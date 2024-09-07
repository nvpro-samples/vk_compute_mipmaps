// Copyright 2021-2024 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#include "mipmaps_app.hpp"

#include <vulkan/vulkan_core.h>
#include "GLFW/glfw3.h"

#include <algorithm>
#include <array>
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <thread>
#include <time.h>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef STB_IMAGE_IMPLEMENTATION

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "imgui.h"
#include "backends/imgui_impl_vulkan.h"

#include "nvh/fileoperations.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "nvvk/swapchain_vk.hpp"

#include "frame_manager.hpp"
#include "image_names.hpp"
#include "scoped_image.hpp"
#include "search_paths.hpp"
#include "timestamps.hpp"

#include "app_args.hpp"
#include "camera_controls.hpp"
#include "drawing.hpp"
#include "julia.hpp"
#include "mipmap_pipelines.hpp"
#include "mipmap_storage.hpp"
#include "gui.hpp"
#include "pipeline_alternative.hpp"

// GLSL polyglots
#include "shaders/camera_transforms.h"
#include "shaders/scene_modes.h"
#include "shaders/swap_image_push_constant.h"

// Class defining main loop of the sample.
class App
{
public:
  // Borrowed from outside.
  nvvk::Context& m_context;
  GLFWwindow*    m_window;
  VkSurfaceKHR   m_surface;
  const AppArgs& m_args;

  // Color format for swap chain, simple render pass, and framebuffers
  // compatible with render pass.
  VkFormat         m_swapColorFormat;
  SwapRenderPass   m_swapRenderPass;
  SwapFramebuffers m_swapFramebuffers;

  // All images use the same sampler so that they have
  // interchangeable pipelines.
  ScopedSampler m_sampler;

  // Image data.
  ScopedImage m_loadedImage;
  Julia       m_julia;
  double      m_lastUpdateTime;

  // Pipelines.
  std::unique_ptr<ComputeMipmapPipelines> m_pComputeMipmapPipelines;

  SwapImagePipeline m_swapImagePipeline;

  nvvk::ProfilerVK m_vkProfiler;
  double           m_lastLogProfilerTime = 0.0;

  // ImGui stuff.
  Gui m_gui;

  // Declared last so that FrameManager's destructor (calls
  // vkQueueWaitIdle) runs first.
  FrameManager m_frameManager;

  // Filename of loaded image. Empty to indicate displaying Julia Set animation.
  std::string m_loadedImageFilename;

  // Used for testing the mipmapped image, and writing images to file.
  std::thread m_testThread, m_writeImageThread;

  // Initialize everything (above object order is very important!)
  App(nvvk::Context& ctx,
      GLFWwindow*    window,
      VkSurfaceKHR   surface,
      const AppArgs& args)
      : m_context(ctx)
      , m_window(window)
      , m_surface(surface)
      , m_args(args)
      , m_swapColorFormat(VK_FORMAT_B8G8R8A8_SRGB)
      , m_swapRenderPass(ctx, m_swapColorFormat)
      , m_swapFramebuffers(ctx, m_swapRenderPass)
      , m_sampler(ctx, ctx.m_physicalDevice)
      , m_loadedImage(ctx, ctx.m_physicalDevice, m_sampler)
      , m_julia(
            ctx,
            ctx.m_physicalDevice,
            args.dumpPipelineStats,
            args.animationTextureWidth,
            args.animationTextureHeight,
            m_sampler)
      , m_lastUpdateTime(glfwGetTime())
      , m_pComputeMipmapPipelines(ComputeMipmapPipelines::make(
            ctx,
            m_loadedImage,
            args.dumpPipelineStats))
      , m_swapImagePipeline(
            ctx,
            ctx.m_physicalDevice,
            m_swapRenderPass,
            m_loadedImage.getTextureDescriptorSetLayout())
      , m_vkProfiler(nullptr)
      , m_frameManager(
            ctx,
            surface,
            1,
            1,
            m_gui.m_vsync,
            m_swapColorFormat,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
  {
    assert((surface == VK_NULL_HANDLE) == !args.openWindow);
    VkCommandBuffer cmdBuf = m_frameManager.recordOneTimeCommandBuffer();

    m_vkProfiler.init(ctx, ctx.m_physicalDevice, ctx.m_queueGCT.familyIndex);

    if(args.openWindow)
    {
      m_gui.cmdInit(cmdBuf, window, ctx, m_frameManager, m_swapRenderPass, 0);
    }

    if(!args.inputFilename.empty())
    {
      // Pipeline alternative used for generating mipmaps.
      const PipelineAlternative* pPipelineAlternative = nullptr;
      for(uint32_t i = 0; i < static_cast<uint32_t>(pipelineAlternativeCount);
          ++i)
      {
        if(pipelineAlternatives[i].label == args.outputPipelineAlternativeLabel)
        {
          pPipelineAlternative = &pipelineAlternatives[i];
          break;
        }
      }
      if(!pPipelineAlternative)
      {
        fprintf(
            stderr, "No such pipeline alternative: %s\n",
            args.outputPipelineAlternativeLabel.c_str());
        exit(1);
      }

      // If requested, load image from file and transfer it to the device.
      m_loadedImage.stageImage(
          nvh::findFile(args.inputFilename, searchPaths), true);
      m_loadedImage.cmdReallocUploadImage(cmdBuf, VK_IMAGE_LAYOUT_GENERAL);

      // Clear mipmaps and generate using the requested pipeline.
      const VkMemoryBarrier clearBarrier = {
          .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT};
      const VkImageSubresourceRange clearRange = {
          .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel   = 1,
          .levelCount     = VK_REMAINING_MIP_LEVELS,
          .baseArrayLayer = 0,
          .layerCount     = 1};
      vkCmdClearColorImage(
          cmdBuf, m_loadedImage.getImage(), VK_IMAGE_LAYOUT_GENERAL,
          m_loadedImage.getPMagenta(), 1, &clearRange);
      vkCmdPipelineBarrier(
          cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &clearBarrier, 0, nullptr,
          0, nullptr);

      m_pComputeMipmapPipelines->cmdBindGenerate(
          cmdBuf, m_loadedImage, *pPipelineAlternative);
      const VkMemoryBarrier downloadBarrier = {
          .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT};
      vkCmdPipelineBarrier(
          cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1, &downloadBarrier, 0, nullptr,
          0, nullptr);

      // Download mipmaps.
      m_loadedImage.cmdDownloadImage(cmdBuf, VK_IMAGE_LAYOUT_GENERAL);
      m_loadedImageFilename = args.inputFilename;
    }

    // Block until operations complete.
    vkEndCommandBuffer(cmdBuf);
    const VkSubmitInfo submitInfo = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &cmdBuf};
    NVVK_CHECK(vkQueueSubmit(
        m_frameManager.getQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    vkQueueWaitIdle(m_frameManager.getQueue());
    vkFreeCommandBuffers(m_context, m_frameManager.getCommandPool(), 1, &cmdBuf);

    // Test mipmap correctness and store images on separate thread.
    if(!args.inputFilename.empty())
    {
      if(args.test)
      {
        auto pMips   = m_loadedImage.copyFromStaging();
        m_testThread = std::thread([pMips = std::move(pMips)] {
          printf("%s\n", testMipmaps(*pMips).c_str());
        });
      }
      if(!args.outputFilename.empty())
      {
        auto pMips         = m_loadedImage.copyFromStaging();
        m_writeImageThread = std::thread(
            [pMips = std::move(pMips), filename = args.outputFilename] {
              writeMipmapsTga(*pMips, filename.c_str());
            });
      }
    }

    // Start benchmark now if requested.
    if(!args.benchmarkFilename.empty())
    {
      fprintf(stderr, "Starting benchmark from command line...\n");
      benchmark(args.benchmarkFilename.c_str(), args.test);
    }
  }

  ~App()
  {
    fprintf(stderr, "Waiting for background threads... (or press ^C))\n");
    if(m_testThread.joinable())
    {
      m_testThread.join();
    }
    if(m_writeImageThread.joinable())
    {
      m_writeImageThread.join();
    }
  }

  bool showAnimation() const { return m_loadedImageFilename.empty(); }

  void doFrame()
  {
    // Get events and window size from GLFW.
    uint32_t width{};
    uint32_t height{};
    glfwPollEvents();
    waitNonzeroFramebufferSize(m_window, &width, &height);

    // Begin the frame, starting primary command buffer recording.
    // beginFrame converts the intended width/height to actual
    // swap chain width/height, which could differ from requested.
    VkCommandBuffer             primaryCmdBuf{};
    nvvk::SwapChainAcquireState acquired{};
    m_frameManager.wantVsync(m_gui.m_vsync);
    m_frameManager.beginFrame(&primaryCmdBuf, &acquired, &width, &height);
    m_vkProfiler.beginFrame();
    auto frameSectionID = m_vkProfiler.beginSection("frame", primaryCmdBuf);

    // Load image if requested.
    const char*& wantLoadImageFilename = m_gui.m_wantLoadImageFilename;
    if(wantLoadImageFilename)
    {
      if(strlen(wantLoadImageFilename) != 0)
      {
        vkQueueWaitIdle(m_frameManager.getQueue());
        m_loadedImage.stageImage(
            nvh::findFile(wantLoadImageFilename, searchPaths), true);
        m_loadedImage.cmdReallocUploadImage(
            primaryCmdBuf, VK_IMAGE_LAYOUT_GENERAL);
      }
      m_loadedImageFilename = wantLoadImageFilename;
      wantLoadImageFilename = nullptr;
    }
    ScopedImage& imageToMipmap =
        showAnimation() ? m_julia.getColorImage() : m_loadedImage;

    // Update timestamps.
    checkReportTimestamps();

    // Update GUI.
    m_gui.m_drawingDynamicImage = showAnimation();
    m_gui.m_imageWidth  = static_cast<int>(imageToMipmap.getImageWidth());
    m_gui.m_imageHeight = static_cast<int>(imageToMipmap.getImageHeight());
    m_gui.doFrame(m_vkProfiler);

    // Resize dynamic image if needed.
    bool animationResized = false;
    if(showAnimation())
    {
      const uint32_t widthU32  = static_cast<uint32_t>(m_gui.m_imageWidth);
      const uint32_t heightU32 = static_cast<uint32_t>(m_gui.m_imageHeight);
      animationResized |= uint32_t(widthU32) != m_julia.getWidth();
      animationResized |= uint32_t(heightU32) != m_julia.getHeight();
      if(animationResized)
      {
        vkQueueWaitIdle(m_frameManager.getQueue());
        m_julia.resize(widthU32, heightU32);
      }
    }

    // Update animation if needed.
    const double newTime = glfwGetTime();
    if(showAnimation() && (m_gui.m_doStep || animationResized))
    {
      m_julia.update(newTime - m_lastUpdateTime, 100);
      m_julia.cmdFillColorTexture(primaryCmdBuf);
    }
    m_lastUpdateTime = newTime;

    // Generate mip-maps, time the operation.
    // Clear the image first to prevent an incorrect pipeline from
    // coincidentally working correctly.
    const VkMemoryBarrier clearBarrier = {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT};
    const VkImageSubresourceRange clearRange = {
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel   = 1,
        .levelCount     = VK_REMAINING_MIP_LEVELS,
        .baseArrayLayer = 0,
        .layerCount     = 1};
    vkCmdClearColorImage(
        primaryCmdBuf, imageToMipmap.getImage(), VK_IMAGE_LAYOUT_GENERAL,
        imageToMipmap.getPMagenta(), 1, &clearRange);
    vkCmdPipelineBarrier(
        primaryCmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &clearBarrier, 0, nullptr,
        0, nullptr);
    const int firstQuery = m_frameManager.evenOdd(0, 2);
    for(int i = 0; i < m_gui.m_mipmapsGeneratedPerFrame; ++i)
    {
      auto scopedSection = m_vkProfiler.timeRecurring("mipmaps", primaryCmdBuf);
      m_pComputeMipmapPipelines->cmdBindGenerate(
          primaryCmdBuf, imageToMipmap,
          pipelineAlternatives[m_gui.m_alternativeIdxSetting]);
    }

    // Clamp explicit lod level.
    auto& lod = m_gui.m_cam.explicitLod;
    m_gui.m_maxExplicitLod =
        static_cast<float>(imageToMipmap.getLevelCount()) - 1.0F;
    lod = std::max(lod, 0.0F);
    lod = std::min(lod, m_gui.m_maxExplicitLod);

    // Fit image to screen if requested.
    if(m_gui.m_wantFitImageToScreen)
    {
      m_gui.m_wantFitImageToScreen = false;

      uint32_t imageWidth = imageToMipmap.getImageWidth();
      if(m_gui.m_cam.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_SHOW_ALL_MIPS)
      {
        // Fit additional width from extra shown mip levels in this mode.
        uint32_t levelWidth  = imageToMipmap.getImageWidth();
        uint32_t levelHeight = imageToMipmap.getImageHeight();
        do
        {
          levelWidth  = levelWidth >= 1 ? levelWidth >> 1U : 1U;
          levelHeight = levelHeight >= 1 ? levelHeight >> 1U : 1U;
          imageWidth += levelWidth;
        } while(levelWidth > 1 && levelHeight > 1);
      }
      const float imageHeight =
          static_cast<float>(imageToMipmap.getImageHeight());
      const float xScale =
          static_cast<float>(imageWidth) / static_cast<float>(width);
      const float yScale   = imageHeight / float(height);
      const float maxScale = std::max(xScale, yScale);

      m_gui.m_cam.scale    = {maxScale, maxScale};
      m_gui.m_cam.offset.x = (static_cast<float>(imageWidth)
                              - static_cast<float>(width) * maxScale)
                             * 0.5F;
      m_gui.m_cam.offset.y =
          (imageHeight - static_cast<float>(height) * maxScale) * 0.5F;
    }

    // Set viewport and scissor.
    const VkViewport viewport{
        0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    const VkRect2D scissor{{0, 0}, {width, height}};
    vkCmdSetViewport(primaryCmdBuf, 0, 1, &viewport);
    vkCmdSetScissor(primaryCmdBuf, 0, 1, &scissor);

    // Select swap chain framebuffer for this frame.
    m_swapFramebuffers.recreateNowIfNeeded(m_frameManager.getSwapChain());
    VkFramebuffer swapFramebuffer = m_swapFramebuffers[acquired.index];

    // Begin render pass.
    const VkRenderPassBeginInfo beginInfo{
        .sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass  = m_swapRenderPass,
        .framebuffer = swapFramebuffer,
        .renderArea  = {{0, 0}, {width, height}}};
    vkCmdBeginRenderPass(primaryCmdBuf, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Bind pipeline + input data, and draw full-screen triangle.
    const auto& imageToDraw =
        showAnimation() ? m_julia.getColorImage() : m_loadedImage;
    VkDescriptorSet baseColorSampler = imageToDraw.getTextureDescriptorSet();
    SwapImagePushConstant swapImagePushConstant{};
    updateFromControls(m_gui.m_cam, swapImagePushConstant);
    CameraTransforms cameraTransforms{};
    if(m_gui.m_cam.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_3D)
      updateFromControls(m_gui.m_cam, viewport, cameraTransforms);
    m_swapImagePipeline.cmdBindDraw(
        primaryCmdBuf, swapImagePushConstant, cameraTransforms,
        baseColorSampler, m_frameManager.evenOdd() != 0);
    // Draw GUI
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), primaryCmdBuf);

    // Submit primary command buffer, and present frame. The primary
    // command buffer is magically cleaned up later.
    vkCmdEndRenderPass(primaryCmdBuf);
    m_vkProfiler.endSection(frameSectionID, primaryCmdBuf);
    m_vkProfiler.endFrame();
    m_frameManager.endFrame(primaryCmdBuf);

    // Download the image for upcoming tests or write-to-disk, if any.
    // Need to do these things NOW in case the image is overwritten next frame.
    if(m_gui.m_wantTestDownloadedImage || m_gui.m_wantWriteImageBaseFilename)
    {
      // Download mipmapped image to staging buffer and wait.
      primaryCmdBuf = m_frameManager.recordOneTimeCommandBuffer();
      const VkMemoryBarrier barrier = {
          .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
          .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT};
      vkCmdPipelineBarrier(
          primaryCmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
          VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1, &barrier, 0, nullptr, 0,
          nullptr);
      imageToMipmap.cmdDownloadImage(primaryCmdBuf, VK_IMAGE_LAYOUT_GENERAL);
      vkEndCommandBuffer(primaryCmdBuf);
      const VkSubmitInfo downloadSubmitInfo = {
          .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
          .commandBufferCount = 1,
          .pCommandBuffers    = &primaryCmdBuf};
      vkQueueSubmit(m_frameManager.getQueue(), 1, &downloadSubmitInfo, {});
      vkQueueWaitIdle(m_frameManager.getQueue());
      vkFreeCommandBuffers(
          m_context, m_frameManager.getCommandPool(), 1, &primaryCmdBuf);

      if(m_gui.m_wantTestDownloadedImage)
      {
        m_gui.m_wantTestDownloadedImage = false;

        // Copy out of staging buffer (avoid race conditions).
        auto pMips = imageToMipmap.copyFromStaging();
        if(m_testThread.joinable())
        {
          m_testThread.join();
        }
        m_testThread = std::thread([pMips = std::move(pMips)] {
          printf("%s\n", testMipmaps(*pMips).c_str());
        });
        printf("Test beginning...\n");
      }

      if(m_gui.m_wantWriteImageBaseFilename)
      {
        std::string baseFilename(m_gui.m_wantWriteImageBaseFilename);
        m_gui.m_wantWriteImageBaseFilename = nullptr;

        auto pMips = imageToMipmap.copyFromStaging();
        if(m_writeImageThread.joinable())
        {
          m_writeImageThread.join();
        }
        m_writeImageThread = std::thread(
            [pMips = std::move(pMips), baseFilename = std::move(baseFilename)] {
              writeMipmapsTga(*pMips, baseFilename.c_str());
            });
      }
    }

    // Run the benchmark if requested by the user.
    if(m_gui.m_wantBenchmark)
    {
      m_gui.m_wantBenchmark = false;
      fprintf(stderr, "Starting benchmark from UI...\n");
      time_t    time_ = time(nullptr);
      struct tm localtime_;
#ifdef WIN32
      localtime_s(&localtime_, &time_);  // They had to disagree
#else
      localtime_r(&time_, &localtime_);  // on the last letter :(
#endif
      char outputFilename[80];
      // Substitute ':' with '_' as ':' is forbidden on Windows.
      const char* format = "nvpro_pyramid_benchmark_%Y-%m-%dT%H_%M_%S%z.json";
      strftime(outputFilename, sizeof outputFilename, format, &localtime_);
      benchmark(outputFilename, true);
    }
  }

  void checkReportTimestamps()
  {
    const double now = glfwGetTime();
    if(m_gui.m_doLogPerformance && floor(now) != floor(m_lastLogProfilerTime))
    {
      m_lastLogProfilerTime = now;
      reportPerformance("frame");
      reportPerformance("mipmaps");
    }
  }

  void reportPerformance(const char* id)
  {
    nvh::Profiler::TimerInfo timerInfo{};
    m_vkProfiler.getTimerInfo(id, timerInfo);
    const double cpuMs = timerInfo.cpu.average * 0.001;
    const double gpuMs = timerInfo.gpu.average * 0.001;
    printf(
        "%10s \x1b[36mCPU:\x1b[0m %7.4f ms| \x1b[32mGPU:\x1b[0m %7.4f ms\n", id,
        cpuMs, gpuMs);
  }

  // Benchmark run, generate mipmaps for each image and each pipeline
  // alternative.  Record times to the named json file.  Repeat each
  // generation multiple times to reduce noise and powerstate effects,
  // spread out over multiple batches. IGNORES THE INITIAL BATCH.
  void benchmark(const char* pOutputFilename, bool enableTesting) const
  {
    VkDevice device      = m_context.m_device;
    VkQueue  queue       = m_context.m_queueGCT;
    auto     queueFamily = m_context.m_queueGCT.familyIndex;

    // Allocate command buffers and fences.
    // Alternate command buffer usage to keep GPU saturated.
    VkCommandBuffer  cmdBufArray[2]{};
    VkCommandBuffer& cmdBuf     = cmdBufArray[0];
    VkCommandBuffer& prevCmdBuf = cmdBufArray[1];
    VkFence          fence{};
    VkFence          prevFence{};

    const VkSubmitInfo submitInfo = {
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &cmdBuf};

    VkCommandPool                     cmdPool = m_frameManager.getCommandPool();
    const VkCommandBufferAllocateInfo cmdBufAllocInfo = {
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = cmdPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 2};
    NVVK_CHECK(vkAllocateCommandBuffers(device, &cmdBufAllocInfo, cmdBufArray));
    const VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};

    const VkFenceCreateInfo fenceInfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT};
    NVVK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));
    NVVK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &prevFence));

    int pipelineAlternative = 0;

    // Load all the test images into staging buffers.
    fprintf(stderr, "Loading test images from disk...\n");
    std::array<std::unique_ptr<ScopedImage>, imageNameArraySize> images;
    std::array<std::thread, imageNameArraySize> loadImageThreads;
    for(int i = 0; i < imageNameArraySize; ++i)
    {
      // Use threads to load images to staging buffer as stb image is sloww
      images[i] = std::make_unique<ScopedImage>(
          device, m_context.m_physicalDevice, m_sampler);
      ScopedImage*      pImage = images[i].get();
      const std::string imageFilename =
          nvh::findFile(imageNameArray[i], searchPaths);
      loadImageThreads[i] = std::thread([pImage, imageFilename]() {
        pImage->stageImage(imageFilename, true);
      });
    }

    NVVK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));

    // Use one command buffer to upload to device.
    for(int i = 0; i < imageNameArraySize; ++i)
    {
      loadImageThreads[i].join();
      images[i]->cmdReallocUploadImage(cmdBuf, VK_IMAGE_LAYOUT_GENERAL);
      // Includes barrier.
    }

    // Also allocate and init enough timestamp queries.
    constexpr size_t BatchCount      = 256;  // Must be even.
    constexpr double RepetitionCount = 8;

    const uint32_t timestampCount = uint32_t(
        2 * BatchCount * imageNameArraySize * pipelineAlternativeCount);
    Timestamps timestamps(m_context, queueFamily, timestampCount);
    // times in second, stored by [pipeline alternative][test image index][batch number]
    std::vector<std::array<std::array<double, BatchCount>, imageNameArraySize>>
             times(pipelineAlternativeCount);
    uint32_t queryIdx = 0;
    timestamps.cmdResetQueries(cmdBuf);

    // Submit, ready to start benchmark.
    NVVK_CHECK(vkEndCommandBuffer(cmdBuf));
    vkResetFences(device, 1, &fence);
    NVVK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
    std::swap(cmdBuf, prevCmdBuf);
    std::swap(fence, prevFence);

    // If testing is enabled, generate expected mipmaps for each image
    // in the background.
    std::unique_ptr<MipmapStorage<uint8_t, 4>>
                expectedResults[imageNameArraySize];
    std::thread expectedResultThreads[imageNameArraySize];
    if(enableTesting)
    {
      for(size_t i = 0; i < imageNameArraySize; ++i)
      {
        const ScopedImage& srcImage = *images[i];
        expectedResults[i]          = srcImage.copyFromStaging();
        expectedResultThreads[i] =
            std::thread(cpuGenerateMipmaps_sRGBA, expectedResults[i].get());
      }
    }

    // Threads and locations for correctness test results,
    // in [pipeline alternative][test image index] order.
    std::thread imageCompareThreads[imageNameArraySize];
    std::vector<std::array<uint8_t, imageNameArraySize>> worstDeltaArray(
        pipelineAlternativeCount);

    // Run the benchmark loops. If testing is enabled, run an extra batch
    // for testing purposes, not counted for timing.
    auto realBatchCount = BatchCount + 1;
    fprintf(stderr, "Generating mipmaps. GPU may now start squeaking...\n");
    for(int batch = 0; batch < realBatchCount; ++batch)
    {
      if(batch == BatchCount)
      {
        fprintf(stderr, "Testing for correctness...\n");
      }
      for(pipelineAlternative = 0;
          pipelineAlternative < pipelineAlternativeCount; ++pipelineAlternative)
      {
        for(size_t imageIdx = 0; imageIdx < images.size(); ++imageIdx)
        {
          // Begin command buffer and timing.
          // Make sure command buffer is done before overwriting.
          NVVK_CHECK(vkWaitForFences(device, 1, &fence, 0, UINT64_MAX));
          NVVK_CHECK(vkResetCommandBuffer(cmdBuf, 0));
          NVVK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));
          if(batch != BatchCount)
          {
            timestamps.cmdWriteTimestamp(cmdBuf, queryIdx++);
          }
          else
          {
            // Clear mips on testing runs.
            const VkImageSubresourceRange range{
                .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel   = 1,
                .levelCount     = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount     = 1};
            vkCmdClearColorImage(
                cmdBuf, images[imageIdx]->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                images[imageIdx]->getPMagenta(), 1, &range);
            const VkMemoryBarrier barrier{
                .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT};
            vkCmdPipelineBarrier(
                cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                nullptr, 0, nullptr);
          }

          const VkMemoryBarrier barrier = {
              .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
              .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
              .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT};
          for(int reps = 0; reps < RepetitionCount; ++reps)
          {
            // Record command to generate mipmaps, plus WAW barrier.
            m_pComputeMipmapPipelines->cmdBindGenerate(
                cmdBuf, *images[imageIdx],
                pipelineAlternatives[pipelineAlternative]);
            vkCmdPipelineBarrier(
                cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                nullptr, 0, nullptr);
          }

          // End timing.
          if(batch != BatchCount)
          {
            timestamps.cmdWriteTimestamp(cmdBuf, queryIdx++);
          }

          // On test batch, download the image for later testing if
          // enabled.  Also, have to ensure that any thread reading
          // from this image's staging buffer is done executing to
          // avoid data race.
          if(enableTesting && batch == BatchCount)
          {
            if(imageCompareThreads[imageIdx].joinable())
            {
              imageCompareThreads[imageIdx].join();
            }
            images[imageIdx]->cmdDownloadImage(cmdBuf, VK_IMAGE_LAYOUT_GENERAL);
          }

          // Submit and swap command buffers.
          NVVK_CHECK(vkEndCommandBuffer(cmdBuf));
          vkResetFences(device, 1, &fence);
          NVVK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
          std::swap(cmdBuf, prevCmdBuf);
          std::swap(fence, prevFence);

          // Start comparison.
          if(enableTesting && batch == BatchCount)
          {
            NVVK_CHECK(vkWaitForFences(device, 1, &prevFence, 0, UINT64_MAX));
            if(pipelineAlternative == 0)
            {
              expectedResultThreads[imageIdx].join();
            }
            imageCompareThreads[imageIdx] = std::thread(
                [pImage    = images[imageIdx].get(),
                 pOutput   = &worstDeltaArray[pipelineAlternative][imageIdx],
                 pExpected = expectedResults[imageIdx].get()] {
                  auto pMips = pImage->copyFromStaging();
                  *pOutput   = pMips->compare(*pExpected);
                });
          }
        }
      }
    }
    assert(queryIdx == timestampCount);
    queryIdx = 0;

    // Read and record the timestamps. Make sure to use the same loop
    // order as earlier to read the timestamps correctly.
    fprintf(stderr, "Waiting for benchmark timestamps...\n");
    for(int batch = 0; batch < BatchCount; ++batch)
    {
      for(pipelineAlternative = 0;
          pipelineAlternative < pipelineAlternativeCount; ++pipelineAlternative)
      {
        for(size_t imageIdx = 0; imageIdx < images.size(); ++imageIdx)
        {
          times[pipelineAlternative][imageIdx][batch] =
              timestamps.subtractTimestampSeconds(queryIdx + 1, queryIdx);
          queryIdx += 2;
        }
      }
    }
    assert(queryIdx == timestampCount);

    // Calculate and print out the info.
    fprintf(stderr, "Writing benchmark json to '%s'...\n", pOutputFilename);
    int         err        = 0;
    const char* fileAction = "opening";
    FILE*       file       = fopen(pOutputFilename, "w");
    if(file == nullptr)
    {
      goto onFileError;
    }
    fileAction = "writing to";
    err        = fprintf(file, "{\n");
    if(err < 0)
      goto onFileError;

    // This order makes more sense now (easily compare between
    // pipelines for same image).
    for(size_t imageIdx = 0; imageIdx < images.size(); ++imageIdx)
    {
      err = fprintf(file, "\"%s\": {\n", imageNameArray[imageIdx]);
      for(pipelineAlternative = 0;
          pipelineAlternative < pipelineAlternativeCount; ++pipelineAlternative)
      {
        // Find min, max, median. Ignore first batch as documented.
        // Thus the median is the first value after the midpoint.
        std::array<double, BatchCount>& batchTimes =
            times[pipelineAlternative][imageIdx];
        std::sort(batchTimes.begin() + 1, batchTimes.end());
        const double min_ = batchTimes[1] / RepetitionCount * 1e9;
        const double median =
            batchTimes[BatchCount / 2] / RepetitionCount * 1e9;
        const double max_ = batchTimes[BatchCount - 1] / RepetitionCount * 1e9;
        static_assert(BatchCount % 2 == 0, "need even BatchCount");

        // Get the test results.
        std::string testResultsString;
        if(enableTesting)
        {
          if(imageCompareThreads[imageIdx].joinable())
          {
            imageCompareThreads[imageIdx].join();
          }
          const int delta = int(worstDeltaArray[pipelineAlternative][imageIdx]);
          testResultsString = ", \"delta\":" + std::to_string(delta) + "";
        }

        // Print the data. Align to make it easier to compare rows.
        // The formatted output is nicer-looking than this weird format string.
        const char* format =
            "  \"%s\":"  // pipeline name
            "%.*s{\"median_ns\":%7.0f, \"min_ns\":%7.0f, \"max_ns\":%7.0f%s}%c\n";
        // padding          median              min            max  test result  trailing comma/}
        const char*  name    = pipelineAlternatives[pipelineAlternative].label;
        const size_t nameLen = strlen(name);
        const int    paddingChars =
            nameLen > 18 ? 0 : 18 - static_cast<int>(nameLen);
        const bool isLastRow =
            pipelineAlternative == pipelineAlternativeCount - 1;
        err = fprintf(
            file, format, name, paddingChars, "                  ", median,
            min_, max_, testResultsString.c_str(), isLastRow ? '}' : ',');
        if(err < 0)
          goto onFileError;
      }

      const bool isLastImage = imageIdx == images.size() - 1;
      err                    = fprintf(file, "%c\n", isLastImage ? '}' : ',');
      if(err < 0)
        goto onFileError;
    }

    // Clean up stuff when done.
    vkQueueWaitIdle(queue);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyFence(device, prevFence, nullptr);
    vkFreeCommandBuffers(device, cmdPool, 2, cmdBufArray);
    fileAction = "closing";
    err        = fclose(file);
    if(err < 0)
      goto onFileError;

    // Timestamps and ScopedImage has destructor.
    fprintf(stderr, "Benchmark complete!\n");
    return;

  onFileError:
    fprintf(
        stderr, "Error %s '%s': %s (%i)\n", fileAction, pOutputFilename,
        strerror(errno), errno);
    exit(1);
  }

  // Get the framebuffer size for the given glfw window; suspend until the glfw
  // window has nonzero size (i.e. not minimized).
  static void waitNonzeroFramebufferSize(
      GLFWwindow* pWindow,
      uint32_t*   pWidth,
      uint32_t*   pHeight)
  {
    int width{};
    int height{};
    glfwGetFramebufferSize(pWindow, &width, &height);
    while(width == 0 || height == 0)
    {
      glfwWaitEvents();
      glfwGetFramebufferSize(pWindow, &width, &height);
    }
    *pWidth  = static_cast<uint32_t>(width);
    *pHeight = static_cast<uint32_t>(height);
  }
};


// Instantiate the App class and run until the user clicks X button or
// equivalent. Immediately exit if no window is opened.
void mipmapsApp(
    nvvk::Context& context,
    GLFWwindow*    window,
    VkSurfaceKHR   surface,
    const AppArgs& args)
{
  App app(context, window, surface, args);
  while(args.openWindow && !glfwWindowShouldClose(window))
  {
    app.doFrame();
  }
}
