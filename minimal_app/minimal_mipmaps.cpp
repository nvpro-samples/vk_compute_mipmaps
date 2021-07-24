// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Minimal usage sample of nvpro_pyramid library
// Uses the out-of-the-box srgba8 shader available in nvpro_pyramid, thus
// it doesn't demonstrate the full flexibility of the library, e.g. custom
//  * pipeline / descriptor set layout
//  * reduction function
// VERY CPU-bound, stb image takes a while to load and write image files.

#include <vulkan/vulkan.h>

#include <array>
#include <cassert>
#include <stdio.h>
#include <string>

// Image file library
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#undef  STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "make_compute_pipeline.hpp"
#include "scoped_image.hpp"
#include "nvpro_pyramid/nvpro_pyramid_dispatch.hpp"

#include "nvh/fileoperations.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/error_vk.hpp"

struct Config
{
  // The fast mipmap pipeline needs some non-guaranteed device functionality;
  // this records if the needed functionality is available and enabled.
  bool canUseFastPipeline{};

  // Testing: force disable fast pipeline.
  bool forceDisableFastPipeline{};

  // Whether the input image does NOT have premultiplied alpha (so we
  // need to do this ourselves).
  bool doPremultiplyAlpha{};

  // Input image name, searched for in working dir and searchPath.
  std::string rawInputFilename = "4096.jpg";

  // Output image name template (modified with mip level number).
  std::string outputFilenameTemplate = "./vk_compute_mipmaps_minimal.tga";

  // Fill from command line arguments (except canUseFastPipeline).
  Config(int argc, char** argv);
};

void app(nvvk::Context& ctx, const Config& config)
{
  // Queue to use: prefer compute only queue.
  VkQueue  queue;
  uint32_t queueFamilyIndex;
  if (ctx.m_queueC.queue)
  {
    queue            = ctx.m_queueC.queue;
    queueFamilyIndex = ctx.m_queueC.familyIndex;
  }
  else
  {
    queue            = ctx.m_queueGCT.queue;
    queueFamilyIndex = ctx.m_queueGCT.familyIndex;
  }

  // Command pool and command buffer setup.
  VkCommandPool           cmdPool;
  VkCommandPoolCreateInfo cmdPoolInfo{};
  cmdPoolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
  NVVK_CHECK(vkCreateCommandPool(ctx, &cmdPoolInfo, nullptr, &cmdPool));
  VkCommandBuffer cmdBuf;
  VkCommandBufferAllocateInfo cmdBufInfo{};
  cmdBufInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdBufInfo.commandPool        = cmdPool;
  cmdBufInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdBufInfo.commandBufferCount = 1u;
  NVVK_CHECK(vkAllocateCommandBuffers(ctx, &cmdBufInfo, &cmdBuf));
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  NVVK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));


  // **************************************************************************
  // Load image from file into staging buffer.
  auto filename = nvh::findFile(config.rawInputFilename, searchPaths, true);
  ScopedImage scopedImage(ctx, ctx.m_physicalDevice);
  fprintf(stderr, "Loading: '%s'...", filename.c_str());
  scopedImage.stageImage(filename, config.doPremultiplyAlpha);
  fprintf(stderr, " done\n");


  // **************************************************************************
  // Allocate an image and copy the staging buffer contents to the image.
  // The details are intentionally hidden in ../local_helpers/ScopedImage.hpp,
  // but the summary is:
  //
  // Allocate an sRGBA8 image with
  //   * VK_IMAGE_USAGE_SAMPLED_BIT and
  //   * VK_IMAGE_USAGE_STORAGE_BIT
  // (plus extra flags to be explained)
  //
  // For read access, create an sRGB view, sampler, and descriptor
  // for the image, ensure all mip levels are included in
  // subresourceRange.
  //
  // For write access, create an array of image views and storage
  // image descriptors, one array entry for each mip level (capped to
  // 16 for the out-of-the-box shader, but the underlying
  // nvproPyramidMain is not limited except by int overflow).
  //
  // Unfortunately, NVIDIA devices do not support imageStore for sRGB
  // images, so the storage views are of type
  //   * VK_FORMAT_R8G8B8A8_UINT
  // and sRGB conversion is done within the shader code. This requires
  // these flags when creating the original VkImage:
  //   * VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT
  // so that the image can be legally reinterpreted as a uint image, and
  //   * VK_IMAGE_CREATE_EXTENDED_USAGE_BIT
  // so that the sRGBA8 image can be given usage VK_IMAGE_USAGE_STORAGE_BIT
  // despite that usage not being supported for sRGBA8 images.
  //
  // See ScopedImage::reallocImage for concrete source code.
  scopedImage.cmdReallocUploadImage(cmdBuf, VK_IMAGE_LAYOUT_GENERAL);

  // Above command includes pipeline barrier.


  // **************************************************************************
  // Compile the pipeline layout and compute pipelines.
  NvproPyramidPipelines pipelines{};

  // Push constant; the library needs a single 32-bit push constant to operate.
  // You only have to include the push constant in the layout, and set
  // NvproPyramidPipelines::pushConstantOffset to 0.
  // Can be customized with glsl macro NVPRO_PYRAMID_PUSH_CONSTANT
  VkPushConstantRange pcRange  = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
  pipelines.pushConstantOffset = 0;

  // Descriptor sets; this is specific to the included example sRGBA8
  // pipeline.  The base nvproPyramidMain implementation does not
  // specify any descriptor set layout; you declare shader
  // inputs/outputs yourself and teach nvproPyramidMain how to load
  // and store image data by providing NVPRO_PYRAMID_LOAD and
  // NVPRO_PYRAMID_STORE macros. For this example, these macros
  // are defined in ../nvpro_pyramid/srgba8_mipmap_preamble.glsl
  std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts;
  // Sampler used for read access.
  descriptorSetLayouts[0] = scopedImage.getTextureDescriptorSetLayout();
  // Array of storage images used for write access.
  descriptorSetLayouts[1] = scopedImage.getStorageDescriptorSetLayout();

  // Set up NvproPyramidPipelines::layout.
  VkPipelineLayoutCreateInfo layoutInfo = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
      2, descriptorSetLayouts.data(),
      1, &pcRange };
  NVVK_CHECK(
      vkCreatePipelineLayout(ctx, &layoutInfo, nullptr, &pipelines.layout));

  // Compile both pipelines, except skip
  // NvproPyramidPipelines::fastPipeline if not usable.
  std::string generalPipelineFilename = nvh::findFile(
      "srgba8_mipmap_general_pipeline.comp.spv", searchPaths, true);
  std::string fastPipelineFilename = nvh::findFile(
      "srgba8_mipmap_fast_pipeline.comp.spv", searchPaths, true);

  makeComputePipeline(ctx, generalPipelineFilename.c_str(), false,
                      pipelines.layout, &pipelines.generalPipeline);
  if (config.canUseFastPipeline)
  {
    makeComputePipeline(ctx, fastPipelineFilename.c_str(), false,
                        pipelines.layout, &pipelines.fastPipeline);
  }
  else
  {
    fprintf(stderr, "Debug: Cannot use NvproPyramidPipelines::fastPipeline\n");
    pipelines.fastPipeline = VK_NULL_HANDLE;
  }


  // **************************************************************************
  // Bind descriptor sets and dispatch mipmap shaders.
  // NOTE: nvproPyramidDispatch does not include barriers before and after.
  // ScopedImage inserts these barriers, but in general you handle it yourself.
  std::array<VkDescriptorSet, 2> descriptorSets;
  descriptorSets[0] = scopedImage.getTextureDescriptorSet();
  descriptorSets[1] = scopedImage.getStorageDescriptorSet();
  vkCmdBindDescriptorSets(
      cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines.layout,
      0, 2, descriptorSets.data(), 0, nullptr);
  uint32_t baseMipWidth  = scopedImage.getImageWidth();
  uint32_t baseMipHeight = scopedImage.getImageHeight();
  nvproCmdPyramidDispatch(cmdBuf, pipelines, baseMipWidth, baseMipHeight);
  // NOTE: nvproCmdPyramidDispatch has a fifth `mipLevels` argument: if not
  // provided, it's assumed that the image has the maximum possible number
  // of mip levels possible given its base size.


  // **************************************************************************
  // Copy back to staging buffer.
  scopedImage.cmdDownloadImage(cmdBuf, VK_IMAGE_LAYOUT_GENERAL);


  // **************************************************************************
  // Execute commands and write to disk.
  vkEndCommandBuffer(cmdBuf);
  VkSubmitInfo submitInfo{};
  submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers    = &cmdBuf;
  vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
  auto mipsUniquePtr = scopedImage.copyFromStaging();
  writeMipmapsTga(*mipsUniquePtr, config.outputFilenameTemplate.c_str());


  // **************************************************************************
  // Cleanup
  vkDestroyPipeline(ctx, pipelines.fastPipeline, nullptr);
  vkDestroyPipeline(ctx, pipelines.generalPipeline, nullptr);
  vkDestroyPipelineLayout(ctx, pipelines.layout, nullptr);
  vkDestroyCommandPool(ctx, cmdPool, nullptr);
  // ScopedImage cleans up descriptors, descriptor layouts, and
  // staging buffer and image.
}



int main(int argc, char** argv)
{
  Config config(argc, argv);

  // Initialize instance and device using helper.
  nvvk::Context ctx;
  nvvk::ContextCreateInfo deviceInfo;
  deviceInfo.apiMajor = 1;
  deviceInfo.apiMinor = 1;
  ctx.init(deviceInfo);
  ctx.ignoreDebugMessage(1303270965);  // Bogus "general layout" perf warning.

  // Check needed device features for fast pipeline.
  config.canUseFastPipeline = true;  // May be overriden later.

  VkPhysicalDeviceSubgroupProperties subgroupProperties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 physicalDeviceProperties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroupProperties};
  vkGetPhysicalDeviceProperties2(ctx.m_physicalDevice, &physicalDeviceProperties);

  if (config.forceDisableFastPipeline)
  {
    fprintf(stderr, "Debug: faking missing subgroup features\n");
    subgroupProperties = {};
  }
  if (subgroupProperties.subgroupSize < 16)
  {
    fprintf(stderr, "fastPipeline not usable: subgroupSize < 16\n");
    config.canUseFastPipeline = false;
  }
  else if (subgroupProperties.subgroupSize != 32)
  {
    fprintf(stderr, "\x1b[35m\x1b[1mWARNING:\x1b[0m "
                    "Only tested with subgroup size 32, not %u.\nI /expect/ "
                    "it to work anyway, notify dakeley@nvidia.com if not.\n",
                    subgroupProperties.subgroupSize);
  }
  if (!(subgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT))
  {
    fprintf(stderr, "fastPipeline not usable: no compute subgroups\n");
    config.canUseFastPipeline = false;
  }
  if (!(subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT))
  {
    fprintf(stderr, "fastPipeline not usable: no subgroup shuffle support\n");
    config.canUseFastPipeline = false;
  }

  app(ctx, config);

  ctx.deinit();
  return 0;
}



const char helpString[] =
"%s:\n    Generates mipmaps for an input image and exports as TGA.\n"
"Not a full-feature texture tool; just a simple nvpro_pyramid demonstration.\n"
"Note that this is heavily CPU bound (file I/O); use the benchmark button\n"
"in vk_compute_mipmaps_demo to test the GPU mipmap generation speed.\n"
"\n"
"    ** Arguments **\n"
"-i [input filename]\n"
"-o [output filename] (will be annotated with mip level numbers)\n"
"    NOTE: I have experienced some image viewers (e.g. eog) that incorrectly\n"
"    show opaque texels as transparent for TGA images, for reasons unknown.\n"
"-force-no-fast-pipeline: debug tool, fake that hardware requirements for\n"
"    NvproPyramidPipelines::fastPipeline are not met.\n"
"-premultiplied-alpha: indicate input image has premultiplied alpha.\n"
"-do-premultiply-alpha: indicate input image does not have premultiplied\n"
"    alpha, so the program must do this itself.\n"
"Note that output images have premultiplied alpha in either case\n"
"(probably will look bad in most image viewers).\n";

Config::Config(int argc, char** argv)
{
  auto checkNeededParam = [argv](const char* arg, const char* needed)
  {
    if (needed == nullptr)
    {
      fprintf(stderr, "%s: %s missing parameter\n", argv[0], arg);
      exit(1);
    }
  };

  for (int i = 1; i < argc; ++i)
  {
    const char* arg    = argv[i];
    const char* param0 = argv[i+1];
    const char* param1 = param0 == nullptr ? nullptr : argv[i+2];

    if (strcmp(arg, "-h") == 0 || strcmp(arg, "/?") == 0)
    {
      printf(helpString, argv[0]);
      exit(0);
    }
    else if (strcmp(arg, "-i") == 0)
    {
      checkNeededParam(arg, param0);
      this->rawInputFilename = param0;
      ++i;
    }
    else if (strcmp(arg, "-o") == 0)
    {
      checkNeededParam(arg, param0);
      this->outputFilenameTemplate = param0;
      ++i;
    }
    else if (strcmp(arg, "-force-no-fast-pipeline") == 0)
    {
      this->forceDisableFastPipeline = true;
    }
    else if (strcmp(arg, "-premultiplied-alpha") == 0)
    {
      this->doPremultiplyAlpha = false;
    }
    else if (strcmp(arg, "-do-premultiply-alpha") == 0)
    {
      this->doPremultiplyAlpha = true;
    }
    else
    {
      fprintf(stderr, "%s: Unknown argument '%s'\n", argv[0], arg);
      exit(1);
    }
  }
}
