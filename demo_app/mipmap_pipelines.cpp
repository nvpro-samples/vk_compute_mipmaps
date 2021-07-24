// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#include "mipmap_pipelines.hpp"

#include <map>
#include <thread>

#include "nvh/container_utils.hpp"
#include "nvvk/shadermodulemanager_vk.hpp"

#include "make_compute_pipeline.hpp"
#include "nvpro_pyramid_dispatch.hpp"
#include "nvpro_pyramid_dispatch_alternative.hpp"
#include "scoped_image.hpp"

class ComputeMipmapPipelinesImpl : public ComputeMipmapPipelines
{
  // Borrowed
  VkDevice m_device{};

  // Managed by us.
  VkPipelineLayout          m_layout{};

  // General-case (NP2) shaders, testing multiple candidates.
  // Map pipeline alternative name + config bits to pipeline object.
  std::map<std::pair<std::string, uint32_t>, VkPipeline> m_generalPipelineMap;

  // Special case 2x2 reduction shaders, testing many candidates; same mapping.
  std::map<std::pair<std::string, uint32_t>, VkPipeline> m_fastPipelineMap;

  using PipelineMapPair = decltype(m_fastPipelineMap)::value_type;

  // Initialize a key-value pair in the fast/general pipeline map, but
  // do not actually add the pipeline yet.
  template <bool IsFastPipeline>
  void addNullPipelineEntry(const PipelineAlternativeDescription& description)
  {
    const std::string& dirname = description.basePipelineName.empty() ?
                                     description.name :
                                     description.basePipelineName;
    // Skip special names.
    if (dirname == "blit")
    {
      assert(!IsFastPipeline);
      return;
    }
    if (dirname == "none")
    {
      assert(IsFastPipeline);
      return;
    }

    auto& pipelineMap = IsFastPipeline ? m_fastPipelineMap : m_generalPipelineMap;
    pipelineMap[{dirname, description.configBits}] = VK_NULL_HANDLE;
  }

  // Compile the pipeline value in a pipeline key-value pair.
  template <bool IsFastPipeline>
  void compilePipelineEntry(PipelineMapPair* pPair, bool dumpPipelineStats)
  {
    const auto& dirname    = pPair->first.first;
    const auto& configBits = pPair->first.second;
    VkPipeline* pPipeline  = &pPair->second;

    // Set up shader module compiler and include path.
    nvvk::ShaderModuleManager shaderModuleManager(m_device);
    if (dirname != "default")
    {
      // Add directories with the wanted pipeline alternative glsl file.
      const auto& alternativeDirectories =
          IsFastPipeline ? getFastPipelineAlternativeDirectories(dirname) :
                           getGeneralPipelineAlternativeDirectories(dirname);
      for (const auto& directory : alternativeDirectories)
      {
        shaderModuleManager.addDirectory(directory);
      }
    }

    // Add other directories.
    for (const auto& directory : searchPaths)
    {
      shaderModuleManager.addDirectory(directory);

    }

    // Add undocumented macro to include alternative implementation if not
    // using the default one.
    std::string prepend = "";
    if (dirname != "default")
    {
      prepend = IsFastPipeline ? "#define NVPRO_USE_FAST_PIPELINE_ALTERNATIVE_ 1\n"
                               : "#define NVPRO_USE_GENERAL_PIPELINE_ALTERNATIVE_ 1\n";
    }

    // Add macros based on config bits.
    using namespace PipelineAlternativeDescriptionConfig;

    if (configBits & srgbSharedBit)
    {
      prepend += "#define SRGB_SHARED 1\n";
    }
    if (configBits & f16SharedBit)
    {
      prepend += "#extension GL_EXT_shader_explicit_arithmetic_types : enable\n";
      prepend += "#define F16_SHARED 1\n";
    }
    if (configBits & noBilinearBit)
    {
      prepend += "#define USE_BILINEAR_SAMPLING 0\n";
    }

    auto id = shaderModuleManager.createShaderModule(
        VK_SHADER_STAGE_COMPUTE_BIT,
        IsFastPipeline ? "./nvpro_pyramid/srgba8_mipmap_fast_pipeline.comp" :
                         "./nvpro_pyramid/srgba8_mipmap_general_pipeline.comp",
        prepend, nvvk::ShaderModuleManager::FILETYPE_GLSL);
    VkShaderModule module = shaderModuleManager.get(id);
    assert(module);

    PipelineAlternativeDescription description{dirname, "", configBits};
    std::string humanName =
        (IsFastPipeline ? "srgba8 fastPipeline " : "srgba8 generalPipeline ")
        + description.toString();
    makeComputePipeline(m_device, module, dumpPipelineStats, m_layout,
                        pPipeline, humanName.c_str());
  }

public:
  ComputeMipmapPipelinesImpl(VkDevice           device,
                             const ScopedImage& image,
                             bool               dumpPipelineStats)
      : m_device(device)
  {
    // Set up pipeline layout inputs.
    VkDescriptorSetLayout setLayouts[] = {image.getTextureDescriptorSetLayout(),
                                          image.getStorageDescriptorSetLayout()};
    VkPushConstantRange pushConstantRange = {
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) };

    // Make pipeline layout.
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
        arraySize(setLayouts), setLayouts, 1, &pushConstantRange };
    NVVK_CHECK(vkCreatePipelineLayout(
        device, &pipelineLayoutInfo, nullptr, &m_layout));

    // Gather all the compute shaders descriptions that have to be compiled.
    for (int i = 0; i < pipelineAlternativeCount; ++i)
    {
      const PipelineAlternative& alt = pipelineAlternatives[i];
      addNullPipelineEntry<false>(alt.generalAlternative);
      addNullPipelineEntry<true>(alt.fastAlternative);
    }

    // Use threads to compile them, except if we're dumping stats.
    std::vector<std::thread> threads;
    for (PipelineMapPair& entry : m_generalPipelineMap)
    {
      auto lambda = [&] { compilePipelineEntry<false>(&entry, dumpPipelineStats); };
      if (dumpPipelineStats)
        lambda();
      else
        threads.emplace_back(std::move(lambda));
    }
    for (PipelineMapPair& entry : m_fastPipelineMap)
    {
      auto lambda = [&] { compilePipelineEntry<true>(&entry, dumpPipelineStats); };
      if (dumpPipelineStats)
        lambda();
      else
        threads.emplace_back(std::move(lambda));
    }

    // Wait.
    for (std::thread& thread : threads)
    {
      thread.join();
    }
  }

  ~ComputeMipmapPipelinesImpl()
  {
    vkDestroyPipelineLayout(m_device, m_layout, nullptr);
    for (auto pair : m_generalPipelineMap)
    {
      vkDestroyPipeline(m_device, pair.second, nullptr);
    }
    for (auto pair : m_fastPipelineMap)
    {
      vkDestroyPipeline(m_device, pair.second, nullptr);
    }
  }

  // Record a command to generate mipmaps for the specified image
  // using info stored in the base level and the named pipeline
  // alternatives. No barrier is included before (i.e. it's your
  // responsibility), but a barrier is included after for read
  // visibility to fragment shaders.
  void cmdBindGenerate(VkCommandBuffer            cmdBuf,
                       const ScopedImage&         imageToMipmap,
                       const PipelineAlternative& alternative) override
  {
#ifdef USE_DEBUG_UTILS
    VkDebugUtilsLabelEXT labelInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
                                      nullptr, "mipmap_generation"};
    vkCmdBeginDebugUtilsLabelEXT(cmdBuf, &labelInfo);
#endif

    bool usingAlternative = alternative.fastAlternative.name != "default"
                            || alternative.fastAlternative.configBits != 0
                            || alternative.generalAlternative.name != "default"
                            || alternative.generalAlternative.configBits != 0;
    if (usingAlternative)
    {
      cmdBindGenerateAlternative(cmdBuf, imageToMipmap, alternative);
    }
    else
    {
      cmdBindGenerateDefault(cmdBuf, imageToMipmap);
    }
#ifdef USE_DEBUG_UTILS
    vkCmdEndDebugUtilsLabelEXT(cmdBuf);
#endif
  }

  // Typical user code for running the nvpro_pyramid shader. Bind
  // descriptors for image, call nvproCmdPyramidDispatch, and insert a
  // barrier after, for visibility.
  void cmdBindGenerateDefault(VkCommandBuffer    cmdBuf,
                              const ScopedImage& imageToMipmap)
  {
    VkDescriptorSet descriptorSets[] = {
        imageToMipmap.getTextureDescriptorSet(),
        imageToMipmap.getStorageDescriptorSet()};
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_layout,
                            0, arraySize(descriptorSets), descriptorSets,
                            0, nullptr);
    NvproPyramidPipelines pipelines;
    pipelines.generalPipeline = m_generalPipelineMap.at({"default", 0});
    pipelines.fastPipeline       = m_fastPipelineMap.at({"default", 0});
    pipelines.layout             = m_layout;
    pipelines.pushConstantOffset = 0;
    nvproCmdPyramidDispatch(cmdBuf, pipelines,
                            imageToMipmap.getImageWidth(),
                            imageToMipmap.getImageHeight());
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                               VK_ACCESS_SHADER_WRITE_BIT,
                               VK_ACCESS_MEMORY_READ_BIT};
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
  }

  // This is NOT typical usage of nvpro_pyramid; see above for that.
  void cmdBindGenerateAlternative(VkCommandBuffer            cmdBuf,
                                  const ScopedImage&         imageToMipmap,
                                  const PipelineAlternative& alternative)
  {
    VkDescriptorSet descriptorSets[] = {
        imageToMipmap.getTextureDescriptorSet(),
        imageToMipmap.getStorageDescriptorSet()};
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_layout,
                            0, arraySize(descriptorSets), descriptorSets,
                            0, nullptr);

    VkMemoryBarrier endBarrier = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT };
    auto barrierBeforePipelineStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    NvproPyramidPipelines pipelines{};
    pipelines.layout = m_layout;

    // Look up the compute pipeline and dispatcher for the fast pipeline,
    // if used.
    nvpro_pyramid_dispatcher_t fastDispatcher = nullptr;
    if (alternative.fastAlternative.name != "none")
    {
      const std::pair<std::string, uint32_t> key = {
          alternative.fastAlternative.basePipelineName.empty() ?
              alternative.fastAlternative.name :
              alternative.fastAlternative.basePipelineName,
          alternative.fastAlternative.configBits};
      auto pipelineIter = m_fastPipelineMap.find(key);
      assert(pipelineIter != m_fastPipelineMap.end());
      pipelines.fastPipeline = pipelineIter->second;

      fastDispatcher = getFastDispatcher(alternative.fastAlternative.name);
      if (fastDispatcher == nullptr)
      {
        assert(!"Fast dispatcher callback not found by name, check CMake was rerun and NVPRO_PYRAMID_ADD_FAST_DISPATCHER used (or I made a mistake)");
      }
    }

    if (alternative.generalAlternative.name != "blit")
    {
      // Same for the general pipeline, unless using blits.
      const std::pair<std::string, uint32_t> key = {
          alternative.generalAlternative.basePipelineName.empty() ?
              alternative.generalAlternative.name :
              alternative.generalAlternative.basePipelineName,
          alternative.generalAlternative.configBits};
      auto pipelineIter = m_generalPipelineMap.find(key);
      assert(pipelineIter != m_generalPipelineMap.end());
      pipelines.generalPipeline = pipelineIter->second;

      nvpro_pyramid_dispatcher_t generalDispatcher =
          getGeneralDispatcher(alternative.generalAlternative.name);
      if (generalDispatcher == nullptr)
      {
        assert(!"General dispatcher callback not found by name, check CMake was rerun and NVPRO_PYRAMID_ADD_GENERAL_DISPATCHER used (or I made a mistake)");
      }
      nvproCmdPyramidDispatch(
          cmdBuf, pipelines, imageToMipmap.getImageWidth(),
          imageToMipmap.getImageHeight(), 0u,
          generalDispatcher, fastDispatcher);
    }
    else
    {
      // Here we use blits when doing downsamples that don't meet
      // the divisibility requirements of the fast pipeline, instead
      // of the general pipeline, but still attempt to use the fast
      // pipeline when suitable. This provides an example of how this library
      // can interop with an alternative domain specific downsampler
      // that may trade "correctness" for performance.
      if (pipelines.fastPipeline)
      {
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelines.fastPipeline);
      }
      NvproPyramidState state;
      state.currentLevel    = 0;
      state.remainingLevels = imageToMipmap.getLevelCount() > 0 ?
                                  imageToMipmap.getLevelCount() - 1 : 0;
      state.currentX        = imageToMipmap.getImageWidth();
      state.currentY        = imageToMipmap.getImageHeight();

      VkMemoryBarrier betweenBarrier = {
          VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
          VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT};

      while (1)
      {
        // printf("NvproPyramidState{%u %u %u %u}\n", state.currentLevel,
        //        state.remainingLevels, state.currentX, state.currentY);
        uint32_t levelsDone = 0;
        if (fastDispatcher != nullptr)
        {
          // Dispatches the fast compute pipeline, iff the current
          // mip level's width, height, etc., allow for it.
          // NOTE: Typically, this is nvproPyramidDefaultFastDispatcher.
          levelsDone = fastDispatcher(
              cmdBuf, m_layout, 0,  // Push constant offset
              VK_NULL_HANDLE, // Pipeline, not supplied as already bound.
              state);
        }
        if (levelsDone != 0)
        {
          // Update progress of mipmap generation based on the
          // number of levels filled by fast pipeline.
          state.currentLevel += levelsDone;
          state.remainingLevels -= levelsDone;
          state.currentX >>= levelsDone;
          state.currentX = state.currentX == 0 ? 1 : state.currentX;
          state.currentY >>= levelsDone;
          state.currentY = state.currentY == 0 ? 1 : state.currentY;

          if (state.remainingLevels == 0)
          {
            break;
          }
        }
        else
        {
          // Fall back to blit if the fast pipeline could not run.
          // This only fills 1 additional level, of course.
          auto nextX = state.currentX >> 1;
          nextX = nextX == 0 ? 1 : nextX;
          auto nextY = state.currentY >> 1;
          nextY = nextY == 0 ? 1 : nextY;

          VkImageBlit blit = {
              {VK_IMAGE_ASPECT_COLOR_BIT, state.currentLevel, 0, 1},
              {{0, 0, 0},
              {int32_t(state.currentX), int32_t(state.currentY), 1}},
              {VK_IMAGE_ASPECT_COLOR_BIT, state.currentLevel + 1, 0, 1},
              {{0, 0, 0}, {int32_t(nextX), int32_t(nextY), 1}}};
          vkCmdBlitImage(cmdBuf,
                         imageToMipmap.getImage(), VK_IMAGE_LAYOUT_GENERAL,
                         imageToMipmap.getImage(), VK_IMAGE_LAYOUT_GENERAL,
                         1, &blit, VK_FILTER_LINEAR);

          // Update progress of mipmap generation.
          state.currentLevel++;
          state.remainingLevels--;
          state.currentX = nextX;
          state.currentY = nextY;

          if (state.remainingLevels == 0)
          {
            // If finishing now, note that we need the barrier for
            // the consumer fragment shader to use transfer instead
            // of compute accesses.
            barrierBeforePipelineStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            endBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            break;
          }
        }
        // Barrier between blit/compute iterations.
        vkCmdPipelineBarrier(
            cmdBuf,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &betweenBarrier, 0, nullptr, 0, nullptr);
      }
    }
    vkCmdPipelineBarrier(cmdBuf, barrierBeforePipelineStage,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                         1, &endBarrier, 0, nullptr, 0, nullptr);
  }
};

ComputeMipmapPipelines* ComputeMipmapPipelines::make(VkDevice           device,
                                                     const ScopedImage& image,
                                                     bool dumpPipelineStats)
{
  return new ComputeMipmapPipelinesImpl(device, image, dumpPipelineStats);
}
