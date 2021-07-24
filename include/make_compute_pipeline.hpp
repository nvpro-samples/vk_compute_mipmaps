// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_MAKE_COMPUTE_PIPELINE_HPP_
#define NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_MAKE_COMPUTE_PIPELINE_HPP_

#include <vulkan/vulkan.h>
#include "nvh/fileoperations.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "search_paths.hpp"

// Create a compute pipeline from the given pipeline layout and
// compute shader module. "main" is the entrypoint function.
inline void makeComputePipeline(VkDevice         device,
                                VkShaderModule   shaderModule,
                                bool             dumpPipelineStats,
                                VkPipelineLayout layout,
                                VkPipeline*      outPipeline,
                                const char* pShaderName = "<generated shader>")
{
  // Shader module must then get packaged into a <shader stage>
  // This is just an ordinary struct, not a Vulkan object.
  VkPipelineShaderStageCreateInfo stageInfo {
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    nullptr,
    0,                                // * Must be 0 by Vulkan spec
    VK_SHADER_STAGE_COMPUTE_BIT,      // * Type of shader (compute shader)
    shaderModule,                     // * Shader module
    "main",                           // * Name of function to call
    nullptr };                        // * I don't use this

  // Create the compute pipeline. Note that the create struct is
  // typed for different pipeline types (compute, rasterization, ray
  // trace, etc.), yet the VkPipeline output type is the same for all.
  VkComputePipelineCreateInfo pipelineInfo {
    VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    nullptr,
    0,
    stageInfo,                // * The compute shader to use
    layout,                   // * Pipeline Layout
    VK_NULL_HANDLE, 0 };      // * Unused advanced feature (pipeline caching)
  if (dumpPipelineStats)
  {
    pipelineInfo.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  }
  NVVK_CHECK(vkCreateComputePipelines(
    device,
    VK_NULL_HANDLE,           // * Unused (pipeline caching)
    1, &pipelineInfo,         // * Array of pipelines to create
    nullptr,                  // * Default host memory allocator
    outPipeline));            // * Pipeline output (array)

  if (dumpPipelineStats)
  {
    nvvk::nvprintPipelineStats(device, *outPipeline, pShaderName, false);
  }
}



// Create a compute pipeline from the given pipeline layout and with
// SPIR-V code loaded from the named file. "main" is the entrypoint
// function.
inline void makeComputePipeline(VkDevice         device,
                                const char*      pFilename,
                                bool             dumpPipelineStats,
                                VkPipelineLayout layout,
                                VkPipeline*      outPipeline)
{
  // Compile SPV shader into a shader module.
  std::string shaderCode = nvh::loadFile(
    pFilename,                // * SPV file name
    true,                     // * Is binary file (needed on Windows)
    searchPaths,              // * Directories to search in
    true);                    // * Warn if not found
  assert(shaderCode.size() > 0);
  VkShaderModuleCreateInfo moduleCreateInfo {
    VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    nullptr,
    0,                                     // * Must be 0 by Vulkan spec
    shaderCode.size(),                     // * SPV code size in bytes
    (const uint32_t*) shaderCode.data() }; // * Pointer to SPV code
  VkShaderModule shaderModule;
  NVVK_CHECK(vkCreateShaderModule(
    device, &moduleCreateInfo, nullptr, &shaderModule));

  makeComputePipeline(device, shaderModule, dumpPipelineStats, layout,
                      outPipeline, pFilename);

  vkDestroyShaderModule(device, shaderModule, nullptr);
}


// Create a compute pipeline and layout from the given
// descriptor/push constant info and with SPIR-V code loaded from
// the named file. "main" is the entrypoint function.
inline void makeComputePipeline(VkDevice                     device,
                                const char*                  pFilename,
                                bool                         dumpPipelineStats,
                                uint32_t                     layoutCount,
                                const VkDescriptorSetLayout* pLayouts,
                                uint32_t                     rangeCount,
                                const VkPushConstantRange*   pRanges,
                                VkPipeline*                  outPipeline,
                                VkPipelineLayout*            outPipelineLayout)
{
  // Make pipeline layout.
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0,
      layoutCount, pLayouts, rangeCount, pRanges };
  NVVK_CHECK(vkCreatePipelineLayout(
      device, &pipelineLayoutInfo, nullptr, outPipelineLayout));

  makeComputePipeline(device, pFilename, dumpPipelineStats, *outPipelineLayout,
                      outPipeline);
}

#endif
