// Copyright 2021-2024 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_COMPUTE_MIPMAPS_NVPRO_PYRAMID_DISPATCH_HPP_
#define NVPRO_SAMPLES_COMPUTE_MIPMAPS_NVPRO_PYRAMID_DISPATCH_HPP_

#include <assert.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

// Struct for passing the pipelines and associated data for the mipmap
// dispatch function.
//
// generalPipeline: compute pipeline, created as described in
// nvpro_pyramid.glsl with NVPRO_PYRAMID_IS_FAST_PIPELINE defined as 0.
// Mandatory; see nvproPyramidDefaultFastDispatcher if you want to use
// fastPipeline alone and handle the non-fast case yourself in an alternate way.
//
// fastPipeline: optional (may be VK_NULL_HANDLE).
// Compute pipeline, created as described in nvpro_pyramid.glsl
// with NVPRO_PYRAMID_IS_FAST_PIPELINE defined as nonzero.
// Must be VK_NULL_HANDLE if the executing device lacks the needed
// features (see nvpro_pyramid.glsl).
//
// layout: shared pipeline layout for both pipelines.
//
// pushConstantOffset: offset of the 32-bit push constant needed by
// nvpro_pyramid.glsl; this must be 0 if the user did not manually
// override the default push constant by defining NVPRO_PYRAMID_PUSH_CONSTANT.
struct NvproPyramidPipelines
{
  VkPipeline       generalPipeline    = VK_NULL_HANDLE;
  VkPipeline       fastPipeline       = VK_NULL_HANDLE;
  VkPipelineLayout layout             = VK_NULL_HANDLE;
  uint32_t         pushConstantOffset = 0;
};

// Record commands for dispatching the compute shaders in NvproPyramidPipelines
// that are appropriate for an image with the given base mip width,
// height, and mip levels (defaults to the maximum number of mip
// levels theoretically allowed for the given image size).
//
// This handles:
//
// * Recording dispatch commands
// * Binding compute pipelines
// * Inserting appropriate barriers strictly between dispatches
//
// The caller is responsible for:
//
// * Performing any needed synchronization before and after
// * Binding any needed descriptor sets
// * Setting any needed push constants, except the push constant declared
//   by NVPRO_PYRAMID_PUSH_CONSTANT (if any)
inline void nvproCmdPyramidDispatch(
    VkCommandBuffer       cmdBuf,
    NvproPyramidPipelines pipelines,
    uint32_t              baseWidth,
    uint32_t              baseHeight,
    uint32_t              mipLevels = 0U);

// Struct used for tracking the progress of scheduling mipmap
// generation commands.
struct NvproPyramidState
{
  // Input level for the next dispatch.
  uint32_t currentLevel = 0;

  // Levels that remain to be filled, i.e.
  // nvproCmdPyramidDispatch::mipLevels - currentLevel - 1.
  // Will never be 0 when passed to an nvpro_pyramid_dispatcher_t instance.
  uint32_t remainingLevels = 0;

  // Width and height of mip level currentLevel.
  uint32_t currentX = 0, currentY = 0;
};

constexpr uint32_t nvproPyramidInputLevelShift = 5U;  // TODO Use consistently


// Callback host function for a pipeline. Attempt to record commands
// for one bind and dispatch of the given pipeline, which may be
// VK_NULL_HANDLE (to indicate that the pipeline is already bound and
// need not be bound again). This function should not record any barriers.
//
// The return value is the number of mip levels filled by the dispatch.
//
// If this is a callback for a fast pipeline, this may fail (return 0)
// if the pipeline is not suitable (e.g. the current mip level fails
// some divisibility requirements).
//
// Callbacks for general pipelines must never return 0.
//
// This function may set the 32-bit push constant at offset
// `pushConstantOffset` (and no other push constant). It may use this
// push constant as it sees fit, but the NVPRO_PYRAMID_INPUT_LEVEL_
// and NVPRO_PYRAMID_LEVEL_COUNT_ macros assume
//
// { input level } << nvproPyramidInputLevelShift | { levels filled }
using nvpro_pyramid_dispatcher_t = uint32_t (*)(
    VkCommandBuffer          cmdBuf,
    VkPipelineLayout         layout,
    uint32_t                 pushConstantOffset,
    VkPipeline               pipelineIfNeeded,
    const NvproPyramidState& state);

// Base implementation function for the typical user-facing
// nvproCmdPyramidDispatch. Try to use the fastPipeline if possible,
// then fall back to the general pipeline if not usable.
inline void nvproCmdPyramidDispatch(
    VkCommandBuffer            cmdBuf,
    NvproPyramidPipelines      pipelines,
    uint32_t                   baseWidth,
    uint32_t                   baseHeight,
    uint32_t                   mipLevels,
    nvpro_pyramid_dispatcher_t generalDispatcher,
    nvpro_pyramid_dispatcher_t fastDispatcher)
{
  const VkMemoryBarrier barrier{
      .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
  if(mipLevels == 0)
  {
    uint32_t srcWidth  = baseWidth;
    uint32_t srcHeight = baseHeight;
    while(srcWidth != 0 || srcHeight != 0)
    {
      srcWidth >>= 1;
      srcHeight >>= 1;
      ++mipLevels;
    }
  }
  NvproPyramidState state;
  state.currentLevel    = 0U;
  state.remainingLevels = mipLevels - 1U;
  state.currentX        = baseWidth;
  state.currentY        = baseHeight;

  VkPipeline fastPipelineIfNeeded    = pipelines.fastPipeline;
  VkPipeline generalPipelineIfNeeded = pipelines.generalPipeline;

  while(true)
  {
    uint32_t levelsDone = 0;

    // Try to use the fast pipeline if possible.
    if(pipelines.fastPipeline)
    {
      levelsDone = fastDispatcher(
          cmdBuf, pipelines.layout, pipelines.pushConstantOffset,
          fastPipelineIfNeeded, state);
    }

    if(levelsDone != 0)
    {
      fastPipelineIfNeeded    = VK_NULL_HANDLE;
      generalPipelineIfNeeded = pipelines.generalPipeline;
    }
    else
    {
      // Otherwise fall back on general pipeline.
      levelsDone = generalDispatcher(
          cmdBuf, pipelines.layout, pipelines.pushConstantOffset,
          generalPipelineIfNeeded, state);

      fastPipelineIfNeeded    = pipelines.fastPipeline;
      generalPipelineIfNeeded = VK_NULL_HANDLE;
    }
    assert(levelsDone != 0);

    // Update the progress.
    assert(levelsDone <= state.remainingLevels);
    state.currentLevel += levelsDone;
    state.remainingLevels -= levelsDone;
    state.currentX >>= levelsDone;
    state.currentX = (state.currentX != 0) ? state.currentX : 1U;
    state.currentY >>= levelsDone;
    state.currentY = (state.currentY != 0) ? state.currentY : 1U;

    // Put barriers only between dispatches.
    if(state.remainingLevels == 0U)
      break;
    vkCmdPipelineBarrier(
        cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
        nullptr);
  }
}


// nvpro_pyramid_dispatcher_t implementation for nvpro_pyramid.glsl
// shaders with NVPRO_PYRAMID_IS_FAST_PIPELINE != 0
//
// Note: this function is referenced by name in ComputeMipmapPipeline::cmdBindGenerate
template <uint32_t DivisibilityRequirement = 4, uint32_t MaxLevels = 6>
static uint32_t nvproPyramidDefaultFastDispatcher(
    VkCommandBuffer          cmdBuf,
    VkPipelineLayout         layout,
    uint32_t                 pushConstantOffset,
    VkPipeline               pipelineIfNeeded,
    const NvproPyramidState& state)
{
  // For maybequad pipeline.
  static_assert(
      DivisibilityRequirement > 0 && DivisibilityRequirement % 2 == 0,
      "Can only handle even sizes.");
  const bool success = state.currentX % DivisibilityRequirement == 0U
                       && state.currentY % DivisibilityRequirement == 0U;
  if(success)
  {
    if(pipelineIfNeeded)
    {
      vkCmdBindPipeline(
          cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineIfNeeded);
    }

    // Choose the number of levels to fill.
    static_assert(MaxLevels <= 6, "Can only handle up to 6 levels");
    uint32_t x      = state.currentX;
    uint32_t y      = state.currentY;
    uint32_t levels = 0U;
    while(x % 2U == 0U && y % 2U == 0U && levels < state.remainingLevels
          && levels < MaxLevels)
    {
      x /= 2U;
      y /= 2U;
      levels++;
    }
    const uint32_t srcLevel = state.currentLevel;
    uint32_t       pc       = srcLevel << nvproPyramidInputLevelShift | levels;
    vkCmdPushConstants(
        cmdBuf, layout, VK_SHADER_STAGE_COMPUTE_BIT, pushConstantOffset,
        sizeof pc, &pc);
    // Each workgroup handles up to 4096 input samples if levels > 5; 1024 otherwise.
    const uint32_t shift   = levels > 5 ? 12U : 10U;
    const uint32_t mask    = levels > 5 ? 4095U : 1023U;
    const uint32_t samples = state.currentX * state.currentY;
    vkCmdDispatch(cmdBuf, (samples + mask) >> shift, 1U, 1U);
    return levels;
  }
  return 0U;
}


// nvpro_pyramid_dispatcher_t implementation for nvpro_pyramid.glsl
// shaders with NVPRO_PYRAMID_IS_FAST_PIPELINE == 0
inline uint32_t nvproPyramidDefaultGeneralDispatcher(
    VkCommandBuffer          cmdBuf,
    VkPipelineLayout         layout,
    uint32_t                 pushConstantOffset,
    VkPipeline               pipelineIfNeeded,
    const NvproPyramidState& state)
{
  // Use py2_4_8_8 pipeline parameters.
  constexpr uint32_t MaxLevels  = 2;
  constexpr uint32_t Warps      = 4;
  constexpr uint32_t TileWidth  = 8;
  constexpr uint32_t TileHeight = 8;

  if(pipelineIfNeeded)
  {
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineIfNeeded);
  }
  static_assert(MaxLevels <= 2U && MaxLevels != 0, "can do 1 or 2 levels");
  const uint32_t levels =
      state.remainingLevels >= MaxLevels ? MaxLevels : state.remainingLevels;
  const uint32_t srcLevel = state.currentLevel;
  const uint32_t pc       = srcLevel << nvproPyramidInputLevelShift | levels;
  vkCmdPushConstants(
      cmdBuf, layout, VK_SHADER_STAGE_COMPUTE_BIT, pushConstantOffset,
      sizeof pc, &pc);
  uint32_t dstWidth  = state.currentX >> levels;
  dstWidth           = (dstWidth != 0) ? dstWidth : 1U;
  uint32_t dstHeight = state.currentY >> levels;
  dstHeight          = (dstHeight != 0) ? dstHeight : 1U;

  if(levels == 1U)
  {
    // Each thread writes one sample.
    const uint32_t samples = dstWidth * dstHeight;
    const uint32_t threads = Warps * 32U;
    vkCmdDispatch(cmdBuf, (samples + (threads - 1U)) / threads, 1U, 1U);
  }
  else
  {
    // Each workgroup handles a tile.
    const uint32_t horizontalTiles = (dstWidth + (TileWidth - 1)) / TileWidth;
    const uint32_t verticalTiles = (dstHeight + (TileHeight - 1)) / TileHeight;
    vkCmdDispatch(cmdBuf, horizontalTiles * verticalTiles, 1U, 1U);
  }
  return levels;
}

inline void nvproCmdPyramidDispatch(
    VkCommandBuffer       cmdBuf,
    NvproPyramidPipelines pipelines,
    uint32_t              baseWidth,
    uint32_t              baseHeight,
    uint32_t              mipLevels)
{
  nvproCmdPyramidDispatch(
      cmdBuf, pipelines, baseWidth, baseHeight, mipLevels,
      nvproPyramidDefaultGeneralDispatcher, nvproPyramidDefaultFastDispatcher);
}
#endif
