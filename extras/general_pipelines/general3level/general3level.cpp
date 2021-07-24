
#include "nvpro_pyramid_dispatch_alternative.hpp"

static uint32_t general3level_dispatch(VkCommandBuffer  cmdBuf,
                                       VkPipelineLayout layout,
                                       uint32_t         pushConstantOffset,
                                       VkPipeline       pipelineIfNeeded,
                                       const NvproPyramidState& state)
{
  if (pipelineIfNeeded)
  {
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipelineIfNeeded);
  }
  uint32_t levels   = state.remainingLevels >= 3u ? 3u : state.remainingLevels;
  uint32_t srcLevel = state.currentLevel;
  uint32_t pc       = srcLevel << nvproPyramidInputLevelShift | levels;
  vkCmdPushConstants(cmdBuf, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                     pushConstantOffset, sizeof pc, &pc);
  // Each workgroup handles an 8x8 output tile.
  uint32_t dstWidth        = state.currentX >> levels;
  dstWidth                 = dstWidth ? dstWidth : 1u;
  uint32_t horizontalTiles = (dstWidth + 7u) / 8u;

  uint32_t dstHeight     = state.currentY >> levels;
  dstHeight              = dstHeight ? dstHeight : 1u;
  uint32_t verticalTiles = (dstHeight + 7u) / 8u;

  vkCmdDispatch(cmdBuf, horizontalTiles * verticalTiles, 1u, 1u);
  return levels;
}

NVPRO_PYRAMID_ADD_GENERAL_DISPATCHER(general3level, general3level_dispatch)
