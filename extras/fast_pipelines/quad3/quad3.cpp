
#include "nvpro_pyramid_dispatch_alternative.hpp"

static uint32_t quad3_dispatch(VkCommandBuffer          cmdBuf,
                               VkPipelineLayout         layout,
                               uint32_t                 pushConstantOffset,
                               VkPipeline               fastPipelineIfNeeded,
                               const NvproPyramidState& state)
{
  bool success = state.currentX % 8u == 0u && state.currentY % 8u == 0u
                 && state.remainingLevels >= 3u;
  if (success)
  {
    if (fastPipelineIfNeeded)
    {
      vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                        fastPipelineIfNeeded);
    }
    uint32_t levels   = 3u;
    uint32_t srcLevel = state.currentLevel;
    uint32_t pc       = srcLevel << nvproPyramidInputLevelShift | levels;
    vkCmdPushConstants(cmdBuf, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       pushConstantOffset, sizeof pc, &pc);
    // Each workgroup handles up to 64 last-level output samples.
    uint32_t dstWidth  = state.currentX >> levels;
    dstWidth           = dstWidth ? dstWidth : 1u;
    uint32_t dstHeight = state.currentY >> levels;
    dstHeight          = dstHeight ? dstHeight : 1u;

    vkCmdDispatch(cmdBuf, (dstWidth * dstHeight + 63u) / 64u, 1u, 1u);
    return levels;
  }
  return 0u;
}

NVPRO_PYRAMID_ADD_FAST_DISPATCHER(quad3, quad3_dispatch)
