
#include "nvpro_pyramid_dispatch_alternative.hpp"

static uint32_t quad4_dispatch(VkCommandBuffer          cmdBuf,
                               VkPipelineLayout         layout,
                               uint32_t                 pushConstantOffset,
                               VkPipeline               pipelineIfNeeded,
                               const NvproPyramidState& state)
{
  bool success = state.currentX % 16u == 0u && state.currentY % 16u == 0u
                 && state.remainingLevels >= 4u;
  if (success)
  {
    if (pipelineIfNeeded)
    {
      vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipelineIfNeeded);
    }
    uint32_t levels   = 4u;
    uint32_t srcLevel = state.currentLevel;
    uint32_t pc       = srcLevel << nvproPyramidInputLevelShift | levels;
    vkCmdPushConstants(cmdBuf, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       pushConstantOffset, sizeof pc, &pc);
    // Each workgroup handles up to 16 16x16 input tile.
    uint32_t tiles = state.currentX * state.currentY / 256u;
    vkCmdDispatch(cmdBuf, (tiles + 15u) / 16u, 1u, 1u);
    return levels;
  }
  return 0u;
}

NVPRO_PYRAMID_ADD_FAST_DISPATCHER(quad4, quad4_dispatch)
