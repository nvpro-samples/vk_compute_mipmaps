#include "nvpro_pyramid_dispatch_alternative.hpp"

static uint32_t baseline_dispatch(VkCommandBuffer          cmdBuf,
                                  VkPipelineLayout         layout,
                                  uint32_t                 pushConstantOffset,
                                  VkPipeline               pipelineIfNeeded,
                                  const NvproPyramidState& state)
{
  if (pipelineIfNeeded)
  {
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pipelineIfNeeded);
  }
  assert(state.currentLevel == 0);
  uint32_t levels = state.remainingLevels;
  vkCmdPushConstants(cmdBuf, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                     pushConstantOffset, sizeof levels, &levels);
  uint32_t dstWidth  = state.currentX >= 1 ? state.currentX >> 1 : 1u;
  uint32_t dstHeight = state.currentY >= 1 ? state.currentY >> 1 : 1u;
  vkCmdDispatch(cmdBuf, (dstWidth * dstHeight + 255u) / 256u, 1, 1);
  return levels;
}

NVPRO_PYRAMID_ADD_GENERAL_DISPATCHER(baseline, baseline_dispatch)
