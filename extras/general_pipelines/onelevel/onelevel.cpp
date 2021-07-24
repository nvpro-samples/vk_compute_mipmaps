
#include "nvpro_pyramid_dispatch_alternative.hpp"

static uint32_t onelevel_dispatch(VkCommandBuffer          cmdBuf,
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
  uint32_t srcLevel = state.currentLevel;
  uint32_t levels   = 1u;
  uint32_t pc       = srcLevel << nvproPyramidInputLevelShift | levels;
  vkCmdPushConstants(cmdBuf, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                     pushConstantOffset, sizeof pc, &pc);
  // Each workgroup handles up to 256 outputs.
  uint32_t dstWidth = state.currentX / 2u;
  dstWidth = dstWidth ? dstWidth : 1u;
  uint32_t dstHeight = state.currentY / 2u;
  dstHeight = dstHeight ? dstHeight : 1u;
  vkCmdDispatch(cmdBuf, (dstWidth * dstHeight + 255u) / 256u, 1u, 1u);
  return levels;
}

NVPRO_PYRAMID_ADD_GENERAL_DISPATCHER(onelevel, onelevel_dispatch)
