
#include "nvpro_pyramid_dispatch_alternative.hpp"

static uint32_t onelevel_dispatch(VkCommandBuffer          cmdBuf,
                                  VkPipelineLayout         layout,
                                  uint32_t                 pushConstantOffset,
                                  VkPipeline               pipelineIfNeeded,
                                  const NvproPyramidState& state)
{
  bool success = state.currentX % 2u == 0u && state.currentY % 2u == 0u;
  if (success)
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
    // Each workgroup handles up to 1024 input samples.
    uint32_t samples = state.currentX * state.currentY;
    vkCmdDispatch(cmdBuf, (samples + 1023u) / 1024u, 1u, 1u);
    return levels;
  }
  return 0u;
}

NVPRO_PYRAMID_ADD_FAST_DISPATCHER(onelevel, onelevel_dispatch)
