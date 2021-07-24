
#include "nvpro_pyramid_dispatch_alternative.hpp"

template <uint32_t DivisibilityRequirement, uint32_t PipelineMaxLevels = 6>
static uint32_t workgroup1024_dispatch(VkCommandBuffer  cmdBuf,
                                       VkPipelineLayout layout,
                                       uint32_t         pushConstantOffset,
                                       VkPipeline       pipelineIfNeeded,
                                       const NvproPyramidState& state)
{
  static_assert(DivisibilityRequirement % 2 == 0, "Handles even size images");
  bool success = state.currentX % DivisibilityRequirement == 0u
                 && state.currentY % DivisibilityRequirement == 0u;
  if (success)
  {
    if (pipelineIfNeeded)
    {
      vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipelineIfNeeded);
    }

    // Choose the number of levels to fill.
    uint32_t x = state.currentX, y = state.currentY;
    uint32_t levels = 0u;
    while (x % 2u == 0u && y % 2u == 0u && levels < state.remainingLevels
           && levels < PipelineMaxLevels)
    {
      static_assert(PipelineMaxLevels <= 6, "Can handle up to 6 levels");
      x /= 2u;
      y /= 2u;
      levels++;
    }
    uint32_t srcLevel = state.currentLevel;
    uint32_t pc       = srcLevel << nvproPyramidInputLevelShift | levels;
    vkCmdPushConstants(cmdBuf, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       pushConstantOffset, sizeof pc, &pc);
    // Each workgroup handles up to 4096 input samples.
    uint32_t samples = state.currentX * state.currentY;
    vkCmdDispatch(cmdBuf, (samples + 4095u) / 4096u, 1u, 1u);
    return levels;
  }
  return 0u;
}

NVPRO_PYRAMID_ADD_FAST_DISPATCHER(workgroup1024, workgroup1024_dispatch<2>)
