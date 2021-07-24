#include "nvpro_pyramid_dispatch_alternative.hpp"

template <uint32_t MaxLevels = 2u>
static uint32_t general2_dispatch(VkCommandBuffer          cmdBuf,
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
  static_assert(MaxLevels <= 2u && MaxLevels != 0, "can do 1 or 2 levels");
  uint32_t levels =
      state.remainingLevels >= MaxLevels ? MaxLevels : state.remainingLevels;
  uint32_t srcLevel = state.currentLevel;
  uint32_t pc       = srcLevel << nvproPyramidInputLevelShift | levels;
  vkCmdPushConstants(cmdBuf, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                     pushConstantOffset, sizeof pc, &pc);
  uint32_t dstWidth        = state.currentX >> levels;
  dstWidth                 = dstWidth ? dstWidth : 1u;
  uint32_t dstHeight     = state.currentY >> levels;
  dstHeight              = dstHeight ? dstHeight : 1u;

  if (levels == 1u)
  {
    // Each workgroup handles 384 output samples.
    uint32_t samples = dstWidth * dstHeight;
    vkCmdDispatch(cmdBuf, (samples + 383u) / 384u, 1u, 1u);
  }
  else
  {
    // Each workgroup handles a 16x16 tile.
    uint32_t horizontalTiles = (dstWidth + 15u) / 16u;
    uint32_t verticalTiles = (dstHeight + 15u) / 16u;
    vkCmdDispatch(cmdBuf, horizontalTiles * verticalTiles, 1u, 1u);
  }
  return levels;
}

NVPRO_PYRAMID_ADD_GENERAL_DISPATCHER(general2, general2_dispatch)
NVPRO_PYRAMID_ADD_GENERAL_DISPATCHER(general2max1, general2_dispatch<1u>)
