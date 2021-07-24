#ifndef VK_COMPUTE_MIPMAPS_PY2_DISPATCH_IMPL_
#define VK_COMPUTE_MIPMAPS_PY2_DISPATCH_IMPL_

template <uint32_t Warps, uint32_t TileWidth, uint32_t TileHeight, uint32_t MaxLevels = 2>
inline uint32_t py2_dispatch_impl(VkCommandBuffer          cmdBuf,
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
  uint32_t dstWidth  = state.currentX >> levels;
  dstWidth           = dstWidth ? dstWidth : 1u;
  uint32_t dstHeight = state.currentY >> levels;
  dstHeight          = dstHeight ? dstHeight : 1u;

  if (levels == 1u)
  {
    // Each thread writes one sample.
    uint32_t samples = dstWidth * dstHeight;
    uint32_t threads = Warps * 32u;
    vkCmdDispatch(cmdBuf, (samples + (threads - 1u)) / threads, 1u, 1u);
  }
  else
  {
    // Each workgroup handles a tile.
    uint32_t horizontalTiles = (dstWidth + (TileWidth - 1)) / TileWidth;
    uint32_t verticalTiles   = (dstHeight + (TileHeight - 1)) / TileHeight;
    vkCmdDispatch(cmdBuf, horizontalTiles * verticalTiles, 1u, 1u);
  }
  return levels;
}

#endif
