// Efficient special case image pyramid generation kernel.  Generates
// three layers at once: each 4 consectively-numbered threads take 8x8
// tiles of the input mip level, and reduces it to 4x4, 2x2, and 1x1
// tiles written to the next 3 mip levels. Subgroup size must be >= 4.
//
// This only works when the source level's size is a multiple of 8x8
// and there are 3 mip levels to generate; see nvproPyramidMain when
// these conditions are not satisfied.
//
// TODO: Test if this works for subgroup size != 32.
layout(local_size_x = 256) in;
void nvproPyramidMain()
{
  // Calculate which 8x8 tile is input for these 4 threads.
  int   inputLevel_      = NVPRO_PYRAMID_INPUT_LEVEL_;
  uint  tilesPerGroup_   = gl_WorkGroupSize.x / 4u;
  ivec2 srcImageSize_    = NVPRO_PYRAMID_LEVEL_SIZE(inputLevel_);
  uint  horizontalTiles_ = uint(srcImageSize_.x) / 8u;
  uint  verticalTiles_   = uint(srcImageSize_.y) / 8u;
  uint  tileIndex_       = tilesPerGroup_ * gl_WorkGroupID.x
                         + gl_LocalInvocationID.x / 4u;
  uint  horizontalIndex_ = tileIndex_ % horizontalTiles_;
  uint  verticalIndex_   = tileIndex_ / horizontalTiles_;
  ivec2 tileOffset_      = 8 * ivec2(horizontalIndex_, verticalIndex_);

  // Bounds-check.
  if (tileIndex_ >= verticalTiles_ * horizontalTiles_) return;

  // Compute 4x4 tile in level firstLevel_ + 1.
  // Think ahead to the 2x2 generation step: each thread generates the 4
  // inputs needed for its own part of the 2x2 output tile.
  NVPRO_PYRAMID_TYPE sample00_, sample01_, sample10_, sample11_;

  // Divide the 8x8 input tile and 4x4 output tile between 4 threads.
  ivec2 threadOffset_ = ivec2(4 * (1 & gl_SubgroupInvocationID),
                              2 * (2 & gl_SubgroupInvocationID));

  ivec2 subTileOffset_ = tileOffset_ + threadOffset_;
  ivec2 srcCoord_;

  // Calculate future upper-left input.
  srcCoord_ = subTileOffset_ + ivec2(0, 0);
  NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord_, inputLevel_, sample00_);
  NVPRO_PYRAMID_STORE((srcCoord_ >> 1), (inputLevel_+1), sample00_);

  // Calculate future lower-left input.
  srcCoord_ = subTileOffset_ + ivec2(0, 2);
  NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord_, inputLevel_, sample01_);
  NVPRO_PYRAMID_STORE((srcCoord_ >> 1), (inputLevel_+1), sample01_);

  // Calculate future upper-right input.
  srcCoord_ = subTileOffset_ + ivec2(2, 0);
  NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord_, inputLevel_, sample10_);
  NVPRO_PYRAMID_STORE((srcCoord_ >> 1), (inputLevel_+1), sample10_);

  // Calculate future lower-right input.
  srcCoord_ = subTileOffset_ + ivec2(2, 2);
  NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord_, inputLevel_, sample11_);
  NVPRO_PYRAMID_STORE((srcCoord_ >> 1), (inputLevel_+1), sample11_);

  NVPRO_PYRAMID_TYPE out_;

  // Compute 2x2 tile in level firstLevel + 2; one sample per thread.
  // Outputs from before are now inputs.
  NVPRO_PYRAMID_REDUCE4(sample00_, sample01_, sample10_, sample11_, out_);
  NVPRO_PYRAMID_STORE((subTileOffset_ >> 2), (inputLevel_+2), out_);

  // Compute 1x1 "tile" in level firstLevel + 3; only one thread out
  // of the four assigned per tile does this. Use shuffle to get the
  // needed data from the other three threads.
  sample00_ = out_;
  sample01_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 1);
  sample10_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 2);
  sample11_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 3);

  if (0 == (gl_SubgroupInvocationID & 3))
  {
    NVPRO_PYRAMID_REDUCE4(sample00_, sample01_, sample10_, sample11_, out_);
    NVPRO_PYRAMID_STORE((tileOffset_ >> 3), (inputLevel_+3), out_);
  }
}
