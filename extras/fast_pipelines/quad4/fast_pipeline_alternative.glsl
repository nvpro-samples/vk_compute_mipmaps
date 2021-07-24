// Efficient special case image pyramid generation kernel.  Generates
// 4 layers at once: each 16 consectively-numbered threads take 16x16
// tiles of the input mip level, and reduces it to 8x8, 4x4, 2x2, and
// 1x1 tiles written to the next 4 mip levels. Subgroup size must be
// at least 16.
//
// This only works when the source level's size is a multiple of 16x16
// and there are 4 mip levels to generate.
//
// TODO: Test if this works for subgroup size != 32.
layout(local_size_x = 256) in; // 16 teams
void nvproPyramidMain()
{
  // Calculate which 16x16 tile is input for these 16 threads.
  int   inputLevel_      = NVPRO_PYRAMID_INPUT_LEVEL_;
  uint  tilesPerGroup_   = gl_WorkGroupSize.x / 16u;
  ivec2 srcImageSize_    = NVPRO_PYRAMID_LEVEL_SIZE(inputLevel_);
  uint  horizontalTiles_ = uint(srcImageSize_.x) / 16u;
  uint  verticalTiles_   = uint(srcImageSize_.y) / 16u;
  uint  tileIndex_       = tilesPerGroup_ * gl_WorkGroupID.x
                         + gl_LocalInvocationID.x / 16u;
  uint  horizontalIndex_ = tileIndex_ % horizontalTiles_;
  uint  verticalIndex_   = tileIndex_ / horizontalTiles_;
  ivec2 tileOffset_      = 16 * ivec2(horizontalIndex_, verticalIndex_);

  // Bounds-check.
  if (tileIndex_ >= verticalTiles_ * horizontalTiles_) return;

  // Break the input tile into 16 sub tiles.  Each thread generates
  // 4 samples (from 16 inputs), writes to inputLevel_ + 1.
  // This means each thread produces the inputs it needs to generate
  // 1 sample for inputLevel_ + 2.
  // +---+---++---+---+
  // | 0 | 1 || 4 | 5 |
  // +---+---++---+---+
  // | 2 | 3 || 6 | 7 |
  // +===+===++===+===+
  // | 8 | 9 ||12 |13 |
  // +---+---++---+---+
  // |10 |11 ||14 |15 |
  // +---+---++---+---+

  NVPRO_PYRAMID_TYPE sample00_, sample01_, sample10_, sample11_;

  // Relative location of sub tile.
  uint idxInTeam_     = gl_LocalInvocationIndex % 16;
  ivec2 threadOffset_ = ivec2((idxInTeam_ & 1) << 2 | (idxInTeam_ & 4) << 1,
                              (idxInTeam_ & 2) << 1 | (idxInTeam_ & 8));
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

  // Compute 4x4 tile in level inputLevel_ + 2; one sample per thread.
  // Outputs from before are now inputs.
  NVPRO_PYRAMID_REDUCE4(sample00_, sample01_, sample10_, sample11_, out_);
  NVPRO_PYRAMID_STORE((subTileOffset_ >> 2), (inputLevel_+2), out_);

  // Compute 2x2 tile in level inputLevel_ + 3; only 1 out of every 4
  // threads does this. Use shuffle to get the needed data from the
  // other three threads.
  sample00_ = out_;
  sample01_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 1);
  sample10_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 2);
  sample11_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 3);

  if (0 == (gl_SubgroupInvocationID & 3))
  {
    NVPRO_PYRAMID_REDUCE4(sample00_, sample01_, sample10_, sample11_, out_);
    NVPRO_PYRAMID_STORE((subTileOffset_ >> 3), (inputLevel_+3), out_);
  }

  // Compute 1x1 "tile" in level inputLevel_ + 4; only 1 thread per team
  // of 16 does this. Shuffle again.
  sample00_ = out_;
  sample01_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 4);
  sample10_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 8);
  sample11_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 12);

  if (0 == (gl_SubgroupInvocationID & 15))
  {
    NVPRO_PYRAMID_REDUCE4(sample00_, sample01_, sample10_, sample11_, out_);
    NVPRO_PYRAMID_STORE((subTileOffset_ >> 4), (inputLevel_+4), out_);
  }
}
