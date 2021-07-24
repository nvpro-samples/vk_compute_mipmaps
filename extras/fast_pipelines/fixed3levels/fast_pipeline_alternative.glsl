// Efficient special case image pyramid generation kernel.  Generates
// up to 3 levels at once: each workgroup reads up to 1024 samples
// of the input mip level and generates the resulting samples for the next
// up to 5 levels. Dispatch with y, z = 1.
//
// This only works when the input mip level has edges divisible by 2
// to the power of NVPRO_PYRAMID_LEVEL_COUNT_, and
// NVPRO_PYRAMID_LEVEL_COUNT_ can be at most 5.
//
// TODO: Test if this works for subgroup size != 32.

layout(local_size_x = 256) in;

// Handle the tile at the given input level and offset (position of
// upper-left corner), and write out the resulting minified tiles in
// the next 3 mip levels. Tiles are squares with edge length 8.
//
// Must be executed by 16 consecutive threads with same inputs, with
// the lowest thread number being a multiple of 16.
void handleTile_(ivec2 srcTileOffset_, int inputLevel_)
{
  // Break the input tile into 16 2x2 sub-tiles.  Each thread in the team
  // generates a samples (from 4 inputs), writes them to inputLevel_ + 1.
  // +---+---++---+---+
  // | 0 | 1 || 4 | 5 |
  // +---+---++---+---+
  // | 2 | 3 || 6 | 7 |
  // +===+===++===+===+
  // | 8 | 9 ||12 |13 |
  // +---+---++---+---+
  // |10 |11 ||14 |15 |
  // +---+---++---+---+
  NVPRO_PYRAMID_TYPE out_;

  // Relative location of sub-tile.
  uint teamMask_   = 15u;
  uint idxInTeam_  = gl_LocalInvocationIndex & teamMask_;
  ivec2 subOffset_ = ivec2((idxInTeam_ & 1) << 1 | (idxInTeam_ & 4),
                           (idxInTeam_ & 2) | (idxInTeam_ & 8) >> 1);

  // Calculate the assigned sample for inputLevel_ + 1
  ivec2 dstCoord_;
  dstCoord_  = ((srcTileOffset_ + subOffset_) >> 1) + ivec2(0, 0);
  NVPRO_PYRAMID_LOAD_REDUCE4((dstCoord_ << 1), inputLevel_, out_);
  NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+1), out_);

  NVPRO_PYRAMID_TYPE in00_, in01_, in10_, in11_;

  // Compute 2x2 tile in level inputLevel_ + 2; only 1 out of every 4
  // threads does this. Use shuffle to get the needed data from the
  // other three threads.
  dstCoord_ >>= 1;
  in00_ = out_;
  in01_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 1);
  in10_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 2);
  in11_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 3);

  if (0 == (gl_SubgroupInvocationID & 3))
  {
    NVPRO_PYRAMID_REDUCE4(in00_, in01_, in10_, in11_, out_);
    NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+2), out_);
  }

  // Compute 1x1 "tile" in level inputLevel_ + 3; only 1 thread per
  // 16 does this. Shuffle again.
  dstCoord_ >>= 1;
  in00_ = out_;
  in01_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 4);
  in10_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 8);
  in11_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 12);

  if (0 == (gl_SubgroupInvocationID & 15))
  {
    NVPRO_PYRAMID_REDUCE4(in00_, in01_, in10_, in11_, out_);
    NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+3), out_);
  }
}


void nvproPyramidMain()
{
  // Cut the input mip level into square tiles of edge length 8.
  int   inputLevel_      = NVPRO_PYRAMID_INPUT_LEVEL_;
  ivec2 srcImageSize_    = NVPRO_PYRAMID_LEVEL_SIZE(inputLevel_);
  uint  horizontalTiles_ = uint(srcImageSize_.x) / 8u;
  uint  verticalTiles_   = uint(srcImageSize_.y) / 8u;

  // Each thread handles 1 sample of the input level.
  // Assign tiles to each team.
  uint  teamSize_        = 16u;
  uint  tileIndex_       = gl_GlobalInvocationID.x / teamSize_;
  uint  horizontalIndex_ = tileIndex_ % horizontalTiles_;
  uint  verticalIndex_   = tileIndex_ / horizontalTiles_;
  ivec2 tileOffset_      = ivec2(horizontalIndex_, verticalIndex_) * 8;
  if (verticalIndex_ < verticalTiles_)
  {
    handleTile_(tileOffset_, inputLevel_);
  }
}
