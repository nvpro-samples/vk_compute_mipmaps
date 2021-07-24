// Efficient special case image pyramid generation kernel.  Generates
// up to 6 levels at once: each workgroup reads up to 4096 samples
// of the input mip level and generates the resulting samples for the next
// up to 6 levels. Dispatch with y, z = 1.
//
// This only works when the input mip level has edges divisible by 2
// to the power of NVPRO_PYRAMID_LEVEL_COUNT_, and
// NVPRO_PYRAMID_LEVEL_COUNT_ can be at most 6.
//
// TODO: Test if this works for subgroup size != 32.

layout(local_size_x = 1024) in;

// Cache for level 3 + NVPRO_PYRAMID_INPUT_LEVEL_.
//
// If 4 == NVPRO_PYRAMID_LEVEL_COUNT_, layout is up to 4 consecutive such tiles:
// +---+---+
// | 0 | 1 |
// +---+---+
// | 2 | 3 |
// +---+---+
//
// If 5 == NVPRO_PYRAMID_LEVEL_COUNT_, layout is up to 4 consecutive such tiles:
// +---+---++---+---+
// | 0 | 1 || 4 | 5 |
// +---+---++---+---+
// | 2 | 3 || 6 | 7 |
// +===+===++===+===+
// | 8 | 9 ||12 |13 |
// +---+---++---+---+
// |10 |11 ||14 |15 |
// +---+---++---+---+
//
// If 6 == NVPRO_PYRAMID_LEVEL_COUNT_, layout is one such tile:
// +---+---++---+---+++-- ...
// | 0 | 1 || 4 | 5 ||| 16
// +---+---++---+---+++-- ...
// | 2 | 3 || 6 | 7 ||| 18
// +===+===++===+===+++== ...
// | 8 | 9 ||12 |13 ||| 24
// +---+---++---+---+++-- ...
// |10 |11 ||14 |15 ||| 26
// +---+---++---+---+++-- ...
// +---+---++---+---+++-- ...
// |32 |33 ||36 |37 ||| 48
// .   .   ..   .   ...
//
// Otherwise, this is unused (in-warp shuffles are enough).
// These diagrams are referenced later.
shared NVPRO_PYRAMID_SHARED_TYPE sharedLevel3_[64];

// Handle the tile at the given input level and offset (position
// of upper-left corner), and write out the resulting minified tiles
// in the next 1 to 3 mip levels (depending on levelCount_).
// Tiles are squares with edge length 1 << levelCount_
//
// Must be executed by N consecutive threads with same inputs, with
// the lowest thread number being a multiple of N; N given by
//
// levelCount_  1   2   3
// N            1   4  16
//
// If sharedMemoryRead_ is true, the input tile is read from sharedLevel3_
// starting from index sharedMemoryReadTileIdx_. Note that the recursive
// tiling pattern makes this work for any tile size.
//
// If inputLevel_ == 3 and sharedMemoryWrite_ == true,
// then the 1x1 sample generated for inputLevel_ + 3 is written
// to sharedLevel3_[sharedMemoryWriteIdx_]
void handleTile_(ivec2 srcTileOffset_, int inputLevel_, int levelCount_,
                 bool sharedMemoryRead_, uint sharedMemoryReadTileIdx_,
                 bool sharedMemoryWrite_, uint sharedMemoryWriteIdx_)
{
  // Discussion for levelCount_ == 3
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
  // For levelCount_ < 3, just cut out the parts of the 8x8 tile
  // that are not applicable.
  NVPRO_PYRAMID_TYPE in00_, in01_, in10_, in11_, out_;

  // Relative location of sub-tile.
  uint teamMask_   = levelCount_ == 1 ? 0u : levelCount_ == 2 ? 3u : 15u;
  uint idxInTeam_  = gl_LocalInvocationIndex & teamMask_;
  ivec2 subOffset_ = ivec2((idxInTeam_ & 1) << 1 | (idxInTeam_ & 4),
                           (idxInTeam_ & 2) | (idxInTeam_ & 8) >> 1);

  // Calculate the assigned sample for inputLevel_ + 1
  ivec2 dstCoord_;
  dstCoord_  = ((srcTileOffset_ + subOffset_) >> 1) + ivec2(0, 0);
  if (sharedMemoryRead_)
  {
    uint smemIdx_ = idxInTeam_ * 4 + sharedMemoryReadTileIdx_;
    NVPRO_PYRAMID_SHARED_LOAD(sharedLevel3_[smemIdx_ + 0], in00_);
    NVPRO_PYRAMID_SHARED_LOAD(sharedLevel3_[smemIdx_ + 1], in10_);
    NVPRO_PYRAMID_SHARED_LOAD(sharedLevel3_[smemIdx_ + 2], in01_);
    NVPRO_PYRAMID_SHARED_LOAD(sharedLevel3_[smemIdx_ + 3], in11_);
    NVPRO_PYRAMID_REDUCE4(in00_, in01_, in10_, in11_, out_);
  }
  else
  {
    NVPRO_PYRAMID_LOAD_REDUCE4((dstCoord_ << 1), inputLevel_, out_);
  }
  NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+1), out_);

  if (levelCount_ == 1) return;

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

  if (levelCount_ == 2) return;

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
    if (sharedMemoryWrite_)
    {
      NVPRO_PYRAMID_SHARED_STORE(sharedLevel3_[sharedMemoryWriteIdx_], out_);
    }
  }
}


void nvproPyramidMain()
{
  // Cut the input mip level into square tiles of edge length
  // 2 to the power of NVPRO_PYRAMID_LEVEL_COUNT_.
  int   levelCount_      = NVPRO_PYRAMID_LEVEL_COUNT_;
  int   inputLevel_      = NVPRO_PYRAMID_INPUT_LEVEL_;
  ivec2 srcImageSize_    = NVPRO_PYRAMID_LEVEL_SIZE(inputLevel_);
  uint  horizontalTiles_ = uint(srcImageSize_.x) >> levelCount_;
  uint  verticalTiles_   = uint(srcImageSize_.y) >> levelCount_;

  // Calculate the team size from the level count.
  // Each thread handles 1 sample of the input level.
  uint  teamSizeLog2_ = levelCount_ * 2u - 2u;

  // Assign tiles to each team.
  uint  tileIndex_       = gl_GlobalInvocationID.x >> teamSizeLog2_;
  uint  horizontalIndex_ = tileIndex_ % horizontalTiles_;
  uint  verticalIndex_   = tileIndex_ / horizontalTiles_;
  ivec2 tileOffset_ = ivec2(horizontalIndex_, verticalIndex_) << levelCount_;
  if (levelCount_ <= 3)
  {
    if (verticalIndex_ < verticalTiles_)
    {
      handleTile_(tileOffset_, inputLevel_, levelCount_, false, 0, false, 0);
    }
  }
  else // levelCount_ 4, 5, or 6.
  {
    // For 4 or more levels, team size is too big for shuffle
    // communication. Need to split the tile into 8x8 sub-tiles
    // and the team into 16-thread sub-teams, communicate inputLevel_ + 3
    // in shared memory, and then handle the last 1 to 3 levels.
    if (verticalIndex_ < verticalTiles_)
    {
      // Break team into sub-teams; 4 sub-teams for 16x16 tile; 16 for 32x32.
      // Assign one sub-tile per sub-team, then nominate one thread
      // per sub-team to write a sample to sharedLevel3_.
      uint subTileMask_  = levelCount_ == 4 ? 3u : levelCount_ == 5 ? 15u : 63u;
      uint subTileIndex_ = (gl_LocalInvocationIndex >> 4) & subTileMask_;

      // Hard to explain; refer to diagram in sharedLevel3_ comment
      // to see how reduced sub-tile get laid out in smem.
      ivec2 subTileOffset_ = ivec2(0, 0);
      subTileOffset_.x |= int(subTileIndex_ &  1) << 3;
      subTileOffset_.x |= int(subTileIndex_ &  4) << 2;
      subTileOffset_.x |= int(subTileIndex_ & 16) << 1;
      subTileOffset_.y |= int(subTileIndex_ &  2) << 2;
      subTileOffset_.y |= int(subTileIndex_ &  8) << 1;
      subTileOffset_.y |= int(subTileIndex_ & 32);
      subTileOffset_ += tileOffset_;

      uint smemIdx_ = (gl_LocalInvocationIndex >> 4u) & 63u;
      handleTile_(subTileOffset_, inputLevel_, 3,  // only 3 levels handled here
                  false, 0, true, smemIdx_);
    }

    // Wait for shared memory to be filled.
    barrier();

    // Fill remaining 1 to 3 levels. At this point it's similar to the
    // original "handle up to 3 levels" case applied to level inputLevel_ + 3,
    // except we are reading from shared memory instead.
    if (gl_LocalInvocationIndex < gl_WorkGroupSize.x / 64u)
    {
      levelCount_      = NVPRO_PYRAMID_LEVEL_COUNT_ - 3;
      inputLevel_      = NVPRO_PYRAMID_INPUT_LEVEL_ + 3;
      srcImageSize_    = NVPRO_PYRAMID_LEVEL_SIZE(inputLevel_);
      horizontalTiles_ = uint(srcImageSize_.x) >> levelCount_;
      verticalTiles_   = uint(srcImageSize_.y) >> levelCount_;

      // Calculate the team size from the level count.
      // Each thread handles 1 sample of the (new) input level.
      teamSizeLog2_ = levelCount_ * 2u - 2u;

      // Assign tiles to each team.
      // This is a bit more tricky due to the "gaps" from inactive threads.
      uint teamsPerWorkgroup_ = (gl_WorkGroupSize.x / 64u) >> teamSizeLog2_;

      uint baseTileIndex_ = gl_WorkGroupID.x * teamsPerWorkgroup_;
      uint teamTileIndex_ = gl_LocalInvocationIndex >> teamSizeLog2_;
      tileIndex_          = baseTileIndex_ + teamTileIndex_;
      horizontalIndex_    = tileIndex_ % horizontalTiles_;
      verticalIndex_      = tileIndex_ / horizontalTiles_;

      tileOffset_ = ivec2(horizontalIndex_, verticalIndex_) << levelCount_;

      if (verticalIndex_ < verticalTiles_)
      {
        // Team reads from the teamTileIndex_-th tile in shared memory,
        // each tile contains 4^{levelCount_} samples.
        uint smemOffset_ = teamTileIndex_ << (2 * levelCount_);
        handleTile_(tileOffset_, inputLevel_, levelCount_,
                    true, smemOffset_, false, 0);
      }
    }
  }
}
