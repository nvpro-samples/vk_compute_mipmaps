// Efficient special case image pyramid generation kernel.  Generates
// up to 6 levels at once: each workgroup reads up to N samples (see below)
// of the input mip level and generates the resulting samples for the next
// up to 6 levels. Dispatch with y, z = 1.
//
// N = 1024 if NVPRO_PYRAMID_LEVEL_COUNT_ <= 5; 4096 otherwise.
//
// This only works when the input mip level has edges divisible by 2
// to the power of NVPRO_PYRAMID_LEVEL_COUNT_, and
// NVPRO_PYRAMID_LEVEL_COUNT_ can be at most 6.
//
// TODO: Test if this works for subgroup size != 32.

layout(local_size_x = 256) in;

// Cache for the tile generated for level NVPRO_PYRAMID_INPUT_LEVEL_ + N
// used only when NVPRO_PYRAMID_LEVEL_COUNT_ >= 4.
// N is 3 if said level count is 4 or 5, 4 if level count is 6.
// If level count is 5 or higher, the layout of the tile is:
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
// If level count is 4, there are up to 4 consecutive tiles of layout:
// +---+---+
// | 0 | 1 |
// +---+---+
// | 2 | 3 |
// +---+---+
//
// These diagrams are referenced later.
shared NVPRO_PYRAMID_SHARED_TYPE sharedTile_[16];


// Handle the tile at the given input level and offset (position
// of upper-left corner), and write out the resulting minified tiles
// in the next 1 to 4 mip levels (depending on levelCount_).
// Tiles are squares with edge length 1 << levelCount_
//
// Only 1 to 3 levels are supported if !sharedMemoryWrite_;
// Only 3 to 4 levels are supported otherwise (micro-optimization).
//
// Must be executed by N consecutive threads with same inputs, with
// the lowest thread number being a multiple of N; N given by
//
// levelCount_  1   2   3   4
// N            1   4  16  16 [NOT 64]
//
// If sharedMemoryWrite_ == true, then the 1x1 sample generated for
// the final output level is written to sharedTile_[sharedMemoryIdx_].
void handleTile_(ivec2 srcTileOffset_, int inputLevel_, int levelCount_,
                 bool sharedMemoryWrite_, uint sharedMemoryIdx_)
{
  // Discussion for levelCount_ == 3
  //
  // Break the input tile into 16 2x2 sub-tiles.  Each thread in the team
  // generates 1 sample (from 4 inputs), writes them to inputLevel_ + 1.
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
  // For levelCount_ < 3, just cut out the parts of the 8x8 tile
  // that are not applicable.
  //
  // For levelCount_ == 4, the input 16x16 tile is broken into 4x4
  // sub-tiles in the same pattern. Each thread generates the
  // corresponding 2x2 sub-tile of the inputLevel_ + 1 8x8 tile.
  // The problem then reduces to the levelCount_ == 3 case.
  NVPRO_PYRAMID_TYPE sample00_, sample01_, sample10_, sample11_, out_;

  int   dstLevel_ = inputLevel_ + 1;
  ivec2 dstSubTile_;

  // Calculate the index of this thread within the team.
  uint teamMask_   = levelCount_ >= 3 ? 15 : levelCount_ == 2 ? 3 : 0;
  uint idxInTeam_  = gl_LocalInvocationIndex & teamMask_;

  // NOTE the extra sharedMemoryWrite_ requirement!!!
  if (sharedMemoryWrite_ && levelCount_ == 4)
  {
    // The location of the sub-tile assigned to this thread in level inputLevel_
    uint  xOffset_    = (idxInTeam_ & 1) << 2 | (idxInTeam_ & 4) << 1;
    uint  yOffset_    = (idxInTeam_ & 2) << 1 | (idxInTeam_ & 8);
    ivec2 srcSubTile_ = srcTileOffset_ + ivec2(xOffset_, yOffset_);
    dstSubTile_       = srcSubTile_ >> 1;

    // Thread calculates upper-left sample of 2x2 output sub-tile.
    ivec2 srcCoord_, dstCoord_;
    srcCoord_ = srcSubTile_;
    dstCoord_ = dstSubTile_;
    NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord_, inputLevel_, sample00_);
    NVPRO_PYRAMID_STORE(dstCoord_, dstLevel_, sample00_);

    // Thread calculates lower-left sample of 2x2 output sub-tile.
    srcCoord_ = srcSubTile_ + ivec2(0, 2);
    dstCoord_ = dstSubTile_ + ivec2(0, 1);
    NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord_, inputLevel_, sample01_);
    NVPRO_PYRAMID_STORE(dstCoord_, dstLevel_, sample01_);

    // Thread calculates upper-right sample of 2x2 output sub-tile.
    srcCoord_ = srcSubTile_ + ivec2(2, 0);
    dstCoord_ = dstSubTile_ + ivec2(1, 0);
    NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord_, inputLevel_, sample10_);
    NVPRO_PYRAMID_STORE(dstCoord_, dstLevel_, sample10_);

    // Thread calculates lower-right sample of 2x2 output sub-tile.
    srcCoord_ = srcSubTile_ + ivec2(2, 2);
    dstCoord_ = dstSubTile_ + ivec2(1, 1);
    NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord_, inputLevel_, sample11_);
    NVPRO_PYRAMID_STORE(dstCoord_, dstLevel_, sample11_);

    // Now the full assigned 2x2 subtile has been filled, move on to
    // the 1x1 sample of the next level assigned to this thread.
    dstLevel_++;
    dstSubTile_ >>= 1;
    NVPRO_PYRAMID_REDUCE4(sample00_, sample01_, sample10_, sample11_, out_);
    NVPRO_PYRAMID_STORE(dstSubTile_, dstLevel_, out_);
  }
  else  // levelCount_ != 4
  {
    // The location of the sub-tile assigned to this thread in the
    // level after level inputLevel_.
    uint  xOffset_    = (idxInTeam_ & 1) << 1 | (idxInTeam_ & 4);
    uint  yOffset_    = (idxInTeam_ & 2) | (idxInTeam_ & 8) >> 1;
    ivec2 srcSubTile_ = srcTileOffset_ + ivec2(xOffset_, yOffset_);
    dstSubTile_       = srcSubTile_ >> 1;

    // Thread calculates the sample in that sub-tile.
    NVPRO_PYRAMID_LOAD_REDUCE4(srcSubTile_, inputLevel_, out_);
    NVPRO_PYRAMID_STORE(dstSubTile_, dstLevel_, out_);
  }

  if (!sharedMemoryWrite_ && levelCount_ == 1) return;

  // The whole team computes the 2x2 tile in the next level; only 1
  // out of every 4 threads does this. Use shuffle to get the needed
  // data from the other three threads.
  dstLevel_++;
  dstSubTile_ >>= 1;
  sample00_ = out_;
  sample01_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 1);
  sample10_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 2);
  sample11_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 3);

  if (0 == (gl_SubgroupInvocationID & 3))
  {
    NVPRO_PYRAMID_REDUCE4(sample00_, sample01_, sample10_, sample11_, out_);
    NVPRO_PYRAMID_STORE(dstSubTile_, dstLevel_, out_);
  }

  if (!sharedMemoryWrite_ && levelCount_ == 2) return;

  // Compute 1x1 "tile" in the last level handled by this function;
  // only 1 thread per 16 does this. Shuffle again.
  // This is also the thread that does the optional shared memory write.
  dstLevel_++;
  dstSubTile_ >>= 1;
  sample00_ = out_;
  sample01_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 4);
  sample10_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 8);
  sample11_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 12);

  if (0 == (gl_SubgroupInvocationID & 15))
  {
    NVPRO_PYRAMID_REDUCE4(sample00_, sample01_, sample10_, sample11_, out_);
    NVPRO_PYRAMID_STORE(dstSubTile_, dstLevel_, out_);
    if (sharedMemoryWrite_)
    {
      NVPRO_PYRAMID_SHARED_STORE(sharedTile_[sharedMemoryIdx_], out_);
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

  // Calculate the team size from the level count.  Each thread
  // handles 4 inupt samples, except when levelCount_ == 6, then each
  // handles 16 samples.
  uint  teamSizeLog2_ = min(8u, uint(levelCount_) * 2u - 2u);

  // Assign tiles to each team.
  uint  tileIndex_       = gl_GlobalInvocationID.x >> teamSizeLog2_;
  uint  horizontalIndex_ = tileIndex_ % horizontalTiles_;
  uint  verticalIndex_   = tileIndex_ / horizontalTiles_;
  ivec2 tileOffset_ = ivec2(horizontalIndex_, verticalIndex_) << levelCount_;

  if (levelCount_ <= 3)
  {
    if (verticalIndex_ < verticalTiles_)
    {
      // Reminder to self: can't handle 4 level case when
      // sharedMemoryWrite_ is false.
      handleTile_(tileOffset_, inputLevel_, levelCount_, false, 0);
    }
    return;
  }

  // For 4 or more levels, team size is too big for shuffle communication.
  // Need to split the tile into sub-tiles and teams into 16 thread sub-teams.
  // Each sub-team writes one sample to shared memory.
  // Refer to sharedTile_ diagram for details.
  if (verticalIndex_ < verticalTiles_)
  {
    // Number of levels to fill for now.
    int subLevelCount_ = levelCount_ == 6 ? 4 : 3;

    // Calculate the index of the sub-team within the team.
    int subTeamMask_ = levelCount_ == 4 ? 3 : 15;
    int subTeamIdx_  = int(gl_GlobalInvocationID.x >> 4) & subTeamMask_;

    // Location of sub-tile; they are 8x8 or 16x16 depending on subLevelCount_
    ivec2 subTeamOffset_;
    subTeamOffset_.x = (subTeamIdx_ & 1) << 3 | (subTeamIdx_ & 4) << 2;
    subTeamOffset_.y = (subTeamIdx_ & 2) << 2 | (subTeamIdx_ & 8) << 1;
    if (subLevelCount_ == 4)
    {
      subTeamOffset_ <<= 1;
    }

    // Index in shared memory that this sub-team will write to.
    uint sharedMemoryIndex_ = (gl_GlobalInvocationID.x >> 4u) & 15u;

    // Handle the sub-tile and write the last level 1x1 sample to shared memory.
    handleTile_(tileOffset_ + subTeamOffset_, inputLevel_, subLevelCount_,
                true, sharedMemoryIndex_);

    // Problem reduces to handling 1 or 2 remaining levels.
    inputLevel_ += subLevelCount_;
    levelCount_ -= subLevelCount_;
  }

  // Wait for shared memory to fill.
  barrier();

  // Handle the remaining 1 or 2 levels.
  // Only 4 threads have to do this per workgroup (NOT per team)
  if (gl_LocalInvocationIndex < 4)
  {
    NVPRO_PYRAMID_TYPE in00_, in01_, in10_, in11_, out_;
    if (levelCount_ == 1)
    {
      // Handle up to 4 2x2 tiles in shared memory, 1 tile per thread.
      // Tile location calculated for the output level (final level).
      tileIndex_       = gl_WorkGroupID.x * 4 + gl_LocalInvocationIndex;
      horizontalIndex_ = tileIndex_ % horizontalTiles_;
      verticalIndex_   = tileIndex_ / horizontalTiles_;
      tileOffset_      = ivec2(horizontalIndex_, verticalIndex_);
      uint smemOffset_ = gl_LocalInvocationIndex * 4u;

      if (verticalIndex_ < verticalTiles_)
      {
        NVPRO_PYRAMID_SHARED_LOAD(sharedTile_[smemOffset_ + 0u], in00_);
        NVPRO_PYRAMID_SHARED_LOAD(sharedTile_[smemOffset_ + 1u], in10_);
        NVPRO_PYRAMID_SHARED_LOAD(sharedTile_[smemOffset_ + 2u], in01_);
        NVPRO_PYRAMID_SHARED_LOAD(sharedTile_[smemOffset_ + 3u], in11_);
        NVPRO_PYRAMID_REDUCE4(in00_, in01_, in10_, in11_, out_);
        NVPRO_PYRAMID_STORE(tileOffset_, (inputLevel_ + 1), out_);
      }
    }
    else  // levelCount_ == 2
    {
      // Handle the 4x4 tile in shared memory, 1 2x2 sub-tile per
      // thread.  Tile location calculated for the final output level
      // (here we first calculate an intermediate level).
      tileIndex_       = gl_WorkGroupID.x;
      horizontalIndex_ = tileIndex_ % horizontalTiles_;
      verticalIndex_   = tileIndex_ / horizontalTiles_;
      tileOffset_      = ivec2(horizontalIndex_, verticalIndex_);
      uint smemOffset_ = gl_LocalInvocationIndex * 4u;

      if (verticalIndex_ < verticalTiles_)
      {
        // Handle first level after inputLevel_
        NVPRO_PYRAMID_SHARED_LOAD(sharedTile_[smemOffset_ + 0u], in00_);
        NVPRO_PYRAMID_SHARED_LOAD(sharedTile_[smemOffset_ + 1u], in10_);
        NVPRO_PYRAMID_SHARED_LOAD(sharedTile_[smemOffset_ + 2u], in01_);
        NVPRO_PYRAMID_SHARED_LOAD(sharedTile_[smemOffset_ + 3u], in11_);
        NVPRO_PYRAMID_REDUCE4(in00_, in01_, in10_, in11_, out_);
        ivec2 threadOffset_ = ivec2(gl_LocalInvocationIndex & 1,
                                    (gl_LocalInvocationIndex & 2) >> 1);
        NVPRO_PYRAMID_STORE((tileOffset_ * 2 + threadOffset_),
                            (inputLevel_ + 1), out_);
        // Shuffle 4 samples and produce sole last level sample.
        in00_ = out_;
        in10_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 1);
        in01_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 2);
        in11_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 3);

        if (gl_LocalInvocationIndex == 0u)
        {
          NVPRO_PYRAMID_REDUCE4(in00_, in01_, in10_, in11_, out_);
          NVPRO_PYRAMID_STORE(tileOffset_, (inputLevel_ + 2), out_);
        }
      }
    }
  }
}
