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

layout(local_size_x = 256) in;

// Cache for level 4 + NVPRO_PYRAMID_INPUT_LEVEL_.
// If 6 == NVPRO_PYRAMID_LEVEL_COUNT_, layout is:
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
// If 5 == NVPRO_PYRAMID_LEVEL_COUNT_, layout is up to 4 consecutive such tiles:
// +---+---+
// | 0 | 1 |
// +---+---+
// | 2 | 3 |
// +---+---+
//
// Otherwise, this is unused (in-warp shuffles are enough).
// These diagrams are referenced later.
shared NVPRO_PYRAMID_SHARED_TYPE sharedLevel4_[16];


// Handle the tile at the given input level and offset (position
// of upper-left corner), and write out the resulting minified tiles
// in the next 2 to 4 mip levels (depending on levelCount_).
// Tiles are squares with edge length 1 << levelCount_
//
// Must be executed by N consecutive threads with same inputs, with
// the lowest thread number being a multiple of N; N given by
//
// levelCount_  2   3   4
// N            1   4  16
//
// If inputLevel_ == 4 and sharedMemoryWrite_ == true,
// then the 1x1 sample generated for inputLevel_ + 4 is written
// to sharedLevel4_[sharedMemoryIdx_]
void handleTile_(ivec2 srcTileOffset_, int inputLevel_, int levelCount_,
                 bool sharedMemoryWrite_, uint sharedMemoryIdx_)
{
  // Discussion for levelCount_ == 4
  // Break the input tile into 16 4x4 sub-tiles.  Each thread in the team
  // generates 4 samples (from 16 inputs), writes them to inputLevel_ + 1.
  // This means each thread produces the inputs it needs to generate 1
  // sample for inputLevel_ + 2.
  // +---+---++---+---+
  // | 0 | 1 || 4 | 5 |
  // +---+---++---+---+
  // | 2 | 3 || 6 | 7 |
  // +===+===++===+===+
  // | 8 | 9 ||12 |13 |
  // +---+---++---+---+
  // |10 |11 ||14 |15 |
  // +---+---++---+---+
  // For levelCount_ < 4, just cut out the parts of the 16x16 tile
  // that are not applicable.
  NVPRO_PYRAMID_TYPE sample00_, sample01_, sample10_, sample11_;

  // Relative location of sub-tile.
  uint teamMask_   = levelCount_ >= 4 ? 15 : levelCount_ == 3 ? 3 : 0;
  uint idxInTeam_  = gl_LocalInvocationIndex & teamMask_;
  ivec2 subOffset_ = ivec2((idxInTeam_ & 1) << 2 | (idxInTeam_ & 4) << 1,
                           (idxInTeam_ & 2) << 1 | (idxInTeam_ & 8));

  // Calculate future upper-left input.
  ivec2 dstCoord_;
  dstCoord_  = ((srcTileOffset_ + subOffset_) >> 1) + ivec2(0, 0);
  NVPRO_PYRAMID_LOAD_REDUCE4((dstCoord_ << 1), inputLevel_, sample00_);
  NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+1), sample00_);

  // Calculate future lower-left input.
  dstCoord_ = ((srcTileOffset_ + subOffset_) >> 1) + ivec2(0, 1);
  NVPRO_PYRAMID_LOAD_REDUCE4((dstCoord_ << 1), inputLevel_, sample01_);
  NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+1), sample01_);

  // Calculate future upper-right input.
  dstCoord_ = ((srcTileOffset_ + subOffset_) >> 1) + ivec2(1, 0);
  NVPRO_PYRAMID_LOAD_REDUCE4((dstCoord_ << 1), inputLevel_, sample10_);
  NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+1), sample10_);

  // Calculate future lower-right input.
  dstCoord_ = ((srcTileOffset_ + subOffset_) >> 1) + ivec2(1, 1);
  NVPRO_PYRAMID_LOAD_REDUCE4((dstCoord_ << 1), inputLevel_, sample11_);
  NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+1), sample11_);

  // Compute 4x4 tile in level inputLevel_ + 2; one sample per thread.
  // Outputs from before are now inputs.
  NVPRO_PYRAMID_TYPE out_;
  dstCoord_ = (srcTileOffset_ + subOffset_) >> 2;
  NVPRO_PYRAMID_REDUCE4(sample00_, sample01_, sample10_, sample11_, out_);
  NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+2), out_);

  if (levelCount_ == 2) return;

  // Compute 2x2 tile in level inputLevel_ + 3; only 1 out of every 4
  // threads does this. Use shuffle to get the needed data from the
  // other three threads.
  dstCoord_ >>= 1;
  sample00_ = out_;
  sample01_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 1);
  sample10_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 2);
  sample11_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 3);

  if (0 == (gl_SubgroupInvocationID & 3))
  {
    NVPRO_PYRAMID_REDUCE4(sample00_, sample01_, sample10_, sample11_, out_);
    NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+3), out_);
  }

  if (levelCount_ == 3) return;

  // Compute 1x1 "tile" in level inputLevel_ + 4; only 1 thread per
  // 16 does this. Shuffle again.
  dstCoord_ >>= 1;
  sample00_ = out_;
  sample01_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 4);
  sample10_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 8);
  sample11_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 12);

  if (0 == (gl_SubgroupInvocationID & 15))
  {
    NVPRO_PYRAMID_REDUCE4(sample00_, sample01_, sample10_, sample11_, out_);
    NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+4), out_);
    if (sharedMemoryWrite_)
    {
      NVPRO_PYRAMID_SHARED_STORE(sharedLevel4_[sharedMemoryIdx_], out_);
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

  // Assign tiles to each team. Handle levelCount_ == 1 as a special case.
  if (levelCount_ == 1)
  {
    // Each thread has to handle 4 2x2 tiles, to maintain the 4096
    // input samples per workgroup invariant.
    uint  baseTileIndex_   = gl_GlobalInvocationID.x * 4u;
    uint  horizontalIndex_ = baseTileIndex_ % horizontalTiles_;
    uint  verticalIndex_   = baseTileIndex_ / horizontalTiles_;
    for (uint i = 0; ; ++i)
    {
      ivec2 dstCoord_        = ivec2(horizontalIndex_, verticalIndex_);
      ivec2 srcCoord_        = dstCoord_ << 1;
      if (verticalIndex_ >= verticalTiles_) break;

      NVPRO_PYRAMID_TYPE sampled_;
      NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord_, inputLevel_, sampled_);
      NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_+1), sampled_);

      if (i == 3u) break;

      horizontalIndex_++;
      bool onNewRow_ = horizontalIndex_ >= horizontalTiles_;
      verticalIndex_   += onNewRow_ ? 1 : 0;
      horizontalIndex_ =  onNewRow_ ? 0 : horizontalIndex_;
    }
  }
  else
  {
    // Calculate the team size from the level count.
    // Each thread handles 4 samples of the input level.
    uint  teamSizeLog2_ = uint(levelCount_) * 2u - 4u;

    // Assign tiles to each team.
    uint  tileIndex_       = gl_GlobalInvocationID.x >> teamSizeLog2_;
    uint  horizontalIndex_ = tileIndex_ % horizontalTiles_;
    uint  verticalIndex_   = tileIndex_ / horizontalTiles_;
    ivec2 tileOffset_ = ivec2(horizontalIndex_, verticalIndex_) << levelCount_;
    if (levelCount_ <= 4)
    {
      if (verticalIndex_ < verticalTiles_)
      {
        handleTile_(tileOffset_, inputLevel_, levelCount_, false, 0);
      }
    }
    else // levelCount_ 5 or 6.
    {
      // For 5 or more levels, team size is too big for shuffle
      // communication. Need to split the tile into 16x16 sub-tiles
      // and the team into 16-thread sub-teams, communicate inputLevel_ + 4
      // in shared memory, and then handle the last 1 or 2 levels.
      if (verticalIndex_ < verticalTiles_)
      {
        // Break team into sub-teams; 4 sub-teams for 32x32 tile; 16 for 64x64
        // Assign one sub-tile per sub-team, then nominate one thread
        // per sub-team to write a sample to sharedLevel4_.
        uint subTileMask_  = levelCount_ == 5 ? 3u : 15u;
        uint subTileIndex_ = (gl_LocalInvocationIndex >> 4) & subTileMask_;
        ivec2 subTileOffset_ =
            tileOffset_
            + ivec2((subTileIndex_ & 1) << 4 | (subTileIndex_ & 4) << 3,
                    (subTileIndex_ & 2) << 3 | (subTileIndex_ & 8) << 2);
        // Hard to explain; refer to diagram in sharedLevel4_ comment
        // to see how reduced sub-tile get laid out in smem.
        uint sharedMemoryWriteIdx_ = (gl_LocalInvocationIndex >> 4u) & 15u;
        handleTile_(subTileOffset_, inputLevel_, 4,  // **NOT** levelCount_
                    true, sharedMemoryWriteIdx_);
      }

      // Wait for shared memory to be filled.
      barrier();

      // Fill last 1 or 2 levels out of 5 or 6. Only 1 of 64 threads
      // has to do this.
      if (gl_LocalInvocationIndex < gl_WorkGroupSize.x / 64u)
      {
        NVPRO_PYRAMID_TYPE in00_, in01_, in10_, in11_, out_;
        if (levelCount_ == 5)
        {
          tileIndex_ = (gl_WorkGroupID.x * gl_WorkGroupSize.x) >> teamSizeLog2_
                       | gl_LocalInvocationIndex;
          horizontalIndex_ = tileIndex_ % horizontalTiles_;
          verticalIndex_   = tileIndex_ / horizontalTiles_;
          tileOffset_ = ivec2(horizontalIndex_, verticalIndex_) << levelCount_;

          // Bounds check depends on earlier threads check.
          if (verticalIndex_ < verticalTiles_)
          {
            uint smemOffset_ = gl_LocalInvocationIndex * 4u;
            NVPRO_PYRAMID_SHARED_LOAD(sharedLevel4_[smemOffset_ + 0u], in00_);
            NVPRO_PYRAMID_SHARED_LOAD(sharedLevel4_[smemOffset_ + 1u], in10_);
            NVPRO_PYRAMID_SHARED_LOAD(sharedLevel4_[smemOffset_ + 2u], in01_);
            NVPRO_PYRAMID_SHARED_LOAD(sharedLevel4_[smemOffset_ + 3u], in11_);
            NVPRO_PYRAMID_REDUCE4(in00_, in01_, in10_, in11_, out_);
            ivec2 dstCoord_ = tileOffset_ >> 5;
            NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_ + 5), out_);
          }
        }
        else // levelCount_ == 6
        {
          // Fill inputLevel_ + 5 from shared memory.
          uint smemOffset_ = gl_LocalInvocationIndex * 4u;
          NVPRO_PYRAMID_SHARED_LOAD(sharedLevel4_[smemOffset_ + 0u], in00_);
          NVPRO_PYRAMID_SHARED_LOAD(sharedLevel4_[smemOffset_ + 1u], in10_);
          NVPRO_PYRAMID_SHARED_LOAD(sharedLevel4_[smemOffset_ + 2u], in01_);
          NVPRO_PYRAMID_SHARED_LOAD(sharedLevel4_[smemOffset_ + 3u], in11_);
          NVPRO_PYRAMID_REDUCE4(in00_, in01_, in10_, in11_, out_);
          ivec2 dstCoord_ = (tileOffset_ >> 5)
                            + ivec2(gl_LocalInvocationIndex & 1u,
                                    (gl_LocalInvocationIndex & 2u) >> 1);
          NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_ + 5), out_);

          // Shuffle 4 samples and produce sole inputLevel_ + 6 sample.
          in00_ = out_;
          in01_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 1);
          in10_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 2);
          in11_ = NVPRO_PYRAMID_SHUFFLE_XOR(out_, 3);
          dstCoord_ >>= 1;

          if (gl_LocalInvocationIndex < gl_WorkGroupSize.x / 256u)
          {
            NVPRO_PYRAMID_REDUCE4(in00_, in01_, in10_, in11_, out_);
            NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_ + 6), out_);
          }
        }
      }
    }
  }
}
