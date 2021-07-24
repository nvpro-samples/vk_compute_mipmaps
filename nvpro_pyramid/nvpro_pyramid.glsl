// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Skeleton code for an image pyramid generation compute shader.  This
// only defines a "schedule" for doing work; you need to provide the
// reduction (kernel) implementation and code for loading and storing
// samples. This code is injected into nvproPyramidMain, the entry
// point to the pyramid generation shader.
//
// Since GLSL doesn't really have advanced meta-programming
// capabilities, this is all configured with preprocessor macros,
// which must be defined before including this file. Your macros, this
// include file, and a main function that does any needed
// initialization before calling nvproPyramidMain, together form the
// complete compute shader.
//
// When you compile the full compute pipeline, the pipeline must
// include a 32-bit integer within its push constants; this is needed
// to communicate to the shader which mip levels to work on.
// By default, this push constant is at offset 0; see
// NvproPyramidPipelines::pushConstantOffset and the
// NVPRO_PYRAMID_PUSH_CONSTANT macro if you need to change this.
//
// This shader is designed to be dispatched by the host code in
// nvpro_pyramid_dispatch.hpp
//
//         The following macros are required:
//
//   * NVPRO_PYRAMID_REDUCE(a0, v0, a1, v1, a2, v2, out_)
// Set out_ to the reduction of the three inputs v0...v2, each using
// a0...a2 as their weights.
//
// Please see the optional NVPRO_PYRAMID_LOAD_REDUCE4 macro as well.
//
//   * NVPRO_PYRAMID_LOAD(coord : ivec2, level : int, out_)
// Load the sample at the given texel and mip level, and store in out_
// You DO NOT have to do bounds-checking; and level is dynamically uniform.
// Note there's nothing stopping you from loading from multiple images.
//
//   * NVPRO_PYRAMID_STORE(coord : ivec2, level : int, in_)
// Store the sample in_ into the given texel of the given mip level.
// You DO NOT have to do bounds-checking; and level is dynamically uniform.
//
//   * NVPRO_PYRAMID_TYPE
// The data type of in_ and out_, above.
// Recommend 32 bytes max to avoid excessive memory usage.
//
//   * NVPRO_PYRAMID_LEVEL_SIZE(level : int)
// Resolve to an ivec2 giving the size of the given mip level.
//
//   * NVPRO_PYRAMID_IS_FAST_PIPELINE
// If nonzero, this shader compiles to the pipeline stored in
// NvproPyramidPipelines::fastPipeline. Otherwise, corresponds to pipeline
// NvproPyramidPipelines::generalPipeline.
//
// NvproPyramidPipelines::generalPipeline has no special hardware requirements.
//
// NvproPyramidPipelines::fastPipeline requires these three abilities:
// #extension GL_KHR_shader_subgroup_shuffle : enable
// VkPhysicalDeviceSubgroupProperties::subgroupSize >= 16
// VkPhysicalDeviceSubgroupProperties::supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT
//
//         The following macros are optional:
//
//   * NVPRO_PYRAMID_PUSH_CONSTANT
// If you need to declare your shader push constant manually, this macro
// must resolve to the name of the 32-bit int variable used for push constant.
// If not provided, the shader uses the 32-bits at offset 0.
// Related: NvproPyramidPipelines::pushConstantOffset
//
//   * NVPRO_PYRAMID_REDUCE2(v0, v1, out_)
// Special case of reducing 2 samples of equal weight.
//
//   * NVPRO_PYRAMID_REDUCE4(v00, v01, v10, v11, out_)
// Special case of reducing a square of 4 samples of equal weight.
// For fast pipelines only, this removes the need for NVPRO_PYRAMID_REDUCE.
//
//   * NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord : ivec2, srcLevel : int, out_)
// Load the 2x2 texel square from srcCoord to (srcCoord + (1,1)), inclusive,
// from mip level srcLevel. Reduce the 4 texels, and write the result to out_.
// This can be used, for example, to take advantage of bilinear filtering.
// For fast pipelines only, this removes the need for NVPRO_PYRAMID_LOAD.
//
//   ***********************************************************************
//   * This macro is optional to simplify getting a minimum viable shader  *
//   * in working condition, but it is STRONGLY recommended to define this *
//   * macro to use hardware samplers whenever practical. Please do not    *
//   * do speed comparisons without defining NVPRO_PYRAMID_LOAD_REDUCE4!!! *
//   ***********************************************************************
//
//   * NVPRO_PYRAMID_SHUFFLE_XOR(in_, mask_)
// Conceptually identical to subgroupShuffleXor(in_, mask_)
// Advanced feature, only needed for potential edge cases.
// This macro is only used when NVPRO_PYRAMID_IS_FAST_PIPELINE != 0
//
//         The following must all be undefined or all be defined:
//
//   * NVPRO_PYRAMID_SHARED_TYPE
// The code needs to cache some texel outputs in shared memory to use
// as inputs for subsequent mip levels. This can be used to customize
// the cached texel type, e.g. to store a compacted reperesentation to
// reduce memory usage.
//
//   * NVPRO_PYRAMID_SHARED_LOAD(smem_, out_)
// Convert the NVPRO_PYRAMID_SHARED_TYPE smem_ to NVPRO_PYRAMID_TYPE out_
//
//   * NVPRO_PYRAMID_SHARED_STORE(smem_, in_)
// Convert the NVPRO_PYRAMID_TYPE in_ to NVPRO_PYRAMID_SHARED_TYPE smem_.
//
//         Macro details:
//
// For function-like macros, it's guaranteed that the output does not
// alias the inputs, and you do not have to parenthesize any
// arguments.  I parenthesize all arguments before calling your macros
// (unless I missed a spot).
//
// You may optionally enclose multiple statements or variable
// declarations in braces in function-like macros; to help with this,
// I declare all variables in this file with an underscore suffix, to
// avoid name collisions.



// Check required macros
#ifndef NVPRO_PYRAMID_IS_FAST_PIPELINE
#error "Missing required macro NVPRO_PYRAMID_IS_FAST_PIPELINE"
#endif
#ifndef NVPRO_PYRAMID_REDUCE
  #if !defined(NVPRO_PYRAMID_REDUCE4) || !NVPRO_PYRAMID_IS_FAST_PIPELINE
  #error "Missing required macro NVPRO_PYRAMID_REDUCE"
  #endif
#endif
#ifndef NVPRO_PYRAMID_LOAD
  #if !defined(NVPRO_PYRAMID_LOAD_REDUCE4) || !NVPRO_PYRAMID_IS_FAST_PIPELINE
  #error "Missing required macro NVPRO_PYRAMID_LOAD"
  #endif
#endif
#ifndef NVPRO_PYRAMID_STORE
#error "Missing required macro NVPRO_PYRAMID_STORE"
#endif
#ifndef NVPRO_PYRAMID_TYPE
#error "Missing required macro NVPRO_PYRAMID_TYPE"
#endif
#ifndef NVPRO_PYRAMID_LEVEL_SIZE
#error "Missing required macro NVPRO_PYRAMID_LEVEL_SIZE"
#endif

// Provide defaults for optional macros.
#ifndef NVPRO_PYRAMID_PUSH_CONSTANT
layout(push_constant) uniform NvproPyramidPushConstantBlock_
{
  uint nvproPyramidPushConstant_;
};
#define NVPRO_PYRAMID_PUSH_CONSTANT nvproPyramidPushConstant_
#endif

// The mip level used as source data for the current dispatch.
// Change nvpro_pyramid_dispatch.hpp nvproPyramidInputLevelShift if changed.
#define NVPRO_PYRAMID_INPUT_LEVEL_ int(uint(NVPRO_PYRAMID_PUSH_CONSTANT) >> 5u)

// Number of subsequent mip levels to fill.
#define NVPRO_PYRAMID_LEVEL_COUNT_ int(uint(NVPRO_PYRAMID_PUSH_CONSTANT) & 31u)

#ifndef NVPRO_PYRAMID_REDUCE2
#define NVPRO_PYRAMID_REDUCE2(v0, v1, out_) \
  NVPRO_PYRAMID_REDUCE(0.5, v0, 0.5, v1, 0, v1, out_)
#endif

#ifndef NVPRO_PYRAMID_REDUCE4
#define NVPRO_PYRAMID_REDUCE4(v00, v01, v10, v11, out_) \
{ \
  NVPRO_PYRAMID_TYPE v0_, v1_; \
  NVPRO_PYRAMID_REDUCE2(v00, v01, v0_); \
  NVPRO_PYRAMID_REDUCE2(v10, v11, v1_); \
  NVPRO_PYRAMID_REDUCE2(v0_, v1_, out_); \
}
#endif

#ifndef NVPRO_PYRAMID_LOAD_REDUCE4
#define NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord_, srcLevel_, out_) \
{ \
  NVPRO_PYRAMID_TYPE v00_, v01_, v10_, v11_; \
  NVPRO_PYRAMID_LOAD((srcCoord_) + ivec2(0, 0), srcLevel_, v00_); \
  NVPRO_PYRAMID_LOAD((srcCoord_) + ivec2(0, 1), srcLevel_, v01_); \
  NVPRO_PYRAMID_LOAD((srcCoord_) + ivec2(1, 0), srcLevel_, v10_); \
  NVPRO_PYRAMID_LOAD((srcCoord_) + ivec2(1, 1), srcLevel_, v11_); \
  NVPRO_PYRAMID_REDUCE4(v00_, v01_, v10_, v11_, out_); \
}
#endif

#if !defined(NVPRO_PYRAMID_SHUFFLE_XOR) && NVPRO_PYRAMID_IS_FAST_PIPELINE != 0
#define NVPRO_PYRAMID_SHUFFLE_XOR(in_, mask_) subgroupShuffleXor(in_, mask_)
#endif

// Handle optional specialized shared memory type.
#ifdef NVPRO_PYRAMID_SHARED_TYPE
  #if !defined(NVPRO_PYRAMID_SHARED_LOAD) || !defined(NVPRO_PYRAMID_SHARED_STORE)
    #error "Missing NVPRO_PYRAMID_SHARED_LOAD or NVPRO_PYRAMID_SHARED_STORE; needed when NVPRO_PYRAMID_SHARED_TYPE is defined."
  #endif
#else
  #if defined(NVPRO_PYRAMID_SHARED_LOAD) || defined(NVPRO_PYRAMID_SHARED_STORE)
    #error "Missing NVPRO_PYRAMID_SHARED_TYPE, needed when NVPRO_PYRAMID_SHARED_LOAD or NVPRO_PYRAMID_SHARED_STORE is defined."
  #endif
  #define NVPRO_PYRAMID_SHARED_TYPE NVPRO_PYRAMID_TYPE
  #define NVPRO_PYRAMID_SHARED_LOAD(smem_, out_) out_ = smem_
  #define NVPRO_PYRAMID_SHARED_STORE(smem_, in_) smem_ = in_
#endif

#if NVPRO_PYRAMID_IS_FAST_PIPELINE != 0

// Code for testing alternative designs during development, can ignore.
#if defined(NVPRO_USE_FAST_PIPELINE_ALTERNATIVE_) && NVPRO_USE_FAST_PIPELINE_ALTERNATIVE_ != 0
#include "fast_pipeline_alternative.glsl"
#else

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
void handleTile_(ivec2 srcTileOffset_, int inputLevel_, uint levelCount_,
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
  int dstLevel_ = inputLevel_ + 1;
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
  uint  teamSizeLog2_ = min(8u, levelCount_ * 2u - 2u);

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

#endif /* !NVPRO_USE_FAST_PIPELINE_ALTERNATIVE_ */

#else /* non-fast path */

// Code for testing alternative designs during development, can ignore.
#if defined(NVPRO_USE_GENERAL_PIPELINE_ALTERNATIVE_) && NVPRO_USE_GENERAL_PIPELINE_ALTERNATIVE_ != 0
#include "general_pipeline_alternative.glsl"
#else

// General-case shader for generating 1 or 2 levels of the mip pyramid.
// When generating 1 level, each workgroup handles up to 128 samples of the
// output mip level. When generating 2 levels, each workgroup handles
// a 8x8 tile of the last (2nd) output mip level, generating up to
// 17x17 samples of the intermediate (1st) output mip level along the way.
//
// Dispatch with y, z = 1
layout(local_size_x = 4 * 32) in;

// When generating 2 levels, the results of generating the intermediate
// level (first level generated) are cached here; this is the input tile
// needed to generate the 8x8 tile of the second level generated.
shared NVPRO_PYRAMID_SHARED_TYPE sharedLevel_[17][17]; // [y][x]

ivec2 kernelSizeFromInputSize_(ivec2 inputSize_)
{
  return ivec2(inputSize_.x == 1 ? 1 : (2 | (inputSize_.x & 1)),
               inputSize_.y == 1 ? 1 : (2 | (inputSize_.y & 1)));
}

NVPRO_PYRAMID_TYPE
loadSample_(ivec2 srcCoord_, int srcLevel_, bool loadFromShared_);

// Handle loading and reducing a rectangle of size kernelSize_
// with the given upper-left coordinate srcCoord_. Samples read from
// mip level srcLevel_ if !loadFromShared_, sharedLevel_ otherwise.
//
// kernelSize_ must range from 1x1 to 3x3.
//
// Once computed, the sample is written to the given coordinate of the
// specified destination mip level, and returned. The destination
// image size is needed to compute the kernel weights.
NVPRO_PYRAMID_TYPE reduceStoreSample_(ivec2 srcCoord_, int srcLevel_,
                                      bool  loadFromShared_,
                                      ivec2 kernelSize_,
                                      ivec2 dstImageSize_,
                                      ivec2 dstCoord_, int dstLevel_)
{
  bool  lfs_ = loadFromShared_;
  float n_   = dstImageSize_.y;
  float rcp_ = 1.0f / (2 * n_ + 1);
  float w0_  = rcp_ * (n_ - dstCoord_.y);
  float w1_  = rcp_ * n_;
  float w2_  = 1.0f - w0_ - w1_;

  NVPRO_PYRAMID_TYPE v0_, v1_, v2_, h0_, h1_, h2_, out_;

  // Reduce vertically up to 3 times (depending on kernel horizontal size)
  switch (kernelSize_.x)
  {
    case 3:
      switch (kernelSize_.y)
      {
        case 3: v2_ = loadSample_(srcCoord_ + ivec2(2, 2), srcLevel_, lfs_);
        case 2: v1_ = loadSample_(srcCoord_ + ivec2(2, 1), srcLevel_, lfs_);
        case 1: v0_ = loadSample_(srcCoord_ + ivec2(2, 0), srcLevel_, lfs_);
      }
      switch (kernelSize_.y)
      {
        case 3: NVPRO_PYRAMID_REDUCE(w0_, v0_, w1_, v1_, w2_, v2_, h2_); break;
        case 2: NVPRO_PYRAMID_REDUCE2(v0_, v1_, h2_); break;
        case 1: h2_ = v0_; break;
      }
      // fallthru
    case 2:
      switch (kernelSize_.y)
      {
        case 3: v2_ = loadSample_(srcCoord_ + ivec2(1, 2), srcLevel_, lfs_);
        case 2: v1_ = loadSample_(srcCoord_ + ivec2(1, 1), srcLevel_, lfs_);
        case 1: v0_ = loadSample_(srcCoord_ + ivec2(1, 0), srcLevel_, lfs_);
      }
      switch (kernelSize_.y)
      {
        case 3: NVPRO_PYRAMID_REDUCE(w0_, v0_, w1_, v1_, w2_, v2_, h1_); break;
        case 2: NVPRO_PYRAMID_REDUCE2(v0_, v1_, h1_); break;
        case 1: h1_ = v0_; break;
      }
    case 1:
      switch (kernelSize_.y)
      {
        case 3: v2_ = loadSample_(srcCoord_ + ivec2(0, 2), srcLevel_, lfs_);
        case 2: v1_ = loadSample_(srcCoord_ + ivec2(0, 1), srcLevel_, lfs_);
        case 1: v0_ = loadSample_(srcCoord_ + ivec2(0, 0), srcLevel_, lfs_);
      }
      switch (kernelSize_.y)
      {
        case 3: NVPRO_PYRAMID_REDUCE(w0_, v0_, w1_, v1_, w2_, v2_, h0_); break;
        case 2: NVPRO_PYRAMID_REDUCE2(v0_, v1_, h0_); break;
        case 1: h0_ = v0_; break;
      }
  }

  // Reduce up to 3 samples horizontally.
  switch (kernelSize_.x)
  {
    case 3:
      n_   = dstImageSize_.x;
      rcp_ = 1.0f / (2 * n_ + 1);
      w0_  = rcp_ * (n_ - dstCoord_.x);
      w1_  = rcp_ * n_;
      w2_  = 1.0f - w0_ - w1_;
      NVPRO_PYRAMID_REDUCE(w0_, h0_, w1_, h1_, w2_, h2_, out_);
      break;
    case 2:
      NVPRO_PYRAMID_REDUCE2(h0_, h1_, out_);
      break;
    case 1:
      out_ = h0_;
  }

  // Write out sample.
  NVPRO_PYRAMID_STORE(dstCoord_, dstLevel_, out_);
  return out_;
}

NVPRO_PYRAMID_TYPE
loadSample_(ivec2 srcCoord_, int srcLevel_, bool loadFromShared_)
{
  NVPRO_PYRAMID_TYPE loaded_;
  if (loadFromShared_)
  {
    NVPRO_PYRAMID_SHARED_LOAD((sharedLevel_[srcCoord_.y][srcCoord_.x]), loaded_);
  }
  else
  {
    NVPRO_PYRAMID_LOAD(srcCoord_, srcLevel_, loaded_);
  }
  return loaded_;
}



// Compute and write out (to the 1st mip level generated) the samples
// at coordinates
//     initDstCoord_,
//     initDstCoord_ + step_, ...
//     initDstCoord_ + (iterations_-1) * step_
// and cache them at in the sharedLevel_ tile at coordinates
//     initSharedCoord_,
//     initSharedCoord_ + step_, ...
//     initSharedCoord_ + (iterations_-1) * step_
// If boundsCheck_ is true, skip coordinates that are out of bounds.
void intermediateLevelLoop_(ivec2 initDstCoord_,
                            ivec2 initSharedCoord_,
                            ivec2 step_,
                            int   iterations_,
                            bool  boundsCheck_)
{
  ivec2 dstCoord_     = initDstCoord_;
  ivec2 sharedCoord_  = initSharedCoord_;
  int   srcLevel_     = int(NVPRO_PYRAMID_INPUT_LEVEL_);
  int   dstLevel_     = srcLevel_ + 1;
  ivec2 srcImageSize_ = NVPRO_PYRAMID_LEVEL_SIZE(srcLevel_);
  ivec2 dstImageSize_ = NVPRO_PYRAMID_LEVEL_SIZE(dstLevel_);
  ivec2 kernelSize_   = kernelSizeFromInputSize_(srcImageSize_);

  for (int i_ = 0; i_ < iterations_; ++i_)
  {
    ivec2 srcCoord_ = dstCoord_ * 2;

    // Optional bounds check.
    if (boundsCheck_)
    {
      if (uint(dstCoord_.x) >= uint(dstImageSize_.x)) continue;
      if (uint(dstCoord_.y) >= uint(dstImageSize_.y)) continue;
    }

    bool loadFromShared_ = false;
    NVPRO_PYRAMID_TYPE sample_ =
        reduceStoreSample_(srcCoord_, srcLevel_, loadFromShared_, kernelSize_,
                           dstImageSize_, dstCoord_, dstLevel_);

    // Above function handles writing to the actual output; manually
    // cache into shared memory here.
    NVPRO_PYRAMID_SHARED_STORE((sharedLevel_[sharedCoord_.y][sharedCoord_.x]),
                               sample_);
    dstCoord_ += step_;
    sharedCoord_ += step_;
  }
}

// Function for the workgroup that handles filling the intermediate level
// (caching it in shared memory as well).
//
// We need somewhere from 16x16 to 17x17 samples, depending
// on what the kernel size for the 2nd mip level generation will be.
//
// dstTileCoord_ : upper left coordinate of the tile to generate.
// boundsCheck_  : whether to skip samples that are out-of-bounds.
void fillIntermediateTile_(ivec2 dstTileCoord_, bool boundsCheck_)
{
  uint localIdx_ = int(gl_LocalInvocationIndex);

  ivec2 initThreadOffset_;
  ivec2 step_;
  int   iterations_;

  ivec2 dstImageSize_ =
      NVPRO_PYRAMID_LEVEL_SIZE((int(NVPRO_PYRAMID_INPUT_LEVEL_) + 1));
  ivec2 futureKernelSize_ = kernelSizeFromInputSize_(dstImageSize_);

  if (futureKernelSize_.x == 3)
  {
    if (futureKernelSize_.y == 3)
    {
      // Fill in 2 17x7 steps and 1 17x3 step (9 idle threads)
      initThreadOffset_ = ivec2(localIdx_ % 17u, localIdx_ / 17u);
      step_             = ivec2(0, 7);
      iterations_       = localIdx_ >= 7 * 17 ? 0 : localIdx_ < 3 * 17 ? 3 : 2;
    }
    else  // Future 3x[2,1] kernel
    {
      // Fill in 2 8x16 steps and 1 1x16 step
      initThreadOffset_ = ivec2(localIdx_ / 16u, localIdx_ % 16u);
      step_             = ivec2(8, 0);
      iterations_       = localIdx_ < 1 * 16 ? 3 : 2;
    }
  }
  else
  {
    if (futureKernelSize_.y == 3)
    {
      // Fill in 2 16x8 steps and 1 16x1 step
      initThreadOffset_ = ivec2(localIdx_ % 16u, localIdx_ / 16u);
      step_             = ivec2(0, 8);
      iterations_       = localIdx_ < 1 * 16 ? 3 : 2;
    }
    else
    {
      // Fill in 2 16x8 steps
      initThreadOffset_ = ivec2(localIdx_ % 16u, localIdx_ / 16u);
      step_             = ivec2(0, 8);
      iterations_       = 2;
    }
  }

  intermediateLevelLoop_(dstTileCoord_ + initThreadOffset_, initThreadOffset_,
                         step_, iterations_, boundsCheck_);
}



// Function for the workgroup that handles filling the last level tile
// (2nd level after the original input level), using as input the
// tile in shared memory.
//
// dstTileCoord_ : upper left coordinate of the tile to generate.
// boundsCheck_  : whether to skip samples that are out-of-bounds.
void fillLastTile_(ivec2 dstTileCoord_, bool boundsCheck_)
{
  uint localIdx_ = gl_LocalInvocationIndex;

  if (localIdx_ < 8 * 8)
  {
    ivec2 threadOffset_ = ivec2(localIdx_ % 8u, localIdx_ / 8u);
    int   srcLevel_     = int(NVPRO_PYRAMID_INPUT_LEVEL_) + 1;
    int   dstLevel_     = int(NVPRO_PYRAMID_INPUT_LEVEL_) + 2;
    ivec2 srcImageSize_ = NVPRO_PYRAMID_LEVEL_SIZE(srcLevel_);
    ivec2 dstImageSize_ = NVPRO_PYRAMID_LEVEL_SIZE(dstLevel_);

    ivec2 srcSharedCoord_ = threadOffset_ * 2;
    bool loadFromShared_  = true;
    ivec2 kernelSize_     = kernelSizeFromInputSize_(srcImageSize_);
    ivec2 dstCoord_       = threadOffset_ + dstTileCoord_;

    bool inBounds_ = true;
    if (boundsCheck_)
    {
      inBounds_ = (uint(dstCoord_.x) < uint(dstImageSize_.x))
                  && (uint(dstCoord_.y) < uint(dstImageSize_.y));
    }
    if (inBounds_)
    {
      reduceStoreSample_(srcSharedCoord_, 0, loadFromShared_, kernelSize_,
                         dstImageSize_, dstCoord_, dstLevel_);
    }
  }
}



void nvproPyramidMain()
{
  int inputLevel_ = int(NVPRO_PYRAMID_INPUT_LEVEL_);

  if (NVPRO_PYRAMID_LEVEL_COUNT_ == 1u)
  {
    ivec2 kernelSize_ =
        kernelSizeFromInputSize_(NVPRO_PYRAMID_LEVEL_SIZE(inputLevel_));
    ivec2 dstImageSize_ = NVPRO_PYRAMID_LEVEL_SIZE((inputLevel_ + 1));
    ivec2 dstCoord_     = ivec2(int(gl_GlobalInvocationID.x) % dstImageSize_.x,
                                int(gl_GlobalInvocationID.x) / dstImageSize_.x);
    ivec2 srcCoord_ = dstCoord_ * 2;

    if (dstCoord_.y < dstImageSize_.y)
    {
      reduceStoreSample_(srcCoord_, inputLevel_, false, kernelSize_,
                         dstImageSize_, dstCoord_, inputLevel_ + 1);
    }
  }
  else  // Handling two levels.
  {
    // Assign a 8x8 tile of mip level inputLevel_ + 2 to this workgroup.
    int   level2_     = inputLevel_ + 2;
    ivec2 level2Size_ = NVPRO_PYRAMID_LEVEL_SIZE(level2_);
    ivec2 tileCount_;
    tileCount_.x   = int(uint(level2Size_.x + 7) / 8u);
    tileCount_.y   = int(uint(level2Size_.y + 7) / 8u);
    ivec2 tileIdx_ = ivec2(gl_WorkGroupID.x % uint(tileCount_.x),
                           gl_WorkGroupID.x / uint(tileCount_.x));
    uint localIdx_ = gl_LocalInvocationIndex;

    // Determine if bounds checking is needed; this is only the case
    // for tiles at the right or bottom fringe that might be cut off
    // by the image border. Note that later, I use if statements rather
    // than passing boundsCheck_ directly to convince the compiler
    // to inline everything.
    bool boundsCheck_ = tileIdx_.x >= tileCount_.x - 1 ||
                        tileIdx_.y >= tileCount_.y - 1;

    if (boundsCheck_)
    {
      // Compute the tile in level inputLevel_ + 1 that's needed to
      // compute the above 8x8 tile.
      fillIntermediateTile_(tileIdx_ * 2 * ivec2(8, 8), true);
      barrier();

      // Compute the inputLevel_ + 2 tile of size 8x8, loading
      // inupts from shared memory.
      fillLastTile_(tileIdx_ * ivec2(8, 8), true);
    }
    else
    {
      // Same with no bounds checking.
      fillIntermediateTile_(tileIdx_ * 2 * ivec2(8, 8), false);
      barrier();
      fillLastTile_(tileIdx_ * ivec2(8, 8), false);
    }
  }
}

#endif /* !NVPRO_USE_GENERAL_PIPELINE_ALTERNATIVE_ */
#endif /* !NVPRO_PYRAMID_IS_FAST_PIPELINE */

#undef NVPRO_PYRAMID_2D_REDUCE_
#undef NVPRO_PYRAMID_LEVEL_COUNT_
#undef NVPRO_PYRAMID_INPUT_LEVEL_
