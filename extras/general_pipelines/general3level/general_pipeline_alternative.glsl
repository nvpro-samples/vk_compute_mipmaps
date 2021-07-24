// Scratch memory for intermediate levels; cache what was also written
// out to main memory. (When T is 16 bytes, this is _just_ under half
// of 48 KiB).
shared NVPRO_PYRAMID_SHARED_TYPE sharedLevel1_[35][35];
shared NVPRO_PYRAMID_SHARED_TYPE sharedLevel2_[17][17];

// Tile sizes for each computed mip level. 0th is the tile size read
// from the input mip level.
shared ivec2 tileSizes_[4];

// Spaghetti code for handling loading and reducing anywhere from 1 to
// 9 samples. Three posibilities for each dimension: reduce 3 (odd
// source image size, not 1), reduce 2 (even source image size), or
// "reduce" 1 (source image size 1, in one dimension).
// For case 3, use the following to compute kernel weights:
// http://download.nvidia.com/developer/Papers/2005/NP2_Mipmapping/NP2_Mipmap_Creation.pdf
//
// loadFromTile_(coord_, out_): macro that loads the given sample of the
// input tile, with coord_ relative to the tile's upper-left corner.
//
// srcOffsetInTile_: Coordinate of the 0,0-th input sample relative to the
// upper-left corner of the input tile.
//
// out_: Where the output is written.
#define NVPRO_PYRAMID_2D_REDUCE_(loadFromTile_, srcOffsetInTile_, out_) \
{ \
  ivec2 kernelSize_ = ivec2(srcTileSize_.x & 1 | (srcTileSize_.x != 1 ? 2 : 0), \
                            srcTileSize_.y & 1 | (srcTileSize_.y != 1 ? 2 : 0)); \
  float n_   = dstImageSize_.y; \
  float rcp_ = 1.0f / (2 * n_ + 1); \
  float w0_  = rcp_ * (n_ - dstCoord_.y); \
  float w1_  = rcp_ * n_; \
  float w2_  = 1.0f - w0_ - w1_; \
  NVPRO_PYRAMID_TYPE v0_, v1_, v2_, h0_, h1_, h2_; \
  /* Reduce vertically up to 3 times (depending on kernel horizontal size). */ \
  switch (kernelSize_.x) { \
  case 3: \
    /* Load up to 3 samples (fallthru intended). */ \
    switch (kernelSize_.y) { \
      case 3: loadFromTile_(((srcOffsetInTile_) + ivec2(2, 2)), v2_); \
      case 2: loadFromTile_(((srcOffsetInTile_) + ivec2(2, 1)), v1_); \
      case 1: loadFromTile_(((srcOffsetInTile_) + ivec2(2, 0)), v0_); \
    } \
    switch (kernelSize_.y) { \
      case 3: NVPRO_PYRAMID_REDUCE(w0_, v0_, w1_, v1_, w2_, v2_, h2_); break; \
      case 2: NVPRO_PYRAMID_REDUCE2(v0_, v1_, h2_); break; \
      case 1: h2_ = v0_; break; \
    } /* fallthru */ \
  case 2: \
    switch (kernelSize_.y) { \
      case 3: loadFromTile_(((srcOffsetInTile_) + ivec2(1, 2)), v2_); \
      case 2: loadFromTile_(((srcOffsetInTile_) + ivec2(1, 1)), v1_); \
      case 1: loadFromTile_(((srcOffsetInTile_) + ivec2(1, 0)), v0_); \
    } \
    switch (kernelSize_.y) { \
      case 3: NVPRO_PYRAMID_REDUCE(w0_, v0_, w1_, v1_, w2_, v2_, h1_); break; \
      case 2: NVPRO_PYRAMID_REDUCE2(v0_, v1_, h1_); break; \
      case 1: h1_ = v0_; break; \
    } \
  case 1: \
    switch (kernelSize_.y) { \
      case 3: loadFromTile_(((srcOffsetInTile_) + ivec2(0, 2)), v2_); \
      case 2: loadFromTile_(((srcOffsetInTile_) + ivec2(0, 1)), v1_); \
      case 1: loadFromTile_(((srcOffsetInTile_) + ivec2(0, 0)), v0_); \
    } \
    switch (kernelSize_.y) { \
      case 3: NVPRO_PYRAMID_REDUCE(w0_, v0_, w1_, v1_, w2_, v2_, h0_); break; \
      case 2: NVPRO_PYRAMID_REDUCE2(v0_, v1_, h0_); break; \
      case 1: h0_ = v0_; break; \
    } \
  } \
  /* Reduce horizontally, calculate horizontal weights if needed. */ \
  switch (kernelSize_.x) \
  { \
  case 3: \
    n_   = dstImageSize_.x; \
    rcp_ = 1.0f / (2 * n_ + 1); \
    w0_  = rcp_ * (n_ - dstCoord_.x); \
    w1_  = rcp_ * n_; \
    w2_  = 1.0f - w0_ - w1_; \
    NVPRO_PYRAMID_REDUCE(w0_, h0_, w1_, h1_, w2_, h2_, out_); \
    break; \
  case 2: \
    NVPRO_PYRAMID_REDUCE2(h0_, h1_, out_); \
    break; \
  case 1: \
    out_ = h0_; \
  } \
}


// Each local workgroup computes a 8x8 tile of the mip level
// firstLevel + levelCount; this takes up to 71x71 texels as input
// from the input mip level (actual amount depends on parity and
// levelCount). Must be dispatched with exactly the right number of
// tiles needed to cover the output.
layout(local_size_x = 256) in;
void nvproPyramidMain()
{
  NVPRO_PYRAMID_TYPE out_;
  ivec2 srcTileSize_, dstTileSize_;
  ivec2 threadOffset_, tileOffset_;
  int   dstTileTexelCount_;
  int   inputLevel_ = NVPRO_PYRAMID_INPUT_LEVEL_;

  ivec2 dstImageSize_ =
      NVPRO_PYRAMID_LEVEL_SIZE((NVPRO_PYRAMID_LEVEL_COUNT_ + inputLevel_));

  // Calculate which output tile this workgroup will handle.
  // Use the coordinate system used by the input mip level.
  ivec2 tileCounts_    = (dstImageSize_ + ivec2(7)) >> 3;
  ivec2 dstTileOffset_ = 8 * ivec2(gl_WorkGroupID.x % tileCounts_.x,
                                  gl_WorkGroupID.x / tileCounts_.x);
  tileOffset_ = dstTileOffset_ << NVPRO_PYRAMID_LEVEL_COUNT_;

  // Calculate the output tile size for the last mip level
  // generated.  Ordinarily 8, but may be less for tiles on the
  // right or bottom fringe. This eliminates the need to do bounds
  // checking later.
  dstTileSize_ = min(ivec2(8, 8), dstImageSize_ - dstTileOffset_);

  if (gl_LocalInvocationID.x == 0)
  {
    tileSizes_[NVPRO_PYRAMID_LEVEL_COUNT_] = dstTileSize_;

    // Calculate more tile sizes by working backwards from the goal.
    // Mostly, multiply the destination tile size by 2, add 1 if the
    // source dimension is odd (in which case kernel will consider 3
    // samples, not 2), but handle source size 1 as a special case.
    int i_ = NVPRO_PYRAMID_LEVEL_COUNT_ - 1;
    do
    {
      ivec2 srcImageSize_ = NVPRO_PYRAMID_LEVEL_SIZE((i_ + inputLevel_));
      srcTileSize_.x = srcImageSize_.x == 1 ? 1
                    : ((dstTileSize_.x * 2) | (srcImageSize_.x & 1));
      srcTileSize_.y = srcImageSize_.y == 1 ? 1
                    : ((dstTileSize_.y * 2) | (srcImageSize_.y & 1));
      tileSizes_[i_]  = srcTileSize_;
      dstTileSize_   = srcTileSize_;
    } while (i_-- != 0);
  }
  barrier();

  // Fill the first mip level after firstLevel, cache in sharedLevel1_.
  if (NVPRO_PYRAMID_LEVEL_COUNT_ >= 1)
  {
    tileOffset_ >>= 1;
    srcTileSize_       = tileSizes_[0];
    dstTileSize_       = tileSizes_[1];
    dstImageSize_      = NVPRO_PYRAMID_LEVEL_SIZE((inputLevel_ + 1));
    dstTileTexelCount_ = dstTileSize_.x * dstTileSize_.y;

    for (int texelIndex_ = int(gl_LocalInvocationID.x);
         texelIndex_ < dstTileTexelCount_;
         texelIndex_ += int(gl_WorkGroupSize.x))
    {
      threadOffset_.x   = texelIndex_ % dstTileSize_.x;
      threadOffset_.y   = texelIndex_ / dstTileSize_.x;
      ivec2 dstCoord_   = tileOffset_ + threadOffset_;

      if ((srcTileSize_ & ivec2(1,1)) == ivec2(0,0))
      {
        // Use special case for even reduction.
        ivec2 srcCoord_ = dstCoord_ * 2;
        NVPRO_PYRAMID_LOAD_REDUCE4(srcCoord_, inputLevel_, out_);
      }
      else
      {
        // NOTE: Implicit dependency on srcTileSize_, dstImageSize_, dstCoord_
        #define loadFromTile_(coordInTile_, out_) \
          NVPRO_PYRAMID_LOAD(((coordInTile_) + 2*tileOffset_), inputLevel_, out_)
        NVPRO_PYRAMID_2D_REDUCE_(loadFromTile_, 2 * threadOffset_, out_)
        #undef loadFromTile_
      }

      // Cache and write output.
      NVPRO_PYRAMID_SHARED_STORE(
          (sharedLevel1_[threadOffset_.y][threadOffset_.x]), out_);
      NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_ + 1), out_);
    }
  }
  barrier();

  // Fill the next mip level, use input from sharedLevel1_ and cache
  // into sharedLevel2_.
  if (NVPRO_PYRAMID_LEVEL_COUNT_ >= 2)
  {
    tileOffset_ >>= 1;
    srcTileSize_       = tileSizes_[1];
    dstTileSize_       = tileSizes_[2];
    dstImageSize_      = NVPRO_PYRAMID_LEVEL_SIZE((inputLevel_ + 2));
    dstTileTexelCount_ = dstTileSize_.x * dstTileSize_.y;

    for (int texelIndex_ = int(gl_LocalInvocationID.x);
         texelIndex_ < dstTileTexelCount_;
         texelIndex_ += int(gl_WorkGroupSize.x))
    {
      threadOffset_.x = texelIndex_ % dstTileSize_.x;
      threadOffset_.y = texelIndex_ / dstTileSize_.x;
      ivec2 dstCoord_ = tileOffset_ + threadOffset_;

      #define loadFromTile_(coordInTile_, out_) \
        NVPRO_PYRAMID_SHARED_LOAD( \
          (sharedLevel1_[coordInTile_.y][coordInTile_.x]), out_)
      NVPRO_PYRAMID_2D_REDUCE_(loadFromTile_, 2 * threadOffset_, out_)
      #undef loadFromTile_

      // Cache and write output.
      NVPRO_PYRAMID_SHARED_STORE(
          (sharedLevel2_[threadOffset_.y][threadOffset_.x]), out_);
      NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_ + 2), out_);
    }
  }
  barrier();

  // Fill in the next mip level, use input from hsaredLevel2_, just
  // write out to main memory.
  if (NVPRO_PYRAMID_LEVEL_COUNT_ >= 3)
  {
    tileOffset_ >>= 1;
    srcTileSize_       = tileSizes_[2];
    dstTileSize_       = tileSizes_[3];
    dstImageSize_      = NVPRO_PYRAMID_LEVEL_SIZE((inputLevel_ + 3));
    dstTileTexelCount_ = dstTileSize_.x * dstTileSize_.y;

    for (int texelIndex_ = int(gl_LocalInvocationID.x);
         texelIndex_ < dstTileTexelCount_;
         texelIndex_ += int(gl_WorkGroupSize.x))
    {
      threadOffset_.x = texelIndex_ % dstTileSize_.x;
      threadOffset_.y = texelIndex_ / dstTileSize_.x;
      ivec2 dstCoord_ = tileOffset_ + threadOffset_;

      #define loadFromTile_(coordInTile_, out_) \
        NVPRO_PYRAMID_SHARED_LOAD( \
          (sharedLevel2_[coordInTile_.y][coordInTile_.x]), out_)
      NVPRO_PYRAMID_2D_REDUCE_(loadFromTile_, 2 * threadOffset_, out_)
      #undef loadFromTile_

      // Write output.
      NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_ + 3), out_);
    }
  }
}
