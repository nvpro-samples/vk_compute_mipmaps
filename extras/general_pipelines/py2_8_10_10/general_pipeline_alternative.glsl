// General-case shader for generating 1 or 2 levels of the mip pyramid.
// When generating 1 level, each workgroup handles up to 256 samples of the
// output mip level. When generating 2 levels, each workgroup handles
// a 10x10 tile of the last (2nd) output mip level, generating up to
// 21x21 samples of the intermediate (1st) output mip level along the way.
//
// Dispatch with y, z = 1
layout(local_size_x = 8 * 32) in;

// When generating 2 levels, the results of generating the intermediate
// level (first level generated) are cached here; this is the input tile
// needed to generate the 10x10 tile of the second level generated.
shared NVPRO_PYRAMID_SHARED_TYPE sharedLevel_[21][21]; // [y][x]

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
// We need somewhere from 20x20 to 21x21 samples, depending
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
      // Fill in 1 21x12 steps and 1 21x9 step (4 idle threads)
      initThreadOffset_ = ivec2(localIdx_ % 21u, localIdx_ / 21u);
      step_             = ivec2(0, 12);
      iterations_       = localIdx_ >= 12 * 21 ? 0 : localIdx_ < 9 * 21 ? 2 : 1;
    }
    else  // Future 3x[2,1] kernel
    {
      // Fill in 1 21x12 steps and 1 21x8 step (4 idle threads)
      initThreadOffset_ = ivec2(localIdx_ % 21u, localIdx_ / 21u);
      step_             = ivec2(0, 12);
      iterations_       = localIdx_ >= 12 * 21 ? 0 : localIdx_ < 8 * 21 ? 2 : 1;
    }
  }
  else
  {
    if (futureKernelSize_.y == 3)
    {
      // Fill in 1 12x21 steps and 1 8x21 step (4 idle threads)
      initThreadOffset_ = ivec2(localIdx_ / 21u, localIdx_ % 21u);
      step_             = ivec2(12, 0);
      iterations_       = localIdx_ >= 12 * 21 ? 0 : localIdx_ < 8 * 21 ? 2 : 1;
    }
    else
    {
      // Fill in 1 20x12 steps and 1 20x8 step (16 idle threads)
      initThreadOffset_ = ivec2(localIdx_ % 20u, localIdx_ / 20u);
      step_             = ivec2(0, 12);
      iterations_       = localIdx_ >= 12 * 20 ? 0 : localIdx_ < 8 * 20 ? 2 : 1;
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

  if (localIdx_ < 10 * 10)
  {
    ivec2 threadOffset_ = ivec2(localIdx_ % 10u, localIdx_ / 10u);
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
    // Assign a 10x10 tile of mip level inputLevel_ + 2 to this workgroup.
    int   level2_     = inputLevel_ + 2;
    ivec2 level2Size_ = NVPRO_PYRAMID_LEVEL_SIZE(level2_);
    ivec2 tileCount_;
    tileCount_.x   = int(uint(level2Size_.x + 9) / 10u);
    tileCount_.y   = int(uint(level2Size_.y + 9) / 10u);
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
      // compute the above 10x10 tile.
      fillIntermediateTile_(tileIdx_ * 2 * ivec2(10, 10), true);
      barrier();

      // Compute the inputLevel_ + 2 tile of size 10x10, loading
      // inupts from shared memory.
      fillLastTile_(tileIdx_ * ivec2(10, 10), true);
    }
    else
    {
      // Same with no bounds checking.
      fillIntermediateTile_(tileIdx_ * 2 * ivec2(10, 10), false);
      barrier();
      fillLastTile_(tileIdx_ * ivec2(10, 10), false);
    }
  }
}
