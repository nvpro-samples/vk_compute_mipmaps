// Baseline comparison shader, creates one sample for the next mip
// level per thread, no shuffles or shared memory optimizations.
// General case shader: works for odd dimensions.
// Each workgroup handles up to 256 output samples.

layout(local_size_x = 256) in;

void nvproPyramidMain()
{
  int   inputLevel_   = NVPRO_PYRAMID_INPUT_LEVEL_;
  ivec2 srcImageSize_ = NVPRO_PYRAMID_LEVEL_SIZE(inputLevel_);
  ivec2 dstImageSize_ = NVPRO_PYRAMID_LEVEL_SIZE((inputLevel_ + 1));
  ivec2 dstCoord_     = ivec2(gl_GlobalInvocationID.x % dstImageSize_.x,
                              gl_GlobalInvocationID.x / dstImageSize_.x);
  ivec2 srcCoord_     = dstCoord_ << 1u;

  if (dstCoord_.y >= dstImageSize_.y) return;

  ivec2 kernelSize_ =
      ivec2(srcImageSize_.x & 1 | (srcImageSize_.x != 1 ? 2 : 0),
            srcImageSize_.y & 1 | (srcImageSize_.y != 1 ? 2 : 0));
  NVPRO_PYRAMID_TYPE v0_, v1_, v2_, h0_, h1_, h2_;

  // http://download.nvidia.com/developer/Papers/2005/NP2_Mipmapping/NP2_Mipmap_Creation.pdf
  float n_   = dstImageSize_.y; \
  float rcp_ = 1.0f / (2 * n_ + 1); \
  float w0_  = rcp_ * (n_ - dstCoord_.y); \
  float w1_  = rcp_ * n_; \
  float w2_  = 1.0f - w0_ - w1_; \

  // Reduce vertically up to 3 times (depending on kernel horizontal size).
  switch (kernelSize_.x)
  {
    case 3:
      // Load up to 3 samples (fallthru intended).
      switch (kernelSize_.y)
      {
        case 3: NVPRO_PYRAMID_LOAD((srcCoord_ + ivec2(2, 2)), inputLevel_, v2_);
        case 2: NVPRO_PYRAMID_LOAD((srcCoord_ + ivec2(2, 1)), inputLevel_, v1_);
        case 1: NVPRO_PYRAMID_LOAD((srcCoord_ + ivec2(2, 0)), inputLevel_, v0_);
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
        case 3: NVPRO_PYRAMID_LOAD((srcCoord_ + ivec2(1, 2)), inputLevel_, v2_);
        case 2: NVPRO_PYRAMID_LOAD((srcCoord_ + ivec2(1, 1)), inputLevel_, v1_);
        case 1: NVPRO_PYRAMID_LOAD((srcCoord_ + ivec2(1, 0)), inputLevel_, v0_);
      }
      switch (kernelSize_.y)
      {
        case 3: NVPRO_PYRAMID_REDUCE(w0_, v0_, w1_, v1_, w2_, v2_, h1_); break;
        case 2: NVPRO_PYRAMID_REDUCE2(v0_, v1_, h1_); break;
        case 1: h1_ = v0_; break;
      }
    default:
      switch (kernelSize_.y)
      {
        case 3: NVPRO_PYRAMID_LOAD((srcCoord_ + ivec2(0, 2)), inputLevel_, v2_);
        case 2: NVPRO_PYRAMID_LOAD((srcCoord_ + ivec2(0, 1)), inputLevel_, v1_);
        case 1: NVPRO_PYRAMID_LOAD((srcCoord_ + ivec2(0, 0)), inputLevel_, v0_);
      }
      switch (kernelSize_.y)
      {
        case 3: NVPRO_PYRAMID_REDUCE(w0_, v0_, w1_, v1_, w2_, v2_, h0_); break;
        case 2: NVPRO_PYRAMID_REDUCE2(v0_, v1_, h0_); break;
        case 1: h0_ = v0_; break;
      }
  }

  // Reduce horizontally, calculate horizontal weights if needed.
  NVPRO_PYRAMID_TYPE out_;
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
    default:
      out_ = h0_;
  }

  // Store the output.
  NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_ + 1), out_);
}
