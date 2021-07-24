// Baseline comparison shader, creates one sample for the next mip
// level per thread, no shuffles or shared memory optimizations.
// Only works for even dimension input mip level.
// Each workgroup handles up to 1024 samples.

layout(local_size_x = 256) in;

void nvproPyramidMain()
{
  int   inputLevel_   = NVPRO_PYRAMID_INPUT_LEVEL_;
  ivec2 dstImageSize_ = NVPRO_PYRAMID_LEVEL_SIZE((inputLevel_ + 1));
  ivec2 dstCoord_     = ivec2(gl_GlobalInvocationID.x % dstImageSize_.x,
                              gl_GlobalInvocationID.x / dstImageSize_.x);
  if (dstCoord_.y < dstImageSize_.y)
  {
    NVPRO_PYRAMID_TYPE sample_;
    NVPRO_PYRAMID_LOAD_REDUCE4((dstCoord_ << 1), inputLevel_, sample_);
    NVPRO_PYRAMID_STORE(dstCoord_, (inputLevel_ + 1), sample_);
  }
}
