// Incorrect shader that reads the entire 0th mip level and writes
// subsets to the remaining mip levels.
// Dispatch with 1 workgroup per 256 samples of mip level 1.
// This gives an idea of just what the thread launch and image
// load/store operations cost.

layout(local_size_x = 256) in;

void nvproPyramidMain()
{
  int   levelCount_ = NVPRO_PYRAMID_LEVEL_COUNT_;
  ivec2 levelSize_  = NVPRO_PYRAMID_LEVEL_SIZE(1);
  ivec2 coord_      = ivec2(gl_GlobalInvocationID.x / levelSize_.y,
                            gl_GlobalInvocationID.x % levelSize_.y);
  if (coord_.y >= levelSize_.y) return;
  
  vec4 value_;
  NVPRO_PYRAMID_LOAD_REDUCE4((coord_ * 2), 0, value_);
  NVPRO_PYRAMID_STORE(coord_, 1, value_);
  for (int level_ = 2; level_ <= levelCount_; ++level_)
  {
    levelSize_ = NVPRO_PYRAMID_LEVEL_SIZE(level_);
    if (coord_.x >= levelSize_.x || coord_.y >= levelSize_.y) return;
    NVPRO_PYRAMID_STORE(coord_, level_, value_);
  }
}
