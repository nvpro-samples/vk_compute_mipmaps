// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Push constant for the pipeline that paints to the screen.

#ifndef NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SWAP_IMAGE_PUSH_CONSTANT_H_
#define NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SWAP_IMAGE_PUSH_CONSTANT_H_

#ifdef __cplusplus
#include <stdint.h>
#include "nvmath/nvmath_glsltypes.h" // emulate glsl types in C++
#define vec2 nvmath::vec2
#define int  int32_t      /* Possibly undefined behavior? */
#endif

struct SwapImagePushConstant
{
  // Used to scale the base color texture.
  // Sampled texel coordinate is
  //     gl_FragCoord.xy * texelScale + texelOffset;
  // Downscale by base mip level size to get normalized texture coordinate.
  vec2 texelScale, texelOffset;

  int   filterMode;
  int   sceneMode;
  float explicitLod;
  float backgroundBrightness;
};

#ifdef __cplusplus
#undef vec2
#undef int
#endif

#endif
