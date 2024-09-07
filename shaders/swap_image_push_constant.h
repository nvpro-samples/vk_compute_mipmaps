// Copyright 2021-2024 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Push constant for the pipeline that paints to the screen.

#ifndef NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SWAP_IMAGE_PUSH_CONSTANT_H_
#define NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SWAP_IMAGE_PUSH_CONSTANT_H_

#ifdef __cplusplus
#include <limits.h>
#include <glm/ext/vector_float2.hpp>
#define vec2 glm::vec2
// Make sure an int is 32 bits
static_assert(sizeof(int) == 4);
static_assert(CHAR_BIT == 8);
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
#endif

#endif
