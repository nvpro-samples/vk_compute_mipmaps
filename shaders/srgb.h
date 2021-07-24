// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SHADERS_SRGB_H_
#define NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SHADERS_SRGB_H_

#ifdef __cplusplus
#include "nvmath/nvmath.h"
#include "nvmath/nvmath_glsltypes.h"
#define uint uint32_t
#define vec4 nvmath::vec4f
#define clamp nvmath::nv_clamp<float>
#else
#define inline
#endif

// Convert 8-bit sRGB red/green/blue component value to linear.
inline float linearFromSrgb(uint arg)
{
  #ifdef __cplusplus
    arg = arg > 255u ? 255u : arg;
  #else
    arg = min(arg, 255u);
  #endif
  float u = arg * (1.0f / 255.0f);
  return u <= 0.04045 ? u * (25.0f / 323.0f)
                      : pow((200.0f * u + 11.0f) * (1.0f / 211.0f), 2.4f);
}

inline uint srgbFromLinearBias(float arg, float bias)
{
  float srgb = arg <= 0.0031308f ? (323.0f / 25.0f) * arg
                                 : 1.055f * pow(arg, 1.0f / 2.4f) - 0.055f;
  return uint(clamp(srgb * 255.0f + bias, 0, 255));
}

// Convert float linear red/green/blue value to 8-bit sRGB component.
inline uint srgbFromLinear(float arg)
{
  return srgbFromLinearBias(arg, 0.5);
}

#ifdef __cplusplus
#undef uint
#undef vec4
#undef clamp

#else
#undef inline
vec3 linearFromSrgbVec(uvec3 arg)
{
  return vec3(linearFromSrgb(arg.r), linearFromSrgb(arg.g),
              linearFromSrgb(arg.b));
}

vec4 linearFromSrgbVec(uvec4 arg)
{
  return vec4(linearFromSrgb(arg.r), linearFromSrgb(arg.g),
              linearFromSrgb(arg.b), arg.a * (1.0f / 255.0f));
}

uvec4 srgbFromLinearVec(vec4 arg)
{
  uint alpha = clamp(uint(arg.a * 255.0f + 0.5f), 0u, 255u);
  return uvec4(srgbFromLinear(arg.r), srgbFromLinear(arg.g),
               srgbFromLinear(arg.b), alpha);
}

#endif

#endif // Include guard
