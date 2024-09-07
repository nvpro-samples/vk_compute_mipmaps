// Copyright 2021-2024 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SHADERS_SRGB_H_
#define NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SHADERS_SRGB_H_

#ifdef __cplusplus
#include <math.h>
#include <stdint.h>
#include <glm/glm.hpp>
#define uint uint32_t
#define vec4 glm::vec4f
#define clamp glm::clamp
#else
#define inline
#endif

// Convert 8-bit sRGB red/green/blue component value to linear.
inline float linearFromSrgb(uint arg)
{
#ifdef __cplusplus
  arg = arg > 255U ? 255U : arg;
#else
  arg = min(arg, 255u);
#endif
  float u = float(arg) * (1.0F / 255.0F);
  return u <= 0.04045F ? u * (25.0F / 323.0F) :
                         pow((200.0F * u + 11.0F) * (1.0F / 211.0F), 2.4F);
}

inline uint srgbFromLinearBias(float arg, float bias)
{
  float srgb = arg <= 0.0031308F ? (323.0F / 25.0F) * arg :
                                   1.055F * pow(arg, 1.0F / 2.4F) - 0.055F;
  return uint(clamp(srgb * 255.0F + bias, 0.F, 255.F));
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
  return vec3(
      linearFromSrgb(arg.r), linearFromSrgb(arg.g), linearFromSrgb(arg.b));
}

vec4 linearFromSrgbVec(uvec4 arg)
{
  return vec4(
      linearFromSrgb(arg.r), linearFromSrgb(arg.g), linearFromSrgb(arg.b),
      arg.a * (1.0f / 255.0f));
}

uvec4 srgbFromLinearVec(vec4 arg)
{
  uint alpha = clamp(uint(arg.a * 255.0f + 0.5f), 0u, 255u);
  return uvec4(
      srgbFromLinear(arg.r), srgbFromLinear(arg.g), srgbFromLinear(arg.b),
      alpha);
}

#endif

#endif  // Include guard
