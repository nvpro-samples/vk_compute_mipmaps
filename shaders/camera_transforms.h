// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Polyglot include file (GLSL and C++) for the camera transforms struct.

#ifndef NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_CAMERA_TRANSFORMS_H_
#define NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_CAMERA_TRANSFORMS_H_

#ifdef __cplusplus
#include <math.h>
#include <nvmath/nvmath_glsltypes.h> // emulate glsl types in C++
#include <stdint.h>
#endif

struct CameraTransforms
{
  #ifdef __cplusplus
    using mat4 = nvmath::mat4;
    using uint = uint32_t;
  #endif

  // View and projection matrices, along with their inverses.
  // If using 2D view, only the proj matrix is meaningful, translates
  // vec4(gl_FragCoord.xy, 0, 1) to texel coordinate.
  // Downscale by base mip level to get normalized texture coord.
  mat4 view;
  mat4 proj;
  mat4 viewInverse;
  mat4 projInverse;
};

#endif
