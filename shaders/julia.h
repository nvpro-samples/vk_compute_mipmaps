// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_JULIA_H_
#define NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_JULIA_H_

#ifdef __cplusplus
#include <stdint.h>
#else
#define int32_t int
#endif

// Polyglot C++/GLSL file for Normal Julia Set shader.
struct JuliaPushConstant
{
  // f_c(z) = z^2 + c_real + i c_imag
  float c_real, c_imag;

  // z_0 = (scale pixel_x - scale i pixel_y) + (offset_real + i offset_imag)
  // Note negation (convert y down to imaginary up)
  float offset_real, offset_imag, scale;

  int32_t maxIterations;
};

#ifndef __cplusplus
#undef int32_t
#endif

#endif /* !NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_JULIA_H_ */
