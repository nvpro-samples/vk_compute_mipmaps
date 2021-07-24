// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Polyglot include file (GLSL and C++).
#ifndef VK_COMPUTE_MIPMAPS_FILTER_MODES_H_
#define VK_COMPUTE_MIPMAPS_FILTER_MODES_H_

#define VK_COMPUTE_MIPMAPS_FILTER_MODE_TRILINEAR                0
#define VK_COMPUTE_MIPMAPS_FILTER_MODE_TRILINEAR_EXPLICIT_LOD   1
#define VK_COMPUTE_MIPMAPS_FILTER_MODE_NEAREST_EXPLICIT_LOD     2
#define VK_COMPUTE_MIPMAPS_FILTER_MODE_COUNT                    3

#ifdef __cplusplus
static const char* filterModeLabels[] = {
    "trilinear",
    "trilinear (LoD explicit)",
    "nearest (LoD explicit)",
    "n/a",  // Used when in "show all mips" scene (filter mode not applicable)
    "[out of bounds]",
};
#endif

#endif
