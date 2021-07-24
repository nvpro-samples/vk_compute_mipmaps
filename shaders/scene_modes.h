// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Polyglot include file (GLSL and C++).
#ifndef VK_COMPUTE_MIPMAPS_SCENE_MODES_H_
#define VK_COMPUTE_MIPMAPS_SCENE_MODES_H_

#define VK_COMPUTE_MIPMAPS_SCENE_MODE_SHOW_ALL_MIPS 0
#define VK_COMPUTE_MIPMAPS_SCENE_MODE_3D            1
#define VK_COMPUTE_MIPMAPS_SCENE_MODE_2D_NOT_TILED  2
#define VK_COMPUTE_MIPMAPS_SCENE_MODE_2D_TILED      3
#define VK_COMPUTE_MIPMAPS_SCENE_MODE_COUNT         4

#ifdef __cplusplus
static const char* sceneModeLabels[VK_COMPUTE_MIPMAPS_SCENE_MODE_COUNT + 1] = {
    "show all mips",
    "3D plane",
    "2D",
    "2D, tiled",
    "[out of bounds]",
};
#endif

#endif
