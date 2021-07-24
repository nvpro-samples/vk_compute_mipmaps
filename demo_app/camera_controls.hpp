// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef VK_COMPUTE_MIPMAPS_CAMERA_CONTROLS_HPP_
#define VK_COMPUTE_MIPMAPS_CAMERA_CONTROLS_HPP_

#include <vulkan/vulkan.h>

#include "nvh/cameramanipulator.hpp"
#include "nvmath/nvmath_glsltypes.h"

#ifdef near /* hax for msvc */
#undef near
#endif
#ifdef far
#undef far
#endif

struct CameraTransforms;
struct SwapImagePushConstant;

struct CameraControls
{
  int   sceneMode   = 0;
  int   filterMode  = 0;
  float explicitLod = 0.0f;

  float backgroundBrightness = 0.01f;

  // 2D camera controls
  // Texel coord is offset + scale * pixelCoordinate.
  nvmath::vec2 offset, scale = {1, 1};

  // 3D camera controls
  nvh::CameraManipulator::Camera camera;
};

void updateFromControls(const CameraControls& controls,
                        VkViewport            viewport,
                        CameraTransforms*     outTransforms);

void updateFromControls(const CameraControls&  controls,
                        SwapImagePushConstant* outPushConstant);

#endif /* !VK_COMPUTE_MIPMAPS_CAMERA_CONTROLS_HPP_ */
