// Copyright 2021-2024 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef VK_COMPUTE_MIPMAPS_CAMERA_CONTROLS_HPP_
#define VK_COMPUTE_MIPMAPS_CAMERA_CONTROLS_HPP_

#include <glm/ext/vector_float2.hpp>
#include <vulkan/vulkan_core.h>

#include "nvh/cameramanipulator.hpp"

struct CameraTransforms;
struct SwapImagePushConstant;

struct CameraControls
{
  int   sceneMode   = 0;
  int   filterMode  = 0;
  float explicitLod = 0.0F;

  float backgroundBrightness = 0.01F;

  // 2D camera controls
  // Texel coord is offset + scale * pixelCoordinate.
  glm::vec2 offset = {0, 0}, scale = {1, 1};

  // 3D camera controls
  nvh::CameraManipulator::Camera camera;
};

void updateFromControls(
    const CameraControls& controls,
    VkViewport            viewport,
    CameraTransforms&     outTransforms);

void updateFromControls(
    const CameraControls&  controls,
    SwapImagePushConstant& outPushConstant);

#endif /* !VK_COMPUTE_MIPMAPS_CAMERA_CONTROLS_HPP_ */
