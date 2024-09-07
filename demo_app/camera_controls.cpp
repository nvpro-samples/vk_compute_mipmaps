// Copyright 2021-2024 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

#include "camera_controls.hpp"

#include <math.h>
#include <glm/glm.hpp>

#include "shaders/camera_transforms.h"
#include "shaders/filter_modes.h"
#include "shaders/scene_modes.h"
#include "shaders/swap_image_push_constant.h"

void updateFromControls(
    const CameraControls& controls,
    VkViewport            viewport,
    CameraTransforms&     outTransforms)
{
  assert(controls.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_3D);
  assert(viewport.minDepth == 0 && viewport.maxDepth == 1);
  float aspectRatio =
      static_cast<float>(viewport.width) / static_cast<float>(viewport.height);

  glm::vec3 up     = controls.camera.up;
  glm::vec3 center = controls.camera.ctr;
  glm::mat4 view   = glm::lookAt(glm::vec3(controls.camera.eye), center, up);
  glm::mat4 proj   = glm::perspectiveRH_ZO(
      glm::radians(controls.camera.fov), aspectRatio, 0.1F, 1.0F);
  proj[1][1] *= -1;

  outTransforms.view        = view;
  outTransforms.proj        = proj;
  outTransforms.viewInverse = glm::inverse(view);
  outTransforms.projInverse = glm::inverse(proj);
}

void updateFromControls(
    const CameraControls&  controls,
    SwapImagePushConstant& outPushConstant)
{
  outPushConstant.texelScale           = controls.scale;
  outPushConstant.texelOffset          = controls.offset;
  outPushConstant.explicitLod          = controls.explicitLod;
  outPushConstant.filterMode           = controls.filterMode;
  outPushConstant.sceneMode            = controls.sceneMode;
  outPushConstant.backgroundBrightness = controls.backgroundBrightness;
}
