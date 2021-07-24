// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#include "camera_controls.hpp"

#include <math.h>
#include "nvmath/nvmath.h"

#include "shaders/camera_transforms.h"
#include "shaders/filter_modes.h"
#include "shaders/scene_modes.h"
#include "shaders/swap_image_push_constant.h"

void updateFromControls(const CameraControls& controls,
                        VkViewport            viewport,
                        CameraTransforms*     outTransforms)
{
  assert(controls.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_3D);
  assert(viewport.minDepth == 0 && viewport.maxDepth == 1);
  float aspectRatio = float(viewport.width) / float(viewport.height);

  nvmath::vec3f up     = controls.camera.up;
  nvmath::vec3f center = controls.camera.ctr;
  nvmath::mat4  view   = nvmath::look_at(controls.camera.eye, center, up);
  nvmath::mat4  proj =
      nvmath::perspectiveVK(controls.camera.fov, aspectRatio, 0.1f, 1.0f);

  outTransforms->view         = view;
  outTransforms->proj         = proj;
  outTransforms->viewInverse  = nvmath::invert(view);
  outTransforms->projInverse  = nvmath::invert(proj);
}

void updateFromControls(const CameraControls&  controls,
                        SwapImagePushConstant* outPushConstant)
{
  outPushConstant->texelScale           = controls.scale;
  outPushConstant->texelOffset          = controls.offset;
  outPushConstant->explicitLod          = controls.explicitLod;
  outPushConstant->filterMode           = controls.filterMode;
  outPushConstant->sceneMode            = controls.sceneMode;
  outPushConstant->backgroundBrightness = controls.backgroundBrightness;
}
