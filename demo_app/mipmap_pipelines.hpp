// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef VK_COMPUTE_MIPMAPS_DEMO_MIPMAP_PIPELINES_HPP_
#define VK_COMPUTE_MIPMAPS_DEMO_MIPMAP_PIPELINES_HPP_

#include "pipeline_alternative.hpp"

#include <vulkan/vulkan.h>

class ScopedImage;
struct PipelineAlternative;

// Class holding compute shader pipelines that computes srgba8
// mipmaps.  There are a lot of pipelines stored here, due to testing
// the performance effects of changes.
class ComputeMipmapPipelines
{
protected:
  ComputeMipmapPipelines() = default;

public:
  virtual ~ComputeMipmapPipelines() = default;

  // Record a command to generate mipmaps for the specified image
  // using info stored in the base level and the named pipeline
  // alternatives. No barrier is included before (i.e. it's your
  // responsibility), but a barrier is included after for read
  // visibility to fragment shaders.
  virtual void cmdBindGenerate(VkCommandBuffer            cmdBuf,
                               const ScopedImage&         imageToMipmap,
                               const PipelineAlternative& alternative) = 0;

  static ComputeMipmapPipelines*
  make(VkDevice device, const ScopedImage& image, bool dumpPipelineStats);
};

#endif
