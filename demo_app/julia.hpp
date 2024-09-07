// Copyright 2021-2024 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef VK_COMPUTE_MIPMAPS_DEMO_JULIA_HPP_
#define VK_COMPUTE_MIPMAPS_DEMO_JULIA_HPP_

#include <vulkan/vulkan.h>

#include <array>
#include "nvvk/resourceallocator_vk.hpp"
#include "scoped_image.hpp"

#include "shaders/julia.h"

class ScopedImage;

// Class holding Julia Set state, compute pipelines, and color
// image used to visualize the fractal.
class Julia
{
  // Borrowed
  VkDevice m_device{};

  // Color texture stored inside.
  ScopedImage m_scopedImage;

  // Compute Pipeline.
  VkPipelineLayout m_pipelineLayout{};
  VkPipeline       m_pipeline{};

  // Push Constant (host copy)
  JuliaPushConstant m_pushConstant{};

  // Used for animation.
  uint32_t m_alphaNormalized = 2109710467;

public:
  Julia(
      VkDevice         device,
      VkPhysicalDevice physicalDevice,
      bool             dumpPipelineStats,
      uint32_t         textureWidth,
      uint32_t         textureHeight,
      VkSampler        sampler);

  ~Julia();

  Julia(Julia&&) = delete;

  // Change the size of the drawn image immediately.
  // Consider vkQueueWaitIdle before.
  void resize(uint32_t x, uint32_t y);

  uint32_t getWidth() const;
  ;
  uint32_t getHeight() const;

  // Call every frame (unless you want the animation paused).
  // dt: frame length in seconds
  // maxIterations: optional, maximum iterations to be performed per sample by shader
  void update(double dt, int maxIterations = 0);

  // Record a command that fills the color texture image with data
  // from the simulation state. Inserts barriers to synchronize read
  // access on the same queue to the color texture image. Only writes
  // to the base mip level, but all levels are transitioned to general
  // layout.
  void cmdFillColorTexture(VkCommandBuffer cmdBuf);

  // Get the ScopedImage holding the color texture data.
  const ScopedImage& getColorImage() const { return m_scopedImage; }

  ScopedImage& getColorImage() { return m_scopedImage; }
};

#endif /* !VK_COMPUTE_MIPMAPS_DEMO_JULIA_HPP_ */
