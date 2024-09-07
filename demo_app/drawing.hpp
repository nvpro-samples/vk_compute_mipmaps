// Copyright 2021-2024 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
// Pipeline, render pass, and framebuffers related to drawing the
// mipmapped image onto the screen. The pipeline paints the screen
// with a scaled and translated base color texture.
#ifndef VK_COMPUTE_MIPMAPS_DRAWING_HPP_
#define VK_COMPUTE_MIPMAPS_DRAWING_HPP_

#include <array>
#include <stdint.h>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/swapchain_vk.hpp"

// Class for managing the simple one-subpass, no depth buffer
// VkRenderPass for drawing to the swap chain image.
class SwapRenderPass
{
  // Managed by us.
  VkRenderPass m_renderPass{};

  // Borrowed device pointer.
  VkDevice m_device{};

public:
  SwapRenderPass(VkDevice device, VkFormat colorFormat);
  SwapRenderPass(SwapRenderPass&&) = delete;
  ~SwapRenderPass() { vkDestroyRenderPass(m_device, m_renderPass, nullptr); }

  operator VkRenderPass() const { return m_renderPass; }
};

// Manager for framebuffers, one per swap chain image.
class SwapFramebuffers
{
  // From nvvk::SwapChain::getChangeID().  Basically, if this
  // doesn't match that of nvvk::SwapChain, the swap chain has been
  // re-created, and we need to re-create the framebuffers here to
  // match.
  uint32_t m_lastChangeId{};

  // Borrowed device pointer and render pass.
  VkDevice     m_device{};
  VkRenderPass m_renderPass{};

  // framebuffer[i] is the framebuffer for swap image i, as you'd
  // expect.  This is cleared to indicate when this class is in an
  // unitinialized state.
  std::vector<VkFramebuffer> m_framebuffers;

public:
  bool initialized() const noexcept { return !m_framebuffers.empty(); }

  SwapFramebuffers(VkDevice device, const SwapRenderPass& renderPass)
      : m_device(device)
      , m_renderPass(renderPass)
  {
  }

  ~SwapFramebuffers() { destroyFramebuffers(); }

  // Check the swap chain and recreate framebuffer now if needed.
  // (now = no synchronization done; note however that we can rely on
  // FrameManager to wait on the main thread queue to idle before
  // re-creating a swap chain).
  void recreateNowIfNeeded(nvvk::SwapChain& swapChain) noexcept;
  void destroyFramebuffers() noexcept;

  VkFramebuffer operator[](size_t i) const { return m_framebuffers.at(i); }
};

struct SwapImagePushConstant;
struct CameraTransforms;

// Dynamic viewport/scissor pipeline for drawing the mip-mapped image;
// disables depth test and write. The vertex shader hard-codes drawing a
// full-screen triangle.
//
// Takes 2 descriptor sets as input, each containing one combined
// 2D image sampler binding. set 0 is the base color texture, set 1 is
// the camera transforms UBO, which is also managed by this class.
// There are 2 UBOs; alternate per frame.
class SwapImagePipeline
{
  // Borrowed device pointer.
  VkDevice m_device{};

  // We manage these.
  VkPipeline       m_pipeline{};
  VkPipelineLayout m_layout{};

  nvvk::ResourceAllocatorDedicated m_allocator{};
  std::array<nvvk::Buffer, 2>      m_cameraBuffers{};
  std::array<CameraTransforms*, 2> m_bufferMaps{};
  nvvk::DescriptorSetContainer m_bufferDescriptors{};  // One set per buffer.

public:
  // Need to borrow a descriptor set layout with one combined image
  // sampler binding that allows fragment shader use.
  SwapImagePipeline(
      VkDevice              device,
      VkPhysicalDevice      physicalDevice,
      const SwapRenderPass& renderPass,
      VkDescriptorSetLayout samplerDescriptorSetLayout);

  SwapImagePipeline(SwapImagePipeline&&) = delete;

  ~SwapImagePipeline()
  {
    m_allocator.destroy(m_cameraBuffers[0]);
    m_allocator.destroy(m_cameraBuffers[1]);
    m_allocator.deinit();
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_layout, nullptr);
  }

  operator VkPipeline() const { return m_pipeline; }

  VkPipelineLayout getLayout() const { return m_layout; }

  // Bind the pipeline, set the push constant and descriptors, and
  // record commands to draw. Must be called within the render pass
  // used to create the pipeline. The descriptor set must contain one
  // combined image sampler2D binding.  Alternating UBOs are used to
  // pass CameraTransforms; parity must alternate per frame.
  void cmdBindDraw(
      VkCommandBuffer       cmdBuf,
      SwapImagePushConstant pushConstant,
      CameraTransforms      cameraTransforms,
      VkDescriptorSet       baseColorSampler,
      bool                  parity);
};

#endif /* !VK_COMPUTE_MIPMAPS_DRAWING_HPP_ */
