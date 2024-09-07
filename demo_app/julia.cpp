// Copyright 2021-2024 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Implementation functions for Julia Set compute shader.
#include "julia.hpp"

#include "make_compute_pipeline.hpp"

// GLSL polyglot
#include "shaders/julia.h"

#include <math.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

Julia::Julia(
    VkDevice         device,
    VkPhysicalDevice physicalDevice,
    bool             dumpPipelineStats,
    uint32_t         textureWidth,
    uint32_t         textureHeight,
    VkSampler        sampler)
    : m_device(device)
    , m_scopedImage(device, physicalDevice, sampler)
{
  // Push constant defaults
  update(0.0, 64);

  // Set up color texture.
  m_scopedImage.reallocImage(textureWidth, textureHeight);

  // Set up compute pipeline.
  const uint32_t pcSize = static_cast<uint32_t>(sizeof(JuliaPushConstant));
  const VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize};
  VkDescriptorSetLayout     descriptorLayout =
      m_scopedImage.getStorageDescriptorSetLayout();
  makeComputePipeline(
      m_device, "julia.comp.spv", dumpPipelineStats, 1, &descriptorLayout, 1,
      &range, &m_pipeline, &m_pipelineLayout);
}

Julia::~Julia()
{
  // Destroy pipelines.
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyPipeline(m_device, m_pipeline, nullptr);
}

void Julia::resize(uint32_t x, uint32_t y)
{
  m_scopedImage.reallocImage(x, y);
  update(0.0, 0);
}

uint32_t Julia::getWidth() const
{
  return m_scopedImage.getImageWidth();
}

uint32_t Julia::getHeight() const
{
  return m_scopedImage.getImageHeight();
}

void Julia::update(double dt, int maxIterations)
{
  m_alphaNormalized += static_cast<uint32_t>(dt * 0x0600'0000);
  const double alphaRadians = m_alphaNormalized * 1.4629180792671596e-09;
  m_pushConstant.c_real     = static_cast<float>(0.7885 * sin(alphaRadians));
  m_pushConstant.c_imag     = static_cast<float>(0.7885 * cos(alphaRadians));

  const float textureWidth = static_cast<float>(m_scopedImage.getImageWidth());
  const float textureHeight =
      static_cast<float>(m_scopedImage.getImageHeight());
  m_pushConstant.offset_real = -2.0F;
  m_pushConstant.scale       = 4.0F / textureWidth;
  m_pushConstant.offset_imag = 2.0F * textureHeight / textureWidth;

  if(maxIterations > 0)
    m_pushConstant.maxIterations = static_cast<int32_t>(maxIterations);
}

void Julia::cmdFillColorTexture(VkCommandBuffer cmdBuf)
{
  // Transition color texture image to general layout, protect
  // earlier reads.
  const VkImageMemoryBarrier imageBarrier = {
      .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask    = VK_ACCESS_MEMORY_READ_BIT,
      .dstAccessMask    = VK_ACCESS_MEMORY_WRITE_BIT,
      .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout        = VK_IMAGE_LAYOUT_GENERAL,
      .image            = m_scopedImage.getImage(),
      .subresourceRange = {
          .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel   = 0,
          .levelCount     = VK_REMAINING_MIP_LEVELS,
          .baseArrayLayer = 0,
          .layerCount     = 1}};
  vkCmdPipelineBarrier(
      cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
      &imageBarrier);

  // Bind pipeline, push constant, and descriptors.
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  VkDescriptorSet descriptorSet = m_scopedImage.getStorageDescriptorSet();
  vkCmdPushConstants(
      cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
      sizeof(m_pushConstant), &m_pushConstant);
  vkCmdBindDescriptorSets(
      cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1,
      &descriptorSet, 0, nullptr);
  // Fill the image.
  vkCmdDispatch(
      cmdBuf, (m_scopedImage.getImageWidth() + 15) / 16,
      (m_scopedImage.getImageHeight() + 15) / 16, 1);

  // Pipeline barrier.
  const VkMemoryBarrier barrier = {
      .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT};
  vkCmdPipelineBarrier(
      cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}
