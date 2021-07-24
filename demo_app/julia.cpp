// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Implementation functions for Julia Set compute shader.
#include "julia.hpp"

#include "nvh/container_utils.hpp"
#include "make_compute_pipeline.hpp"

// GLSL polyglot
#include "shaders/julia.h"

Julia::Julia(VkDevice         device,
             VkPhysicalDevice physicalDevice,
             bool             dumpPipelineStats,
             uint32_t         textureWidth,
             uint32_t         textureHeight)
    : m_device(device)
    , m_scopedImage(device, physicalDevice)
{
  // Push constant defaults
  update(0.0, 64);

  // Set up color texture.
  m_scopedImage.reallocImage(textureWidth, textureHeight);

  // Set up compute pipeline.
  uint32_t              pcSize = uint32_t(sizeof(JuliaPushConstant));
  VkPushConstantRange   range  = {VK_SHADER_STAGE_COMPUTE_BIT, 0, pcSize};
  VkDescriptorSetLayout descriptorLayout =
      m_scopedImage.getStorageDescriptorSetLayout();
  makeComputePipeline(m_device, "julia.comp.spv", dumpPipelineStats,
                      1, &descriptorLayout, 1, &range,
                      &m_pipeline, &m_pipelineLayout);
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
  m_alphaNormalized += uint32_t(dt * 0x0600'0000);
  double alphaRadians   = m_alphaNormalized * 1.4629180792671596e-09;
  m_pushConstant.c_real = float(0.7885 * sin(alphaRadians));
  m_pushConstant.c_imag = float(0.7885 * cos(alphaRadians));

  float textureWidth         = float(m_scopedImage.getImageWidth());
  float textureHeight        = float(m_scopedImage.getImageHeight());
  m_pushConstant.offset_real = -2.0f;
  m_pushConstant.scale       = 4.0f / textureWidth;
  m_pushConstant.offset_imag = 2.0f * textureHeight / textureWidth;

  if (maxIterations > 0)
    m_pushConstant.maxIterations = int32_t(maxIterations);
}

void Julia::cmdFillColorTexture(VkCommandBuffer cmdBuf)
{
  // Transition color texture image to general layout, protect
  // earlier reads.
  VkImageMemoryBarrier imageBarrier = {
      VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr,
      VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
      0, 0, m_scopedImage.getImage(),
      {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1} };
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, nullptr, 0, nullptr, 1, &imageBarrier);

  // Bind pipeline, push constant, and descriptors.
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  VkDescriptorSet descriptorSet = m_scopedImage.getStorageDescriptorSet();
  vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(m_pushConstant), &m_pushConstant);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                          m_pipelineLayout, 0, 1, &descriptorSet,
                          0, nullptr);
  // Fill the image.
  vkCmdDispatch(cmdBuf, (m_scopedImage.getImageWidth() + 15) / 16,
                (m_scopedImage.getImageHeight() + 15) / 16, 1);

  // Pipeline barrier.
  VkMemoryBarrier barrier = {
      VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
      VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT };
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 1, &barrier,
                       0, nullptr, 0, nullptr);
}
