// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#include "drawing.hpp"

#include "nvh/container_utils.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/shaders_vk.hpp"

#include "search_paths.hpp"

// GLSL polyglots
#include "shaders/camera_transforms.h"
#include "shaders/swap_image_push_constant.h"

SwapRenderPass::SwapRenderPass(VkDevice device, VkFormat colorFormat)
{
  m_device = device;

  VkAttachmentDescription colorAttachment{};
  colorAttachment.format         = colorFormat;
  colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments    = &colorAttachmentRef;

  VkSubpassDependency dependency{};
  dependency.srcSubpass   = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass   = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                            | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  dependency.srcAccessMask =
      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments    = &colorAttachment;
  renderPassInfo.subpassCount    = 1;
  renderPassInfo.pSubpasses      = &subpass;
  renderPassInfo.dependencyCount = 1;
  renderPassInfo.pDependencies   = &dependency;

  NVVK_CHECK(
      vkCreateRenderPass(device, &renderPassInfo, nullptr, &m_renderPass));
}

void SwapFramebuffers::recreateNowIfNeeded(nvvk::SwapChain& swapChain) noexcept
{
  if (initialized() && swapChain.getChangeID() == m_lastChangeId)
  {
    return;
  }

  // Destroy old resources.
  destroyFramebuffers();

  auto width  = swapChain.getWidth();
  auto height = swapChain.getHeight();

  // Make a framebuffer for every swap chain image.
  uint32_t imageCount = swapChain.getImageCount();
  m_framebuffers.resize(imageCount);
  for (uint32_t i = 0; i < imageCount; ++i)
  {
    std::array<VkImageView, 1> attachments = {swapChain.getImageView(i)};

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass      = m_renderPass;
    framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebufferInfo.pAttachments    = attachments.data();
    framebufferInfo.width           = swapChain.getWidth();
    framebufferInfo.height          = swapChain.getHeight();
    framebufferInfo.layers          = 1;

    NVVK_CHECK(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr,
                                   &m_framebuffers.at(i)));
  }

  m_lastChangeId = swapChain.getChangeID();
}

void SwapFramebuffers::destroyFramebuffers()
{
  if (initialized())
  {
    for (VkFramebuffer fb : m_framebuffers)
    {
      vkDestroyFramebuffer(m_device, fb, nullptr);
    }
    m_framebuffers.clear();
  }
  assert(!initialized());
}

SwapImagePipeline::SwapImagePipeline(
    VkDevice              device,
    VkPhysicalDevice      physicalDevice,
    const SwapRenderPass& renderPass,
    VkDescriptorSetLayout samplerDescriptorSetLayout)
    : m_device(device)
{
  // Set up camera UBOs
  m_allocator.init(device, physicalDevice);
  VkBufferCreateInfo bufferInfo = {
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0,
      sizeof(CameraTransforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT };
  for (int i = 0; i < 2; ++i)
  {
    m_cameraBuffers[i] = m_allocator.createBuffer(
        bufferInfo, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_bufferMaps[i] =
        static_cast<CameraTransforms*>(m_allocator.map(m_cameraBuffers[i]));
  }

  // Set up descriptors delivering uniform buffers.
  m_bufferDescriptors.init(device);
  m_bufferDescriptors.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_ALL);
  m_bufferDescriptors.initLayout();
  m_bufferDescriptors.initPool(2);

  VkDescriptorBufferInfo bufferInfos[2];
  VkWriteDescriptorSet   writes[2];
  for (int i = 0; i < 2; ++i)
  {
    bufferInfos[i] = VkDescriptorBufferInfo{m_cameraBuffers[i].buffer, 0,
                                            sizeof(CameraTransforms)};
    writes[i]      = m_bufferDescriptors.makeWrite(i, 0, &bufferInfos[i], 0);
  }
  vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

  // Set up pipeline layout.
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  VkDescriptorSetLayout setLayouts[] = {samplerDescriptorSetLayout,
                                        m_bufferDescriptors.getLayout()};
  pipelineLayoutInfo.setLayoutCount = arraySize(setLayouts);
  pipelineLayoutInfo.pSetLayouts    = setLayouts;
  VkPushConstantRange range{VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                            sizeof(SwapImagePushConstant)};
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges    = &range;
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr,
                                    &m_layout));

  // Hides all the graphics pipeline boilerplate (in particular
  // enabling dynamic viewport and scissor). We just have to
  // disable the depth test and write.
  nvvk::GraphicsPipelineState pipelineState;
  pipelineState.depthStencilState.depthTestEnable  = false;
  pipelineState.depthStencilState.depthWriteEnable = false;

  // Compile shaders and state into graphics pipeline.
  auto vertSpv =
      nvh::loadFile("fullscreen_triangle.vert.spv", true, searchPaths, true);
  auto fragSpv =
      nvh::loadFile("swap_image_pipeline.frag.spv", true, searchPaths, true);
  nvvk::GraphicsPipelineGenerator generator(m_device, m_layout, renderPass,
                                            pipelineState);
  generator.addShader(vertSpv, VK_SHADER_STAGE_VERTEX_BIT);
  generator.addShader(fragSpv, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_pipeline = generator.createPipeline();
}

void SwapImagePipeline::cmdBindDraw(VkCommandBuffer       cmdBuf,
                                    SwapImagePushConstant pushConstant,
                                    CameraTransforms      cameraTransforms,
                                    VkDescriptorSet       baseColorSampler,
                                    bool                  parity)
{
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
  vkCmdPushConstants(cmdBuf, m_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     sizeof pushConstant, &pushConstant);
  *m_bufferMaps[parity]               = cameraTransforms;
  VkDescriptorSet cameraUniformBuffer = m_bufferDescriptors.getSet(parity);
  VkDescriptorSet descriptorSets[]    = {baseColorSampler, cameraUniformBuffer};
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout,
                          0, arraySize(descriptorSets), descriptorSets,
                          0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);
}
