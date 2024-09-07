// Copyright 2021-2024 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SCOPED_IMAGE_HPP_
#define NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SCOPED_IMAGE_HPP_

#include <array>
#include <assert.h>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vulkan/vulkan_core.h>

#include "stb_image.h"
#include <glm/ext/vector_uint3.hpp>
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"

#include "mipmap_storage.hpp"

#include "shaders/srgb.h"

// Class managing the image (including all mip levels). sRGB RGBA8-only for now.
//
// * Optional CPU-side storage for the images (MipmapStorage)
// * Vulkan Staging Buffer
// * Vulkan Image
// * Descriptors for accessing the image as sampler and storage image.
//
// The storage image samplers show the underlying 8-bit integers
// instead of sRGB, as sRGB images can't be bound as storage images
// (at least at time of writing for NVIDIA cards). The sampler is
// still sRGB-correct.
class ScopedImage
{
  // Borrowed from outside.
  VkDevice m_device;

  // CPU-side mipmap storage. This also defines the structure of the
  // staging buffer (i.e. what portions correspond to what mip levels).
  std::unique_ptr<MipmapStorage<uint8_t, 4>> m_pCpuMipmap;

  // Maximum number of mip levels supported, bounds image edge size to 65536.
  static constexpr uint32_t s_maxMipLevels = 16;

  static constexpr VkDeviceSize s_texelSize = 4;

  // We manage the lifetimes of these. The staging buffer's data layout
  // matches the way MipmapStorage packs its mip levels.

  // allocates 1 VkDeviceMemory per image/buffer.
  nvvk::ResourceAllocatorDedicated m_allocator;

  nvvk::Image  m_imageDedicated{};  // may be null
  uint32_t     m_imageWidth{0};
  uint32_t     m_imageHeight{0};
  uint32_t     m_imageLevels{0};
  nvvk::Buffer m_stagingBufferDedicated{};  // may be null
  void*        m_pStagingBufferMap{};

  // Image views and descriptor sets for images.
  VkImageView m_view{};
  // One view per mip level.
  std::array<VkImageView, s_maxMipLevels> m_uintViews{};

  // 1 descriptor, for binding image as sampled texture (binding=0).
  // Uses the above m_sampler (immutable sampler).
  nvvk::DescriptorSetContainer m_textureDescriptorContainer;

  // Array of descriptors, for binding image as storage image
  // (binding=0).  Each entry corresponds to one mip level.
  // Load and store raw 8-bit unsigned red/green/blue/alpha values.
  nvvk::DescriptorSetContainer m_storageDescriptorContainer;

  // For debug purposes.
  VkClearColorValue m_magenta{{1.F, 0.F, 1.F, 1.F}};

public:
  ScopedImage(VkDevice device, VkPhysicalDevice physicalDevice, VkSampler sampler)
      : m_device(device)
      , m_textureDescriptorContainer(device)
      , m_storageDescriptorContainer(device)
  {
    m_allocator.init(device, physicalDevice);

    // Set up descriptor sets. Both are assuming general layout image.
    m_textureDescriptorContainer.addBinding(
        0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL,
        &sampler);
    m_textureDescriptorContainer.initLayout();
    m_textureDescriptorContainer.initPool(1);

    m_storageDescriptorContainer.addBinding(
        0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, s_maxMipLevels,
        VK_SHADER_STAGE_ALL);
    m_storageDescriptorContainer.initLayout();
    m_storageDescriptorContainer.initPool(1);
  }

  ScopedImage(ScopedImage&&) = delete;

  ~ScopedImage()
  {
    destroyStagingBuffer();
    destroyImage();
    m_allocator.deinit();
  }

  VkDescriptorSetLayout getTextureDescriptorSetLayout() const
  {
    return m_textureDescriptorContainer.getLayout();
  }

  VkDescriptorSet getTextureDescriptorSet() const
  {
    return m_textureDescriptorContainer.getSet(0);
  }

  VkDescriptorSetLayout getStorageDescriptorSetLayout() const
  {
    return m_storageDescriptorContainer.getLayout();
  }

  VkDescriptorSet getStorageDescriptorSet() const
  {
    return m_storageDescriptorContainer.getSet(0);
  }

  // Helper for image barrier boilerplate
  void cmdImageBarrier(
      VkCommandBuffer      cmdBuf,
      VkPipelineStageFlags srcStage,
      VkPipelineStageFlags dstStage,
      VkAccessFlags        srcAccess,
      VkAccessFlags        dstAccess,
      VkImageLayout        oldLayout,
      VkImageLayout        newLayout,
      uint32_t             baseMipLevel = 0,
      uint32_t             levelCount   = VK_REMAINING_MIP_LEVELS) const
  {
    const VkImageMemoryBarrier barrier = {
        .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask    = srcAccess,
        .dstAccessMask    = dstAccess,
        .oldLayout        = oldLayout,
        .newLayout        = newLayout,
        .image            = m_imageDedicated.image,
        .subresourceRange = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = baseMipLevel,
            .levelCount     = levelCount,
            .baseArrayLayer = 0,
            .layerCount     = 1}};
    vkCmdPipelineBarrier(
        cmdBuf, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  }

  // Set the staging buffer size to be as needed to store an image
  // of the given base dimensions and all its mipmap levels.
  void resizeStaging(uint32_t width, uint32_t height)
  {
    bool needReallocate = m_pCpuMipmap == nullptr;
    if(!needReallocate)
    {
      const auto& baseDim = m_pCpuMipmap->getWidthHeight()[0];
      needReallocate      = width != baseDim.x || height != baseDim.y;
    }
    if(!needReallocate)
    {
      return;
    }

    // Re-allocate storage.
    destroyStagingBuffer();
    m_pCpuMipmap = std::make_unique<MipmapStorage<uint8_t, 4>>(width, height);
    const VkBufferCreateInfo stagingBufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = m_pCpuMipmap->getByteSize(),
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                 | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    m_stagingBufferDedicated = m_allocator.createBuffer(
        stagingBufferInfo, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                               | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    m_pStagingBufferMap = m_allocator.map(m_stagingBufferDedicated);
  }

  // Load named image file's contents to staging buffer.
  // Staging buffer is immediately (re)-allocated, be careful.
  // Everything in this program assumes premultiplied alpha, set
  // doPremultiplyAlpha to true if the source image does not already
  // premultiply alpha.
  //
  // This function must be safe to call by different threads, as long
  // as they are operating on different objects!
  void stageImage(const std::string& filename, bool doPremultiplyAlpha)
  {
    stageImage(filename.c_str(), doPremultiplyAlpha);
  }

  void stageImage(const char* pFilename, bool doPremultiplyAlpha)
  {
    // Load the image from file.
    FILE* file = fopen(pFilename, "rb");
    if(file == nullptr)
    {
      fprintf(
          stderr, "Could not open '%s': %s (%i)\n", pFilename, strerror(errno),
          errno);
      exit(1);
    }
    int            loadWidth{};
    int            loadHeight{};
    int            loadChannels{};
    const uint8_t* pPixels = reinterpret_cast<uint8_t*>(stbi_load_from_file(
        file, &loadWidth, &loadHeight, &loadChannels, STBI_rgb_alpha));
    if(pPixels == nullptr)
    {
      fprintf(stderr, "stbi failed to load '%s'\n", pFilename);
      exit(1);
    }

    const uint32_t width  = static_cast<uint32_t>(loadWidth);
    const uint32_t height = static_cast<uint32_t>(loadHeight);
    resizeStaging(width, height);

    // Copy pixels over to staging buffer and cpu mipmap storage, and
    // free stbi memory. If doPremultipyAlpha, have to handle the alpha
    // stuff instead of copying directly.
    const size_t byteCount = static_cast<size_t>(4) * width * height;
    if(!doPremultiplyAlpha)
    {
      memcpy(m_pStagingBufferMap, pPixels, byteCount);
    }
    else
    {
      uint32_t* pOut = static_cast<uint32_t*>(m_pStagingBufferMap);
      for(size_t i = 0; i < byteCount; i += 4)
      {
        const float alpha = pPixels[i + 3] * (1.F / 255.F);
        const float red   = linearFromSrgb(pPixels[i + 0]) * alpha;
        const float green = linearFromSrgb(pPixels[i + 1]) * alpha;
        const float blue  = linearFromSrgb(pPixels[i + 2]) * alpha;

        // Assuming little endian.
        const uint32_t packed =
            static_cast<uint32_t>(pPixels[i + 3]) << 24
            | static_cast<uint32_t>(srgbFromLinear(blue)) << 16
            | static_cast<uint32_t>(srgbFromLinear(green)) << 8
            | static_cast<uint32_t>(srgbFromLinear(red));
        *pOut++ = packed;
      }
    }
    stbi_image_free(const_cast<uint8_t*>(pPixels));
  }

  uint32_t getStagedWidth() const
  {
    if(m_pCpuMipmap == nullptr)
      return 0;
    return m_pCpuMipmap->getWidthHeight()[0].x;
  }

  uint32_t getStagedHeight() const
  {
    if(m_pCpuMipmap == nullptr)
      return 0;
    return m_pCpuMipmap->getWidthHeight()[0].y;
  }

  uint32_t getImageWidth() const { return m_imageWidth; }

  uint32_t getImageHeight() const { return m_imageHeight; }

  uint32_t getLevelCount() const { return m_imageLevels; }

  VkImage getImage() const { return m_imageDedicated.image; }

  // Re-allocate the image to the specified size, and reset the descriptors
  // to point to the new images.
  void reallocImage(uint32_t width, uint32_t height)
  {
    destroyImage();

    // Calculate number of mip levels.
    uint32_t mipLevels = 0, tmpWidth = width, tmpHeight = height;
    while(tmpWidth != 0 || tmpHeight != 0)
    {
      mipLevels++;
      tmpWidth >>= 1;
      tmpHeight >>= 1;
    }
    assert(mipLevels <= s_maxMipLevels);

    m_imageLevels = mipLevels;
    m_imageWidth  = width;
    m_imageHeight = height;

    // Create image and memory.
    auto usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                 | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                 | VK_IMAGE_USAGE_STORAGE_BIT;

    VkImageCreateInfo imageInfo = {
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        nullptr,
        VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT  // Will be sampled as integer later.
            | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT,
        VK_IMAGE_TYPE_2D,
        VK_FORMAT_R8G8B8A8_SRGB,
        {width, height, 1},
        mipLevels,
        1,  // Mip levels / array layers
        VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_OPTIMAL,
        VkImageUsageFlags(usage),
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        nullptr,
        VK_IMAGE_LAYOUT_UNDEFINED};
    m_imageDedicated = m_allocator.createImage(imageInfo);

    m_magenta.float32[0] = 1.0f;
    m_magenta.float32[1] = 0.0f;
    m_magenta.float32[2] = 1.0f;
    m_magenta.float32[3] = 1.0f;

    // Create image views.
    for(uint32_t level = 0; level < mipLevels; ++level)
    {
      VkImageViewCreateInfo viewInfo = {
          VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
          nullptr,
          0,
          m_imageDedicated.image,
          VK_IMAGE_VIEW_TYPE_2D,
          VK_FORMAT_R8G8B8A8_UINT,
          {},
          {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1}};
      NVVK_CHECK(
          vkCreateImageView(m_device, &viewInfo, nullptr, &m_uintViews[level]));
    }
    VkImageViewUsageCreateInfo sampleUsageOnly = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, nullptr,
        VK_IMAGE_USAGE_SAMPLED_BIT};  // sRGB might not support storage usage.
    VkImageViewCreateInfo viewInfo = {
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        &sampleUsageOnly,
        0,
        m_imageDedicated.image,
        VK_IMAGE_VIEW_TYPE_2D,
        VK_FORMAT_R8G8B8A8_SRGB,
        {},
        {VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1}};
    NVVK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_view));

    // Update descriptor sets.
    VkWriteDescriptorSet  write{};
    VkDescriptorImageInfo descriptorInfo = {
        VK_NULL_HANDLE, m_view, VK_IMAGE_LAYOUT_GENERAL};
    // Texture descriptor set just has the base mip level (plus
    // subsequent levels) bound; don't worry about the immutable sampler.
    write = m_textureDescriptorContainer.makeWrite(0, 0, &descriptorInfo, 0);
    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
    // Update every entry of the storage image descriptor array with
    // each mip level, use dummy data for excess mip levels (Vulkan requires
    // each descriptor to be updated if it's statically used).
    for(uint32_t i = 0; i < s_maxMipLevels; ++i)
    {
      descriptorInfo.imageView =
          m_uintViews[i < mipLevels ? i : mipLevels - 1U];
      write = m_storageDescriptorContainer.makeWrite(0, 0, &descriptorInfo, i);
      vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
    }
  }

  // Record a command to copy the base mip level of the staging buffer
  // to the base mip level of the image (all other mip levels become
  // undefined). Includes a command for transitioning ALL mip levels
  // of the image to the specified layout afterwards (also makes
  // visible to all future operations of any type on the queue).
  //
  // Image and descriptors are immediately re-allocated, be careful.
  void cmdReallocUploadImage(VkCommandBuffer cmdBuf, VkImageLayout finalLayout)
  {
    assert(m_pCpuMipmap != nullptr);  // Forgot to stage image?

    uint32_t width  = m_pCpuMipmap->getWidthHeight()[0].x;
    uint32_t height = m_pCpuMipmap->getWidthHeight()[0].y;
    reallocImage(width, height);

    // Transition to transfer dst layout.
    cmdImageBarrier(
        cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Copy it over.
    VkBufferImageCopy region = {
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset       = {0, 0, 0},
        .imageExtent       = {width, height, 1}};
    vkCmdCopyBufferToImage(
        cmdBuf, m_stagingBufferDedicated.buffer, m_imageDedicated.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Layout transition.
    cmdImageBarrier(
        cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, finalLayout);
  }

  // Record a command to download all mip levels from the Image to the
  // staging buffer. Includes needed pipeline barriers to ensure prior
  // commands' visibility and correct future host reads.
  // Staging buffer is immediately resized if needed.
  void cmdDownloadImage(VkCommandBuffer cmdBuf, VkImageLayout currentLayout)
  {
    resizeStaging(m_imageWidth, m_imageHeight);

    VkMemoryBarrier barrier = {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(
        cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    VkBufferImageCopy regions[s_maxMipLevels]{};
    uint32_t levelCount = uint32_t(m_pCpuMipmap->getWidthHeight().size());
    for(uint32_t level = 0; level < levelCount; ++level)
    {
      const auto& dim = m_pCpuMipmap->getWidthHeight()[level];
      const auto& off = m_pCpuMipmap->getLevelOffsets()[level] * s_texelSize;

      regions[level] = VkBufferImageCopy{
          .bufferOffset      = off,
          .bufferRowLength   = 0,
          .bufferImageHeight = 0,
          .imageSubresource =
              {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
               .mipLevel   = level,
               .layerCount = 1},
          .imageOffset = {0, 0, 0},
          .imageExtent = {dim.x, dim.y, 1}};
    }
    vkCmdCopyImageToBuffer(
        cmdBuf, m_imageDedicated.image, currentLayout,
        m_stagingBufferDedicated.buffer, levelCount, regions);
    VkBufferMemoryBarrier bufferBarrier = {
        .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        .buffer        = m_stagingBufferDedicated.buffer,
        .size          = m_pCpuMipmap->getByteSize()};
    vkCmdPipelineBarrier(
        cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
        0, nullptr, 1, &bufferBarrier, 0, nullptr);
  }

  // Copy and return data from staging buffer.
  std::unique_ptr<MipmapStorage<uint8_t, 4>> copyFromStaging() const
  {
    assert(m_pCpuMipmap);

    const auto& stagingWidthHeight = m_pCpuMipmap->getWidthHeight()[0];
    std::unique_ptr<MipmapStorage<uint8_t, 4>> result(
        new MipmapStorage<uint8_t, 4>(
            stagingWidthHeight.x, stagingWidthHeight.y));
    uint32_t level = 0;
    for(auto offset : m_pCpuMipmap->getLevelOffsets())
    {
      const void* srcData =
          static_cast<const char*>(m_pStagingBufferMap) + offset * 4;
      void* dstData = result->levelData(level);
      memcpy(dstData, srcData, m_pCpuMipmap->getLevelByteSize(level));
      level++;
    }
    return result;
  }

  uint8_t compareWithStaging(
      const MipmapStorage<uint8_t, 4>& mips,
      glm::uvec3*                      outCoordinate = nullptr,
      uint32_t*                        outChannel    = nullptr) const
  {
    return mips.compare(m_pStagingBufferMap, outCoordinate, outChannel);
  }

  // Idempotent
  void destroyImage()
  {
    if(m_imageDedicated.image)
    {
      m_allocator.destroy(m_imageDedicated);
    }
    m_imageDedicated = {};

    for(VkImageView& view : m_uintViews)
    {
      if(view)
        vkDestroyImageView(m_device, view, nullptr);
      view = VK_NULL_HANDLE;
    }
    vkDestroyImageView(m_device, m_view, nullptr);
    m_view = VK_NULL_HANDLE;

    m_imageWidth  = 0;
    m_imageHeight = 0;
  }

  void destroyStagingBuffer()
  {
    if(m_stagingBufferDedicated.buffer)
    {
      m_allocator.destroy(m_stagingBufferDedicated);
    }
    m_stagingBufferDedicated = {};
  }

  const VkClearColorValue* getPMagenta() const { return &m_magenta; }
};

// A sampler that is automatically destroyed.
// Uses the maximum anisotropy available on the device.
class ScopedSampler
{
  VkDevice  m_device;
  VkSampler m_sampler = VK_NULL_HANDLE;

public:
  ScopedSampler(VkDevice device, VkPhysicalDevice physicalDevice)
      : m_device(device)
  {
    // Somewhat inefficient to query all properties just for one...
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    float maxAnisotropy = 4;
    if(maxAnisotropy > props.limits.maxSamplerAnisotropy)
    {
      maxAnisotropy = props.limits.maxSamplerAnisotropy;
    }

    // Set up sampler.
    const VkSamplerCreateInfo samplerInfo{
        .sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter        = VK_FILTER_LINEAR,
        .minFilter        = VK_FILTER_LINEAR,
        .mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .anisotropyEnable = VK_TRUE,
        .maxAnisotropy    = maxAnisotropy,
        .maxLod           = VK_LOD_CLAMP_NONE};
    NVVK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &m_sampler));
  }

  ~ScopedSampler() { vkDestroySampler(m_device, m_sampler, nullptr); }

  operator VkSampler() const { return m_sampler; }
};

#endif
