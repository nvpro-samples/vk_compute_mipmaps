// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// CPU-side storage structure for mipmap tower, and functions for
// CPU-only mipmap generation, difference comparison, and disk output
// (for testing the GPU mipmap generator).

#ifndef NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_MIPMAP_STORAGE_HPP_
#define NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_MIPMAP_STORAGE_HPP_

#include <array>
#include <cassert>
#include <stdint.h>
#include <string.h>
#include <vector>

#include "nvmath/nvmath.h"

#include "shaders/srgb.h"

template <typename T=uint8_t, uint32_t Channels=4>
class MipmapStorage
{
  using Texel = std::array<T, Channels>;

  // Data for all mip levels is in one vector.  Allocated only if
  // needed (i.e. this class can be used just to store the "layout"
  // for data that is actually stored elsewhere).
  std::vector<Texel> m_data;

  // The offset within data at which data for each mip level starts.
  // Each level is packed in [y][x] order as expected by Vulkan.  Base
  // mip level is always at offset 0, subsequent levels stored in
  // increasing order.
  std::vector<uint64_t> m_levelOffsets;

  // Width and height of each mip level.
  std::vector<nvmath::vec2ui> m_widthHeight;

  void allocateData()
  {
    if (m_data.empty())
    {
      m_data.resize(m_levelOffsets.back() + 1);
    }
  }

public:
  MipmapStorage(uint32_t width, uint32_t height)
  {
    assert(width != 0 && height != 0);
    uint64_t offset = 0;
    m_widthHeight.push_back({width, height});
    m_levelOffsets.push_back(0);

    // Calculate the number and dimensions of subsequent mip levels.
    while (width != 1 || height != 1)
    {
      // Compute offset for this level by adding the data size of the
      // previous level.
      offset += uint64_t(width) * uint64_t(height);

      // Divide by 2 rounding down, but don't go below 1.
      width  =  width >> 1 | (width  == 1u);
      height = height >> 1 | (height == 1u);

      m_widthHeight.push_back({width, height});
      m_levelOffsets.push_back(offset);
    }

    allocateData();
  }

  // Return data at (x, y, mip level)
  Texel& operator[] (nvmath::vec3ui coord)
  {
    allocateData();
    uint32_t x     = coord.x;
    uint32_t y     = coord.y;
    uint32_t level = coord.z;
    assert(level < m_levelOffsets.size());
    auto dim = m_widthHeight[level];
    assert(x < dim.x && y < dim.y);
    return m_data[m_levelOffsets[level] + dim.x*y + x];
  }

  const Texel& operator[] (nvmath::vec3ui coord) const
  {
    assert(!m_data.empty());
    return const_cast<MipmapStorage&>(*this)[coord];
  }

  // Get list of mip level width/heights.
  const std::vector<nvmath::vec2ui>& getWidthHeight() const
  {
    return m_widthHeight;
  }

  // Get list of offsets for each mip level [units = Texels, not bytes]
  const std::vector<uint64_t> getLevelOffsets() const
  {
    return m_levelOffsets;
  }

  // // Get raw data for all mip levels; interpret with getOffsets().
  // const std::vector<Texel>& getData() const
  // {
  //   return m_data;
  // }

  // Get bytes needed to store all data
  size_t getByteSize() const
  {
    return sizeof(Texel) * m_data.size();
  }

  // Return data for the given mip level; packed in [y][x] order.
  Texel* levelData(uint32_t level)
  {
    allocateData();
    assert(level < m_levelOffsets.size());
    return &m_data[m_levelOffsets[level]];
  }

  const Texel* levelData(uint32_t level) const
  {
    assert(!m_data.empty());
    assert(level < m_levelOffsets.size());
    return &m_data[m_levelOffsets[level]];
  }

  // Get bytes needed to store level.
  size_t getLevelByteSize(uint32_t level) const
  {
    assert(level < m_widthHeight.size());
    return sizeof(Texel) * m_widthHeight[level].x * m_widthHeight[level].y;
  }

  // Compare the two mipmaps (must have same size), and find the texel
  // with the greatest difference. Skip level 0. Return that
  // difference and optionally write out texel coordinate + channel at
  // which that difference was found.
  T compare(const MipmapStorage& other,
            nvmath::vec3ui*      outCoordinate=nullptr,
            uint32_t*            outChannel=nullptr) const
  {
    assert(m_widthHeight == other.m_widthHeight);
    assert(m_levelOffsets == other.m_levelOffsets);
    assert(!other.m_data.empty());
    return compare(other.m_data.data(), outCoordinate, outChannel);
  }

  // Like above, but compares to the raw data buffer given; assumed
  // to be in same layout as used in MipmapStorage.
  T compare(const void*     pBuffer,
            nvmath::vec3ui* outCoordinate = nullptr,
            uint32_t*       outChannel    = nullptr) const
  {
    assert(!m_data.empty());
    const Texel*   pOtherTexels = static_cast<const Texel*>(pBuffer);
    T              worstDelta{0};
    nvmath::vec3ui worstCoordinate{0, 0, 0};
    uint32_t       worstChannel = 0;

    for (uint32_t level = 1; level != m_levelOffsets.size(); ++level)
    {
      auto dim = m_widthHeight[level];
      const Texel*  thisLevelData = this->levelData(level);
      const Texel* otherLevelData = &pOtherTexels[thisLevelData - &m_data[0]];
      for (uint32_t y = 0; y < dim.y; ++y)
      {
        for (uint32_t x = 0; x < dim.x; ++x)
        {
          const Texel& thisTexel  = thisLevelData[dim.x*y + x];
          const Texel& otherTexel = otherLevelData[dim.x*y + x];
          for (uint32_t c = 0; c < Channels; ++c)
          {
            T a = thisTexel[c];
            T b = otherTexel[c];
            T delta = a > b ? a - b : b - a;
            if (delta > worstDelta)
            {
              worstDelta      = delta;
              worstCoordinate = {x, y, level};
              worstChannel    = c;
            }
          }
        }
      }
    }

    if (outCoordinate) *outCoordinate = worstCoordinate;
    if (outChannel)    *outChannel    = worstChannel;
    return worstDelta;
  }

  // Fill in mip levels 1+ using data from mip level 0. Provide
  // functions for converting texels (std::array<T, Channels>) to/from
  // linear color space (std::array<float, Channels>)
  template <typename ToLinear, typename FromLinear>
  void generateMipmaps(ToLinear&& toLinear, FromLinear&& fromLinear)
  {
    assert(!m_data.empty());
    for (uint32_t level = 1; level < m_levelOffsets.size(); ++level)
    {
      auto srcDim        = m_widthHeight[level - 1];
      bool srcWidthEven  = !(srcDim.x & 1);
      bool srcHeightEven = !(srcDim.y & 1);

      // Reducing a dimension of even size is fundamentally different
      // from reducing an odd size dimension. Use templates to avoid
      // excessive run-time branching in the hot loop.
      if (srcWidthEven)
      {
        if (srcHeightEven)
        {
          generateLevel<true, true>(toLinear, fromLinear, level);
        }
        else
        {
          generateLevel<true, false>(toLinear, fromLinear, level);
        }
      }
      else
      {
        if (srcHeightEven)
        {
          generateLevel<false, true>(toLinear, fromLinear, level);
        }
        else
        {
          generateLevel<false, false>(toLinear, fromLinear, level);
        }
      }
    }
  }

private:
  template <bool SrcWidthEven, bool SrcHeightEven, typename ToLinear, typename FromLinear>
  void generateLevel(ToLinear&& toLinear, FromLinear&& fromLinear, uint32_t level)
  {
    assert(level > 0 && level < m_levelOffsets.size());

    auto srcDim = m_widthHeight[level - 1];
    auto dstDim = m_widthHeight[level];
    assert(SrcWidthEven  == !(srcDim.x & 1));
    assert(SrcHeightEven == !(srcDim.y & 1));

    const Texel* pSrcLevel = levelData(level - 1);
    Texel*       pDstLevel = levelData(level);

    for (uint32_t y = 0; y < dstDim.y; ++y)
    {
      for (uint32_t x = 0; x < dstDim.x; ++x)
      {
        // A bit tricky to handle different even/odd width/height cases:
        // kernel size ranges from 2x2 to 3x3.
        using Sample = std::array<float, Channels>;
        auto loadSample = [&] (uint32_t xOffset, uint32_t yOffset, Sample& sample)
        {
          const Texel& texel = pSrcLevel[(2*x+xOffset) + srcDim.x * (2*y+yOffset)];
          sample = toLinear(texel);
        };
        auto mac = [] (Sample& lhs, const Sample& rhs, float wt)
        {
          for (uint32_t c = 0; c < Channels; ++c)
          {
            lhs[c] += rhs[c] * wt;
          }
        };

        Sample sample00, sample10, sample20,
               sample01, sample11, sample21,
               sample02, sample12, sample22; // Some may be unused.

        loadSample(0, 0, sample00);
        if (SrcWidthEven || srcDim.x != 1)
        {
          loadSample(1, 0, sample10);
        }
        if (SrcHeightEven || srcDim.y != 1)
        {
          loadSample(0, 1, sample01);
          if (SrcWidthEven || srcDim.x != 1) loadSample(1, 1, sample11);
        }

        if (!SrcWidthEven && srcDim.x != 1)
        {
          loadSample(2, 0, sample20);
          loadSample(2, 1, sample21);
          if (!SrcHeightEven && srcDim.y != 1) loadSample(2, 2, sample22);
        }
        if (!SrcHeightEven && srcDim.y != 1)
        {
          loadSample(0, 2, sample02);
          loadSample(1, 2, sample12);
        }

        // Reduce vertically.
        Sample sample0{}, sample1{}, sample2{};

        // 2 samples vertically for even source level height.
        if (SrcHeightEven)
        {
          mac(sample0, sample00, 0.5f);
          mac(sample0, sample01, 0.5f);
          mac(sample1, sample10, 0.5f);
          mac(sample1, sample11, 0.5f);
          if (!SrcWidthEven)
          {
            mac(sample2, sample20, 0.5f);
            mac(sample2, sample21, 0.5f);
          }
        }
        // 3 samples vertically for odd source level height, except when
        // the source height is 1.
        else if (srcDim.y == 1)
        {
          sample0 = sample00;
          if (SrcWidthEven || srcDim.x != 1) sample1 = sample10;
          if (!SrcWidthEven) sample2 = sample20;
        }
        else
        {
          // http://download.nvidia.com/developer/Papers/2005/NP2_Mipmapping/NP2_Mipmap_Creation.pdf
          // Page 4.
          float n   = float(dstDim.y);
          float rcp = 1.0f / (2 * n + 1);
          float w0  = rcp * (n - y);
          float w1  = rcp * n;
          float w2  = rcp * (1 + y);

          mac(sample0, sample00, w0);
          mac(sample0, sample01, w1);
          mac(sample0, sample02, w2);
          if (SrcWidthEven || srcDim.x != 1)
          {
            mac(sample1, sample10, w0);
            mac(sample1, sample11, w1);
            mac(sample1, sample12, w2);
          }
          if (!SrcWidthEven)
          {
            mac(sample2, sample20, w0);
            mac(sample2, sample21, w1);
            mac(sample2, sample22, w2);
          }
        }

        // Reduce horizontally
        Sample result{};

        if (SrcWidthEven)
        {
          mac(result, sample0, 0.5f);
          mac(result, sample1, 0.5f);
        }
        else if (srcDim.x == 1)
        {
          result = sample0;
        }
        else
        {
          float n   = float(dstDim.x);
          float rcp = 1.0f / (2 * n + 1);
          float w0  = rcp * (n - x);
          float w1  = rcp * n;
          float w2  = rcp * (1 + x);

          mac(result, sample0, w0);
          mac(result, sample1, w1);
          mac(result, sample2, w2);
        }

        // Write output texel.
        Texel& output = pDstLevel[dstDim.x*y + x];
        output = fromLinear(result);
      }
    }
  }
};

inline void cpuGenerateMipmaps_sRGBA(MipmapStorage<uint8_t, 4>* pMips)
{
  auto toLinear = [] (std::array<uint8_t, 4> texel) -> std::array<float, 4>
  {
    return { linearFromSrgb(texel[0]), linearFromSrgb(texel[1]),
             linearFromSrgb(texel[2]), texel[3] * (1.f/255.f) };
  };
  auto fromLinear = [] (std::array<float, 4> linear) -> std::array<uint8_t, 4>
  {
    uint8_t alpha = uint8_t(nvmath::nv_clamp(linear[3] * 255.f, 0.f, 255.f));
    return { uint8_t(srgbFromLinear(linear[0])),
             uint8_t(srgbFromLinear(linear[1])),
             uint8_t(srgbFromLinear(linear[2])), alpha };
  };
  auto x = pMips->getWidthHeight()[0].x;
  auto y = pMips->getWidthHeight()[0].y;
  pMips->generateMipmaps(toLinear, fromLinear);
}

// Compare contents of the given mipmap pyramid with CPU-generated mipmap.
// Return human-readable info about worst difference.
inline std::string testMipmaps(const MipmapStorage<uint8_t, 4>& input)
{
  auto x = input.getWidthHeight()[0].x;
  auto y = input.getWidthHeight()[0].y;
  MipmapStorage<uint8_t, 4> expected(x, y);
  memcpy(expected.levelData(0), input.levelData(0), input.getLevelByteSize(0));
  cpuGenerateMipmaps_sRGBA(&expected);

  nvmath::vec3ui worstCoordinate;
  uint32_t       worstChannel;
  uint8_t        worstDelta = input.compare(
      expected, &worstCoordinate, &worstChannel);

  return "Worst delta=" + std::to_string(worstDelta)
       + " at texel (" + std::to_string(worstCoordinate.x)
       + ", " + std::to_string(worstCoordinate.y)
       + "), level=" + std::to_string(worstCoordinate.z)
       + ", channel=" + std::to_string(worstChannel);
}

// Write the mip levels of the given mipmap pyramid to tga images.
// TGA names are converted as image.name.tga to
// image.name.mipLevel.tga except that level 0 is written to the base
// filename (so that file overwrite warnings work correctly).
inline void writeMipmapsTga(const MipmapStorage<uint8_t, 4>& mips,
                            const char*                      pBaseFilename)
{
  const char* pLastDot = strrchr(pBaseFilename, '.');
  std::string prefix = pLastDot ?
      std::string(pBaseFilename, pLastDot - pBaseFilename + 1)
    : std::string(pBaseFilename);
  std::string suffix = std::string(pLastDot ? pLastDot : "");

  const auto& levelOffsets = mips.getLevelOffsets();
  const auto& widthHeights = mips.getWidthHeight();
  uint32_t    levelCount   = uint32_t(widthHeights.size());

  auto threadFunction = [&](uint32_t level) {
    std::string filename = level == 0 ? std::string(pBaseFilename) :
                                        prefix + std::to_string(level) + suffix;
    auto widthHeight     = widthHeights[level];
    const void* data     = mips.levelData(level);
    stbi_write_tga(filename.c_str(), widthHeight.x, widthHeight.y, 4, data);
    fprintf(stderr, "Wrote %s\n", filename.c_str());
  };

  std::vector<std::thread> threads;
  uint32_t parallelLevelCount = levelCount > 8u ? levelCount - 8u : 0u;
  for (uint32_t level = 0; level < parallelLevelCount; ++level)
  {
    threads.emplace_back(threadFunction, level);
  }
  for (uint32_t level = parallelLevelCount; level < levelCount; ++level)
  {
    threadFunction(level);
  }
  for (std::thread& thread : threads)
  {
    thread.join();
  }
}

#endif
