// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_TIMESTAMPS_HPP_
#define NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_TIMESTAMPS_HPP_

#include <cassert>
#include <vulkan/vulkan.h>

#include "nvvk/context_vk.hpp"
#include "nvvk/error_vk.hpp"

// Simple class for managing a timestamp query pool specialized for a
// specific queue family.
class Timestamps
{
  VkDevice    m_device;
  uint32_t    m_queryCount;
  VkQueryPool m_pool;
  double      m_tickSeconds;
  uint64_t    m_timestampMask;

public:
  Timestamps(const nvvk::Context& ctx,
             uint32_t             queueFamily,
             uint32_t             queryCount)
  {
    assert(queryCount != 0);
    m_device     = ctx;
    m_queryCount = queryCount;

    VkQueryPoolCreateInfo info = {
      VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0,
      VK_QUERY_TYPE_TIMESTAMP, queryCount, 0 };
    NVVK_CHECK(vkCreateQueryPool(m_device, &info, nullptr, &m_pool));

    m_tickSeconds =
        1e-9 * ctx.m_physicalInfo.properties10.limits.timestampPeriod;

    assert(queueFamily < ctx.m_physicalInfo.queueProperties.size());
    uint32_t timestampValidBits =
        ctx.m_physicalInfo.queueProperties[queueFamily].timestampValidBits;
    assert(timestampValidBits >= 36);
    m_timestampMask = timestampValidBits >= 64 ? ~uint64_t(0)
                                               : (uint64_t(1) << timestampValidBits) - 1u;
  }

  ~Timestamps()
  {
    vkDestroyQueryPool(m_device, m_pool, nullptr);
  }

  Timestamps(Timestamps&&) = delete;

  void cmdResetQueries(VkCommandBuffer cmdBuf)
  {
    cmdResetQueries(cmdBuf, 0, m_queryCount);
  }

  void cmdResetQueries(VkCommandBuffer cmdBuf,
                       uint32_t        firstQuery,
                       uint32_t        resetQueryCount)
  {
    assert(firstQuery + resetQueryCount <= m_queryCount);
    vkCmdResetQueryPool(cmdBuf, m_pool, firstQuery, resetQueryCount);
  }

  // Record a command to write the timestamp with the given
  // index. Index must be below queryCount.
  void cmdWriteTimestamp(
      VkCommandBuffer         cmdBuf,
      uint32_t                idx,
      VkPipelineStageFlagBits stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
  {
    assert(idx < m_queryCount);
    vkCmdWriteTimestamp(cmdBuf, stage, m_pool, idx);
  }

  // Return the difference in seconds between the two timestamps with
  // the given timestamp indices.
  double subtractTimestampSeconds(uint32_t leftIdx, uint32_t rightIdx)
  {
    return m_tickSeconds
         * (m_timestampMask & (getTimestamp(leftIdx) - getTimestamp(rightIdx)));
  }

private:
  // Could be public if needed.
  uint64_t getTimestamp(uint32_t idx)
  {
    assert(idx < m_queryCount);
    uint64_t result;
    vkGetQueryPoolResults(m_device, m_pool, idx, 1, 8, &result, 8,
                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    return result & m_timestampMask;
  }
};

#endif
