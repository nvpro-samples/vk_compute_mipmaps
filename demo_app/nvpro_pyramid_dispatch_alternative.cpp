// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Dictionary of dispatcher callbacks for tested alternative mipmap pipelines.

#include "nvpro_pyramid_dispatch_alternative.hpp"

#include <unordered_map>

using DispatcherMap =
    std::unordered_map<std::string, nvpro_pyramid_dispatcher_t>;

static DispatcherMap& getFastDispatcherMap()
{
  static DispatcherMap map;
  return map;
}

nvpro_pyramid_dispatcher_t getFastDispatcher(const std::string& name)
{
  const auto& fastDispatcherMap = getFastDispatcherMap();
  auto it = fastDispatcherMap.find(name);
  return it == fastDispatcherMap.end() ? nullptr : it->second;
}

NvproPyramidFastDispatcherAdder::NvproPyramidFastDispatcherAdder(
    const std::string&         name,
    nvpro_pyramid_dispatcher_t dispatcher)
{
  auto& fastDispatcherMap = getFastDispatcherMap();
  fastDispatcherMap[name] = dispatcher;
}

NVPRO_PYRAMID_ADD_FAST_DISPATCHER(default, nvproPyramidDefaultFastDispatcher)
NVPRO_PYRAMID_ADD_FAST_DISPATCHER(levels_1_5, (nvproPyramidDefaultFastDispatcher<2, 5>))
NVPRO_PYRAMID_ADD_FAST_DISPATCHER(levels_1_6, nvproPyramidDefaultFastDispatcher<2>)

static DispatcherMap& getGeneralDispatcherMap()
{
  static DispatcherMap map;
  return map;
}

nvpro_pyramid_dispatcher_t getGeneralDispatcher(const std::string& name)
{
  const auto& generalDispatcherMap = getGeneralDispatcherMap();
  auto it = generalDispatcherMap.find(name);
  return it == generalDispatcherMap.end() ? nullptr : it->second;
}

NvproPyramidGeneralDispatcherAdder::NvproPyramidGeneralDispatcherAdder(
    const std::string&         name,
    nvpro_pyramid_dispatcher_t dispatcher)
{
  auto& generalDispatcherMap = getGeneralDispatcherMap();
  generalDispatcherMap[name] = dispatcher;
}

NVPRO_PYRAMID_ADD_GENERAL_DISPATCHER(default, nvproPyramidDefaultGeneralDispatcher)

// For testing, dispatcher that does not do anything (but pretends it did).
static uint32_t nullDispatcher(VkCommandBuffer,
                               VkPipelineLayout,
                               uint32_t,
                               VkPipeline,
                               const NvproPyramidState& state)
{
  return state.remainingLevels;
}

NVPRO_PYRAMID_ADD_FAST_DISPATCHER(null, nullDispatcher)
NVPRO_PYRAMID_ADD_GENERAL_DISPATCHER(null, nullDispatcher)
