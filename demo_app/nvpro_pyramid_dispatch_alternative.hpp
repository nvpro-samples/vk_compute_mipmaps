// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Dictionary of dispatcher callbacks for tested alternative mipmap pipelines.
#ifndef NVPRO_PYRAMID_DISPATCH_ALTERNATIVE_HPP_
#define NVPRO_PYRAMID_DISPATCH_ALTERNATIVE_HPP_

#include "nvpro_pyramid_dispatch.hpp"
#include <string>

nvpro_pyramid_dispatcher_t getFastDispatcher(const std::string& name);

struct NvproPyramidFastDispatcherAdder
{
  NvproPyramidFastDispatcherAdder(const std::string&,
                                  nvpro_pyramid_dispatcher_t);
};

#define NVPRO_PYRAMID_ADD_FAST_DISPATCHER(name, dispatcher) \
  static NvproPyramidFastDispatcherAdder NvproPyramidFastDispatcherAdder_##name \
  {#name, dispatcher};


nvpro_pyramid_dispatcher_t getGeneralDispatcher(const std::string& name);

struct NvproPyramidGeneralDispatcherAdder
{
  NvproPyramidGeneralDispatcherAdder(const std::string&,
                                  nvpro_pyramid_dispatcher_t);
};

#define NVPRO_PYRAMID_ADD_GENERAL_DISPATCHER(name, dispatcher) \
  static NvproPyramidGeneralDispatcherAdder NvproPyramidGeneralDispatcherAdder_##name \
  {#name, dispatcher};

#endif
