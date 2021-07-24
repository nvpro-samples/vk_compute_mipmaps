// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef VK_COMPUTE_MIPMAPS_MIPMAPS_APP_HPP_
#define VK_COMPUTE_MIPMAPS_MIPMAPS_APP_HPP_

#include <vulkan/vulkan.h>

namespace nvvk {
class Context;
}

struct GLFWwindow;
struct AppArgs;

void mipmapsApp(nvvk::Context& context,
                GLFWwindow*    window,
                VkSurfaceKHR   surface,
                const AppArgs& args);

#endif
