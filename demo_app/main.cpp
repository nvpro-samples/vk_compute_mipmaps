// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Main function of the sample.
// Mostly just initializing GLFW, instance, device, extensions,
// then parsing arguments and passing control to the App implementation.

#include <cassert>
#include <stdio.h>
#include <string>
#include <vulkan/vulkan.h>

#include "GLFW/glfw3.h"
#include "nvvk/context_vk.hpp"
#include "nvvk/error_vk.hpp"

#include "app_args.hpp"
#include "mipmaps_app.hpp"

int main(int argc, char** argv)
{
  AppArgs args;
  parseArgs(argc, argv, &args);

  // Create Vulkan glfw window unless disabled.
  GLFWwindow*  pWindow            = nullptr;
  uint32_t     glfwExtensionCount = 0;
  const char** glfwExtensions     = nullptr;
  if (args.openWindow)
  {
    glfwInit();
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    const char* pTitle = "Compute Mipmaps";
    pWindow = glfwCreateWindow(1920, 1080, pTitle, nullptr, nullptr);
    assert(pWindow != nullptr);

    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    assert(glfwExtensions != nullptr);
  }

  // Init Vulkan 1.1 device.
  nvvk::Context ctx;
  nvvk::ContextCreateInfo deviceInfo;
  deviceInfo.apiMajor = 1;
  deviceInfo.apiMinor = 1;
  for (uint32_t i = 0; i < glfwExtensionCount; ++i)
  {
    deviceInfo.addInstanceExtension(glfwExtensions[i]);
  }
#ifdef USE_DEBUG_UTILS
  deviceInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
  if (args.openWindow)
  {
    deviceInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Pipeline stats flag requires extension.
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pipelinePropertyFeatures =
      {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
  if (args.dumpPipelineStats)
  {
    deviceInfo.addDeviceExtension(
        VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME,
        false,
        &pipelinePropertyFeatures);
  }

  // Also need half floats.
  VkPhysicalDeviceShaderFloat16Int8Features shaderFloat16Features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, nullptr,
      true /* float16 */, false /* int8 */};
  deviceInfo.addDeviceExtension(
      VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
      false,
      &shaderFloat16Features);

  ctx.init(deviceInfo);
  ctx.ignoreDebugMessage(1303270965); // Bogus "general layout" perf warning.

  // Query needed subgroup properties.
  VkPhysicalDeviceSubgroupProperties subgroupProperties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
  VkPhysicalDeviceProperties2 physicalDeviceProperties = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &subgroupProperties};
  vkGetPhysicalDeviceProperties2(ctx.m_physicalDevice, &physicalDeviceProperties);

  if (subgroupProperties.subgroupSize < 16)
  {
    fprintf(stderr, "Expected subgroup size at least 16.\n");
    return 1;
  }
  else if (subgroupProperties.subgroupSize != 32)
  {
    fprintf(stderr, "\x1b[35m\x1b[1mWARNING:\x1b[0m "
                    "Only tested with subgroup size 32, not %u.\nI /expect/ "
                    "it to work anyway, notify dakeley@nvidia.com if not.\n",
                    subgroupProperties.subgroupSize);
  }
  #define NEED_BIT(var, bit) \
  if (!(var & bit)) \
  { \
    fprintf(stderr, "Needed capability: %s\n", #bit); \
    return -1; \
  }
  NEED_BIT(subgroupProperties.supportedStages, VK_SHADER_STAGE_COMPUTE_BIT);
  NEED_BIT(subgroupProperties.supportedOperations, VK_SUBGROUP_FEATURE_SHUFFLE_BIT);
  #undef NEED_BIT

  // Query needed feature for pipeline stats.
  if (args.dumpPipelineStats && !pipelinePropertyFeatures.pipelineExecutableInfo)
  {
    fprintf(stderr,
            "missing VK_KHR_pipeline_executable_properties;\n"
            "needed for -stats flag\n");
    return 1;
  }

  // Query half float feature.
  if (!shaderFloat16Features.shaderFloat16)
  {
    fprintf(stderr, "missing shaderFloat16 feature\n");
    return 1;
  }

  // Get the surface to draw to.
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  if (args.openWindow)
  {
    NVVK_CHECK(glfwCreateWindowSurface(
      ctx.m_instance, pWindow, nullptr, &surface));
  }
  else
  {
    fprintf(stderr, "Window implicitly disabled.\n");
  }

  // Start the main loop.
  mipmapsApp(ctx, pWindow, surface, args);

  // At this point, FrameManager's destructor in mainLoop ensures all
  // pending commands are complete. So, we can clean up the surface,
  // Vulkan device, and glfw.
  if (args.openWindow)
  {
    vkDestroySurfaceKHR(ctx.m_instance, surface, nullptr);
    glfwTerminate();
  }
  ctx.deinit();
}
