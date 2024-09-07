// Copyright 2021-2024 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

#ifndef VK_COMPUTE_MIPMAPS_DEMO_GUI_HPP_
#define VK_COMPUTE_MIPMAPS_DEMO_GUI_HPP_

#include <vulkan/vulkan_core.h>
#include "GLFW/glfw3.h"

#include <string>
#include <vector>

#include "nvh/cameramanipulator.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/profiler_vk.hpp"

#include "camera_controls.hpp"
class FrameManager;

struct ImGuiContext;

// This is the data stored behind the GLFW window's user pointer.
// Simple container for ImGui stuff, useful only for my basic needs.
// Unfortunately I couldn't initialize everything in a constructor for
// this class, you have to call cmdInit to initialize the state.
class Gui
{
  static const long        s_magicNumber = 0x697547;  // "Gui"
  long                     m_magicNumber = s_magicNumber;
  GLFWwindow*              m_pWindow{};
  VkDevice                 m_device{};
  VkDescriptorPool         m_pool{};
  ImGuiContext*            m_guiContext{};
  bool                     m_firstTime                   = true;
  bool                     m_lockAmbient                 = true;
  bool                     m_showAllPipelineAlternatives = false;
  std::vector<const char*> m_imageMenuOptions;

  // For fps counter, updated once per second.
  float   m_displayedFPS         = 0;
  float   m_displayedFrameTime   = 0;
  float   m_frameCountThisSecond = 1;
  float   m_frameTimeThisSecond  = 0;
  int64_t m_thisSecond           = 0;
  double  m_lastUpdateTime       = 0;

  // Filters for images types supported by stb_image
  static const char* s_openExts;
  static const char* s_saveExts;

  std::string m_userSelectedOpenImageFilename;
  std::string m_userSelectedSaveImageFilename;

public:
  // These are parameters set modified by the gui controls and used by
  // the App class.

  // Parameters used to build camera transforms (and other
  // appearance-related info) passed to graphics shaders.
  CameraControls m_cam;

  // Internal state of 3D camera.
  nvh::CameraManipulator m_cameraManipulator;

  std::vector<float> m_mipmapGpuRuntimeHistory;

  // Used by input callbacks.
  float m_mouseX = 0, m_mouseY = 0;
  float m_zoomMouseX = 0, m_zoomMouseY = 0;  // For centering zoom.
  bool  m_rmb = false, m_mmb = false, m_lmb = false;
  int   m_glfwMods{};

  // Index of the PipelineAlternative chosen
  int m_alternativeIdxSetting = 0;

  // Used to select and load an image file; set to empty string to view
  // animation. nullptr indicates no change wanted.
  const char* m_wantLoadImageFilename = nullptr;

  // Used to select the template filename for writing generated mip
  // levels to disk. nullptr indicates no write wanted.
  const char* m_wantWriteImageBaseFilename = nullptr;

  // Used to signal some other user inputs.
  bool m_wantTestDownloadedImage = false;
  bool m_wantBenchmark           = false;
  bool m_wantFitImageToScreen    = true;

  // Other Controls
  bool m_doStep           = true;
  bool m_vsync            = false;
  bool m_doLogPerformance = false;
  bool m_guiVisible       = true;
  bool m_doGaussianBlur   = true;

  int m_mipmapsGeneratedPerFrame = 1;

  // This does communication in the reverse direction, the graphics app
  // sets this to tell the GUI the bounds of the LoD slider.
  float m_maxExplicitLod = 1.0f;

  // The graphics app sets this to indicate whether the
  // dynamically-generated image is being shown.
  bool m_drawingDynamicImage = false;

  // Bidirectional, when m_drawingDynamicImage is true, GUI indicates
  // how big the image is to be; otherwise, app informs GUI of how big
  // the drawn static image is.
  int m_imageWidth = -1, m_imageHeight = -1;

  // Must be called once after FrameManager initialized, so that the
  // correct queue is chosen. Some initialization is done directly,
  // some by recording commands to the given command buffer.
  void cmdInit(
      VkCommandBuffer      cmdBuf,
      GLFWwindow*          pWindow,
      const nvvk::Context& ctx,
      const FrameManager&  frameManager,
      VkRenderPass         renderPass,
      uint32_t             subpass);
  Gui()
      : m_cameraManipulator(CameraManip)
  {
  }
  Gui(Gui&&) = delete;
  ~Gui();

  // Per-frame ImGui code, except for actual Vulkan draw commands.
  void doFrame(nvvk::ProfilerVK& vkProfiler);

private:
  void updateCamera();
  void doInputOutputControls();
  void doMipmapGenerationControls(nvvk::ProfilerVK& vkProfiler);
  void doVisualizationControls();
  void doOpenImageFileDialog();
  void doSaveImageFileDialog();
  void doFramePerformanceControls(nvvk::ProfilerVK& vkProfiler);
  void doToolsControls();
  void showCpuGpuTime(
      nvvk::ProfilerVK& vkProfiler,
      const char*       id,
      const char*       label = nullptr);
  void updateFpsSample();
  void zoomCallback2d(double dy);
  void zoomCallback3d(double dy);
  void mouseMoveCallback2d(float dx, float dy);
  void mouseMoveCallback3d(float dx, float dy);

  static Gui& getData(GLFWwindow* pWindow);
  static void scrollCallback(GLFWwindow* pWindow, double x, double y);
  static void mouseCallback(GLFWwindow* pWindow, int, int, int);
  static void cursorPositionCallback(GLFWwindow* pWindow, double x, double y);
  void        charCallbackImpl(unsigned chr);
  static void charCallback(GLFWwindow* pWindow, unsigned chr);
  static void keyCallback(GLFWwindow*, int, int, int, int);
  static void addCallbacks(GLFWwindow* pWindow);
};

#endif
