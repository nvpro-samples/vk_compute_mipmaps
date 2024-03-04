#include "gui.hpp"

#include <math.h>

#include "nvh/cameramanipulator.hpp"
#include "nvh/container_utils.hpp"
#include <glm/glm.hpp>

#include "frame_manager.hpp"
#include "image_names.hpp"
#include "pipeline_alternative.hpp"

#include "shaders/filter_modes.h"
#include "shaders/scene_modes.h"

void Gui::cmdInit(VkCommandBuffer cmdBuf, GLFWwindow* pWindow, const nvvk::Context& ctx, const FrameManager& frameManager, VkRenderPass renderPass, uint32_t subpass)
{
  m_pWindow = pWindow;
  m_device  = ctx;
  assert(m_device != nullptr);

  void* oldUserPointer = glfwGetWindowUserPointer(pWindow);
  assert(oldUserPointer == nullptr);
  glfwSetWindowUserPointer(pWindow, this);  // Class must be non-moveable
  addCallbacks(pWindow);

  m_cameraManipulator.setLookat({5000, 5000, 5000}, {0, 0, 0}, {0, 1, 0});

  // Special understanding of first two options is hard coded later; careful.
  m_imageMenuOptions.push_back("Select Drawn Image");
  m_imageMenuOptions.push_back("Dynamically-Generated");
  for(const char* pImageName : imageNameArray)
  {
    m_imageMenuOptions.push_back(pImageName);
  }

  m_guiContext = ImGui::CreateContext(nullptr);
  assert(m_guiContext != nullptr);
  ImGui::SetCurrentContext(m_guiContext);

  ImGuiH::Init(1920, 1080, nullptr, ImGuiH::FONT_PROPORTIONAL_SCALED);
  ImGuiH::setFonts(ImGuiH::FONT_PROPORTIONAL_SCALED);
  ImGuiH::setStyle(true);

  VkDescriptorPoolSize       poolSizes[] = {VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1},
                                            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                                            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}};
  VkDescriptorPoolCreateInfo poolInfo    = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                            nullptr,
                                            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                            arraySize(poolSizes),
                                            arraySize(poolSizes),
                                            poolSizes};
  assert(m_pool == VK_NULL_HANDLE);
  NVVK_CHECK(vkCreateDescriptorPool(ctx, &poolInfo, nullptr, &m_pool));

  ImGui_ImplVulkan_InitInfo info{};
  info.Instance            = ctx.m_instance;
  info.PhysicalDevice      = ctx.m_physicalDevice;
  info.Device              = ctx.m_device;
  info.QueueFamily         = frameManager.getQueueFamilyIndex();
  info.Queue               = frameManager.getQueue();
  info.DescriptorPool      = m_pool;
  info.RenderPass          = renderPass;
  info.Subpass             = subpass;
  info.MinImageCount       = frameManager.getSwapChain().getImageCount();
  info.ImageCount          = frameManager.getSwapChain().getImageCount();
  info.MSAASamples         = VK_SAMPLE_COUNT_1_BIT;
  info.UseDynamicRendering = false;
  info.Allocator           = nullptr;
  info.CheckVkResultFn     = [](VkResult err) { NVVK_CHECK(err); };

  ImGui_ImplVulkan_Init(&info);
  ImGui_ImplVulkan_CreateFontsTexture();

  ImGui_ImplGlfw_InitForVulkan(pWindow, false);
}

Gui::~Gui()
{
  if(m_device != nullptr)
  {
    vkDestroyDescriptorPool(m_device, m_pool, nullptr);
    ImGui_ImplVulkan_DestroyFontsTexture();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
  }
  if(m_pWindow != nullptr)
  {
    glfwSetWindowUserPointer(m_pWindow, nullptr);
  }
}

void Gui::doFrame(nvvk::ProfilerVK& vkProfiler)
{
  updateFpsSample();
  updateCamera();
  ImGui::NewFrame();
  ImGui_ImplGlfw_NewFrame();
  float dpiScale = float(ImGuiH::getDPIScale());

  if(m_guiVisible)
  {
    if(m_firstTime)
    {
      ImGui::SetNextWindowPos({0, 0});
      ImGui::SetNextWindowSize({dpiScale * 300, dpiScale * 800});
      ImGui::SetNextItemOpen(true);
    }
    ImGui::Begin("Toggle UI [u]");
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);

    if(ImGui::CollapsingHeader("Input/Output"))
    {
      doInputOutputControls();
    }
    if(ImGui::CollapsingHeader("Mipmap Generation"))
    {
      doMipmapGenerationControls(vkProfiler);
    }
    if(ImGui::CollapsingHeader("Visualization"))
    {
      doVisualizationControls();
    }
    if(ImGui::CollapsingHeader("Frame Performance"))
    {
      doFramePerformanceControls(vkProfiler);
    }
    if(ImGui::CollapsingHeader("Tools"))
    {
      doToolsControls();
    }
    ImGui::End();
  }
  ImGui::Render();

  m_firstTime = false;
}

void Gui::updateCamera()
{
  int x, y;
  glfwGetWindowSize(m_pWindow, &x, &y);
  m_cameraManipulator.setWindowSize(x, y);
  m_cameraManipulator.updateAnim();
  m_cam.camera = m_cameraManipulator.getCamera();
}

void Gui::doInputOutputControls()
{
  int showImageIdx = 0;
  ImGui::Combo(" " /* breaks if empty string */, &showImageIdx, m_imageMenuOptions.data(), int(m_imageMenuOptions.size()));
  if(showImageIdx == 1)
  {
    m_wantLoadImageFilename = "";
  }
  else if(showImageIdx != 0)
  {
    m_wantLoadImageFilename = imageNameArray[showImageIdx - 2];
  }

  ImGui::SameLine();
  if(ImGui::Button("Open File [o]"))
  {
    doOpenImageFileDialog();
  }

  if(ImGui::Button("Write Generated Mipmaps [w]"))
  {
    doSaveImageFileDialog();
  }

  if(m_drawingDynamicImage)
  {
    ImGui::SliderInt("width", &m_imageWidth, 1, 32768);
    ImGui::SliderInt("height", &m_imageHeight, 1, 32768);
  }
  else
  {
    ImGui::Text("Image Width: %i", m_imageWidth);
    ImGui::Text("Image Height: %i", m_imageHeight);
  }
}

void Gui::doMipmapGenerationControls(nvvk::ProfilerVK& vkProfiler)
{
  if(m_showAllPipelineAlternatives || m_alternativeIdxSetting >= 2)
  {
    std::vector<const char*> labels;
    for(size_t i = 0; i < pipelineAlternativeCount; ++i)
    {
      labels.push_back(pipelineAlternatives[i].label);
    }
    ImGui::Combo("##fixesSurprisingImGuiDesign", &m_alternativeIdxSetting, labels.data(), int(labels.size()));
    m_showAllPipelineAlternatives = true;
  }
  else
  {
    ImGui::RadioButton("nvpro_pyramid", &m_alternativeIdxSetting, defaultPipelineAlternativeIdx);
    ImGui::SameLine();
    ImGui::RadioButton("blit", &m_alternativeIdxSetting, blitPipelineAlternativeIdx);
    ImGui::SameLine();
    int showMore = 0;
    ImGui::RadioButton("more...", &showMore, 1);
    if(PIPELINE_ALTERNATIVES && showMore)
    {
      m_showAllPipelineAlternatives = true;
    }
    if(!PIPELINE_ALTERNATIVES && ImGui::IsItemHovered())
      ImGui::SetTooltip("Note: Re-run CMake with -DPIPELINE_ALTERNATIVES=1");
  }
  showCpuGpuTime(vkProfiler, "mipmaps", "Mipmap Generation");
  auto&                    points = m_mipmapGpuRuntimeHistory;
  nvh::Profiler::TimerInfo timerInfo{};
  vkProfiler.getTimerInfo("mipmaps", timerInfo);
  float gpu_ms = float(timerInfo.gpu.average * 0.001);
  points.push_back(gpu_ms);
  if(points.size() > 256)
    points.erase(points.begin());  // Forget oldest sample
  ImGui::PlotLines("GPU Time History", points.data(), int(points.size()));
}

void Gui::doVisualizationControls()
{
  // Filter Mode and LoD controls.
  // These have no effect in "show all mips" scene mode, so to avoid confusion,
  // implicitly exit that mode if these controls are used.
  if(m_cam.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_SHOW_ALL_MIPS)
  {
    int filterMode = VK_COMPUTE_MIPMAPS_FILTER_MODE_COUNT;
    ImGui::Combo("Filter Mode [f]", &filterMode, filterModeLabels, VK_COMPUTE_MIPMAPS_FILTER_MODE_COUNT + 1);
    if(filterMode != VK_COMPUTE_MIPMAPS_FILTER_MODE_COUNT)
    {
      m_cam.filterMode = filterMode;
      m_cam.sceneMode  = VK_COMPUTE_MIPMAPS_SCENE_MODE_2D_NOT_TILED;
    }
  }
  else
  {
    ImGui::Combo("Filter Mode [f]", &m_cam.filterMode, filterModeLabels, VK_COMPUTE_MIPMAPS_FILTER_MODE_COUNT);
  }
  float newLod     = m_cam.explicitLod;
  float upperBound = std::max(m_maxExplicitLod, 0.0001f);
  if(m_cam.filterMode == VK_COMPUTE_MIPMAPS_FILTER_MODE_NEAREST_EXPLICIT_LOD)
  {
    int intLod = int(roundf(newLod));
    ImGui::SliderInt("Explicit LoD", &intLod, 0, int(upperBound));
    newLod = float(intLod);
  }
  else
  {
    ImGui::SliderFloat("Explicit LoD", &newLod, 0.0f, upperBound);
  }
  if(newLod != m_cam.explicitLod)
  {
    if(m_cam.filterMode == VK_COMPUTE_MIPMAPS_FILTER_MODE_TRILINEAR)
      m_cam.filterMode = VK_COMPUTE_MIPMAPS_FILTER_MODE_TRILINEAR_EXPLICIT_LOD;
    if(m_cam.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_SHOW_ALL_MIPS)
      m_cam.sceneMode = VK_COMPUTE_MIPMAPS_SCENE_MODE_2D_NOT_TILED;
  }
  m_cam.explicitLod = newLod;

  if(m_cam.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_3D)
  {
    float fov = m_cameraManipulator.getFov();
    ImGui::SliderFloat("FOV", &fov, 1, 170);
    m_cameraManipulator.setFov(fov);
  }
  else
  {
    if(ImGui::Button("Reset Position (1:1 zoom)"))
    {
      m_cam.offset = {0, 0};
      m_cam.scale  = {1, 1};
    }
    ImGui::SameLine();
    if(ImGui::Button("Fit Image"))
    {
      m_wantFitImageToScreen = true;
    }
  }
  auto oldSceneMode = m_cam.sceneMode;
  ImGui::Combo("Scene [s]", &m_cam.sceneMode, sceneModeLabels, VK_COMPUTE_MIPMAPS_SCENE_MODE_COUNT);
  auto showAllMips = VK_COMPUTE_MIPMAPS_SCENE_MODE_SHOW_ALL_MIPS;
  if(oldSceneMode != m_cam.sceneMode && m_cam.sceneMode == showAllMips)
  {
    m_wantFitImageToScreen = true;  // Requested by Pascal
  }
  if(m_drawingDynamicImage)
    ImGui::Checkbox("Animate [space]", &m_doStep);
  else
    ImGui::Text("Not showing animated image");
}

// Open a dialog box and record the image file that the user wants opened.
void Gui::doOpenImageFileDialog()
{
  m_userSelectedOpenImageFilename = NVPSystem::windowOpenFileDialog(m_pWindow, "Select Image", s_openExts);
  if(!m_userSelectedOpenImageFilename.empty())
  {
    m_wantLoadImageFilename = m_userSelectedOpenImageFilename.c_str();
  }
}

// Open a dialog box and record the image file that the user wants saved.
void Gui::doSaveImageFileDialog()
{
  m_userSelectedSaveImageFilename = NVPSystem::windowSaveFileDialog(m_pWindow, "Save Image", s_saveExts);
  if(!m_userSelectedSaveImageFilename.empty())
  {
    m_wantWriteImageBaseFilename = m_userSelectedSaveImageFilename.c_str();
  }
}

void Gui::doFramePerformanceControls(nvvk::ProfilerVK& vkProfiler)
{
  ImGui::Text("FPS: %.0f", m_displayedFPS);
  ImGui::Text("Max Frame Time: %7.4f ms", m_displayedFrameTime * 1000.);
  showCpuGpuTime(vkProfiler, "frame", "Frame");
  ImGui::Checkbox("vsync [v] (may reduce timing accuracy)", &m_vsync);
}

void Gui::doToolsControls()
{
  ImGui::Text("Note: see console for following");
  if(ImGui::Button("Start Benchmark [B]"))
  {
    m_wantBenchmark = true;
  }
  if(ImGui::Button("Test Downloaded Image [T]"))
  {
    m_wantTestDownloadedImage = true;
  }
  ImGui::Checkbox("Log Performance [G]", &m_doLogPerformance);
}

void Gui::showCpuGpuTime(nvvk::ProfilerVK& vkProfiler, const char* id, const char* label)
{
  label = label ? label : id;
  nvh::Profiler::TimerInfo timerInfo{};
  vkProfiler.getTimerInfo(id, timerInfo);
  ImGui::Text("%s", label);
  double cpu_ms = timerInfo.cpu.average * 0.001;
  double gpu_ms = timerInfo.gpu.average * 0.001;
  double max_ms = std::max(std::max(cpu_ms, gpu_ms), 0.0001);
  ImGui::Text("CPU: %.4f ms", cpu_ms);
  ImGui::SameLine();
  ImGui::ProgressBar(float(cpu_ms / max_ms), ImVec2(0.0f, 0.0f));
  ImGui::Text("GPU: %.4f ms", gpu_ms);
  ImGui::SameLine();
  ImGui::ProgressBar(float(gpu_ms / max_ms), ImVec2(0.0f, 0.0f));
}

void Gui::updateFpsSample()
{
  double now = glfwGetTime();
  if(m_lastUpdateTime == 0)
  {
    m_lastUpdateTime = now;
    return;
  }

  if(int64_t(now) != m_thisSecond)
  {
    m_displayedFPS       = m_frameCountThisSecond;
    m_displayedFrameTime = m_frameTimeThisSecond;

    m_thisSecond           = int64_t(now);
    m_frameCountThisSecond = 1;
    m_frameTimeThisSecond  = 0;
  }
  else
  {
    float frameTime = float(now - m_lastUpdateTime);
    m_frameCountThisSecond++;
    m_frameTimeThisSecond = std::max(m_frameTimeThisSecond, frameTime);
  }
  m_lastUpdateTime = now;
}

// 2d camera zoom callback. Set it up so that the mouse
// is always "in the same position" as it scrolls.
void Gui::zoomCallback2d(double dy)
{
  glm::vec2 mouseTexCoord = m_cam.offset + m_cam.scale * glm::vec2(m_zoomMouseX, m_zoomMouseY);
  float     scale         = expf(float(dy));
  m_cam.scale *= scale;

  glm::vec2 wrongTexCoord = m_cam.offset + m_cam.scale * glm::vec2(m_zoomMouseX, m_zoomMouseY);

  m_cam.offset += (mouseTexCoord - wrongTexCoord);
}

// 3d camera scroll wheel callback, moves you forwards and backwards.
void Gui::zoomCallback3d(double dy)
{
  m_cameraManipulator.wheel(int(copysign(1.0, dy)), {m_lmb, m_mmb, m_rmb, bool(m_glfwMods & GLFW_MOD_SHIFT),
                                                     bool(m_glfwMods & GLFW_MOD_CONTROL), bool(m_glfwMods & GLFW_MOD_ALT)});
}

// 2d mouse move callback, left mouse button pans, right mouse zooms
// slowly (makes aliasing issues more obvious).
void Gui::mouseMoveCallback2d(float x, float y)
{
  float dx = float(x - m_mouseX);
  float dy = float(y - m_mouseY);

  if(m_lmb)
  {
    m_cam.offset.x -= dx * m_cam.scale.x;
    m_cam.offset.y -= dy * m_cam.scale.y;
  }

  if(m_rmb)
  {
    zoomCallback2d(dy * .002);
  }
  else
  {
    m_zoomMouseX = x;
    m_zoomMouseY = y;
  }
}

// 3d mouse move callback.
void Gui::mouseMoveCallback3d(float x, float y)
{
  m_cameraManipulator.mouseMove(int(x), int(y),
                                {m_lmb, m_mmb, m_rmb, bool(m_glfwMods & GLFW_MOD_SHIFT),
                                 bool(m_glfwMods & GLFW_MOD_CONTROL), bool(m_glfwMods & GLFW_MOD_ALT)});
}

Gui& Gui::getData(GLFWwindow* pWindow)
{
  void* userPointer = glfwGetWindowUserPointer(pWindow);
  assert(userPointer != nullptr);
  Gui& data = *static_cast<Gui*>(userPointer);
  assert(data.m_magicNumber == data.s_magicNumber);
  return data;
}

void Gui::scrollCallback(GLFWwindow* pWindow, double x, double y)
{
  Gui& g = getData(pWindow);
  ImGui_ImplGlfw_ScrollCallback(pWindow, x, y);
  if(ImGui::GetIO().WantCaptureMouse)
  {
  }
  else if(g.m_cam.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_3D)
  {
    g.zoomCallback3d(y * -0.25);
  }
  else
  {
    g.zoomCallback2d(y * -0.25);
  }
}

void Gui::mouseCallback(GLFWwindow* pWindow, int button, int action, int mods)
{
  Gui& g       = getData(pWindow);
  g.m_glfwMods = mods;
  ImGui_ImplGlfw_MouseButtonCallback(pWindow, button, action, mods);
  bool mouseFlag = (action != GLFW_RELEASE) && !ImGui::GetIO().WantCaptureMouse;

  if(action == GLFW_PRESS)
  {
    g.m_cameraManipulator.setMousePosition(int(g.m_mouseX), int(g.m_mouseY));
  }

  switch(button)
  {
    case GLFW_MOUSE_BUTTON_RIGHT:
      g.m_rmb = mouseFlag;
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      g.m_mmb = mouseFlag;
      break;
    case GLFW_MOUSE_BUTTON_LEFT:
      g.m_lmb = mouseFlag;
      break;
    case 3:
      if(action == GLFW_PRESS)
        g.m_cam.explicitLod -= 1.0f;
      goto updateLodMode;
    case 4:
      if(action == GLFW_PRESS)
        g.m_cam.explicitLod += 1.0f;
      goto updateLodMode;
    default:
      break;  // Get rid of warning.
  }
  return;

updateLodMode:
  if(g.m_cam.filterMode == VK_COMPUTE_MIPMAPS_FILTER_MODE_TRILINEAR)
  {
    g.m_cam.filterMode = VK_COMPUTE_MIPMAPS_FILTER_MODE_NEAREST_EXPLICIT_LOD;
  }
  if(g.m_cam.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_SHOW_ALL_MIPS)
  {
    g.m_cam.sceneMode = VK_COMPUTE_MIPMAPS_SCENE_MODE_2D_NOT_TILED;
  }
}

void Gui::cursorPositionCallback(GLFWwindow* pWindow, double x, double y)
{
  Gui& g = getData(pWindow);
  if(g.m_cam.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_3D)
  {
    g.mouseMoveCallback3d(float(x), float(y));
  }
  else
  {
    g.mouseMoveCallback2d(float(x), float(y));
  }
  g.m_mouseX = float(x);
  g.m_mouseY = float(y);
}

void Gui::charCallbackImpl(unsigned chr)
{
  switch(chr)
  {
    case ' ':
      m_doStep = !m_doStep;
      break;
    case 'B':
      m_wantBenchmark = true;
      break;
    case 'f':
      m_cam.filterMode++;
      if(m_cam.filterMode >= VK_COMPUTE_MIPMAPS_FILTER_MODE_COUNT)
      {
        m_cam.filterMode = VK_COMPUTE_MIPMAPS_FILTER_MODE_TRILINEAR;
      }
      if(m_cam.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_SHOW_ALL_MIPS)
      {
        m_cam.sceneMode = VK_COMPUTE_MIPMAPS_SCENE_MODE_2D_NOT_TILED;
      }
      break;
    case 'g':
      m_doGaussianBlur = !m_doGaussianBlur;
      break;
    case 'G':
      m_doLogPerformance ^= 1;
      break;
    case 'k':
      if(m_cam.backgroundBrightness == 0.5f)
      {
        m_cam.backgroundBrightness = 0.01f;
      }
      else
      {
        m_cam.backgroundBrightness = 0.5f;
      }
      break;
    case 'm':
      m_mipmapsGeneratedPerFrame++;
      break;
    case 'M':
      m_mipmapsGeneratedPerFrame--;
      if(m_mipmapsGeneratedPerFrame <= 0)
        m_mipmapsGeneratedPerFrame = 1;
      break;
    case 'n':
      m_alternativeIdxSetting++;
      if(m_alternativeIdxSetting >= pipelineAlternativeCount)
      {
        m_alternativeIdxSetting = 0;
      }
      break;
    case 'o':
      Gui::doOpenImageFileDialog();
      break;
    case 'p':
      m_alternativeIdxSetting--;
      if(m_alternativeIdxSetting < 0)
      {
        m_alternativeIdxSetting = pipelineAlternativeCount - 1;
      }
      break;
    case 's':
      m_cam.sceneMode++;
      if(m_cam.sceneMode >= VK_COMPUTE_MIPMAPS_SCENE_MODE_COUNT)
      {
        m_cam.sceneMode = 0;
      }
      break;
    case 'T':
      m_wantTestDownloadedImage = true;
      break;
    case 'u':
      m_guiVisible ^= 1;
      break;
    case 'v':
      m_vsync ^= 1;
      break;
    case 'w':
      Gui::doSaveImageFileDialog();
      break;
  }
}

void Gui::charCallback(GLFWwindow* pWindow, unsigned chr)
{
  ImGui_ImplGlfw_CharCallback(pWindow, chr);
  getData(pWindow).charCallbackImpl(chr);
}

void Gui::keyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
{
  ImGui_ImplGlfw_KeyCallback(pWindow, key, scancode, action, mods);
}

void Gui::addCallbacks(GLFWwindow* pWindow)
{
  glfwSetScrollCallback(pWindow, scrollCallback);
  glfwSetMouseButtonCallback(pWindow, mouseCallback);
  glfwSetCursorPosCallback(pWindow, cursorPositionCallback);
  glfwSetCharCallback(pWindow, charCallback);
  glfwSetKeyCallback(pWindow, keyCallback);
}

const char* Gui::s_openExts = "Image Files|*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.psd;*.gif;*.hdr;*.pic";
const char* Gui::s_saveExts = "TGA files|*.TGA";
