cmake_minimum_required(VERSION 3.5)
set(PROJNAME vk_compute_mipmaps)
project(${PROJNAME})

#####################################################################################
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
#
if(NOT BASE_DIRECTORY)

  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../..
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/utilities.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

#####################################################################################
# Add sample packages
#
_add_package_VulkanSDK()
_add_package_ShaderC()
_add_package_ImGui()
_add_nvpro_core_lib()

#####################################################################################
# Include paths for this project
#
include_directories(AFTER .)
include_directories(AFTER ./include)
include_directories(AFTER ./nvpro_pyramid)

#####################################################################################
# Individual Programs
#
add_subdirectory(demo_app)
add_subdirectory(minimal_app)
