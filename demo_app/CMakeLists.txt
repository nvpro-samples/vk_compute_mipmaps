#####################################################################################
# Mipmap Usage Demo Executable
#
SET(PROJNAME vk_compute_mipmaps_demo)
add_executable(${PROJNAME})
_add_project_definitions(${PROJNAME})

#####################################################################################
# User Project Configuration Options
#
#
set(PIPELINE_ALTERNATIVES 0 CACHE STRING "Whether to compile in alternative mipmap shaders; 0:no, 1:some, 2:baseline, 3:all ")
option(USE_DEBUG_UTILS "Whether to include Vulkan debug markers" OFF)
if(USE_DEBUG_UTILS)
  add_definitions(-DUSE_DEBUG_UTILS)
endif(USE_DEBUG_UTILS)
add_definitions(-DPIPELINE_ALTERNATIVES=${PIPELINE_ALTERNATIVES})

#####################################################################################
# Source files for this project
#
file(GLOB TOP_LEVEL_SOURCE_FILES *.c *.cpp)
if(PIPELINE_ALTERNATIVES)
  file(GLOB_RECURSE FAST_DISPATCHER_SOURCE_FILES ../extras/fast_pipelines/*.cpp)
  file(GLOB_RECURSE GENERAL_DISPATCHER_SOURCE_FILES ../extras/general_pipelines/*.cpp)
  list(APPEND SOURCE_FILES "${TOP_LEVEL_SOURCE_FILES}" "${FAST_DISPATCHER_SOURCE_FILES}" "${GENERAL_DISPATCHER_SOURCE_FILES}")
else()
  set(SOURCE_FILES "${TOP_LEVEL_SOURCE_FILES}")
endif(PIPELINE_ALTERNATIVES)
file(GLOB HEADER_FILES *.h *.hpp ../include/*.h ../include/*.hpp)
file(GLOB SHADER_FILES ../shaders/*.comp ../shaders/*.vert ../shaders/*.frag ../nvpro_pyramid/*.comp)
file(GLOB NVPRO_PYRAMID_LIBRARY_FILES ../nvpro_pyramid/*.glsl ../nvpro_pyramid/*.hpp)  # Skip .comp

target_sources(${PROJNAME} PUBLIC ${SOURCE_FILES} ${HEADER_FILES} ${SHADER_FILES} ${NVPRO_PYRAMID_LIBRARY_FILES})

#####################################################################################
# Pre-compiled shaders for this project -> sets SPV_OUTPUT variable
#
compile_glsl(
    SOURCE_FILES ${SHADER_FILES}
    DST ${CMAKE_CURRENT_SOURCE_DIR}/../spv)

#####################################################################################
# Linkage
#
target_link_libraries(${PROJNAME} ${PLATFORM_LIBRARIES} nvpro_core)
foreach(DEBUGLIB ${LIBRARIES_DEBUG})
  target_link_libraries(${PROJNAME} debug ${DEBUGLIB})
endforeach(DEBUGLIB)

foreach(RELEASELIB ${LIBRARIES_OPTIMIZED})
  target_link_libraries(${PROJNAME} optimized ${RELEASELIB})
endforeach(RELEASELIB)

#####################################################################################
# Source code groups for Visual Studio (I don't really use that so contact me if there's a mistake)
#
source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Header Files" FILES ${HEADER_FILES})
source_group("Shader Files" FILES ${SHADER_FILES})
if(PIPELINE_ALTERNATIVES)
  source_group("Fast Pipeline Dispatchers" FILES ${FAST_DISPATCHER_SOURCE_FILES})
  source_group("General Pipeline Dispatchers" FILES ${GENERAL_DISPATCHER_SOURCE_FILES})
endif()
source_group("nvpro_pyramid library" FILES ${NVPRO_PYRAMID_LIBRARY_FILES})

#####################################################################################
# copies binaries that need to be put next to the exe files (shaders, etc.)
#
_finalize_target(${PROJNAME})
install(FILES ${SPV_OUTPUT} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}/spv")
install(FILES ${SPV_OUTPUT} CONFIGURATIONS Debug   DESTINATION "bin_${ARCH}_debug/${PROJNAME}/spv")
install(DIRECTORY ../extras CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}" FILES_MATCHING PATTERN *.glsl)
install(DIRECTORY ../extras CONFIGURATIONS Debug   DESTINATION "bin_${ARCH}_debug/${PROJNAME}" FILES_MATCHING PATTERN *.glsl)
install(DIRECTORY ../nvpro_pyramid CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}")
install(DIRECTORY ../nvpro_pyramid CONFIGURATIONS Debug   DESTINATION "bin_${ARCH}_debug/${PROJNAME}")
install(DIRECTORY ../test_images CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}")
install(DIRECTORY ../test_images CONFIGURATIONS Debug   DESTINATION "bin_${ARCH}_debug/${PROJNAME}")
