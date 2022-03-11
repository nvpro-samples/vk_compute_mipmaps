#####################################################################################
# Mipmap Minimal Usage Demo Executable
#
SET(PROJNAME vk_compute_mipmaps_minimal)
add_executable(${PROJNAME})
_add_project_definitions(${PROJNAME})

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.c *.cpp)
file(GLOB HEADER_FILES *.h *.hpp ../include/*.h ../include/*.hpp)
file(GLOB SHADER_FILES ../nvpro_pyramid/*.comp)

target_sources(${PROJNAME} PUBLIC ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${HEADER_FILES} ${SHADER_FILES})

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

#####################################################################################
# copies binaries that need to be put next to the exe files (shaders, etc.)
#
_finalize_target(${PROJNAME})
install(FILES ${SPV_OUTPUT} CONFIGURATIONS Release DESTINATION "bin_${ARCH}/${PROJNAME}/spv")
install(FILES ${SPV_OUTPUT} CONFIGURATIONS Debug   DESTINATION "bin_${ARCH}_debug/${PROJNAME}/spv")
