// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef VK_COMPUTE_MIPMAPS_DEMO_PIPELINE_ALTERNATIVE_HPP_
#define VK_COMPUTE_MIPMAPS_DEMO_PIPELINE_ALTERNATIVE_HPP_

#include <stdint.h>
#include <string>

// Struct used to identify "pipeline alternatives", i.e. tested
// alternate mipmap compute algorithms.
//
// Pipeline alternatives are identified by name, used to lookup their
// dispatch callback and glsl file. Sometimes an alternative might use
// the glsl file defined by another alternative; basePipelineName is
// nonempty in that case and defines the directory of the glsl file used.
// Finally, configBits alters the macros used to configure nvproPyramidMain
//
// `name` values with special interpretation:
//
// default: don't use an alternative (use what is defined for to the lib user)
// none:    don't use a pipeline at all (only valid for fastAlternative)
// blit:    use blits instead of compute (only valid for generalAlternative)
struct PipelineAlternativeDescription
{
  std::string name             = "default";
  std::string basePipelineName = "";
  uint32_t    configBits       = 0;

  inline std::string toString() const;
};

struct PipelineAlternative
{
  const char* label;
  PipelineAlternativeDescription generalAlternative, fastAlternative;
};

namespace PipelineAlternativeDescriptionConfig
{
constexpr uint32_t srgbSharedBit = 1;
constexpr uint32_t f16SharedBit  = 2;
constexpr uint32_t noBilinearBit = 4;
};

inline std::string PipelineAlternativeDescription::toString() const
{
  using namespace PipelineAlternativeDescriptionConfig;
  std::string result = name;
  if (configBits & srgbSharedBit) result += " srgbSharedBit";
  if (configBits & f16SharedBit) result += " f16SharedBit";
  if (configBits & noBilinearBit) result += " noBilinearBit";
  return result;
}

// List of pipeline alternatives compiled into the application.
// 0th and 1st must be the nvpro_pyramid default shader and blit, respectively.
extern PipelineAlternative pipelineAlternatives[];
extern const int           pipelineAlternativeCount;
constexpr int              defaultPipelineAlternativeIdx = 0;
constexpr int              blitPipelineAlternativeIdx    = 1;

#endif /* !VK_COMPUTE_MIPMAPS_DEMO_PIPELINE_ALTERNATIVE_HPP_ */
