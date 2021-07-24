// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#include "pipeline_alternative.hpp"

// List of pipeline alternatives to compile into the application.

using namespace PipelineAlternativeDescriptionConfig;

PipelineAlternative pipelineAlternatives[] = {
    {"default", {}, {}},             // See defaultPipelineAlternativeIdx
    {"blit", {"blit"}, {"none"}},    // See blitPipelineAlternativeIdx

#if PIPELINE_ALTERNATIVES
#if PIPELINE_ALTERNATIVES != 2
/* Most relevant alternative algorithms */
    {"generalblit", {"blit"}, {"default"}},
    {"onelevel", {"onelevel"}, {"onelevel"}},
    {"generalonly", {}, {"none"}},
    {"levels_1_3", {}, {"noshared"}},
    {"levels_3_3", {}, {"fixed3levels"}},
    {"levels_1_5", {}, {"levels_1_5", "default"}},
    {"levels_1_6", {}, {"levels_1_6", "default"}},
    {"workgroup1024", {}, {"workgroup1024", "workgroup1024"}},

/* Testing alternative configuration macros e.g. no hardware samplers */
    {"srgbShared",
     {"default", "", srgbSharedBit},
     {"default", "", srgbSharedBit}},
    {"srgbSharedGeneral", {"default", "", srgbSharedBit}, {}},
    {"f16Shared", {"default", "", f16SharedBit}, {"default", "", f16SharedBit}},
    {"f16SharedGeneral", {"default", "", f16SharedBit}, {}},
    {"noBilinear", {}, {"default", "", noBilinearBit}},
#endif

#if PIPELINE_ALTERNATIVES >= 3
/* Testing more alternative general pipelines. */
    {"generalonelevel", {"onelevel"}, {"default"}},
    {"general3level", {"general3level"}, {}},
    {"general2", {"general2", "general2"}, {}},
    {"general2max1", {"general2max1", "general2"}, {}},
    {"general2s", {"general2s"}, {}},
    {"general2smax1", {"general2smax1", "general2s"}, {}},

/* Testing more alternative fast pipelines. */
    {"quad3", {}, {"quad3"}},
    {"quad4", {}, {"quad4"}},
    {"quadflex", {}, {"quadflex"}},
    {"quadflexmin1", {}, {"quadflexmin1", "quadflex"}},
    {"quadshared", {}, {"quadshared"}},
    {"quadsharedmin1", {}, {"quadsharedmin1", "quadshared"}},
    {"maybequad", {}, {"maybequad"}},
    {"maybequadmin1", {}, {"maybequadmin1", "maybequad"}},

/* Testing a lot of different workgroup and tile sizes. */
     #include "py2_pipeline_alternatives.inc"

/* Incorrect pipelines, for testing purposes */
    {"null", {"null", "default"}, {"null", "default"}},
    {"generalnull", {"null", "default"}},
#endif
#if PIPELINE_ALTERNATIVES != 1
    {"baseline", {"baseline"}, {"none"}},
#endif
#endif  /* PIPELINE_ALTERNATIVES */
};

const int pipelineAlternativeCount =
    sizeof(pipelineAlternatives) / sizeof(pipelineAlternatives[0]);
