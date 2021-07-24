// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SEARCH_PATHS_HPP_
#define NVPRO_SAMPLES_VK_COMPUTE_MIPMAPS_SEARCH_PATHS_HPP_

#include <array>

#include "nvpsystem.hpp"

static const std::string installPath  = NVPSystem::exePath() + PROJECT_NAME "/";
static const std::string repoRootPath = NVPSystem::exePath() + PROJECT_RELDIRECTORY "../";

// This is used by nvh::findFile to search for shader files.
static const std::vector<std::string> searchPaths = {
  installPath + "test_images/",
  repoRootPath + "test_images/",
  installPath,
  repoRootPath,
  installPath + "spv/",
  repoRootPath + "spv/",
};

#endif
