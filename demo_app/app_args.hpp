// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef VK_COMPUTE_MIPMAPS_DEMO_APP_ARGS_HPP_
#define VK_COMPUTE_MIPMAPS_DEMO_APP_ARGS_HPP_

#include <stdint.h>
#include <string>

// Arguments for the App.
struct AppArgs
{
  std::string inputFilename = "";
  static const char inputFilenameHelpString[];

  std::string outputFilename = "";
  static const char outputFilenameHelpString[];

  std::string outputPipelineAlternativeLabel = "default";
  static const char outputPipelineAlternativeLabelHelpString[];

  bool test;
  static const char testHelpString[];

  // Size of texture that the animation is drawn to.
  uint32_t animationTextureWidth = 16384, animationTextureHeight = 16384;
  static const char animationTextureHelpString[];

  // Output filename for benchmark; if specified, run the benchmark
  // on startup.
  std::string benchmarkFilename = "";
  static const char benchmarkFilenameHelpString[];

  // Flag that enables static performance statistics for compute pipelines.
  bool dumpPipelineStats = false;
  static const char dumpPipelineStatsHelpString[];

  // Flag that forces window to be open even if implicitly disabled.
  bool openWindow = false;
  static const char openWindowHelpString[];
};

void parseArgs(int argc, char** argv, AppArgs* outArgs);

#endif
