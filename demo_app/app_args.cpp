// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#include "app_args.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char AppArgs::inputFilenameHelpString[] =
    "-i [file] : Specify the input filename for the mipmap generator.\n"
    "If not specified, the input texture is instead a dynamically-updated\n"
    "texture.\n";

const char AppArgs::outputFilenameHelpString[] =
    "-o [file] : If specified, the generated mipmap levels are stored to\n"
    "output files of this name (with the mip level number added).\n"
    "Should specify -i as well.\n"
    "Only supports tga output, not meant as a full-feature texture exporter.\n"
    "Implicitly disables opening a window.\n";

const char AppArgs::outputPipelineAlternativeLabelHelpString[] =
    "-pipeline [name] : Pipeline alternative to be used to generate the -o output.\n";

const char AppArgs::testHelpString[] =
    "-test : If specified, compare the GPU-generated mipmaps to CPU-generated\n"
    "mipmaps; affects benchmark and -i images if any.\n";

const char AppArgs::animationTextureHelpString[] =
    "-texture [int] [int] : Specify the texture size that the state of the\n"
     "animation is drawn to.\n";

const char AppArgs::benchmarkFilenameHelpString[] =
    "-benchmark [filename] : dump json nanosecond timing info to named file.\n"
    "Implicitly disables opening a window.\n";

const char AppArgs::dumpPipelineStatsHelpString[] =
    "-stats : print static performance statistics for compute pipelines.\n";

const char AppArgs::openWindowHelpString[] =
    "-window : open a window even if implicitly disabled.\n";

void parseArgs(int argc, char** argv, AppArgs* outArgs)
{
  auto badNumber = [argv](const char* badStr)
  {
    fprintf(stderr, "%s: Expected positive integer, not '%s'\n", argv[0],
            badStr);
  };
  auto checkNeededParam = [argv](const char* arg, const char* needed)
  {
    if (needed == nullptr)
    {
      fprintf(stderr, "%s: %s missing parameter\n", argv[0], arg);
      exit(1);
    }
  };

  bool windowExplicitlyEnabled  = false;
  bool windowImplicitlyDisabled = false;

  for (int i = 1; i < argc; ++i)
  {
    const char* arg    = argv[i];
    const char* param0 = argv[i+1];
    const char* param1 = param0 == nullptr ? nullptr : argv[i+2];
    long x, y;
    char* endptr;

    if (strcmp(arg, "-h") == 0 || strcmp(arg, "/?") == 0)
    {
      printf("%s:\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s",
        argv[0],
        AppArgs::inputFilenameHelpString,
        AppArgs::outputFilenameHelpString,
        AppArgs::outputPipelineAlternativeLabelHelpString,
        AppArgs::testHelpString,
        AppArgs::animationTextureHelpString,
        AppArgs::benchmarkFilenameHelpString,
        AppArgs::dumpPipelineStatsHelpString,
        AppArgs::openWindowHelpString);
      exit(0);
    }
    else if (strcmp(arg, "-i") == 0)
    {
      checkNeededParam(arg, param0);
      outArgs->inputFilename = param0;
      ++i;
    }
    else if (strcmp(arg, "-o") == 0)
    {
      windowImplicitlyDisabled = true;
      checkNeededParam(arg, param0);
      outArgs->outputFilename = param0;
      ++i;
    }
    else if (strcmp(arg, "-pipeline") == 0)
    {
      checkNeededParam(arg, param0);
      outArgs->outputPipelineAlternativeLabel = param0;
      ++i;
    }
    else if (strcmp(arg, "-test") == 0)
    {
      outArgs->test = true;
    }
    else if (strcmp(arg, "-texture") == 0)
    {
      checkNeededParam(arg, param0);
      x = strtol(param0, &endptr, 0);
      if (*endptr != '\0' || x <= 0) badNumber(param0);

      checkNeededParam(arg, param1);
      y = strtol(param1, &endptr, 0);
      if (*endptr != '\0' || y <= 0) badNumber(param1);

      outArgs->animationTextureWidth  = uint32_t(x);
      outArgs->animationTextureHeight = uint32_t(y);
      i += 2;
    }
    else if (strcmp(arg, "-benchmark") == 0)
    {
      windowImplicitlyDisabled = true;
      checkNeededParam(arg, param0);
      outArgs->benchmarkFilename = param0;
      ++i;
    }
    else if (strcmp(arg, "-stats") == 0)
    {
      outArgs->dumpPipelineStats = true;
    }
    else if (strcmp(arg, "-window") == 0)
    {
      windowExplicitlyEnabled = true;
    }
    else
    {
      fprintf(stderr, "%s: Unknown argument '%s'\n", argv[0], arg);
      exit(1);
    }
  }

  outArgs->openWindow = !windowImplicitlyDisabled || windowExplicitlyEnabled;
}
