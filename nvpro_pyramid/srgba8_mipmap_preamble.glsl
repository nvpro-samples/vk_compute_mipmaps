// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Defines the pipeline interface and macros for nvproPyramidMain for
// the srgba8-mipmap-generation shader; EXCEPT that
// NVPRO_PYRAMID_IS_FAST_PIPELINE is not defined.

// ************************************************************************
// Input: Entire sRGBA8 texture with bilinear filtering.
layout(set=0, binding=0) uniform sampler2D srgbTex;
// Output: Same texture, imageMipLevels[n] refers to mip level n.
//         Requires manual linear (vec4) color to sRGBA8 conversion.
layout(set=1, binding=0, rgba8ui) uniform writeonly uimage2D imageMipLevels[16];

// ************************************************************************
// Mandatory macros, except NVPRO_PYRAMID_IS_FAST_PIPELINE
uvec4 srgbFromLinearVec(vec4 arg);

#define NVPRO_PYRAMID_TYPE vec4

#define NVPRO_PYRAMID_LOAD(coord, level, out_) \
  out_ = texelFetch(srgbTex, coord, level)

#define NVPRO_PYRAMID_REDUCE(a0, v0, a1, v1, a2, v2, out_) \
   out_ = a0 * v0 + a1 * v1 + a2 * v2

#define NVPRO_PYRAMID_STORE(coord, level, in_) \
  imageStore(imageMipLevels[level], coord, srgbFromLinearVec(in_))

ivec2 levelSize(int level) { return imageSize(imageMipLevels[level]); }
#define NVPRO_PYRAMID_LEVEL_SIZE levelSize

// ************************************************************************
// Optional macros (including recommended NVPRO_PYRAMID_LOAD_REDUCE4)
#define NVPRO_PYRAMID_REDUCE2(v0, v1, out_) out_ = 0.5 * (v0 + v1)

#define NVPRO_PYRAMID_REDUCE4(v00, v01, v10, v11, out_) \
  out_ = 0.25 * ((v00 + v01) + (v10 + v11))

#if !defined(USE_BILINEAR_SAMPLING) || USE_BILINEAR_SAMPLING
  void loadReduce4(in ivec2 srcTexelCoord, in int srcLevel, out vec4 out_)
  {
    // Construct the normalized coordinate in the exact center of the 4
    // texels we want and use it to sample the input texture.
    // +--+--+ -- srcTexelCoord.y / imageSize.y
    // |  |  |
    // +--X <--- normCoord
    // |  |  |
    // +--+--+ -- (srcTexelCoord.y + 2) / imageSize.y
    // |     |
    // |   (srcTexelCoord.x + 2) / imageSize.x
    // srcTexelCoord.x / imageSize.x
    vec2 normCoord = (vec2(srcTexelCoord) + vec2(1))
                   / vec2(imageSize(imageMipLevels[srcLevel]));
    out_ = textureLod(srgbTex, normCoord, srcLevel);
  }
  #define NVPRO_PYRAMID_LOAD_REDUCE4 loadReduce4
#endif

#if defined(SRGB_SHARED) && SRGB_SHARED
  #define RED_SHIFT 0
  #define GREEN_SHIFT 8
  #define BLUE_SHIFT 16
  #define ALPHA_SHIFT 24
  #define NVPRO_PYRAMID_SHARED_TYPE uint

  // Convert float (0-1) sRGB red/green/blue component value to linear.
  float linearFromSrgbComponent(float srgb)
  {
    return srgb <= 0.04045 ? srgb * (25 / 323.)
                           : pow((200 * srgb + 11) * (1/211.), 2.4);
  }

  // Convert linear red/green/blue component to float (0-1) sRGB
  float srgbComponentFromLinear(float linear)
  {
    return linear <= 0.0031308 ? (323 / 25.) * linear
                               : 1.055 * pow(linear, 1 / 2.4) - 0.055;
  }

  void srgbUnpack(in uint srgbPacked, out vec4 linearColor)
  {
    vec4 srgba    = unpackUnorm4x8(srgbPacked);
    linearColor.r = linearFromSrgbComponent(srgba.r);
    linearColor.g = linearFromSrgbComponent(srgba.g);
    linearColor.b = linearFromSrgbComponent(srgba.b);
    linearColor.a = srgba.a;
  }
  #define NVPRO_PYRAMID_SHARED_LOAD srgbUnpack

  void srgbPack(out uint srgbPacked, in vec4 linearColor)
  {
    vec4 srgba;
    srgba.r    = srgbComponentFromLinear(linearColor.r);
    srgba.g    = srgbComponentFromLinear(linearColor.g);
    srgba.b    = srgbComponentFromLinear(linearColor.b);
    srgba.a    = linearColor.a;
    srgbPacked = packUnorm4x8(srgba);
  }
  #define NVPRO_PYRAMID_SHARED_STORE srgbPack
#endif

#if defined(F16_SHARED) && F16_SHARED
  // Requires GL_EXT_shader_explicit_arithmetic_types
  #define NVPRO_PYRAMID_SHARED_TYPE f16vec4
  #define NVPRO_PYRAMID_SHARED_LOAD(smem_, out_) out_ = vec4(smem_)
  #define NVPRO_PYRAMID_SHARED_STORE(smem_, in_) smem_ = f16vec4(in_)
#endif

uint srgbFromLinearBias(float arg, float bias)
{
  float srgb = arg <= 0.0031308 ? (323/25.) * arg
                                : 1.055 * pow(arg, 1/2.4) - 0.055;
  return uint(clamp(srgb * 255. + bias, 0, 255));
}

// Convert float linear red/green/blue value to 8-bit sRGB component.
uint srgbFromLinear(float arg)
{
  return srgbFromLinearBias(arg, 0.5);
}

uvec4 srgbFromLinearVec(vec4 arg)
{
  uint alpha = clamp(uint(arg.a * 255.0f + 0.5f), 0u, 255u);
  return uvec4(srgbFromLinear(arg.r), srgbFromLinear(arg.g),
               srgbFromLinear(arg.b), alpha);
}
