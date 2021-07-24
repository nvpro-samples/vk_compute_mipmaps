// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

// Paint the whole screen with the given base color texture.
#version 460
#extension GL_GOOGLE_include_directive : enable

#include "camera_transforms.h"
#include "filter_modes.h"
#include "scene_modes.h"
#include "swap_image_push_constant.h"
#include "srgb.h"

layout (push_constant) uniform PushConstantBlock
{
  SwapImagePushConstant pushConstant;
};

layout(set=0, binding=0) uniform sampler2D baseColorTex;
layout(set=1, binding=0) uniform CameraTransformsBuffer
{
  CameraTransforms cameraTransforms;
};

layout(location=0) in  vec2 normalizedPixel;
layout(location=0) out vec4 color;

vec4 sampleTexture()
{
  vec2 baseTexelCoord =
      gl_FragCoord.xy * pushConstant.texelScale + pushConstant.texelOffset;
  if (pushConstant.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_SHOW_ALL_MIPS)
  {
    // Show all mip levels side-by-side.
    float x = baseTexelCoord.x;
    for (int lod = 0; ; ++lod)
    {
      vec2 levelSize = textureSize(baseColorTex, lod);
      if (x < levelSize.x)
      {
        if (baseTexelCoord.y < 0 || baseTexelCoord.y >= levelSize.y)
          return vec4(0);
        else
          return texelFetch(baseColorTex, ivec2(x, baseTexelCoord.y), lod);
      }
      x -= levelSize.x;
      if (levelSize.x <= 1 && levelSize.y <= 1)
      {
        return vec4(0);
      }
    }
  }

  vec2  texSize = textureSize(baseColorTex, 0);
  vec2  normCoord;
  vec4  sampledColor;
  float opacityScale = 1.0;

  if (pushConstant.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_3D)
  {
    // Imagine the texture to be on the y=0 plane.
    // Convert NDC to ray and sample the intersection point, if any.
    // Fade the texture near the "horizon" to avoid artifacts from
    // fragment derivatives.
    vec3 rayOrigin, rayNormalizedDirection;
    mat4 invV              = cameraTransforms.viewInverse,
         invP              = cameraTransforms.projInverse;
    rayOrigin              = (invV * vec4(0, 0, 0, 1)).xyz;
    vec3 target            = (invP * vec4(normalizedPixel, 1, 1)).xyz;
    rayNormalizedDirection = normalize((invV * vec4(normalize(target), 0)).xyz);

    if (rayNormalizedDirection.y == 0)
    {
      return vec4(0);
    }
    else
    {
      float t        = -rayOrigin.y / rayNormalizedDirection.y;
      baseTexelCoord = rayOrigin.xz + t * rayNormalizedDirection.xz;
      if (t < 0)
      {
        return vec4(0);
      }
      // Fade texture near horizon.
      opacityScale = clamp(abs(rayNormalizedDirection.y) * 32, 0, 1);
    }
  }
  normCoord = baseTexelCoord / texSize;

  switch (pushConstant.filterMode)
  {
    case VK_COMPUTE_MIPMAPS_FILTER_MODE_TRILINEAR: default:
      sampledColor = texture(baseColorTex, normCoord);
      break;
    case VK_COMPUTE_MIPMAPS_FILTER_MODE_TRILINEAR_EXPLICIT_LOD:
      sampledColor = textureLod(baseColorTex, normCoord, pushConstant.explicitLod);
      break;
    case VK_COMPUTE_MIPMAPS_FILTER_MODE_NEAREST_EXPLICIT_LOD:
    {
      int  lod        = int(round(pushConstant.explicitLod));
      vec2 texelCoord = fract(normCoord) * textureSize(baseColorTex, lod);
      sampledColor    = texelFetch(baseColorTex, ivec2(texelCoord), lod);
      break;
    }
  }

  if (pushConstant.sceneMode == VK_COMPUTE_MIPMAPS_SCENE_MODE_2D_NOT_TILED)
  {
    if (clamp(normCoord, vec2(0), vec2(1)) != normCoord)
    {
      return vec4(0);
    }
  }
  return sampledColor * opacityScale;  // Note premultiplied alpha
}

// Return a typical "transparent" checkerboard pattern.
vec3 getBackgroundColor()
{
  uvec2 tileCoord = uvec2(gl_FragCoord.xy) / 32;
  bool  parity    = (tileCoord.x + tileCoord.y) % 2 == 1;
  return vec3(parity ? 0.875 : 1.125) * pushConstant.backgroundBrightness;
}

void main()
{
  vec4 loadedColor = sampleTexture();
  vec3 background  = getBackgroundColor();

  // Assuming premultiplied alpha.
  color = vec4(background * (1 - loadedColor.a) + loadedColor.rgb, 1);
}
