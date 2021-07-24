// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#version 460
#extension GL_GOOGLE_include_directive : enable

// Simple vertex shader for filling the whole screen, when drawn with
// three vertices as a triangle.
const float MAX_DEPTH = 1.0;

layout(location=0) out vec2 normalizedPixel;

void main() {
  switch (gl_VertexIndex) {
    case 0: gl_Position = vec4(-1, -1, MAX_DEPTH, 1.0); break;
    case 1: gl_Position = vec4(-1, +3, MAX_DEPTH, 1.0); break;
    default:gl_Position = vec4(+3, -1, MAX_DEPTH, 1.0); break;
  }
  normalizedPixel = gl_Position.xy;
}
