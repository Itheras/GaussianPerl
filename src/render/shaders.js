// GLSL sources (M8). Pass 1: fullscreen two-layer inverse-depth raymarch over
// the LDI heightfield textures — photo-native sharpness (one bilinear tap of
// the full-res photo), structurally incapable of splat confetti, no sorting.
// Pass 2: composite over background + depth-of-field gather (unchanged
// contract: premult color + alpha-composited depth MRT).

export const RAYMARCH_VS = /* glsl */ `#version 300 es
precision highp float;
out vec2 vUv; // OUTPUT-window uv in [0,1] (interior image space)
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

// Camera model (translation-only, window-anchored — rotation is what turns
// FoV error into face shear):
//   1/Z(d) = d / dSub  (subject-anchored reciprocal, Z in subject units)
//   lateral: u_src(d) = uv + kxy * (d - dConv),  kxy = fPx*e_xy / (dSub*[W,H])
//   dolly:   u_src(d) += (uv - 0.5) * kz * (d - dConv),  kz = -e_z / dSub
// Both terms are LINEAR in candidate disparity d, so the march advances the
// sample position by a constant delta per step.
export const RAYMARCH_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 vUv;
layout(location = 0) out vec4 oColor;   // premultiplied
layout(location = 1) out vec4 oDepth;   // (depth * a, a, 0, a)

uniform sampler2D uColor0;  // padded working-res photo + outpaint ring
uniform sampler2D uDisp0;   // padded depth-res disparity (R16F)
uniform sampler2D uColor1;  // padded bg fill color (alpha = feathered mask)
uniform sampler2D uDisp1;   // padded bg fill disparity

uniform vec2 uCropScale;    // interior->padded uv: uvTex = uv*scale + off
uniform vec2 uCropOff;
uniform vec2 uFitScale;     // viewport uv -> interior image uv (contain fit)
uniform vec2 uFitOff;
uniform vec2 uKxy;          // fPx*e_xy / (dSub*[W,H])
uniform float uKz;          // -e_z / dSub
uniform float uDConv;       // convergence (pivot) disparity
uniform float uDMin;        // far disparity bound of the march
uniform float uDMax;        // near disparity bound
uniform float uDSub;        // subject disparity (depth anchor)
uniform float uDFloor;      // far clamp for 1/Z
uniform int uSteps;         // coarse march steps
uniform float uDepthEncode; // 0 => raw depth, else 1/range for 8-bit mode

const int MAX_STEPS = 96;
const int REFINE = 7;

vec2 gUvImg; // this pixel's window position in IMAGE space (v down)

vec2 srcUv(float d) {
  // sample position in INTERIOR image space for candidate disparity d
  return gUvImg + uKxy * (d - uDConv) + (gUvImg - 0.5) * (uKz * (d - uDConv));
}

vec2 toTex(vec2 uv) { return uv * uCropScale + uCropOff; }

float disp0At(vec2 uv) { return texture(uDisp0, toTex(uv)).r; }
float disp1At(vec2 uv) { return texture(uDisp1, toTex(uv)).r; }

float zOf(float d) { return uDSub / max(d, uDFloor); }

void main() {
  // viewport uv -> image uv: contain-fit, y flipped (image v grows down)
  gUvImg = vec2(vUv.x, 1.0 - vUv.y) * uFitScale + uFitOff;
  if (gUvImg.x < -0.002 || gUvImg.x > 1.002 ||
      gUvImg.y < -0.002 || gUvImg.y > 1.002) {
    oColor = vec4(0.0);
    oDepth = vec4(0.0);
    return;
  }
  float range = uDMax - uDMin;
  float stepD = range / float(uSteps);

  // ---- coarse march layer 0: near -> far ----
  float d = uDMax;
  float dPrev = uDMax;
  bool hit = false;
  for (int i = 0; i < MAX_STEPS; i++) {
    if (i >= uSteps) break;
    if (disp0At(srcUv(d)) >= d) { hit = true; break; }
    dPrev = d;
    d -= stepD;
  }
  if (!hit) { d = uDMin; } // clamp to far shell

  // ---- binary refinement between dPrev (miss) and d (hit) ----
  float lo = d, hi = dPrev;
  for (int i = 0; i < REFINE; i++) {
    float mid = 0.5 * (lo + hi);
    if (disp0At(srcUv(mid)) >= mid) lo = mid; else hi = mid;
  }
  float dHit = lo;
  vec2 uvHit = srcUv(dHit);
  float surf = disp0At(uvHit);

  // ---- gap detection: did the ray skip past a torn silhouette wall? ----
  // if the surface here is much nearer than the shell that first touched it,
  // the ray fell THROUGH a discontinuity — the stretched wall between fg and
  // bg. Take layer 1 (generated background) there instead.
  float gapThresh = 0.02 * range * (140.0 / float(uSteps));
  float gap = surf - dHit;
  vec3 rgb;
  float outD = dHit;
  if (gap > gapThresh) {
    // march layer 1 from just behind the foreground surface downward
    float d1 = dHit - stepD * 0.5;
    bool hit1 = false;
    for (int i = 0; i < MAX_STEPS; i++) {
      if (i >= uSteps) break;
      if (d1 <= uDMin) break;
      if (disp1At(srcUv(d1)) >= d1) { hit1 = true; break; }
      d1 -= stepD;
    }
    if (!hit1) d1 = uDMin;
    float lo1 = d1, hi1 = min(d1 + stepD, dHit);
    for (int i = 0; i < REFINE; i++) {
      float mid = 0.5 * (lo1 + hi1);
      if (disp1At(srcUv(mid)) >= mid) lo1 = mid; else hi1 = mid;
    }
    vec2 uv1 = srcUv(lo1);
    vec4 fill = texture(uColor1, toTex(uv1));
    // stretched layer-0 wall sample as the never-void fallback
    vec3 wall = texture(uColor0, toTex(uvHit)).rgb;
    rgb = mix(wall, fill.rgb, fill.a);
    outD = mix(dHit, lo1, fill.a);
  } else {
    rgb = texture(uColor0, toTex(uvHit)).rgb;
  }

  float z = zOf(outD);
  oColor = vec4(rgb, 1.0);
  float dEnc = (uDepthEncode > 0.0) ? clamp(z * uDepthEncode, 0.0, 1.0) : z;
  oDepth = vec4(dEnc, 1.0, 0.0, 1.0);
}
`;

export const COMPOSITE_VS = /* glsl */ `#version 300 es
precision highp float;
out vec2 vTex;
void main() {
  // fullscreen triangle
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vTex = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export const COMPOSITE_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 vTex;
out vec4 oColor;

uniform sampler2D uTexColor;   // premultiplied color
uniform sampler2D uTexDepth;   // (depth*a, a, 0, a), possibly encoded
uniform vec2 uViewport;
uniform vec3 uBgTop;
uniform vec3 uBgBottom;
uniform float uFocusDist;      // positive view depth (subject units)
uniform float uDofStrength;    // 0 disables
uniform float uMaxCoC;         // px
uniform float uDepthDecode;    // 0 => raw, else range for 8-bit mode

const int NTAPS = 16;
const vec2 POISSON[16] = vec2[16](
  vec2(0.0, 0.0),        vec2(0.54, 0.17),   vec2(-0.42, 0.31),  vec2(0.13, -0.58),
  vec2(-0.18, -0.29),    vec2(0.87, -0.27),  vec2(-0.79, -0.19), vec2(0.33, 0.68),
  vec2(-0.51, 0.74),     vec2(0.62, -0.71),  vec2(-0.93, 0.35),  vec2(0.21, 0.21),
  vec2(-0.16, 0.55),     vec2(0.44, -0.28),  vec2(-0.61, -0.56), vec2(0.95, 0.31)
);

vec3 bgAt(vec2 uv) {
  float t = clamp(uv.y, 0.0, 1.0);
  return mix(uBgBottom, uBgTop, t);
}

float decodeDepth(float d) {
  return (uDepthDecode > 0.0) ? d * uDepthDecode : d;
}

vec4 resolveAt(vec2 uv, out float depth, out float coverage) {
  vec4 c = texture(uTexColor, uv);
  vec4 dz = texture(uTexDepth, uv);
  coverage = c.a;
  depth = (dz.g > 1e-4) ? decodeDepth(dz.r / dz.g) : 1e6;
  vec3 rgb = c.rgb + bgAt(uv) * (1.0 - c.a);
  return vec4(rgb, 1.0);
}

float cocOf(float depth, float coverage) {
  float invZ = (coverage > 0.01) ? 1.0 / max(depth, 1e-3) : 0.0;
  return clamp(uDofStrength * abs(invZ - 1.0 / max(uFocusDist, 1e-3)), 0.0, uMaxCoC);
}

void main() {
  float depth, cov;
  vec4 base = resolveAt(vTex, depth, cov);
  if (uDofStrength <= 0.0) { oColor = base; return; }

  float coc = cocOf(depth, cov);

  float ang = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  float ca = cos(ang), sa = sin(ang);
  mat2 rot = mat2(ca, sa, -sa, ca);

  float gatherR = max(coc, 1.5);
  vec3 acc = base.rgb;
  float wacc = 1.0;
  for (int i = 1; i < NTAPS; i++) {
    vec2 offs = rot * POISSON[i] * gatherR;
    vec2 uv = vTex + offs / uViewport;
    float td, tc;
    vec4 tap = resolveAt(uv, td, tc);
    float tapCoC = cocOf(td, tc);
    float dist = length(offs);
    float w = clamp((max(tapCoC, coc) - dist) + 1.0, 0.0, 1.0);
    acc += tap.rgb * w;
    wacc += w;
  }
  oColor = vec4(acc / wacc, 1.0);
}
`;
