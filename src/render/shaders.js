// GLSL sources. Splat pass: instanced quads, EWA projection of 3D covariance.
// Composite pass: background + depth-of-field gather from alpha-composited depth.

// Splat data textures are 2048 texels wide; index -> texel via (idx & 2047, idx >> 11).
export const TEX_WIDTH = 2048;

export const SPLAT_VS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;

layout(location = 0) in uint aIndex;      // per-instance: sorted splat index

uniform sampler2D uTexPos;    // RGBA32F: x, y, z, (unused)
uniform sampler2D uTexCovA;   // RGBA32F: xx, xy, xz, yy
uniform sampler2D uTexCovB;   // RG32F:   yz, zz
uniform sampler2D uTexColor;  // RGBA8:   straight-alpha color

uniform mat4 uView;
uniform mat4 uProj;
uniform vec2 uFocal;          // focal length in pixels (x, y)
uniform vec2 uViewport;       // render target size in pixels
uniform float uSplatScale;    // global size multiplier (quality/artistic)

out vec4 vColor;
out vec2 vUV;                 // in sigma units, corner at +-KSIGMA
out float vDepth;             // positive view-space depth

const float KSIGMA = 2.5;

ivec2 texelOf(uint idx) {
  return ivec2(int(idx & 2047u), int(idx >> 11u));
}

void main() {
  ivec2 tc = texelOf(aIndex);
  vec3 p = texelFetch(uTexPos, tc, 0).xyz;

  vec4 cam = uView * vec4(p, 1.0);
  float s = -cam.z;                       // positive depth (camera looks -Z)
  vec4 clip = uProj * cam;
  vec3 ndc = clip.xyz / clip.w;
  // cull behind camera / far off-screen (quad extent handled by margin)
  if (s < 0.02 || abs(ndc.x) > 1.4 || abs(ndc.y) > 1.4) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vColor = vec4(0.0); vUV = vec2(0.0); vDepth = 0.0;
    return;
  }

  vec4 covA = texelFetch(uTexCovA, tc, 0);
  vec2 covB = texelFetch(uTexCovB, tc, 0).xy;
  mat3 Vrk = mat3(
    covA.x, covA.y, covA.z,
    covA.y, covA.w, covB.x,
    covA.z, covB.x, covB.y
  );

  // Jacobian of perspective projection at t = cam.xyz (s = -t.z):
  //   u = fx * t.x / s  =>  du/dt = (fx/s, 0, fx*t.x/s^2)
  //   v = fy * t.y / s  =>  dv/dt = (0, fy/s, fy*t.y/s^2)
  // GLSL mat3 constructor is column-major: columns are d/dt.x, d/dt.y, d/dt.z.
  mat3 J = mat3(
    uFocal.x / s, 0.0, 0.0,
    0.0, uFocal.y / s, 0.0,
    uFocal.x * cam.x / (s * s), uFocal.y * cam.y / (s * s), 0.0
  );
  mat3 R = mat3(uView);                   // world -> camera rotation
  mat3 T = J * R;
  mat3 cov2d = T * Vrk * transpose(T);    // sigma_screen = (JR) sigma (JR)^T

  float scale2 = uSplatScale * uSplatScale;
  float a = cov2d[0][0] * scale2 + 0.3;
  float d = cov2d[1][1] * scale2 + 0.3;
  float b = cov2d[0][1] * scale2;

  float mid = 0.5 * (a + d);
  float rad = sqrt(max(0.25 * (a - d) * (a - d) + b * b, 0.0));
  float l1 = mid + rad;
  float l2 = max(mid - rad, 0.02);
  if (l1 > 4096.0 * 4096.0) {             // degenerate: kill
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vColor = vec4(0.0); vUV = vec2(0.0); vDepth = 0.0;
    return;
  }
  vec2 dir = (abs(b) > 1e-9) ? normalize(vec2(b, l1 - a)) : ((a >= d) ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
  vec2 axis1 = min(sqrt(l1), 1024.0) * dir;
  vec2 axis2 = min(sqrt(l2), 1024.0) * vec2(-dir.y, dir.x);

  vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;
  vUV = corner * KSIGMA;
  vec2 offsetPx = (corner.x * axis1 + corner.y * axis2) * KSIGMA;

  vColor = texelFetch(uTexColor, tc, 0);
  vDepth = s;
  gl_Position = vec4(ndc.xy + offsetPx * 2.0 / uViewport, 0.0, 1.0);
}
`;

export const SPLAT_FS = /* glsl */ `#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vUV;
in float vDepth;

layout(location = 0) out vec4 oColor;   // premultiplied
layout(location = 1) out vec4 oDepth;   // (depth * a, a, 0, a)

uniform float uDepthEncode;             // 0 => raw depth, else 1/range for 8-bit mode

void main() {
  float r2 = dot(vUV, vUV);
  if (r2 > 6.25) discard;
  float alpha = vColor.a * exp(-0.5 * r2);
  if (alpha < 0.004) discard;
  oColor = vec4(vColor.rgb * alpha, alpha);
  float dEnc = (uDepthEncode > 0.0) ? clamp(vDepth * uDepthEncode, 0.0, 1.0) : vDepth;
  oDepth = vec4(dEnc * alpha, alpha, 0.0, alpha);
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

uniform sampler2D uTexColor;   // premultiplied splat color
uniform sampler2D uTexDepth;   // (depth*a, a, 0, a), possibly encoded
uniform vec2 uViewport;
uniform vec3 uBgTop;
uniform vec3 uBgBottom;
uniform float uFocusDist;      // positive view depth in world units
uniform float uDofStrength;    // 0 disables; roughly px of CoC per unit of |1/z - 1/zf|
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

// resolve premultiplied splat color over background; also outputs depth + coverage
vec4 resolveAt(vec2 uv, out float depth, out float coverage) {
  vec4 c = texture(uTexColor, uv);
  vec4 dz = texture(uTexDepth, uv);
  coverage = c.a;
  depth = (dz.g > 1e-4) ? decodeDepth(dz.r / dz.g) : 1e6;
  vec3 rgb = c.rgb + bgAt(uv) * (1.0 - c.a);
  return vec4(rgb, 1.0);
}

float cocOf(float depth, float coverage) {
  float invZ = (coverage > 0.01) ? 1.0 / max(depth, 1e-3) : 0.0;  // bg at infinity
  return clamp(uDofStrength * abs(invZ - 1.0 / max(uFocusDist, 1e-3)), 0.0, uMaxCoC);
}

void main() {
  float depth, cov;
  vec4 base = resolveAt(vTex, depth, cov);
  if (uDofStrength <= 0.0) { oColor = base; return; }

  float coc = cocOf(depth, cov);

  // interleaved gradient noise rotation to hide undersampling
  float ang = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  float ca = cos(ang), sa = sin(ang);
  mat2 rot = mat2(ca, sa, -sa, ca);

  float gatherR = max(coc, 1.5);   // gather a bit even in focus so blurred fg bleeds over
  vec3 acc = base.rgb;
  float wacc = 1.0;
  for (int i = 1; i < NTAPS; i++) {
    vec2 offs = rot * POISSON[i] * gatherR;
    vec2 uv = vTex + offs / uViewport;
    float td, tc;
    vec4 tap = resolveAt(uv, td, tc);
    float tapCoC = cocOf(td, tc);
    float dist = length(offs);
    // scatter-as-gather: a tap contributes if its own CoC reaches us,
    // or if we're blurred enough to gather the neighborhood
    float w = clamp((max(tapCoC, coc) - dist) + 1.0, 0.0, 1.0);
    acc += tap.rgb * w;
    wacc += w;
  }
  oColor = vec4(acc / wacc, 1.0);
}
`;
