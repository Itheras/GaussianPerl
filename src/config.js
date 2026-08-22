// Tunables and per-device quality presets.

export function isMobile() {
  const ua = navigator.userAgent || '';
  const touchMac = ua.includes('Macintosh') && navigator.maxTouchPoints > 2; // iPadOS
  return /iPhone|iPad|iPod|Android/i.test(ua) || touchMac;
}

// WebKit engine (not Chromium/Gecko). Gates ORT bundle choice and webgpu
// attempts — see MODEL/FILL notes below. Works in workers too.
// EVERY iOS browser is WebKit (Chrome/Edge/Firefox included) — check that
// before the Chromium-token exclusion or CriOS/EdgiOS slip through the gate.
// Chrome on iPadOS ships the desktop-Mac UA and is only detectable on the
// main thread (maxTouchPoints) — main.js forwards that as params.webkitHint.
export function isSafariEngine() {
  const ua = (globalThis.navigator && navigator.userAgent) || '';
  if (/iPhone|iPad|iPod/.test(ua)) return true;
  return /AppleWebKit/i.test(ua) && !/Chrome|Chromium|CriOS|Edg|OPR|Android/i.test(ua);
}

export function defaultQuality() {
  if (isMobile()) return 'medium';
  // modern desktop GPUs handle ~5M splats; webgpu presence is the proxy.
  // 4.2MP working res is display-limited on a 2x canvas — going higher only
  // burns memory without visible sharpness.
  return navigator.gpu ? 'ultra' : 'high';
}

// maxPixels: budget for the fine splat layer (≈ splat count before extras)
export const QUALITY = {
  low: { maxPixels: 380_000 },
  medium: { maxPixels: 730_000 },
  high: { maxPixels: 1_400_000 },
  ultra: { maxPixels: 4_200_000 },
};

export const DEFAULTS = {
  fovYDeg: 55,          // rendering FOV; also assumed capture FOV of the photo
  depthStrength: 1.0,   // scales scene depth range
  splatScale: 1.0,
  dofStrength: 0,       // 0 = off; UI maps aperture slider here
  maxCoC: 26,
  bgTop: [0.075, 0.08, 0.10],
  bgBottom: [0.02, 0.02, 0.03],

  // scene depth mapping (view depth, world units)
  zNearScene: 1.0,      // closest content
  zRange: 7.0,          // depth span across disparity range (scaled by depthStrength)

  // pipeline
  edgeDispJump: 0.055,  // disparity jump (fraction of full range) => discontinuity
  farKnee: 0.16,        // disparity below this counts as "far field"…
  farKeep: 0.25,        // …and keeps only this fraction of its depth spread
  bgBandPx: 0,          // 0 = auto (~7% of the short side, clamped 12..56)
  skirtPx: 0,           // 0 = auto (~10% of the short side, clamped 16..88)
  underlayerStep: 4,    // downsample step for the crack-filling underlayer
};

// transformers.js v4 pinned: v4.2.0 forces ORT's plain (non-JSEP) wasm build on
// Safari — the JSEP binary melts WebKit 26.2's JIT (ORT #26827) — and its
// rewritten WebGPU runtime is current. v3 ships ONLY the JSEP wasm: unsafe on iOS.
export const MODEL = {
  id: 'onnx-community/depth-anything-v2-small',
  idHQ: 'onnx-community/depth-anything-v2-base', // sharper edges; desktop webgpu only
  cdn: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0',
};

// Generative fill: MI-GAN-512-Places2 full ONNX pipeline (first-party export by
// the paper author; MIT repo). uint8 image[1,3,H,W] + mask[1,1,H,W] at ARBITRARY
// resolution; mask 255=known, 0=hole; crop-around-mask(+128px)/resize-512/
// feathered-blend all baked into the graph, so per-cluster calls keep detail.
// Revision-pinned; fetched at runtime (never vendored) and cached via Cache API.
export const FILL = {
  modelUrl: 'https://huggingface.co/andraniksargsyan/migan/resolve/1538c135034b8cfe7a8472f34d09c8a5a45b17a7/migan_pipeline_v2.onnx',
  // byte-identical mirror (unaffiliated packager — sha256 is the trust boundary)
  modelMirrorUrl: 'https://cdn.jsdelivr.net/npm/migan-onnx@1.0.0/models/migan_pipeline_v2.onnx',
  modelSha256: '6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b',
  modelSizeMB: 28,
  // ort bundle choice is a SAFETY matter, not a perf knob (see MODEL note):
  // Safari gets the plain wasm build; others may try the native WebGPU EP.
  // .bundle variants inline the Emscripten loader — only the .wasm is fetched.
  ortWebgpu: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.bundle.min.mjs',
  ortWasm: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.wasm.bundle.min.mjs',
  ortDist: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/',
  // fill planning (see fill-plan.js)
  cellPx: 32,          // hole-clustering grid
  maxBoxPx: 512,       // per-call mask bbox target (graph resizes crop to 512)
  overlapPx: 96,       // tile overlap when splitting oversized boxes
  mergeGapPx: 48,      // merge clusters closer than this
  maxCalls: 6,         // per stage (interior / ring): coarsen boxes beyond this
  minHolePx: 24,       // clusters smaller than this keep the classical fill
};
