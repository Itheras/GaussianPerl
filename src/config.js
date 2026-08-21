// Tunables and per-device quality presets.

export function isMobile() {
  const ua = navigator.userAgent || '';
  const touchMac = ua.includes('Macintosh') && navigator.maxTouchPoints > 2; // iPadOS
  return /iPhone|iPad|iPod|Android/i.test(ua) || touchMac;
}

export function defaultQuality() {
  if (isMobile()) return 'medium';
  return 'high';
}

// maxPixels: budget for the fine splat layer (≈ splat count before extras)
export const QUALITY = {
  low: { maxPixels: 380_000, inpaintScale: 0.5 },
  medium: { maxPixels: 730_000, inpaintScale: 0.5 },
  high: { maxPixels: 1_400_000, inpaintScale: 0.6 },
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
  bgBandPx: 0,          // 0 = auto (~7% of the short side, clamped 12..56)
  skirtPx: 24,          // border outpaint skirt width (in working pixels)
  underlayerStep: 4,    // downsample step for the crack-filling underlayer
};

export const MODEL = {
  id: 'onnx-community/depth-anything-v2-small',
  cdn: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3',
};
