// Disparity processing: normalization, fallback heuristic, GT decode,
// disparity->depth mapping, discontinuity detection. Pure — runs in the worker.

import { jointBilateral, boxBlurFloat, percentile, gradients } from '../util/imageops.js';

/** Robust-normalize raw disparity (bigger = closer) to [0,1]. */
export function normalizeDisparity(raw) {
  const lo = percentile(raw, 0.015);
  const hi = percentile(raw, 0.985);
  const range = Math.max(hi - lo, 1e-6);
  const out = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const v = (raw[i] - lo) / range;
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return out;
}

/** Decode our RG-packed 16-bit GT disparity PNG (R=hi, G=lo). */
export function decodeGtDisparity(rgba, w, h) {
  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    out[i] = (rgba[i * 4] * 256 + rgba[i * 4 + 1]) / 65535;
  }
  return out;
}

/**
 * No-AI fallback: vertical gradient (bottom = near) + mild center prior,
 * then edge-aware smoothing guided by the image so it hugs large structures.
 * Marked approximate — used only when the model can't load.
 */
export function heuristicDisparity(rgba, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const ty = y / (h - 1);
    for (let x = 0; x < w; x++) {
      const tx = x / (w - 1);
      const cx = (tx - 0.5) * 2, cy = (ty - 0.55) * 2;
      const center = Math.exp(-(cx * cx * 1.4 + cy * cy * 2.2)) * 0.22;
      out[y * w + x] = Math.pow(ty, 1.35) * 0.8 + center;
    }
  }
  const sm = jointBilateral(out, rgba, w, h, 3, 30, 3);
  return normalizeDisparity(sm);
}

/**
 * Compress the far tail of the disparity range: in real life the far field
 * (mountains vs sky) shows ~zero relative parallax for a head-sized camera
 * move, but a normalized-disparity range mapped onto a finite depth span
 * exaggerates it into floating "cardboard" slabs — and makes the horizon line
 * read as a giant disocclusion edge (hallucination-prone fill band, slab
 * curtain under yaw). Monotone map: [0, knee] -> [knee*(1-keep), knee],
 * everything >= knee unchanged. Run BEFORE snap/edge detection so a
 * compressed horizon stops being a silhouette; true near-vs-sky silhouettes
 * still jump far beyond the edge threshold.
 */
export function compressFarField(disp, knee = 0.08, keep = 0.35) {
  const out = new Float32Array(disp.length);
  for (let i = 0; i < disp.length; i++) {
    const d = disp[i];
    out[i] = d < knee ? knee - (knee - d) * keep : d;
  }
  return out;
}

/**
 * Discontinuity mask: 1 where the 4-neighborhood disparity jump exceeds
 * `jump` (in units of the [0,1] disparity range).
 */
export function edgeMask(disp, w, h, jump) {
  const mask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const ym = Math.max(y - 1, 0) * w, yp = Math.min(y + 1, h - 1) * w;
    for (let x = 0; x < w; x++) {
      const d = disp[row + x];
      const xm = Math.max(x - 1, 0), xp = Math.min(x + 1, w - 1);
      const m = Math.max(
        Math.abs(d - disp[row + xm]), Math.abs(d - disp[row + xp]),
        Math.abs(d - disp[ym + x]), Math.abs(d - disp[yp + x]));
      if (m > jump) mask[row + x] = 1;
    }
  }
  return mask;
}

/**
 * Foreground silhouette pixels: on an edge AND locally nearer than the far
 * side. These are the pixels whose parallax will reveal a hole behind them.
 */
export function fgBoundary(disp, w, h, jump) {
  const mask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const ym = Math.max(y - 1, 0) * w, yp = Math.min(y + 1, h - 1) * w;
    for (let x = 0; x < w; x++) {
      const d = disp[row + x];
      const xm = Math.max(x - 1, 0), xp = Math.min(x + 1, w - 1);
      if (d - disp[row + xm] > jump || d - disp[row + xp] > jump ||
          d - disp[ym + x] > jump || d - disp[yp + x] > jump) {
        mask[row + x] = 1;
      }
    }
  }
  return mask;
}

/** Simple smoothing utility re-export used by the worker. */
export { boxBlurFloat, gradients };
