// Disparity processing: normalization, fallback heuristic, GT decode,
// disparity->depth mapping, discontinuity detection. Pure — runs in the worker.

import {
  jointBilateral, boxBlurFloat, percentile, gradients, dilateMask,
} from '../util/imageops.js';

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
 * Edge-preserving refinement of model disparity guided by the color image.
 * The model works at ~518px internally, so its output is upsampled ~2-4.5x to
 * the working res — the refine radius must scale with that ratio or, at
 * 'ultra' resolutions, a 2px kernel can't pull depth edges onto color edges
 * and silhouettes shred into confetti under yaw.
 */
export function refineDisparity(disp, rgba, w, h) {
  const r = Math.max(2, Math.min(4, Math.round(Math.min(w, h) / 560)));
  return jointBilateral(disp, rgba, w, h, r, 24, r);
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
 * Map disparity [0,1] to positive view depth. Physically disparity ~ 1/z:
 * 1/z = d*(1/zn - 1/zf) + 1/zf, with zf = zn + range.
 */
export function disparityToDepth(disp, zNear, zRange) {
  const zf = zNear + zRange;
  const a = 1 / zNear - 1 / zf, b = 1 / zf;
  const out = new Float32Array(disp.length);
  for (let i = 0; i < disp.length; i++) {
    out[i] = 1 / (disp[i] * a + b);
  }
  return out;
}

/**
 * Snap soft disparity ramps at discontinuities back into steps: wherever the
 * local 3x3 range exceeds ~the discontinuity threshold, move the value to the
 * nearer of (local min, local max). Kills the mixed-depth "streak" splats that
 * bilinear resampling (and soft AI output) creates along silhouettes.
 * Iterations move the ramp ~1px each — callers scale with resolution (the
 * model works at ~518px; at 'ultra' the upsampled ramp is ~10px wide).
 */
export function snapDepthEdges(disp, w, h, jump, iterations = 2) {
  let cur = disp;
  for (let it = 0; it < iterations; it++) {
    const out = new Float32Array(cur.length);
    for (let y = 0; y < h; y++) {
      const ym = Math.max(y - 1, 0), yp = Math.min(y + 1, h - 1);
      for (let x = 0; x < w; x++) {
        const xm = Math.max(x - 1, 0), xp = Math.min(x + 1, w - 1);
        let lo = Infinity, hi = -Infinity;
        for (const yy of [ym, y, yp]) {
          const row = yy * w;
          for (const xx of [xm, x, xp]) {
            const v = cur[row + xx];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
        }
        // threshold matches the > jump test used by edgeMask/fgBoundary/dz():
        // anything we steepen into a step WILL be treated as a silhouette;
        // anything below stays smooth for every consumer (no blind window)
        const d = cur[y * w + x];
        out[y * w + x] = (hi - lo > jump)
          ? ((d - lo < hi - d) ? lo : hi)
          : d;
      }
    }
    cur = out;
  }
  return cur;
}

/**
 * Pull depth boundaries onto COLOR boundaries. The model's silhouette can sit
 * several px off the true edge even after snapping (it saw the image at
 * ~518px): body-colored pixels stranded at background depth render as dark
 * streak sheets under yaw. For every pixel within `radius` of a depth edge,
 * group the (2r+1)^2 neighborhood into near/far by disparity, and if the
 * pixel's color decisively matches the OTHER side, move it there (to that
 * side's mean disparity). Radius should match the model->working upsample
 * ratio. Run AFTER snapDepthEdges (needs steps, not ramps).
 */
export function alignEdgesToColor(disp, rgba, w, h, jump, radius = 4) {
  const edges = edgeMask(disp, w, h, jump);
  const zone = dilateMask(edges, w, h, radius);
  const out = Float32Array.from(disp);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(y - radius, 0), y1 = Math.min(y + radius, h - 1);
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!zone[i]) continue;
      const x0 = Math.max(x - radius, 0), x1 = Math.min(x + radius, w - 1);
      let lo = Infinity, hi = -Infinity;
      for (let yy = y0; yy <= y1; yy++) {
        const row = yy * w;
        for (let xx = x0; xx <= x1; xx++) {
          const v = disp[row + xx];
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      }
      if (hi - lo <= jump) continue;
      const mid = (lo + hi) * 0.5;
      let lr = 0, lg = 0, lb = 0, ld = 0, ln = 0;
      let hr = 0, hg = 0, hb = 0, hd = 0, hn = 0;
      for (let yy = y0; yy <= y1; yy++) {
        const row = yy * w;
        for (let xx = x0; xx <= x1; xx++) {
          const j = row + xx;
          const o = j * 4;
          if (disp[j] >= mid) {
            hr += rgba[o]; hg += rgba[o + 1]; hb += rgba[o + 2]; hd += disp[j]; hn++;
          } else {
            lr += rgba[o]; lg += rgba[o + 1]; lb += rgba[o + 2]; ld += disp[j]; ln++;
          }
        }
      }
      if (ln === 0 || hn === 0) continue;
      const o = i * 4;
      const r0 = rgba[o], g0 = rgba[o + 1], b0 = rgba[o + 2];
      const dLo = Math.abs(r0 - lr / ln) + Math.abs(g0 - lg / ln) + Math.abs(b0 - lb / ln);
      const dHi = Math.abs(r0 - hr / hn) + Math.abs(g0 - hg / hn) + Math.abs(b0 - hb / hn);
      // decisive margin only — ambiguous colors keep the model's depth
      if (Math.abs(dLo - dHi) <= 20) continue;
      out[i] = dLo < dHi ? ld / ln : hd / hn;
    }
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
