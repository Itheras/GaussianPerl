// Disocclusion synthesis: build a background layer (color + disparity) behind
// foreground silhouettes so parallax reveals plausible content instead of holes.
// Classical multi-directional background pull + smoothing + variance-matched
// noise — fast, deterministic, fully on-device.

import { dilateMask } from '../util/imageops.js';

const DIRS = [
  [1, 0], [-1, 0], [0, 1], [0, -1],
  [1, 1], [-1, 1], [1, -1], [-1, -1],
];

/**
 * rgba: Uint8ClampedArray, disp: Float32Array [0,1], fgB: Uint8Array silhouette
 * pixels (near side of discontinuities).
 * Returns {bgColor, bgDisp, bgMask} — synthesized only where bgMask=1.
 */
export function synthesizeBackground(rgba, disp, w, h, fgB, opts = {}) {
  const bandPx = opts.bandPx ?? 20;
  const jump = opts.jump ?? 0.05;
  const band = dilateMask(fgB, w, h, bandPx);

  const bgColor = new Uint8ClampedArray(w * h * 4);
  const bgDisp = new Float32Array(w * h);
  const bgMask = new Uint8Array(w * h);
  const maxMarch = bandPx * 2 + 6;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!band[i]) continue;
      const dHere = disp[i];
      let wr = 0, wg = 0, wb = 0, wd = 0, wsum = 0;
      for (let dir = 0; dir < 8; dir++) {
        const dx = DIRS[dir][0], dy = DIRS[dir][1];
        let px = x, py = y;
        for (let step = 1; step <= maxMarch; step++) {
          px += dx; py += dy;
          if (px < 0 || py < 0 || px >= w || py >= h) break;
          const j = py * w + px;
          // a genuine background sample: clearly farther than this pixel
          if (disp[j] < dHere - jump * 0.6) {
            const dist = step * (dir >= 4 ? 1.4142 : 1);
            const wgt = 1 / (dist + 2);
            wr += rgba[j * 4] * wgt;
            wg += rgba[j * 4 + 1] * wgt;
            wb += rgba[j * 4 + 2] * wgt;
            wd += disp[j] * wgt;
            wsum += wgt;
            break;
          }
        }
      }
      if (wsum > 1e-6) {
        bgMask[i] = 1;
        bgColor[i * 4] = wr / wsum;
        bgColor[i * 4 + 1] = wg / wsum;
        bgColor[i * 4 + 2] = wb / wsum;
        bgColor[i * 4 + 3] = 255;
        // force strictly behind the local foreground
        bgDisp[i] = Math.min(wd / wsum, dHere - jump);
      }
    }
  }

  smoothWithinMask(bgColor, bgDisp, bgMask, w, h);
  addGrain(bgColor, bgMask, w, h);
  return { bgColor, bgDisp, bgMask };
}

// 3x3 blur of color and disparity restricted to the synthesized mask
function smoothWithinMask(bgColor, bgDisp, bgMask, w, h) {
  const cSrc = bgColor.slice();
  const dSrc = bgDisp.slice();
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!bgMask[i]) continue;
      let r = 0, g = 0, b = 0, d = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = Math.min(Math.max(y + dy, 0), h - 1);
        for (let dx = -1; dx <= 1; dx++) {
          const xx = Math.min(Math.max(x + dx, 0), w - 1);
          const j = yy * w + xx;
          if (!bgMask[j]) continue;
          r += cSrc[j * 4]; g += cSrc[j * 4 + 1]; b += cSrc[j * 4 + 2];
          d += dSrc[j]; n++;
        }
      }
      if (n > 0) {
        bgColor[i * 4] = r / n; bgColor[i * 4 + 1] = g / n; bgColor[i * 4 + 2] = b / n;
        bgDisp[i] = d / n;
      }
    }
  }
}

// subtle deterministic grain so the fill doesn't read as an airbrushed smear
function addGrain(bgColor, bgMask, w, h) {
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!bgMask[i]) continue;
      let hsh = (x * 374761393 + y * 668265263) | 0;
      hsh = (hsh ^ (hsh >>> 13)) * 1274126177;
      const n = (((hsh ^ (hsh >>> 16)) >>> 0) / 4294967295 - 0.5) * 9;
      bgColor[i * 4] += n; bgColor[i * 4 + 1] += n; bgColor[i * 4 + 2] += n;
    }
  }
}
