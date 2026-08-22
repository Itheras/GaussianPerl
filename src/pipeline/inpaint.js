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
  // a band pixel at distance d <= bandPx from the silhouette needs a march of
  // ~d to cross it plus a few px of clean background — bandPx*2 was pure waste
  // and the march dominates the classical stage at high resolutions
  const maxMarch = bandPx + 16;

  const hitD = new Float32Array(8);
  const hitJ = new Int32Array(8);
  const hitW = new Float32Array(8);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!band[i]) continue;
      const dHere = disp[i];
      let nHits = 0;
      for (let dir = 0; dir < 8; dir++) {
        const dx = DIRS[dir][0], dy = DIRS[dir][1];
        let px = x, py = y;
        let prevD = dHere;
        let crossedEdge = false;
        for (let step = 1; step <= maxMarch; step++) {
          px += dx; py += dy;
          if (px < 0 || py < 0 || px >= w || py >= h) break;
          const j = py * w + px;
          const dj = disp[j];
          // must exit through a silhouette — a smooth gradient (open ground)
          // never counts, no matter how far it drops over the march
          if (prevD - dj > jump * 0.8) crossedEdge = true;
          prevD = dj;
          if (crossedEdge && dj < dHere - jump * 0.6) {
            const dist = step * (dir >= 4 ? 1.4142 : 1);
            hitD[nHits] = dj;
            hitJ[nHits] = j;
            hitW[nHits] = 1 / (dist + 2);
            nHits++;
            break;
          }
        }
      }
      if (nHits === 0) continue;
      // Median of hit disparities picks the DOMINANT background
      // surface (ground behind a stone's base, sky behind its top) — never a
      // mid-air average between layers (the "veil" artifact), never a lone
      // far outlier (holes at ground level).
      const order = [];
      for (let k = 0; k < nHits; k++) order.push(k);
      order.sort((a, b) => hitD[a] - hitD[b]);
      // plain median — distance weights would bias toward the near side
      const dMed = hitD[order[nHits >> 1]];
      // only synthesize where a real disocclusion gap exists
      if (dHere - dMed <= jump * 1.2) continue;
      // blend within the median's cluster only
      let wr = 0, wg = 0, wb = 0, wd = 0, wsum = 0;
      for (let k = 0; k < nHits; k++) {
        if (Math.abs(hitD[k] - dMed) > jump * 0.7) continue;
        const j = hitJ[k], wgt = hitW[k];
        wr += rgba[j * 4] * wgt;
        wg += rgba[j * 4 + 1] * wgt;
        wb += rgba[j * 4 + 2] * wgt;
        wd += hitD[k] * wgt;
        wsum += wgt;
      }
      if (wsum <= 1e-6) continue;
      bgMask[i] = 1;
      bgColor[i * 4] = wr / wsum;
      bgColor[i * 4 + 1] = wg / wsum;
      bgColor[i * 4 + 2] = wb / wsum;
      bgColor[i * 4 + 3] = 255;
      // force strictly behind the local foreground
      bgDisp[i] = Math.min(wd / wsum, dHere - jump);
    }
  }

  // the 8-direction march fails on cluttered backgrounds (a crowd at noisy
  // depths defeats the crossed-edge test) leaving COVERAGE HOLES inside the
  // fill region — revealed as literal void at sharp angles. Close them by
  // flooding from successfully filled neighbors.
  closeBandHoles({ bgColor, bgDisp, bgMask }, disp, band, w, h, jump, bandPx);

  smoothWithinMask(bgColor, bgDisp, bgMask, w, h, jump);
  addGrain(bgColor, bgMask, w, h);
  return { bgColor, bgDisp, bgMask };
}

/**
 * BFS from every filled bg pixel into band pixels the march failed on: a
 * pixel joins when it sits genuinely IN FRONT of the neighbor's fill surface
 * (disp > fill disp + jump — i.e. a real disocclusion the march missed).
 * First-come = nearest donor; inherits its color and disparity. Exported for
 * tests.
 */
export function closeBandHoles(bg, disp, band, w, h, jump, maxSteps) {
  const { bgColor, bgDisp, bgMask } = bg;
  const n = w * h;
  const dist = new Int16Array(n).fill(-1);
  let queue = new Int32Array(Math.max(256, n >> 3));
  let qHead = 0, qTail = 0;
  const push = (i) => {
    if (qTail === queue.length) {
      const bigger = new Int32Array(queue.length * 2);
      bigger.set(queue);
      queue = bigger;
    }
    queue[qTail++] = i;
  };
  for (let i = 0; i < n; i++) if (bgMask[i]) { dist[i] = 0; push(i); }
  while (qHead < qTail) {
    const i = queue[qHead++];
    const d = dist[i];
    if (d >= maxSteps) continue;
    const x = i % w, y = (i / w) | 0;
    for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1,
      y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) {
      if (j < 0 || dist[j] >= 0 || !band[j]) continue;
      // only a genuine disocclusion: the pixel is clearly nearer than the
      // fill surface it would sit in front of (a different donor with a
      // farther surface may still claim it later — don't mark rejections)
      if (disp[j] - bgDisp[i] <= jump) continue;
      dist[j] = d + 1;
      bgMask[j] = 1;
      bgDisp[j] = Math.min(bgDisp[i], disp[j] - jump);
      bgColor[j * 4] = bgColor[i * 4];
      bgColor[j * 4 + 1] = bgColor[i * 4 + 1];
      bgColor[j * 4 + 2] = bgColor[i * 4 + 2];
      bgColor[j * 4 + 3] = 255;
      push(j);
    }
  }
}

// 3x3 blur of color and disparity restricted to the synthesized mask —
// and to neighbors at a SIMILAR fill depth, so a sky-depth fill never
// bleeds milky color into a ground-depth fill next to it
function smoothWithinMask(bgColor, bgDisp, bgMask, w, h, jump) {
  const cSrc = bgColor.slice();
  const dSrc = bgDisp.slice();
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!bgMask[i]) continue;
      const d0 = dSrc[i];
      let r = 0, g = 0, b = 0, d = 0, n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = Math.min(Math.max(y + dy, 0), h - 1);
        for (let dx = -1; dx <= 1; dx++) {
          const xx = Math.min(Math.max(x + dx, 0), w - 1);
          const j = yy * w + xx;
          if (!bgMask[j] || Math.abs(dSrc[j] - d0) > jump) continue;
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
// (also applied over AI fills to hide the model's 512-crop softness)
export function addGrain(bgColor, bgMask, w, h) {
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
