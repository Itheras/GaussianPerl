// Production depth post-filter stack (M8) — replaces joint-bilateral refine +
// snap + color-align, which had two structural flaws: color-guided bilateral
// IMPRINTS image texture into geometry (faces "swim" when the camera moves),
// and 3x3 snapping cannot fix boundaries offset across flat ramps.
//
// Stack (all pure typed-array code, run at DEPTH resolution <=~1.5MP):
//   1. fgsSmooth      — Fast Global Smoother (Min et al., TIP 2014): exact
//                       1D tridiagonal solves alternating H/V with gradient-
//                       based conductance. Flattens interior noise WITHOUT
//                       halos or texture imprint; doubles as the edge-aware
//                       upsampler from model res.
//   2. weightedMedianDepth — Kopf One-Shot-3D-Photography 5x5 weighted median
//                       with edge-sample rejection: turns silhouette ramps
//                       into ~1px steps.
//   3. mergeFloaters  — connected components under a disparity tolerance;
//                       tiny floaters (depth debris over textured regions —
//                       crowds!) merge into the neighbor with the largest
//                       contact perimeter.
//   4. relocateEdges  — gated bilateral-median ON DISPARITY at discontinuity
//                       pixels only (Shih), moving depth boundaries onto
//                       image edges when a local image edge agrees in
//                       orientation (mutual-structure gate).

/** conductance weight LUT: w = exp(-|dI| / sigmaColor), |dI| in 0..255 */
function colorWeightLUT(sigmaColor) {
  const lut = new Float32Array(256);
  for (let i = 0; i < 256; i++) lut[i] = Math.exp(-i / sigmaColor);
  return lut;
}

function lumaDiff(rgba, i, j) {
  const oi = i * 4, oj = j * 4;
  return (Math.abs(rgba[oi] - rgba[oj]) +
    Math.abs(rgba[oi + 1] - rgba[oj + 1]) +
    Math.abs(rgba[oi + 2] - rgba[oj + 2])) / 3;
}

/**
 * Fast Global Smoother. Solves (I + lambda_t * A) u = d per row then per
 * column, T times, with the lambda schedule from the paper:
 * lambda_t = 1.5 * lambda * 4^(T-t) / (4^T - 1).
 * Weights w_pq = exp(-|I_p - I_q| / sigmaColor) between 4-neighbors.
 * Returns a new Float32Array.
 */
export function fgsSmooth(disp, rgba, w, h, opts = {}) {
  const lambda = opts.lambda ?? 900;
  const sigmaColor = opts.sigmaColor ?? 7;
  const T = opts.iterations ?? 3;
  const lut = colorWeightLUT(sigmaColor);

  // inter-pixel conductances (image-constant across iterations)
  const wH = new Float32Array(w * h); // between (x,y) and (x+1,y)
  const wV = new Float32Array(w * h); // between (x,y) and (x,y+1)
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      if (x < w - 1) wH[i] = lut[lumaDiff(rgba, i, i + 1) | 0];
      if (y < h - 1) wV[i] = lut[lumaDiff(rgba, i, i + w) | 0];
    }
  }

  let u = Float32Array.from(disp);
  const cPrime = new Float32Array(Math.max(w, h));
  const dPrime = new Float32Array(Math.max(w, h));

  const denomTotal = Math.pow(4, T) - 1;
  for (let t = 1; t <= T; t++) {
    const lt = 1.5 * lambda * Math.pow(4, T - t) / denomTotal;

    // horizontal pass: tridiagonal solve per row (Thomas)
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const i = row + x;
        const wl = x > 0 ? wH[i - 1] : 0;
        const wr = x < w - 1 ? wH[i] : 0;
        const a = -lt * wl;             // sub-diagonal
        const b = 1 + lt * (wl + wr);   // diagonal
        const c = -lt * wr;             // super-diagonal
        if (x === 0) {
          cPrime[0] = c / b;
          dPrime[0] = u[i] / b;
        } else {
          const m = b - a * cPrime[x - 1];
          cPrime[x] = c / m;
          dPrime[x] = (u[i] - a * dPrime[x - 1]) / m;
        }
      }
      u[row + w - 1] = dPrime[w - 1];
      for (let x = w - 2; x >= 0; x--) {
        u[row + x] = dPrime[x] - cPrime[x] * u[row + x + 1];
      }
    }

    // vertical pass: tridiagonal solve per column
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        const i = y * w + x;
        const wu = y > 0 ? wV[i - w] : 0;
        const wd = y < h - 1 ? wV[i] : 0;
        const a = -lt * wu;
        const b = 1 + lt * (wu + wd);
        const c = -lt * wd;
        if (y === 0) {
          cPrime[0] = c / b;
          dPrime[0] = u[i] / b;
        } else {
          const m = b - a * cPrime[y - 1];
          cPrime[y] = c / m;
          dPrime[y] = (u[i] - a * dPrime[y - 1]) / m;
        }
      }
      u[(h - 1) * w + x] = dPrime[h - 1];
      for (let y = h - 2; y >= 0; y--) {
        u[y * w + x] = dPrime[y] - cPrime[y] * u[(y + 1) * w + x];
      }
    }
  }
  return u;
}

/**
 * Kopf-style 5x5 weighted median: weight_i = exp(-(d_i - d_c)^2 / (2*0.2^2)),
 * with samples that sit ON an edge (any 4-neighbor differing > tau) excluded
 * entirely — the median then comes from one SIDE of the silhouette, turning a
 * soft ramp into a step instead of averaging across it.
 */
export function weightedMedianDepth(disp, w, h, tau = 0.055) {
  const n = w * h;
  // edge-sample flags
  const onEdge = new Uint8Array(n);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      const d = disp[i];
      if ((x > 0 && Math.abs(disp[i - 1] - d) > tau) ||
          (x < w - 1 && Math.abs(disp[i + 1] - d) > tau) ||
          (y > 0 && Math.abs(disp[i - w] - d) > tau) ||
          (y < h - 1 && Math.abs(disp[i + w] - d) > tau)) {
        onEdge[i] = 1;
      }
    }
  }
  const out = new Float32Array(n);
  const vals = new Float32Array(25);
  const wgts = new Float32Array(25);
  const order = new Int32Array(25);
  const invTwoSigma2 = 1 / (2 * 0.2 * 0.2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const dc = disp[i];
      let m = 0, wsum = 0;
      for (let dy = -2; dy <= 2; dy++) {
        const yy = Math.min(Math.max(y + dy, 0), h - 1);
        for (let dx = -2; dx <= 2; dx++) {
          const xx = Math.min(Math.max(x + dx, 0), w - 1);
          const j = yy * w + xx;
          if (onEdge[j] && j !== i) continue;
          const dv = disp[j];
          const diff = dv - dc;
          const wgt = Math.exp(-diff * diff * invTwoSigma2);
          vals[m] = dv; wgts[m] = wgt; wsum += wgt; m++;
        }
      }
      if (m === 0 || wsum <= 0) { out[i] = dc; continue; }
      for (let k = 0; k < m; k++) order[k] = k;
      // insertion sort by value (m <= 25)
      for (let k = 1; k < m; k++) {
        const ok = order[k];
        const v = vals[ok];
        let p = k - 1;
        while (p >= 0 && vals[order[p]] > v) { order[p + 1] = order[p]; p--; }
        order[p + 1] = ok;
      }
      const half = wsum / 2;
      let acc = 0, med = vals[order[m - 1]];
      for (let k = 0; k < m; k++) {
        acc += wgts[order[k]];
        if (acc >= half) { med = vals[order[k]]; break; }
      }
      out[i] = med;
    }
  }
  return out;
}

/**
 * Merge tiny disparity components ("floaters" — depth debris the model
 * hallucinates over textured regions) into the neighboring component with the
 * largest contact perimeter. Components are 4-connected under |dd| <= tau.
 * minArea should scale with resolution: ~20px * (min(w,h)/384)^2.
 * Mutates and returns disp.
 */
export function mergeFloaters(disp, w, h, tau = 0.055, minArea = 20) {
  const n = w * h;
  const label = new Int32Array(n).fill(-1);
  const areas = [];
  let queue = new Int32Array(n);
  for (let seed = 0; seed < n; seed++) {
    if (label[seed] >= 0) continue;
    const id = areas.length;
    let qh = 0, qt = 0;
    queue[qt++] = seed;
    label[seed] = id;
    let area = 0;
    while (qh < qt) {
      const i = queue[qh++];
      area++;
      const x = i % w, y = (i / w) | 0;
      const d = disp[i];
      if (x > 0 && label[i - 1] < 0 && Math.abs(disp[i - 1] - d) <= tau) { label[i - 1] = id; queue[qt++] = i - 1; }
      if (x < w - 1 && label[i + 1] < 0 && Math.abs(disp[i + 1] - d) <= tau) { label[i + 1] = id; queue[qt++] = i + 1; }
      if (y > 0 && label[i - w] < 0 && Math.abs(disp[i - w] - d) <= tau) { label[i - w] = id; queue[qt++] = i - w; }
      if (y < h - 1 && label[i + w] < 0 && Math.abs(disp[i + w] - d) <= tau) { label[i + w] = id; queue[qt++] = i + w; }
    }
    areas.push(area);
  }
  // for each small component: count contacts per neighbor label, take the max
  const contactCount = new Map();
  const contactSum = new Map();
  for (let ci = 0; ci < areas.length; ci++) {
    if (areas[ci] >= minArea) continue;
    contactCount.clear();
    contactSum.clear();
    for (let i = 0; i < n; i++) {
      if (label[i] !== ci) continue;
      const x = i % w, y = (i / w) | 0;
      for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1,
        y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) {
        if (j < 0) continue;
        const lj = label[j];
        if (lj === ci) continue;
        contactCount.set(lj, (contactCount.get(lj) || 0) + 1);
        contactSum.set(lj, (contactSum.get(lj) || 0) + disp[j]);
      }
    }
    let best = -1, bestCount = 0;
    for (const [lj, c] of contactCount) {
      if (c > bestCount) { best = lj; bestCount = c; }
    }
    if (best < 0) continue;
    const target = contactSum.get(best) / bestCount;
    for (let i = 0; i < n; i++) if (label[i] === ci) disp[i] = target;
  }
  return disp;
}

/**
 * Gated edge relocation (Shih-style): iterative bilateral median ON DISPARITY
 * restricted to discontinuity pixels, gated so relocation happens only where
 * a local IMAGE edge exists whose orientation agrees with the disparity
 * gradient (within ~30 deg) — prevents spurious snaps in textured regions.
 */
export function relocateEdges(disp, rgba, w, h, opts = {}) {
  const tau = opts.tau ?? 0.04;
  const windows = opts.windows ?? [7, 7, 5, 5, 5];
  const sigmaS = opts.sigmaS ?? 4.0;
  const sigmaR = opts.sigmaR ?? 0.5;
  const edgeGradMin = opts.edgeGradMin ?? 8; // luma units/px
  const cos30 = Math.cos(Math.PI / 6);
  const n = w * h;

  // image luma gradient (constant)
  const luma = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    luma[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
  }
  let igx = new Float32Array(n), igy = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    const ym = Math.max(y - 1, 0) * w, yp = Math.min(y + 1, h - 1) * w, row = y * w;
    for (let x = 0; x < w; x++) {
      const xm = Math.max(x - 1, 0), xp = Math.min(x + 1, w - 1);
      igx[row + x] = (luma[row + xp] - luma[row + xm]) * 0.5;
      igy[row + x] = (luma[yp + x] - luma[ym + x]) * 0.5;
    }
  }
  // the gate must see an image edge NEARBY (the depth boundary may sit a few
  // px off the image edge — that offset is what relocation fixes): propagate
  // the strongest gradient vector over a ~3px radius (3x3 argmax passes)
  for (let it = 0; it < 3; it++) {
    const nx = Float32Array.from(igx), ny = Float32Array.from(igy);
    for (let y = 0; y < h; y++) {
      const ym = Math.max(y - 1, 0), yp = Math.min(y + 1, h - 1), row = y * w;
      for (let x = 0; x < w; x++) {
        const i = row + x;
        let bm = igx[i] * igx[i] + igy[i] * igy[i];
        for (const j of [row + Math.max(x - 1, 0), row + Math.min(x + 1, w - 1),
          ym * w + x, yp * w + x]) {
          const m2 = igx[j] * igx[j] + igy[j] * igy[j];
          if (m2 > bm) { bm = m2; nx[i] = igx[j]; ny[i] = igy[j]; }
        }
      }
    }
    igx = nx; igy = ny;
  }

  let cur = Float32Array.from(disp);
  const vals = new Float32Array(49);
  const wgts = new Float32Array(49);
  const order = new Int32Array(49);

  for (const win of windows) {
    const r = win >> 1;
    const next = Float32Array.from(cur);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const i = row + x;
        const d = cur[i];
        // discontinuity pixels only (mask recomputed each iteration via cur)
        const isEdge =
          (x > 0 && Math.abs(cur[i - 1] - d) > tau) ||
          (x < w - 1 && Math.abs(cur[i + 1] - d) > tau) ||
          (y > 0 && Math.abs(cur[i - w] - d) > tau) ||
          (y < h - 1 && Math.abs(cur[i + w] - d) > tau);
        if (!isEdge) continue;
        // mutual-structure gate: a real image edge, aligned with the depth edge
        const gI = Math.hypot(igx[i], igy[i]);
        if (gI < edgeGradMin) continue;
        const xm2 = Math.max(x - 1, 0), xp2 = Math.min(x + 1, w - 1);
        const ym2 = Math.max(y - 1, 0), yp2 = Math.min(y + 1, h - 1);
        const dgx = (cur[row + xp2] - cur[row + xm2]) * 0.5;
        const dgy = (cur[yp2 * w + x] - cur[ym2 * w + x]) * 0.5;
        const gD = Math.hypot(dgx, dgy);
        if (gD > 1e-9) {
          const cosA = Math.abs(igx[i] * dgx + igy[i] * dgy) / (gI * gD);
          if (cosA < cos30) continue;
        }
        // bilateral median over the window (disparity-range weighted)
        let m = 0, wsum = 0;
        for (let dy = -r; dy <= r; dy++) {
          const yy = Math.min(Math.max(y + dy, 0), h - 1);
          for (let dx = -r; dx <= r; dx++) {
            const xx = Math.min(Math.max(x + dx, 0), w - 1);
            const dv = cur[yy * w + xx];
            const sw = Math.exp(-(dx * dx + dy * dy) / (2 * sigmaS * sigmaS));
            const dr = (dv - d) / sigmaR;
            const wgt = sw * Math.exp(-dr * dr / 2);
            vals[m] = dv; wgts[m] = wgt; wsum += wgt; m++;
          }
        }
        for (let k = 0; k < m; k++) order[k] = k;
        for (let k = 1; k < m; k++) {
          const ok = order[k];
          const v = vals[ok];
          let p = k - 1;
          while (p >= 0 && vals[order[p]] > v) { order[p + 1] = order[p]; p--; }
          order[p + 1] = ok;
        }
        const half = wsum / 2;
        let acc = 0;
        for (let k = 0; k < m; k++) {
          acc += wgts[order[k]];
          if (acc >= half) { next[i] = vals[order[k]]; break; }
        }
      }
    }
    cur = next;
  }
  return cur;
}
