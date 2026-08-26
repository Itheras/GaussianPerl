// Pure typed-array image operations (no DOM) — usable in workers and node tests.

// Bilinear-resample a single-channel Float32 image.
export function resizeFloat(src, sw, sh, dw, dh) {
  const dst = new Float32Array(dw * dh);
  const xr = sw / dw, yr = sh / dh;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min((y + 0.5) * yr - 0.5, sh - 1.001);
    const y0 = Math.max(Math.floor(sy), 0);
    const y1 = Math.min(y0 + 1, sh - 1);
    const fy = Math.min(Math.max(sy - y0, 0), 1);
    for (let x = 0; x < dw; x++) {
      const sx = Math.min((x + 0.5) * xr - 0.5, sw - 1.001);
      const x0 = Math.max(Math.floor(sx), 0);
      const x1 = Math.min(x0 + 1, sw - 1);
      const fx = Math.min(Math.max(sx - x0, 0), 1);
      const a = src[y0 * sw + x0], b = src[y0 * sw + x1];
      const c = src[y1 * sw + x0], d = src[y1 * sw + x1];
      dst[y * dw + x] = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
    }
  }
  return dst;
}

// Bilinear-resample RGBA Uint8 image.
export function resizeRGBA(src, sw, sh, dw, dh) {
  const dst = new Uint8ClampedArray(dw * dh * 4);
  const xr = sw / dw, yr = sh / dh;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min((y + 0.5) * yr - 0.5, sh - 1.001);
    const y0 = Math.max(Math.floor(sy), 0);
    const y1 = Math.min(y0 + 1, sh - 1);
    const fy = Math.min(Math.max(sy - y0, 0), 1);
    for (let x = 0; x < dw; x++) {
      const sx = Math.min((x + 0.5) * xr - 0.5, sw - 1.001);
      const x0 = Math.max(Math.floor(sx), 0);
      const x1 = Math.min(x0 + 1, sw - 1);
      const fx = Math.min(Math.max(sx - x0, 0), 1);
      const i00 = (y0 * sw + x0) * 4, i01 = (y0 * sw + x1) * 4;
      const i10 = (y1 * sw + x0) * 4, i11 = (y1 * sw + x1) * 4;
      const o = (y * dw + x) * 4;
      for (let k = 0; k < 4; k++) {
        const a = src[i00 + k], b = src[i01 + k], c = src[i10 + k], d = src[i11 + k];
        dst[o + k] = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
      }
    }
  }
  return dst;
}

// In-place-ish separable box blur (returns new array). radius in pixels.
export function boxBlurFloat(src, w, h, radius) {
  if (radius <= 0) return src.slice();
  const tmp = new Float32Array(w * h);
  const dst = new Float32Array(w * h);
  const win = radius * 2 + 1;
  for (let y = 0; y < h; y++) {
    let acc = 0;
    const row = y * w;
    for (let x = -radius; x <= radius; x++) acc += src[row + Math.min(Math.max(x, 0), w - 1)];
    for (let x = 0; x < w; x++) {
      tmp[row + x] = acc / win;
      const xAdd = Math.min(x + radius + 1, w - 1);
      const xSub = Math.max(x - radius, 0);
      acc += src[row + xAdd] - src[row + xSub];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let y = -radius; y <= radius; y++) acc += tmp[Math.min(Math.max(y, 0), h - 1) * w + x];
    for (let y = 0; y < h; y++) {
      dst[y * w + x] = acc / win;
      const yAdd = Math.min(y + radius + 1, h - 1);
      const ySub = Math.max(y - radius, 0);
      acc += tmp[yAdd * w + x] - tmp[ySub * w + x];
    }
  }
  return dst;
}

// Joint bilateral filter: smooth `val` (Float32, single channel) guided by RGBA image.
// Snaps value edges to color edges. sigmaColor in 0..255 luminance-ish units.
// The color weight uses a 2048-entry LUT — the naive form costs ~35M Math.exp
// calls on a 1.4MP image, ~1s of a phone core for no visible difference.
export function jointBilateral(val, rgba, w, h, radius = 2, sigmaColor = 22, sigmaSpace = 2) {
  const dst = new Float32Array(w * h);
  const invS2 = 1 / (2 * sigmaSpace * sigmaSpace);
  const invC2 = 1 / (2 * sigmaColor * sigmaColor);
  // Precompute spatial weights
  const size = radius * 2 + 1;
  const sw = new Float32Array(size * size);
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      sw[(dy + radius) * size + (dx + radius)] = Math.exp(-(dx * dx + dy * dy) * invS2);
    }
  }
  // Color-weight LUT over squared RGB distance [0, 3*255^2]
  const LUT_N = 2048;
  const maxD2 = 3 * 255 * 255;
  const lutScale = (LUT_N - 1) / maxD2;
  const lut = new Float32Array(LUT_N);
  for (let i = 0; i < LUT_N; i++) {
    lut[i] = Math.exp(-(i / lutScale) * invC2 / 3);
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ci = (y * w + x) * 4;
      const r0 = rgba[ci], g0 = rgba[ci + 1], b0 = rgba[ci + 2];
      let sum = 0, wsum = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = Math.min(Math.max(y + dy, 0), h - 1);
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = Math.min(Math.max(x + dx, 0), w - 1);
          const ni = (yy * w + xx);
          const nc = ni * 4;
          const dr = rgba[nc] - r0, dg = rgba[nc + 1] - g0, db = rgba[nc + 2] - b0;
          const cw = lut[((dr * dr + dg * dg + db * db) * lutScale) | 0];
          const wgt = sw[(dy + radius) * size + (dx + radius)] * cw;
          sum += val[ni] * wgt;
          wsum += wgt;
        }
      }
      dst[y * w + x] = wsum > 1e-9 ? sum / wsum : val[y * w + x];
    }
  }
  return dst;
}

// Central-difference gradients of a float field. Returns {gx, gy}.
export function gradients(val, w, h) {
  const gx = new Float32Array(w * h);
  const gy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const ym = Math.max(y - 1, 0) * w, yp = Math.min(y + 1, h - 1) * w, row = y * w;
    for (let x = 0; x < w; x++) {
      const xm = Math.max(x - 1, 0), xp = Math.min(x + 1, w - 1);
      gx[row + x] = (val[row + xp] - val[row + xm]) * 0.5;
      gy[row + x] = (val[yp + x] - val[ym + x]) * 0.5;
    }
  }
  return { gx, gy };
}

// Binary dilate a Uint8 mask (nonzero = set) by `radius` using chamfer passes.
export function dilateMask(mask, w, h, radius) {
  // Two-pass distance-ish dilation: repeated 3x3 max, radius times.
  let cur = Uint8Array.from(mask);
  let nxt = new Uint8Array(w * h);
  for (let it = 0; it < radius; it++) {
    for (let y = 0; y < h; y++) {
      const ym = Math.max(y - 1, 0) * w, yp = Math.min(y + 1, h - 1) * w, row = y * w;
      for (let x = 0; x < w; x++) {
        const xm = Math.max(x - 1, 0), xp = Math.min(x + 1, w - 1);
        nxt[row + x] = (cur[row + x] | cur[row + xm] | cur[row + xp] |
          cur[ym + x] | cur[yp + x]) ? 1 : 0;
      }
    }
    [cur, nxt] = [nxt, cur];
  }
  return cur;
}

// Approximate percentile of a Float32Array via histogram (fast, good enough).
export function percentile(arr, p, bins = 1024) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!(hi > lo)) return lo;
  const hist = new Uint32Array(bins);
  const scale = (bins - 1) / (hi - lo);
  for (let i = 0; i < arr.length; i++) hist[((arr[i] - lo) * scale) | 0]++;
  const target = p * arr.length;
  let acc = 0;
  for (let b = 0; b < bins; b++) {
    acc += hist[b];
    if (acc >= target) return lo + b / scale;
  }
  return hi;
}

// Luminance of an RGBA pixel index (i = pixel index, not byte index)
export function luma(rgba, i) {
  const o = i * 4;
  return 0.299 * rgba[o] + 0.587 * rgba[o + 1] + 0.114 * rgba[o + 2];
}

// Float32Array -> IEEE 754 half floats (Uint16Array). Disparity lives in
// [0,1] where half precision (~1e-3 relative) is ample; R16F textures filter
// LINEARly in core WebGL2 on every platform (float32 linear needs an ext).
export function toHalfFloat(src) {
  const out = new Uint16Array(src.length);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let i = 0; i < src.length; i++) {
    f32[0] = src[i];
    const x = u32[0];
    const sign = (x >>> 16) & 0x8000;
    let exp = (x >>> 23) & 0xff;
    let mant = x & 0x7fffff;
    if (exp === 255) { out[i] = sign | 0x7c00 | (mant ? 1 : 0); continue; } // inf/nan
    let e = exp - 127 + 15;
    if (e >= 31) { out[i] = sign | 0x7c00; continue; }        // overflow -> inf
    if (e <= 0) {
      if (e < -10) { out[i] = sign; continue; }               // underflow -> 0
      mant = (mant | 0x800000) >> (1 - e);
      out[i] = sign | (mant >> 13);
      continue;
    }
    out[i] = sign | (e << 10) | (mant >> 13);
  }
  return out;
}

// 3x3 min-filter, iterated — erodes NEAR (large) disparity values so fg
// silhouettes shrink a hair and the photo's mixed-color rim lands on the
// background side (Immersity's edge-alignment trick).
/**
 * Per-channel sensor-noise sigma, by Immerkaer's method: convolve with
 *   [[1,-2,1],[-2,4,-2],[1,-2,1]]
 * which annihilates any locally-linear signal, so what survives is noise, then
 * take a ROBUST average of the magnitudes (the plain mean is dominated by
 * edges, which the kernel does not annihilate). Returns sigma in 0..1 units.
 *
 * We need this because generated regions have to carry the SAME grain as the
 * photograph. Grain is the single strongest tell: a region with no noise reads
 * as plastic instantly, long before anyone notices the geometry is invented.
 */
export function estimateNoiseSigma(rgba, w, h) {
  const out = [0, 0, 0];
  if (w < 5 || h < 5) return out;
  const n = (w - 2) * (h - 2);
  const mags = new Float32Array(n);
  for (let c = 0; c < 3; c++) {
    let k = 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = (y * w + x) * 4 + c;
        const v = 4 * rgba[i]
          - 2 * (rgba[i - 4] + rgba[i + 4] + rgba[i - w * 4] + rgba[i + w * 4])
          + rgba[i - w * 4 - 4] + rgba[i - w * 4 + 4]
          + rgba[i + w * 4 - 4] + rgba[i + w * 4 + 4];
        mags[k++] = Math.abs(v);
      }
    }
    // median, not mean: edges survive the kernel and would inflate the estimate
    const med = percentile(mags, 0.5);
    // |N(0,s)| has median 0.6745s; the kernel multiplies sigma by sqrt(36)=6
    out[c] = Math.min(med / 0.6745 / 6 / 255, 0.25);
  }
  return out;
}

export function erodeMaxima(field, w, h, iterations = 2) {
  let cur = Float32Array.from(field);
  for (let it = 0; it < iterations; it++) {
    const out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      const ym = Math.max(y - 1, 0) * w, yp = Math.min(y + 1, h - 1) * w, row = y * w;
      for (let x = 0; x < w; x++) {
        const xm = Math.max(x - 1, 0), xp = Math.min(x + 1, w - 1);
        out[row + x] = Math.min(
          cur[row + x], cur[row + xm], cur[row + xp], cur[ym + x], cur[yp + x]);
      }
    }
    cur = out;
  }
  return cur;
}
