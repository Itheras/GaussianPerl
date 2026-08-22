// Planning for generative (AI) fill — pure typed-array code, no DOM, no model:
//   - hole-mask construction: which pixels the inpainting model must synthesize
//     (disocclusion band + a foreground "collar" so the model never sees the
//     occluder as context and can't continue it into the hole),
//   - cluster boxes: partition holes into per-call masks (the MI-GAN pipeline
//     graph crops around each mask's bbox, so small masks keep native detail),
//   - replicate-padded plates + ring masks for border outpainting,
//   - NCHW uint8 tensor packing.
// Runs in the pipeline worker and in node tests.

import { dilateMask, boxBlurFloat } from '../util/imageops.js';

/**
 * Grow a mask from `seed` outward, but only into pixels whose disparity stays
 * within `drop` of the pixel they were reached from — i.e. along the same
 * (near) surface, never down a silhouette cliff into the background.
 * BFS limited to `radius` steps. Returns Uint8Array (includes the seeds).
 */
export function collarGrow(disp, seed, w, h, radius, drop) {
  const n = w * h;
  const dist = new Int16Array(n).fill(-1);
  // queue of pixel indices; head pointer instead of shift()
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
  for (let i = 0; i < n; i++) if (seed[i]) { dist[i] = 0; push(i); }
  while (qHead < qTail) {
    const i = queue[qHead++];
    const d = dist[i];
    if (d >= radius) continue;
    const x = i % w, y = (i / w) | 0;
    const lim = disp[i] - drop;
    let j;
    // 4-neighborhood, inlined
    if (x > 0 && dist[j = i - 1] < 0 && disp[j] >= lim) { dist[j] = d + 1; push(j); }
    if (x < w - 1 && dist[j = i + 1] < 0 && disp[j] >= lim) { dist[j] = d + 1; push(j); }
    if (y > 0 && dist[j = i - w] < 0 && disp[j] >= lim) { dist[j] = d + 1; push(j); }
    if (y < h - 1 && dist[j = i + w] < 0 && disp[j] >= lim) { dist[j] = d + 1; push(j); }
  }
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = dist[i] >= 0 ? 1 : 0;
  return out;
}

/**
 * Build the model's input for disocclusion fill.
 * holes = bgMask (pixels that get background splats)
 *       ∪ collar (foreground surface near silhouettes — masked so the model
 *         cannot use the occluder as context; its output there is discarded)
 *       ∪ 2px rim around fg boundary (mixed silhouette pixels are contaminated).
 * prefilled = image with classical fill colors at bgMask pixels, so holes seen
 * as *context by other per-cluster calls* look like plausible background.
 * Returns {holes, prefilled}.
 */
export function buildFillInput(rgba, disp, bg, fgB, w, h, opts = {}) {
  const jump = opts.jump ?? 0.055;
  // wide enough that the occluder sits outside the model's near context at the
  // graph's 512-crop scale (Shih dilates 5px at 960; GAN receptive fields want
  // more), narrow enough not to balloon the hole bbox — 2% of the short side
  const collarPx = opts.collarPx
    ?? Math.max(6, Math.min(48, Math.round(Math.min(w, h) * 0.02)));
  const collar = collarGrow(disp, fgB, w, h, collarPx, jump * 0.75);
  const rim = dilateMask(fgB, w, h, 2);
  const n = w * h;
  const holes = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    holes[i] = (bg.bgMask[i] | collar[i] | rim[i]) ? 1 : 0;
  }
  const prefilled = new Uint8ClampedArray(rgba);
  for (let i = 0; i < n; i++) {
    if (bg.bgMask[i]) {
      prefilled[i * 4] = bg.bgColor[i * 4];
      prefilled[i * 4 + 1] = bg.bgColor[i * 4 + 1];
      prefilled[i * 4 + 2] = bg.bgColor[i * 4 + 2];
      prefilled[i * 4 + 3] = 255;
    }
  }
  return { holes, prefilled, collarPx };
}

/**
 * Partition hole pixels into cluster boxes for per-call masks.
 * Grid-downsampled connected components (8-conn), nearby boxes merged, boxes
 * bigger than maxBoxPx split into overlapping tiles, empty tiles dropped.
 * If that yields more than maxCalls boxes, re-plan with a coarser maxBox —
 * fewer, softer fills beat a 30-second build.
 * Returns array of {x0, y0, x1, y1} (x1/y1 exclusive), scanline order.
 */
export function planClusters(holes, w, h, opts = {}) {
  const cell = opts.cellPx ?? 32;
  const overlap = opts.overlapPx ?? 96;
  const mergeGap = opts.mergeGapPx ?? 48;
  const maxCalls = opts.maxCalls ?? 6;
  let maxBox = opts.maxBoxPx ?? 512;

  for (;;) {
    const boxes = planOnce(holes, w, h, cell, maxBox, overlap, mergeGap);
    if (boxes.length <= maxCalls) return boxes;
    if (maxBox >= Math.max(w, h)) {
      // disjoint clusters survive any maxBox — last resort: one union bbox
      const u = boxes.reduce((a, b) => ({
        x0: Math.min(a.x0, b.x0), y0: Math.min(a.y0, b.y0),
        x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1),
      }));
      return [u];
    }
    maxBox = Math.round(maxBox * 1.6);
  }
}

function planOnce(holes, w, h, cell, maxBox, overlap, mergeGap) {
  const gw = Math.ceil(w / cell), gh = Math.ceil(h / cell);
  const grid = new Uint8Array(gw * gh);
  for (let y = 0; y < h; y++) {
    const row = y * w, grow = ((y / cell) | 0) * gw;
    for (let x = 0; x < w; x++) {
      if (holes[row + x]) grid[grow + ((x / cell) | 0)] = 1;
    }
  }
  // connected components on the cell grid (8-conn)
  const label = new Int32Array(gw * gh).fill(-1);
  const boxes = [];
  const stack = [];
  for (let i = 0; i < gw * gh; i++) {
    if (!grid[i] || label[i] >= 0) continue;
    const id = boxes.length;
    let bx0 = gw, by0 = gh, bx1 = -1, by1 = -1;
    stack.push(i);
    label[i] = id;
    while (stack.length) {
      const c = stack.pop();
      const cx = c % gw, cy = (c / gw) | 0;
      if (cx < bx0) bx0 = cx;
      if (cy < by0) by0 = cy;
      if (cx > bx1) bx1 = cx;
      if (cy > by1) by1 = cy;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = cx + dx, ny = cy + dy;
          if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
          const j = ny * gw + nx;
          if (grid[j] && label[j] < 0) { label[j] = id; stack.push(j); }
        }
      }
    }
    boxes.push({
      x0: bx0 * cell, y0: by0 * cell,
      x1: Math.min((bx1 + 1) * cell, w), y1: Math.min((by1 + 1) * cell, h),
    });
  }

  // merge boxes whose Chebyshev gap is below mergeGap (until stable)
  let merged = true;
  while (merged) {
    merged = false;
    outer:
    for (let a = 0; a < boxes.length; a++) {
      for (let b = a + 1; b < boxes.length; b++) {
        const A = boxes[a], B = boxes[b];
        const gapX = Math.max(A.x0, B.x0) - Math.min(A.x1, B.x1);
        const gapY = Math.max(A.y0, B.y0) - Math.min(A.y1, B.y1);
        if (Math.max(gapX, gapY) < mergeGap) {
          A.x0 = Math.min(A.x0, B.x0); A.y0 = Math.min(A.y0, B.y0);
          A.x1 = Math.max(A.x1, B.x1); A.y1 = Math.max(A.y1, B.y1);
          boxes.splice(b, 1);
          merged = true;
          break outer;
        }
      }
    }
  }

  // split oversized boxes into overlapping tiles; keep only tiles with holes
  const out = [];
  for (const B of boxes) {
    const tilesX = splitSpan(B.x0, B.x1, maxBox, overlap);
    const tilesY = splitSpan(B.y0, B.y1, maxBox, overlap);
    for (const [ty0, ty1] of tilesY) {
      for (const [tx0, tx1] of tilesX) {
        if (anyHole(holes, w, tx0, ty0, tx1, ty1)) {
          out.push({ x0: tx0, y0: ty0, x1: tx1, y1: ty1 });
        }
      }
    }
  }
  out.sort((a, b) => (a.y0 - b.y0) || (a.x0 - b.x0));
  return out;
}

function splitSpan(s0, s1, maxLen, overlap) {
  const len = s1 - s0;
  if (len <= maxLen) return [[s0, s1]];
  const step = Math.max(maxLen - overlap, 32);
  const count = Math.ceil((len - overlap) / step);
  const spans = [];
  for (let i = 0; i < count; i++) {
    const a = s0 + i * step;
    const b = Math.min(a + maxLen, s1);
    spans.push([Math.min(a, b - maxLen < s0 ? s0 : b - maxLen), b]);
    if (b >= s1) break;
  }
  return spans;
}

function anyHole(holes, w, x0, y0, x1, y1) {
  for (let y = y0; y < y1; y++) {
    const row = y * w;
    for (let x = x0; x < x1; x++) if (holes[row + x]) return true;
  }
  return false;
}

function mirrorIdx(v, size) {
  let m = v;
  if (m < 0) m = -m - 1;
  if (m >= size) m = 2 * size - 1 - m;
  return Math.min(Math.max(m, 0), size - 1);
}

/**
 * Mirror-pad an RGBA image by padPx on every side. Mirrored content is only an
 * INIT for the outpaint ring: the model erases hole pixels, but un-run ring
 * areas act as context for neighboring ring tiles (and as the visual fallback
 * if a ring call fails), and mirrored structure beats replicate streaks there.
 * Returns {plate, pw, ph}.
 */
export function padPlate(rgba, w, h, padPx) {
  const pw = w + 2 * padPx, ph = h + 2 * padPx;
  const plate = new Uint8ClampedArray(pw * ph * 4);
  for (let y = 0; y < ph; y++) {
    const sy = mirrorIdx(y - padPx, h);
    for (let x = 0; x < pw; x++) {
      const sx = mirrorIdx(x - padPx, w);
      const si = (sy * w + sx) * 4, di = (y * pw + x) * 4;
      plate[di] = rgba[si];
      plate[di + 1] = rgba[si + 1];
      plate[di + 2] = rgba[si + 2];
      plate[di + 3] = rgba[si + 3];
    }
  }
  return { plate, pw, ph };
}

/** Replicate-pad a float field (disparity) by padPx. */
export function padFloat(field, w, h, padPx) {
  const pw = w + 2 * padPx, ph = h + 2 * padPx;
  const out = new Float32Array(pw * ph);
  for (let y = 0; y < ph; y++) {
    const sy = Math.min(Math.max(y - padPx, 0), h - 1);
    for (let x = 0; x < pw; x++) {
      const sx = Math.min(Math.max(x - padPx, 0), w - 1);
      out[y * pw + x] = field[sy * w + sx];
    }
  }
  return out;
}

/** 1 on the padPx-wide border band of a pw×ph canvas. */
export function ringMask(pw, ph, padPx) {
  const out = new Uint8Array(pw * ph);
  for (let y = 0; y < ph; y++) {
    const border = y < padPx || y >= ph - padPx;
    const row = y * pw;
    if (border) { out.fill(1, row, row + pw); continue; }
    for (let x = 0; x < padPx; x++) out[row + x] = 1;
    for (let x = pw - padPx; x < pw; x++) out[row + x] = 1;
  }
  return out;
}

/**
 * Anchor the LOW frequencies of an AI fill to a trustworthy reference while
 * keeping the AI's high-frequency texture. GAN inpainting hallucinates objects
 * and tone shifts at scale (dark blobs in grass, invented structures behind a
 * horizon line); the classical fill has boringly-right color but smeared
 * texture. correction = maskedBlur(ref - ai); out = ai + correction on mask.
 * The blur statistics use ONLY mask pixels, so contaminated collar/foreground
 * colors around the mask never bleed into the anchor. Mutates `ai` in place.
 */
export function anchorToReference(ai, ref, mask, w, h, radius) {
  const n = w * h;
  const wgt = new Float32Array(n);
  for (let i = 0; i < n; i++) wgt[i] = mask[i] ? 1 : 0;
  const wBlur = boxBlurFloat(wgt, w, h, radius);
  const ch = new Float32Array(n);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < n; i++) {
      ch[i] = mask[i] ? (ref[i * 4 + c] - ai[i * 4 + c]) : 0;
    }
    const dBlur = boxBlurFloat(ch, w, h, radius);
    for (let i = 0; i < n; i++) {
      if (!mask[i] || wBlur[i] < 1e-6) continue;
      ai[i * 4 + c] = ai[i * 4 + c] + dBlur[i] / wBlur[i];
    }
  }
  return ai;
}

/**
 * Soften the depth field inside the outpaint ring: replicated disparity drags
 * the image's biggest depth cliffs (horizon line!) into the ring, where they
 * render as a staircase of floating slabs under yaw. Blend toward a blurred
 * field, ramping from exact at the interior boundary to fully blurred a few
 * pixels out — continuity where the skirt meets the image, calm beyond.
 * Mutates and returns plateDisp.
 */
export function smoothRingDisparity(plateDisp, pw, ph, padPx, radius) {
  const blurred = boxBlurFloat(plateDisp, pw, ph, radius);
  const ramp = Math.max(4, padPx >> 2);
  for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      // distance OUTSIDE the interior rect (0 = inside)
      const dx = Math.max(padPx - x, x - (pw - 1 - padPx), 0);
      const dy = Math.max(padPx - y, y - (ph - 1 - padPx), 0);
      const d = Math.max(dx, dy);
      if (d <= 0) continue;
      const t = Math.min(d / ramp, 1);
      const i = y * pw + x;
      plateDisp[i] = plateDisp[i] * (1 - t) + blurred[i] * t;
    }
  }
  return plateDisp;
}

/** RGBA interleaved -> uint8 NCHW [1,3,h,w] RGB. */
export function packImageNCHW(rgba, w, h) {
  const n = w * h;
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    out[i] = rgba[i * 4];
    out[n + i] = rgba[i * 4 + 1];
    out[2 * n + i] = rgba[i * 4 + 2];
  }
  return out;
}

/**
 * Mask tensor for one cluster call: 255 = known, 0 = hole — holes only inside
 * the given box. [1,1,h,w].
 */
export function packMaskForBox(holes, w, h, box) {
  const out = new Uint8Array(w * h).fill(255);
  for (let y = box.y0; y < box.y1; y++) {
    const row = y * w;
    for (let x = box.x0; x < box.x1; x++) {
      if (holes[row + x]) out[row + x] = 0;
    }
  }
  return out;
}

/**
 * uint8 NCHW [1,3,h,w] RGB -> RGBA interleaved. The model carries no alpha, so
 * pass the input RGBA as `srcRgba` to preserve transparency through the
 * round-trip (synthesized holes are already opaque in the prefilled input);
 * without it alpha is stamped 255.
 */
export function unpackNCHW(data, w, h, srcRgba = null) {
  const n = w * h;
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = data[i];
    out[i * 4 + 1] = data[n + i];
    out[i * 4 + 2] = data[2 * n + i];
    out[i * 4 + 3] = srcRgba ? srcRgba[i * 4 + 3] : 255;
  }
  return out;
}
