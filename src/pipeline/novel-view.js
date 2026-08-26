// Novel-view completion maths (M9) — pure, node-testable.
//
// The inpaint-and-lift loop: render a novel view, generatively fill the pixels
// no anchor could explain, estimate depth on the completed frame, then make
// that depth agree with the scene we already have. This module owns the
// "make it agree" half, which is where these systems normally fall apart.
//
// Three ideas do the work:
//  1. ALIGN in disparity space with a ROBUST affine fit (a*d_est + b ~ d_ref)
//     over the pixels the renderer already knew. Monocular depth is only
//     affine-invariant, so scale+shift is exactly the right family; robustness
//     matters because the reference contains stretched-wall garbage near
//     silhouettes.
//  2. Absorb the leftover disagreement as a RESIDUAL FIELD, extended into the
//     holes by push-pull interpolation. After that the aligned depth matches
//     the existing scene EXACTLY at known pixels and blends smoothly across the
//     hole boundary — no step, no floating slab.
//  3. Keep the reference depth verbatim where it is trustworthy and use the
//     model only inside the holes, with a feathered handover.

import { dilateMask, boxBlurFloat, percentile } from '../util/imageops.js';

/**
 * Robust affine fit  ref ~ a*est + b  over mask pixels.
 * Trimmed least squares: fit, drop the worst residuals, refit. Falls back to
 * percentile matching when the fit degenerates (too few points, flat input,
 * negative slope — all of which mean "do not trust this").
 * Returns {a, b, n, inliers, method}.
 */
export function robustAffine(est, ref, mask, opts = {}) {
  const maxPoints = opts.maxPoints ?? 40000;
  const rounds = opts.rounds ?? 3;
  const keep = opts.keep ?? 0.8;
  const bins = opts.bins ?? 16;
  const n = est.length;

  const idx = [];
  for (let i = 0; i < n; i++) {
    if (mask[i] && Number.isFinite(est[i]) && Number.isFinite(ref[i])) idx.push(i);
  }
  if (idx.length < 64) {
    return { a: 1, b: 0, n: idx.length, inliers: 0, mad: Infinity, method: 'identity' };
  }
  let pts = idx;
  if (pts.length > maxPoints) {
    const stride = Math.ceil(pts.length / maxPoints);
    pts = pts.filter((_, k) => k % stride === 0);
  }

  // De-bias the fit across depth. The known pixels are the NON-hole region,
  // which is heavily near- and mid-field — holes live at silhouettes and past
  // the frame — so an unweighted fit lets the near field set the scale and
  // mis-places the far field, exactly where depth disagreements are hardest to
  // resolve later. One vote per disparity bin fixes it.
  let lo = Infinity, hi = -Infinity;
  for (const i of pts) { if (est[i] < lo) lo = est[i]; if (est[i] > hi) hi = est[i]; }
  const span = Math.max(hi - lo, 1e-6);
  const counts = new Float64Array(bins);
  const binOf = (i) => Math.min(bins - 1, Math.max(0, Math.floor((est[i] - lo) / span * bins)));
  for (const i of pts) counts[binOf(i)]++;
  const wOf = (i) => 1 / Math.max(counts[binOf(i)], 1);

  let a = 1, b = 0, inliers = pts.length;
  let active = pts;
  for (let r = 0; r < rounds; r++) {
    let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    const m = active.length;
    if (m < 32) break;
    for (const i of active) {
      const w = wOf(i), x = est[i], y = ref[i];
      sw += w; sx += w * x; sy += w * y; sxx += w * x * x; sxy += w * x * y;
    }
    const den = sw * sxx - sx * sx;
    if (Math.abs(den) < 1e-12) break;
    a = (sw * sxy - sx * sy) / den;
    b = (sy - a * sx) / sw;
    inliers = m;
    if (r === rounds - 1) break;
    const res = active.map((i) => Math.abs(a * est[i] + b - ref[i]));
    const sorted = Float64Array.from(res).sort();
    const cut = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * keep))];
    const next = active.filter((i) => Math.abs(a * est[i] + b - ref[i]) <= cut + 1e-9);
    if (next.length < 32) break;
    active = next;
  }

  const madOf = () => {
    const res = pts.map((i) => Math.abs(a * est[i] + b - ref[i])).sort((x, y) => x - y);
    return res.length ? res[Math.floor(res.length * 0.5)] * 1.4826 : Infinity;
  };

  if (!Number.isFinite(a) || !Number.isFinite(b) || a <= 1e-4) {
    // percentile matching: preserves ordering even when the fit blows up
    const e = Float32Array.from(pts.map((i) => est[i]));
    const rf = Float32Array.from(pts.map((i) => ref[i]));
    const eLo = percentile(e, 0.1), eHi = percentile(e, 0.9);
    const rLo = percentile(rf, 0.1), rHi = percentile(rf, 0.9);
    const span = Math.max(eHi - eLo, 1e-6);
    a = (rHi - rLo) / span;
    if (!Number.isFinite(a) || a <= 1e-4) a = 1;
    b = percentile(rf, 0.5) - a * percentile(e, 0.5);
    return { a, b, n: pts.length, inliers: pts.length, mad: madOf(), method: 'percentile' };
  }
  return { a, b, n: pts.length, inliers, mad: madOf(), method: 'trimmed-lsq' };
}

/**
 * Push-pull interpolation (Gortler et al.): extend a sparse field over the
 * whole grid by averaging up a pyramid and filling down it. Cheap, stable,
 * and it relaxes to the global mean far from any known pixel — exactly the
 * behaviour we want for a residual field inside a big hole.
 */
export function pushPullFill(values, known, w, h) {
  const levels = [];
  let cw = w, ch = h;
  let v = new Float32Array(w * h);
  let m = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    if (known[i] && Number.isFinite(values[i])) { v[i] = values[i]; m[i] = 1; }
  }
  levels.push({ v, m, w: cw, h: ch });
  while (cw > 1 || ch > 1) {
    const nw = Math.max(1, cw >> 1), nh = Math.max(1, ch >> 1);
    const nv = new Float32Array(nw * nh);
    const nm = new Float32Array(nw * nh);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        let sv = 0, sm = 0;
        for (let dy = 0; dy < 2; dy++) {
          const sy = Math.min(y * 2 + dy, ch - 1);
          for (let dx = 0; dx < 2; dx++) {
            const sx = Math.min(x * 2 + dx, cw - 1);
            const si = sy * cw + sx;
            sv += v[si] * m[si];
            sm += m[si];
          }
        }
        const o = y * nw + x;
        nm[o] = Math.min(sm / 4, 1);
        nv[o] = sm > 0 ? sv / sm : 0;
      }
    }
    levels.push({ v: nv, m: nm, w: nw, h: nh });
    v = nv; m = nm; cw = nw; ch = nh;
  }

  // Pull with BILINEAR reconstruction, not nearest: a nearest pull stamps each
  // coarse texel over its 2x2 children and the blocks survive all the way up,
  // which is exactly the quilt of rectangles a naive push-pull fill is known
  // for. Bilinear costs three extra taps and removes them.
  for (let l = levels.length - 2; l >= 0; l--) {
    const fine = levels[l], coarse = levels[l + 1];
    for (let y = 0; y < fine.h; y++) {
      const cyf = (y + 0.5) * 0.5 - 0.5;
      const cy0 = Math.floor(cyf), ty = cyf - cy0;
      for (let x = 0; x < fine.w; x++) {
        const i = y * fine.w + x;
        if (fine.m[i] >= 1) continue;
        const cxf = (x + 0.5) * 0.5 - 0.5;
        const cx0 = Math.floor(cxf), tx = cxf - cx0;
        let sv = 0, sm = 0;
        for (let j = 0; j < 2; j++) {
          const yy = Math.min(Math.max(cy0 + j, 0), coarse.h - 1);
          const wy = j ? ty : 1 - ty;
          for (let k = 0; k < 2; k++) {
            const xx = Math.min(Math.max(cx0 + k, 0), coarse.w - 1);
            const ci = yy * coarse.w + xx;
            const w = (k ? tx : 1 - tx) * wy * coarse.m[ci];
            sv += coarse.v[ci] * w;
            sm += w;
          }
        }
        const cv = sm > 1e-6 ? sv / sm : 0;
        const cm = Math.min(sm, 1);
        const t = fine.m[i];
        fine.v[i] = fine.v[i] * t + cv * (1 - t);
        fine.m[i] = Math.max(t, cm);
      }
    }
  }
  return levels[0].v;
}

/**
 * Bring an estimated disparity map into the scene's frame.
 *   est   normalized model disparity of the completed frame (0..1)
 *   ref   the renderer's disparity for this view (dSub / novel depth)
 *   known 1 where ref is trustworthy
 * Returns {disp, fit} — disp equals ref at known pixels and continues it
 * smoothly through the holes.
 */
export function alignDisparity({ est, ref, known, w, h, smoothPx = 0 }) {
  const n = w * h;
  const fit = robustAffine(est, ref, known);
  const scaled = new Float32Array(n);
  for (let i = 0; i < n; i++) scaled[i] = fit.a * est[i] + fit.b;

  const residual = new Float32Array(n);
  for (let i = 0; i < n; i++) residual[i] = known[i] ? ref[i] - scaled[i] : 0;
  let field = pushPullFill(residual, known, w, h);
  if (smoothPx > 0) field = boxBlurFloat(field, w, h, smoothPx);

  const disp = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = scaled[i] + field[i];
    disp[i] = known[i] ? ref[i] : (Number.isFinite(v) ? v : ref[i]);
  }
  return { disp, fit };
}

/**
 * Pixels no anchor could explain. `conf` is the renderer's per-pixel
 * confidence; `dilate` grows the mask so the generator repaints the smeared
 * rim around each hole instead of blending against it.
 */
export function holeMask(conf, w, h, opts = {}) {
  const threshold = opts.threshold ?? 0.6;
  const n = w * h;
  const m = new Uint8Array(n);
  for (let i = 0; i < n; i++) m[i] = conf[i] < threshold ? 1 : 0;
  const grow = opts.dilate ?? 0;
  return grow > 0 ? dilateMask(m, w, h, grow) : m;
}

/** Fraction of the frame that is a hole (the trigger signal). */
export function holeFraction(conf, threshold = 0.6) {
  let n = 0;
  for (let i = 0; i < conf.length; i++) if (conf[i] < threshold) n++;
  return n / conf.length;
}

/**
 * Hole fraction inside a sub-rectangle, as fractions of the frame. Anchors are
 * captured wider than the visible frame, so the TRIGGER must look only at what
 * the viewer can actually see — otherwise the permanently-unknown border ring
 * keeps demanding generation passes nobody asked for.
 */
export function holeFractionRect(conf, w, h, inset, threshold = 0.6) {
  const x0 = Math.max(0, Math.round(w * inset));
  const x1 = Math.min(w, Math.round(w * (1 - inset)));
  const y0 = Math.max(0, Math.round(h * inset));
  const y1 = Math.min(h, Math.round(h * (1 - inset)));
  if (x1 <= x0 || y1 <= y0) return holeFraction(conf, threshold);
  let n = 0, total = 0;
  for (let y = y0; y < y1; y++) {
    const row = y * w;
    for (let x = x0; x < x1; x++) {
      total++;
      if (conf[row + x] < threshold) n++;
    }
  }
  return total ? n / total : 0;
}

/**
 * Per-texel VALIDITY for a generated anchor's colour texture (its alpha).
 * 1.0 where this anchor actually generated something; `keep` — still above the
 * shader's CONF_OK — for re-rendered pixels other anchors already own.
 *
 * Re-rendered pixels are real geometry and must be allowed to occlude: walk
 * around a pillar and the pillar's near face in the new anchor IS re-render
 * content. Parking them below CONF_OK would both forbid that and mark them as
 * holes forever, so every pass would regenerate content that is already right.
 * Colour ownership is a separate matter and lives in the shader's prio term.
 * Feathered at the frame border so anchors show no rectangular seam.
 */
export function trustAlpha(holes, w, h, opts = {}) {
  const keep = opts.keep ?? 0.8;
  const feather = opts.featherPx ?? 6;
  const borderPx = opts.borderPx ?? Math.max(4, Math.round(Math.min(w, h) * 0.02));
  const n = w * h;
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = holes[i] ? 1 : keep;
  const soft = feather > 0 ? boxBlurFloat(a, w, h, feather) : a;
  const out = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    const dy = Math.min(y, h - 1 - y);
    for (let x = 0; x < w; x++) {
      const dx = Math.min(x, w - 1 - x);
      const edge = Math.min(1, Math.min(dx, dy) / borderPx);
      const i = y * w + x;
      out[i] = Math.min(1, Math.max(keep * edge, soft[i] * edge));
    }
  }
  return out;
}

/**
 * Force a disocclusion to sit at BACKGROUND depth.
 *
 * This is the fix for the artifact that dominates everything else. When the
 * camera moves, the band revealed beside a foreground object is filled with
 * invented colour, and then a monocular depth model is asked what depth that
 * invention is at. It sees texture that continues the occluder and answers
 * "somewhere between the person and the wall" — a RAMP. A ramp is a surface,
 * so the renderer dutifully stretches colour across it, the result looks
 * locally plausible, no confidence term flags it, and the next anchor bakes it
 * in. That is the rubber-sheet smear, and no amount of blending fixes it
 * because the geometry itself is wrong.
 *
 * A disocclusion is, by definition, background. So: label each hole, look at
 * the depths on its rim, and if that rim straddles a real depth cliff (near on
 * one side, far on the other — which is exactly what makes it a disocclusion
 * rather than a plain gap), clamp the hole to the FAR side. Components whose
 * rim is all at one depth — a frame-border extension, a small texture hole —
 * are left alone, because for them there is no cliff and the model's answer is
 * as good as any.
 */
export function clampHolesToBackground(disp, holes, w, h, opts = {}) {
  const jump = opts.jump ?? 0.055;
  const margin = opts.margin ?? 0.02;
  const lowPct = opts.lowPercentile ?? 0.3;
  const n = w * h;
  const seen = new Uint8Array(n);
  let queue = new Int32Array(Math.max(256, n >> 3));
  const out = Float32Array.from(disp);
  let clamped = 0, components = 0;

  for (let start = 0; start < n; start++) {
    if (!holes[start] || seen[start]) continue;
    // flood the component, collecting its pixels and its known-side rim
    let qHead = 0, qTail = 0;
    const push = (i) => {
      if (qTail === queue.length) {
        const bigger = new Int32Array(queue.length * 2);
        bigger.set(queue);
        queue = bigger;
      }
      queue[qTail++] = i;
    };
    seen[start] = 1;
    push(start);
    const cells = [];
    const rim = [];
    while (qHead < qTail) {
      const i = queue[qHead++];
      cells.push(i);
      const x = i % w, y = (i / w) | 0;
      const visit = (j) => {
        if (holes[j]) {
          if (!seen[j]) { seen[j] = 1; push(j); }
        } else if (Number.isFinite(disp[j])) {
          rim.push(disp[j]);
        }
      };
      if (x > 0) visit(i - 1);
      if (x < w - 1) visit(i + 1);
      if (y > 0) visit(i - w);
      if (y < h - 1) visit(i + w);
    }
    if (rim.length < 24 || cells.length < 8) continue;
    components++;
    const r = Float32Array.from(rim);
    const lo = percentile(r, lowPct);
    const hi = percentile(r, 0.9);
    // no cliff on the rim => not a disocclusion => leave the model alone
    if (hi - lo < jump) continue;
    const limit = lo + margin;
    for (const i of cells) {
      if (out[i] > limit) { out[i] = limit; clamped++; }
    }
  }
  return { disp: out, clamped, components };
}

/**
 * Split a hole mask by connected-component area. A diffusion inpainter works
 * at an 8x-downsampled latent on a frame we have already shrunk to ~1 MP, so
 * a hole a few pixels wide is invisible to it and comes back as a smudge of
 * the initialisation colour — while a classical fill handles exactly those
 * perfectly. Only the LARGE components are worth a generator call.
 * Returns {large, small} masks (Uint8Array) and the component count.
 */
export function splitHolesByArea(holes, w, h, minArea) {
  const n = w * h;
  const seen = new Uint8Array(n);
  const large = new Uint8Array(n);
  const small = new Uint8Array(n);
  let queue = new Int32Array(Math.max(256, n >> 3));
  let components = 0;
  for (let start = 0; start < n; start++) {
    if (!holes[start] || seen[start]) continue;
    let qHead = 0, qTail = 0;
    const push = (i) => {
      if (qTail === queue.length) {
        const bigger = new Int32Array(queue.length * 2);
        bigger.set(queue);
        queue = bigger;
      }
      queue[qTail++] = i;
    };
    seen[start] = 1;
    push(start);
    while (qHead < qTail) {
      const i = queue[qHead++];
      const x = i % w, y = (i / w) | 0;
      const visit = (j) => { if (holes[j] && !seen[j]) { seen[j] = 1; push(j); } };
      if (x > 0) visit(i - 1);
      if (x < w - 1) visit(i + 1);
      if (y > 0) visit(i - w);
      if (y < h - 1) visit(i + w);
    }
    components++;
    const dst = qTail >= minArea ? large : small;
    for (let k = 0; k < qTail; k++) dst[queue[k]] = 1;
  }
  return { large, small, components };
}

/**
 * Pixels on the NEAR side of a disocclusion: the occluder's own surface, in
 * a band around each hole. A hole beside a person is revealed BACKGROUND, and
 * anything that seeds or fills it must draw from the far side only — a seed
 * averaged over all neighbours smears the arm into the gap, and a diffusion
 * model started from that seed keeps the smear as a ghost limb. For each hole
 * component the rim's low disparity sets the background level; within
 * `radius` of the component, known pixels standing `jump` nearer than that
 * level are the occluder. Returns a Uint8Array mask of those pixels.
 */
export function nearSideMask(disp, holes, w, h, opts = {}) {
  const jump = opts.jump ?? 0.055;
  const radius = opts.radius ?? 24;
  // The hole's rim is PART occluder: the collar puts the mask's inner edge
  // inside the subject, so a thin disocclusion's rim can be half skin. A
  // 30th-percentile "background level" then landed ON the subject, nothing
  // counted as nearer, and the seed was sourced from his body. Take the far
  // end of the rim distribution, and never accept a level nearer than the
  // frame's own median as "background".
  const lowPct = opts.lowPercentile ?? 0.1;
  const frameMedian = percentile(disp, 0.5);
  const n = w * h;
  const label = new Int32Array(n).fill(-1);
  const limits = [];
  let queue = new Int32Array(Math.max(256, n >> 3));
  const push = (i) => {
    if (qTail === queue.length) {
      const bigger = new Int32Array(queue.length * 2);
      bigger.set(queue);
      queue = bigger;
    }
    queue[qTail++] = i;
  };
  let qHead = 0, qTail = 0;

  // 1. label components, record each rim's background level
  for (let start = 0; start < n; start++) {
    if (!holes[start] || label[start] >= 0) continue;
    const id = limits.length;
    qHead = 0; qTail = 0;
    label[start] = id;
    push(start);
    const rim = [];
    while (qHead < qTail) {
      const i = queue[qHead++];
      const x = i % w, y = (i / w) | 0;
      const visit = (j) => {
        if (holes[j]) { if (label[j] < 0) { label[j] = id; push(j); } }
        else if (Number.isFinite(disp[j])) rim.push(disp[j]);
      };
      if (x > 0) visit(i - 1);
      if (x < w - 1) visit(i + 1);
      if (y > 0) visit(i - w);
      if (y < h - 1) visit(i + w);
    }
    let lim = rim.length >= 8 ? percentile(Float32Array.from(rim), lowPct) : Infinity;
    if (lim > frameMedian) lim = frameMedian;
    limits.push(lim);
  }

  // 2. grow outward from every hole by `radius`, carrying the component id;
  //    a known pixel standing clearly nearer than its component's background
  //    level is the occluder
  const near = new Uint8Array(n);
  const dist = new Int16Array(n).fill(-1);
  qHead = 0; qTail = 0;
  for (let i = 0; i < n; i++) if (holes[i]) { dist[i] = 0; push(i); }
  while (qHead < qTail) {
    const i = queue[qHead++];
    const d = dist[i];
    if (d >= radius) continue;
    const x = i % w, y = (i / w) | 0;
    const lim = limits[label[i]];
    const visit = (j) => {
      if (dist[j] >= 0) return;
      dist[j] = d + 1;
      label[j] = label[i];
      if (!holes[j] && disp[j] > lim + jump) near[j] = 1;
      push(j);
    };
    if (x > 0) visit(i - 1);
    if (x < w - 1) visit(i + 1);
    if (y > 0) visit(i - w);
    if (y < h - 1) visit(i + w);
  }
  return near;
}

/**
 * Row-wise mirror fill: replace masked pixels with a mirrored copy of the
 * nearest KNOWN pixels on the same row, blending the two sides by distance.
 * A standing subject against a landscape hides background that is, row by
 * row, much like what is visible beside them — sky stays sky, the horizon
 * stays level, grass stays grass. Where a smooth push-pull fill gives a
 * diffusion model a featureless blob to match (and it dutifully paints
 * featureless blobs), this gives it plausible texture to continue.
 */
export function mirrorFillRows(rgba, mask, w, h) {
  const out = new Uint8ClampedArray(rgba);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let x = 0;
    while (x < w) {
      if (!mask[row + x]) { x++; continue; }
      let x1 = x;
      while (x1 < w && mask[row + x1]) x1++;
      const left = x - 1, right = x1;            // nearest known on each side
      const hasL = left >= 0, hasR = right < w;
      if (!hasL && !hasR) { x = x1; continue; }
      for (let xx = x; xx < x1; xx++) {
        const dl = xx - left, dr = right - xx;
        let r = 0, g = 0, b = 0, wsum = 0;
        if (hasL) {
          const src = Math.max(left - dl, 0);            // mirror about `left`
          const k = (row + src) * 4, wl = hasR ? dr : 1;
          r += rgba[k] * wl; g += rgba[k + 1] * wl; b += rgba[k + 2] * wl; wsum += wl;
        }
        if (hasR) {
          const src = Math.min(right + dr, w - 1);       // mirror about `right`
          const k = (row + src) * 4, wr = hasL ? dl : 1;
          r += rgba[k] * wr; g += rgba[k + 1] * wr; b += rgba[k + 2] * wr; wsum += wr;
        }
        const o = (row + xx) * 4;
        out[o] = r / wsum; out[o + 1] = g / wsum; out[o + 2] = b / wsum; out[o + 3] = 255;
      }
      x = x1;
    }
  }
  return out;
}

/**
 * Mask of the hole components that touch the frame border — the beyond-frame
 * band that needs OUTPAINTING. Interior components are disocclusions: the
 * background the occluder hid, which is continuation, not invention.
 */
export function borderComponents(holes, w, h, margin = 2) {
  const n = w * h;
  const seen = new Uint8Array(n);
  const out = new Uint8Array(n);
  let queue = new Int32Array(Math.max(256, n >> 3));
  for (let start = 0; start < n; start++) {
    if (!holes[start] || seen[start]) continue;
    let qHead = 0, qTail = 0;
    const push = (i) => {
      if (qTail === queue.length) {
        const bigger = new Int32Array(queue.length * 2);
        bigger.set(queue);
        queue = bigger;
      }
      queue[qTail++] = i;
    };
    seen[start] = 1;
    push(start);
    let touches = false;
    while (qHead < qTail) {
      const i = queue[qHead++];
      const x = i % w, y = (i / w) | 0;
      if (x < margin || y < margin || x >= w - margin || y >= h - margin) touches = true;
      const visit = (j) => { if (holes[j] && !seen[j]) { seen[j] = 1; push(j); } };
      if (x > 0) visit(i - 1);
      if (x < w - 1) visit(i + 1);
      if (y > 0) visit(i - w);
      if (y < h - 1) visit(i + w);
    }
    if (touches) for (let k = 0; k < qTail; k++) out[queue[k]] = 1;
  }
  return out;
}

/** Novel-frame depth -> scene disparity (the mapping the renderer uses). */
export function depthToDisparity(depth, dSub, dFloor = 0.04, out = null) {
  const n = depth.length;
  const d = out || new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const z = depth[i];
    d[i] = (z > 1e-4 && Number.isFinite(z)) ? Math.min(dSub / z, 4) : dFloor;
  }
  return d;
}

/** Pack rgb + trust alpha into an RGBA texture payload. */
export function packAnchorColor(rgba, alpha, w, h) {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    out[i * 4] = rgba[i * 4];
    out[i * 4 + 1] = rgba[i * 4 + 1];
    out[i * 4 + 2] = rgba[i * 4 + 2];
    out[i * 4 + 3] = Math.round(Math.min(1, Math.max(0, alpha[i])) * 255);
  }
  return out;
}
