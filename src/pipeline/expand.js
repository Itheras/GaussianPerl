// Scene expansion (M9): turn a rendered novel view into a new ANCHOR.
//
// Given what the renderer could show from a new viewpoint — colour, per-pixel
// confidence, and the scene depth it already knew — produce a COMPLETE
// synthetic photograph of that viewpoint plus geometry that agrees with the
// scene. That anchor is then permanent: walk away and back, and the invented
// content is still there, unchanged.
//
//   1. holes      = pixels no anchor could explain (confidence below CONF), grown
//   2. colour     = MI-GAN over those holes, low frequencies anchored to the
//                   marched frame so exposure/colour never drifts, grained
//   3. geometry   = Depth Anything on the COMPLETED frame, run through the same
//                   silhouette stack the photo gets, then robustly aligned to
//                   the reference disparity and stitched across the boundary
//   4. trust      = 1 where this anchor generated something, low elsewhere, so
//                   generated pixels fill gaps but never outrank the photograph
//
// Degrades all the way down: no fill model -> push-pull colour completion; no
// depth model -> push-pull geometry. Something is always returned.

import { normalizeDisparity, compressFarField } from './depthproc.js';
import { fgsSmooth, weightedMedianDepth, mergeFloaters, relocateEdges } from './depth-filter.js';
import { anchorToReference, collarGrow } from './fill-plan.js';
import {
  alignDisparity, holeMask, holeFraction, trustAlpha, packAnchorColor, pushPullFill,
  clampHolesToBackground, splitHolesByArea, nearSideMask, mirrorFillRows, borderComponents,
} from './novel-view.js';
import { resizeFloat, resizeRGBA, dilateMask } from '../util/imageops.js';

const ANCHOR_DEPTH_MAX_PIXELS = 480_000;
// bumped on every behavioural change so a stale worker module is detectable
export const EXPAND_VERSION = 'm11-final-1';

export function anchorDepthDims(w, h, maxPixels = ANCHOR_DEPTH_MAX_PIXELS) {
  const scale = Math.min(1, Math.sqrt(maxPixels / (w * h)));
  return { dw: Math.max(8, Math.round(w * scale)), dh: Math.max(8, Math.round(h * scale)) };
}

/**
 * Push-pull the known colour into the holes, then relax it — the offline fill,
 * and the seed MI-GAN sees. A few Jacobi sweeps constrained to the hole pixels
 * turn the pyramid's residual banding into a smooth gradient; a hole filled
 * this way should read as defocus, never as a mosaic.
 */
export function classicalComplete(rgba, holes, w, h, relax = 8) {
  const n = w * h;
  const known = new Uint8Array(n);
  for (let i = 0; i < n; i++) known[i] = holes[i] ? 0 : 1;
  const out = new Uint8ClampedArray(rgba);
  const chan = new Float32Array(n);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < n; i++) chan[i] = rgba[i * 4 + c];
    let filled = pushPullFill(chan, known, w, h);
    for (let it = 0; it < relax; it++) {
      const next = Float32Array.from(filled);
      for (let y = 0; y < h; y++) {
        const row = y * w;
        const up = Math.max(y - 1, 0) * w, dn = Math.min(y + 1, h - 1) * w;
        for (let x = 0; x < w; x++) {
          const i = row + x;
          if (known[i]) continue;
          const l = row + Math.max(x - 1, 0), r = row + Math.min(x + 1, w - 1);
          next[i] = (filled[l] + filled[r] + filled[up + x] + filled[dn + x]) * 0.25;
        }
      }
      filled = next;
    }
    for (let i = 0; i < n; i++) if (holes[i]) out[i * 4 + c] = filled[i];
  }
  for (let i = 0; i < n; i++) out[i * 4 + 3] = 255;
  return out;
}

/**
 * job: {rgba, conf, refDisp, w, h, dSub, params}
 *   rgba     marched colour of the novel view (holes hold the smeared fallback)
 *   conf     per-pixel confidence 0..1 from the renderer
 *   refDisp  scene disparity for this view (dSub / novel depth)
 * deps: {inpainter, depth, onProgress, shouldAbort}
 * Returns {color, disp, dw, dh, stats} or null when there is nothing to do.
 */
export async function expandView(job, deps = {}) {
  const { rgba, conf, refDisp, w, h } = job;
  const p = job.params || {};
  const confThreshold = p.confThreshold ?? 0.6;
  const onProgress = deps.onProgress || (() => {});
  const shouldAbort = deps.shouldAbort || (() => false);

  const frac = holeFraction(conf, confThreshold);
  if (frac < (p.minHoleFraction ?? 0.004)) return null;

  const growPx = p.dilatePx ?? Math.max(2, Math.round(Math.min(w, h) * 0.006));
  const seed = holeMask(conf, w, h, { threshold: confThreshold, dilate: growPx });
  // Grow the hole ALONG THE NEAR SURFACE into the occluder — never down the
  // cliff into the background. Two things depend on this:
  //  - the model must not see the occluder as context, or it continues the
  //    foreground straight into the gap it is meant to be hiding behind;
  //  - more importantly, the depth conditioning below must be anchored on the
  //    BACKGROUND side only. Condition a disocclusion on both sides and the
  //    residual field interpolates between foreground and background depth,
  //    giving the hole a gentle ramp instead of a step — and a ramp is a
  //    surface, so the colour smears across it, it looks locally plausible, no
  //    confidence term flags it, and the next anchor bakes the smear in. That
  //    is the exact mechanism by which these loops rot.
  const collarR = p.collarPx ?? Math.max(6, Math.round(Math.min(w, h) * 0.02));
  const holes = collarGrow(refDisp, seed, w, h, collarR, p.edgeDispJump ?? 0.055);

  // ---------------------------------------------------------------- colour
  let filled = null;
  let fillKind = 'classical';
  let fillBackend = null;
  let generated = null;   // pixels this anchor actually invented
  onProgress({ stage: 'fill', done: 0, total: 1 });

  // Seed the model with a smooth completion instead of a directional smear —
  // and draw that completion from the FAR side only. A seed that averages
  // the occluder's own pixels smears an arm into the gap, and a diffusion
  // prior started from it keeps the smear as a ghost limb. Computed outside
  // the try so the fallback path below can use it too.
  const nearSide = nearSideMask(refDisp, holes, w, h, {
    jump: p.edgeDispJump ?? 0.055,
    radius: Math.max(24, Math.round(Math.min(w, h) * 0.04)),
  });
  // The seed must not see the occluder AT ALL. Push-pull averages over a
  // pyramid whose reach grows with level, so a torso 40 px beyond a thin
  // fringe still dictates the colour of the sky band beside it — measured as
  // a brown strip (alpha 1.0, far-shell depth) that survived every other fix.
  const occluder = nearSideMask(refDisp, holes, w, h, {
    jump: p.edgeDispJump ?? 0.055,
    radius: Math.round(Math.min(w, h) * (p.occluderRadius ?? 0.6)),
  });
  const seedHoles = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) seedHoles[i] = (holes[i] || nearSide[i] || occluder[i]) ? 1 : 0;
  const seededAll = classicalComplete(rgba, seedHoles, w, h);
  // the occluder itself stays the photograph; only holes take the seed
  const seeded = new Uint8ClampedArray(rgba);
  for (let i = 0; i < w * h; i++) {
    if (!holes[i]) continue;
    seeded[i * 4] = seededAll[i * 4]; seeded[i * 4 + 1] = seededAll[i * 4 + 1];
    seeded[i * 4 + 2] = seededAll[i * 4 + 2]; seeded[i * 4 + 3] = 255;
  }

  if (deps.inpainter) {
    try {
      const budget = p.deviceClass === 'mobile'
        ? { interior: { maxBoxPx: 768, maxCalls: 4 }, ring: { maxBoxPx: 768, maxCalls: 0 } }
        : { interior: { maxBoxPx: 512, maxCalls: 8 }, ring: { maxBoxPx: 512, maxCalls: 0 } };
      // a diffusion prior cannot see pinholes; hand it the real holes only.
      // The model's mask includes the collar; what we KEEP is the true hole.
      let genHoles = holes;
      let keep = seed;
      let modelInput = seeded;
      if (deps.inpainter.backend === 'sidecar') {
        const minArea = Math.round(w * h * (p.minGenFraction ?? 0.0015));
        const split = splitHolesByArea(holes, w, h, minArea);
        // Only components that reach the frame edge go to the diffusion prior:
        // that is outpainting, which it does well. Interior components are
        // disocclusions — the background an occluder hid — and a diffusion
        // model given a thin hole beside a person paints the PERSON into it
        // (measured with every prompt, mask shape and strength tried: alpha
        // 1.0, dark skin tones, where the true background was sky). Those
        // keep the far-side classical continuation, which is what they are.
        const border = borderComponents(split.large, w, h, 2);
        // ...but at wider orbits the displaced silhouette MERGES with the
        // beyond-frame band into one edge-touching component, and the model
        // then paints the subject's continuation into the part beside them
        // (measured: alpha 1.0 skin tones at far depth, every prompt tried).
        // So the split is by DISTANCE FROM THE OCCLUDER: within this band of
        // the subject the far-side classical fill stands; only beyond it does
        // the model get to invent.
        // Default 0: measured on the Khomami portrait, the far-side classical
        // seed in this band is a flat sky-blue patch even below the horizon —
        // worse than the model's continuation. Left as a knob because the
        // band is where the remaining artifact lives (see SCRATCHPAD M11).
        const bandR = Math.round(Math.min(w, h) * (p.occluderBandFraction ?? 0));
        let near2 = bandR > 0 ? dilateMask(occluder, w, h, bandR >> 1) : new Uint8Array(w * h);
        if (bandR > 0) near2 = dilateMask(near2, w, h, bandR - (bandR >> 1));
        keep = new Uint8Array(w * h);
        for (let i = 0; i < w * h; i++) keep[i] = (seed[i] && border[i] && !near2[i]) ? 1 : 0;
        if (p.generateInterior) for (let i = 0; i < w * h; i++) keep[i] = (seed[i] && split.large[i]) ? 1 : 0;
        // A disocclusion is the occluder's SILHOUETTE, displaced: a
        // person-shaped gap beside a person. A diffusion model that can SEE
        // the person continues them into the gap (measured repeatedly, with a
        // rounded mask, a background-only negative prompt and lower
        // strength); MASKING the person instead invites a tall object into a
        // tall mask (a new man with the subject prompt, an acacia without).
        // What works: the occluder is REPLACED by far-side background in the
        // model's conditioning image and left UNMASKED — the model then sees
        // a quiet patch of grass and sky where the person stood, is asked to
        // paint only the true holes beside it, and has nothing to continue.
        // The subject itself is never touched: only `keep` is composited.
        const whole = occluder;
        const grow = Math.round(Math.min(w, h) * (p.maskGrowFraction ?? 0.03));
        let big = dilateMask(split.large, w, h, grow >> 1);
        big = dilateMask(big, w, h, grow - (grow >> 1));   // two passes: rounder
        genHoles = new Uint8Array(w * h);
        for (let i = 0; i < w * h; i++) {
          // the model's mask: what we keep, plus the collar hygiene around it
          genHoles[i] = (keep[i] || (holes[i] && border[i]) || (big[i] && border[i] && !whole[i])) ? 1 : 0;
        }
        if (p.generateInterior) for (let i = 0; i < w * h; i++) genHoles[i] = (holes[i] || (big[i] && !whole[i])) ? 1 : 0;
        const seedAll = new Uint8Array(w * h);
        for (let i = 0; i < w * h; i++) seedAll[i] = (genHoles[i] || whole[i] || nearSide[i]) ? 1 : 0;
        // Context replacement was tried both ways and both failed: a smooth
        // push-pull blob in the subject's place makes the model paint smooth
        // blobs into the holes, and a row-mirrored one paints streaks. The
        // subject therefore stays VISIBLE in the context. What stops the
        // model continuing them into the gap is the PROMPT: it must describe
        // the background, never the subject (the UI field says so), and the
        // negative prompt lists people. With that, the holes come out as
        // background. `seedAll`/`whole` remain for the seed so the hole's
        // initial colour never comes from the occluder.
        void mirrorFillRows; void seedAll;
        modelInput = seeded;
      }
      const res = await deps.inpainter.fill({
        rgba: modelInput, holes: genHoles, w, h,
        padPx: 0,
        budget,
        consumable: keep,
        shouldAbort,
        options: p.fillOptions || {},
        onProgress: ({ phase, done, total }) => onProgress({ stage: 'fill', phase, done, total }),
      });
      filled = res.filled;
      fillBackend = deps.inpainter.backend || 'wasm';
      // The collar was masked so the model could not lean on the occluder,
      // and its output there is discarded — but the INPUT there was the seed
      // blur, and that is what came back. Restore the render: the occluder's
      // rim is the photograph, not a smear of it.
      generated = new Uint8Array(keep);
      for (let i = 0; i < w * h; i++) {
        if (keep[i]) continue;
        if (seed[i]) {
          // interior disocclusion: the far-side continuation stands, and it
          // is invented content (trust 1.0), not a re-render
          filled[i * 4] = seeded[i * 4]; filled[i * 4 + 1] = seeded[i * 4 + 1];
          filled[i * 4 + 2] = seeded[i * 4 + 2]; filled[i * 4 + 3] = 255;
          generated[i] = 1;
          continue;
        }
        if (!(holes[i] || genHoles[i])) continue;
        filled[i * 4] = rgba[i * 4]; filled[i * 4 + 1] = rgba[i * 4 + 1];
        filled[i * 4 + 2] = rgba[i * 4 + 2]; filled[i * 4 + 3] = 255;
      }
      // Low-frequency anchoring exists for MI-GAN: a 512px GAN drifts in tone,
      // so its output is pulled toward the classical estimate at ~2% radius.
      // A diffusion prior conditioned on the WHOLE frame does not drift — and
      // anchoring it to a smooth seed at that radius erases every structure
      // coarser than the radius, i.e. exactly the people and objects it was
      // asked to invent. Skip it for the sidecar.
      if (fillBackend !== 'sidecar') {
        const anchorR = Math.max(12, Math.min(48, Math.round(Math.min(w, h) * 0.02)));
        anchorToReference(filled, seeded, res.genMask, w, h, anchorR);
      }
      // NO baked grain here — the composite pass applies it in screen space,
      // weighted by how synthetic each pixel is. Baking it into the texture
      // means the raymarch resamples it away as the camera moves.
      fillKind = 'ai';
    } catch (err) {
      if (err && err.name === 'AbortError') throw err;
      fillKind = 'classical';
      console.warn('novel-view fill failed:', err);
      filled = null;
      // a dead or hung sidecar must not quietly downgrade every view to a
      // blur: try the in-browser model before giving up on AI entirely
      if (deps.inpainter.backend === 'sidecar' && deps.fallback) {
        try {
          const fb = await deps.fallback();
          if (fb) {
            const res2 = await fb.fill({
              rgba: seeded, holes, w, h, padPx: 0,
              budget: { interior: { maxBoxPx: 512, maxCalls: 8 }, ring: { maxBoxPx: 512, maxCalls: 0 } },
              consumable: seed, shouldAbort,
              onProgress: ({ done, total }) => onProgress({ stage: 'fill', done, total }),
            });
            filled = res2.filled;
            fillBackend = fb.backend || 'wasm';
            fillKind = 'ai';
            const anchorR = Math.max(12, Math.min(48, Math.round(Math.min(w, h) * 0.02)));
            anchorToReference(filled, seeded, res2.genMask, w, h, anchorR);
            for (let i = 0; i < w * h; i++) {
              if (!holes[i] || seed[i]) continue;
              filled[i * 4] = rgba[i * 4]; filled[i * 4 + 1] = rgba[i * 4 + 1];
              filled[i * 4 + 2] = rgba[i * 4 + 2]; filled[i * 4 + 3] = 255;
            }
            generated = seed;
          }
        } catch (err2) {
          if (err2 && err2.name === 'AbortError') throw err2;
          console.warn('fallback fill failed too:', err2);
          filled = null;
        }
      }
    }
  }
  if (!filled) { filled = classicalComplete(rgba, holes, w, h); generated = seed; }
  if (shouldAbort()) return null;

  // -------------------------------------------------------------- geometry
  const { dw, dh } = anchorDepthDims(w, h, p.depthMaxPixels);
  const refD = resizeFloat(refDisp, w, h, dw, dh);
  const confD = resizeFloat(conf, w, h, dw, dh);
  const holesD = resizeFloat(Float32Array.from(holes), w, h, dw, dh);
  // The true holes, WITHOUT the collar. The collar is the occluder's own rim
  // — masked for the model's benefit, but geometrically it is the subject.
  // Treating it as a hole here excluded it from `known` and then clamped it
  // to BACKGROUND depth with the real holes, so every generated anchor
  // carried a collar-wide copy of the subject's rim at far depth: from any
  // other angle, a strip of their skin floating beside them.
  const seedD = resizeFloat(Float32Array.from(seed), w, h, dw, dh);
  const known = new Uint8Array(dw * dh);
  // A usable depth reference: known to the renderer AND outside the collar,
  // so the fit and the residual field never hear from the occluder.
  // The threshold MUST match what the renderer treats as a real candidate
  // (CONF_OK 0.55): an earlier generated anchor's invention is rendered at
  // ~0.7 after its pose-distance fade, and a stricter cut here left those
  // pixels "unknown" — so this anchor re-estimated their depth from the
  // completed frame, disagreed with the anchor that invented them, and the
  // two ghosted against each other ever after. Stretched walls are already
  // below 0.5 via the smear term, so this admits no garbage.
  for (let i = 0; i < dw * dh; i++) {
    known[i] = (confD[i] >= Math.max(confThreshold, 0.58) && seedD[i] < 0.35) ? 1 : 0;
  }
  void holesD;

  let est = null;
  let depthKind = 'pushpull';
  if (deps.depth) {
    onProgress({ stage: 'depth' });
    try {
      const raw = await deps.depth.estimate(filled, w, h, dw, dh);
      est = normalizeDisparity(raw);
      depthKind = 'ai';
    } catch (err) {
      console.warn('novel-view depth failed; using push-pull geometry:', err);
      est = null;
    }
  }
  if (shouldAbort()) return null;

  let disp;
  let fit = null;
  let clampStats = null;
  if (est) {
    const rgbaD = resizeRGBA(filled, w, h, dw, dh);
    let d = fgsSmooth(est, rgbaD, dw, dh, { lambda: 700, sigmaColor: 7, iterations: 3 });
    d = weightedMedianDepth(d, dw, dh, p.edgeDispJump ?? 0.055);
    const minArea = Math.round(20 * Math.pow(Math.min(dw, dh) / 384, 2));
    mergeFloaters(d, dw, dh, p.edgeDispJump ?? 0.055, Math.max(16, minArea));
    d = relocateEdges(d, rgbaD, dw, dh);
    d = compressFarField(d, p.farKnee ?? 0.16, p.farKeep ?? 0.25);
    const aligned = alignDisparity({ est: d, ref: refD, known, w: dw, h: dh });
    fit = aligned.fit;
    // A fit this bad means the estimate disagrees with the scene in a way no
    // scale+shift can reconcile — wrong focal length, or the model read the
    // completed frame as a different scene. Committing it would bake permanent
    // distortion into the anchor, so fall back to continuing the geometry we
    // already trust and keep only the generated COLOUR.
    // Loose on purpose. The residual field makes the result EXACT at known
    // pixels whatever the fit, so the gate only has to catch a fit that is
    // garbage (flipped, degenerate) — not a model whose disparity curve is
    // merely not affine to ours, which is every monocular model on every real
    // scene. At 0.05 it rejected all three anchors on the beach photo and
    // left every invented person as a flat billboard.
    const madLimit = (p.madLimit ?? 0.15) * (p.dispRange ?? 1);
    if (!Number.isFinite(fit.mad) || fit.mad > madLimit) {
      disp = pushPullFill(refD, known, dw, dh);
      depthKind = 'pushpull-rejected';
    } else {
      disp = aligned.disp;
    }
    // a disocclusion is background, whatever the depth model thinks it saw
    const holeD = new Uint8Array(dw * dh);
    for (let i = 0; i < dw * dh; i++) holeD[i] = seedD[i] >= 0.35 ? 1 : 0;
    const bg = clampHolesToBackground(disp, holeD, dw, dh, {
      jump: p.edgeDispJump ?? 0.055,
    });
    disp = bg.disp;
    clampStats = { clamped: bg.clamped, components: bg.components };
  } else {
    // no depth model: continue the known geometry into the holes
    disp = pushPullFill(refD, known, dw, dh);
  }

  // ----------------------------------------------------------------- trust
  // trust 1.0 means "this anchor invented this pixel" — the kept set, never
  // the collar, which is re-rendered photograph and gets `keep` like the rest
  const alpha = trustAlpha(generated || seed, w, h, {
    keep: p.keepTrust ?? 0.8,
    featherPx: p.featherPx ?? Math.max(3, Math.round(Math.min(w, h) * 0.01)),
  });
  const color = packAnchorColor(filled, alpha, w, h);

  let dMin = Infinity, dMax = -Infinity;
  for (let i = 0; i < disp.length; i++) {
    const v = disp[i];
    if (!Number.isFinite(v)) { disp[i] = 0.04; continue; }
    if (v < dMin) dMin = v;
    if (v > dMax) dMax = v;
  }
  if (!Number.isFinite(dMin)) { dMin = 0.04; dMax = 1; }

  return {
    color, disp, dw, dh,
    stats: {
      version: EXPAND_VERSION,
      holeFraction: frac, fillKind, fillBackend, depthKind, clamp: clampStats,
      fit: fit ? { a: fit.a, b: fit.b, mad: fit.mad, method: fit.method, inliers: fit.inliers } : null,
      dMin: Math.max(dMin - 0.02, 0.005), dMax: Math.min(dMax + 0.02, 4),
    },
  };
}
