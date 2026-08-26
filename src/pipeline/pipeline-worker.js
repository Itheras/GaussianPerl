// Pipeline worker (M8): image (+ optional GT disparity) -> two-layer LDI
// heightfield textures, off the main thread. Owns BOTH AI models (depth via
// transformers.js, generative fill via onnxruntime-web).
//
// Resolutions: COLOR stays at working res (photo-native sharpness — the
// renderer taps it directly); DEPTH runs at <=~1.75MP end to end (it is a
// smooth field; every heavy filter stage stays bounded at any quality tier).
//
// Build flow per message:
//   depth (GT | AI | heuristic, at depth res) -> normalize -> FGS interior
//   smoothing -> weighted-median edge snapping -> floater merge -> gated edge
//   relocation -> far-field compression -> classical background synth ->
//   [preview 'built' ships: photo + classical fill layers] -> MI-GAN fills +
//   outpaint ring -> final 'built' (AI-colored layers).
// A newer 'build' message abandons the older continuation (buildGen guard).

import {
  normalizeDisparity, heuristicDisparity, fgBoundary, compressFarField,
} from './depthproc.js';
import {
  fgsSmooth, weightedMedianDepth, mergeFloaters, relocateEdges,
} from './depth-filter.js';
import { synthesizeBackground, addGrain } from './inpaint.js';
import { buildFillInput, anchorToReference } from './fill-plan.js';
import { buildLayers, upsampleBackgroundTo } from './layer-build.js';
import { resizeFloat, resizeRGBA, percentile, estimateNoiseSigma } from '../util/imageops.js';
import { DepthEstimator } from './depth-ai.js';
import { Inpainter } from './inpaint-ai.js';
import { expandView } from './expand.js';
import { NativeInpainter } from '../backend/native-fill.js';

const DEPTH_MAX_PIXELS = 1_750_000;

// Stage cache keyed by sourceId: depth, filtered disparity, classical bg and
// the AI fill all survive parameter-only rebuilds.
let stageCache = null;

let buildGen = 0;
let depthDlSink = null;
let fillDlSink = null;

let expandGen = 0;

// The guesser. A local sidecar (4B diffusion prior) if one is reachable, else
// the in-browser MI-GAN. Probed per call: the sidecar may start after the
// page did, and a dead one must degrade instantly, not stall the build.
async function chooseInpainter(params, onDownload) {
  if (params.sidecar) {
    const native = await NativeInpainter.probe(params.sidecar);
    if (native) return native;
  }
  if (!params.aiFill) return null;
  return Inpainter.load(onDownload, { forceWasm: params.forceWasmFill || params.webkitHint });
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'build') {
    const gen = ++buildGen;
    expandGen++; // a rebuild invalidates every in-flight expansion
    build(msg, gen).catch((err) => {
      if (gen === buildGen) {
        self.postMessage({ type: 'error', id: msg.id, message: String(err && err.stack || err) });
      }
    });
    return;
  }
  if (msg.type === 'expand') {
    const gen = ++expandGen;
    const buildAt = buildGen;
    expand(msg, gen, buildAt).catch((err) => {
      if (err && err.name === 'AbortError') return;
      if (gen === expandGen) {
        self.postMessage({ type: 'expand-failed', id: msg.id, message: String(err && err.message || err) });
      }
    });
  }
};

/**
 * Grow the scene: a rendered novel view (colour + confidence + the depth the
 * renderer already knew) becomes a fully generated anchor. Runs on the same
 * two models the build uses, so the weights are already resident.
 */
async function expand(msg, gen, buildAt) {
  const { id, w, h, params = {} } = msg;
  const stale = () => gen !== expandGen || buildAt !== buildGen;
  const post = (stage, extra) => self.postMessage({ type: 'expand-progress', id, stage, ...extra });

  const inpainter = (params.aiFill || params.sidecar)
    ? await chooseInpainter(params, () => post('fill-download'))
    : null;
  if (stale()) return;
  const depth = params.wantAiDepth
    ? await DepthEstimator.load(() => post('depth-download'), {
      deviceClass: params.deviceClass, forceWasm: params.webkitHint,
    })
    : null;
  if (stale()) return;

  const out = await expandView({
    rgba: new Uint8ClampedArray(msg.rgba),
    conf: new Float32Array(msg.conf),
    refDisp: new Float32Array(msg.refDisp),
    w, h, params,
  }, {
    inpainter, depth,
    fallback: params.aiFill
      ? () => Inpainter.load(() => post('fill-download'), { forceWasm: params.forceWasmFill || params.webkitHint })
      : null,
    onProgress: (pr) => { if (!stale()) post(pr.stage, pr); },
    shouldAbort: stale,
  });

  if (stale()) return;
  if (!out) { self.postMessage({ type: 'expand-skipped', id }); return; }
  self.postMessage({
    type: 'expanded', id, anchorId: msg.anchorId,
    color: out.color, disp: out.disp, dw: out.dw, dh: out.dh, stats: out.stats,
  }, [out.color.buffer, out.disp.buffer]);
}

function depthDims(w, h) {
  // never upscale, never distort aspect (a per-axis floor would)
  const scale = Math.min(1, Math.sqrt(DEPTH_MAX_PIXELS / (w * h)));
  return { dw: Math.max(1, Math.round(w * scale)), dh: Math.max(1, Math.round(h * scale)) };
}

function upsampleMask(m, dw, dh, w, h) {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = Math.min((y * dh / h) | 0, dh - 1);
    for (let x = 0; x < w; x++) {
      const sx = Math.min((x * dw / w) | 0, dw - 1);
      out[y * w + x] = m[sy * dw + sx];
    }
  }
  return out;
}

// The subject anchor d_s: Z_subject = 1 is the ORBIT PIVOT, so this decides
// what the camera moves around. The centre-box median is wrong whenever the
// subject is off-centre: on a photo of two people standing at the right edge
// in front of a wave pool it picked the WATER (0.14), leaving the couple seven
// times nearer than the pivot — a 5-degree orbit swept them 64% of the frame
// and clean off screen, which looked exactly like "the fill destroyed the
// people". People in photos are the NEAR thing: take the nearest 15% of the
// frame (floor band at the camera's feet excluded) and pivot on its median
// when it stands well in front of the frame's typical depth; else the old rule.
export function subjectDisparity(disp, w, h) {
  const yFloor = Math.round(h * 0.92);
  const all = new Float32Array(w * yFloor);
  let n = 0;
  for (let y = 0; y < yFloor; y++) for (let x = 0; x < w; x++) all[n++] = disp[y * w + x];
  // the near 15% of the frame: a person-sized subject is often well under a
  // quarter of the pixels, and a looser cut lets the background vote
  const p85 = percentile(all, 0.85);
  const near = [];
  for (let i = 0; i < n; i++) if (all[i] >= p85) near.push(all[i]);
  const nearMed = percentile(Float32Array.from(near), 0.5);
  const x0 = w >> 2, x1 = w - (w >> 2), y0 = h >> 2, y1 = h - (h >> 2);
  const box = new Float32Array((x1 - x0) * (y1 - y0));
  let m = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) box[m++] = disp[y * w + x];
  const centre = percentile(box, 0.5);
  // a real near subject stands well in front of the frame's typical depth;
  // a flat scene (everything at one distance) keeps the centre rule
  return nearMed > centre * 1.6 ? nearMed : centre;
}

async function build(msg, gen) {
  const { id, w, h, params } = msg;
  const rgba = new Uint8ClampedArray(msg.rgba);
  const post = (stage, frac, extra) =>
    self.postMessage({ type: 'progress', id, stage, frac, ...extra });
  const superseded = () => gen !== buildGen;

  const jump = params.edgeDispJump;
  const { dw, dh } = depthDims(w, h);
  const cacheHit = stageCache && msg.sourceId !== undefined &&
    stageCache.sourceId === msg.sourceId &&
    stageCache.w === w && stageCache.h === h;

  // ---------- depth (at depth res) ----------
  let dispD, rgbaD, depthMeta;
  if (cacheHit) {
    dispD = stageCache.dispD;
    rgbaD = stageCache.rgbaD;
    depthMeta = stageCache.depthMeta;
  } else {
    rgbaD = resizeRGBA(rgba, w, h, dw, dh);
    let raw = msg.disparity
      ? resizeFloat(new Float32Array(msg.disparity), w, h, dw, dh)
      : null;
    depthMeta = { depthKind: raw ? 'gt' : 'heuristic', depthBackend: null, depthTier: null };
    if (!raw && params.wantAiDepth) {
      depthDlSink = (p) => {
        if (!superseded() && p.phase === 'download') post('depth-download', 0.02, { pct: p.pct });
      };
      const est = await DepthEstimator.load((p) => depthDlSink && depthDlSink(p), {
        deviceClass: params.deviceClass,
        forceWasm: params.webkitHint,
        hq: params.hq,
      });
      if (superseded()) return;
      if (est) {
        post('depth', 0.05);
        try {
          raw = await est.estimate(rgba, w, h, dw, dh);
          depthMeta = { depthKind: 'ai', depthBackend: est.backend, depthTier: est.tier };
        } catch (err) {
          console.warn('depth inference failed; falling back to heuristic:', err);
          raw = null;
        }
        if (superseded()) return;
      }
    }

    post('normalize', 0.12);
    let d = raw ? normalizeDisparity(raw) : heuristicDisparity(rgbaD, dw, dh);
    if (depthMeta.depthKind === 'ai') {
      // production depth stack: interior smoothing WITHOUT texture imprint,
      // silhouette ramps -> steps, floater debris merged, edges relocated
      // onto image edges
      post('refine', 0.18);
      d = fgsSmooth(d, rgbaD, dw, dh, { lambda: 900, sigmaColor: 7, iterations: 3 });
      post('edges', 0.26);
      d = weightedMedianDepth(d, dw, dh, jump);
      const minArea = Math.round(20 * Math.pow(Math.min(dw, dh) / 384, 2));
      mergeFloaters(d, dw, dh, jump, Math.max(20, minArea));
      d = relocateEdges(d, rgbaD, dw, dh);
    } else {
      post('edges', 0.26);
      d = weightedMedianDepth(d, dw, dh, jump);
    }
    // real far fields barely parallax — flatten the tail so the horizon is a
    // backdrop, not a giant silhouette
    dispD = compressFarField(d, params.farKnee ?? 0.16, params.farKeep ?? 0.25);
    stageCache = {
      sourceId: msg.sourceId, w, h, dw, dh, dispD, rgbaD,
      bg: null, depthMeta, fill: null, fillFailed: false,
    };
  }

  // padded ring (working-res px); depth-res pad keeps the same fraction
  const padPx = params.withSkirt
    ? (params.skirtPx || Math.max(16, Math.min(192, Math.round(Math.min(w, h) * 0.1))))
    : 0;
  const padD = Math.round(padPx * dw / w);

  // ---------- classical background synthesis (depth res) ----------
  let bgD = null;
  if (params.withBg) {
    if (cacheHit && stageCache.bg) {
      bgD = stageCache.bg;
    } else {
      post('inpaint', 0.4);
      const fgB = fgBoundary(dispD, dw, dh, jump);
      const bandPx = params.bgBandPx
        || Math.max(12, Math.round(Math.min(dw, dh) * 0.11));
      bgD = synthesizeBackground(rgbaD, dispD, dw, dh, fgB, { bandPx, jump });
      bgD.fgB = fgB;
      bgD.bandPx = bandPx;
      if (stageCache) stageCache.bg = bgD;
    }
  }

  // ---------- anchors + march bounds ----------
  const dSub = Math.max(subjectDisparity(dispD, dw, dh), 0.08);
  const dMin = Math.max(percentile(dispD, 0.01) - 0.02, 0);
  const dMax = Math.min(percentile(dispD, 0.995) + 0.02, 1.05);

  // the photograph's own grain, so invented pixels can be given the same
  const noiseSigma = estimateNoiseSigma(rgba, w, h);

  const bandRows = Math.max(1, Math.floor(dh * 0.15));
  const rowMean = (y0, y1) => {
    let s = 0, n = 0;
    for (let y = y0; y < y1; y++) {
      for (let x = 0; x < dw; x += 3) { s += dispD[y * dw + x]; n++; }
    }
    return n ? s / n : 0;
  };

  const finish = (phase, fill) => {
    post('build', 0.75);
    // working-res fill colors: AI (final) or classical upsample (preview)
    let bgW = null;
    if (bgD) {
      bgW = fill && fill.bgW ? fill.bgW : upsampleBackgroundTo(bgD, dw, dh, w, h);
    }
    const layers = buildLayers({
      rgba, w, h, dispD, dw, dh,
      bgW, bgD,
      padPx, padD,
      plateRgba: fill ? fill.plateRgba : null,
      shade: fill ? 1.0 : 0.94,
      erodeIterations: 1,
    });
    const meta = {
      ...depthMeta,
      noiseSigma,
      w, h, padPx, dw, dh, padD,
      pw: layers.pw, ph: layers.ph, pdw: layers.pdw, pdh: layers.pdh,
      dSub, dMin, dMax, dFloor: 0.04,
      bandFrac: bgD ? bgD.bandPx / Math.min(dw, dh) : 0,
      dispTopMean: rowMean(0, bandRows),
      dispBottomMean: rowMean(dh - bandRows, dh),
      phase,
      fillKind: fill ? 'ai' : (params.withBg ? 'classical' : 'off'),
      fillBackend: fill ? fill.backend : null,
    };
    post('done', 1);
    self.postMessage({
      type: 'built', id, meta,
      color0: layers.color0, disp0: layers.disp0,
      color1: layers.color1, disp1: layers.disp1,
    }, [layers.color0.buffer, layers.disp0.buffer, layers.color1.buffer, layers.disp1.buffer]);
  };

  // ---------- generative fill ----------
  const wantFill = (params.aiFill || (!!params.sidecar && !!params.sidecarForBase)) && (bgD || padPx > 0);
  if (!wantFill) {
    finish('final', null);
    return;
  }
  if (stageCache && stageCache.fillFailed) {
    finish('final', null);
    return;
  }
  const cachedFill = stageCache && stageCache.fill;
  if (cachedFill &&
      (padPx === 0 || (cachedFill.plateRgba && cachedFill.padPx >= padPx)) &&
      (!bgD || cachedFill.bgW)) {
    finish('final', cachedFill);
    return;
  }

  // ship an interactive classical preview before touching the model
  finish('preview', null);

  const markFailed = () => {
    if (stageCache && stageCache.sourceId === msg.sourceId) stageCache.fillFailed = true;
  };

  fillDlSink = (p) => {
    if (!superseded() && p.phase === 'download') post('fill-download', 0.8, { pct: p.pct });
  };
  // The base build's holes are thin silhouette bands + the outpaint ring:
  // MI-GAN handles those in ~200 ms. The 4B prior costs ~50 s a call and is
  // reserved for novel-view expansion, where a person's far side has to be
  // INVENTED — unless explicitly asked for on the base too.
  const inpainter = await chooseInpainter(
    params.sidecarForBase ? params : { ...params, sidecar: null },
    (p) => fillDlSink && fillDlSink(p));
  if (superseded()) return;
  if (!inpainter) {
    markFailed();
    self.postMessage({ type: 'fill-failed', id, message: 'fill model unavailable' });
    return;
  }

  try {
    // working-res masks: nearest-upsampled from depth res (crisp), collar
    // grown on the bilinear-upsampled disparity (smooth is fine there)
    let holes, prefilled, bgWMask = null, bgWColor = null;
    if (bgD) {
      const dispW = resizeFloat(dispD, dw, dh, w, h);
      const bgUp = upsampleBackgroundTo(bgD, dw, dh, w, h);
      bgWMask = bgUp.bgMask;
      bgWColor = bgUp.bgColor;
      const fgBW = upsampleMask(bgD.fgB, dw, dh, w, h);
      ({ holes, prefilled } = buildFillInput(
        rgba, dispW, { bgMask: bgWMask, bgColor: bgWColor }, fgBW, w, h, { jump }));
    } else {
      holes = new Uint8Array(w * h);
      prefilled = rgba;
    }
    const budget = params.deviceClass === 'mobile'
      ? { interior: { maxBoxPx: 768, maxCalls: 4 }, ring: { maxBoxPx: 1024, maxCalls: 4 } }
      : { interior: { maxBoxPx: 512, maxCalls: 10 }, ring: { maxBoxPx: 768, maxCalls: 8 } };
    const { filled, genMask, plate, plateInit, ring } = await inpainter.fill({
      rgba: prefilled, holes, w, h,
      padPx,
      budget,
      consumable: bgWMask,
      shouldAbort: () => gen !== buildGen,
      onProgress: ({ phase, done, total }) => {
        if (superseded()) return;
        if (phase === 'wait') post('fill', 0.8, { waiting: true });
        else post('fill', 0.8 + 0.15 * (done / total), { done, total });
      },
    });

    const anchorR = Math.max(8, Math.min(32, Math.round(Math.min(w, h) * 0.015)));
    if (plate && padPx > 0) {
      // anchor the outpaint ring to the exact plate init the model saw and
      // grain it — same treatment the interior fills get
      const ppw = w + 2 * padPx, pph = h + 2 * padPx;
      anchorToReference(plate, plateInit, ring, ppw, pph, anchorR);
      addGrain(plate, ring, ppw, pph);
    }

    let bgW = null;
    if (bgD) {
      anchorToReference(filled, prefilled, genMask, w, h, anchorR);
      const bgColor = new Uint8ClampedArray(bgWColor);
      const grainMask = new Uint8Array(w * h);
      for (let i = 0; i < w * h; i++) {
        if (!bgWMask[i]) continue;
        if (genMask[i]) {
          bgColor[i * 4] = filled[i * 4];
          bgColor[i * 4 + 1] = filled[i * 4 + 1];
          bgColor[i * 4 + 2] = filled[i * 4 + 2];
          bgColor[i * 4 + 3] = 255;
          grainMask[i] = 1;
        } else {
          bgColor[i * 4] = bgWColor[i * 4] * 0.94;
          bgColor[i * 4 + 1] = bgWColor[i * 4 + 1] * 0.94;
          bgColor[i * 4 + 2] = bgWColor[i * 4 + 2] * 0.94;
        }
      }
      addGrain(bgColor, grainMask, w, h);
      bgW = { bgColor, bgMask: bgWMask };
    }

    const fill = {
      bgW,
      plateRgba: plate, // padded working-res outpaint (null if padPx = 0)
      padPx,
      backend: inpainter.backend,
    };
    if (stageCache && stageCache.sourceId === msg.sourceId) stageCache.fill = fill;
    if (superseded()) return;
    finish('final', fill);
  } catch (err) {
    if (err && err.name === 'AbortError') return;
    console.warn('generative fill failed; classical preview stands:', err);
    markFailed();
    if (!superseded()) {
      self.postMessage({ type: 'fill-failed', id, message: String(err && err.message || err) });
    }
  }
}
