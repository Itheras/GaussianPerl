// Pipeline worker: image (+ optional GT disparity) -> splat cloud, off the main
// thread. Owns BOTH AI models (depth via transformers.js, generative fill via
// onnxruntime-web): wasm inference in here never janks the UI, and transformers
// v4 no longer proxies wasm off the calling thread.
//
// Build flow per message:
//   depth (GT | AI | heuristic) -> normalize/refine/snap/edges -> classical
//   background synth -> [preview 'built' ships immediately] -> generative fill
//   (MI-GAN: disocclusion clusters + border outpaint ring) -> final 'built'.
// The preview keeps first-splat latency at classical speed while the fill model
// downloads/runs; a newer 'build' message abandons the older continuation
// (buildGen guard) — its posts are also filtered by id on the main thread.

import {
  normalizeDisparity, heuristicDisparity, refineDisparity, edgeMask, fgBoundary,
  snapDepthEdges, compressFarField,
} from './depthproc.js';
import { synthesizeBackground, addGrain } from './inpaint.js';
import {
  buildFillInput, padFloat, anchorToReference, smoothRingDisparity,
} from './fill-plan.js';
import { buildSplats } from './splat-build.js';
import { encodeSplatFile } from '../io/save.js';
import { DepthEstimator } from './depth-ai.js';
import { Inpainter } from './inpaint-ai.js';

// Stage cache: everything derived from the source image alone (keyed by
// sourceId) — depth, edges, classical bg, and the AI fill — so a Depth-slider
// rebuild skips seconds of recomputation (and the model calls entirely).
let stageCache = null; // {sourceId, w, h, disp, edges, bg, depthMeta, fill, fillFailed}

let buildGen = 0;

// Model loaders pin the FIRST caller's progress callback for the shared
// in-flight load; these mutable sinks let the newest build re-route download
// progress to its own id (stale-id progress is dropped by the main thread,
// which would starve the watchdog during a perfectly healthy download).
let depthDlSink = null;
let fillDlSink = null;

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'export') {
    try {
      const bytes = encodeSplatFile({
        count: msg.count,
        positions: new Float32Array(msg.positions),
        cov: new Float32Array(msg.cov),
        colors: new Uint8Array(msg.colors),
      });
      self.postMessage({ type: 'exported', id: msg.id, bytes }, [bytes.buffer]);
    } catch (err) {
      self.postMessage({ type: 'error', id: msg.id, message: String(err && err.stack || err) });
    }
    return;
  }
  if (msg.type !== 'build') return;
  const gen = ++buildGen;
  build(msg, gen).catch((err) => {
    if (gen === buildGen) {
      self.postMessage({ type: 'error', id: msg.id, message: String(err && err.stack || err) });
    }
  });
};

async function build(msg, gen) {
  const { id, w, h, params } = msg;
  const rgba = new Uint8ClampedArray(msg.rgba);
  const post = (stage, frac, extra) =>
    self.postMessage({ type: 'progress', id, stage, frac, ...extra });
  const superseded = () => gen !== buildGen;

  const jump = params.edgeDispJump;
  const cacheHit = stageCache && msg.sourceId !== undefined &&
    stageCache.sourceId === msg.sourceId &&
    stageCache.w === w && stageCache.h === h;

  // ---------- depth ----------
  let disp, edges, depthMeta;
  if (cacheHit) {
    disp = stageCache.disp;
    edges = stageCache.edges;
    depthMeta = stageCache.depthMeta;
  } else {
    let raw = msg.disparity ? new Float32Array(msg.disparity) : null;
    depthMeta = { depthKind: raw ? 'gt' : 'heuristic', depthBackend: null, depthTier: null };
    if (!raw && params.wantAiDepth) {
      depthDlSink = (p) => {
        if (!superseded() && p.phase === 'download') post('depth-download', 0.02, { pct: p.pct });
      };
      const est = await DepthEstimator.load((p) => depthDlSink && depthDlSink(p), {
        deviceClass: params.deviceClass,
        forceWasm: params.webkitHint,
      });
      if (superseded()) return;
      if (est) {
        post('depth', 0.05);
        try {
          raw = await est.estimate(rgba, w, h, w, h);
          depthMeta = { depthKind: 'ai', depthBackend: est.backend, depthTier: est.tier };
        } catch (err) {
          // inference itself can die (OOM on a tight phone) — a heuristic
          // splat beats no splat; load() failures already degrade the same way
          console.warn('depth inference failed; falling back to heuristic:', err);
          raw = null;
        }
        if (superseded()) return;
      }
    }

    if (raw) {
      post('normalize', 0.12);
      disp = normalizeDisparity(raw);
      if (depthMeta.depthKind === 'ai') {
        post('refine', 0.18);
        disp = refineDisparity(disp, rgba, w, h);
        disp = normalizeDisparity(disp);
      }
    } else {
      post('heuristic', 0.12);
      disp = heuristicDisparity(rgba, w, h);
    }
    post('edges', 0.3);
    // real far fields barely parallax — flatten the tail before edge
    // detection so the horizon is a backdrop, not a giant silhouette
    disp = compressFarField(disp, params.farKnee ?? 0.08, params.farKeep ?? 0.35);
    // consolidate soft silhouette ramps into clean steps before edge detection
    disp = snapDepthEdges(disp, w, h, jump, 2);
    edges = edgeMask(disp, w, h, jump);
    stageCache = {
      sourceId: msg.sourceId, w, h, disp, edges,
      bg: null, depthMeta, fill: null, fillFailed: false,
    };
  }

  // skirt width scales with resolution (fixed px would vanish at high res)
  const skirtPx = params.withSkirt
    ? (params.skirtPx || Math.max(16, Math.min(88, Math.round(Math.min(w, h) * 0.1))))
    : 0;

  // ---------- classical background synthesis ----------
  let bg = null;
  if (params.withBg) {
    if (cacheHit && stageCache.bg) {
      bg = stageCache.bg;
    } else {
      post('inpaint', 0.4);
      const fgB = fgBoundary(disp, w, h, jump);
      // band scales with resolution: parallax reveals ~11% of the short side
      const bandPx = params.bgBandPx
        || Math.max(12, Math.min(72, Math.round(Math.min(w, h) * 0.11)));
      bg = synthesizeBackground(rgba, disp, w, h, fgB, { bandPx, jump });
      bg.fgB = fgB;
      if (stageCache) stageCache.bg = bg;
    }
  }

  // quick disparity sanity stats for tests/diagnostics (top vs bottom band)
  const bandRows = Math.max(1, Math.floor(h * 0.15));
  const rowMean = (y0, y1) => {
    let s = 0, n = 0;
    for (let y = y0; y < y1; y++) {
      const row = y * w;
      for (let x = 0; x < w; x += 3) { s += disp[row + x]; n++; }
    }
    return n ? s / n : 0;
  };
  const dispTopMean = rowMean(0, bandRows);
  const dispBottomMean = rowMean(h - bandRows, h);

  const finish = (phase, fill) => {
    post('build', 0.75);
    const bgOut = (fill && fill.bgColor && bg) ? { ...bg, bgColor: fill.bgColor } : bg;
    const cloud = buildSplats({
      rgba, w, h, disp, edges,
      bg: bgOut,
      plate: fill ? fill.plateObj : null,
      params: { ...params, skirtPx, bgShade: fill ? 1.0 : 0.94 },
    });
    const meta = {
      ...cloud.meta,
      ...depthMeta,
      dispTopMean, dispBottomMean,
      phase,
      fillKind: fill ? 'ai' : (params.withBg ? 'classical' : 'off'),
      fillBackend: fill ? fill.backend : null,
    };
    post('done', 1);
    self.postMessage({
      type: 'built', id,
      count: cloud.count,
      positions: cloud.positions,
      cov: cloud.cov,
      colors: cloud.colors,
      meta,
    }, [cloud.positions.buffer, cloud.cov.buffer, cloud.colors.buffer]);
  };

  // ---------- generative fill ----------
  // the fill has work even without the bg layer: the outpainted skirt
  const wantFill = params.aiFill && (bg || skirtPx > 0);
  if (!wantFill) {
    finish('final', null);
    return;
  }
  // a failed fill for this source stays failed until a new source retries it —
  // rebuilds go straight to an honest classical final, no churn, no refetch
  if (stageCache && stageCache.fillFailed) {
    finish('final', null);
    return;
  }
  // cached fill must satisfy the CURRENT skirt request (a plateless cache from
  // a skirt-off build must not downgrade a skirt-on build to the mirror skirt)
  // and carry bg colors when the bg layer is now wanted
  const cachedFill = stageCache && stageCache.fill;
  if (cachedFill &&
      (skirtPx === 0 || (cachedFill.plateObj && cachedFill.plateObj.padPx >= skirtPx)) &&
      (!bg || cachedFill.bgColor)) {
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
  const inpainter = await Inpainter.load((p) => fillDlSink && fillDlSink(p), {
    forceWasm: params.forceWasmFill || params.webkitHint,
  });
  if (superseded()) return;
  if (!inpainter) {
    // preview stands; classical fill is the result — main must hear about it
    // (it keeps a hung-AI watchdog armed until the fill resolves either way)
    markFailed();
    self.postMessage({ type: 'fill-failed', id, message: 'fill model unavailable' });
    return;
  }

  try {
    const { holes, prefilled } = bg
      ? buildFillInput(rgba, disp, bg, bg.fgB, w, h, { jump })
      : { holes: new Uint8Array(w * h), prefilled: rgba }; // ring-only outpaint
    const budget = params.deviceClass === 'mobile'
      ? { interior: { maxBoxPx: 768, maxCalls: 4 }, ring: { maxBoxPx: 1024, maxCalls: 4 } }
      : { interior: { maxBoxPx: 512, maxCalls: 6 }, ring: { maxBoxPx: 768, maxCalls: 6 } };
    const { filled, genMask, plate, plateInit, ring } = await inpainter.fill({
      rgba: prefilled, holes, w, h,
      padPx: skirtPx,
      budget,
      consumable: bg ? bg.bgMask : null,
      shouldAbort: () => gen !== buildGen,
      onProgress: ({ done, total }) => {
        if (!superseded()) post('fill', 0.8 + 0.15 * (done / total), { done, total });
      },
    });
    const pw = w + 2 * skirtPx, ph = h + 2 * skirtPx;

    const anchorR = Math.max(8, Math.min(20, Math.round(Math.min(w, h) * 0.015)));
    let bgColor = null;
    if (bg) {
      // anchor the fill's low frequencies to the classical estimate — GAN
      // hallucinations (dark blobs, invented structures behind the horizon)
      // get pulled back to plausible color while the AI texture survives
      anchorToReference(filled, prefilled, genMask, w, h, anchorR);

      // extract colors where background splats live. Model-generated pixels
      // get grain (hides 512-crop softness); pixels of skipped tiny clusters
      // keep the classical color with its shade baked in (the global bgShade
      // is 1.0 on AI builds) and their existing grain — never grained twice.
      bgColor = new Uint8ClampedArray(bg.bgColor);
      const grainMask = new Uint8Array(w * h);
      for (let i = 0; i < w * h; i++) {
        if (!bg.bgMask[i]) continue;
        if (genMask[i]) {
          bgColor[i * 4] = filled[i * 4];
          bgColor[i * 4 + 1] = filled[i * 4 + 1];
          bgColor[i * 4 + 2] = filled[i * 4 + 2];
          bgColor[i * 4 + 3] = 255;
          grainMask[i] = 1;
        } else {
          bgColor[i * 4] = bg.bgColor[i * 4] * 0.94;
          bgColor[i * 4 + 1] = bg.bgColor[i * 4 + 1] * 0.94;
          bgColor[i * 4 + 2] = bg.bgColor[i * 4 + 2] * 0.94;
        }
      }
      addGrain(bgColor, grainMask, w, h);
    }

    let plateObj = null;
    if (plate && skirtPx > 0) {
      // same anchoring for the outpaint ring, against the exact plate init the
      // model saw (real edge content) — also evens out ring-tile seams
      anchorToReference(plate, plateInit, ring, pw, ph, anchorR);
      addGrain(plate, ring, pw, ph);
      // replicated disparity drags the horizon cliff into the ring where it
      // renders as floating slabs — calm it down
      const plateDisp = smoothRingDisparity(
        padFloat(disp, w, h, skirtPx), pw, ph, skirtPx, Math.max(8, skirtPx >> 1));
      plateObj = { rgba: plate, disp: plateDisp, padPx: skirtPx, pw, ph };
    }
    const fill = { bgColor, plateObj, backend: inpainter.backend };
    // cache even when superseded: the completed model work must not be thrown
    // away — the successor build (same source) hits the cache instantly
    if (stageCache && stageCache.sourceId === msg.sourceId) stageCache.fill = fill;
    if (superseded()) return;
    finish('final', fill);
  } catch (err) {
    if (err && err.name === 'AbortError') return; // superseded mid-fill: silent
    console.warn('generative fill failed; classical preview stands:', err);
    markFailed();
    if (!superseded()) {
      self.postMessage({ type: 'fill-failed', id, message: String(err && err.message || err) });
    }
  }
}
