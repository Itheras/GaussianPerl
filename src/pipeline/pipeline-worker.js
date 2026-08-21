// Pipeline worker: image + disparity -> splat cloud, off the main thread.
// Input disparity is raw AI output, decoded GT, or null (heuristic fallback).

import {
  normalizeDisparity, heuristicDisparity, refineDisparity, edgeMask, fgBoundary,
  snapDepthEdges,
} from './depthproc.js';
import { synthesizeBackground } from './inpaint.js';
import { buildSplats } from './splat-build.js';
import { encodeSplatFile } from '../io/save.js';

// Stage cache: normalize/refine/snap/edges/inpaint depend only on the source
// image (keyed by sourceId), not on depthStrength or splat params — so a
// Depth-slider rebuild skips seconds of recomputation and goes straight to
// buildSplats.
let stageCache = null; // {sourceId, w, h, disp, edges, bg}

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
  const { id, w, h, params } = msg;
  const rgba = new Uint8ClampedArray(msg.rgba);
  const post = (stage, frac) => self.postMessage({ type: 'progress', id, stage, frac });

  try {
    const jump = params.edgeDispJump;
    const cacheHit = stageCache && msg.sourceId !== undefined &&
      stageCache.sourceId === msg.sourceId &&
      stageCache.w === w && stageCache.h === h;

    let disp, edges;
    if (cacheHit) {
      disp = stageCache.disp;
      edges = stageCache.edges;
    } else {
      if (msg.disparity) {
        post('normalize', 0.05);
        disp = normalizeDisparity(new Float32Array(msg.disparity));
        if (params.refine) {
          post('refine', 0.15);
          disp = refineDisparity(disp, rgba, w, h);
          disp = normalizeDisparity(disp);
        }
      } else {
        post('heuristic', 0.1);
        disp = heuristicDisparity(rgba, w, h);
      }
      post('edges', 0.35);
      // consolidate soft silhouette ramps into clean steps before edge detection
      disp = snapDepthEdges(disp, w, h, jump, 2);
      edges = edgeMask(disp, w, h, jump);
      stageCache = { sourceId: msg.sourceId, w, h, disp, edges, bg: null };
    }

    let bg = null;
    if (params.withBg) {
      if (cacheHit && stageCache.bg) {
        bg = stageCache.bg;
      } else {
        post('inpaint', 0.45);
        const fgB = fgBoundary(disp, w, h, jump);
        // band scales with resolution: parallax reveals ~11% of the short side
        const bandPx = params.bgBandPx
          || Math.max(12, Math.min(72, Math.round(Math.min(w, h) * 0.11)));
        bg = synthesizeBackground(rgba, disp, w, h, fgB, { bandPx, jump });
        if (stageCache) stageCache.bg = bg;
      }
    }

    post('build', 0.7);
    // skirt width scales with resolution (fixed px would vanish at high res)
    if (params.withSkirt && !params.skirtPx) {
      params.skirtPx = Math.max(16, Math.min(88, Math.round(Math.min(w, h) * 0.1)));
    }
    const cloud = buildSplats({ rgba, w, h, disp, edges, bg, params });

    post('done', 1);
    self.postMessage({
      type: 'built', id,
      count: cloud.count,
      positions: cloud.positions,
      cov: cloud.cov,
      colors: cloud.colors,
      meta: cloud.meta,
    }, [cloud.positions.buffer, cloud.cov.buffer, cloud.colors.buffer]);
  } catch (err) {
    self.postMessage({ type: 'error', id, message: String(err && err.stack || err) });
  }
};
