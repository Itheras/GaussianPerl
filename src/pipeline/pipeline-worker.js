// Pipeline worker: image + disparity -> splat cloud, off the main thread.
// Input disparity is raw AI output, decoded GT, or null (heuristic fallback).

import {
  normalizeDisparity, heuristicDisparity, refineDisparity, edgeMask, fgBoundary,
} from './depthproc.js';
import { synthesizeBackground } from './inpaint.js';
import { buildSplats } from './splat-build.js';

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type !== 'build') return;
  const { id, w, h, params } = msg;
  const rgba = new Uint8ClampedArray(msg.rgba);
  const post = (stage, frac) => self.postMessage({ type: 'progress', id, stage, frac });

  try {
    let disp;
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
    const jump = params.edgeDispJump;
    const edges = edgeMask(disp, w, h, jump);

    let bg = null;
    if (params.withBg) {
      post('inpaint', 0.45);
      const fgB = fgBoundary(disp, w, h, jump);
      bg = synthesizeBackground(rgba, disp, w, h, fgB, {
        bandPx: params.bgBandPx, jump,
      });
    }

    post('build', 0.7);
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
