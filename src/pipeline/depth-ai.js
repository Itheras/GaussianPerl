// On-device monocular depth via transformers.js v4 (Depth Anything V2).
// Runs INSIDE the pipeline worker: v4 no longer proxies wasm inference to a
// side worker, so calling it from the main thread would freeze the UI.
// Model tiering:
//   - desktop + WebGPU (non-Safari): DA V2 *base* q4f16 (~72 MB) — visibly
//     sharper depth edges, the thing 3D-photo realism lives or dies on,
//   - WebGPU (non-Safari): DA V2 small fp16,
//   - everywhere else (incl. all Safari, see below): DA V2 small wasm q8.
// Safari/WebKit gets NO webgpu attempt: transformers v4 pins Safari to ORT's
// plain wasm build (WebKit 26.2 JIT meltdown on the JSEP/asyncify builds,
// ORT #26827) — we gate here too rather than trusting a fast failure.
// Weights download once and are cached by the library; wrapped so the app
// still works when the CDN/model is unreachable (caller falls back).

import { MODEL, isSafariEngine } from '../config.js';
import { resizeFloat } from '../util/imageops.js';

let _instance = null;
let _loading = null;

export class DepthEstimator {
  constructor(pipe, backend, tier) {
    this.pipe = pipe;
    this.backend = backend; // 'webgpu' | 'wasm'
    this.tier = tier;       // 'base' | 'small'
    this.RawImage = null;
  }

  /**
   * onProgress({phase, pct, note}) — phases: 'download', 'init'.
   * opts.deviceClass: 'desktop' | 'mobile' (main thread decides; worker UA
   * can't detect iPadOS). Returns null if no model can be loaded.
   * Concurrent callers share one in-flight load; a failed load may be retried.
   */
  static load(onProgress = () => {}, opts = {}) {
    if (_instance) return Promise.resolve(_instance);
    if (!_loading) {
      _loading = DepthEstimator._doLoad(onProgress, opts)
        .finally(() => { _loading = null; });
    }
    return _loading;
  }

  static async _doLoad(onProgress, opts) {
    let tf;
    try {
      tf = await import(/* @vite-ignore */ `${MODEL.cdn}`);
    } catch (err) {
      console.warn('transformers.js unavailable:', err);
      return null;
    }
    const { pipeline, RawImage, env } = tf;
    env.allowLocalModels = false;

    const progress = (p) => {
      if (p.status === 'progress') {
        // forward even without a total: each event proves the download is
        // alive (feeds the main-thread watchdog), pct just stays undefined
        onProgress({
          phase: 'download',
          pct: p.total ? p.loaded / p.total : undefined,
          note: p.file,
        });
      } else if (p.status === 'ready') {
        onProgress({ phase: 'init', pct: 1 });
      }
    };

    const attempts = [];
    // navigator.gpu existing is NOT enough (headless/blocked environments
    // expose it with no adapter) — and a FAILED webgpu pipeline attempt can
    // leave the library's ORT state unable to fall back to wasm afterwards,
    // so probe the adapter before committing to any webgpu attempt.
    // opts.forceWasm: main-thread WebKit hint (Chrome-on-iPadOS wears the
    // desktop-Mac UA; only maxTouchPoints on the main thread can tell).
    let hasGpu = false;
    if (!opts.forceWasm && !isSafariEngine() && globalThis.navigator && navigator.gpu) {
      try { hasGpu = !!(await navigator.gpu.requestAdapter()); } catch { hasGpu = false; }
    }
    if (hasGpu && opts.deviceClass === 'desktop') {
      attempts.push({ id: MODEL.idHQ, device: 'webgpu', dtype: 'q4f16', tier: 'base' });
    }
    if (hasGpu) {
      attempts.push({ id: MODEL.id, device: 'webgpu', dtype: 'fp16', tier: 'small' });
    }
    attempts.push({ id: MODEL.id, device: 'wasm', dtype: 'q8', tier: 'small' });

    for (const opt of attempts) {
      try {
        const pipe = await pipeline('depth-estimation', opt.id, {
          device: opt.device, dtype: opt.dtype, progress_callback: progress,
        });
        const inst = new DepthEstimator(pipe, opt.device, opt.tier);
        inst.RawImage = RawImage;
        _instance = inst;
        return inst;
      } catch (err) {
        console.warn(`depth model ${opt.tier} on ${opt.device} failed:`, err);
      }
    }
    return null;
  }

  /**
   * rgba/width/height: working-resolution image. Returns Float32Array disparity
   * resampled to (outW, outH), raw model units (bigger = closer), not yet
   * normalized. Serialized: the worker can be re-entered by a newer build
   * while an older one still awaits inference on this shared session.
   */
  async estimate(rgba, width, height, outW, outH) {
    const run = () => this._estimate(rgba, width, height, outW, outH);
    const p = (this._mutex ?? Promise.resolve()).then(run, run);
    this._mutex = p.catch(() => {});
    return p;
  }

  async _estimate(rgba, width, height, outW, outH) {
    const img = new this.RawImage(new Uint8ClampedArray(rgba), width, height, 4);
    const out = await this.pipe(img);
    const t = out.predicted_depth;
    const dims = t.dims;
    const mh = dims[dims.length - 2], mw = dims[dims.length - 1];
    let data = t.data;
    if (!(data instanceof Float32Array)) data = Float32Array.from(data);
    if (mw === outW && mh === outH) return data.slice();
    return resizeFloat(data, mw, mh, outW, outH);
  }
}
