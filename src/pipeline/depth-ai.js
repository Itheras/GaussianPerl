// On-device monocular depth via transformers.js (Depth Anything V2 small).
// WebGPU when available, else WASM (quantized). Weights are fetched once by the
// library and cached in browser storage; inference never leaves the device.
// Everything is wrapped so the app still works when the CDN/model is unreachable.

import { MODEL } from '../config.js';
import { resizeFloat } from '../util/imageops.js';

let _instance = null;

export class DepthEstimator {
  constructor(pipe, backend) {
    this.pipe = pipe;
    this.backend = backend; // 'webgpu' | 'wasm'
    this.RawImage = null;
  }

  /**
   * onProgress({phase, pct, note}) — phases: 'download', 'init'
   * Returns null if the model can't be loaded (offline etc.) — caller falls back.
   */
  static async load(onProgress = () => {}) {
    if (_instance) return _instance;
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
      if (p.status === 'progress' && p.total) {
        onProgress({ phase: 'download', pct: p.loaded / p.total, note: p.file });
      } else if (p.status === 'ready') {
        onProgress({ phase: 'init', pct: 1 });
      }
    };

    const attempts = [];
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      attempts.push({ device: 'webgpu', dtype: 'fp16' });
    }
    attempts.push({ device: 'wasm', dtype: 'q8' });

    for (const opt of attempts) {
      try {
        if (opt.device === 'wasm' && env.backends?.onnx?.wasm) {
          env.backends.onnx.wasm.proxy = true; // keep the main thread alive
        }
        const pipe = await pipeline('depth-estimation', MODEL.id, {
          device: opt.device, dtype: opt.dtype, progress_callback: progress,
        });
        const inst = new DepthEstimator(pipe, opt.device);
        inst.RawImage = RawImage;
        _instance = inst;
        return inst;
      } catch (err) {
        console.warn(`depth model on ${opt.device} failed:`, err);
      }
    }
    return null;
  }

  /**
   * imageData: ImageData (RGBA). Returns Float32Array disparity resampled to
   * (outW, outH), raw model units (bigger = closer), not yet normalized.
   */
  async estimate(imageData, outW, outH) {
    const img = new this.RawImage(
      new Uint8ClampedArray(imageData.data), imageData.width, imageData.height, 4);
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
