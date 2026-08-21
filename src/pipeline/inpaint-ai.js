// On-device generative fill via MI-GAN (512, Places2) under onnxruntime-web.
// Runs inside the pipeline worker (wasm EP is off-main-thread by construction;
// the native WebGPU EP is used on browsers where it works).
//
// Safety-critical bundle choice (do not "simplify"):
//   - Safari/WebKit: ORT's JSEP and asyncify wasm builds melt WebKit 26.2's JIT
//     (ORT #26827, unfixed) and the WebGPU EP fails on iPhones (ORT #26480).
//     WebKit therefore ALWAYS gets ort.wasm.bundle.min.mjs (plain build, wasm
//     EP) — gated by isSafariEngine() plus the main thread's webkitHint (via
//     opts.forceWasm) for UA-camouflaged cases like Chrome on iPadOS.
//   - Elsewhere with WebGPU: ort.webgpu.bundle.min.mjs, falling back to the
//     plain wasm bundle if session creation or the first run fails.
//
// Model I/O (verified from the graph + browser precedent in lxfater/inpaint-web):
//   image  uint8 [1,3,H,W] RGB 0..255, dynamic H/W
//   mask   uint8 [1,1,H,W], 255 = known, 0 = hole (binary)
//   result uint8 [1,3,H,W]; known pixels are preserved, holes filled with a
//   feathered blend; the graph internally crops around the mask bbox (+128px),
//   resizes the crop to 512, infers, and scatters back.
// Weights are fetched from the author's HF repo at runtime (revision-pinned,
// see config.js) and cached via the Cache API. Everything degrades to null so
// the classical fill path keeps working offline.

import { FILL, isSafariEngine } from '../config.js';
import {
  planClusters, packImageNCHW, packMaskForBox, unpackNCHW,
  padPlate, ringMask,
} from './fill-plan.js';

const CACHE_NAME = 'gaussianperl-models';

// ORT's .mjs bundles ship both named exports and a default namespace object —
// normalize so either shape works.
async function importOrt(url) {
  const m = await import(/* @vite-ignore */ url);
  return m.default ?? m;
}

// crypto.subtle needs a secure context; on the documented http-over-LAN iPhone
// dev flow it is absent. Policy there: the AUTHOR's https URL may be used
// unverified (TLS is the trust), the unaffiliated mirror may NOT (the hash is
// its only trust boundary). Returns null when hashing is unavailable.
async function sha256Hex(bytes) {
  if (!globalThis.crypto?.subtle) return null;
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

async function fetchWithProgress(url, onProgress) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`model fetch failed: ${resp.status}`);
  const total = parseInt(resp.headers.get('content-length') || '0', 10);
  if (!(resp.body && resp.body.getReader)) {
    return new Uint8Array(await resp.arrayBuffer());
  }
  const reader = resp.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    // report every chunk even without a Content-Length — each event proves
    // liveness to the main-thread watchdog; pct is undefined when unknown
    onProgress(total ? Math.min(got / total, 1) : undefined);
  }
  const bytes = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }
  return bytes;
}

// CacheStorage hit -> HF -> jsdelivr mirror; sha256-gated wherever hashing is
// available (the hash is the trust boundary for the mirror AND guards a
// corrupted cache). Without crypto.subtle only the author's URL is allowed.
async function fetchModelBytes(onProgress) {
  const canHash = !!globalThis.crypto?.subtle;
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch { /* private mode etc. */ }
  if (cache) {
    try {
      const hit = await cache.match(FILL.modelUrl);
      if (hit) {
        const bytes = new Uint8Array(await hit.arrayBuffer());
        const hash = await sha256Hex(bytes);
        if (hash === FILL.modelSha256 || (hash === null && bytes.length > 0)) return bytes;
        await cache.delete(FILL.modelUrl);
      }
    } catch { /* fall through to network */ }
  }
  const sources = canHash ? [FILL.modelUrl, FILL.modelMirrorUrl] : [FILL.modelUrl];
  if (!canHash) {
    console.warn('crypto.subtle unavailable (insecure context): fetching the fill '
      + 'model unverified from the author\'s URL only; the mirror is disabled');
  }
  let lastErr = null;
  for (const url of sources) {
    try {
      const bytes = await fetchWithProgress(url, onProgress);
      const hash = await sha256Hex(bytes);
      if (hash !== null && hash !== FILL.modelSha256) {
        throw new Error(`sha256 mismatch from ${url}`);
      }
      if (cache) {
        try { await cache.put(FILL.modelUrl, new Response(bytes)); } catch { /* quota */ }
      }
      return bytes;
    } catch (err) {
      lastErr = err;
      console.warn('fill model source failed:', url, err);
    }
  }
  throw lastErr;
}

let _instance = null;
let _loading = null;

export class Inpainter {
  constructor(ort, session, backend, bytes) {
    this.ort = ort;
    this.session = session;
    this.backend = backend; // 'webgpu' | 'wasm'
    // keep bytes only while a webgpu->wasm fallback is still possible
    this._bytes = backend === 'webgpu' ? bytes : null;
  }

  /**
   * onProgress({phase:'download'|'init', pct}). Returns null when the CDN or
   * model is unreachable — callers fall back to the classical fill.
   * Concurrent callers share one in-flight load; a failed load may be retried.
   */
  static load(onProgress = () => {}, opts = {}) {
    if (_instance) return Promise.resolve(_instance);
    if (!_loading) {
      _loading = Inpainter._doLoad(onProgress, opts)
        .finally(() => { _loading = null; });
    }
    return _loading;
  }

  static async _doLoad(onProgress, opts) {
    try {
      const bytes = await fetchModelBytes(
        (pct) => onProgress({ phase: 'download', pct }));
      let wantGpu = !opts.forceWasm && !isSafariEngine() &&
        !!(globalThis.navigator && navigator.gpu);
      if (wantGpu) {
        // navigator.gpu can exist with no usable adapter (headless, blocklists)
        try { wantGpu = !!(await navigator.gpu.requestAdapter()); } catch { wantGpu = false; }
      }
      let ort = null, session = null, backend = 'wasm';
      if (wantGpu) {
        try {
          ort = await importOrt(FILL.ortWebgpu);
          ort.env.wasm.wasmPaths = FILL.ortDist;
          session = await ort.InferenceSession.create(bytes, {
            executionProviders: ['webgpu'],
          });
          backend = 'webgpu';
        } catch (err) {
          console.warn('fill model on webgpu failed, trying wasm:', err);
          ort = null; session = null;
        }
      }
      if (!session) {
        ort = await importOrt(FILL.ortWasm);
        ort.env.wasm.wasmPaths = FILL.ortDist;
        session = await ort.InferenceSession.create(bytes, {
          executionProviders: ['wasm'],
        });
        backend = 'wasm';
      }
      if (!session.inputNames.includes('image') || !session.inputNames.includes('mask')) {
        throw new Error(`unexpected model inputs: ${session.inputNames}`);
      }
      onProgress({ phase: 'init', pct: 1 });
      _instance = new Inpainter(ort, session, backend, bytes);
      return _instance;
    } catch (err) {
      console.warn('fill model unavailable:', err);
      return null;
    }
  }

  // Serialized: the worker can be re-entered by a newer build while an older
  // one still awaits inference — the mutex makes each run (INCLUDING the
  // webgpu->wasm session swap in the fallback) atomic w.r.t. other runs.
  _run(feeds) {
    const run = () => this._runExclusive(feeds);
    const p = (this._mutex ?? Promise.resolve()).then(run, run);
    this._mutex = p.catch(() => {});
    return p;
  }

  async _runExclusive(feeds) {
    try {
      const out = await this.session.run(feeds);
      // first successful webgpu run proves the EP works — drop the 28 MB
      // fallback copy of the weights
      this._bytes = null;
      return out;
    } catch (err) {
      if (this.backend === 'webgpu' && this._bytes) {
        console.warn('webgpu run failed, rebuilding session on wasm:', err);
        const dead = this.session;
        const ort = await importOrt(FILL.ortWasm);
        ort.env.wasm.wasmPaths = FILL.ortDist;
        this.session = await ort.InferenceSession.create(this._bytes, {
          executionProviders: ['wasm'],
        });
        // best-effort, fire-and-forget: a hanging release on a lost GPU
        // device must not stall the wasm retry
        try { Promise.resolve(dead.release()).catch(() => {}); } catch { /* ok */ }
        this.ort = ort;
        this.backend = 'wasm';
        this._bytes = null;
        return await this.session.run(feeds);
      }
      throw err;
    }
  }

  /**
   * One model call: fill `holes` limited to `box`, on rgba (w×h).
   * Returns new rgba; source ALPHA is preserved through the round-trip so
   * transparent-background images keep their transparency guards alive.
   */
  async _fillBox(rgba, holes, w, h, box) {
    const img = packImageNCHW(rgba, w, h);
    const mask = packMaskForBox(holes, w, h, box);
    const feeds = {
      image: new this.ort.Tensor('uint8', img, [1, 3, h, w]),
      mask: new this.ort.Tensor('uint8', mask, [1, 1, h, w]),
    };
    const out = await this._run(feeds);
    return unpackNCHW(out.result.data, w, h, rgba);
  }

  /**
   * Full generative fill: disocclusion holes on the image, then border
   * outpainting on a mirror-padded plate. Sequential per-cluster calls, each
   * seeing the previous results as context.
   * opts: {rgba, holes, w, h, padPx, budget, consumable, shouldAbort, onProgress}
   *   budget: {interior:{maxBoxPx,maxCalls}, ring:{maxBoxPx,maxCalls}}
   *   consumable: mask of hole pixels whose output is actually USED (bgMask);
   *     collar/rim pixels are masked for context-hygiene only — clusters with
   *     no consumable pixels are skipped, and discarded model output at
   *     non-consumable pixels never leaks into the outpaint plate.
   *   shouldAbort(): checked between model calls; throws AbortError when true.
   * Returns {filled, genMask, plate, plateInit, ring, pw, ph} — plate* null/empty
   * if padPx is 0. genMask marks hole pixels the model actually generated.
   */
  async fill({ rgba, holes, w, h, padPx, budget, consumable = null,
    shouldAbort = () => false, onProgress = () => {} }) {
    const base = { cellPx: FILL.cellPx, overlapPx: FILL.overlapPx, mergeGapPx: FILL.mergeGapPx };
    const boxes = planClusters(holes, w, h, {
      ...base,
      maxBoxPx: budget?.interior?.maxBoxPx ?? FILL.maxBoxPx,
      maxCalls: budget?.interior?.maxCalls ?? FILL.maxCalls,
    });

    let ringBoxes = [], ring = null, pw = 0, ph = 0;
    if (padPx > 0) {
      pw = w + 2 * padPx; ph = h + 2 * padPx;
      ring = ringMask(pw, ph, padPx);
      ringBoxes = planClusters(ring, pw, ph, {
        ...base,
        maxBoxPx: budget?.ring?.maxBoxPx ?? 768,
        maxCalls: budget?.ring?.maxCalls ?? FILL.maxCalls,
      });
    }
    const total = boxes.length + ringBoxes.length;
    let done = 0;
    // yield a real macrotask between model calls so a queued 'build' message
    // can run and flip shouldAbort — microtask-only chains starve onmessage
    const checkpoint = async () => {
      await new Promise((r) => setTimeout(r));
      if (shouldAbort()) {
        const e = new Error('fill aborted (superseded build)');
        e.name = 'AbortError';
        throw e;
      }
    };

    let filled = new Uint8ClampedArray(rgba);
    const genMask = new Uint8Array(w * h);
    for (const box of boxes) {
      await checkpoint();
      // gate on pixels we will actually USE: a cluster of pure collar/rim is
      // a full model call whose entire output would be discarded — and tiny
      // consumable slivers keep the classical prefill (Shih drops <10px curves)
      if (countHoles(consumable ?? holes, w, box) < FILL.minHolePx) {
        onProgress({ phase: 'run', done: ++done, total });
        continue;
      }
      filled = await this._fillBox(filled, holes, w, h, box);
      for (let y = box.y0; y < box.y1; y++) {
        const row = y * w;
        for (let x = box.x0; x < box.x1; x++) {
          if (holes[row + x]) genMask[row + x] = 1;
        }
      }
      onProgress({ phase: 'run', done: ++done, total });
    }

    let plate = null, plateInit = null;
    if (padPx > 0) {
      // plate base honors the discard contract: model output survives only at
      // consumable pixels; discarded collar/rim content must not become the
      // ring's context (or its anchor reference)
      let plateSrc = filled;
      if (consumable) {
        plateSrc = new Uint8ClampedArray(filled);
        for (let i = 0; i < w * h; i++) {
          if (holes[i] && !consumable[i]) {
            plateSrc[i * 4] = rgba[i * 4];
            plateSrc[i * 4 + 1] = rgba[i * 4 + 1];
            plateSrc[i * 4 + 2] = rgba[i * 4 + 2];
            plateSrc[i * 4 + 3] = rgba[i * 4 + 3];
          }
        }
      }
      ({ plate } = padPlate(plateSrc, w, h, padPx));
      plateInit = plate.slice(); // what the model saw — the anchor reference
      for (const box of ringBoxes) {
        await checkpoint();
        plate = await this._fillBox(plate, ring, pw, ph, box);
        onProgress({ phase: 'run', done: ++done, total });
      }
    }
    return { filled, genMask, plate, plateInit, ring, pw, ph };
  }
}

function countHoles(holes, w, box) {
  let n = 0;
  for (let y = box.y0; y < box.y1; y++) {
    const row = y * w;
    for (let x = box.x0; x < box.x1; x++) n += holes[row + x];
  }
  return n;
}
