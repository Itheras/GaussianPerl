// Native generative fill: drive the local inference sidecar (M11).
//
// Same contract as Inpainter.fill() in inpaint-ai.js, so expand.js and the
// base build in pipeline-worker.js do not know which guesser they are talking
// to. The difference is what is on the other end: MI-GAN is a 28 MB
// hole-filler with no idea what a person is; the sidecar runs a 4B-parameter
// diffusion model with a learned prior over bodies, clothes, hair and scenes —
// the "guess what is in the missing region given the scene so far" the anchor
// loop has been missing.
//
// Discovery: {url, token} arrive via params (dev: ?sidecar=PORT:TOKEN, app:
// injected by the shell). probe() is cheap and non-fatal; everything degrades
// to the in-browser path when the sidecar is absent.

import { padPlate, ringMask } from '../pipeline/fill-plan.js';

function frame(header, payload) {
  const hb = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(4 + hb.length + payload.length);
  new DataView(out.buffer).setUint32(0, hb.length, true);
  out.set(hb, 4);
  out.set(payload, 4 + hb.length);
  return out;
}

function unframe(buf) {
  const view = new DataView(buf);
  const n = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n)));
  return { header, payload: new Uint8Array(buf, 4 + n) };
}

export class NativeInpainter {
  constructor(sidecar, health) {
    this.url = sidecar.url.replace(/\/$/, '');
    this.token = sidecar.token;
    this.backend = 'sidecar';
    this.model = health.model;
    this.loaded = !!health.loaded;
    this._mutex = null;
  }

  /** Returns an instance if the sidecar answers, else null. Never throws. */
  static async probe(sidecar, timeoutMs = 1500) {
    if (!sidecar || !sidecar.url || !sidecar.token) return null;
    try {
      const ctl = new AbortController();
      const t = setTimeout(() => ctl.abort(), timeoutMs);
      const r = await fetch(`${sidecar.url.replace(/\/$/, '')}/v1/health`, {
        headers: { Authorization: `Bearer ${sidecar.token}` },
        signal: ctl.signal,
      });
      clearTimeout(t);
      if (!r.ok) return null;
      const health = await r.json();
      if (!health.ok || health.loadError) return null;
      return new NativeInpainter(sidecar, health);
    } catch {
      return null;
    }
  }

  async _call(rgba, mask, w, h, opts, onProgress = null, shouldAbort = () => false) {
    // Liveness AND a deadline. The first call may block for minutes while the
    // sidecar loads weights, and the main thread's watchdog treats 150 s of
    // worker silence as a hang — so a heartbeat keeps it fed. But a heartbeat
    // with no deadline turns a hung sidecar into a session-long freeze (the
    // watchdog can never fire), so the same timer enforces an upper bound and
    // honours supersession. A deadline throws a plain Error (not AbortError)
    // so the caller falls back to the in-browser model.
    const t0 = Date.now();
    const ctl = new AbortController();
    const deadlineMs = (this.loaded ? 3 : 10) * 60 * 1000;
    let why = null;
    const tick = () => {
      const elapsed = Date.now() - t0;
      if (shouldAbort()) { why = 'superseded'; ctl.abort(); return; }
      if (elapsed > deadlineMs) { why = 'deadline'; ctl.abort(); return; }
      if (onProgress) onProgress({ phase: 'wait', elapsedS: Math.round(elapsed / 1000) });
    };
    const beat = setInterval(tick, 10000);
    try {
      const out = await this._callInner(rgba, mask, w, h, opts, ctl.signal);
      this.loaded = true;
      return out;
    } catch (err) {
      if (why === 'superseded') {
        const e = new Error('fill aborted (superseded build)');
        e.name = 'AbortError';
        throw e;
      }
      if (why === 'deadline') throw new Error(`sidecar deadline (${Math.round(deadlineMs / 1000)} s)`);
      throw err;
    } finally {
      clearInterval(beat);
    }
  }

  async _callInner(rgba, mask, w, h, opts, signal) {
    const payload = new Uint8Array(w * h * 5);
    payload.set(rgba instanceof Uint8Array ? rgba
      : new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength), 0);
    payload.set(mask, w * h * 4);
    const body = frame({ w, h, ...opts }, payload);
    const r = await fetch(`${this.url}/v1/inpaint`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.token}`,
        'Content-Type': 'application/octet-stream',
      },
      body,
      signal,
    });
    if (!r.ok) {
      let msg = `sidecar ${r.status}`;
      try { msg += `: ${(await r.json()).error}`; } catch { /* keep status */ }
      throw new Error(msg);
    }
    const { header, payload: out } = unframe(await r.arrayBuffer());
    return { rgba: new Uint8ClampedArray(out.buffer, out.byteOffset, out.byteLength).slice(), info: header };
  }

  /**
   * Inpainter.fill() contract. One model call covers ALL interior holes at
   * once — a diffusion model wants the whole frame as context, which is the
   * entire reason to use it — plus one call for the outpaint ring if padPx>0.
   */
  async fill({ rgba, holes, w, h, padPx = 0, consumable = null,
    shouldAbort = () => false, onProgress = () => {}, options = {} }) {
    const run = () => this._fill({ rgba, holes, w, h, padPx, consumable, shouldAbort, onProgress, options });
    const p = (this._mutex ?? Promise.resolve()).then(run, run);
    this._mutex = p.catch(() => {});
    return p;
  }

  async _fill({ rgba, holes, w, h, padPx, consumable, shouldAbort, onProgress, options }) {
    const abort = () => {
      if (shouldAbort()) {
        const e = new Error('fill aborted (superseded build)');
        e.name = 'AbortError';
        throw e;
      }
    };
    const total = 1 + (padPx > 0 ? 1 : 0);
    let done = 0;
    let filled = new Uint8ClampedArray(rgba);
    const modelMask = new Uint8Array(w * h);
    const genMask = new Uint8Array(w * h);
    let any = 0;
    for (let i = 0; i < w * h; i++) {
      if (!holes[i]) continue;
      modelMask[i] = 1;
      any++;
      // The mask the MODEL sees includes the collar (so it cannot use the
      // occluder as context); the pixels we KEEP are only the consumable ones.
      // Compositing the collar back repaints 18 px of a person's face.
      if (!consumable || consumable[i]) genMask[i] = 1;
    }
    if (any > 0) {
      abort();
      const res = await this._call(filled, modelMask, w, h, options, onProgress, shouldAbort);
      const out = res.rgba;
      for (let i = 0; i < w * h; i++) {
        if (!genMask[i]) continue;
        filled[i * 4] = out[i * 4]; filled[i * 4 + 1] = out[i * 4 + 1];
        filled[i * 4 + 2] = out[i * 4 + 2]; filled[i * 4 + 3] = 255;
      }
      onProgress({ phase: 'run', done: ++done, total, info: res.info });
    } else {
      onProgress({ phase: 'run', done: ++done, total });
    }

    let plate = null, plateInit = null, ring = null, pw = 0, ph = 0;
    if (padPx > 0) {
      abort();
      pw = w + 2 * padPx; ph = h + 2 * padPx;
      let plateSrc = filled;
      if (consumable) {
        // discard contract: model output survives only at consumable pixels
        plateSrc = new Uint8ClampedArray(filled);
        for (let i = 0; i < w * h; i++) {
          if (holes[i] && !consumable[i]) {
            plateSrc[i * 4] = rgba[i * 4]; plateSrc[i * 4 + 1] = rgba[i * 4 + 1];
            plateSrc[i * 4 + 2] = rgba[i * 4 + 2]; plateSrc[i * 4 + 3] = rgba[i * 4 + 3];
          }
        }
      }
      ({ plate } = padPlate(plateSrc, w, h, padPx));
      plateInit = plate.slice();
      ring = ringMask(pw, ph, padPx);
      const res = await this._call(plate, ring, pw, ph, options, onProgress, shouldAbort);
      plate = res.rgba;
      onProgress({ phase: 'run', done: ++done, total, info: res.info });
    }
    return { filled, genMask, plate, plateInit, ring, pw, ph };
  }
}
