// Persistent scene memory and the binary prompt handed to a future learned
// completion model.
//
// The important analogy with an LLM is not "put every raw splat in a sentence".
// It is the contract:
//
//   committed scene state + requested camera pose -> sampled continuation
//
// The renderer's small GPU anchor array is only a cache. This object retains
// every committed observation/guess on the CPU, retrieves the views most
// relevant to a camera query, and serialises them as world-space point tokens.
// A model can later replace the sampler without changing scene identity,
// retrieval, provenance, or the refusal gate in points.js.

import { poseDistance } from '../render/pose.js';
import { anchorToPoints } from './points.js';

export const POINT_PROMPT_SCHEMA = 'gaussianperl.point-prompt.v1';
export const POINT_MEMORY_STATE = Object.freeze({
  SOURCE: 'source',
  CONFIRMED: 'confirmed',
  SPECULATIVE: 'speculative',
});

export const POINT_MEMORY_STATE_CODE = Object.freeze({
  source: 1,
  confirmed: 2,
  speculative: 3,
});

function copyPose(pose) {
  if (!pose || !pose.R || pose.R.length !== 9 || !pose.C || pose.C.length !== 3 ||
      !pose.K || pose.K.length !== 2) {
    throw new Error('pose must contain R[9], C[3], and K[2]');
  }
  return {
    R: Float32Array.from(pose.R),
    C: Float32Array.from(pose.C),
    K: Float32Array.from(pose.K),
  };
}

function validateAnchor(anchor) {
  copyPose(anchor);
  if (!Number.isInteger(anchor.w) || !Number.isInteger(anchor.h) ||
      anchor.w <= 0 || anchor.h <= 0) {
    throw new Error('anchor must have positive integer w and h');
  }
  if (!anchor.disp || anchor.disp.length !== anchor.w * anchor.h) {
    throw new Error('anchor disparity dimensions do not match w*h');
  }
  const colorW = anchor.colorW ?? anchor.w;
  const colorH = anchor.colorH ?? anchor.h;
  if (!Number.isInteger(colorW) || !Number.isInteger(colorH) || colorW <= 0 || colorH <= 0) {
    throw new Error('anchor color dimensions must be positive integers');
  }
  if (anchor.color && anchor.color.length !== colorW * colorH * 4) {
    throw new Error('anchor color dimensions do not match colorW*colorH*4');
  }
  if (anchor.rgb && anchor.rgb.length !== colorW * colorH * 3) {
    throw new Error('anchor RGB dimensions do not match colorW*colorH*3');
  }
}

function byte(v) {
  const x = Number.isFinite(v) ? v : 0;
  return Math.max(0, Math.min(255, Math.round(x <= 1 ? x * 255 : x)));
}

function writeColor(anchor, texel, out, offset) {
  const colorW = anchor.colorW ?? anchor.w;
  const colorH = anchor.colorH ?? anchor.h;
  const x = texel % anchor.w;
  const y = Math.floor(texel / anchor.w);
  const cx = Math.min(colorW - 1, Math.floor((x + 0.5) * colorW / anchor.w));
  const cy = Math.min(colorH - 1, Math.floor((y + 0.5) * colorH / anchor.h));
  if (anchor.color) {
    const j = (cy * colorW + cx) * 4;
    out[offset] = byte(anchor.color[j]);
    out[offset + 1] = byte(anchor.color[j + 1]);
    out[offset + 2] = byte(anchor.color[j + 2]);
    out[offset + 3] = byte(anchor.color[j + 3]);
    return out[offset + 3];
  }
  if (anchor.rgb) {
    const j = (cy * colorW + cx) * 3;
    out[offset] = byte(anchor.rgb[j]);
    out[offset + 1] = byte(anchor.rgb[j + 1]);
    out[offset + 2] = byte(anchor.rgb[j + 2]);
    out[offset + 3] = 255;
    return 255;
  }
  out[offset + 3] = 255;
  return 255;
}

function uniformIndex(k, take, count) {
  if (take <= 1) return 0;
  return Math.min(count - 1, Math.round(k * (count - 1) / (take - 1)));
}

/**
 * A non-evicting CPU scene memory. `payload` arrays are retained by reference
 * deliberately: copying a dense RGB-D anchor doubles memory for no semantic
 * benefit. Pose and metadata are snapshotted so later camera mutations cannot
 * silently rewrite the prompt.
 */
export class ScenePointMemory {
  constructor() {
    this._entries = [];
    this._nextId = 1;
  }

  get size() { return this._entries.length; }

  clear() {
    this._entries.length = 0;
    this._nextId = 1;
  }

  commit(anchor, metadata = {}) {
    validateAnchor(anchor);
    const id = metadata.id ?? this._nextId++;
    if (this._entries.some((x) => x.id === id)) throw new Error(`duplicate scene-memory id ${id}`);
    const pose = copyPose(anchor);
    const uncertainty = Math.max(0, Math.min(1, metadata.uncertainty ?? 0));
    const state = metadata.state || (metadata.observed
      ? POINT_MEMORY_STATE.SOURCE : POINT_MEMORY_STATE.SPECULATIVE);
    if (!(state in POINT_MEMORY_STATE_CODE)) throw new Error(`invalid point-memory state ${state}`);
    const entry = Object.freeze({
      id,
      pose,
      anchor: { ...anchor, ...pose },
      observed: metadata.observed ?? state === POINT_MEMORY_STATE.SOURCE,
      state,
      branchId: metadata.branchId ?? (state === POINT_MEMORY_STATE.SPECULATIVE ? id : null),
      seed: metadata.seed ?? null,
      stats: metadata.stats ?? null,
      uncertainty,
      provenance: metadata.provenance || (metadata.observed ? 'photo' : 'generated'),
      committedAt: metadata.committedAt ?? Date.now(),
    });
    this._entries.push(entry);
    return entry;
  }

  get(id) { return this._entries.find((x) => x.id === id) || null; }

  /** Nearest first. Retrieval does not mutate or evict committed state. */
  retrieve(targetPose, limit = 4, options = {}) {
    copyPose(targetPose); // validate without retaining the caller's arrays
    const n = Math.max(0, Math.floor(limit));
    const states = options.states ? new Set(options.states) : null;
    return this._entries.filter((entry) => !states || states.has(entry.state))
      .map((entry, order) => ({ entry, order, d: poseDistance(entry.pose, targetPose) }))
      .sort((a, b) => a.d - b.d || a.order - b.order)
      .slice(0, n)
      .map((x) => x.entry);
  }

  buildPrompt(targetPose, options = {}) {
    return buildPointPrompt(this.retrieve(targetPose, options.maxAnchors ?? 4), targetPose, options);
  }
}

/**
 * Serialise retrieved entries into a deterministic, model-ready point prompt.
 * The typed arrays form the binary payload; `anchors` and `target` are the
 * control header. A fair per-anchor budget prevents the nearest dense view
 * from consuming the entire context window.
 */
export function buildPointPrompt(entries, targetPose, options = {}) {
  const target = copyPose(targetPose);
  const selected = Array.from(entries || []);
  const maxPoints = Math.max(0, Math.floor(options.maxPoints ?? 200_000));
  const pointOptions = {
    dSub: options.dSub,
    dFloor: options.dFloor,
    stride: options.stride ?? 2,
    skipBelow: options.skipBelow,
    edgeJump: options.edgeJump ?? 0.055,
  };
  // Undefined values should not override anchorToPoints defaults.
  for (const key of Object.keys(pointOptions)) {
    if (pointOptions[key] === undefined) delete pointOptions[key];
  }

  const clouds = selected.map((entry, i) => {
    const anchor = entry.anchor || entry;
    validateAnchor(anchor);
    return {
      entry,
      anchor,
      cloud: anchorToPoints(anchor, pointOptions),
      // The binary payload deliberately uses dense numeric IDs. External IDs
      // may be strings, UUIDs, or numbers and are preserved in the header.
      sourceId: i + 1,
      externalId: entry.id ?? i + 1,
      observed: entry.observed ?? false,
      uncertainty: Math.max(0, Math.min(1, entry.uncertainty ?? 0)),
      state: entry.state || (entry.observed
        ? POINT_MEMORY_STATE.SOURCE : POINT_MEMORY_STATE.SPECULATIVE),
    };
  });

  const take = [];
  let remaining = maxPoints;
  for (let i = 0; i < clouds.length; i++) {
    const fair = Math.ceil(remaining / Math.max(1, clouds.length - i));
    const n = Math.min(clouds[i].cloud.count, fair);
    take.push(n);
    remaining -= n;
  }
  const count = take.reduce((a, b) => a + b, 0);
  const positions = new Float32Array(count * 3);
  const colors = new Uint8Array(count * 4);
  const sourceIds = new Uint32Array(count);
  const observed = new Uint8Array(count);
  const uncertainty = new Float32Array(count);
  const states = new Uint8Array(count);
  let outPoint = 0;
  for (let i = 0; i < clouds.length; i++) {
    const { entry, anchor, cloud, sourceId } = clouds[i];
    const n = take[i];
    for (let k = 0; k < n; k++) {
      const j = uniformIndex(k, n, cloud.count);
      positions[outPoint * 3] = cloud.positions[j * 3];
      positions[outPoint * 3 + 1] = cloud.positions[j * 3 + 1];
      positions[outPoint * 3 + 2] = cloud.positions[j * 3 + 2];
      const trust = writeColor(anchor, cloud.texel[j], colors, outPoint * 4) / 255;
      sourceIds[outPoint] = sourceId;
      observed[outPoint] = entry.observed ? 1 : 0;
      uncertainty[outPoint] = Math.max(
        Math.max(0, Math.min(1, entry.uncertainty ?? 0)),
        1 - trust,
      );
      states[outPoint] = POINT_MEMORY_STATE_CODE[clouds[i].state];
      outPoint++;
    }
  }

  return {
    schema: POINT_PROMPT_SCHEMA,
    target,
    count,
    positions,
    colors,
    sourceIds,
    observed,
    uncertainty,
    states,
    anchors: clouds.map(({ entry, sourceId, externalId, cloud }, i) => ({
      id: externalId,
      sourceId,
      provenance: entry.provenance || (entry.observed ? 'photo' : 'generated'),
      observed: !!entry.observed,
      uncertainty: Math.max(0, Math.min(1, entry.uncertainty ?? 0)),
      state: entry.state || (entry.observed
        ? POINT_MEMORY_STATE.SOURCE : POINT_MEMORY_STATE.SPECULATIVE),
      branchId: entry.branchId ?? null,
      seed: entry.seed ?? null,
      availablePoints: cloud.count,
      emittedPoints: take[i],
    })),
  };
}
