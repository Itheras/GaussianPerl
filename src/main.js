// App bootstrap (M9): free camera + progressive generative scene completion.
//
// The scene is a growing set of anchors. Anchor 0 is the photograph. When the
// camera comes to rest somewhere the photograph cannot explain, that view is
// captured, generatively completed, given depth, and committed as a new anchor
// — so the invented content is permanent and identical when you come back.

import { LayerRenderer } from './render/renderer.js';
import { FreeCam } from './controls/free-cam.js';
import { M3, intrinsicsK } from './render/pose.js';
import { clamp } from './util/math3d.js';
import { holeFraction, holeFractionRect } from './pipeline/novel-view.js';
import { POINT_MEMORY_STATE, ScenePointMemory } from './pipeline/point-prompt.js';
import { anchorToPoints, evaluatePointCandidate } from './pipeline/points.js';
import { QUALITY, DEFAULTS, defaultQuality, isMobile, isSafariEngine } from './config.js';
import { loadImageBlob, loadSample, bindImageDrop } from './io/load.js';
import { savePixelsAsPNG } from './io/save.js';

const $ = (id) => document.getElementById(id);
const urlParams = new URLSearchParams(location.search);

const canvas = $('gl');
let renderer;
try {
  renderer = new LayerRenderer(canvas);
} catch (err) {
  document.querySelector('#welcome p').textContent =
    'Sorry — this browser does not support WebGL2, which is required.';
  document.querySelector('.welcomeButtons').remove();
  throw err;
}

// Anchors are captured with a wider field of view than the visible frame, so a
// small extra move does not immediately demand another generation pass.
const ANCHOR_PAD = 0.12;
// Must sit at or below the shader's CONF_OK (0.55). Above it and there is a
// band of pixels that are simultaneously good enough to draw and bad enough to
// regenerate — a permanent annulus that costs a full generation pass forever.
const CONF_HOLE = 0.5;
// How much ORIGINAL PHOTOGRAPH must still be in frame. This — not a fixed
// angle — is the real governor, and it is scene-adaptive for free: a landscape
// keeps the photo dominant through a large orbit, a close-up portrait does not,
// and the same number expresses both. Below it, a new anchor would be built
// mostly out of earlier anchors' output, which is how these loops rot: each
// pass re-inpaints and re-estimates depth on the last pass's invention.
// Calibrated on a hard photo (two people filling the near field). Measured:
// at 0.88 and 0.64 the generated view is sharp and the subjects survive; at
// 0.50 the fill model has replaced them with mush. A committed anchor is
// PERMANENT pollution — there is no undo — so this threshold is deliberately
// the conservative side of the last known-good measurement.
const MIN_BASE_SHARE = 0.55;
// With a true inpainter the drift guard can be far looser: geometry stays
// consistent because a later anchor adopts an earlier anchor's depth as
// KNOWN, and a full-frame diffusion prior does not drift in tone the way a
// 512px GAN does. Going around a subject legitimately leaves well under
// half the frame as original photograph.
const MIN_BASE_SHARE_SIDECAR = 0.25;
const minBaseShare = () => (sidecarConfig() ? MIN_BASE_SHARE_SIDECAR : MIN_BASE_SHARE);
// Past this much missing, a 512px hole-filler is being asked to invent most of
// a photograph. It will produce mush, and that mush becomes a permanent anchor.
//
// This number is the whole ballgame. Lateral motion runs off the EDGE of the
// photo far sooner than it opens disocclusions: the source sample for a
// near-field pixel shifts by d*K*e/dSub, so at the frame border the nearest
// content leaves the plate after a couple of percent of subject distance —
// which is exactly why M8's envelope was ~3% and not timidity. M9 gets past it
// by outpainting a wider frame on every pass, so the covered region GROWS as
// you explore. That only works if each pass invents a modest band; ask for
// half a frame at once and the fill model returns mush.
const HOLE_CEILING = 0.30;
// A true inpainter with a scene prior copes with far larger holes than a 512px
// GAN (measured: SDXL fills 35% of a frame coherently). The ceiling only has
// to stop "invent most of a photograph" requests.
const HOLE_CEILING_SIDECAR = 0.45;
const holeCeiling = () => (sidecarConfig() ? HOLE_CEILING_SIDECAR : HOLE_CEILING);
const ARM_AT = 0.05;      // hysteresis: a binary per-pixel count crosses any
const DISARM_AT = 0.025;  // single threshold many times a second while moving

// Local inference sidecar (M11): dev passes ?sidecar=PORT:TOKEN, the desktop
// shell injects globalThis.__GP_SIDECAR__ = {url, token} before modules load.
function sidecarConfig() {
  if (globalThis.__GP_SIDECAR__) return globalThis.__GP_SIDECAR__;
  const v = urlParams.get('sidecar');
  if (!v) return null;
  const i = v.indexOf(':');
  if (i < 0) return null;
  return { url: `http://127.0.0.1:${v.slice(0, i)}`, token: v.slice(i + 1) };
}

const settings = {
  quality: urlParams.get('quality') || defaultQuality(),
  speed: 1,
  aperture: 0,          // sharp by default — the photo had no synthetic DoF
  wiggle: !urlParams.has('nowiggle'),
  withBg: true,
  withSkirt: true,
  autoFill: !urlParams.has('noauto'),
  freeRoam: urlParams.has('roam'),
  lookMode: false,
  // nomodel = no AI at all (depth + fill); nofill = generative fill only
  aiDepth: !urlParams.has('nomodel'),
  aiFill: !urlParams.has('nomodel') && !urlParams.has('nofill'),
  // generator knobs for the sidecar; a scene description helps a diffusion
  // inpainter enormously (dev: ?prompt=...; the app will ask the user / caption)
  // Measured on the beach photo: strength 0.99 from a flat grey init reads the
  // frame edge as an architectural wall; starting from the classical seed at
  // 0.9 continues the scene instead. 20 steps ~= 27 s at 768x1024 on M2 Max.
  fillOptions: {
    steps: 20, strength: 0.85, guidance: 7.0, seed: 1, init: 'seed',
    prompt: urlParams.get('prompt') || '',
    // what is revealed beside a subject is BACKGROUND by definition; without
    // this the model fills a person-shaped gap with a person
    negative: 'person, people, human figure, man, woman, face, extra limbs, extra heads, '
      + 'hat, cap, headwear, blurry, smeared, distorted, deformed, text, watermark, frame, border',
  },
};

// Permanent CPU context for the future camera-conditioned point guesser. The
// renderer's texture array remains a deliberately tiny GPU cache; evicting a
// texture must never erase a committed observation from the model's prompt.
const sceneMemory = new ScenePointMemory();

const app = {
  layers: null,
  meta: null,
  source: null,
  imageData: null,
  intrinsics: null,
  natW: 0, natH: 0,
  disparity: null,
  disparityKind: 'none',
  sourceId: 0,
  focusDist: 1,
  targetFocus: 1,
  buildId: 0,
  setupId: 0,
  needsRender: true,
  aiBroken: false,
  aiFillBroken: false,
  workerDead: false,
  // scene growth
  anchors: [],           // {slot, pose, born}
  expandId: 0,
  expandInFlight: false,
  pendingPose: null,
  lastProbeAt: -1e9,
  lastExpandAt: -1e9,
  restSince: 0,
  holeFrac: 0,
  expandFailed: false,
  armed: false,
  holeBefore: 0,
  barren: [],            // poses where generating provably did not help
  lastGood: null,        // furthest pose the scene could actually explain
  recovering: null,      // rubber-band animation back to it
  baseShare: 1,          // how much of the view is still the photograph
};

// ---------- worker ----------
let pipelineWorker = null;

function onPipelineMessage(e) {
  const msg = e.data;
  feedWatchdog();
  if (msg.type === 'expanded' || msg.type === 'expand-skipped' ||
      msg.type === 'expand-failed' || msg.type === 'expand-progress') {
    onExpandMessage(msg);
    return;
  }
  if (msg.id !== app.buildId) return; // stale build
  if (msg.type === 'progress') {
    const pct = msg.pct !== undefined ? ` ${Math.round(msg.pct * 100)}%` : '';
    const stageText = {
      'depth-download': `Downloading depth model…${pct}`,
      depth: 'Estimating depth (on-device AI)…',
      normalize: 'Normalizing depth…', refine: 'Refining depth…',
      heuristic: 'Estimating depth (heuristic)…', edges: 'Cleaning silhouettes…',
      inpaint: 'Synthesizing hidden areas…',
      'fill-download': `Downloading fill model…${pct}`,
      fill: msg.total ? `Generative fill… ${msg.done}/${msg.total}` : 'Generative fill…',
      build: 'Assembling layers…', done: 'Almost there…',
    }[msg.stage] || 'Working…';
    status(stageText, true);
  } else if (msg.type === 'built') {
    onBuilt(msg);
  } else if (msg.type === 'fill-failed') {
    console.warn('fill failed:', msg.message);
    status('AI fill unavailable — using fast fill', false, 4000);
    stopWatchdog();
  } else if (msg.type === 'error') {
    console.error('pipeline error:', msg.message);
    status('Something went wrong building the scene', false, 5000);
    app.expandInFlight = false;
    app.pendingPose = null;
    stopWatchdog();
  }
}

function spawnPipelineWorker() {
  // cache-busted in dev: a stale worker module silently runs old code
  const workerUrl = new URL('./pipeline/pipeline-worker.js', import.meta.url);
  if (urlParams.has('sidecar') || urlParams.has('dev')) workerUrl.searchParams.set('v', String(Date.now()));
  pipelineWorker = new Worker(workerUrl, { type: 'module' });
  pipelineWorker.onmessage = onPipelineMessage;
  pipelineWorker.onerror = (e) => {
    console.error('pipeline worker failed:', e.message || e);
    stopWatchdog();
    app.workerDead = true;
    app.expandInFlight = false;
    status('Pipeline worker failed — reload the page', false, 8000);
  };
  pipelineWorker.onmessageerror = (e) => console.error('pipeline messageerror:', e);
}
spawnPipelineWorker();

// Watchdog: wasm model inference blocks the worker; a wedged call can never
// time itself out. Quiet too long -> respawn without AI.
let watchdogTimer = 0;
let watchdogArmed = false;
function feedWatchdog() {
  if (!watchdogArmed) return;
  clearTimeout(watchdogTimer);
  watchdogTimer = setTimeout(onWatchdog, 150000);
}
function armWatchdog() {
  watchdogArmed = true;
  feedWatchdog();
}
function stopWatchdog() {
  watchdogArmed = false;
  clearTimeout(watchdogTimer);
}
let watchdogJob = 'build';   // what armed it: a hung EXPANSION must not be
                             // treated as a hung base build (that disables AI
                             // for the session and throws every anchor away)
function onWatchdog() {
  if (!watchdogArmed) return;
  stopWatchdog();
  try { pipelineWorker.terminate(); } catch { /* already dead */ }
  spawnPipelineWorker();
  app.expandInFlight = false;
  if (watchdogJob === 'expand') {
    console.error('expansion watchdog tripped — generator hung');
    app.pendingPose = null;
    app.expandFailed = true;
    status('Could not generate that view', false, 3500);
    return;
  }
  const previewStands = app.meta && app.meta.phase === 'preview' && app.setupId === app.buildId;
  if (previewStands) {
    console.error('fill watchdog tripped — keeping the classical preview');
    app.aiFillBroken = true;
    status('AI fill hung — keeping fast fill', false, 4000);
  } else {
    console.error('pipeline watchdog tripped — respawning worker without AI');
    app.aiBroken = true;
    status('AI hung — rebuilding without it', false, 4000);
    kickBuild();
  }
}

// ---------- camera ----------
const cam = new FreeCam(canvas, {
  onChange: () => {
    app.needsRender = true;
    app.restSince = performance.now();
    app.recovering = null;   // the user is driving; never fight them
  },
  onPick: (cx, cy) => pickFocus(cx, cy),
});
cam.wiggle = settings.wiggle;
cam.freeRoam = settings.freeRoam;

// ---------- status ----------
let statusTimer = 0;
function status(text, busy = false, autoHideMs = 0) {
  const el = $('status');
  clearTimeout(statusTimer);
  if (!text) { el.hidden = true; return; }
  el.hidden = false;
  el.textContent = text;
  el.classList.toggle('busy', busy);
  if (autoHideMs) statusTimer = setTimeout(() => { el.hidden = true; }, autoHideMs);
}

// ---------- pipeline flow ----------
function maxPixels() {
  const q = QUALITY[settings.quality] || QUALITY.medium;
  const override = parseInt(urlParams.get('maxpx') || '', 10);
  return Number.isFinite(override) && override >= 10000 ? override : q.maxPixels;
}

function buildParams() {
  const q = QUALITY[settings.quality] || QUALITY.medium;
  return {
    edgeDispJump: DEFAULTS.edgeDispJump,
    farKnee: DEFAULTS.farKnee,
    farKeep: DEFAULTS.farKeep,
    bgBandPx: DEFAULTS.bgBandPx,
    skirtPx: DEFAULTS.skirtPx,
    withBg: settings.withBg,
    withSkirt: settings.withSkirt,
    wantAiDepth: settings.aiDepth && !app.aiBroken,
    aiFill: settings.aiFill && !app.aiBroken && !app.aiFillBroken,
    deviceClass: isMobile() ? 'mobile' : 'desktop',
    forceWasmFill: urlParams.get('fillep') === 'wasm',
    hq: urlParams.has('hq'),   // opt in to the non-commercial depth checkpoint
    sidecar: sidecarConfig(),
    webkitHint: isSafariEngine() ||
      (navigator.userAgent.includes('Macintosh') && navigator.maxTouchPoints > 2),
    quality: q,
  };
}

function kickBuild() {
  if (!app.imageData) return;
  if (app.workerDead) {
    status('Pipeline worker failed — reload the page', false, 8000);
    return;
  }
  app.buildId++;
  // a rebuild invalidates any expansion in flight (the worker drops it)
  app.expandInFlight = false;
  app.pendingPose = null;
  status('Building the scene…', true);
  watchdogJob = 'build';
  armWatchdog();
  const rgbaCopy = app.imageData.data.slice();
  const dispCopy = app.disparity ? app.disparity.slice() : null;
  const transfer = [rgbaCopy.buffer];
  if (dispCopy) transfer.push(dispCopy.buffer);
  pipelineWorker.postMessage({
    type: 'build', id: app.buildId, sourceId: app.sourceId,
    rgba: rgbaCopy.buffer,
    w: app.imageData.width, h: app.imageData.height,
    disparity: dispCopy ? dispCopy.buffer : null,
    params: buildParams(),
  }, transfer);
}

function modelInfoText(meta) {
  const fov = app.intrinsics
    ? ` · fov ${Math.max(app.intrinsics.fovXDeg, app.intrinsics.fovYDeg).toFixed(0)}°${app.intrinsics.source === 'exif' ? '' : '*'}`
    : '';
  const depth = meta.depthKind === 'ai'
    ? `depth: DA V2 ${meta.depthTier} · ${meta.depthBackend}`
    : meta.depthKind === 'gt' ? 'depth: bundled ground truth'
      : 'depth: heuristic (AI model unavailable)';
  const fill = meta.fillKind === 'ai'
    ? ` · fill: MI-GAN · ${meta.fillBackend}`
    : meta.fillKind === 'classical' ? ' · fill: classical' : '';
  return depth + fill + fov;
}

/** working-res focal length in pixels */
function fPxWorking() {
  if (!app.intrinsics || !app.meta) return Math.max(app.meta?.w || 1, app.meta?.h || 1);
  const scale = app.natW ? app.meta.w / app.natW : 1;
  return app.intrinsics.fPx * scale;
}

function baseK() {
  const m = app.meta;
  return intrinsicsK(fPxWorking(), m.w, m.h);
}

function anchorK() {
  const k = baseK();
  const s = 1 / (1 + 2 * ANCHOR_PAD);
  return [k[0] * s, k[1] * s];
}

function anchorDims() {
  const m = app.meta;
  const maxPx = isMobile() ? 700_000 : 1_800_000;
  const w0 = m.w * (1 + 2 * ANCHOR_PAD), h0 = m.h * (1 + 2 * ANCHOR_PAD);
  const s = Math.min(1, Math.sqrt(maxPx / (w0 * h0)));
  return { cw: Math.max(64, Math.round(w0 * s)), ch: Math.max(64, Math.round(h0 * s)) };
}

function anchorCapacity() {
  // GPU memory, not ambition: each anchor is a colour slice plus a disparity
  // slice, and iOS sheds the WebGL context long before it runs out of ideas.
  return isMobile() ? 2 : 4;
}

function onBuilt(msg) {
  const m = msg.meta;
  app.meta = m;
  app.disparityKind = m.depthKind === 'heuristic' ? 'none' : m.depthKind;
  $('modelInfo').textContent = modelInfoText(m);

  const layers = {
    color0: msg.color0, disp0: msg.disp0,
    color1: msg.color1, disp1: msg.disp1,
    pw: m.pw, ph: m.ph, pdw: m.pdw, pdh: m.pdh,
    w: m.w, h: m.h, dw: m.dw, dh: m.dh, padPx: m.padPx, padD: m.padD,
  };
  app.layers = layers;
  try { renderer.setLayers(layers); } catch (err) { console.error('setLayers failed:', err); }

  const revisit = m.phase === 'final' && app.setupId === msg.id;
  if (!revisit) {
    // the base geometry changed: every generated anchor was derived from the
    // old one, so they are all stale
    resetAnchors();
    sceneMemory.clear();
    app.lastGood = null;
    app.recovering = null;
    cam.K = baseK();
    cam.setHome({ pivotZ: -1, dist: 1 });
    app.targetFocus = app.focusDist = 1;
    syncFocusSlider();
    app.setupId = msg.id;
    cam.home();
    commitBaseObservation();
  }
  app.needsRender = true;
  if (m.phase === 'preview') {
    status('Enhancing hidden areas with AI…', true);
  } else {
    stopWatchdog();
    status(revisit ? '✨ AI fill applied' : 'Ready — drag to orbit', false, 3200);
  }
  $('welcome').hidden = true;
  showHint();
}

function commitBaseObservation() {
  const m = app.meta;
  const L = app.layers;
  if (!m || !L || !app.imageData) return;
  // Geometry is stored at depth resolution inside a padded texture. Colour is
  // the exact source photo at working resolution; both span the same 0..1 UV.
  const disp = new Float32Array(m.dw * m.dh);
  for (let y = 0; y < m.dh; y++) {
    const start = (y + m.padD) * m.pdw + m.padD;
    disp.set(L.disp0.subarray(start, start + m.dw), y * m.dw);
  }
  sceneMemory.commit({
    R: M3.identity(), C: [0, 0, 0], K: baseK(),
    disp, w: m.dw, h: m.dh,
    color: app.imageData.data, colorW: m.w, colorH: m.h,
  }, {
    id: 0,
    observed: true,
    state: POINT_MEMORY_STATE.SOURCE,
    provenance: 'source-photo',
    uncertainty: app.disparityKind === 'gt' ? 0 : 0.15,
  });
  updateAnchorInfo();
}

let hintShown = false;
function showHint() {
  if (hintShown) return;
  hintShown = true;
  const el = $('hint');
  el.hidden = false;
  setTimeout(() => { el.style.opacity = '0'; }, 9000);
  setTimeout(() => { el.hidden = true; }, 9800);
}

// ---------- scene growth ----------
function resetAnchors() {
  const { cw, ch } = anchorDims();
  const dims = anchorDepthDimsFor(cw, ch);
  renderer.initAnchorStore({ cw, ch, dw: dims.dw, dh: dims.dh, capacity: anchorCapacity() });
  app.anchors = [];
  app.expandInFlight = false;
  app.pendingPose = null;
  app.expandFailed = false;
  updateAnchorInfo();
}

// mirrors expand.js's anchorDepthDims so the texture array matches the payload
function anchorDepthDimsFor(w, h, maxPixels = 480_000) {
  const s = Math.min(1, Math.sqrt(maxPixels / (w * h)));
  return { dw: Math.max(8, Math.round(w * s)), dh: Math.max(8, Math.round(h * s)) };
}

function anchorState(withDof = false) {
  const st = renderState();
  return {
    ...st,
    cam: { R: st.cam.R, C: st.cam.C, K: anchorK() },
    trustBase: false,
    dofStrength: withDof ? st.dofStrength : 0,
  };
}

// the visible frame occupies this fraction inset inside an anchor capture
const VISIBLE_INSET = ANCHOR_PAD / (1 + 2 * ANCHOR_PAD);

/**
 * Cheap look at how much of what the VIEWER SEES nothing can explain.
 * Deliberately the photo frame, not the wider anchor frame: the border ring an
 * anchor capture reaches into is unknown by construction and must never be the
 * thing that triggers a generation pass.
 */
function probeHoles() {
  if (!app.meta || !app.layers) return 0;
  const m = app.meta;
  const s = Math.min(1, Math.sqrt(26000 / (m.w * m.h)));
  const pw = Math.max(24, Math.round(m.w * s)), ph = Math.max(24, Math.round(m.h * s));
  const st = renderState();
  const cap = renderer.captureAnchorFrame({
    ...st, cam: { R: st.cam.R, C: st.cam.C, K: baseK() },
    trustBase: false, dofStrength: 0,
  }, pw, ph);
  if (!cap) return 0;
  app.holeFrac = holeFraction(cap.conf, CONF_HOLE);
  let bs = 0;
  for (let i = 0; i < cap.baseShare.length; i++) bs += cap.baseShare[i];
  app.baseShare = bs / cap.baseShare.length;
  return app.holeFrac;
}

function nearestAnchorDistance() {
  if (!app.anchors.length) return Infinity;
  const pose = cam.pose();
  let best = Infinity;
  for (const a of app.anchors) {
    const dx = a.pose.C[0] - pose.C[0], dy = a.pose.C[1] - pose.C[1], dz = a.pose.C[2] - pose.C[2];
    const r = M3.multiply(a.pose.R, M3.transpose(pose.R));
    const ang = Math.acos(Math.max(-1, Math.min(1, (r[0] + r[4] + r[8] - 1) / 2)));
    best = Math.min(best, Math.hypot(dx, dy, dz) + ang * (6 / Math.PI));
  }
  return best;
}

/**
 * Poses where a generation pass measurably failed to help. Without this, a
 * viewpoint whose holes lie outside every anchor's reach re-triggers forever:
 * generate, no improvement, generate again.
 */
function isBarren() {
  const pose = cam.pose();
  const fwd = M3.mulVecT(pose.R, [0, 0, -1]);
  for (const b of app.barren) {
    const dx = b.C[0] - pose.C[0], dy = b.C[1] - pose.C[1], dz = b.C[2] - pose.C[2];
    if (Math.hypot(dx, dy, dz) > 0.15) continue;
    if (b.fwd[0] * fwd[0] + b.fwd[1] * fwd[1] + b.fwd[2] * fwd[2] > 0.98) return true;
  }
  return false;
}

function markBarren() {
  const pose = cam.pose();
  app.barren.push({ C: Array.from(pose.C), fwd: Array.from(M3.mulVecT(pose.R, [0, 0, -1])) });
  if (app.barren.length > 24) app.barren.shift();
}

function requestExpand(force = false) {
  if (!app.meta || !app.layers || app.workerDead) return false;
  if (app.expandInFlight) return false;
  if (app.meta.phase === 'preview') return false;   // base fill still running
  if (app.expandFailed && !force) return false;
  const { cw, ch } = anchorDims();
  const cap = renderer.captureAnchorFrame(anchorState(), cw, ch);
  if (!cap) return false;
  const frac = holeFractionRect(cap.conf, cw, ch, VISIBLE_INSET, CONF_HOLE);
  app.holeFrac = frac;
  // force still needs SOMETHING to do — check the whole capture, border included
  const anyHole = holeFraction(cap.conf, CONF_HOLE);
  if (!force && frac < 0.015) return false;
  if (force && anyHole < 0.002) return false;
  if (frac > holeCeiling()) {
    status('Too far from the photo to fill this in', false, 2500);
    return false;
  }
  // never build an anchor out of other anchors' output
  let bs = 0;
  for (let i = 0; i < cap.baseShare.length; i++) bs += cap.baseShare[i];
  if (bs / cap.baseShare.length < minBaseShare()) {
    status('Too little of the original photo left here', false, 2500);
    return false;
  }
  app.holeBefore = frac;

  const dSub = app.meta.dSub, dFloor = app.meta.dFloor;
  const refDisp = new Float32Array(cw * ch);
  for (let i = 0; i < refDisp.length; i++) {
    const z = cap.depth[i];
    refDisp[i] = (z > 1e-4 && Number.isFinite(z)) ? Math.min(dSub / z, 4) : dFloor;
  }
  const rgba = cap.rgba instanceof Uint8ClampedArray
    ? new Uint8ClampedArray(cap.rgba) : cap.rgba;

  const pose = cam.pose();
  app.pendingPose = { R: Float32Array.from(pose.R), C: Float32Array.from(pose.C), K: anchorK() };
  app.expandInFlight = true;
  app.expandId++;
  app.lastExpandAt = performance.now();
  status('Generating what the camera never saw…', true);

  pipelineWorker.postMessage({
    type: 'expand', id: app.expandId,
    rgba: rgba.buffer, conf: cap.conf.buffer, refDisp: refDisp.buffer,
    w: cw, h: ch,
    params: {
      confThreshold: CONF_HOLE,
      aiFill: settings.aiFill && !app.aiBroken && !app.aiFillBroken,
      wantAiDepth: settings.aiDepth && !app.aiBroken,
      deviceClass: isMobile() ? 'mobile' : 'desktop',
      forceWasmFill: urlParams.get('fillep') === 'wasm',
      webkitHint: isSafariEngine() ||
        (navigator.userAgent.includes('Macintosh') && navigator.maxTouchPoints > 2),
      edgeDispJump: DEFAULTS.edgeDispJump,
      farKnee: DEFAULTS.farKnee,
      farKeep: DEFAULTS.farKeep,
      dispRange: Math.max(app.meta.dMax - app.meta.dMin, 1e-3),
      sidecar: sidecarConfig(),
      fillOptions: settings.fillOptions,
    },
  }, [rgba.buffer, cap.conf.buffer, refDisp.buffer]);
  watchdogJob = 'expand';
  armWatchdog();
  return true;
}

function onExpandMessage(msg) {
  if (msg.type === 'expand-progress') {
    if (msg.id !== app.expandId) return;
    if (msg.stage === 'fill' && msg.phase === 'wait') {
      status('Generating (model warming up)…', true);
    } else if (msg.stage === 'fill' && msg.total > 1) {
      status(`Generating… ${msg.done}/${msg.total}`, true);
    } else if (msg.stage === 'depth') {
      status('Giving the new view depth…', true);
    }
    return;
  }
  if (msg.id !== app.expandId) return;
  app.expandInFlight = false;
  stopWatchdog();
  if (msg.type === 'expand-skipped') { status('', false); return; }
  if (msg.type === 'expand-failed') {
    console.warn('expand failed:', msg.message);
    app.expandFailed = true;
    status('Could not generate that view', false, 3500);
    return;
  }

  const pose = app.pendingPose;
  if (!pose) return;
  if (app.anchors.length >= anchorCapacity()) evictFarthestAnchor();
  const slot = renderer.addAnchor({
    R: pose.R, C: pose.C, K: pose.K,
    dMin: msg.stats.dMin, dMax: msg.stats.dMax,
    color: msg.color, disp: msg.disp,
    weight: 0,
  });
  if (slot < 0) { status('', false); return; }
  // keep the payload: iOS sheds the GL context under memory pressure, and
  // re-uploading is the difference between a blink and losing everything the
  // user explored
  const anchorRecord = {
    slot, pose, born: performance.now(), stats: msg.stats,
    color: msg.color, disp: msg.disp,
  };
  app.anchors.push(anchorRecord);
  const dims = renderer.genDims;
  const proposal = {
    R: pose.R, C: pose.C, K: pose.K,
    disp: msg.disp, w: msg.dw || dims.dw, h: msg.dh || dims.dh,
    color: msg.color, colorW: dims.cw, colorH: dims.ch,
    dMin: msg.stats.dMin, dMax: msg.stats.dMax,
  };

  // A full-looking render is not enough to become scene truth. First require
  // measurable coverage, then test the proposed geometry against every nearby
  // committed witness. Empty clouds, known-surface copies, and geometry that a
  // trusted camera sees through all refuse instead of polluting the prompt.
  renderer.anchors.forEach((a) => { if (a.slot === slot) a.weight = 1; });
  const after = probeHoles();
  renderer.anchors.forEach((a) => { if (a.slot === slot) a.weight = 0; });
  const points = anchorToPoints(proposal, {
    dSub: app.meta.dSub, dFloor: app.meta.dFloor,
    stride: 4, edgeJump: DEFAULTS.edgeDispJump,
  });
  const witnesses = sceneMemory.retrieve(pose, 4, {
    // One generated guess must not become evidence against an equally
    // plausible branch. Only observations (and future externally-confirmed
    // views) are allowed to veto a proposal.
    states: [POINT_MEMORY_STATE.SOURCE, POINT_MEMORY_STATE.CONFIRMED],
  }).map((entry) => entry.anchor);
  const pointGate = evaluatePointCandidate(points.positions, witnesses, {
    dSub: app.meta.dSub,
    tol: 0.025,
  });
  const noRenderGain = app.holeBefore > 0 && after > 0.7 * app.holeBefore;
  if (noRenderGain || !pointGate.accepted) {
    renderer.removeAnchor(slot);
    app.anchors = app.anchors.filter((a) => a !== anchorRecord);
    markBarren();
    probeHoles();
    app.needsRender = true;
    updateAnchorInfo();
    const why = noRenderGain ? 'no coverage gain' : pointGate.reasons.join(', ');
    console.warn('generated view refused:', why, pointGate);
    status('Generated view refused — evidence did not support it', false, 3200);
    app.armed = false;
    return;
  }

  anchorRecord.stats = { ...msg.stats, pointGate };
  const memoryEntry = sceneMemory.commit(proposal, {
    observed: false,
    state: POINT_MEMORY_STATE.SPECULATIVE,
    branchId: 'live',
    seed: settings.fillOptions.seed,
    provenance: msg.stats.fillBackend || msg.stats.fillKind || 'generated-view',
    uncertainty: msg.stats.fillKind === 'ai' ? 0.35 : 0.8,
    stats: anchorRecord.stats,
  });
  anchorRecord.memoryId = memoryEntry.id;
  app.needsRender = true;
  updateAnchorInfo();
  status(msg.stats.fillKind === 'ai' ? '✨ Filled in' : 'Filled in', false, 1800);
  app.armed = false;
}

function evictFarthestAnchor() {
  const pose = cam.pose();
  let worst = null, worstD = -1;
  for (const a of app.anchors) {
    const dx = a.pose.C[0] - pose.C[0], dy = a.pose.C[1] - pose.C[1], dz = a.pose.C[2] - pose.C[2];
    const d = Math.hypot(dx, dy, dz);
    if (d > worstD) { worstD = d; worst = a; }
  }
  if (!worst) return;
  renderer.removeAnchor(worst.slot);
  app.anchors = app.anchors.filter((a) => a !== worst);
}

function updateAnchorInfo() {
  const el = $('anchorInfo');
  if (!el) return;
  const n = app.anchors.length;
  const remembered = Math.max(0, sceneMemory.size - 1);
  const via = app.anchors.some((a) => a.stats && a.stats.fillBackend === 'sidecar') ? ' · SDXL sidecar' : '';
  const memory = remembered > n ? ` · ${remembered} remembered` : '';
  el.textContent = n ? `${n} resident view${n === 1 ? '' : 's'}${memory}${via}`
    : remembered ? `${remembered} remembered view${remembered === 1 ? '' : 's'}`
      : 'no generated views yet';
}

/**
 * Make the nearest retained views resident in the fixed-size WebGL texture
 * array. This closes the old semantic bug where GPU eviction also made an
 * explored view impossible to revisit. Source geometry is rendered through
 * the immutable base layers, so only generated entries need slots here.
 */
function syncAnchorResidency() {
  if (!renderer.genDims || !sceneMemory.size) return;
  const desired = sceneMemory.retrieve(cam.pose(), sceneMemory.size)
    .filter((entry) => entry.state !== POINT_MEMORY_STATE.SOURCE &&
      (entry.state === POINT_MEMORY_STATE.CONFIRMED || entry.branchId === 'live'))
    .slice(0, anchorCapacity());
  const desiredIds = new Set(desired.map((entry) => entry.id));
  let changed = false;

  for (const resident of app.anchors.slice()) {
    if (resident.memoryId == null || desiredIds.has(resident.memoryId)) continue;
    renderer.removeAnchor(resident.slot);
    app.anchors = app.anchors.filter((a) => a !== resident);
    changed = true;
  }

  for (const entry of desired) {
    if (app.anchors.some((a) => a.memoryId === entry.id)) continue;
    if (app.anchors.length >= anchorCapacity()) evictFarthestAnchor();
    const a = entry.anchor;
    const slot = renderer.addAnchor({
      R: a.R, C: a.C, K: a.K,
      dMin: a.dMin, dMax: a.dMax,
      color: a.color, disp: a.disp,
      weight: 1,
    });
    if (slot < 0) continue;
    app.anchors.push({
      slot,
      pose: entry.pose,
      born: performance.now() - FADE_MS,
      stats: entry.stats,
      color: a.color,
      disp: a.disp,
      memoryId: entry.id,
    });
    changed = true;
  }

  if (changed) {
    app.needsRender = true;
    updateAnchorInfo();
  }
}

// Every open flow takes a token; after each await, a newer token means the
// user opened something else — abandon, never mix state from two images.
let loadToken = 0;

async function openBlob(blob) {
  const token = ++loadToken;
  try {
    status('Reading photo…', true);
    const { imageData, natW, natH, intrinsics } = await loadImageBlob(blob, maxPixels());
    if (token !== loadToken) return;
    app.source = { sample: false, blob };
    app.imageData = imageData;
    app.natW = natW; app.natH = natH;
    app.intrinsics = intrinsics;
    app.disparity = null;
    app.disparityKind = 'none';
    app.sourceId++;
    kickBuild();
  } catch (err) {
    console.error(err);
    if (token === loadToken) status('Could not open that image', false, 5000);
  }
}

async function openSample() {
  const token = ++loadToken;
  try {
    status('Loading sample…', true);
    const { imageData, disparity, natW, natH } = await loadSample(maxPixels());
    if (token !== loadToken) return;
    app.source = { sample: true };
    app.imageData = imageData;
    app.natW = natW ?? imageData.width; app.natH = natH ?? imageData.height;
    // the procedural sample's GT depth was generated at exactly 55° vertical
    const fPx = (app.natH) / (2 * Math.tan((DEFAULTS.fovYDeg * Math.PI / 180) / 2));
    app.intrinsics = {
      fPx,
      fovYDeg: DEFAULTS.fovYDeg,
      fovXDeg: 2 * Math.atan(app.natW / (2 * fPx)) * 180 / Math.PI,
      f35: null, source: 'sample',
    };
    app.disparity = disparity;
    app.disparityKind = 'gt';
    app.sourceId++;
    kickBuild();
  } catch (err) {
    console.error(err);
    if (token === loadToken) status('Could not load the sample', false, 5000);
  }
}

async function reopenCurrent() {
  if (!app.source) return;
  if (app.source.sample) return openSample();
  return openBlob(app.source.blob);
}

// ---------- focus / pivot picking ----------
function viewportToImageUv(clientX, clientY) {
  const m = app.meta;
  const rect = canvas.getBoundingClientRect();
  const vpU = (clientX - rect.left) / rect.width;
  const vpV = (clientY - rect.top) / rect.height;
  const imgAspect = m.w / m.h;
  const vpAspect = rect.width / Math.max(rect.height, 1);
  let fsx = 1, fsy = 1;
  if (vpAspect > imgAspect) fsx = vpAspect / imgAspect;
  else fsy = imgAspect / vpAspect;
  return [(vpU - 0.5) * fsx + 0.5, (vpV - 0.5) * fsy + 0.5];
}

function pickFocus(clientX, clientY) {
  if (!app.layers || !app.meta) return;
  const [u, v] = viewportToImageUv(clientX, clientY);
  if (u < 0 || u > 1 || v < 0 || v > 1) return;
  const hit = renderer.probeAt(renderState(), u, v);
  if (!hit || !(hit.depth > 1e-3)) return;

  // re-pivot onto the tapped surface AND refocus there
  const pose = cam.pose();
  const K = pose.K;
  const dir = [(u - 0.5) / K[0], -(v - 0.5) / K[1], -1];
  const dirW = M3.mulVecT(pose.R, dir);
  cam.setPivot([
    pose.C[0] + dirW[0] * hit.depth,
    pose.C[1] + dirW[1] * hit.depth,
    pose.C[2] + dirW[2] * hit.depth,
  ]);
  app.targetFocus = clamp(hit.depth, 0.05, 50);
  syncFocusSlider();
  app.needsRender = true;

  const ring = $('focusRing');
  ring.hidden = false;
  ring.style.left = `${clientX}px`;
  ring.style.top = `${clientY}px`;
  ring.style.animation = 'none';
  void ring.offsetWidth;
  ring.style.animation = '';
  setTimeout(() => { ring.hidden = true; }, 600);
}

function focusRange() {
  const m = app.meta;
  const near = m ? m.dSub / Math.max(m.dMax, 1e-3) : 0.3;
  const far = m ? m.dSub / Math.max(m.dFloor, 1e-3) : 8;
  return { near: Math.max(near * 0.8, 0.05), far: far * 1.1 };
}

function syncFocusSlider() {
  const { near, far } = focusRange();
  const t = clamp(Math.log(app.targetFocus / near) / Math.log(far / near), 0, 1);
  $('sFocus').value = String(t);
}

// ---------- render loop ----------
function renderState() {
  const m = app.meta;
  const px = Math.min(devicePixelRatio || 1, 2);
  const dof = settings.aperture * settings.aperture * 240 * px;
  const mobile = isMobile();
  return {
    cam: cam.pose(),
    baseK: baseK(),
    dSub: m.dSub,
    dMin: m.dMin,
    dMax: m.dMax,
    dFloor: m.dFloor,
    steps: mobile ? 30 : 52,
    stepsGen: mobile ? 20 : 34,
    trustBase: cam.atHome(),
    focusDist: app.focusDist,
    dofStrength: dof,
    maxCoC: DEFAULTS.maxCoC * px,
    bgTop: DEFAULTS.bgTop,
    bgBottom: DEFAULTS.bgBottom,
    grainSigma: m.noiseSigma || [0, 0, 0],
  };
}

const FADE_MS = 260;

// Keep the camera out of the scene's own geometry. One-fragment probe down the
// view axis, throttled — a full-rate readback would stall the pipeline, and a
// purely analytic guard cannot know where the surfaces actually are.
let lastClearanceAt = -1e9;
function enforceClearance(t, force = false) {
  if (!app.meta || !app.layers) return;
  if (!force && t - lastClearanceAt < 80) return;
  lastClearanceAt = t;
  const hit = renderer.probeAt(renderState(), 0.5, 0.5);
  if (!hit || !(hit.depth > 1e-4)) return;
  const near = app.meta.dSub / Math.max(app.meta.dMax, 1e-3);
  const minZ = Math.max(0.1, near * 0.55);
  if (hit.depth < minZ) {
    cam.backOff((minZ - hit.depth) * 1.15);
    app.needsRender = true;
  }
}

let lastT = performance.now();
let lastResidencyAt = -1e9;
function frame(t) {
  const dt = (t - lastT) / 1000;
  lastT = t;
  const moved = cam.update(dt);
  if (moved) { app.needsRender = true; app.restSince = t; }
  if (moved && !app.expandInFlight && t - lastResidencyAt > 120) {
    lastResidencyAt = t;
    syncAnchorResidency();
  }
  if (moved || t - app.restSince < 200) enforceClearance(t);
  if (Math.abs(app.focusDist - app.targetFocus) > 1e-3) {
    app.focusDist += (app.targetFocus - app.focusDist) * Math.min(dt * 8, 1);
    app.needsRender = true;
  }
  // fade newly committed anchors in rather than popping
  for (const a of app.anchors) {
    const rec = renderer.anchors.find((r) => r.slot === a.slot);
    if (!rec || rec.weight >= 1) continue;
    rec.weight = Math.min(1, (t - a.born) / FADE_MS);
    app.needsRender = true;
  }
  if (app.needsRender && app.layers && app.meta) {
    app.needsRender = false;
    renderer.render(renderState());
  }
  // once the camera settles, fill in whatever the scene could not explain
  if (settings.autoFill && app.layers && app.meta && !app.expandInFlight &&
      t - app.restSince > 420 && t - app.lastProbeAt > 900 && t - app.lastExpandAt > 700) {
    app.lastProbeAt = t;
    const d = probeHoles();
    const thin = app.baseShare < minBaseShare();
    if (d < DISARM_AT && !thin) { app.armed = false; app.lastGood = cam.snapshot(); }
    else if (d > ARM_AT) app.armed = true;
    if ((d > holeCeiling() || thin) && !settings.freeRoam && app.lastGood && !app.recovering) {
      // Past the frontier of what the scene can explain. Rubber-band back
      // rather than committing an anchor that is mostly invention — the
      // frontier moves outward on its own as anchors accumulate.
      app.recovering = { from: cam.snapshot(), to: app.lastGood, t: 0 };
      status(thin ? 'Mostly invention out here — easing back'
        : 'That is past what this photo can show — easing back', false, 2600);
    } else if (app.armed && d <= holeCeiling() && app.baseShare >= minBaseShare() &&
               nearestAnchorDistance() > 0.05 && !isBarren()) {
      requestExpand();
    }
  }

  if (app.recovering) {
    const R = app.recovering;
    R.t = Math.min(1, R.t + dt / 0.45);
    const e = R.t * R.t * (3 - 2 * R.t);
    const lerp = (a, b) => a + (b - a) * e;
    cam.pivot = [0, 1, 2].map((i) => lerp(R.from.pivot[i], R.to.pivot[i]));
    cam.yaw = lerp(R.from.yaw, R.to.yaw);
    cam.pitch = lerp(R.from.pitch, R.to.pitch);
    cam.dist = lerp(R.from.dist, R.to.dist);
    cam.velYaw = 0; cam.velPitch = 0;
    app.needsRender = true;
    if (R.t >= 1) app.recovering = null;
  }
  requestAnimationFrame(frame);
}

function onResize() {
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  renderer.resize(w, h);
  app.needsRender = true;
}
window.addEventListener('resize', onResize);
if (window.visualViewport) visualViewport.addEventListener('resize', onResize);

// iOS Safari sheds GL contexts under memory pressure — recover transparently
renderer.onContextLost = () => status('Graphics context lost — recovering…', true);
renderer.onContextRestored = () => {
  if (app.layers) renderer.setLayers(app.layers);
  // the texture arrays died with the context — rebuild and re-upload, do not
  // silently discard everything the user generated
  const keep = app.anchors.slice();
  resetAnchors();
  for (const a of keep) {
    if (!a.color || !a.disp) continue;
    const slot = renderer.addAnchor({
      R: a.pose.R, C: a.pose.C, K: a.pose.K,
      dMin: a.stats.dMin, dMax: a.stats.dMax,
      color: a.color, disp: a.disp, weight: 1,
    });
    if (slot >= 0) app.anchors.push({ ...a, slot });
  }
  updateAnchorInfo();
  onResize();
  status('Recovered', false, 2000);
};

// ---------- UI wiring ----------
$('btnOpen').onclick = () => $('file').click();
$('btnWelcomeOpen').onclick = () => $('file').click();
$('btnSample').onclick = openSample;
$('btnWelcomeSample').onclick = openSample;
$('file').onchange = (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) openBlob(f);
  e.target.value = '';
};
$('btnReset').onclick = () => { cam.reset(); app.needsRender = true; };
$('btnPanel').onclick = () => { $('panel').hidden = !$('panel').hidden; };
$('btnFill').onclick = () => {
  app.expandFailed = false;
  if (!requestExpand(true)) status('Nothing missing from this view', false, 2000);
};

$('btnSave').onclick = async () => {
  if (!app.layers) { status('Nothing to save yet', false, 2500); return; }
  try {
    status('Rendering PNG…', true);
    await new Promise((r) => requestAnimationFrame(r));
    const cap = renderer.capture(renderState(), 2);
    if (!cap) throw new Error('graphics context lost');
    await savePixelsAsPNG(cap.pixels, cap.width, cap.height, pngName());
    app.needsRender = true;
    status('Saved ✓', false, 2500);
  } catch (err) {
    console.error(err);
    status('Save failed', false, 4000);
  }
};

function pngName() {
  const d = new Date();
  const p = (x) => String(x).padStart(2, '0');
  return `gaussianperl-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}.png`;
}

$('sSpeed').addEventListener('input', (e) => {
  settings.speed = parseFloat(e.target.value);
  cam.orbitSpeed = 2.6 * settings.speed;
  cam.flySpeed = 0.9 * settings.speed;
});
$('sAperture').addEventListener('input', (e) => {
  settings.aperture = parseFloat(e.target.value);
  app.needsRender = true;
});
$('sFocus').addEventListener('input', (e) => {
  const { near, far } = focusRange();
  const t = parseFloat(e.target.value);
  app.targetFocus = near * Math.pow(far / near, t);
});
$('selQuality').value = settings.quality;
$('selQuality').addEventListener('change', (e) => {
  settings.quality = e.target.value;
  reopenCurrent();
});
for (const [id, key] of [['tBg', 'withBg'], ['tSkirt', 'withSkirt'], ['tAiFill', 'aiFill']]) {
  $(id).addEventListener('change', (e) => {
    settings[key] = e.target.checked;
    kickBuild();
  });
}
$('tAiFill').checked = settings.aiFill;
$('tWiggle').checked = settings.wiggle;
$('tWiggle').addEventListener('change', (e) => {
  settings.wiggle = e.target.checked;
  cam.wiggle = settings.wiggle;
});
$('tAuto').checked = settings.autoFill;
$('tAuto').addEventListener('change', (e) => { settings.autoFill = e.target.checked; });
$('tRoam').checked = settings.freeRoam;
$('tRoam').addEventListener('change', (e) => {
  settings.freeRoam = e.target.checked;
  cam.freeRoam = settings.freeRoam;
});
$('iPrompt').value = settings.fillOptions.prompt || '';
$('iPrompt').addEventListener('input', (e) => { settings.fillOptions.prompt = e.target.value; });
$('tLook').checked = settings.lookMode;
$('tLook').addEventListener('change', (e) => {
  settings.lookMode = e.target.checked;
  cam.lookMode = settings.lookMode;
});

bindImageDrop(document.body, openBlob);

// ---------- boot ----------
onResize();
requestAnimationFrame(frame);
if (urlParams.has('demo')) openSample();

// expose for e2e tests
window.__gp = {
  app, cam, renderer, settings, sceneMemory, openBlob,
  captureNow: (scale = 1) => renderer.capture(renderState(), scale),
  probeHoles,
  requestExpand,
  renderState,
  anchorState,
  pointPrompt: (options = {}) => sceneMemory.buildPrompt(cam.pose(), options),
};
