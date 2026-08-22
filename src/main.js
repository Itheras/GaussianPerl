// App bootstrap (M8): wires the layered-heightfield renderer, translation-only
// window camera, pipeline worker, and UI.

import { LayerRenderer } from './render/renderer.js';
import { WindowCam } from './controls/window-cam.js';
import { clamp } from './util/math3d.js';
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

const settings = {
  quality: urlParams.get('quality') || defaultQuality(),
  boost: 1,             // baseline gain on the motion envelope (never geometry)
  aperture: 0,          // sharp by default — the photo had no synthetic DoF
  wiggle: !urlParams.has('nowiggle'),
  withBg: true,
  withSkirt: true,
  // nomodel = no AI at all (depth + fill); nofill = generative fill only
  aiDepth: !urlParams.has('nomodel'),
  aiFill: !urlParams.has('nomodel') && !urlParams.has('nofill'),
};

const app = {
  layers: null,          // renderer-side layer set (kept for pickFocus reads)
  meta: null,
  source: null,          // {sample:bool, blob?:Blob}
  imageData: null,       // working-resolution ImageData of current source
  intrinsics: null,      // {fPx, fovXDeg, fovYDeg, f35, source} — capture FoV
  natW: 0, natH: 0,
  disparity: null,       // Float32Array at working res (gt), or null
  disparityKind: 'none',
  sourceId: 0,
  dConv: 0.5,            // convergence (pivot) disparity
  focusDist: 1,          // subject units
  targetFocus: 1,
  buildId: 0,
  setupId: 0,
  needsRender: true,
  aiBroken: false,
  aiFillBroken: false,
  workerDead: false,
};

// ---------- worker ----------
let pipelineWorker = null;

function onPipelineMessage(e) {
  const msg = e.data;
  feedWatchdog();
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
    stopWatchdog();
  }
}

function spawnPipelineWorker() {
  pipelineWorker = new Worker(new URL('./pipeline/pipeline-worker.js', import.meta.url), { type: 'module' });
  pipelineWorker.onmessage = onPipelineMessage;
  pipelineWorker.onerror = (e) => {
    console.error('pipeline worker failed:', e.message || e);
    stopWatchdog();
    app.workerDead = true;
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
function onWatchdog() {
  if (!watchdogArmed) return;
  stopWatchdog();
  try { pipelineWorker.terminate(); } catch { /* already dead */ }
  spawnPipelineWorker();
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
const cam = new WindowCam(canvas, {
  onChange: () => { app.needsRender = true; },
  onPick: (cx, cy) => pickFocus(cx, cy),
});
cam.wiggle = settings.wiggle;

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
  status('Building the scene…', true);
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

// per-photo motion envelope: keep demanded disocclusion within what the fills
// cover (the stretched-wall fallback tolerates a moderate overrun — hence the
// 1.75x allowance), and within the stereography parallax budget (~3% of frame)
function applyEnvelope() {
  const m = app.meta;
  if (!m) return;
  const f = fPxWorking();
  const base = 0.05 * settings.boost;
  const parallaxCap = 0.03 * m.w * m.dSub / Math.max(f * (m.dMax - m.dMin), 1e-6);
  const bandW = m.bandFrac * Math.min(m.w, m.h);
  const holeCap = m.dMax - app.dConv > 1e-4
    ? 1.75 * bandW * m.dSub / Math.max(f * (m.dMax - app.dConv), 1e-6)
    : base;
  cam.setEnvelope(Math.max(0.006, Math.min(base, parallaxCap, holeCap)));
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
    w: m.w, h: m.h, padPx: m.padPx, padD: m.padD,
  };
  app.layers = layers;
  try { renderer.setLayers(layers); } catch (err) { console.error('setLayers failed:', err); }

  const revisit = m.phase === 'final' && app.setupId === msg.id;
  if (!revisit) {
    app.dConv = m.dSub;
    app.targetFocus = app.focusDist = 1; // subject plane
    syncFocusSlider();
    app.setupId = msg.id;
    cam.home();
  }
  applyEnvelope();
  app.needsRender = true;
  if (m.phase === 'preview') {
    status('Enhancing hidden areas with AI…', true);
  } else {
    stopWatchdog();
    status(revisit ? '✨ AI fill applied' : 'Ready — drag to look around', false, 3200);
  }
  $('welcome').hidden = true;
  showHint();
}

let hintShown = false;
function showHint() {
  if (hintShown) return;
  hintShown = true;
  const el = $('hint');
  el.hidden = false;
  setTimeout(() => { el.style.opacity = '0'; }, 7000);
  setTimeout(() => { el.hidden = true; }, 7800);
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
function pickFocus(clientX, clientY) {
  const L = app.layers, m = app.meta;
  if (!L || !m) return;
  const rect = canvas.getBoundingClientRect();
  const vpU = (clientX - rect.left) / rect.width;
  const vpV = (clientY - rect.top) / rect.height;
  // same contain-fit mapping the shader uses (v measured from the top here)
  const imgAspect = m.w / m.h;
  const vpAspect = rect.width / Math.max(rect.height, 1);
  let fsx = 1, fsy = 1;
  if (vpAspect > imgAspect) fsx = vpAspect / imgAspect;
  else fsy = imgAspect / vpAspect;
  const u = (vpU - 0.5) * fsx + 0.5;
  const v = (vpV - 0.5) * fsy + 0.5;
  if (u < 0 || u > 1 || v < 0 || v > 1) return;
  const dx = Math.min(Math.max(Math.round(u * m.dw), 0), m.dw - 1) + m.padD;
  const dy = Math.min(Math.max(Math.round(v * m.dh), 0), m.dh - 1) + m.padD;
  const d = L.disp0[dy * m.pdw + dx];
  if (!Number.isFinite(d)) return;
  // re-pivot AND refocus on the tapped surface
  app.dConv = Math.min(Math.max(d, m.dMin), m.dMax);
  app.targetFocus = clamp(m.dSub / Math.max(d, m.dFloor), 0.05, 50);
  applyEnvelope();
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
  return {
    eye: cam.eye(),
    dConv: app.dConv,
    dSub: m.dSub,
    dMin: m.dMin,
    dMax: m.dMax,
    dFloor: m.dFloor,
    fPx: fPxWorking(),
    steps: isMobile() ? 28 : 48,
    focusDist: app.focusDist,
    dofStrength: dof,
    maxCoC: DEFAULTS.maxCoC * px,
    bgTop: DEFAULTS.bgTop,
    bgBottom: DEFAULTS.bgBottom,
  };
}

let lastT = performance.now();
function frame(t) {
  const dt = (t - lastT) / 1000;
  lastT = t;
  const moved = cam.update(dt);
  if (moved) app.needsRender = true;
  if (Math.abs(app.focusDist - app.targetFocus) > 1e-3) {
    app.focusDist += (app.targetFocus - app.focusDist) * Math.min(dt * 8, 1);
    app.needsRender = true;
  }
  if (app.needsRender && app.layers && app.meta) {
    app.needsRender = false;
    renderer.render(renderState());
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
$('btnReset').onclick = () => { cam.reset(); app.dConv = app.meta ? app.meta.dSub : 0.5; applyEnvelope(); };
$('btnPanel').onclick = () => { $('panel').hidden = !$('panel').hidden; };

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

$('sDepth').addEventListener('input', (e) => {
  settings.boost = parseFloat(e.target.value);
  applyEnvelope();
  app.needsRender = true;
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

bindImageDrop(document.body, openBlob);

// ---------- boot ----------
onResize();
requestAnimationFrame(frame);
if (urlParams.has('demo')) openSample();

// expose for e2e tests
window.__gp = {
  app, cam, renderer, settings, openBlob,
  captureNow: (scale = 1) => renderer.capture(renderState(), scale),
};
