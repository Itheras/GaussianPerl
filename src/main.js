// App bootstrap: wires renderer, controls, workers, pipeline, and UI.

import { SplatRenderer } from './render/renderer.js';
import { OrbitControls } from './controls/orbit.js';
import { M4, clamp } from './util/math3d.js';
import { QUALITY, DEFAULTS, defaultQuality, isMobile, isSafariEngine } from './config.js';
import { loadImageBlob, loadSample, bindImageDrop } from './io/load.js';
import { savePixelsAsPNG, downloadBlob } from './io/save.js';

const $ = (id) => document.getElementById(id);
const urlParams = new URLSearchParams(location.search);

const canvas = $('gl');
let renderer;
try {
  renderer = new SplatRenderer(canvas);
} catch (err) {
  document.querySelector('#welcome p').textContent =
    'Sorry — this browser does not support WebGL2, which is required.';
  document.querySelector('.welcomeButtons').remove();
  throw err;
}

const settings = {
  quality: urlParams.get('quality') || defaultQuality(),
  depthStrength: 1,
  splatScale: 1,
  aperture: 0.25,
  fovYDeg: DEFAULTS.fovYDeg,
  wiggle: !urlParams.has('nowiggle'),
  withBg: true,
  withSkirt: true,
  withUnder: true,
  // nomodel = no AI at all (depth + fill); nofill = generative fill only
  aiDepth: !urlParams.has('nomodel'),
  aiFill: !urlParams.has('nomodel') && !urlParams.has('nofill'),
};

const app = {
  cloud: null,
  meta: null,
  source: null,          // {sample:bool, blob?:Blob}
  imageData: null,       // working-resolution ImageData of current source
  disparity: null,       // Float32Array at working res (gt), or null (worker estimates)
  disparityKind: 'none', // 'ai' | 'gt' | 'none' — mirrored from build meta
  sourceId: 0,           // bumps per opened image/quality; keys the worker's stage cache
  focusDist: 2,
  targetFocus: 2,
  buildId: 0,
  setupId: 0,            // last build id whose PREVIEW/first result set up the view
  needsRender: true,
  sortBusy: false,
  sortDirty: false,
  sortGen: 0,
  spareIdx: null,
  lastSortView: null,
  pendingCloud: null,    // final AI rebuild parked until its sort arrives
  aiBroken: false,       // watchdog tripped pre-preview: ALL AI off this session
  aiFillBroken: false,   // watchdog tripped mid-fill: only the fill off
  workerDead: false,     // worker script itself failed — reload required
};

// ---------- workers ----------
const sortWorker = new Worker(new URL('./render/sort-worker.js', import.meta.url), { type: 'module' });
let pipelineWorker = null;

function onPipelineMessage(e) {
  const msg = e.data;
  // any message proves the worker is alive — feed BEFORE all filtering
  // (stale-build progress during a long wasm fill must not false-trip)
  feedWatchdog();
  // export replies (success OR error) are routed by their own id, never
  // filtered by buildId — otherwise an export error strands the button
  if (msg.id === 'export') {
    if (msg.type === 'exported') {
      downloadBlob(new Blob([msg.bytes], { type: 'application/octet-stream' }), 'gaussianperl.splat');
      status('Saved .splat', false, 2500);
    } else if (msg.type === 'error') {
      console.error('export error:', msg.message);
      status('Export failed', false, 4000);
    }
    $('btnExport').disabled = false;
    return;
  }
  if (msg.id !== app.buildId) return; // stale build
  if (msg.type === 'progress') {
    const pct = msg.pct !== undefined ? ` ${Math.round(msg.pct * 100)}%` : '';
    const stageText = {
      'depth-download': `Downloading depth model…${pct}`,
      depth: 'Estimating depth (on-device AI)…',
      normalize: 'Normalizing depth…', refine: 'Refining depth edges…',
      heuristic: 'Estimating depth (heuristic)…', edges: 'Finding silhouettes…',
      inpaint: 'Synthesizing hidden areas…',
      'fill-download': `Downloading fill model…${pct}`,
      fill: msg.total ? `Generative fill… ${msg.done}/${msg.total}` : 'Generative fill…',
      build: 'Building splats…', done: 'Almost there…',
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
    status('Something went wrong building the splat', false, 5000);
    stopWatchdog();
  }
}

function spawnPipelineWorker() {
  pipelineWorker = new Worker(new URL('./pipeline/pipeline-worker.js', import.meta.url), { type: 'module' });
  pipelineWorker.onmessage = onPipelineMessage;
  pipelineWorker.onerror = (e) => {
    console.error('pipeline worker failed:', e.message || e);
    // stop the watchdog or it would respawn-and-fail in a loop forever
    stopWatchdog();
    app.workerDead = true;
    status('Pipeline worker failed — reload the page', false, 8000);
    $('btnExport').disabled = false;
  };
  pipelineWorker.onmessageerror = (e) => console.error('pipeline messageerror:', e);
}
spawnPipelineWorker();

// Watchdog: wasm model inference blocks the worker, so a wedged ORT call can
// never time itself out. If a build goes quiet for too long, respawn the
// worker without AI and rebuild — the app must never end up hung.
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
  // the terminated worker takes any in-flight export with it
  $('btnExport').disabled = false;
  // if this build's classical preview is already on screen, the hang was in
  // the FILL stage: keep the standing cloud and camera, disable only the fill
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

sortWorker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type !== 'sorted') return;
  app.sortBusy = false;
  // gen must match AND the index count must match the target cloud —
  // a sort of the previous cloud must never index the new one
  if (msg.indices && msg.gen === app.sortGen) {
    const pend = app.pendingCloud;
    if (pend && msg.indices.length === pend.cloud.count) {
      // promote the parked AI-final: swapping only when its sort is ready
      // means the first visible frame is fully sorted (no popping flicker)
      app.cloud = pend.cloud;
      app.meta = pend.meta;
      app.pendingCloud = null;
      try { renderer.setCloud(app.cloud); } catch (err) { console.error('setCloud failed:', err); }
      renderer.setSortedIndices(msg.indices);
      app.spareIdx = msg.indices;
      app.needsRender = true;
    } else if (!pend && app.cloud && msg.indices.length === app.cloud.count) {
      renderer.setSortedIndices(msg.indices);
      app.spareIdx = msg.indices;
      app.needsRender = true;
    }
  }
  maybeSort();
};
sortWorker.onerror = (e) => {
  console.error('sort worker failed:', e.message || e);
  app.sortBusy = false;
};

function requestSort() {
  app.sortDirty = true;
  maybeSort();
}

function maybeSort() {
  if (app.sortBusy || !app.sortDirty || !app.cloud) return;
  app.sortBusy = true;
  app.sortDirty = false;
  const view = controls.viewMatrix(new Float32Array(16));
  app.lastSortView = view.slice();
  app.sortGen++;
  const transfer = [];
  let idx = null;
  if (app.spareIdx && app.spareIdx.length === app.cloud.count) {
    idx = app.spareIdx;
    transfer.push(idx.buffer);
    app.spareIdx = null;
  }
  sortWorker.postMessage({ type: 'sort', view, gen: app.sortGen, indices: idx }, transfer);
}

function checkSortNeeded() {
  if (!app.cloud || !app.lastSortView) { if (app.cloud) requestSort(); return; }
  const v = controls.viewMatrix(tmpView);
  const l = app.lastSortView;
  const dirDot = v[2] * l[2] + v[6] * l[6] + v[10] * l[10];
  const eyeDx = v[12] - l[12], eyeDy = v[13] - l[13], eyeDz = v[14] - l[14];
  if (dirDot < 0.99995 || (eyeDx * eyeDx + eyeDy * eyeDy + eyeDz * eyeDz) > 1e-4) {
    requestSort();
  }
}

// ---------- controls ----------
const controls = new OrbitControls(canvas, {
  onChange: () => { app.needsRender = true; },
  onPick: (cx, cy) => pickFocus(cx, cy),
});
controls.wiggle = settings.wiggle;

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
    fovYDeg: DEFAULTS.fovYDeg,
    zNear: DEFAULTS.zNearScene,
    zRange: DEFAULTS.zRange * settings.depthStrength,
    sizeFactor: 0.65,
    edgeDispJump: DEFAULTS.edgeDispJump,
    farKnee: DEFAULTS.farKnee,
    farKeep: DEFAULTS.farKeep,
    bgBandPx: DEFAULTS.bgBandPx,
    skirtPx: DEFAULTS.skirtPx,
    underStep: DEFAULTS.underlayerStep,
    withBg: settings.withBg,
    withSkirt: settings.withSkirt,
    withUnder: settings.withUnder,
    wantAiDepth: settings.aiDepth && !app.aiBroken,
    aiFill: settings.aiFill && !app.aiBroken && !app.aiFillBroken,
    deviceClass: isMobile() ? 'mobile' : 'desktop',
    forceWasmFill: urlParams.get('fillep') === 'wasm',
    // Chrome on iPadOS wears the desktop-Mac UA — only maxTouchPoints here on
    // the main thread can unmask it; the worker must never run the JSEP/webgpu
    // ORT builds on a WebKit engine (ORT #26827/#26480)
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
  app.pendingCloud = null; // a newer build supersedes any parked final swap
  status('Building splats…', true);
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
  const depth = meta.depthKind === 'ai'
    ? `depth: DA V2 ${meta.depthTier} · ${meta.depthBackend}`
    : meta.depthKind === 'gt' ? 'depth: bundled ground truth'
      : 'depth: heuristic (AI model unavailable)';
  const fill = meta.fillKind === 'ai'
    ? ` · fill: MI-GAN · ${meta.fillBackend}`
    : meta.fillKind === 'classical' ? ' · fill: classical' : '';
  return depth + fill;
}

function onBuilt(msg) {
  // buffers arrive TRANSFERRED from the worker — use them directly
  // (a deep copy here would cost ~110 MB at 'high')
  const cloud = {
    count: msg.count,
    positions: msg.positions,
    cov: msg.cov,
    colors: msg.colors,
  };
  app.disparityKind = msg.meta.depthKind === 'heuristic' ? 'none' : msg.meta.depthKind;
  $('modelInfo').textContent = modelInfoText(msg.meta);

  // a 'final' rebuild for a build we already set up (AI fill arriving after
  // the preview): keep showing the sorted preview and PARK the final until
  // its own sort lands — no unsorted popping frames, no camera/focus yank
  const revisit = msg.meta.phase === 'final' && app.setupId === msg.id && app.cloud;
  if (revisit) {
    app.pendingCloud = { cloud, meta: msg.meta };
    app.sortGen++; // in-flight sorts of the preview must not promote/apply
    app.spareIdx = null;
    const posCopy = cloud.positions.slice();
    sortWorker.postMessage(
      { type: 'points', positions: posCopy, count: cloud.count }, [posCopy.buffer]);
    app.lastSortView = null;
    requestSort();
    stopWatchdog();
    status(`✨ AI fill applied · ${(msg.count / 1e6).toFixed(2)}M splats`, false, 3200);
    return;
  }

  app.cloud = cloud;
  app.meta = msg.meta;
  app.pendingCloud = null;
  // invalidate any in-flight sort of the previous cloud NOW — its reply must
  // not index into the new textures
  app.sortGen++;
  app.spareIdx = null;
  try { renderer.setCloud(app.cloud); } catch (err) { console.error('setCloud failed:', err); }
  const posCopy = app.cloud.positions.slice();
  sortWorker.postMessage(
    { type: 'points', positions: posCopy, count: app.cloud.count },
    [posCopy.buffer]);

  controls.setHome(-msg.meta.centerZ, msg.meta.centerZ);
  app.targetFocus = app.focusDist = msg.meta.centerZ;
  syncFocusSlider();
  app.setupId = msg.id;
  app.lastSortView = null;
  app.needsRender = true;
  requestSort();
  if (msg.meta.phase === 'preview') {
    status('Enhancing hidden areas with AI…', true);
  } else {
    stopWatchdog();
    status(`${(msg.count / 1e6).toFixed(2)}M splats`, false, 3200);
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
// Depth estimation happens in the pipeline worker (part of the build).
let loadToken = 0;

async function openBlob(blob) {
  const token = ++loadToken;
  try {
    status('Reading photo…', true);
    const { imageData } = await loadImageBlob(blob, maxPixels());
    if (token !== loadToken) return;
    app.source = { sample: false, blob };
    app.imageData = imageData;
    app.disparity = null; // worker estimates (AI or heuristic fallback)
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
    const { imageData, disparity } = await loadSample(maxPixels());
    if (token !== loadToken) return;
    app.source = { sample: true };
    app.imageData = imageData;
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

// ---------- focus picking ----------
const tmpView = new Float32Array(16);

function pickFocus(clientX, clientY) {
  if (!app.cloud) return;
  const rect = canvas.getBoundingClientRect();
  const xN = ((clientX - rect.left) / rect.width) * 2 - 1;
  const yN = 1 - ((clientY - rect.top) / rect.height) * 2;
  const v = controls.viewMatrix(tmpView);
  const eye = controls.eye();
  const tanF = Math.tan((settings.fovYDeg * Math.PI / 180) / 2);
  const aspect = rect.width / Math.max(rect.height, 1);
  // camera basis from view-matrix rows
  const right = [v[0], v[4], v[8]];
  const up = [v[1], v[5], v[9]];
  const back = [v[2], v[6], v[10]];
  let dx = right[0] * xN * tanF * aspect + up[0] * yN * tanF - back[0];
  let dy = right[1] * xN * tanF * aspect + up[1] * yN * tanF - back[1];
  let dz = right[2] * xN * tanF * aspect + up[2] * yN * tanF - back[2];
  const dl = Math.hypot(dx, dy, dz) || 1;
  dx /= dl; dy /= dl; dz /= dl;

  const P = app.cloud.positions;
  const n = app.cloud.count;
  const stride = Math.max(1, Math.floor(n / 250000));
  const tanPick = 0.03; // ~1.7° cone
  let bestT = Infinity;
  for (let i = 0; i < n; i += stride) {
    const rx = P[i * 3] - eye[0], ry = P[i * 3 + 1] - eye[1], rz = P[i * 3 + 2] - eye[2];
    const t = rx * dx + ry * dy + rz * dz;
    if (t < 0.05 || t >= bestT) continue;
    const perp2 = rx * rx + ry * ry + rz * rz - t * t;
    const rad = t * tanPick;
    if (perp2 < rad * rad) bestT = t;
  }
  if (!Number.isFinite(bestT)) return;
  const px = eye[0] + dx * bestT, py = eye[1] + dy * bestT, pz = eye[2] + dz * bestT;
  const s = -(v[2] * px + v[6] * py + v[10] * pz + v[14]);
  app.targetFocus = clamp(s, 0.15, 500);
  syncFocusSlider();

  const ring = $('focusRing');
  ring.hidden = false;
  ring.style.left = `${clientX}px`;
  ring.style.top = `${clientY}px`;
  ring.style.animation = 'none';
  void ring.offsetWidth; // restart animation
  ring.style.animation = '';
  setTimeout(() => { ring.hidden = true; }, 600);
}

function focusRange() {
  const near = app.meta ? Math.max(app.meta.nearZ * 0.8, 0.2) : 0.5;
  const far = app.meta ? app.meta.farZ * 1.3 : 20;
  return { near, far };
}

function syncFocusSlider() {
  const { near, far } = focusRange();
  const t = clamp(Math.log(app.targetFocus / near) / Math.log(far / near), 0, 1);
  $('sFocus').value = String(t);
}

// ---------- render loop ----------
const proj = new Float32Array(16);

function renderState(w, h) {
  M4.perspective(settings.fovYDeg * Math.PI / 180, w / Math.max(h, 1), 0.1, 300, proj);
  // CoC is computed in render-target pixels: BOTH strength and cap must scale
  // with dpr or blur strength varies between 1x and 2x displays
  const px = Math.min(devicePixelRatio || 1, 2);
  const dof = settings.aperture * settings.aperture * 240 * px;
  return {
    view: controls.viewMatrix(tmpView),
    proj,
    splatScale: settings.splatScale,
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
  const moved = controls.update(dt);
  if (moved) app.needsRender = true;
  if (Math.abs(app.focusDist - app.targetFocus) > 1e-3) {
    app.focusDist += (app.targetFocus - app.focusDist) * Math.min(dt * 8, 1);
    app.needsRender = true;
  }
  checkSortNeeded();
  if (app.needsRender && app.cloud) {
    app.needsRender = false;
    renderer.render(renderState(canvas.width, canvas.height));
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
  if (app.cloud) {
    renderer.setCloud(app.cloud);
    app.sortGen++;
    app.spareIdx = null;
    app.lastSortView = null;
    requestSort();
  }
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
$('btnReset').onclick = () => controls.reset();
$('btnPanel').onclick = () => { $('panel').hidden = !$('panel').hidden; };

$('btnSave').onclick = async () => {
  if (!app.cloud) { status('Nothing to save yet', false, 2500); return; }
  try {
    status('Rendering PNG…', true);
    await new Promise((r) => requestAnimationFrame(r));
    const cap = renderer.capture(renderState(canvas.width, canvas.height), 2);
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

$('btnExport').onclick = () => {
  if (!app.cloud) { status('Nothing to export yet', false, 2500); return; }
  $('btnExport').disabled = true;
  status('Encoding .splat…', true);
  const pos = app.cloud.positions.slice();
  const cov = app.cloud.cov.slice();
  const col = app.cloud.colors.slice();
  pipelineWorker.postMessage({
    type: 'export', id: 'export', count: app.cloud.count,
    positions: pos.buffer, cov: cov.buffer, colors: col.buffer,
  }, [pos.buffer, cov.buffer, col.buffer]);
};

$('sDepth').addEventListener('change', (e) => {
  settings.depthStrength = parseFloat(e.target.value);
  kickBuild();
});
$('sSize').addEventListener('input', (e) => {
  settings.splatScale = parseFloat(e.target.value);
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
$('sFov').addEventListener('input', (e) => {
  settings.fovYDeg = parseFloat(e.target.value);
  controls.fovY = settings.fovYDeg * Math.PI / 180; // keeps pan scale pixel-true
  app.needsRender = true;
});
$('selQuality').value = settings.quality;
$('selQuality').addEventListener('change', (e) => {
  settings.quality = e.target.value;
  reopenCurrent();
});
for (const [id, key] of [['tBg', 'withBg'], ['tSkirt', 'withSkirt'], ['tUnder', 'withUnder'], ['tAiFill', 'aiFill']]) {
  $(id).addEventListener('change', (e) => {
    settings[key] = e.target.checked;
    kickBuild();
  });
}
$('tAiFill').checked = settings.aiFill;
$('tWiggle').checked = settings.wiggle;
$('tWiggle').addEventListener('change', (e) => {
  settings.wiggle = e.target.checked;
  controls.wiggle = settings.wiggle;
});

bindImageDrop(document.body, openBlob);

// ---------- boot ----------
onResize();
requestAnimationFrame(frame);
if (urlParams.has('demo')) openSample();

// expose for e2e tests
window.__gp = {
  app, controls, renderer, settings, openBlob,
  captureNow: (scale = 1) => renderer.capture(renderState(canvas.width, canvas.height), scale),
};
