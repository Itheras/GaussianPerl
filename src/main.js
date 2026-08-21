// App bootstrap: wires renderer, controls, workers, pipeline, and UI.

import { SplatRenderer } from './render/renderer.js';
import { OrbitControls } from './controls/orbit.js';
import { M4, clamp } from './util/math3d.js';
import { QUALITY, DEFAULTS, defaultQuality } from './config.js';
import { DepthEstimator } from './pipeline/depth-ai.js';
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
};

const app = {
  cloud: null,
  meta: null,
  source: null,          // {sample:bool, blob?:Blob}
  imageData: null,       // working-resolution ImageData of current source
  disparity: null,       // Float32Array at working res (ai/gt), or null (heuristic)
  disparityKind: 'none', // 'ai' | 'gt' | 'none'
  focusDist: 2,
  targetFocus: 2,
  buildId: 0,
  needsRender: true,
  sortBusy: false,
  sortDirty: false,
  sortGen: 0,
  spareIdx: null,
  lastSortView: null,
  estimator: undefined,  // undefined = not tried, null = unavailable
};

// ---------- workers ----------
const pipelineWorker = new Worker(new URL('./pipeline/pipeline-worker.js', import.meta.url), { type: 'module' });
const sortWorker = new Worker(new URL('./render/sort-worker.js', import.meta.url), { type: 'module' });

pipelineWorker.onmessage = (e) => {
  const msg = e.data;
  if (msg.id !== app.buildId && msg.type !== 'exported') return; // stale build
  if (msg.type === 'progress') {
    const stageText = {
      normalize: 'Normalizing depth…', refine: 'Refining depth edges…',
      heuristic: 'Estimating depth (heuristic)…', edges: 'Finding silhouettes…',
      inpaint: 'Synthesizing hidden areas…', build: 'Building splats…', done: 'Almost there…',
    }[msg.stage] || 'Working…';
    status(stageText, true);
  } else if (msg.type === 'built') {
    onBuilt(msg);
  } else if (msg.type === 'exported') {
    downloadBlob(new Blob([msg.bytes], { type: 'application/octet-stream' }), 'gaussianperl.splat');
    status('Saved .splat', false, 2500);
    $('btnExport').disabled = false;
  } else if (msg.type === 'error') {
    console.error('pipeline error:', msg.message);
    status('Something went wrong building the splat', false, 5000);
    $('btnExport').disabled = false;
  }
};

sortWorker.onmessage = (e) => {
  const msg = e.data;
  if (msg.type !== 'sorted') return;
  app.sortBusy = false;
  if (msg.indices && msg.gen === app.sortGen) {
    renderer.setSortedIndices(msg.indices);
    app.spareIdx = msg.indices;
    app.needsRender = true;
  }
  maybeSort();
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
function buildParams() {
  const q = QUALITY[settings.quality] || QUALITY.medium;
  return {
    fovYDeg: DEFAULTS.fovYDeg,
    zNear: DEFAULTS.zNearScene,
    zRange: DEFAULTS.zRange * settings.depthStrength,
    sizeFactor: 0.65,
    edgeDispJump: DEFAULTS.edgeDispJump,
    bgBandPx: DEFAULTS.bgBandPx,
    skirtPx: DEFAULTS.skirtPx,
    underStep: DEFAULTS.underlayerStep,
    withBg: settings.withBg,
    withSkirt: settings.withSkirt,
    withUnder: settings.withUnder,
    refine: app.disparityKind === 'ai',
    quality: q,
  };
}

function kickBuild() {
  if (!app.imageData) return;
  app.buildId++;
  status('Building splats…', true);
  const rgbaCopy = app.imageData.data.slice();
  const dispCopy = app.disparity ? app.disparity.slice() : null;
  const transfer = [rgbaCopy.buffer];
  if (dispCopy) transfer.push(dispCopy.buffer);
  pipelineWorker.postMessage({
    type: 'build', id: app.buildId,
    rgba: rgbaCopy.buffer,
    w: app.imageData.width, h: app.imageData.height,
    disparity: dispCopy ? dispCopy.buffer : null,
    params: buildParams(),
  }, transfer);
}

function onBuilt(msg) {
  app.cloud = {
    count: msg.count,
    positions: new Float32Array(msg.positions),
    cov: new Float32Array(msg.cov),
    colors: new Uint8Array(msg.colors),
  };
  app.meta = msg.meta;
  renderer.setCloud(app.cloud);
  sortWorker.postMessage(
    { type: 'points', positions: app.cloud.positions.slice(), count: app.cloud.count },
    []);
  controls.setHome(-msg.meta.centerZ, msg.meta.centerZ);
  app.targetFocus = app.focusDist = msg.meta.centerZ;
  syncFocusSlider();
  app.lastSortView = null;
  app.needsRender = true;
  requestSort();
  status(`${(msg.count / 1e6).toFixed(2)}M splats`, false, 3200);
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

async function ensureEstimator() {
  if (urlParams.has('nomodel')) return null;
  if (app.estimator !== undefined) return app.estimator;
  status('Loading depth model (first time: ~25 MB)…', true);
  app.estimator = await DepthEstimator.load(({ phase, pct }) => {
    if (phase === 'download') {
      status(`Downloading depth model… ${Math.round(pct * 100)}%`, true);
    }
  });
  if (app.estimator) {
    $('modelInfo').textContent = `depth: Depth Anything V2 · ${app.estimator.backend}`;
  }
  return app.estimator;
}

async function openBlob(blob) {
  try {
    status('Reading photo…', true);
    const q = QUALITY[settings.quality] || QUALITY.medium;
    const { imageData } = await loadImageBlob(blob, q.maxPixels);
    app.source = { sample: false, blob };
    app.imageData = imageData;

    const est = await ensureEstimator();
    if (est) {
      status('Estimating depth (on-device AI)…', true);
      await new Promise((r) => setTimeout(r, 30)); // let the status paint
      app.disparity = await est.estimate(imageData, imageData.width, imageData.height);
      app.disparityKind = 'ai';
    } else {
      app.disparity = null;
      app.disparityKind = 'none';
      $('modelInfo').textContent = 'depth: heuristic (AI model unavailable)';
      status('AI model unavailable — using heuristic depth', true, 0);
    }
    kickBuild();
  } catch (err) {
    console.error(err);
    status('Could not open that image', false, 5000);
  }
}

async function openSample() {
  try {
    status('Loading sample…', true);
    const q = QUALITY[settings.quality] || QUALITY.medium;
    const { imageData, disparity } = await loadSample(q.maxPixels);
    app.source = { sample: true };
    app.imageData = imageData;
    app.disparity = disparity;
    app.disparityKind = 'gt';
    $('modelInfo').textContent = 'depth: bundled ground truth';
    kickBuild();
  } catch (err) {
    console.error(err);
    status('Could not load the sample', false, 5000);
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
  const dof = settings.aperture * settings.aperture * 240;
  return {
    view: controls.viewMatrix(tmpView),
    proj,
    splatScale: settings.splatScale,
    focusDist: app.focusDist,
    dofStrength: dof,
    maxCoC: DEFAULTS.maxCoC * Math.min(devicePixelRatio || 1, 2),
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
  app.needsRender = true;
});
$('selQuality').value = settings.quality;
$('selQuality').addEventListener('change', (e) => {
  settings.quality = e.target.value;
  reopenCurrent();
});
for (const [id, key] of [['tBg', 'withBg'], ['tSkirt', 'withSkirt'], ['tUnder', 'withUnder']]) {
  $(id).addEventListener('change', (e) => {
    settings[key] = e.target.checked;
    kickBuild();
  });
}
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
  app, controls, renderer, settings,
  captureNow: (scale = 1) => renderer.capture(renderState(canvas.width, canvas.height), scale),
};
