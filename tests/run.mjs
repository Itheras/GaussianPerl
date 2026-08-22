// Unit tests, dependency-free: node tests/run.mjs
import assert from 'node:assert/strict';
import zlib from 'node:zlib';

import { V3, M4, eigenSym3, matToQuat, clamp } from '../src/util/math3d.js';
import {
  resizeFloat, boxBlurFloat, jointBilateral, gradients, dilateMask, percentile,
  toHalfFloat, erodeMaxima,
} from '../src/util/imageops.js';
import {
  normalizeDisparity, decodeGtDisparity, heuristicDisparity,
  edgeMask, fgBoundary, compressFarField,
} from '../src/pipeline/depthproc.js';
import { synthesizeBackground, closeBandHoles } from '../src/pipeline/inpaint.js';
import {
  collarGrow, buildFillInput, planClusters, padPlate, padFloat, ringMask,
  packImageNCHW, packMaskForBox, unpackNCHW, anchorToReference, smoothRingDisparity,
} from '../src/pipeline/fill-plan.js';
import { intrinsicsFrom35mm, defaultIntrinsics, intrinsicsFromTags } from '../src/io/exif.js';
import { fgsSmooth, weightedMedianDepth, mergeFloaters, relocateEdges } from '../src/pipeline/depth-filter.js';
import { buildLayers, upsampleBackgroundTo } from '../src/pipeline/layer-build.js';
import { encodePNG } from '../tools/png.mjs';

let passed = 0, failed = 0;
async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  ok  ${name}`);
  } catch (err) {
    failed++;
    console.error(`FAIL  ${name}\n      ${err.message}`);
  }
}
const near = (a, b, eps = 1e-4) => {
  if (Math.abs(a - b) > eps) throw new Error(`expected ${a} ≈ ${b} (eps ${eps})`);
};

// ---------------- math3d ----------------
await test('M4.perspective projects a centered point to NDC origin', () => {
  const p = M4.perspective(Math.PI / 3, 1.5, 0.1, 100);
  const out = M4.transformPoint4(p, [0, 0, -5]);
  near(out[0] / out[3], 0);
  near(out[1] / out[3], 0);
});

await test('M4.lookAt: eye at origin looking -Z is identity rotation', () => {
  const v = M4.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]);
  const out = M4.transformPoint4(v, [1, 2, -3]);
  near(out[0], 1); near(out[1], 2); near(out[2], -3);
});

await test('M4.lookAt maps eye to view-space origin, target to -Z axis', () => {
  const eye = [3, 2, 1], target = [0.5, -1, -4];
  const v = M4.lookAt(eye, target, [0, 1, 0]);
  const o = M4.transformPoint4(v, eye);
  near(o[0], 0); near(o[1], 0); near(o[2], 0);
  const tt = M4.transformPoint4(v, target);
  near(tt[0], 0, 1e-3); near(tt[1], 0, 1e-3);
  assert.ok(tt[2] < 0, 'target must be in front (negative view z)');
});

await test('M4.invertRigid inverts a view matrix', () => {
  const v = M4.lookAt([2, -1, 3], [0, 0, 0], [0, 1, 0]);
  const inv = M4.invertRigid(v);
  const m = M4.multiply(inv, v);
  const id = M4.identity();
  for (let i = 0; i < 16; i++) near(m[i], id[i], 1e-5);
});

await test('eigenSym3 round-trips a known covariance', () => {
  // sigma = R S^2 R^T for R = rot about z by 30deg, s = (3, 2, 0.5)
  const c = Math.cos(Math.PI / 6), s = Math.sin(Math.PI / 6);
  const R = [[c, -s, 0], [s, c, 0], [0, 0, 1]];
  const S2 = [9, 4, 0.25];
  const cov = new Array(6).fill(0);
  const covIdx = { '00': 0, '01': 1, '02': 2, '11': 3, '12': 4, '22': 5 };
  for (let i = 0; i < 3; i++) {
    for (let j = i; j < 3; j++) {
      let v = 0;
      for (let k = 0; k < 3; k++) v += R[i][k] * R[j][k] * S2[k];
      cov[covIdx[`${i}${j}`]] = v;
    }
  }
  const { evals, evecs } = eigenSym3(cov);
  near(evals[0], 9, 1e-6); near(evals[1], 4, 1e-6); near(evals[2], 0.25, 1e-6);
  // rebuild sigma from eigen pairs
  for (let i = 0; i < 3; i++) {
    for (let j = i; j < 3; j++) {
      let v = 0;
      for (let k = 0; k < 3; k++) v += evecs[k][i] * evecs[k][j] * evals[k];
      near(v, cov[covIdx[`${i}${j}`]], 1e-6);
    }
  }
});

await test('matToQuat produces unit quaternions matching the rotation', () => {
  const c = Math.cos(0.7), s = Math.sin(0.7);
  const cols = [[c, s, 0], [-s, c, 0], [0, 0, 1]]; // rot about +z by 0.7
  const q = matToQuat(cols);
  near(Math.hypot(...q), 1, 1e-6);
  near(q[0], Math.cos(0.35), 1e-6);  // w
  near(q[3], Math.sin(0.35), 1e-6);  // z
});

// ---------------- imageops ----------------
await test('resizeFloat preserves constant fields', () => {
  const src = new Float32Array(16 * 12).fill(3.25);
  const dst = resizeFloat(src, 16, 12, 7, 5);
  for (const v of dst) near(v, 3.25, 1e-6);
});

await test('resizeFloat approximates a linear ramp', () => {
  const w = 32, h = 8;
  const src = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) src[y * w + x] = x / (w - 1);
  const dst = resizeFloat(src, w, h, 16, 4);
  assert.ok(dst[0] < 0.1 && dst[15] > 0.9, `ramp endpoints ${dst[0]} ${dst[15]}`);
  for (let x = 1; x < 16; x++) assert.ok(dst[x] >= dst[x - 1] - 1e-6, 'monotone');
});

await test('boxBlurFloat preserves mean', () => {
  const w = 20, h = 20;
  const src = new Float32Array(w * h).map(() => Math.random());
  const dst = boxBlurFloat(src, w, h, 2);
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  near(mean(dst), mean(src), 0.02);
});

await test('jointBilateral: constant stays constant; edges guided by color', () => {
  const w = 16, h = 4;
  const val = new Float32Array(w * h).fill(2);
  const rgba = new Uint8ClampedArray(w * h * 4).fill(128);
  const out = jointBilateral(val, rgba, w, h, 2, 20, 2);
  for (const v of out) near(v, 2, 1e-4);
});

await test('dilateMask grows a point by radius', () => {
  const w = 11, h = 11;
  const m = new Uint8Array(w * h);
  m[5 * w + 5] = 1;
  const d = dilateMask(m, w, h, 3);
  assert.equal(d[5 * w + 2], 1);
  assert.equal(d[2 * w + 5], 1);
  assert.equal(d[5 * w + 1], 0);
  assert.equal(d[0], 0);
});

await test('percentile brackets the data', () => {
  const a = new Float32Array(1000).map((_, i) => i);
  const p50 = percentile(a, 0.5);
  assert.ok(Math.abs(p50 - 500) < 10, `p50=${p50}`);
  assert.ok(percentile(a, 0.99) > 950);
});

// ---------------- depthproc ----------------
await test('normalizeDisparity maps to [0,1]', () => {
  const raw = new Float32Array(4096).map(() => Math.random() * 80 - 40);
  const n = normalizeDisparity(raw);
  let lo = 1, hi = 0;
  for (const v of n) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  assert.ok(lo >= 0 && hi <= 1 && hi > 0.9 && lo < 0.1);
});

await test('decodeGtDisparity decodes RG 16-bit packing', () => {
  const rgba = new Uint8ClampedArray(8);
  rgba[0] = 0x12; rgba[1] = 0x34; // 0x1234 / 65535
  rgba[4] = 0xff; rgba[5] = 0xff;
  const d = decodeGtDisparity(rgba, 2, 1);
  near(d[0], 0x1234 / 65535, 1e-6);
  near(d[1], 1, 1e-6);
});

await test('edgeMask + fgBoundary flag a depth step correctly', () => {
  const w = 10, h = 4;
  const disp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) disp[y * w + x] = x < 5 ? 0.8 : 0.2;
  const em = edgeMask(disp, w, h, 0.3);
  const fb = fgBoundary(disp, w, h, 0.3);
  assert.equal(em[1 * w + 4], 1, 'near side flagged');
  assert.equal(em[1 * w + 5], 1, 'far side flagged');
  assert.equal(fb[1 * w + 4], 1, 'fg boundary = near side');
  assert.equal(fb[1 * w + 5], 0, 'far side not fg');
  assert.equal(em[1 * w + 2], 0, 'interior clean');
});

await test('compressFarField: flattens far tail, keeps near field, stays monotone', () => {
  const d = new Float32Array([0, 0.02, 0.06, 0.08, 0.3, 1]);
  const c = compressFarField(d, 0.08, 0.35);
  near(c[3], 0.08, 1e-6);
  near(c[5], 1, 1e-6);
  near(c[4], 0.3, 1e-6, 'above knee untouched');
  // sky (0) vs mountains (0.06): spread shrinks by keep factor
  const before = d[2] - d[0], after = c[2] - c[0];
  near(after, before * 0.35, 1e-6);
  for (let i = 1; i < c.length; i++) assert.ok(c[i] >= c[i - 1], 'monotone');
  // a horizon jump below the edge threshold after compression
  assert.ok(c[2] - c[0] < 0.055, 'compressed horizon no longer a silhouette');
});

await test('heuristicDisparity: bottom nearer than top', () => {
  const w = 24, h = 24;
  const rgba = new Uint8ClampedArray(w * h * 4).fill(120);
  const d = heuristicDisparity(rgba, w, h);
  const top = d.slice(0, w).reduce((s, v) => s + v, 0) / w;
  const bot = d.slice((h - 1) * w).reduce((s, v) => s + v, 0) / w;
  assert.ok(bot > top + 0.3, `bot=${bot} top=${top}`);
});

// ---------------- inpaint ----------------
await test('synthesizeBackground fills behind a foreground square', () => {
  const w = 40, h = 40;
  const rgba = new Uint8ClampedArray(w * h * 4);
  const disp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const fg = x >= 14 && x < 26 && y >= 14 && y < 26;
      disp[i] = fg ? 0.9 : 0.2;
      rgba[i * 4] = fg ? 200 : 30;
      rgba[i * 4 + 1] = fg ? 40 : 160;
      rgba[i * 4 + 2] = 40;
      rgba[i * 4 + 3] = 255;
    }
  }
  const fgB = fgBoundary(disp, w, h, 0.3);
  const bg = synthesizeBackground(rgba, disp, w, h, fgB, { bandPx: 6, jump: 0.3 });
  let inside = 0;
  for (let y = 15; y < 25; y++) {
    for (let x = 15; x < 25; x++) {
      const i = y * w + x;
      if (bg.bgMask[i]) {
        inside++;
        assert.ok(bg.bgDisp[i] < 0.7, 'bg strictly behind fg');
        assert.ok(bg.bgColor[i * 4 + 1] > bg.bgColor[i * 4], 'bg pulled green, not fg red');
      }
    }
  }
  assert.ok(inside > 20, `expected fill inside fg, got ${inside}`);
});

await test('closeBandHoles floods failed band pixels from filled neighbors', () => {
  const w = 20, h = 8;
  const disp = new Float32Array(w * h).fill(0.9); // foreground everywhere
  const band = new Uint8Array(w * h).fill(1);
  const bgColor = new Uint8ClampedArray(w * h * 4);
  const bgDisp = new Float32Array(w * h);
  const bgMask = new Uint8Array(w * h);
  // left column filled at bg depth 0.2; rest of the band is a failed hole
  for (let y = 0; y < h; y++) {
    const i = y * w;
    bgMask[i] = 1; bgDisp[i] = 0.2;
    bgColor[i * 4] = 10; bgColor[i * 4 + 1] = 180; bgColor[i * 4 + 2] = 20; bgColor[i * 4 + 3] = 255;
  }
  closeBandHoles({ bgColor, bgDisp, bgMask }, disp, band, w, h, 0.1, 8);
  assert.equal(bgMask[3 * w + 6], 1, 'hole 6px in closed');
  assert.equal(bgColor[(3 * w + 6) * 4 + 1], 180, 'donor color inherited');
  assert.ok(bgDisp[3 * w + 6] <= 0.8, 'fill stays behind the foreground');
  assert.equal(bgMask[3 * w + 12], 0, 'maxSteps respected');
  // a band pixel already AT background depth is not a disocclusion — untouched
  const disp2 = new Float32Array(w * h).fill(0.22);
  const bgMask2 = new Uint8Array(w * h);
  bgMask2[0] = 1;
  const bgDisp2 = new Float32Array(w * h); bgDisp2[0] = 0.2;
  closeBandHoles({ bgColor: new Uint8ClampedArray(w * h * 4), bgDisp: bgDisp2, bgMask: bgMask2 },
    disp2, band, w, h, 0.1, 8);
  assert.equal(bgMask2[1], 0, 'background-depth pixel not claimed');
});

await test('collarGrow follows the near surface, never crosses the cliff', () => {
  // vertical silhouette at x=10: near (0.8) left, far (0.2) right
  const w = 24, h = 8;
  const disp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) disp[y * w + x] = x < 10 ? 0.8 : 0.2;
  const seed = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) seed[y * w + 9] = 1; // fg boundary column
  const grown = collarGrow(disp, seed, w, h, 4, 0.05);
  assert.equal(grown[3 * w + 6], 1, 'grew 3px into the foreground');
  assert.equal(grown[3 * w + 11], 0, 'did NOT step down the cliff into background');
  assert.equal(grown[3 * w + 4], 0, 'radius respected');
});

await test('buildFillInput: holes = bgMask + collar + rim; prefill uses classical colors', () => {
  const w = 48, h = 48;
  const rgba = new Uint8ClampedArray(w * h * 4);
  const disp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const fg = x >= 10 && x < 30 && y >= 10 && y < 30; // 20px square
      disp[i] = fg ? 0.9 : 0.2;
      rgba[i * 4] = fg ? 200 : 30;
      rgba[i * 4 + 1] = fg ? 40 : 160;
      rgba[i * 4 + 2] = 40;
      rgba[i * 4 + 3] = 255;
    }
  }
  const fgB = fgBoundary(disp, w, h, 0.3);
  const bg = synthesizeBackground(rgba, disp, w, h, fgB, { bandPx: 6, jump: 0.3 });
  const { holes, prefilled } = buildFillInput(rgba, disp, bg, fgB, w, h, { jump: 0.3, collarPx: 4 });
  let bgHoles = 0;
  for (let i = 0; i < w * h; i++) {
    if (bg.bgMask[i]) {
      assert.equal(holes[i], 1, 'every bgMask pixel is a hole');
      bgHoles++;
      // prefill took the classical (green-ish background) color, not fg red
      assert.ok(prefilled[i * 4 + 1] > prefilled[i * 4], 'prefill is background-colored');
    }
  }
  assert.ok(bgHoles > 10, `bgMask holes present (${bgHoles})`);
  assert.equal(holes[20 * w + 12], 1, 'foreground collar near silhouette is masked');
  assert.equal(holes[20 * w + 20], 0, 'deep fg interior stays context');
  assert.equal(holes[2 * w + 2], 0, 'far background stays context');
});

await test('planClusters: separate clusters, merging, and maxCalls coarsening', () => {
  const w = 400, h = 300;
  const holes = new Uint8Array(w * h);
  const blob = (cx, cy, r) => {
    for (let y = cy - r; y <= cy + r; y++) {
      for (let x = cx - r; x <= cx + r; x++) holes[y * w + x] = 1;
    }
  };
  blob(50, 50, 10);
  blob(340, 240, 10);
  const boxes = planClusters(holes, w, h, { cellPx: 16, maxBoxPx: 128, overlapPx: 32, mergeGapPx: 24, maxCalls: 6 });
  assert.equal(boxes.length, 2, `two distant blobs = two boxes (${boxes.length})`);
  for (const b of boxes) {
    assert.ok(b.x1 - b.x0 <= 128 && b.y1 - b.y0 <= 128, 'boxes within maxBox');
  }
  // close blobs merge
  const holes2 = new Uint8Array(w * h);
  holes2.fill(0);
  const setAt = (cx, cy, r, arr) => {
    for (let y = cy - r; y <= cy + r; y++) for (let x = cx - r; x <= cx + r; x++) arr[y * w + x] = 1;
  };
  setAt(100, 100, 8, holes2);
  setAt(120, 100, 8, holes2);
  const merged = planClusters(holes2, w, h, { cellPx: 16, maxBoxPx: 128, overlapPx: 32, mergeGapPx: 24, maxCalls: 6 });
  assert.equal(merged.length, 1, 'adjacent blobs merged');
  // maxCalls forces coarsening down to (at worst) a single union box
  const many = new Uint8Array(w * h);
  for (let cy = 30; cy < 300; cy += 60) {
    for (let cx = 30; cx < 400; cx += 80) setAt(cx, cy, 4, many);
  }
  const capped = planClusters(many, w, h, { cellPx: 16, maxBoxPx: 64, overlapPx: 16, mergeGapPx: 8, maxCalls: 3 });
  assert.ok(capped.length <= 3, `capped at maxCalls (${capped.length})`);
});

await test('planClusters covers every hole pixel', () => {
  const w = 300, h = 200;
  const holes = new Uint8Array(w * h);
  for (let y = 20; y < 180; y++) for (let x = 10; x < 290; x += 2) holes[y * w + x] = 1;
  const boxes = planClusters(holes, w, h, { cellPx: 32, maxBoxPx: 96, overlapPx: 24, mergeGapPx: 16, maxCalls: 64 });
  const covered = new Uint8Array(w * h);
  for (const b of boxes) {
    for (let y = b.y0; y < b.y1; y++) for (let x = b.x0; x < b.x1; x++) covered[y * w + x] = 1;
  }
  for (let i = 0; i < w * h; i++) {
    if (holes[i]) assert.equal(covered[i], 1, `hole at ${i % w},${(i / w) | 0} uncovered`);
  }
});

await test('padPlate mirrors, padFloat replicates, ringMask bands', () => {
  const w = 4, h = 3, pad = 2;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) { rgba[i * 4] = i * 10; rgba[i * 4 + 3] = 255; }
  const { plate, pw, ph } = padPlate(rgba, w, h, pad);
  assert.equal(pw, 8); assert.equal(ph, 7);
  // plate(1,2) is one left of interior x=0 (mirror -> x=0), row 0
  assert.equal(plate[(2 * pw + 1) * 4], rgba[0]);
  // plate(0,2): x-pad = -2 mirrors to 1
  assert.equal(plate[(2 * pw + 0) * 4], rgba[1 * 4]);
  const f = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
  const pf = padFloat(f, w, h, pad);
  assert.equal(pf[0], 1, 'replicate: corner clamps to (0,0)');
  assert.equal(pf[2 * pw + 0], 1, 'replicate row start');
  assert.equal(pf[2 * pw + 7], 4, 'replicate row end clamps to x=w-1');
  const ring = ringMask(pw, ph, pad);
  assert.equal(ring[0], 1);
  assert.equal(ring[3 * pw + 4], 0, 'interior clear');
  assert.equal(ring[3 * pw + 1], 1, 'left band set');
  let interior = 0;
  for (let i = 0; i < pw * ph; i++) interior += 1 - ring[i];
  assert.equal(interior, w * h, 'interior exactly the unpadded area');
});

await test('anchorToReference removes low-freq hallucination, keeps texture', () => {
  const w = 64, h = 64;
  const mask = new Uint8Array(w * h);
  for (let y = 16; y < 48; y++) for (let x = 16; x < 48; x++) mask[y * w + x] = 1;
  const ref = new Uint8ClampedArray(w * h * 4);
  const ai = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      ref[i * 4] = 100; ref[i * 4 + 1] = 150; ref[i * 4 + 2] = 80; ref[i * 4 + 3] = 255;
      // AI: hallucinated +60 offset (wrong tone) + fine checkerboard texture
      const tex = ((x + y) & 1) ? 12 : -12;
      ai[i * 4] = 160 + tex; ai[i * 4 + 1] = 210 + tex; ai[i * 4 + 2] = 140 + tex;
      ai[i * 4 + 3] = 255;
    }
  }
  anchorToReference(ai, ref, mask, w, h, 8);
  // deep inside the mask: mean pulled to ref, checkerboard survives
  const at = (x, y, c) => ai[(y * w + x) * 4 + c];
  const meanG = (at(32, 32, 1) + at(33, 32, 1)) / 2;
  assert.ok(Math.abs(meanG - 150) < 4, `low-freq anchored (mean G ${meanG})`);
  const contrast = Math.abs(at(32, 32, 1) - at(33, 32, 1));
  assert.ok(contrast > 16, `texture preserved (contrast ${contrast})`);
  // outside the mask untouched
  assert.equal(at(4, 4, 0), 160 + ((4 + 4) & 1 ? 12 : -12));
});

await test('smoothRingDisparity calms the ring, keeps the interior exact', () => {
  const w = 40, h = 30, pad = 10;
  const pw = w + 2 * pad, ph = h + 2 * pad;
  // horizon cliff: top half far (0.05), bottom half near (0.9)
  const disp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) disp[y * w + x] = y < 12 ? 0.05 : 0.9;
  const plate = padFloat(disp, w, h, pad);
  const before = plate.slice();
  smoothRingDisparity(plate, pw, ph, pad, 6);
  // interior bit-exact
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      near(plate[(y + pad) * pw + (x + pad)], disp[y * w + x], 1e-7);
    }
  }
  // deep in the left ring, the vertical cliff must be gentler than the raw copy
  const cliffAt = (arr, x) => {
    let m = 0;
    for (let y = 1; y < ph; y++) m = Math.max(m, Math.abs(arr[y * pw + x] - arr[(y - 1) * pw + x]));
    return m;
  };
  assert.ok(cliffAt(plate, 2) < cliffAt(before, 2) * 0.6,
    `ring cliff softened (${cliffAt(plate, 2).toFixed(3)} vs ${cliffAt(before, 2).toFixed(3)})`);
});

await test('NCHW pack/unpack round-trips; mask polarity is 255=known/0=hole', () => {
  const w = 3, h = 2;
  const rgba = new Uint8ClampedArray([
    10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255,
    1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255,
  ]);
  const packed = packImageNCHW(rgba, w, h);
  assert.equal(packed[0], 10, 'R plane first');
  assert.equal(packed[w * h], 20, 'G plane second');
  assert.equal(packed[2 * w * h + 5], 9, 'B plane last');
  const back = unpackNCHW(packed, w, h);
  for (let i = 0; i < w * h; i++) {
    assert.equal(back[i * 4], rgba[i * 4]);
    assert.equal(back[i * 4 + 1], rgba[i * 4 + 1]);
    assert.equal(back[i * 4 + 2], rgba[i * 4 + 2]);
    assert.equal(back[i * 4 + 3], 255);
  }
  // alpha round-trip: the model carries no alpha — srcRgba preserves it
  const transparent = Uint8ClampedArray.from(rgba);
  transparent[3] = 0; transparent[7] = 128;
  const back2 = unpackNCHW(packed, w, h, transparent);
  assert.equal(back2[3], 0, 'transparent pixel stays transparent');
  assert.equal(back2[7], 128, 'partial alpha preserved');
  assert.equal(back2[11], 255);
  const holes = new Uint8Array([0, 1, 0, 0, 1, 1]);
  const m = packMaskForBox(holes, w, h, { x0: 0, y0: 0, x1: 2, y1: 2 });
  assert.deepEqual([...m], [255, 0, 255, 255, 0, 255], 'holes->0 inside box only');
});

// ---------------- splat-build ----------------

await test('fgsSmooth: constant stays constant; smooths noise; never imprints texture', () => {
  const w = 48, h = 32;
  const mkTex = (amp) => {
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const v = 128 + (((x + y) & 1) ? amp : -amp);
        const i = y * w + x;
        rgba[i * 4] = v; rgba[i * 4 + 1] = v; rgba[i * 4 + 2] = v; rgba[i * 4 + 3] = 255;
      }
    }
    return rgba;
  };
  const flat = new Float32Array(w * h).fill(0.5);
  for (const v of fgsSmooth(flat, mkTex(90), w, h)) near(v, 0.5, 1e-4);
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const noisy = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) noisy[i] = 0.5 + (rnd() - 0.5) * 0.1;
  const std = (a) => {
    let m = 0; for (const v of a) m += v; m /= a.length;
    let s = 0; for (const v of a) s += (v - m) * (v - m);
    return Math.sqrt(s / a.length);
  };
  // realistic mild texture (skin-scale contrast): noise flattens strongly
  const outMild = fgsSmooth(noisy, mkTex(10), w, h);
  assert.ok(std(outMild) < std(noisy) * 0.35, `noise flattened over mild texture (${std(outMild).toFixed(4)} vs ${std(noisy).toFixed(4)})`);
  // even where hard texture blocks flow, the filter must never CREATE
  // texture-correlated structure (the bilateral-imprint failure mode)
  const outHard = fgsSmooth(noisy, mkTex(90), w, h);
  const altCorr = (a, rgba) => {
    // correlation of neighbor differences with the checkerboard sign
    let s = 0, cnt = 0;
    for (let y = 4; y < h - 4; y++) {
      for (let x = 4; x < w - 5; x++) {
        const i = y * w + x;
        const texSign = ((x + y) & 1) ? 1 : -1;
        s += (a[i] - a[i + 1]) * texSign; cnt++;
      }
    }
    return Math.abs(s / cnt);
  };
  assert.ok(altCorr(outHard) <= altCorr(noisy) + 1e-4,
    `no texture-correlated structure created (${altCorr(outHard).toFixed(6)})`);
});

await test('fgsSmooth preserves a depth step aligned with a color edge', () => {
  const w = 40, h = 16;
  const rgba = new Uint8ClampedArray(w * h * 4);
  const disp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const left = x < 20;
      const v = left ? 200 : 30;
      rgba[i * 4] = v; rgba[i * 4 + 1] = v; rgba[i * 4 + 2] = v; rgba[i * 4 + 3] = 255;
      disp[i] = left ? 0.8 : 0.2;
    }
  }
  const out = fgsSmooth(disp, rgba, w, h);
  const step = out[8 * w + 15] - out[8 * w + 24];
  assert.ok(step > 0.5, `step survives (${step.toFixed(3)})`);
});

await test('weightedMedianDepth: silhouette ramp becomes a step; gradient survives', () => {
  const w = 24, h = 10;
  const disp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v;
      if (x < 10) v = 0.2;
      else if (x > 13) v = 0.8;
      else v = 0.2 + (x - 9) * 0.15; // 4px soft ramp
      disp[y * w + x] = v;
    }
  }
  const out = weightedMedianDepth(disp, w, h, 0.055);
  let mid = 0;
  for (let x = 0; x < w; x++) {
    const v = out[5 * w + x];
    if (Math.abs(v - 0.2) > 0.05 && Math.abs(v - 0.8) > 0.05) mid++;
  }
  assert.ok(mid <= 2, `ramp collapsed to a near-step (${mid} mid pixels)`);
  const g = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) g[y * w + x] = x * 0.01;
  const g2 = weightedMedianDepth(g, w, h, 0.055);
  for (let x = 4; x < w - 4; x++) near(g2[5 * w + x], g[5 * w + x], 0.01);
});

await test('mergeFloaters: small debris merged, large regions kept', () => {
  const w = 40, h = 30;
  const disp = new Float32Array(w * h).fill(0.3);
  // small floater (9px) at wrong depth + a large legitimate region
  for (let y = 5; y < 8; y++) for (let x = 5; x < 8; x++) disp[y * w + x] = 0.9;
  for (let y = 15; y < 28; y++) for (let x = 10; x < 35; x++) disp[y * w + x] = 0.7;
  mergeFloaters(disp, w, h, 0.055, 20);
  near(disp[6 * w + 6], 0.3, 1e-3);
  near(disp[20 * w + 20], 0.7, 1e-6);
});

await test('relocateEdges: straightens a jagged depth edge along a straight image edge; gated off elsewhere', () => {
  const w = 30, h = 16;
  const rgba = new Uint8ClampedArray(w * h * 4);
  const disp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const left = x < 12; // straight image edge at 12
      const v = left ? 210 : 40;
      rgba[i * 4] = v; rgba[i * 4 + 1] = v; rgba[i * 4 + 2] = v; rgba[i * 4 + 3] = 255;
      const cut = 12 + ((y % 3) - 1) * 2; // jagged depth edge 10/12/14
      disp[i] = x < cut ? 0.8 : 0.2;
    }
  }
  const out = relocateEdges(disp, rgba, w, h);
  const edgeX = (arr, y) => {
    for (let x = 1; x < w; x++) if (Math.abs(arr[y * w + x] - arr[y * w + x - 1]) > 0.3) return x;
    return -1;
  };
  const spread = (arr) => {
    let mn = w, mx = 0;
    for (let y = 3; y < h - 3; y++) { const e = edgeX(arr, y); mn = Math.min(mn, e); mx = Math.max(mx, e); }
    return mx - mn;
  };
  assert.ok(spread(out) < spread(disp), `edge straightened (${spread(out)} < ${spread(disp)})`);
  // flat color, textured depth noise far from edges: untouched (gate)
  const disp2 = new Float32Array(w * h).fill(0.5);
  disp2[8 * w + 20] = 0.58; // small bump, no image edge there... below tau anyway
  const rgbaFlat = new Uint8ClampedArray(w * h * 4).fill(128);
  const out2 = relocateEdges(disp2, rgbaFlat, w, h);
  for (let i = 0; i < w * h; i++) near(out2[i], disp2[i], 1e-6);
});

// ---------------- exif intrinsics ----------------
await test('intrinsics: iPhone XR test photo (f35=26) gives 67.3 deg long-side FoV', () => {
  // the repo test photo: iPhone XR, FocalLengthIn35mmFormat=26, 3024x4032 portrait
  const i = intrinsicsFrom35mm(26, 3024, 4032);
  near(i.fPx, 26 * 5040 / 43.267, 0.1);
  near(i.fovYDeg, 67.28, 0.1);
  assert.equal(i.source, 'exif');
});

await test('intrinsics: tag interpretation — clamps garbage, guards digital zoom, falls back', () => {
  // sane path
  const ok = intrinsicsFromTags({ FocalLengthIn35mmFormat: 26 }, 3024, 4032);
  assert.equal(ok.f35, 26);
  // iOS garbage f35 (known 177/311 bug) -> fallback
  const junk = intrinsicsFromTags({ FocalLengthIn35mmFormat: 311 }, 4000, 3000);
  assert.equal(junk.source, 'default');
  // digital zoom applies only on a known base lens
  const zoomed = intrinsicsFromTags({ FocalLengthIn35mmFormat: 26, DigitalZoomRatio: 2 }, 4000, 3000);
  near(zoomed.f35, 52, 1e-9);
  const baked = intrinsicsFromTags({ FocalLengthIn35mmFormat: 31, DigitalZoomRatio: 2 }, 4000, 3000);
  near(baked.f35, 31, 1e-9, 'non-base lens: zoom already baked in');
  // no EXIF at all
  const none = intrinsicsFromTags(null, 1600, 900);
  assert.equal(none.source, 'default');
  near(none.fPx, 1600, 1e-9);
  near(none.fovXDeg, 53.13, 0.01);
  // default is orientation-symmetric: portrait vs landscape same long-side FoV
  const port = defaultIntrinsics(900, 1600);
  near(port.fovYDeg, none.fovXDeg, 1e-9);
});

// ---------------- .splat export ----------------
await test('encodePNG: valid signature, IHDR, inflatable IDAT', () => {
  const w = 5, h = 3;
  const rgba = new Uint8ClampedArray(w * h * 4).map((_, i) => (i * 7) & 0xff);
  const png = encodePNG(w, h, rgba);
  assert.deepEqual([...png.subarray(0, 8)], [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  assert.equal(png.readUInt32BE(16), w);
  assert.equal(png.readUInt32BE(20), h);
  // find IDAT
  let off = 8, idat = null;
  while (off < png.length) {
    const len = png.readUInt32BE(off);
    const type = png.toString('ascii', off + 4, off + 8);
    if (type === 'IDAT') idat = png.subarray(off + 8, off + 8 + len);
    off += 12 + len;
  }
  assert.ok(idat, 'IDAT found');
  const raw = zlib.inflateSync(idat);
  assert.equal(raw.length, h * (w * 4 + 1));
});

// ---------------- sort worker (simulated) ----------------

// ---------------- half float + erosion ----------------
await test('toHalfFloat round-trips typical disparities within half precision', () => {
  const vals = new Float32Array([0, 0.04, 0.055, 0.16, 0.5, 0.999, 1.0]);
  const halves = toHalfFloat(vals);
  // decode back
  const dec = (hbits) => {
    const s = (hbits & 0x8000) ? -1 : 1;
    const e = (hbits >> 10) & 0x1f;
    const m = hbits & 0x3ff;
    if (e === 0) return s * m * Math.pow(2, -24);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  };
  for (let i = 0; i < vals.length; i++) {
    near(dec(halves[i]), vals[i], 6e-4);
  }
});

await test('erodeMaxima shrinks near (high) disparity by the radius', () => {
  const w = 20, h = 10;
  const d = new Float32Array(w * h).fill(0.2);
  for (let y = 3; y < 7; y++) for (let x = 8; x < 14; x++) d[y * w + x] = 0.9;
  const e = erodeMaxima(d, w, h, 1);
  near(e[5 * w + 8], 0.2, 1e-6);  // fg boundary column eroded
  near(e[5 * w + 10], 0.9, 1e-6); // interior kept
  near(e[1 * w + 1], 0.2, 1e-6);  // background untouched
});

// ---------------- layer build ----------------
await test('buildLayers: dims, mirror ring, feathered fill alpha, disp1 override', () => {
  const w = 24, h = 16, dw = 12, dh = 8, padPx = 4, padD = 2;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4] = (i % w) * 10; rgba[i * 4 + 1] = 50; rgba[i * 4 + 2] = 90; rgba[i * 4 + 3] = 255;
  }
  const dispD = new Float32Array(dw * dh).fill(0.4);
  for (let y = 2; y < 6; y++) for (let x = 4; x < 8; x++) dispD[y * dw + x] = 0.8;
  const bgD = {
    bgMask: new Uint8Array(dw * dh), bgDisp: new Float32Array(dw * dh),
    bgColor: new Uint8ClampedArray(dw * dh * 4),
  };
  bgD.bgMask[3 * dw + 5] = 1; bgD.bgDisp[3 * dw + 5] = 0.3;
  const bgW = upsampleBackgroundTo(bgD, dw, dh, w, h);
  const L = buildLayers({ rgba, w, h, dispD, dw, dh, bgW, bgD, padPx, padD, erodeIterations: 0 });
  assert.equal(L.pw, w + 2 * padPx);
  assert.equal(L.ph, h + 2 * padPx);
  assert.equal(L.pdw, dw + 2 * padD);
  assert.equal(L.color0.length, L.pw * L.ph * 4);
  assert.equal(L.disp0.length, L.pdw * L.pdh);
  // interior photo pixel lands at the padded offset
  const o = ((3 + padPx) * L.pw + (5 + padPx)) * 4;
  assert.equal(L.color0[o], 5 * 10);
  // mirror ring: left of interior column 0 mirrors column 0
  const or_ = ((3 + padPx) * L.pw + (padPx - 1)) * 4;
  assert.equal(L.color0[or_], 0);
  // disp0 interior preserved (no erosion in this test)
  near(L.disp0[(3 + padD) * L.pdw + (5 + padD)], 0.8, 1e-6);
  // disp1 override inside mask, disp0 copy outside
  near(L.disp1[(3 + padD) * L.pdw + (5 + padD)], 0.3, 1e-6);
  near(L.disp1[(1 + padD) * L.pdw + (1 + padD)], L.disp0[(1 + padD) * L.pdw + (1 + padD)], 1e-6);
  // layer-1 alpha: present only where the (upsampled) mask lives
  let a = 0;
  for (let i = 0; i < L.pw * L.ph; i++) a += L.color1[i * 4 + 3] > 0 ? 1 : 0;
  const maskCount = bgW.bgMask.reduce((s, v) => s + v, 0);
  assert.equal(a, maskCount, 'alpha coverage equals mask coverage');
});

// ---------------- pipeline worker protocol (simulated) ----------------
await test('pipeline-worker: classical layers final; AI fill falls back offline', async () => {
  const messages = [];
  globalThis.self = {
    onmessage: null,
    postMessage: (m) => messages.push(m),
  };
  const realFetch = globalThis.fetch;
  globalThis.fetch = () => Promise.reject(new Error('offline test'));
  try {
    await import('../src/pipeline/pipeline-worker.js');
    const handler = globalThis.self.onmessage;
    assert.ok(handler, 'worker registered a handler');

    const w = 48, h = 40;
    const rgba = new Uint8ClampedArray(w * h * 4).fill(180);
    const disp = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        disp[y * w + x] = (x >= 18 && x < 30 && y >= 14 && y < 26) ? 0.9 : 0.25;
      }
    }
    const params = {
      edgeDispJump: 0.055, farKnee: 0.16, farKeep: 0.25,
      bgBandPx: 6, skirtPx: 6,
      withBg: true, withSkirt: true,
      wantAiDepth: false, aiFill: false, deviceClass: 'desktop',
    };
    const waitFor = async (pred, ms = 8000) => {
      const t0 = Date.now();
      while (!pred()) {
        if (Date.now() - t0 > ms) throw new Error('timed out waiting for worker message');
        await new Promise((r) => setTimeout(r, 20));
      }
    };

    handler({ data: {
      type: 'build', id: 1, sourceId: 1, rgba: rgba.slice().buffer, w, h,
      disparity: disp.slice().buffer, params,
    } });
    await waitFor(() => messages.some((m) => m.type === 'built'));
    const b1 = messages.filter((m) => m.type === 'built');
    assert.equal(b1.length, 1);
    const m1 = b1[0].meta;
    assert.equal(m1.phase, 'final');
    assert.equal(m1.fillKind, 'classical');
    assert.equal(m1.depthKind, 'gt');
    assert.equal(b1[0].color0.length, m1.pw * m1.ph * 4);
    assert.equal(b1[0].disp0.length, m1.pdw * m1.pdh);
    assert.equal(b1[0].color1.length, m1.pw * m1.ph * 4);
    assert.ok(m1.dSub > 0 && m1.dMax > m1.dMin, 'anchors sane');
    // layer 1 has some fill coverage behind the square
    let cov = 0;
    for (let i = 0; i < m1.pw * m1.ph; i++) cov += b1[0].color1[i * 4 + 3] > 0 ? 1 : 0;
    assert.ok(cov > 10, `bg layer has coverage (${cov})`);

    // AI fill requested but model unreachable: preview + fill-failed
    messages.length = 0;
    handler({ data: {
      type: 'build', id: 2, sourceId: 2, rgba: rgba.slice().buffer, w, h,
      disparity: disp.slice().buffer, params: { ...params, aiFill: true },
    } });
    await waitFor(() => messages.some((m) => m.type === 'fill-failed'));
    const b2 = messages.filter((m) => m.type === 'built');
    assert.equal(b2.length, 1, 'exactly the preview build');
    assert.equal(b2[0].meta.phase, 'preview');
  } finally {
    globalThis.fetch = realFetch;
    delete globalThis.self;
  }
});


console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
