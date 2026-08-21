// Unit tests, dependency-free: node tests/run.mjs
import assert from 'node:assert/strict';
import zlib from 'node:zlib';

import { V3, M4, eigenSym3, matToQuat, clamp } from '../src/util/math3d.js';
import {
  resizeFloat, boxBlurFloat, jointBilateral, gradients, dilateMask, percentile,
} from '../src/util/imageops.js';
import {
  normalizeDisparity, decodeGtDisparity, heuristicDisparity, disparityToDepth,
  edgeMask, fgBoundary, snapDepthEdges, compressFarField,
} from '../src/pipeline/depthproc.js';
import { synthesizeBackground } from '../src/pipeline/inpaint.js';
import {
  collarGrow, buildFillInput, planClusters, padPlate, padFloat, ringMask,
  packImageNCHW, packMaskForBox, unpackNCHW, anchorToReference, smoothRingDisparity,
} from '../src/pipeline/fill-plan.js';
import { buildSplats } from '../src/pipeline/splat-build.js';
import { encodeSplatFile } from '../src/io/save.js';
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

await test('disparityToDepth: monotone, correct endpoints', () => {
  const d = disparityToDepth(new Float32Array([1, 0.5, 0]), 1, 7);
  near(d[0], 1, 1e-5);       // nearest
  near(d[2], 8, 1e-4);       // farthest = zn + range
  assert.ok(d[0] < d[1] && d[1] < d[2]);
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

await test('snapDepthEdges: soft silhouette ramp becomes a step, gradients survive', () => {
  const w = 20, h = 6;
  const disp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // step 0.2 -> 0.8 with a 3px soft ramp at x=9..11, like bilinear mixing
      let v;
      if (x < 9) v = 0.2;
      else if (x > 11) v = 0.8;
      else v = 0.2 + (x - 8) * 0.2;
      disp[y * w + x] = v;
    }
  }
  const out = snapDepthEdges(disp, w, h, 0.055, 2);
  for (let x = 0; x < w; x++) {
    const v = out[2 * w + x];
    assert.ok(Math.abs(v - 0.2) < 0.02 || Math.abs(v - 0.8) < 0.02,
      `x=${x} still mid-ramp: ${v}`);
  }
  // gentle gradient untouched
  const g = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) g[y * w + x] = x * 0.005;
  const g2 = snapDepthEdges(g, w, h, 0.055, 2);
  for (let i = 0; i < g.length; i++) near(g2[i], g[i], 1e-6);
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

// ---------------- fill-plan ----------------
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
function tinyScene(w = 12, h = 10) {
  const rgba = new Uint8ClampedArray(w * h * 4);
  const disp = new Float32Array(w * h).fill(0.5);
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4] = 100; rgba[i * 4 + 1] = 150; rgba[i * 4 + 2] = 200; rgba[i * 4 + 3] = 255;
  }
  return { rgba, disp, w, h };
}
const buildDefaults = {
  fovYDeg: 55, zNear: 1, zRange: 7, sizeFactor: 0.65, edgeDispJump: 0.05,
  bgBandPx: 8, skirtPx: 8, underStep: 4,
  withBg: false, withSkirt: false, withUnder: false,
};

await test('buildSplats: fine layer count and geometry for a flat plane', () => {
  const { rgba, disp, w, h } = tinyScene();
  const edges = new Uint8Array(w * h);
  const out = buildSplats({ rgba, w, h, disp, edges, bg: null, params: { ...buildDefaults } });
  assert.equal(out.count, w * h);
  // constant disparity 0.5 => z = 1/(0.5*(1-1/8)+1/8) = 1.7777
  const zExpect = 1 / (0.5 * (1 - 1 / 8) + 1 / 8);
  for (let i = 0; i < out.count; i++) {
    near(-out.positions[i * 3 + 2], zExpect, 1e-3);
  }
  // covariance: symmetric PSD-ish, positive diagonals
  for (let i = 0; i < out.count; i++) {
    assert.ok(out.cov[i * 6] > 0 && out.cov[i * 6 + 3] > 0 && out.cov[i * 6 + 5] >= 0);
  }
  near(out.meta.centerZ, zExpect, 1e-3);
});

await test('buildSplats: layers add splats; skirt fades out', () => {
  const { rgba, disp, w, h } = tinyScene();
  const edges = new Uint8Array(w * h);
  const out = buildSplats({
    rgba, w, h, disp, edges, bg: null,
    params: { ...buildDefaults, withSkirt: true, withUnder: true },
  });
  assert.ok(out.meta.underCount > 0, 'underlayer present');
  assert.ok(out.meta.skirtCount > 0, 'skirt present');
  assert.equal(out.count, out.meta.fineCount + out.meta.bgCount + out.meta.underCount + out.meta.skirtCount);
  // skirt alphas < 255
  const start = out.meta.fineCount + out.meta.bgCount + out.meta.underCount;
  let sawFade = false;
  for (let i = start; i < out.count; i++) {
    const a = out.colors[i * 4 + 3];
    assert.ok(a <= 235, 'skirt alpha faded');
    if (a < 200) sawFade = true;
  }
  assert.ok(sawFade, 'skirt should fade with distance');
});

await test('buildSplats: slanted surface stretches covariance anisotropically', () => {
  // 64px grid: per-pixel disparity delta stays below the discontinuity guard
  // (edgeDispJump) while the surface slant is strong, like real photos
  const w = 64, h = 64;
  const rgba = new Uint8ClampedArray(w * h * 4).fill(255);
  const disp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) disp[y * w + x] = 0.2 + 0.6 * (y / (h - 1)); // ground-like
  }
  const edges = new Uint8Array(w * h);
  const out = buildSplats({ rgba, w, h, disp, edges, bg: null, params: { ...buildDefaults } });
  // mid pixel: covariance must not be isotropic — the slant direction (y/z for a
  // ground plane) carries the 1/cos(theta) stretch while x keeps sigma0
  const i = (32 * w + 32);
  const cxx = out.cov[i * 6], cyy = out.cov[i * 6 + 3], czz = out.cov[i * 6 + 5];
  assert.ok(cyy + czz > cxx * 1.35, `expected slant stretch, got xx=${cxx} yy=${cyy} zz=${czz}`);
});

await test('buildSplats: AI plate skirt uses plate colors unshaded; bgShade honored', () => {
  const { rgba, disp, w, h } = tinyScene();
  const edges = new Uint8Array(w * h);
  const pad = 8;
  const pw = w + 2 * pad, ph = h + 2 * pad;
  const plate = new Uint8ClampedArray(pw * ph * 4);
  for (let i = 0; i < pw * ph; i++) {
    plate[i * 4] = 250; plate[i * 4 + 1] = 10; plate[i * 4 + 2] = 10; plate[i * 4 + 3] = 255;
  }
  const pDisp = new Float32Array(pw * ph).fill(0.5);
  const out = buildSplats({
    rgba, w, h, disp, edges, bg: null,
    plate: { rgba: plate, disp: pDisp, padPx: pad, pw, ph },
    params: { ...buildDefaults, withSkirt: true, skirtPx: pad },
  });
  assert.ok(out.meta.skirtCount > 0, 'plate skirt present');
  const start = out.meta.fineCount + out.meta.bgCount + out.meta.underCount;
  let sawFade = false;
  for (let i = start; i < out.count; i++) {
    assert.equal(out.colors[i * 4], 250, 'plate color used verbatim (no shade)');
    assert.equal(out.colors[i * 4 + 1], 10);
    if (out.colors[i * 4 + 3] < 200) sawFade = true;
  }
  assert.ok(sawFade, 'outer rim still fades');
  // same depth mapping as interior: constant disparity 0.5 => same z
  const zExpect = 1 / (0.5 * (1 - 1 / 8) + 1 / 8);
  near(-out.positions[start * 3 + 2], zExpect, 1e-3);

  // bgShade: classical darkens, AI (bgShade=1) does not
  const bgMask = new Uint8Array(w * h);
  const bgColor = new Uint8ClampedArray(w * h * 4);
  const bgDisp = new Float32Array(w * h).fill(0.3);
  bgMask[5 * w + 5] = 1;
  bgColor[(5 * w + 5) * 4] = 100; bgColor[(5 * w + 5) * 4 + 1] = 100;
  bgColor[(5 * w + 5) * 4 + 2] = 100; bgColor[(5 * w + 5) * 4 + 3] = 255;
  const mkBg = (shade) => buildSplats({
    rgba, w, h, disp, edges, bg: { bgColor, bgDisp, bgMask },
    params: { ...buildDefaults, withBg: true, bgShade: shade },
  });
  const shaded = mkBg(0.94), unshaded = mkBg(1.0);
  const bi = shaded.meta.fineCount; // first bg splat
  assert.equal(unshaded.colors[bi * 4], 100, 'bgShade=1 keeps AI fill brightness');
  assert.ok(shaded.colors[bi * 4] < 100, 'classical fill still shaded');
});

// ---------------- pipeline worker protocol (simulated) ----------------
await test('pipeline-worker: classical build posts a single final; AI fill falls back offline', async () => {
  const messages = [];
  globalThis.self = {
    onmessage: null,
    postMessage: (m) => messages.push(m),
  };
  // block all network so Inpainter.load fails deterministically in node
  const realFetch = globalThis.fetch;
  globalThis.fetch = () => Promise.reject(new Error('offline test'));
  try {
    await import('../src/pipeline/pipeline-worker.js');
    const handler = globalThis.self.onmessage;
    assert.ok(handler, 'worker registered a handler');

    const w = 32, h = 32;
    const rgba = new Uint8ClampedArray(w * h * 4).fill(200);
    const disp = new Float32Array(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        disp[y * w + x] = (x >= 12 && x < 20 && y >= 12 && y < 20) ? 0.9 : 0.2;
      }
    }
    const params = {
      fovYDeg: 55, zNear: 1, zRange: 7, sizeFactor: 0.65, edgeDispJump: 0.055,
      bgBandPx: 6, skirtPx: 6, underStep: 4,
      withBg: true, withSkirt: true, withUnder: true,
      wantAiDepth: false, aiFill: false, deviceClass: 'desktop',
    };
    const waitFor = async (pred, ms = 4000) => {
      const t0 = Date.now();
      while (!pred()) {
        if (Date.now() - t0 > ms) throw new Error('timed out waiting for worker message');
        await new Promise((r) => setTimeout(r, 20));
      }
    };

    // classical: one 'built', phase final
    handler({ data: {
      type: 'build', id: 1, sourceId: 1, rgba: rgba.slice().buffer, w, h,
      disparity: disp.slice().buffer, params,
    } });
    await waitFor(() => messages.some((m) => m.type === 'built'));
    const built1 = messages.filter((m) => m.type === 'built');
    assert.equal(built1.length, 1);
    assert.equal(built1[0].meta.phase, 'final');
    assert.equal(built1[0].meta.fillKind, 'classical');
    assert.equal(built1[0].meta.depthKind, 'gt');
    assert.ok(built1[0].meta.bgCount > 0, 'classical bg layer present');

    // AI fill requested but model unreachable: preview built + fill-failed
    messages.length = 0;
    handler({ data: {
      type: 'build', id: 2, sourceId: 2, rgba: rgba.slice().buffer, w, h,
      disparity: disp.slice().buffer, params: { ...params, aiFill: true },
    } });
    await waitFor(() => messages.some((m) => m.type === 'fill-failed'));
    const built2 = messages.filter((m) => m.type === 'built');
    assert.equal(built2.length, 1, 'exactly the preview build');
    assert.equal(built2[0].meta.phase, 'preview');
    assert.equal(built2[0].meta.fillKind, 'classical');
    assert.ok(messages.some((m) => m.type === 'fill-failed'), 'fill failure reported');
  } finally {
    globalThis.fetch = realFetch;
    delete globalThis.self;
  }
});

// ---------------- .splat export ----------------
await test('encodeSplatFile: 32-byte records, sane scales and quats', () => {
  const cloud = {
    count: 2,
    positions: new Float32Array([1, 2, -3, -0.5, 0, -2]),
    // isotropic sigma=0.1 and anisotropic diag(0.04, 0.01, 0.0025)
    cov: new Float32Array([0.01, 0, 0, 0.01, 0, 0.01, 0.04, 0, 0, 0.01, 0, 0.0025]),
    colors: new Uint8Array([255, 128, 0, 255, 10, 20, 30, 200]),
  };
  const bytes = encodeSplatFile(cloud);
  assert.equal(bytes.length, 64);
  const f32 = new Float32Array(bytes.buffer);
  near(f32[0], 1); near(f32[1], 2); near(f32[2], -3);
  near(f32[3], 0.1, 1e-4); near(f32[4], 0.1, 1e-4); near(f32[5], 0.1, 1e-4);
  near(f32[8 + 3], 0.2, 1e-4); // second splat, largest scale first
  const q = [bytes[28], bytes[29], bytes[30], bytes[31]].map((v) => (v - 128) / 128);
  near(Math.hypot(...q), 1, 0.03);
  assert.equal(bytes[24], 255); assert.equal(bytes[25], 128);
});

// ---------------- png encoder ----------------
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
await test('sort-worker: orders back-to-front under a view matrix', async () => {
  const messages = [];
  globalThis.self = {
    onmessage: null,
    postMessage: (m) => messages.push(m),
  };
  await import('../src/render/sort-worker.js');
  const handler = globalThis.self.onmessage;
  assert.ok(handler, 'worker registered a handler');

  // camera at origin looking -Z: depth = -z
  const positions = new Float32Array([
    0, 0, -1,   // nearest
    0, 0, -5,   // farthest
    0, 0, -3,   // middle
  ]);
  handler({ data: { type: 'points', positions, count: 3 } });
  const view = M4.lookAt([0, 0, 0], [0, 0, -1], [0, 1, 0]);
  handler({ data: { type: 'sort', view, gen: 1, indices: null } });
  const out = messages.find((m) => m.type === 'sorted');
  assert.ok(out && out.indices, 'sorted reply');
  assert.deepEqual([...out.indices], [1, 2, 0], 'far first, near last');
  delete globalThis.self;
});

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
