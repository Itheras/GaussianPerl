// Analytic 3D ground truth + a CPU mirror of the shader's march, so the
// multi-anchor renderer can be checked against a scene whose right answer is
// known exactly — including from viewpoints no single anchor can explain.
//
// This exists to settle one architectural question: can the anchor
// representation support walking BEHIND a subject, given a generator able to
// supply that view? If it can, "see a person\'s back" is a generator problem,
// not a renderer problem. Pure JS, no GPU, no DOM.
import { camRotation, rayDir, relativePose, marchParams, uvAt, depthAt }
  from '../src/render/pose.js';

export const DSUB = 1.0, DFLOOR = 0.02;

// ---------- analytic scene: a ground plane + axis-aligned boxes ----------
const GROUND_Y = -1;
const boxes = [
  // a "person": a standing slab with a distinct front and back
  { min: [-0.30, -1.0, -3.15], max: [0.30, 0.05, -2.85], id: 1 },
  // a back wall
  { min: [-8, -1.0, -8.2], max: [8, 3.0, -8.0], id: 2 },
  // a side block, so there is real occlusion structure
  { min: [1.2, -1.0, -5.0], max: [2.2, 0.4, -4.0], id: 3 },
];

function hash3(a, b, c) {
  let h = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263) ^ Math.imul(c | 0, 2246822519);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}

// distinct texture per face, so a wrong face is obvious in the error metric
function shade(p, face, id) {
  const u = face === 0 ? p[2] : p[0];
  const v = face === 1 ? p[2] : p[1];
  const chk = ((Math.floor(u * 6) + Math.floor(v * 6)) & 1) ? 0.75 : 0.35;
  const n = hash3(Math.floor(u * 40), Math.floor(v * 40), id * 7 + face);
  const t = chk * (0.85 + 0.3 * n);
  if (id === 1) return face === 2 ? [t, t * 0.35, t * 0.3]   // person FRONT: red
    : face === 5 ? [t * 0.3, t * 0.4, t]                      // person BACK: blue
      : [t * 0.6, t * 0.6, t * 0.6];
  if (id === 2) return [t * 0.45, t * 0.75, t * 0.5];         // wall: green
  if (id === 3) return [t, t * 0.85, t * 0.35];               // block: yellow
  return [t * 0.5, t * 0.45, t * 0.4];                        // ground
}

function traceBox(o, d, b) {
  let tmin = -Infinity, tmax = Infinity, axis = -1, sign = 1;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(d[i]) < 1e-12) {
      if (o[i] < b.min[i] || o[i] > b.max[i]) return null;
      continue;
    }
    const inv = 1 / d[i];
    let t1 = (b.min[i] - o[i]) * inv, t2 = (b.max[i] - o[i]) * inv, s = -1;
    if (t1 > t2) { const tt = t1; t1 = t2; t2 = tt; s = 1; }
    if (t1 > tmin) { tmin = t1; axis = i; sign = s; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return null;
  }
  if (tmin < 1e-4) return null;
  // face code: 0,1,2 = -x,-y,-z ; 3,4,5 = +x,+y,+z
  return { t: tmin, face: axis + (sign > 0 ? 3 : 0), id: b.id };
}

function trace(o, d) {
  let best = null;
  for (const b of boxes) {
    const h = traceBox(o, d, b);
    if (h && (!best || h.t < best.t)) best = h;
  }
  if (Math.abs(d[1]) > 1e-9) {
    const t = (GROUND_Y - o[1]) / d[1];
    if (t > 1e-4 && (!best || t < best.t)) best = { t, face: 4, id: 0 };
  }
  if (!best) return null;
  const p = [o[0] + best.t * d[0], o[1] + best.t * d[1], o[2] + best.t * d[2]];
  return { ...best, p, rgb: shade(p, best.face, best.id) };
}

// ---------- render ground truth from any camera ----------
export function renderGT(cam, w, h) {
  const rgb = new Float32Array(w * h * 3);
  const disp = new Float32Array(w * h);
  const R = cam.R;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const uv = [(x + 0.5) / w, (y + 0.5) / h];
      const dl = rayDir(uv, cam.K);
      // camera -> world: R is world->camera, so apply R^T
      const d = [
        R[0] * dl[0] + R[1] * dl[1] + R[2] * dl[2],
        R[3] * dl[0] + R[4] * dl[1] + R[5] * dl[2],
        R[6] * dl[0] + R[7] * dl[1] + R[8] * dl[2],
      ];
      const hit = trace(cam.C, d);
      const i = y * w + x;
      if (!hit) { disp[i] = DFLOOR; continue; }
      // s along the ray IS the camera-frame depth because dl.z == -1
      disp[i] = Math.min(DSUB / Math.max(hit.t, 1e-4), 4);
      rgb[i * 3] = hit.rgb[0]; rgb[i * 3 + 1] = hit.rgb[1]; rgb[i * 3 + 2] = hit.rgb[2];
    }
  }
  return { rgb, disp, w, h };
}

// ---------- CPU mirror of the shader's march ----------
function sampleDisp(a, u, v) {
  const { w, h, disp } = a;
  const x = Math.min(Math.max(u * w - 0.5, 0), w - 1.001);
  const y = Math.min(Math.max(v * h - 0.5, 0), h - 1.001);
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const g = (xx, yy) => disp[Math.min(yy, h - 1) * w + Math.min(xx, w - 1)];
  return g(x0, y0) * (1 - fx) * (1 - fy) + g(x0 + 1, y0) * fx * (1 - fy)
    + g(x0, y0 + 1) * (1 - fx) * fy + g(x0 + 1, y0 + 1) * fx * fy;
}
function sampleRgb(a, u, v) {
  const x = Math.min(Math.max(Math.round(u * a.w - 0.5), 0), a.w - 1);
  const y = Math.min(Math.max(Math.round(v * a.h - 0.5), 0), a.h - 1);
  const i = (y * a.w + x) * 3;
  return [a.rgb[i], a.rgb[i + 1], a.rgb[i + 2]];
}

const STEPS = 96, REFINE = 10;

function marchAnchor(anchor, cam, uv) {
  const dir = rayDir(uv, cam.K);
  const { m, c } = relativePose(anchor, cam);
  const p = marchParams(m, c, anchor.K, dir, DSUB, anchor.dMax);
  if (!p) return null;
  const dMin = anchor.dMin;
  const range = p.dStart - dMin;
  if (range <= 0) return null;
  const step = range / STEPS;
  let d = p.dStart, dPrev = d, hit = false;
  for (let i = 0; i < STEPS; i++) {
    const [u, v] = uvAt(p, d);
    if (sampleDisp(anchor, u, v) >= d) { hit = true; break; }
    dPrev = d; d -= step;
  }
  if (!hit) return null;
  let lo = d, hi = dPrev;
  for (let i = 0; i < REFINE; i++) {
    const mid = 0.5 * (lo + hi);
    const [u, v] = uvAt(p, mid);
    if (sampleDisp(anchor, u, v) >= mid) lo = mid; else hi = mid;
  }
  const [u, v] = uvAt(p, lo);
  if (u < 0 || u > 1 || v < 0 || v > 1) return null;
  // epipolar-direction stretch, exactly as the shader computes it
  const sl = Math.hypot(p.slope[0], p.slope[1]);
  let conf = p.graze;
  if (sl > 1e-9) {
    const sh = [p.slope[0] / sl, p.slope[1] / sl];
    const du = 1.5 * (Math.abs(sh[0]) / anchor.w + Math.abs(sh[1]) / anchor.h);
    const g = (sampleDisp(anchor, u + sh[0] * du, v + sh[1] * du)
      - sampleDisp(anchor, u - sh[0] * du, v - sh[1] * du)) / (2 * du) * sl;
    const k = Math.abs(1 - g);
    const stretch = Math.max(k, 1 / Math.max(k, 1e-4));
    const t = Math.min(Math.max((stretch - 2.2) / (9.0 - 2.2), 0), 1);
    conf *= 1 - t * t * (3 - 2 * t);
  }
  return { rgb: sampleRgb(anchor, u, v), s: depthAt(p, Math.max(lo, DFLOOR)), conf };
}

export function renderFromAnchors(anchors, cam, w, h) {
  const rgb = new Float32Array(w * h * 3);
  const conf = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const uv = [(x + 0.5) / w, (y + 0.5) / h];
      const cands = [];
      for (const a of anchors) {
        const c = marchAnchor(a, cam, uv);
        if (c && c.conf > 0) cands.push(c);
      }
      const i = y * w + x;
      if (!cands.length) continue;
      let qFront = -1;
      const ok = cands.filter((c) => c.conf >= 0.55);
      for (const c of (ok.length ? ok : cands)) qFront = Math.max(qFront, DSUB / Math.max(c.s, 1e-4));
      const band = Math.min(Math.max(0.012 * 1.0 + 0.035 * qFront, 1e-4), 0.0275);
      let acc = [0, 0, 0], wacc = 0, cacc = 0;
      for (const c of cands) {
        const q = DSUB / Math.max(c.s, 1e-4);
        const behind = Math.max(0, qFront - q);
        const wt = c.conf * Math.exp(-(behind * behind) / (2 * band * band));
        if (wt <= 1e-6) continue;
        acc[0] += c.rgb[0] * wt; acc[1] += c.rgb[1] * wt; acc[2] += c.rgb[2] * wt;
        cacc += c.conf * wt; wacc += wt;
      }
      if (wacc <= 1e-6) continue;
      rgb[i * 3] = acc[0] / wacc; rgb[i * 3 + 1] = acc[1] / wacc; rgb[i * 3 + 2] = acc[2] / wacc;
      conf[i] = cacc / wacc;
    }
  }
  return { rgb, conf };
}

// ---------- helpers for the test ----------
export const W = 128, H = 96, K = [0.9, 1.2];
export const mkCam = (C, yaw, pitch = 0) => ({ R: camRotation(yaw, pitch), C, K });

export function mkAnchor(cam, w = 192, h = 144) {
  const g = renderGT(cam, w, h);
  let dMin = Infinity, dMax = -Infinity;
  for (const d of g.disp) { if (d < dMin) dMin = d; if (d > dMax) dMax = d; }
  return { ...cam, ...g, dMin: Math.max(dMin - 0.02, 0.005), dMax: dMax + 0.02 };
}

/** mean |error| against a ground-truth render, over CONFIDENT pixels only */
export function compare(res, gt, w, h) {
  let err = 0, n = 0, covered = 0;
  for (let i = 0; i < w * h; i++) {
    if (res.conf[i] < 0.5) continue;
    covered++;
    for (let c = 0; c < 3; c++) err += Math.abs(res.rgb[i * 3 + c] - gt.rgb[i * 3 + c]);
    n += 3;
  }
  return { coverage: covered / (w * h), meanAbsErr: n ? err / n : NaN };
}
