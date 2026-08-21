// Build gaussian splats from an image + disparity:
//   - fine layer: one anisotropic, normal-oriented gaussian per pixel
//   - background layer: synthesized disocclusion fill behind silhouettes
//   - underlayer: coarse crack-filling splats
//   - skirt: mirrored fading rim beyond the image borders
// Covariance is Σ = Σᵢ sᵢ² aᵢaᵢᵀ from an orthonormal frame — no quaternions.

import { disparityToDepth } from './depthproc.js';
import { percentile } from '../util/imageops.js';

function addCov(cov, o, ax, ay, az, s2) {
  cov[o] += s2 * ax * ax;
  cov[o + 1] += s2 * ax * ay;
  cov[o + 2] += s2 * ax * az;
  cov[o + 3] += s2 * ay * ay;
  cov[o + 4] += s2 * ay * az;
  cov[o + 5] += s2 * az * az;
}

/**
 * args: {rgba, w, h, disp, edges, bg, plate, params}
 * params: {fovYDeg, zNear, zRange, sizeFactor, skirtPx, underStep,
 *          withSkirt, withUnder, withBg, edgeDispJump, bgShade}
 * plate (optional): {rgba, disp, padPx, pw, ph} — AI-outpainted border plate;
 *   when present the skirt reads real synthesized content from it instead of
 *   mirroring the photo.
 * returns {count, positions, cov, colors, meta}
 */
export function buildSplats({ rgba, w, h, disp, edges, bg, plate, params }) {
  const fovY = (params.fovYDeg ?? 55) * Math.PI / 180;
  const f = (h / 2) / Math.tan(fovY / 2);
  const cx = w / 2, cy = h / 2;
  const sizeFactor = params.sizeFactor ?? 0.65;
  const jump = params.edgeDispJump ?? 0.05;

  const depth = disparityToDepth(disp, params.zNear, params.zRange);

  // capacity: fine + bg + underlayer + skirt (bg counted exactly — a
  // worst-case w*h there would waste ~50 MB of peak memory on phones)
  const underStep = params.underStep ?? 4;
  const skirtPx = params.withSkirt ? (params.skirtPx ?? 24) : 0;
  const skirtCap = skirtPx > 0
    ? Math.ceil(((w + 2 * skirtPx) * (h + 2 * skirtPx) - w * h) / 4) + 16 : 0;
  let bgCap = 0;
  if (params.withBg && bg) {
    for (let i = 0; i < w * h; i++) bgCap += bg.bgMask[i];
  }
  const cap = w * h
    + bgCap
    + (params.withUnder ? (Math.ceil(w / underStep) + 1) * (Math.ceil(h / underStep) + 1) : 0)
    + skirtCap;

  const positions = new Float32Array(cap * 3);
  const cov = new Float32Array(cap * 6);
  const colors = new Uint8Array(cap * 4);
  let n = 0;

  const unproject = (u, v, z) => [(u - cx) * z / f, -(v - cy) * z / f, -z];

  // one-sided-safe depth derivative (avoid crossing discontinuities)
  const dz = (iC, iA, iB, dC, dA, dB, zC, zA, zB) => {
    const okA = Math.abs(dA - dC) < jump;
    const okB = Math.abs(dB - dC) < jump;
    if (okA && okB) return (zA - zB) * 0.5;
    if (okA) return zA - zC;
    if (okB) return zC - zB;
    return 0;
  };

  const emit = (x, y, z, c6, r, g, b, a) => {
    positions[n * 3] = x; positions[n * 3 + 1] = y; positions[n * 3 + 2] = z;
    cov.set(c6, n * 6);
    colors[n * 4] = r; colors[n * 4 + 1] = g; colors[n * 4 + 2] = b; colors[n * 4 + 3] = a;
    n++;
  };

  const c6 = new Float32Array(6);

  // camera-facing disc at position p with radius sigma
  const facingCov = (px, py, pz, sigma, out) => {
    out.fill(0);
    const pl = Math.hypot(px, py, pz) || 1;
    const vx = -px / pl, vy = -py / pl, vz = -pz / pl; // toward camera
    // t1 = normalize(cross(v, up)), degenerate-safe
    let t1x = vz, t1y = 0, t1z = -vx;
    const t1l = Math.hypot(t1x, t1y, t1z);
    if (t1l < 1e-5) { t1x = 1; t1y = 0; t1z = 0; }
    else { t1x /= t1l; t1z /= t1l; }
    const t2x = vy * t1z - vz * t1y;
    const t2y = vz * t1x - vx * t1z;
    const t2z = vx * t1y - vy * t1x;
    const s2 = sigma * sigma;
    addCov(out, 0, t1x, t1y, t1z, s2);
    addCov(out, 0, t2x, t2y, t2z, s2);
    addCov(out, 0, vx, vy, vz, s2 * 0.02);
  };

  // surface-oriented disc: tangent frame from the depth-field gradient (zu, zv),
  // stretched 1/cos(theta) along the tilt so grazing surfaces stay closed.
  // Falls back to camera-facing when the normal is degenerate or view-aligned.
  const orientedCov = (x, y, z, zu, zv, px, py, pz, sigma0, out) => {
    const u = x + 0.5 - cx, v = y + 0.5 - cy;
    const dux = z / f + u * zu / f, duy = -v * zu / f, duz = -zu;
    const dvx = u * zv / f, dvy = -z / f - v * zv / f, dvz = -zv;
    let nx = duy * dvz - duz * dvy;
    let ny = duz * dvx - dux * dvz;
    let nz = dux * dvy - duy * dvx;
    const nl = Math.hypot(nx, ny, nz) || 1;
    nx /= nl; ny /= nl; nz /= nl;
    if (nx * px + ny * py + nz * pz > 0) { nx = -nx; ny = -ny; nz = -nz; }
    const pl = Math.hypot(px, py, pz) || 1;
    const vx = -px / pl, vy = -py / pl, vz = -pz / pl;
    let cosT = nx * vx + ny * vy + nz * vz;
    if (cosT < 0) cosT = 0;
    let t1x = vx - nx * cosT, t1y = vy - ny * cosT, t1z = vz - nz * cosT;
    const t1l = Math.hypot(t1x, t1y, t1z);
    if (t1l < 1e-4) {
      facingCov(px, py, pz, sigma0, out);
      return;
    }
    t1x /= t1l; t1y /= t1l; t1z /= t1l;
    const t2x = ny * t1z - nz * t1y;
    const t2y = nz * t1x - nx * t1z;
    const t2z = nx * t1y - ny * t1x;
    const s1 = sigma0 / Math.max(cosT, 0.35);
    out.fill(0);
    addCov(out, 0, t1x, t1y, t1z, s1 * s1);
    addCov(out, 0, t2x, t2y, t2z, sigma0 * sigma0);
    addCov(out, 0, nx, ny, nz, sigma0 * sigma0 * 0.02);
  };

  // ---------- fine layer ----------
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const ym = Math.max(y - 1, 0), yp = Math.min(y + 1, h - 1);
    for (let x = 0; x < w; x++) {
      const i = row + x;
      const a = rgba[i * 4 + 3];
      if (a < 8) continue; // transparent source pixel
      const z = depth[i];
      const [px, py, pz] = unproject(x + 0.5, y + 0.5, z);
      const sigma0 = sizeFactor * z / f;
      const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];

      if (edges[i]) {
        // silhouette: isotropic camera-facing, feathered
        facingCov(px, py, pz, sigma0 * 0.9, c6);
        emit(px, py, pz, c6, r, g, b, 200);
        continue;
      }

      const xm = Math.max(x - 1, 0), xp = Math.min(x + 1, w - 1);
      const zu = dz(i, row + xp, row + xm, disp[i], disp[row + xp], disp[row + xm],
        z, depth[row + xp], depth[row + xm]);
      const zv = dz(i, yp * w + x, ym * w + x, disp[i], disp[yp * w + x], disp[ym * w + x],
        z, depth[yp * w + x], depth[ym * w + x]);
      orientedCov(x, y, z, zu, zv, px, py, pz, sigma0, c6);
      emit(px, py, pz, c6, r, g, b, 255);
    }
  }
  const fineCount = n;

  // ---------- background (disocclusion) layer ----------
  if (params.withBg && bg) {
    const bgDepth = disparityToDepth(bg.bgDisp, params.zNear, params.zRange);
    // depth-field gradient within the mask, so fills continue the background
    // surface's slant (camera-facing discs open into a dashed lattice at
    // grazing angles)
    const bgGrad = (i, iA, iB) => {
      const okA = bg.bgMask[iA] && Math.abs(bg.bgDisp[iA] - bg.bgDisp[i]) < jump;
      const okB = bg.bgMask[iB] && Math.abs(bg.bgDisp[iB] - bg.bgDisp[i]) < jump;
      if (okA && okB) return (bgDepth[iA] - bgDepth[iB]) * 0.5;
      if (okA) return bgDepth[iA] - bgDepth[i];
      if (okB) return bgDepth[i] - bgDepth[iB];
      return 0;
    };
    // classical fill is darkened to read as soft shadow (hides its smear);
    // AI fill matches surroundings and ships unshaded (bgShade = 1)
    const shade = params.bgShade ?? 0.94;
    for (let y = 0; y < h; y++) {
      const ym = Math.max(y - 1, 0), yp = Math.min(y + 1, h - 1);
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!bg.bgMask[i]) continue;
        const z = bgDepth[i];
        const [px, py, pz] = unproject(x + 0.5, y + 0.5, z);
        const xm = Math.max(x - 1, 0), xp = Math.min(x + 1, w - 1);
        const zu = bgGrad(i, y * w + xp, y * w + xm);
        const zv = bgGrad(i, yp * w + x, ym * w + x);
        const sigma = sizeFactor * z / f * 1.35;
        orientedCov(x, y, z, zu, zv, px, py, pz, sigma, c6);
        emit(px, py, pz, c6,
          bg.bgColor[i * 4] * shade, bg.bgColor[i * 4 + 1] * shade,
          bg.bgColor[i * 4 + 2] * shade, 252);
      }
    }
  }
  const bgCount = n - fineCount;

  // ---------- coarse underlayer (fills cracks when dollying in) ----------
  if (params.withUnder) {
    for (let y = underStep >> 1; y < h; y += underStep) {
      for (let x = underStep >> 1; x < w; x += underStep) {
        const i = y * w + x;
        if (rgba[i * 4 + 3] < 8) continue;
        const z = depth[i] * 1.012; // nudged behind the fine layer
        const [px, py, pz] = unproject(x + 0.5, y + 0.5, z);
        const sigma = sizeFactor * z / f * underStep * 0.8;
        facingCov(px, py, pz, sigma, c6);
        emit(px, py, pz, c6, rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], 255);
      }
    }
  }
  const underCount = n - fineCount - bgCount;

  // ---------- border skirt ----------
  if (skirtPx > 0 && plate && plate.padPx >= skirtPx) {
    // AI-outpainted plate: real synthesized content beyond the borders —
    // full brightness, gentle fade only near the outer rim
    const { rgba: pRgba, disp: pDisp, padPx, pw } = plate;
    const zf2 = params.zNear + params.zRange;
    const dA = 1 / params.zNear - 1 / zf2, dB = 1 / zf2;
    for (let y = -skirtPx; y < h + skirtPx; y += 2) {
      for (let x = -skirtPx; x < w + skirtPx; x += 2) {
        if (x >= 0 && x < w && y >= 0 && y < h) continue;
        const pi = (y + padPx) * pw + (x + padPx);
        if (pRgba[pi * 4 + 3] < 8) continue;
        const distOut = Math.max(x < 0 ? -x : x - w + 1, y < 0 ? -y : y - h + 1, 0);
        const fade = Math.pow(1 - Math.min(distOut / skirtPx, 1), 1.2);
        const alpha = Math.round(240 * fade);
        if (alpha < 10) continue;
        const z = 1 / (pDisp[pi] * dA + dB);
        const [px, py, pz] = unproject(x + 0.5, y + 0.5, z);
        const sigma = sizeFactor * z / f * 2.0;
        facingCov(px, py, pz, sigma, c6);
        emit(px, py, pz, c6,
          pRgba[pi * 4], pRgba[pi * 4 + 1], pRgba[pi * 4 + 2], alpha);
      }
    }
  } else if (skirtPx > 0) {
    // fallback: mirrored fading outpaint from the photo itself
    const mirror = (v, size) => {
      let m = v;
      if (m < 0) m = -m - 1;
      if (m >= size) m = 2 * size - 1 - m;
      return Math.min(Math.max(m, 0), size - 1);
    };
    for (let y = -skirtPx; y < h + skirtPx; y += 2) {
      for (let x = -skirtPx; x < w + skirtPx; x += 2) {
        if (x >= 0 && x < w && y >= 0 && y < h) continue;
        const sx = mirror(x, w), sy = mirror(y, h);
        const i = sy * w + sx;
        if (rgba[i * 4 + 3] < 8) continue;
        const distOut = Math.max(x < 0 ? -x : x - w + 1, y < 0 ? -y : y - h + 1, 0);
        const fade = Math.pow(1 - Math.min(distOut / skirtPx, 1), 1.6);
        const alpha = Math.round(235 * fade);
        if (alpha < 10) continue;
        const z = depth[i];
        const [px, py, pz] = unproject(x + 0.5, y + 0.5, z);
        const sigma = sizeFactor * z / f * 2.4;
        facingCov(px, py, pz, sigma, c6);
        const shade = 0.92;
        emit(px, py, pz, c6,
          rgba[i * 4] * shade, rgba[i * 4 + 1] * shade, rgba[i * 4 + 2] * shade, alpha);
      }
    }
  }

  const meta = {
    fineCount, bgCount, underCount, skirtCount: n - fineCount - bgCount - underCount,
    centerZ: depth[(Math.floor(cy) * w + Math.floor(cx))],
    medianZ: percentile(depth, 0.5),
    nearZ: percentile(depth, 0.03),
    farZ: percentile(depth, 0.97),
    focalPx: f, width: w, height: h,
  };

  // return the cap-sized buffers unsliced — every consumer (renderer textures,
  // sort worker, .splat export) honors `count`, and slicing here would double
  // the peak allocation (~64 MB at 'high') for nothing
  return { count: n, positions, cov, colors, meta };
}
