// Generates assets/sample.png + assets/sample_depth.png (ground-truth disparity,
// 16-bit packed into R=hi, G=lo). Deterministic, no deps. Run: node tools/gen-sample.mjs
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { encodePNG } from './png.mjs';

const W = 1024, H = 768;
const rgb = new Float32Array(W * H * 3);   // 0..1
const disp = new Float32Array(W * H);      // 0 (far) .. 1 (near)

// ---------- deterministic noise ----------
function hash2(x, y, seed) {
  let h = (x | 0) * 374761393 + (y | 0) * 668265263 + seed * 2246822519;
  h = (h ^ (h >>> 13)) >>> 0;
  h = Math.imul(h, 1274126177) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}
function vnoise(x, y, seed) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const fx = x - xi, fy = y - yi;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = hash2(xi, yi, seed), b = hash2(xi + 1, yi, seed);
  const c = hash2(xi, yi + 1, seed), d = hash2(xi + 1, yi + 1, seed);
  return a + (b - a) * sx + (c - a) * sy + (a - b - c + d) * sx * sy;
}
function fbm(x, y, seed, oct = 5) {
  let v = 0, amp = 0.5, f = 1;
  for (let o = 0; o < oct; o++) {
    v += amp * vnoise(x * f, y * f, seed + o * 101);
    amp *= 0.5; f *= 2.03;
  }
  return v;
}
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const mix = (a, b, t) => a + (b - a) * t;
const smooth = (a, b, x) => {
  const t = clamp01((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};

function put(x, y, r, g, b, d) {
  const i = y * W + x;
  rgb[i * 3] = r; rgb[i * 3 + 1] = g; rgb[i * 3 + 2] = b;
  if (d !== undefined) disp[i] = d;
}
function blend(x, y, r, g, b, a, d) {
  const i = y * W + x;
  rgb[i * 3] = mix(rgb[i * 3], r, a);
  rgb[i * 3 + 1] = mix(rgb[i * 3 + 1], g, a);
  rgb[i * 3 + 2] = mix(rgb[i * 3 + 2], b, a);
  if (d !== undefined && a > 0.5) disp[i] = d;
}

// ---------- scene parameters ----------
const HOR = Math.round(H * 0.44);           // horizon line
const SUN = { x: W * 0.76, y: H * 0.20 };
const groundDisp = (y) => {
  const t = clamp01((y - HOR) / (H - 1 - HOR));
  return mix(0.30, 1.0, Math.pow(t, 1.15));
};
// haze: mix color toward horizon tint with distance
const hazeCol = [0.94, 0.77, 0.62];
function hazed(r, g, b, d) {
  const f = Math.pow(1 - clamp01(d), 3.2) * 0.55;
  return [mix(r, hazeCol[0], f), mix(g, hazeCol[1], f), mix(b, hazeCol[2], f)];
}

// ---------- sky ----------
for (let y = 0; y < H; y++) {
  const t = y / (H - 1);
  for (let x = 0; x < W; x++) {
    const skyT = clamp01(y / HOR);
    let r = mix(0.22, 0.95, Math.pow(skyT, 1.6));
    let g = mix(0.34, 0.74, Math.pow(skyT, 1.5));
    let b = mix(0.55, 0.58, skyT);
    // sun glow
    const dx = (x - SUN.x) / W, dy = (y - SUN.y) / W;
    const dsun = Math.sqrt(dx * dx + dy * dy);
    const glow = Math.exp(-dsun * dsun * 90) * 0.9 + Math.exp(-dsun * 9) * 0.25;
    r = clamp01(r + glow * 0.95); g = clamp01(g + glow * 0.72); b = clamp01(b + glow * 0.38);
    put(x, y, r, g, b, 0.02);
  }
}
// clouds (upper sky only, soft)
for (let y = 0; y < Math.round(HOR * 0.72); y++) {
  for (let x = 0; x < W; x++) {
    const n = fbm(x / 260 + 7, y / 90 + 2, 31, 5);
    const band = smooth(0.04, 0.30, y / HOR) * (1 - smooth(0.5, 0.72, y / HOR));
    const dens = smooth(0.60, 0.85, n) * band * 0.7;
    if (dens > 0.02) {
      const warm = Math.exp(-Math.hypot((x - SUN.x) / W, (y - SUN.y) / W) * 4);
      blend(x, y, mix(0.98, 1.0, warm), mix(0.92, 0.85, warm), mix(0.92, 0.72, warm), dens);
    }
  }
}

// ---------- mountain ridges ----------
function ridge(seedN, amp, base, dispV, col) {
  const ys = new Int32Array(W);
  for (let x = 0; x < W; x++) {
    const n = fbm(x / 340, seedN, seedN * 7, 5);
    const n2 = fbm(x / 90, seedN + 3, seedN * 13, 4) * 0.25;
    ys[x] = Math.round(base - amp * (n + n2));
  }
  for (let x = 0; x < W; x++) {
    for (let y = Math.max(ys[x], 0); y <= HOR; y++) {
      // subtle slope shading
      const dl = ys[Math.min(x + 3, W - 1)] - ys[Math.max(x - 3, 0)];
      const lit = clamp01(0.5 - dl * 0.03);
      let r = col[0] * mix(0.85, 1.15, lit);
      let g = col[1] * mix(0.85, 1.15, lit);
      let b = col[2] * mix(0.85, 1.12, lit);
      [r, g, b] = hazed(r, g, b, dispV);
      put(x, y, clamp01(r), clamp01(g), clamp01(b), dispV);
    }
  }
}
ridge(11, 84, HOR + 4, 0.055, [0.46, 0.44, 0.58]);
ridge(23, 54, HOR + 3, 0.10, [0.34, 0.36, 0.46]);
ridge(37, 27, HOR + 2, 0.16, [0.24, 0.29, 0.31]);

// ---------- ground ----------
for (let y = HOR; y < H; y++) {
  const d = groundDisp(y);
  for (let x = 0; x < W; x++) {
    const scaleUV = 20 / (0.15 + d);      // pseudo-perspective texture scale
    const n = fbm(x / scaleUV, y / (scaleUV * 0.55), 55, 4);
    const n2 = vnoise(x / 2.3, y / 2.1, 99);
    let r = mix(0.20, 0.34, n) + (n2 - 0.5) * 0.05;
    let g = mix(0.30, 0.46, n) + (n2 - 0.5) * 0.05;
    let b = mix(0.13, 0.20, n) + (n2 - 0.5) * 0.04;
    // dry patches
    const dry = smooth(0.62, 0.8, fbm(x / 300, y / 160, 141, 4));
    r = mix(r, 0.48, dry * 0.5); g = mix(g, 0.40, dry * 0.5); b = mix(b, 0.24, dry * 0.5);
    [r, g, b] = hazed(r, g, b, d);
    put(x, y, clamp01(r), clamp01(g), clamp01(b), d);
  }
}
// path winding from bottom to horizon
for (let y = HOR + 2; y < H; y++) {
  const t = (y - HOR) / (H - HOR);
  const d = groundDisp(y);
  const cx = W * 0.5 + Math.sin(t * 3.1 + 0.6) * W * 0.13 * t + (t - 1) * W * 0.05;
  const halfw = mix(1.5, 46, Math.pow(t, 1.7));
  for (let x = Math.max(0, Math.floor(cx - halfw - 6)); x < Math.min(W, cx + halfw + 6); x++) {
    const e = Math.abs(x - cx) / halfw + (vnoise(x / 6, y / 6, 171) - 0.5) * 0.35;
    const a = 1 - smooth(0.75, 1.05, e);
    if (a > 0.01) {
      const n = vnoise(x / 3, y / 3, 181);
      const rr = mix(0.62, 0.72, n), gg = mix(0.52, 0.60, n), bb = mix(0.38, 0.45, n);
      const [r2, g2, b2] = hazed(rr, gg, bb, d);
      blend(x, y, r2, g2, b2, a * 0.9);
    }
  }
}

// ---------- bushes (mid-ground blobs) ----------
function bush(bx, byF, size, seed) {
  const by = Math.round(byF);
  const d = groundDisp(by);
  const x0 = Math.max(0, Math.round(bx - size * 1.6)), x1 = Math.min(W - 1, Math.round(bx + size * 1.6));
  const y0 = Math.max(0, Math.round(by - size * 1.15)), y1 = Math.min(H - 1, by);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const nx = (x - bx) / (size * 1.35), ny = (y - (by - size * 0.42)) / (size * 0.62);
      const rr = nx * nx + ny * ny;
      const n = fbm(x / 14, y / 14, seed, 4);
      if (rr + (0.5 - n) * 0.9 < 1) {
        const lit = clamp01(0.35 + n * 0.9 - ny * 0.35);
        let r = 0.10 + 0.16 * lit, g = 0.18 + 0.24 * lit, b = 0.07 + 0.09 * lit;
        [r, g, b] = hazed(r, g, b, d);
        put(x, y, r, g, b, d);
      }
    }
  }
}
bush(W * 0.115, H * 0.62, 34, 211);
bush(W * 0.855, H * 0.565, 22, 223);
bush(W * 0.60, H * 0.525, 15, 227);
bush(W * 0.335, H * 0.505, 11, 229);

// ---------- standing stones ----------
function shadow(bx, by, wPx) {
  // sun is upper-right → shadow stretches to the lower-left
  const sx = wPx * 2.2, sy = wPx * 0.38;
  const cx = bx - sx * 0.62;
  for (let y = Math.max(HOR, Math.round(by - sy * 2)); y < Math.min(H, by + sy * 2); y++) {
    for (let x = Math.max(0, Math.round(cx - sx * 1.4)); x < Math.min(W, bx + wPx * 1.8); x++) {
      const nx = (x - cx) / sx, ny = (y - by) / sy;
      // cast shadow + tight contact occlusion under the base
      const cnx = (x - bx) / (wPx * 1.25), cny = (y - by) / (wPx * 0.30);
      const a = Math.exp(-(nx * nx + ny * ny) * 2.2) * 0.42 +
        Math.exp(-(cnx * cnx + cny * cny) * 2.5) * 0.38;
      if (a > 0.01) {
        const i = (y * W + x) * 3;
        const aa = Math.min(a, 0.72);
        rgb[i] *= 1 - aa; rgb[i + 1] *= 1 - aa; rgb[i + 2] *= 1 - aa * 0.9;
      }
    }
  }
}
function stone(bx, byF, hPx, wPx, seed) {
  const by = Math.round(byF);
  const d = groundDisp(by);
  shadow(bx, by, wPx);
  const cy = by - hPx * 0.52;
  const shape = (x, y) => {
    const nx = (x - bx) / wPx, ny = (y - cy) / (hPx * 0.55);
    const ang = Math.atan2(ny, nx);
    const wob = (fbm(Math.cos(ang) * 1.4 + seed, Math.sin(ang) * 1.4, seed * 3, 4) - 0.5) * 0.5
      + (fbm(x / 26, y / 26, seed * 5, 3) - 0.5) * 0.22;
    // taper toward top
    const taper = 1 - 0.35 * clamp01(-ny * 0.5 + 0.5) * 0; // keep simple
    return nx * nx / (taper || 1) + ny * ny + wob - 1;
  };
  const x0 = Math.max(0, Math.floor(bx - wPx * 1.9)), x1 = Math.min(W - 1, Math.ceil(bx + wPx * 1.9));
  const y0 = Math.max(0, Math.floor(cy - hPx * 0.8)), y1 = Math.min(H - 1, by + 2);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const f = shape(x, y);
      if (f < 0) {
        // Normal from the smooth ellipse (avoids spoke artifacts from angular
        // noise), pushed to face the camera near the center; bumped by a
        // spatial fbm gradient for rocky relief.
        const exn = (x - bx) / (wPx * wPx);
        const eyn = (y - cy) / (hPx * 0.55 * hPx * 0.55);
        const el = Math.hypot(exn, eyn) || 1;
        const inside = clamp01(-f);                 // 0 at edge → ~1 at center
        const rim = Math.sqrt(clamp01(1 - inside)); // 1 at edge → 0 at center
        let nx = (exn / el) * rim, ny2 = (eyn / el) * rim;
        const bumpE = 1.5;
        const bumpX = (fbm((x + bumpE) / 16, y / 16, seed * 7, 3) - fbm((x - bumpE) / 16, y / 16, seed * 7, 3)) * 2.2;
        const bumpY = (fbm(x / 16, (y + bumpE) / 16, seed * 7, 3) - fbm(x / 16, (y - bumpE) / 16, seed * 7, 3)) * 2.2;
        nx += bumpX; ny2 += bumpY;
        const nl = Math.hypot(nx, ny2, 1);
        const nz = 1 / nl; nx /= nl; ny2 /= nl;
        // sun from upper-right-front
        const lit = clamp01(nx * 0.55 + -ny2 * 0.35 + nz * 0.55) + 0.08;
        const tex = fbm(x / 9, y / 9, seed * 7, 4);
        let base = mix(0.26, 0.56, lit) + (tex - 0.5) * 0.14;
        let r = base * 1.02, g = base * 0.98, b = base * 0.95;
        // warm rim on sun side
        const rimLight = clamp01(nx * 0.9 - 0.25) * clamp01(1 - nz * 1.4) * 0.9;
        r += rimLight * 0.30; g += rimLight * 0.16; b += rimLight * 0.03;
        // moss near bottom
        const moss = smooth(0.35, 0.9, (y - cy) / (hPx * 0.55)) * smooth(0.45, 0.75, tex);
        r = mix(r, 0.18, moss * 0.5); g = mix(g, 0.30, moss * 0.5); b = mix(b, 0.10, moss * 0.5);
        [r, g, b] = hazed(r, g, b, d);
        put(x, y, clamp01(r), clamp01(g), clamp01(b), d);
      }
    }
  }
}
stone(W * 0.290, H * 0.955, H * 0.375, W * 0.062, 3);   // big fg left
stone(W * 0.735, H * 0.800, H * 0.230, W * 0.047, 5);   // mid right
stone(W * 0.492, H * 0.655, H * 0.105, W * 0.023, 9);   // small center
stone(W * 0.618, H * 0.560, H * 0.050, W * 0.011, 13);  // tiny far
stone(W * 0.170, H * 0.585, H * 0.058, W * 0.013, 17);  // tiny far-left

// ---------- grade: vignette + grain ----------
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 3;
    const vx = (x / W - 0.5) * 2, vy = (y / H - 0.5) * 2;
    const vig = 1 - 0.22 * Math.pow(vx * vx + vy * vy, 1.4);
    const grain = (hash2(x, y, 999) - 0.5) * 0.015;
    for (let k = 0; k < 3; k++) rgb[i + k] = clamp01(rgb[i + k] * vig + grain);
  }
}

// ---------- write files ----------
const outRGBA = new Uint8ClampedArray(W * H * 4);
for (let i = 0; i < W * H; i++) {
  // mild filmic-ish curve
  for (let k = 0; k < 3; k++) {
    const v = rgb[i * 3 + k];
    outRGBA[i * 4 + k] = Math.round(clamp01(v * (1.02 + 0.35 * v - 0.37 * v * v)) * 255);
  }
  outRGBA[i * 4 + 3] = 255;
}
const depthRGBA = new Uint8ClampedArray(W * H * 4);
for (let i = 0; i < W * H; i++) {
  const q = Math.round(clamp01(disp[i]) * 65535);
  depthRGBA[i * 4] = q >> 8;         // R = hi byte
  depthRGBA[i * 4 + 1] = q & 0xff;   // G = lo byte
  depthRGBA[i * 4 + 2] = q >> 8;     // B duplicates hi for human preview
  depthRGBA[i * 4 + 3] = 255;
}

const here = path.dirname(fileURLToPath(import.meta.url));
const assets = path.join(here, '..', 'assets');
fs.mkdirSync(assets, { recursive: true });
fs.writeFileSync(path.join(assets, 'sample.png'), encodePNG(W, H, outRGBA));
fs.writeFileSync(path.join(assets, 'sample_depth.png'), encodePNG(W, H, depthRGBA));
console.log('wrote assets/sample.png and assets/sample_depth.png', `${W}x${H}`);
