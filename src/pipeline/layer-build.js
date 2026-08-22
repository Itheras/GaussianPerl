// Assemble the two-layer LDI textures the raymarch renderer consumes (M8).
// Layer 0 = the complete photo heightfield, padded by the outpaint ring.
// Layer 1 = band-limited background revealed by parallax (classical fill for
// the instant preview, MI-GAN colors in the final).
// Color lives at WORKING resolution (photo-native sharpness); disparity lives
// at DEPTH resolution (smooth field — the march samples it bilinearly), each
// padded proportionally. Pure typed-array code; unit-tested in node.

import { padPlate, padFloat, smoothRingDisparity } from './fill-plan.js';
import { erodeMaxima } from '../util/imageops.js';

/**
 * Nearest-upsample a depth-res background synthesis onto an arbitrary target
 * grid (generalizes the old 2x block copy).
 */
export function upsampleBackgroundTo(bgD, dw, dh, w, h) {
  const bgColor = new Uint8ClampedArray(w * h * 4);
  const bgDisp = new Float32Array(w * h);
  const bgMask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = Math.min((y * dh / h) | 0, dh - 1);
    for (let x = 0; x < w; x++) {
      const sx = Math.min((x * dw / w) | 0, dw - 1);
      const si = sy * dw + sx, di = y * w + x;
      if (!bgD.bgMask[si]) continue;
      bgMask[di] = 1;
      bgDisp[di] = bgD.bgDisp[si];
      bgColor[di * 4] = bgD.bgColor[si * 4];
      bgColor[di * 4 + 1] = bgD.bgColor[si * 4 + 1];
      bgColor[di * 4 + 2] = bgD.bgColor[si * 4 + 2];
      bgColor[di * 4 + 3] = 255;
    }
  }
  return { bgColor, bgDisp, bgMask };
}

/**
 * Layer 0 color: photo centered in a padded canvas whose ring comes from the
 * outpainted plate when available, else from mirror padding. The PRISTINE
 * photo is always stamped over the interior — the AI plate's interior carries
 * fill colors at bgMask pixels (that content belongs to layer 1, never on top
 * of the photo).
 */
export function buildColor0(rgba, w, h, padPx, plateRgba = null) {
  const pw = w + 2 * padPx;
  const base = plateRgba
    ? Uint8ClampedArray.from(plateRgba)
    : padPlate(rgba, w, h, padPx).plate;
  for (let y = 0; y < h; y++) {
    const src = y * w * 4;
    const dst = ((y + padPx) * pw + padPx) * 4;
    base.set(rgba.subarray(src, src + w * 4), dst);
  }
  return base;
}

/**
 * Layer 0 disparity at depth res, padded: interior = filtered disparity,
 * ring = replicate + smoothRingDisparity (no horizon staircase), then a
 * gentle erosion of near values so silhouettes shrink inside the color edge.
 */
export function buildDisp0(dispD, dw, dh, padD, opts = {}) {
  const pw = dw + 2 * padD, ph = dh + 2 * padD;
  let out = padFloat(dispD, dw, dh, padD);
  if (padD > 0) {
    out = smoothRingDisparity(out, pw, ph, padD, Math.max(8, padD >> 1));
  }
  const erode = opts.erodeIterations ?? 1;
  if (erode > 0) out = erodeMaxima(out, pw, ph, erode);
  return out;
}

/**
 * Layer 1 color at working res, padded: bg fill color where the mask lives,
 * transparent elsewhere, alpha feathered ~1.5px at the mask boundary so the
 * march composites softly. shade darkens classical preview fills (0.94).
 */
export function buildColor1(bg, w, h, padPx, shade = 1) {
  const pw = w + 2 * padPx, ph = h + 2 * padPx;
  const out = new Uint8ClampedArray(pw * ph * 4);
  const m = bg.bgMask;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!m[i]) continue;
      // feather: boundary mask pixels get partial alpha
      let interior = true;
      if (x === 0 || !m[i - 1] || x === w - 1 || !m[i + 1] ||
          y === 0 || !m[i - w] || y === h - 1 || !m[i + w]) interior = false;
      const o = ((y + padPx) * pw + (x + padPx)) * 4;
      out[o] = bg.bgColor[i * 4] * shade;
      out[o + 1] = bg.bgColor[i * 4 + 1] * shade;
      out[o + 2] = bg.bgColor[i * 4 + 2] * shade;
      out[o + 3] = interior ? 255 : 140;
    }
  }
  return out;
}

/**
 * Layer 1 disparity at depth res, padded: bgDisp inside the mask, disp0
 * elsewhere (sane bilinear reads everywhere).
 */
export function buildDisp1(bgD, dw, dh, padD, disp0Padded) {
  const pw = dw + 2 * padD;
  const out = Float32Array.from(disp0Padded);
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const i = y * dw + x;
      if (!bgD.bgMask[i]) continue;
      out[(y + padD) * pw + (x + padD)] = bgD.bgDisp[i];
    }
  }
  return out;
}

/**
 * Assemble everything the renderer needs.
 * args: {rgba, w, h, dispD, dw, dh, padPx, padD,
 *        bgW: working-res {bgColor, bgMask} (fill colors — AI in the final,
 *             classical-upsampled in the preview) or null,
 *        bgD: depth-res {bgDisp, bgMask} (fill geometry) or null,
 *        plateRgba (padded working-res outpaint or null), shade}
 * returns {color0, disp0, color1, disp1, pw, ph, pdw, pdh}
 */
export function buildLayers({ rgba, w, h, dispD, dw, dh, bgW, bgD, padPx, padD,
  plateRgba = null, shade = 1, erodeIterations = 1 }) {
  const color0 = buildColor0(rgba, w, h, padPx, plateRgba);
  const disp0 = buildDisp0(dispD, dw, dh, padD, { erodeIterations });
  const pw = w + 2 * padPx, ph = h + 2 * padPx;
  const pdw = dw + 2 * padD, pdh = dh + 2 * padD;
  let color1, disp1;
  if (bgW && bgD) {
    color1 = buildColor1(bgW, w, h, padPx, shade);
    disp1 = buildDisp1(bgD, dw, dh, padD, disp0);
  } else {
    color1 = new Uint8ClampedArray(pw * ph * 4); // fully transparent
    disp1 = Float32Array.from(disp0);
  }
  return { color0, disp0, color1, disp1, pw, ph, pdw, pdh };
}
