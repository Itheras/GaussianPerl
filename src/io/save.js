// Saving: captured pixels -> PNG download (share-sheet friendly on iOS),
// and .splat export encoding (antimatter15 format) via eigendecomposition.

import { eigenSym3, matToQuat } from '../util/math3d.js';

export function pixelsToBlob(pixels, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(new ImageData(pixels, width, height), 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), 'image/png');
  });
}

export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

export async function savePixelsAsPNG(pixels, width, height, filename) {
  const blob = await pixelsToBlob(pixels, width, height);
  downloadBlob(blob, filename);
}

/**
 * Encode a cloud {count, positions, cov, colors} as antimatter15 .splat:
 * 32 bytes/splat: pos f32x3, scale f32x3, rgba u8x4, rot u8x4 (q*128+128, wxyz).
 * Scales/rotation recovered from covariance by eigendecomposition.
 */
export function encodeSplatFile(cloud) {
  const n = cloud.count;
  const buf = new ArrayBuffer(n * 32);
  const f32 = new Float32Array(buf);
  const u8 = new Uint8Array(buf);
  const c = new Float32Array(6);
  for (let i = 0; i < n; i++) {
    const o = i * 8; // f32 stride per splat
    f32[o] = cloud.positions[i * 3];
    f32[o + 1] = cloud.positions[i * 3 + 1];
    f32[o + 2] = cloud.positions[i * 3 + 2];
    for (let k = 0; k < 6; k++) c[k] = cloud.cov[i * 6 + k];
    const { evals, evecs } = eigenSym3(c);
    f32[o + 3] = Math.sqrt(Math.max(evals[0], 1e-12));
    f32[o + 4] = Math.sqrt(Math.max(evals[1], 1e-12));
    f32[o + 5] = Math.sqrt(Math.max(evals[2], 1e-12));
    const bo = i * 32 + 24;
    u8[bo] = cloud.colors[i * 4];
    u8[bo + 1] = cloud.colors[i * 4 + 1];
    u8[bo + 2] = cloud.colors[i * 4 + 2];
    u8[bo + 3] = cloud.colors[i * 4 + 3];
    const q = matToQuat(evecs); // [w,x,y,z], evecs are columns of R
    u8[bo + 4] = Math.max(0, Math.min(255, Math.round(q[0] * 128 + 128)));
    u8[bo + 5] = Math.max(0, Math.min(255, Math.round(q[1] * 128 + 128)));
    u8[bo + 6] = Math.max(0, Math.min(255, Math.round(q[2] * 128 + 128)));
    u8[bo + 7] = Math.max(0, Math.min(255, Math.round(q[3] * 128 + 128)));
  }
  return new Uint8Array(buf);
}
