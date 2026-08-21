// Image loading: file/drag/paste/sample -> working-resolution ImageData.
// iOS HEIC decodes natively through the browser's own decoder.

import { resizeFloat } from '../util/imageops.js';
import { decodeGtDisparity } from '../pipeline/depthproc.js';

export function workingSize(natW, natH, maxPixels) {
  const scale = Math.min(1, Math.sqrt(maxPixels / (natW * natH)));
  const w = Math.max(64, Math.round(natW * scale));
  const h = Math.max(64, Math.round(natH * scale));
  return { w, h };
}

async function decodeToSource(blob) {
  // <img> first: Safari's createImageBitmap(blob) IGNORES EXIF orientation
  // (WebKit bug 237895) and succeeds anyway, so a try/catch can't save us —
  // iPhone portrait photos would build sideways splats. The <img> path applies
  // orientation correctly in every engine, and we rasterize to 2D right after,
  // so ImageBitmap had no advantage here.
  const url = URL.createObjectURL(blob);
  try {
    const img = new Image();
    img.decoding = 'async';
    img.src = url;
    await img.decode();
    return img;
  } catch {
    if (typeof createImageBitmap === 'function') {
      return await createImageBitmap(blob); // exotic formats <img> rejects
    }
    throw new Error('could not decode image');
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }
}

function drawToImageData(src, w, h) {
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

/** blob -> {imageData} at working resolution */
export async function loadImageBlob(blob, maxPixels) {
  const src = await decodeToSource(blob);
  const natW = src.naturalWidth || src.width;
  const natH = src.naturalHeight || src.height;
  if (!natW || !natH) throw new Error('could not decode image');
  const { w, h } = workingSize(natW, natH, maxPixels);
  const imageData = drawToImageData(src, w, h);
  if (src.close) src.close();
  return { imageData, natW, natH };
}

/** Bundled sample: image + ground-truth disparity (works fully offline). */
export async function loadSample(maxPixels, baseUrl = '.') {
  const [imgResp, depResp] = await Promise.all([
    fetch(`${baseUrl}/assets/sample.png`),
    fetch(`${baseUrl}/assets/sample_depth.png`),
  ]);
  if (!imgResp.ok || !depResp.ok) throw new Error('sample assets missing');
  const [imgBlob, depBlob] = await Promise.all([imgResp.blob(), depResp.blob()]);

  const src = await decodeToSource(imgBlob);
  const natW = src.naturalWidth || src.width;
  const natH = src.naturalHeight || src.height;
  const { w, h } = workingSize(natW, natH, maxPixels);
  const imageData = drawToImageData(src, w, h);
  if (src.close) src.close();

  const depSrc = await decodeToSource(depBlob);
  const depData = drawToImageData(depSrc, depSrc.width, depSrc.height);
  if (depSrc.close) depSrc.close();
  const gtNative = decodeGtDisparity(depData.data, depData.width, depData.height);
  const disparity = (depData.width === w && depData.height === h)
    ? gtNative : resizeFloat(gtNative, depData.width, depData.height, w, h);

  return { imageData, disparity, natW, natH };
}

/** Wire up drag-drop + paste on an element; cb(blob) on any image received. */
export function bindImageDrop(el, cb) {
  el.addEventListener('dragover', (e) => { e.preventDefault(); el.classList.add('dropping'); });
  el.addEventListener('dragleave', () => el.classList.remove('dropping'));
  el.addEventListener('drop', (e) => {
    e.preventDefault();
    el.classList.remove('dropping');
    const f = [...(e.dataTransfer?.files || [])].find((f) => f.type.startsWith('image/'));
    if (f) cb(f);
  });
  window.addEventListener('paste', (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
    const f = item?.getAsFile();
    if (f) cb(f);
  });
}
