// Saving: captured pixels -> PNG download (share-sheet friendly on iOS).
// (.splat export retired with the splat renderer in M8.)

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
