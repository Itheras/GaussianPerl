// Minimal PNG encoder for node (no deps). RGBA8 input, Sub/None/Up filter heuristic.
import zlib from 'node:zlib';

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const out = Buffer.alloc(8 + data.length + 4);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'ascii');
  data.copy(out, 8);
  const body = out.subarray(4, 8 + data.length);
  out.writeUInt32BE(crc32(body), 8 + data.length);
  return out;
}

export function encodePNG(width, height, rgba) {
  if (rgba.length !== width * height * 4) throw new Error('bad rgba size');
  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);
  const src = Buffer.from(rgba.buffer, rgba.byteOffset, rgba.byteLength);
  for (let y = 0; y < height; y++) {
    const rowOff = y * stride;
    // score None / Sub / Up, pick smallest sum of abs (as signed bytes)
    let sNone = 0, sSub = 0, sUp = 0;
    for (let x = 0; x < stride; x++) {
      const v = src[rowOff + x];
      const left = x >= 4 ? src[rowOff + x - 4] : 0;
      const up = y > 0 ? src[rowOff - stride + x] : 0;
      sNone += Math.abs((v << 24) >> 24);
      sSub += Math.abs(((v - left) << 24) >> 24);
      sUp += Math.abs(((v - up) << 24) >> 24);
    }
    const best = Math.min(sNone, sSub, sUp);
    const o = y * (stride + 1);
    if (best === sSub) {
      raw[o] = 1;
      for (let x = 0; x < stride; x++) {
        raw[o + 1 + x] = (src[rowOff + x] - (x >= 4 ? src[rowOff + x - 4] : 0)) & 0xff;
      }
    } else if (best === sUp) {
      raw[o] = 2;
      for (let x = 0; x < stride; x++) {
        raw[o + 1 + x] = (src[rowOff + x] - (y > 0 ? src[rowOff - stride + x] : 0)) & 0xff;
      }
    } else {
      raw[o] = 0;
      src.copy(raw, o + 1, rowOff, rowOff + stride);
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;   // bit depth
  ihdr[9] = 6;   // color type RGBA
  ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', zlib.deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}
