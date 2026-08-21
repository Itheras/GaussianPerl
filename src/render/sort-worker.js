// Depth-sort worker: 16-bit counting sort of splats by view-space depth,
// back-to-front (far first). Keeps its own copy of positions; main thread
// sends one sort request at a time (ping-pong on the returned buffer).

let positions = null;  // Float32Array, xyz per splat
let count = 0;
const HIST_SIZE = 65536;
let counts = new Uint32Array(HIST_SIZE);
let depthQ = null;     // Uint32Array quantized keys

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'points') {
    positions = msg.positions;
    count = msg.count;
    depthQ = new Uint32Array(count);
    return;
  }
  if (msg.type === 'sort') {
    if (!positions || count === 0) {
      self.postMessage({ type: 'sorted', indices: null, gen: msg.gen });
      return;
    }
    const v = msg.view; // column-major mat4 (world->camera)
    // depth s = -(row2 . p + tz): positive in front of camera
    const r0 = -v[2], r1 = -v[6], r2 = -v[10], r3 = -v[14];
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < count; i++) {
      const s = r0 * positions[i * 3] + r1 * positions[i * 3 + 1] + r2 * positions[i * 3 + 2] + r3;
      if (s < min) min = s;
      if (s > max) max = s;
      // stash raw float bits temporarily via scale below (second pass quantizes)
    }
    const range = max - min;
    const scale = range > 1e-9 ? (HIST_SIZE - 1) / range : 0;
    counts.fill(0);
    for (let i = 0; i < count; i++) {
      const s = r0 * positions[i * 3] + r1 * positions[i * 3 + 1] + r2 * positions[i * 3 + 2] + r3;
      // far -> small bucket index so prefix order = back-to-front
      const q = (HIST_SIZE - 1 - ((s - min) * scale)) | 0;
      depthQ[i] = q;
      counts[q]++;
    }
    let acc = 0;
    for (let b = 0; b < HIST_SIZE; b++) {
      const c = counts[b];
      counts[b] = acc;
      acc += c;
    }
    const indices = (msg.indices && msg.indices.length === count)
      ? msg.indices : new Uint32Array(count);
    for (let i = 0; i < count; i++) {
      indices[counts[depthQ[i]]++] = i;
    }
    self.postMessage({ type: 'sorted', indices, gen: msg.gen }, [indices.buffer]);
  }
};
