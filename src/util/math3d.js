// Minimal 3D math. Column-major mat4 (OpenGL convention): m[c*4+r].
// Right-handed, camera looks down -Z, Y up.

export const V3 = {
  make: (x = 0, y = 0, z = 0) => new Float32Array([x, y, z]),
  add: (a, b, out = new Float32Array(3)) => {
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2]; return out;
  },
  sub: (a, b, out = new Float32Array(3)) => {
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2]; return out;
  },
  scale: (a, s, out = new Float32Array(3)) => {
    out[0] = a[0] * s; out[1] = a[1] * s; out[2] = a[2] * s; return out;
  },
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b, out = new Float32Array(3)) => {
    const x = a[1] * b[2] - a[2] * b[1];
    const y = a[2] * b[0] - a[0] * b[2];
    const z = a[0] * b[1] - a[1] * b[0];
    out[0] = x; out[1] = y; out[2] = z; return out;
  },
  len: (a) => Math.hypot(a[0], a[1], a[2]),
  normalize: (a, out = new Float32Array(3)) => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    out[0] = a[0] / l; out[1] = a[1] / l; out[2] = a[2] / l; return out;
  },
  lerp: (a, b, t, out = new Float32Array(3)) => {
    out[0] = a[0] + (b[0] - a[0]) * t;
    out[1] = a[1] + (b[1] - a[1]) * t;
    out[2] = a[2] + (b[2] - a[2]) * t; return out;
  },
};

export const M4 = {
  identity: (out = new Float32Array(16)) => {
    out.fill(0); out[0] = out[5] = out[10] = out[15] = 1; return out;
  },

  // a*b (both column-major)
  multiply: (a, b, out = new Float32Array(16)) => {
    const r = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let row = 0; row < 4; row++) {
        r[c * 4 + row] =
          a[row] * b[c * 4] + a[4 + row] * b[c * 4 + 1] +
          a[8 + row] * b[c * 4 + 2] + a[12 + row] * b[c * 4 + 3];
      }
    }
    out.set(r); return out;
  },

  perspective: (fovyRad, aspect, near, far, out = new Float32Array(16)) => {
    const f = 1 / Math.tan(fovyRad / 2);
    out.fill(0);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) / (near - far);
    out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
    return out;
  },

  // World->view matrix from camera basis: right/up/back(+Z of camera) and eye pos.
  view: (right, up, back, eye, out = new Float32Array(16)) => {
    out[0] = right[0]; out[4] = right[1]; out[8] = right[2];
    out[1] = up[0]; out[5] = up[1]; out[9] = up[2];
    out[2] = back[0]; out[6] = back[1]; out[10] = back[2];
    out[12] = -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]);
    out[13] = -(up[0] * eye[0] + up[1] * eye[1] + up[2] * eye[2]);
    out[14] = -(back[0] * eye[0] + back[1] * eye[1] + back[2] * eye[2]);
    out[3] = out[7] = out[11] = 0; out[15] = 1;
    return out;
  },

  lookAt: (eye, target, upHint, out = new Float32Array(16)) => {
    const back = V3.normalize(V3.sub(eye, target));      // camera +Z
    let right = V3.cross(upHint, back);
    if (V3.len(right) < 1e-6) right = V3.make(1, 0, 0);
    right = V3.normalize(right);
    const up = V3.cross(back, right);
    return M4.view(right, up, back, eye, out);
  },

  // Transform point (w=1); returns [x,y,z,w]
  transformPoint4: (m, p) => {
    return [
      m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
      m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
      m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
      m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15],
    ];
  },

  // Inverse of a rigid transform (rotation+translation only).
  invertRigid: (m, out = new Float32Array(16)) => {
    const r00 = m[0], r01 = m[4], r02 = m[8];
    const r10 = m[1], r11 = m[5], r12 = m[9];
    const r20 = m[2], r21 = m[6], r22 = m[10];
    const tx = m[12], ty = m[13], tz = m[14];
    out[0] = r00; out[4] = r10; out[8] = r20;
    out[1] = r01; out[5] = r11; out[9] = r21;
    out[2] = r02; out[6] = r12; out[10] = r22;
    out[12] = -(r00 * tx + r10 * ty + r20 * tz);
    out[13] = -(r01 * tx + r11 * ty + r21 * tz);
    out[14] = -(r02 * tx + r12 * ty + r22 * tz);
    out[3] = out[7] = out[11] = 0; out[15] = 1;
    return out;
  },
};

export const clamp = (x, lo, hi) => (x < lo ? lo : x > hi ? hi : x);
export const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

// Jacobi eigendecomposition of a symmetric 3x3 given as upper triangle
// [xx, xy, xz, yy, yz, zz]. Returns { evals: [l0,l1,l2], evecs: 3 column vectors }
// (descending eigenvalues). Used for cov -> scale/rotation (.splat export).
export function eigenSym3(c) {
  let a = [
    [c[0], c[1], c[2]],
    [c[1], c[3], c[4]],
    [c[2], c[4], c[5]],
  ];
  let v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 24; sweep++) {
    let off = Math.abs(a[0][1]) + Math.abs(a[0][2]) + Math.abs(a[1][2]);
    if (off < 1e-12) break;
    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        if (Math.abs(a[p][q]) < 1e-15) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1)) || 1 / (theta + Math.sqrt(theta * theta + 1));
        const cth = 1 / Math.sqrt(t * t + 1);
        const s = t * cth;
        // Rotate a
        for (let k = 0; k < 3; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = cth * akp - s * akq;
          a[k][q] = s * akp + cth * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = cth * apk - s * aqk;
          a[q][k] = s * apk + cth * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = cth * vkp - s * vkq;
          v[k][q] = s * vkp + cth * vkq;
        }
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[j][j] - a[i][i]);
  const evals = order.map((i) => Math.max(a[i][i], 0));
  const evecs = order.map((i) => [v[0][i], v[1][i], v[2][i]]);
  // Ensure right-handed basis
  const cx = evecs[0][1] * evecs[1][2] - evecs[0][2] * evecs[1][1];
  const cy = evecs[0][2] * evecs[1][0] - evecs[0][0] * evecs[1][2];
  const cz = evecs[0][0] * evecs[1][1] - evecs[0][1] * evecs[1][0];
  if (cx * evecs[2][0] + cy * evecs[2][1] + cz * evecs[2][2] < 0) {
    evecs[2] = evecs[2].map((x) => -x);
  }
  return { evals, evecs };
}

// Rotation matrix (3 column vectors) -> quaternion [w,x,y,z]
export function matToQuat(cols) {
  const m00 = cols[0][0], m10 = cols[0][1], m20 = cols[0][2];
  const m01 = cols[1][0], m11 = cols[1][1], m21 = cols[1][2];
  const m02 = cols[2][0], m12 = cols[2][1], m22 = cols[2][2];
  const tr = m00 + m11 + m22;
  let w, x, y, z;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s;
  }
  const n = Math.hypot(w, x, y, z) || 1;
  return [w / n, x / n, y / n, z / n];
}
