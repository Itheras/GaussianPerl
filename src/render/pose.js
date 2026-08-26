// Camera poses and per-anchor raymarch parameters (M9 — free camera).
//
// World frame == the ORIGINAL photo's camera frame: origin at the capture
// point, looking down -Z, Y up, and Z_subject = 1 (subject units, the M8
// convention). Everything else is expressed relative to that.
//
// An ANCHOR is an RGB-D view of the world:
//   R  3x3 world->anchor rotation (column-major, GL mat3 layout)
//   C  anchor camera centre, in world
//   K  (f/W, f/H) — intrinsics in FRAME-NORMALISED units, so they are
//      independent of the pixel resolution the anchor is stored at.
// Anchor 0 is the photo itself: R = I, C = 0.
//
// The load-bearing identity of the renderer: for a ray of the NOVEL camera,
// the sample position in an anchor's image is AFFINE in that anchor's stored
// disparity d. With v = (R_a R_b^T) dir_b, C = R_a (C_b - C_a):
//
//   invVz = 1/v.z,  E = C - v*(C.z*invVz),  F = -v*invVz     (E.z=0, F.z=-1)
//   P(zeta) = E + zeta*F,  zeta = dSub/d  (positive depth in the anchor)
//   u(d) = (0.5 + K.x*F.x) + d * ( K.x*E.x/dSub)
//   v(d) = (0.5 - K.y*F.y) + d * (-K.y*E.y/dSub)
//
// i.e. uv(d) = uv0 + d*slope EXACTLY, for arbitrary relative ROTATION and
// translation. That is why M9 can rotate freely at the same cost per march
// step that M8 paid for translation alone — and with no shear approximation.
// (M8's "rotation shears faces" note was about focal-length ERROR under the
// old approximate model, not about rotation per se.)
//
// The ray parameter s (with dir_b.z == -1 by construction) IS the novel-frame
// positive depth of the sample:  s(d) = sBias + sScale/d.
// Pure functions, no GL, no DOM — unit-tested in node.

/** Column-major 3x3 helpers (m[col*3 + row], the GL mat3 layout). */
export const M3 = {
  identity: () => new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]),

  /** rows r0,r1,r2 (each a 3-vector) -> column-major matrix */
  fromRows: (r0, r1, r2) => new Float32Array([
    r0[0], r1[0], r2[0],
    r0[1], r1[1], r2[1],
    r0[2], r1[2], r2[2],
  ]),

  transpose: (m, out = new Float32Array(9)) => {
    const t = [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
    out.set(t);
    return out;
  },

  /** a*b */
  multiply: (a, b, out = new Float32Array(9)) => {
    const r = new Float32Array(9);
    for (let c = 0; c < 3; c++) {
      for (let row = 0; row < 3; row++) {
        r[c * 3 + row] = a[row] * b[c * 3] + a[3 + row] * b[c * 3 + 1] + a[6 + row] * b[c * 3 + 2];
      }
    }
    out.set(r);
    return out;
  },

  mulVec: (m, v, out = new Float32Array(3)) => {
    const x = m[0] * v[0] + m[3] * v[1] + m[6] * v[2];
    const y = m[1] * v[0] + m[4] * v[1] + m[7] * v[2];
    const z = m[2] * v[0] + m[5] * v[1] + m[8] * v[2];
    out[0] = x; out[1] = y; out[2] = z;
    return out;
  },

  /** v * m == m^T * v — i.e. apply the INVERSE of a rotation matrix. */
  mulVecT: (m, v, out = new Float32Array(3)) => {
    const x = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    const y = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    const z = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
    out[0] = x; out[1] = y; out[2] = z;
    return out;
  },
};

/**
 * Camera basis for a yaw/pitch orientation (no roll — a free camera with roll
 * makes photos read as broken). yaw > 0 turns right, pitch > 0 looks up.
 * Returns {right, up, back, forward} in world coords.
 */
export function cameraBasis(yaw, pitch) {
  const sy = Math.sin(yaw), cy = Math.cos(yaw);
  const sp = Math.sin(pitch), cp = Math.cos(pitch);
  const forward = [sy * cp, sp, -cy * cp];
  const right = [cy, 0, sy];
  const back = [-forward[0], -forward[1], -forward[2]];
  // up = back x right
  const up = [
    back[1] * right[2] - back[2] * right[1],
    back[2] * right[0] - back[0] * right[2],
    back[0] * right[1] - back[1] * right[0],
  ];
  return { right, up, back, forward };
}

/** World->camera rotation (column-major) for a yaw/pitch orientation. */
export function camRotation(yaw, pitch) {
  const { right, up, back } = cameraBasis(yaw, pitch);
  return M3.fromRows(right, up, back);
}

/** Recover (yaw, pitch) from a world->camera rotation built by camRotation. */
export function rotationToYawPitch(R) {
  // forward = -back; back is row 2 of R => (R[2], R[5], R[8])
  const fx = -R[2], fy = -R[5], fz = -R[8];
  const pitch = Math.asin(Math.max(-1, Math.min(1, fy)));
  const yaw = Math.atan2(fx, -fz);
  return { yaw, pitch };
}

/**
 * Novel-camera pixel -> ray direction in the novel camera's own coords.
 * uv is IMAGE uv (v grows down). dir.z is exactly -1, so the ray parameter is
 * the novel-frame positive depth.
 */
export function rayDir(uv, K, out = new Float32Array(3)) {
  out[0] = (uv[0] - 0.5) / K[0];
  out[1] = -(uv[1] - 0.5) / K[1];
  out[2] = -1;
  return out;
}

/**
 * Relative pose of the novel camera as seen from an anchor.
 * Returns {m: novel->anchor rotation (column-major), c: novel centre in anchor
 * coords}. These are exactly the two per-anchor uniforms the shader wants.
 */
export function relativePose(anchor, novel) {
  const m = M3.multiply(anchor.R, M3.transpose(novel.R));
  const d = [novel.C[0] - anchor.C[0], novel.C[1] - anchor.C[1], novel.C[2] - anchor.C[2]];
  const c = M3.mulVec(anchor.R, d);
  return { m, c };
}

/**
 * Per-pixel march parameters for one anchor — the CPU mirror of the shader.
 *   dirNovel: ray direction in NOVEL camera coords (from rayDir)
 *   m, c:     from relativePose
 *   K:        the ANCHOR's (f/W, f/H)
 * Returns null when the anchor cannot see along this ray at all (the ray runs
 * parallel to, or away from, the anchor's optical axis), else
 *   {uv0, slope, sBias, sScale, dStart}
 * with uv(d) = uv0 + d*slope and s(d) = sBias + sScale/d, and dStart the
 * NEAREST disparity the march may begin at (the s >= 0 clamp: samples behind
 * the novel camera are not on the ray).
 */
export function marchParams(m, c, K, dirNovel, dSub, dMax) {
  const v = M3.mulVec(m, dirNovel);
  // gate on the normalised z (a cosine), but keep v unnormalised: scaling v
  // rescales sBias/sScale and turns s from depth into radial distance
  const cosz = v[2] / (Math.hypot(v[0], v[1], v[2]) || 1);
  if (cosz > -0.1) return null; // grazing: the source step goes hyperbolic
  const invVz = 1 / v[2];
  const k = c[2] * invVz;
  const E = [c[0] - v[0] * k, c[1] - v[1] * k, 0];
  const F = [-v[0] * invVz, -v[1] * invVz, -1];
  const uv0 = [0.5 + K[0] * F[0], 0.5 - K[1] * F[1]];
  const slope = [K[0] * E[0] / dSub, -K[1] * E[1] / dSub];
  const sBias = -c[2] * invVz;
  const sScale = -dSub * invVz; // > 0 because v.z < 0
  // s(d) = sBias + sScale/d decreases as d grows; s >= 0 caps the near end
  let dStart = dMax;
  if (sBias < 0) dStart = Math.min(dStart, -sScale / sBias);
  const graze = Math.min(1, Math.max(0, (cosz - (-0.1)) / (-0.25 - (-0.1))));
  return { uv0, slope, sBias, sScale, dStart, graze, E, F, v };
}

/** Anchor-image uv at candidate disparity d. */
export function uvAt(p, d) {
  return [p.uv0[0] + d * p.slope[0], p.uv0[1] + d * p.slope[1]];
}

/** Novel-frame positive depth at candidate disparity d. */
export function depthAt(p, d) {
  return p.sBias + p.sScale / d;
}

/**
 * How well an anchor matches a novel view — used to rank which anchors to
 * march this frame and which to evict. Lower is better.
 * Translation is measured in subject units; rotation in radians, weighted so
 * that ~30 degrees costs about as much as moving one subject distance.
 */
export function poseDistance(a, b) {
  const dx = a.C[0] - b.C[0], dy = a.C[1] - b.C[1], dz = a.C[2] - b.C[2];
  const trans = Math.hypot(dx, dy, dz);
  // relative rotation angle from the trace of R_a R_b^T
  const r = M3.multiply(a.R, M3.transpose(b.R));
  const tr = r[0] + r[4] + r[8];
  const ang = Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2)));
  return trans + ang * (1 / (Math.PI / 6));
}

/** Frame-normalised intrinsics for a photo of w x h pixels with focal fPx. */
export function intrinsicsK(fPx, w, h) {
  return [fPx / w, fPx / h];
}
