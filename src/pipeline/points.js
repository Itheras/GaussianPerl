// The scene as an explicit world-space point cloud (M11 exploration).
//
// Every generative-point-cloud direction — a next-splat predictor, a pointmap
// model with persistent state, a splat-completion network — needs the same
// two primitives before any model enters the picture:
//   1. serialise the committed anchors into world-space points (the CONTEXT
//      the guesser conditions on),
//   2. measure whether points from one anchor agree with another anchor's
//      geometry (the consistency check that decides what is context and what
//      is noise).
// Pure typed-array code, node-testable, reuses the pose algebra.

import { M3 } from '../render/pose.js';

// Per-point relation to a witnessed depth map. Kept numeric so a candidate
// with hundreds of thousands of points can be classified without allocating
// one JavaScript object per point.
export const POINT_RELATION = Object.freeze({
  OUTSIDE: 0,
  MATCHED: 1,
  OCCLUDED: 2,
  FLOATING: 3,
});

/**
 * Unproject an anchor's disparity grid into world-space points.
 *   anchor: {R (world->anchor, col-major), C, K:(f/W,f/H), disp, w, h}
 *   opts:   {dSub, dFloor, stride, skipBelow, edgeJump}
 * skipBelow drops far-shell/sky texels (their "depth" is a clamp, not a
 * surface). edgeJump drops texels sitting ON a silhouette cliff: a bilinear
 * disparity grid turns a depth step into a ramp, and unprojecting ramp texels
 * produces points floating in mid-air between the two true surfaces — the
 * point-cloud version of the stretched-wall artifact the renderer already
 * detects with its epipolar smear term.
 * Returns {positions: Float32Array(3n), texel: Int32Array(n), count}.
 */
export function anchorToPoints(anchor, opts = {}) {
  const { R, C, K, disp, w, h } = anchor;
  const dSub = opts.dSub ?? 1;
  const dFloor = opts.dFloor ?? 0.02;
  const stride = opts.stride ?? 1;
  const skipBelow = opts.skipBelow ?? dFloor * 1.05;
  const edgeJump = opts.edgeJump ?? 0;

  const positions = [];
  const texel = [];
  const pCam = new Float32Array(3);
  const pWorld = new Float32Array(3);
  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      const i = y * w + x;
      const d = disp[i];
      if (!(d > skipBelow)) continue;
      if (edgeJump > 0) {
        const xm = Math.max(x - 1, 0), xp = Math.min(x + 1, w - 1);
        const ym = Math.max(y - 1, 0), yp = Math.min(y + 1, h - 1);
        const m = Math.max(
          Math.abs(d - disp[y * w + xm]), Math.abs(d - disp[y * w + xp]),
          Math.abs(d - disp[ym * w + x]), Math.abs(d - disp[yp * w + x]));
        if (m > edgeJump) continue; // cliff texel: a ramp sample, not a surface
      }
      const u = (x + 0.5) / w, v = (y + 0.5) / h;
      const zeta = dSub / Math.max(d, dFloor);
      // dir has z = -1 exactly, so P_cam = dir * zeta
      pCam[0] = (u - 0.5) / K[0] * zeta;
      pCam[1] = -(v - 0.5) / K[1] * zeta;
      pCam[2] = -zeta;
      M3.mulVecT(R, pCam, pWorld); // R^T: anchor -> world rotation
      positions.push(pWorld[0] + C[0], pWorld[1] + C[1], pWorld[2] + C[2]);
      texel.push(i);
    }
  }
  return {
    positions: Float32Array.from(positions),
    texel: Int32Array.from(texel),
    count: texel.length,
  };
}

/**
 * Test world points against another anchor's geometry.
 * Each point projects into anchor B and lands in exactly one bucket:
 *   matched   agrees with B's surface within tol (disparity units)
 *   occluded  behind B's surface — consistent, B simply cannot see it
 *   floating  IN FRONT of B's surface — a genuine inconsistency: B looks
 *             straight through where the point claims solid matter is
 *   outside   outside B's frustum / behind B's camera
 * Floating fraction is the number that matters: it is what a fusion step must
 * drive to zero, and what a learned guesser must not add to.
 */
export function classifyPointsAgainstAnchor(positions, anchorB, opts = {}) {
  const { R, C, K, disp, w, h } = anchorB;
  const dSub = opts.dSub ?? 1;
  const tol = opts.tol ?? 0.02;
  const n = positions.length / 3;
  if (!Number.isInteger(n)) throw new Error('positions length must be divisible by 3');
  const q = new Float32Array(3);
  const pB = new Float32Array(3);
  const labels = new Uint8Array(n);
  let matched = 0, occluded = 0, floating = 0, outside = 0;

  const dispAt = (u, v) => {
    const x = Math.min(Math.max(u * w - 0.5, 0), w - 1.001);
    const y = Math.min(Math.max(v * h - 0.5, 0), h - 1.001);
    const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    const g = (xx, yy) => disp[Math.min(yy, h - 1) * w + Math.min(xx, w - 1)];
    return g(x0, y0) * (1 - fx) * (1 - fy) + g(x0 + 1, y0) * fx * (1 - fy)
      + g(x0, y0 + 1) * (1 - fx) * fy + g(x0 + 1, y0 + 1) * fx * fy;
  };

  for (let i = 0; i < n; i++) {
    q[0] = positions[i * 3] - C[0];
    q[1] = positions[i * 3 + 1] - C[1];
    q[2] = positions[i * 3 + 2] - C[2];
    M3.mulVec(R, q, pB);
    const zeta = -pB[2];
    if (zeta < 1e-4) {
      labels[i] = POINT_RELATION.OUTSIDE;
      outside++;
      continue;
    }
    const u = 0.5 + K[0] * (pB[0] / zeta);
    const v = 0.5 - K[1] * (pB[1] / zeta);
    if (u < 0 || u > 1 || v < 0 || v > 1) {
      labels[i] = POINT_RELATION.OUTSIDE;
      outside++;
      continue;
    }
    const dPoint = dSub / zeta;
    const dSurf = dispAt(u, v);
    if (dPoint > dSurf + tol) {
      labels[i] = POINT_RELATION.FLOATING;
      floating++;
    } else if (dPoint >= dSurf - tol) {
      labels[i] = POINT_RELATION.MATCHED;
      matched++;
    } else {
      labels[i] = POINT_RELATION.OCCLUDED;
      occluded++;
    }
  }
  const tested = matched + occluded + floating;
  return {
    labels,
    tested, matched, occluded, floating, outside,
    matchedFrac: tested ? matched / tested : 0,
    floatingFrac: tested ? floating / tested : 0,
  };
}

/** Aggregate form retained for callers that do not need per-point labels. */
export function crossViewConsistency(positions, anchorB, opts = {}) {
  const { labels: _labels, ...summary } = classifyPointsAgainstAnchor(positions, anchorB, opts);
  return summary;
}

/**
 * Score one sampled completion before it is allowed into permanent memory.
 *
 * A candidate is a complete point prediction, not just the newly invented
 * pixels. That gives the gate three independent signals:
 *   supported     overlaps a surface already committed to the scene
 *   hidden        sits behind a witnessed surface (possible, not verified)
 *   unobserved    no witness can see it (possible, maximally uncertain)
 *   contradicted  floats in front of a witnessed surface (known wrong)
 *
 * Crucially, an empty candidate and a copy of only the already-known surface
 * are both refusals. Closing a confidence mask is not evidence of scene
 * completion. `options` can tune the conservative defaults for a trained
 * model, but every threshold is reported in the result for reproducibility.
 */
export function evaluatePointCandidate(positions, witnesses, options = {}) {
  if (!positions || positions.length % 3 !== 0) {
    throw new Error('candidate positions must be xyz triples');
  }
  const count = positions.length / 3;
  const minPoints = options.minPoints ?? 64;
  const minSupportedPoints = options.minSupportedPoints ?? 32;
  const minSupportFraction = options.minSupportFraction ?? 0.02;
  const minNovelPoints = options.minNovelPoints ?? 16;
  const minNovelFraction = options.minNovelFraction ?? 0.002;
  const maxContradictionFraction = options.maxContradictionFraction ?? 0.02;
  const allowUnwitnessed = options.allowUnwitnessed ?? false;
  const witnessList = Array.from(witnesses || []);

  const supported = new Uint8Array(count);
  const hidden = new Uint8Array(count);
  const contradicted = new Uint8Array(count);
  for (const witness of witnessList) {
    const { labels } = classifyPointsAgainstAnchor(positions, witness, options);
    for (let i = 0; i < count; i++) {
      if (labels[i] === POINT_RELATION.FLOATING) contradicted[i] = 1;
      else if (labels[i] === POINT_RELATION.MATCHED) supported[i] = 1;
      else if (labels[i] === POINT_RELATION.OCCLUDED) hidden[i] = 1;
    }
  }

  let supportedCount = 0, hiddenCount = 0, unobservedCount = 0, contradictedCount = 0;
  for (let i = 0; i < count; i++) {
    // A contradiction wins over support from another noisy view. Permanent
    // memory should be conservative: one trusted camera seeing through a
    // proposed point is enough to make the proposal unsafe.
    if (contradicted[i]) contradictedCount++;
    else if (supported[i]) supportedCount++;
    else if (hidden[i]) hiddenCount++;
    else unobservedCount++;
  }

  const novelCount = hiddenCount + unobservedCount;
  const supportFraction = count ? supportedCount / count : 0;
  const novelFraction = count ? novelCount / count : 0;
  const constrainedCount = supportedCount + contradictedCount;
  const contradictionFraction = constrainedCount ? contradictedCount / constrainedCount : 0;
  const reasons = [];
  if (count < minPoints) reasons.push('insufficient-points');
  if (!witnessList.length && !allowUnwitnessed) reasons.push('no-witnesses');
  if (witnessList.length &&
      (supportedCount < minSupportedPoints || supportFraction < minSupportFraction)) {
    reasons.push('insufficient-known-overlap');
  }
  if (novelCount < minNovelPoints || novelFraction < minNovelFraction) {
    reasons.push('no-coverage-gain');
  }
  if (contradictionFraction > maxContradictionFraction) reasons.push('geometry-contradiction');

  return {
    accepted: reasons.length === 0,
    reasons,
    count,
    supported: supportedCount,
    hidden: hiddenCount,
    unobserved: unobservedCount,
    novel: novelCount,
    contradicted: contradictedCount,
    supportFraction,
    novelFraction,
    contradictionFraction,
    thresholds: {
      minPoints, minSupportedPoints, minSupportFraction,
      minNovelPoints, minNovelFraction, maxContradictionFraction,
    },
  };
}

/**
 * Choose one coherent stochastic sample. Never average candidate geometry:
 * averaging mutually exclusive guesses is the 3D analogue of regressing a
 * blurry mean. If every sample fails the guard, the explicit result is a
 * refusal and nothing should be committed.
 */
export function selectPointCandidate(candidates, witnesses, options = {}) {
  const evaluated = Array.from(candidates || []).map((candidate, index) => {
    const evaluation = evaluatePointCandidate(candidate.positions, witnesses, options);
    const uncertainty = Math.max(0, Math.min(1, candidate.uncertainty ?? 0));
    const score = evaluation.supported
      + 0.35 * evaluation.hidden
      + 0.05 * evaluation.unobserved
      - 20 * evaluation.contradicted
      - uncertainty * evaluation.count;
    return { index, candidate, evaluation, score };
  });
  const accepted = evaluated.filter((x) => x.evaluation.accepted)
    .sort((a, b) => b.score - a.score || a.index - b.index);
  return {
    accepted: accepted.length > 0,
    chosen: accepted.length ? accepted[0].candidate : null,
    chosenIndex: accepted.length ? accepted[0].index : -1,
    score: accepted.length ? accepted[0].score : -Infinity,
    evaluated,
    reason: accepted.length ? null : 'all-candidates-refused',
  };
}
