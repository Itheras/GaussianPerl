// GLSL sources (M9 — free camera, multi-anchor scene).
//
// Pass 1: fullscreen raymarch over N ANCHORS. An anchor is an RGB-D view of
// the world stored as a heightfield (colour at photo resolution, disparity at
// depth resolution). Anchor 0 is the original photo and keeps M8's two layers
// (photo + pre-generated background fill). Anchors 1..N are views GENERATED at
// run time by the inpaint-and-lift loop and live in texture arrays.
//
// For each anchor the sample position is AFFINE in that anchor's disparity —
// see src/render/pose.js for the derivation — so a 6-DoF novel camera costs
// exactly what M8's translation-only camera cost: one constant uv delta per
// march step. Rotation is EXACT here, not an approximation.
//
// Per pixel each anchor yields a candidate {rgb, s, conf} where s is the
// NOVEL-frame depth (the ray parameter, since dir.z == -1). The winner is
// chosen by confidence first, then by "substantially nearer occluder wins".
// Anchor 0 is considered first so it wins every tie: at rest the render is the
// photograph, pixel for pixel.
//
// Pass 2: composite over the background gradient + depth-of-field gather.

export const RAYMARCH_VS = /* glsl */ `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export const MAX_GEN_ANCHORS = 4;

export const RAYMARCH_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
precision highp sampler2DArray;

#define MAXGEN ${MAX_GEN_ANCHORS}
const int MAX_STEPS = 128;
const int REFINE = 7;
const float CONF_OK = 0.55;      // "this anchor really knows this pixel"

in vec2 vUv;
layout(location = 0) out vec4 oColor;  // premultiplied rgb, a = coverage
layout(location = 1) out vec4 oAux;    // depth / confidence / coverage

// ---- anchor 0: the photograph (two layers, padded by the outpaint ring) ----
uniform sampler2D uColor0;
uniform sampler2D uDisp0;
uniform sampler2D uColor1;
uniform sampler2D uDisp1;
uniform vec2 uCropScale;   // interior uv -> padded COLOUR texture uv
uniform vec2 uCropOff;
uniform vec2 uCropScaleD;  // ...and -> padded DISPARITY texture uv. The two pads
uniform vec2 uCropOffD;    // round independently once depth res != colour res,
                           // and sharing one mapping shears depth against colour.
uniform mat3 uBaseM;       // novel -> anchor rotation
uniform vec3 uBaseC;       // novel centre in anchor coords
uniform vec2 uBaseK;       // (f/W, f/H)
uniform vec2 uBaseRange;   // (dMin, dMax)
uniform vec4 uBaseValid;   // valid interior-uv rect, ring included
uniform float uBaseMargin; // border feather width, interior-uv units
uniform vec2 uBaseTexel;   // one disparity texel, in interior-uv units

// ---- generated anchors ----
uniform sampler2DArray uGenColor;  // rgb + a = per-texel trust
uniform sampler2DArray uGenDisp;
uniform mat3 uGenM[MAXGEN];
uniform vec3 uGenC[MAXGEN];
uniform vec2 uGenK[MAXGEN];
uniform vec2 uGenRange[MAXGEN];
uniform float uGenLayer[MAXGEN];   // slice index in the texture arrays
uniform float uGenWeight[MAXGEN];  // fade-in x distance falloff (blend)
uniform float uGenValid[MAXGEN];   // distance falloff only (validity)
uniform int uNumGen;
uniform vec2 uGenTexel;            // one generated-anchor disparity texel, uv

// ---- novel camera / framing ----
uniform vec2 uFitScale;    // viewport uv -> novel image uv (contain fit)
uniform vec2 uFitOff;
uniform vec2 uNovelK;      // (f/W, f/H) of the novel view
uniform float uDSub;       // subject disparity: zeta = uDSub / d
uniform float uDFloor;
uniform int uSteps;
uniform int uStepsGen;
uniform int uTrustBase;    // 1 => a pristine base hit ends the search
uniform float uDepthEncode; // 0 => raw depth in aux.r, else 1/range (packed)
uniform vec2 uStretchRange; // (clean, hopeless) source-texel stretch factors
uniform float uSmearJump;   // min raw disparity jump (over 3 texels) for a wall

struct Ray {
  vec2 uv0;
  vec2 slope;
  float sBias;
  float sScale;
  float dStart;
  float graze;   // 1 = the anchor faces this ray squarely, 0 = edge on
  bool ok;
};

struct Cand {
  vec3 rgb;
  float s;
  float conf;   // validity: does this anchor genuinely know this pixel?
  float prio;   // ownership prior: whose colour should win a tie
};

// uv(d) = uv0 + d*slope ; s(d) = sBias + sScale/d  (s = novel-frame depth)
Ray rayParams(mat3 M, vec3 C, vec2 K, vec3 dirB, float dMin, float dMax) {
  Ray r;
  r.uv0 = vec2(0.0); r.slope = vec2(0.0);
  r.sBias = 0.0; r.sScale = 1.0; r.dStart = dMax; r.graze = 0.0; r.ok = false;
  vec3 v = M * dirB;
  // Rays nearly parallel to an anchor's image plane are useless: the source
  // step per unit disparity goes hyperbolic (measured: median 3.3 uv/disparity
  // below cos 0.1, worst case 5.6e4), so a fixed-step march samples noise.
  // Reject them outright and fade the ones approaching that regime.
  // NOTE: gate on the NORMALISED z, but keep v unnormalised — scaling v would
  // rescale sBias/sScale and quietly turn s from depth into radial distance,
  // which every depth comparison, the DoF pass and the alignment reference all
  // assume it is not.
  float cosz = v.z / max(length(v), 1e-6);
  if (cosz > -0.1) return r;
  r.graze = smoothstep(-0.1, -0.25, cosz);
  float invVz = 1.0 / v.z;
  vec3 E = C - v * (C.z * invVz);       // E.z == 0
  vec3 F = -v * invVz;                  // F.z == -1
  r.uv0 = vec2(0.5 + K.x * F.x, 0.5 - K.y * F.y);
  r.slope = vec2(K.x * E.x, -K.y * E.y) / uDSub;
  r.sBias = -C.z * invVz;
  r.sScale = -uDSub * invVz;            // > 0
  float dStart = dMax;
  // s >= 0: samples behind the novel camera are not on the ray
  if (r.sBias < 0.0) dStart = min(dStart, -r.sScale / r.sBias);
  r.dStart = dStart;
  r.ok = dStart > dMin * 1.0001;
  return r;
}

float depthOf(Ray r, float d) { return r.sBias + r.sScale / d; }

/**
 * How badly this hit is smeared.
 *
 * A heightfield sampled with LINEAR filtering has no discontinuities, so the
 * old "did the ray skip a wall?" test can never fire: the binary search always
 * lands exactly on the ramp that a silhouette becomes, and reports a clean hit.
 * That ramp IS the disocclusion — a handful of source texels dragged across
 * hundreds of screen pixels — and under M8's 3% envelope it was one pixel wide,
 * which is why it never mattered before and matters enormously now.
 *
 * The honest measure is how much the source is magnified ALONG THE EPIPOLAR
 * DIRECTION. Differentiating the hit condition disp(uv0 + d*slope) = d gives a
 * footprint scale of exactly 1/(1 - G) along the epipolar
 * direction, with G = grad(disp) dot slope,
 * and 1 across it. So (1 - G) alone says whether this pixel is looking at real
 * texture or at a rubber sheet — and it is 1 at rest, for any camera, which is
 * what keeps the photograph pixel-exact.
 */
float stretchConf(float g) {
  float k = abs(1.0 - g);
  float stretch = max(k, 1.0 / max(k, 1e-4));
  return 1.0 - smoothstep(uStretchRange.x, uStretchRange.y, stretch);
}

// how far inside its own frame a sample sits, feathered at the border
float frameConf(vec2 uv, vec4 valid, float margin) {
  vec2 a = (uv - valid.xy) / margin;
  vec2 b = (valid.zw - uv) / margin;
  return clamp(min(min(a.x, a.y), min(b.x, b.y)), 0.0, 1.0);
}

/**
 * The outpainted skirt is not the photograph. It is a 10% mirror-and-inpaint
 * guess made once at build time, and where it meets a freshly generated anchor
 * the two inventions disagree — which is visible as a rectangular tonal step
 * exactly on the plate boundary. Rating it below CONF_OK lets it serve as the
 * never-void fallback it was designed to be while the generator replaces it
 * with something that agrees with the rest of the frame.
 * Exactly 1.0 for every uv inside the photo itself, so rest stays pixel-exact.
 */
float ringConf(vec2 uv) {
  vec2 rw = max(-uBaseValid.xy, vec2(1e-4));
  float dx = max(max(-uv.x, uv.x - 1.0), 0.0) / rw.x;
  float dy = max(max(-uv.y, uv.y - 1.0), 0.0) / rw.y;
  return mix(1.0, 0.40, clamp(max(dx, dy), 0.0, 1.0));
}

vec2 baseTex(vec2 uv) { return uv * uCropScale + uCropOff; }
vec2 baseTexD(vec2 uv) { return uv * uCropScaleD + uCropOffD; }
// A sample outside the plate is NOT a surface. With CLAMP_TO_EDGE it would
// return the edge texel — and when a subject touches the frame edge, that is
// the subject's own disparity replicated outward. A ray whose path leaves the
// plate before reaching the subject then "hits" that phantom copy first, at a
// position with zero confidence, and a person standing in plain view becomes
// a hole. Measured on the beach photo: the whole couple, at 5 degrees.
bool inPlate(vec2 uv) {
  return uv.x >= uBaseValid.x && uv.y >= uBaseValid.y &&
         uv.x <= uBaseValid.z && uv.y <= uBaseValid.w;
}
float bDisp0(vec2 uv) { return inPlate(uv) ? texture(uDisp0, baseTexD(uv)).r : -1.0; }
float bDisp1(vec2 uv) { return inPlate(uv) ? texture(uDisp1, baseTexD(uv)).r : -1.0; }
float gDisp(vec2 uv, float layer) {
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return -1.0;
  return texture(uGenDisp, vec3(uv, layer)).r;
}

// ---------------------------------------------------------------- anchor 0
Cand marchBase(vec3 dirB) {
  Cand c;
  c.rgb = vec3(0.0); c.s = 1e6; c.conf = 0.0; c.prio = 1.0;
  float dMin = uBaseRange.x;
  Ray r = rayParams(uBaseM, uBaseC, uBaseK, dirB, dMin, uBaseRange.y);
  if (!r.ok) return c;

  float range = r.dStart - dMin;
  float stepD = range / float(uSteps);
  vec2 duv = -stepD * r.slope;
  vec2 uv = r.uv0 + r.dStart * r.slope;
  float d = r.dStart;
  float dPrev = d;
  bool hit = false;
  for (int i = 0; i < MAX_STEPS; i++) {
    if (i >= uSteps) break;
    if (bDisp0(uv) >= d) { hit = true; break; }
    dPrev = d;
    d -= stepD;
    uv += duv;
  }

  float dHit;
  if (hit) {
    float lo = d, hi = dPrev;
    for (int i = 0; i < REFINE; i++) {
      float mid = 0.5 * (lo + hi);
      if (bDisp0(r.uv0 + mid * r.slope) >= mid) lo = mid; else hi = mid;
    }
    dHit = lo;
  } else {
    dHit = dMin;                        // far shell: sky / backdrop
  }

  vec2 uvHit = r.uv0 + dHit * r.slope;
  float border = frameConf(uvHit, uBaseValid, uBaseMargin) * ringConf(uvHit);
  float surf = bDisp0(uvHit);

  // Epipolar-direction disparity gradient at the hit -> smear factor.
  // GATED on the raw jump: a wall is a CLIFF — a silhouette step the bilinear
  // filter turned into a 1-texel ramp, so the difference over 3 texels is
  // most of the step. Interior relief (a shoulder against a chest) has a tiny
  // per-texel gradient, yet at near range the slope factor inflates it into
  // a "grazing" verdict — measured: 51% of a couple standing in plain view
  // flagged as missing. Only cliffs may be walls.
  float sl = length(r.slope);
  float smear = 1.0;
  if (sl > 1e-9 && hit) {
    vec2 sh = r.slope / sl;
    float du = 1.5 * dot(abs(sh), uBaseTexel);
    float dp = bDisp0(uvHit + sh * du), dm = bDisp0(uvHit - sh * du);
    // The cliff gate must look WIDER than the local gradient: the depth
    // filters can smear a silhouette into a 15-20 texel ramp whose 3-texel
    // jump reads as relief, and a wall that wide was captured into a
    // generated anchor as a real surface (an opaque arm beside a man).
    // Over 12 texels a real surface moves ~0.02 in disparity; a silhouette
    // still moves most of the subject-to-background jump.
    float dp4 = bDisp0(uvHit + sh * du * 4.0), dm4 = bDisp0(uvHit - sh * du * 4.0);
    if (dp >= 0.0 && dm >= 0.0) {
      float dd = dp - dm;
      float cliff = max(abs(dd), (dp4 >= 0.0 && dm4 >= 0.0) ? abs(dp4 - dm4) : 0.0);
      if (cliff > uSmearJump) smear = stretchConf(dd / (2.0 * du) * sl);
    }
  }

  // Did the ray skip THROUGH a torn silhouette wall? Then this pixel is a
  // disocclusion: take layer 1, the pre-generated background.
  float gapThresh = 0.02 * range * (140.0 / float(uSteps));
  float outD = dHit;
  if (hit && surf - dHit > gapThresh) {
    float d1 = dHit - stepD * 0.5;
    bool hit1 = false;
    for (int i = 0; i < MAX_STEPS; i++) {
      if (i >= uSteps) break;
      if (d1 <= dMin) break;
      if (bDisp1(r.uv0 + d1 * r.slope) >= d1) { hit1 = true; break; }
      d1 -= stepD;
    }
    if (!hit1) d1 = dMin;
    float lo1 = d1, hi1 = min(d1 + stepD, dHit);
    for (int i = 0; i < REFINE; i++) {
      float mid = 0.5 * (lo1 + hi1);
      if (bDisp1(r.uv0 + mid * r.slope) >= mid) lo1 = mid; else hi1 = mid;
    }
    vec2 uv1 = r.uv0 + lo1 * r.slope;
    vec4 fill = texture(uColor1, baseTex(uv1));
    vec3 wall = texture(uColor0, baseTex(uvHit)).rgb;   // never-void fallback
    c.rgb = mix(wall, fill.rgb, fill.a);
    outD = mix(dHit, lo1, fill.a);
    // the classical/MI-GAN fill is good but not the photograph; bare wall is a
    // smear and must read as a hole so the generator comes for it
    // The pre-baked background band is a build-time GUESS (a 512px GAN with
    // no scene prior). Rate it drawn-but-replaceable: above the hole line so
    // it fills thin bands before anything better exists, below CONF_OK so a
    // generated anchor that knows the pixel replaces it outright instead of
    // cross-dissolving with it.
    c.conf = border * mix(0.14, 0.52, fill.a) * max(smear, 0.15 * fill.a);
    c.conf *= frameConf(uv1, uBaseValid, uBaseMargin) * ringConf(uv1) * r.graze;
  } else {
    c.rgb = texture(uColor0, baseTex(uvHit)).rgb;
    c.conf = border * (hit ? smear : 0.45) * r.graze;
  }
  c.s = max(depthOf(r, max(outD, uDFloor)), 1e-3);
  return c;
}

// -------------------------------------------------------- generated anchors
Cand marchGen(int i, vec3 dirB) {
  Cand c;
  c.rgb = vec3(0.0); c.s = 1e6; c.conf = 0.0;
  // a stated 10% handicap, so the photograph owns any pixel it can explain —
  // and the fade-in lives HERE, never in validity, or the whole anchor would
  // snap into existence the instant it crossed the validity threshold
  c.prio = 0.9 * uGenWeight[i];
  float layer = uGenLayer[i];
  vec2 rng = uGenRange[i];
  Ray r = rayParams(uGenM[i], uGenC[i], uGenK[i], dirB, rng.x, rng.y);
  if (!r.ok) return c;

  float range = r.dStart - rng.x;
  int steps = uStepsGen;
  float stepD = range / float(steps);
  vec2 duv = -stepD * r.slope;
  vec2 uv = r.uv0 + r.dStart * r.slope;
  float d = r.dStart;
  float dPrev = d;
  bool hit = false;
  for (int k = 0; k < MAX_STEPS; k++) {
    if (k >= steps) break;
    if (gDisp(uv, layer) >= d) { hit = true; break; }
    dPrev = d;
    d -= stepD;
    uv += duv;
  }
  float lo;
  if (hit) {
    lo = d;
    float hi = dPrev;
    for (int k = 0; k < REFINE; k++) {
      float mid = 0.5 * (lo + hi);
      if (gDisp(r.uv0 + mid * r.slope, layer) >= mid) lo = mid;
      else hi = mid;
    }
  } else {
    // Far shell, like the base anchor: invented sky sits at this view's
    // minimum disparity and a ray that gets there without a hit is looking
    // at it. Without this the sky a generated anchor painted is unreachable
    // and the frame goes black where it should be blue.
    lo = rng.x;
  }
  vec2 uvHit = r.uv0 + lo * r.slope;
  if (uvHit.x < 0.0 || uvHit.x > 1.0 || uvHit.y < 0.0 || uvHit.y > 1.0) return c;

  // a ray that fell through a silhouette in THIS view knows nothing here
  float surf = gDisp(uvHit, layer);
  float gapThresh = 0.02 * range * (140.0 / float(steps));
  vec4 texel = texture(uGenColor, vec3(uvHit, layer));
  float trust = texel.a;                 // baked at commit time (see expand.js)
  if (surf - lo > gapThresh) trust *= 0.25;
  float slg = length(r.slope);
  if (slg > 1e-9) {
    vec2 sh = r.slope / slg;
    float du = 1.5 * dot(abs(sh), uGenTexel);
    float dd = gDisp(uvHit + sh * du, layer) - gDisp(uvHit - sh * du, layer);
    float dd4 = gDisp(uvHit + sh * du * 4.0, layer) - gDisp(uvHit - sh * du * 4.0, layer);
    if (max(abs(dd), abs(dd4)) > uSmearJump) trust *= stretchConf(dd / (2.0 * du) * slg);
  }

  c.rgb = texel.rgb;
  // uGenWeight carries BOTH the commit fade-in and the distance-from-capture
  // falloff. The fade-in must not touch validity (it would snap), but the
  // falloff must: a stale anchor that still reads as valid keeps its own smear
  // out of the hole budget, and the pixel is never repaired.
  c.conf = trust * r.graze * uGenValid[i] * (hit ? 1.0 : 0.45);
  c.s = max(depthOf(r, max(lo, uDFloor)), 1e-3);
  return c;
}

/**
 * Agreement band, in DISPARITY. A relative depth margin is the wrong shape:
 * s is linear depth, so a fixed ratio is a couple of thousandths of disparity
 * at the far shell — pure noise, and the whole sky would flip anchors every
 * frame — while at the near plane the same ratio demands almost a full
 * silhouette step, so a generated anchor that correctly sees a thin foreground
 * object loses. Absolute floor + relative term, capped at half a silhouette.
 */
float agreeBand(float q) {
  float b = 0.012 * (uBaseRange.y - uBaseRange.x) + 0.035 * max(q, 0.0);
  return clamp(b, 1e-4, 0.0275);
}

void main() {
  // viewport uv -> novel image uv (contain fit; v grows down)
  vec2 uvImg = vec2(vUv.x, 1.0 - vUv.y) * uFitScale + uFitOff;
  if (uvImg.x < -0.002 || uvImg.x > 1.002 || uvImg.y < -0.002 || uvImg.y > 1.002) {
    oColor = vec4(0.0);
    oAux = vec4(0.0);
    return;
  }
  vec3 dirB = vec3((uvImg.x - 0.5) / uNovelK.x, -(uvImg.y - 0.5) / uNovelK.y, -1.0);

  vec3 cRgb[1 + MAXGEN];
  float cS[1 + MAXGEN];
  float cConf[1 + MAXGEN];
  float cPrio[1 + MAXGEN];
  int nc = 0;

  Cand b = marchBase(dirB);
  cRgb[0] = b.rgb; cS[0] = b.s; cConf[0] = b.conf; cPrio[0] = b.prio;
  nc = 1;
  if (uTrustBase == 0 || b.conf < 0.995) {
    for (int i = 0; i < MAXGEN; i++) {
      if (i >= uNumGen) break;
      Cand g = marchGen(i, dirB);
      cRgb[nc] = g.rgb; cS[nc] = g.s; cConf[nc] = g.conf; cPrio[nc] = g.prio;
      nc++;
    }
  }

  // Pass 1: the front bucket. Whoever is nearest AND sure of itself defines
  // the surface; anything meaningfully behind it is occluded. Computing this
  // before any comparison is what makes the result independent of the order
  // anchors happen to sit in — otherwise committing a fourth anchor can flip
  // pixels that the first two were arguing over.
  // The photograph gets the benefit of the doubt in the ELECTION, not just in
  // the blend. Electing the front surface as an unweighted max over confident
  // candidates lets a single over-near generated depth sample declare itself
  // the surface — and the Gaussian below then treats the photograph as
  // occluded and suppresses it. The colour prior alone cannot save it, because
  // by then the weight is already ~0. A generated anchor must be a clear band
  // nearer than the photo to displace it; a genuinely nearer occluder still
  // can, which is what makes walking around something work.
  float qBase = (cConf[0] >= CONF_OK) ? uDSub / max(cS[0], 1e-4) : -1.0;
  float qFront = qBase;
  bool anyOk = qBase > 0.0;
  for (int k = 1; k < 1 + MAXGEN; k++) {
    if (k >= nc || cConf[k] < CONF_OK) continue;
    float q = uDSub / max(cS[k], 1e-4);
    if (qBase < 0.0 || q > qBase + agreeBand(qBase)) {
      anyOk = true;
      qFront = max(qFront, q);
    }
  }
  if (!anyOk) {
    for (int k = 0; k < 1 + MAXGEN; k++) {
      if (k >= nc || cConf[k] <= 0.0) continue;
      qFront = max(qFront, uDSub / max(cS[k], 1e-4));
    }
  }
  float band = agreeBand(qFront);

  // Pass 2: soft agreement. Anchors that agree on where the surface is
  // cross-dissolve; anchors behind the front are suppressed by a Gaussian
  // that is effectively a hard occlusion test outside the band. Hard argmax
  // instead would draw a one-pixel seam along the locus where the winner
  // flips — and that locus slides across the image as the camera moves, which
  // reads far worse than the double image the blend is trading against.
  vec3 acc = vec3(0.0);
  float wacc = 0.0, sAcc = 0.0, confAcc = 0.0, baseAcc = 0.0;
  for (int k = 0; k < 1 + MAXGEN; k++) {
    if (k >= nc || cConf[k] <= 0.0) continue;
    // A fallback (stretched wall, far shell, faded anchor) exists so that a
    // pixel nobody knows is never void. The moment somebody KNOWS the pixel
    // the fallback must vanish entirely: the Gaussian below only suppresses
    // candidates BEHIND the front, and a silhouette wall sits in FRONT of the
    // background that replaced it — measured as a 25%-opacity ghost arm
    // beside a person at 10 degrees.
    if (anyOk && cConf[k] < CONF_OK) continue;
    float q = uDSub / max(cS[k], 1e-4);
    float behind = max(0.0, qFront - q);
    float w = cConf[k] * cPrio[k] * exp(-(behind * behind) / (2.0 * band * band));
    if (w <= 1e-6) continue;
    acc += cRgb[k] * w;
    sAcc += cS[k] * w;
    confAcc += cConf[k] * w;
    if (k == 0) baseAcc += w;   // provenance: how much of this pixel is PHOTO
    wacc += w;
  }
  if (wacc <= 1e-6) {
    oColor = vec4(0.0);
    oAux = vec4(0.0);
    return;
  }
  vec3 rgb = acc / wacc;
  float sOut = sAcc / wacc;
  float confOut = confAcc / wacc;
  // Provenance, carried out to the completion pass. Content that came from
  // other GENERATED anchors is second-hand: re-inpainting over it, and then
  // re-estimating depth on that, is how these systems rot. Knowing which
  // pixels are still the photograph is what lets the loop refuse to build a
  // scene out of its own output.
  float baseShare = baseAcc / wacc;

  oColor = vec4(rgb, 1.0);
  if (uDepthEncode > 0.0) {
    float e = clamp(sOut * uDepthEncode, 0.0, 1.0) * 65535.0;
    float hi = floor(e / 256.0);
    oAux = vec4(hi / 255.0, (e - hi * 256.0) / 255.0, confOut, baseShare);
  } else {
    oAux = vec4(sOut, confOut, baseShare, 1.0);
  }
}
`;

export const COMPOSITE_VS = /* glsl */ `#version 300 es
precision highp float;
out vec2 vTex;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vTex = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export const COMPOSITE_FS = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 vTex;
out vec4 oColor;

uniform sampler2D uTexColor;   // premultiplied colour, a = coverage
uniform sampler2D uTexAux;     // depth / confidence / coverage
uniform vec2 uViewport;
uniform vec3 uBgTop;
uniform vec3 uBgBottom;
uniform float uFocusDist;
uniform float uDofStrength;    // 0 disables
uniform float uMaxCoC;         // px
uniform float uDepthDecode;    // 0 => raw depth, else range for 8-bit mode
uniform vec3 uGrainSigma;      // the SOURCE photo's measured per-channel noise

const int NTAPS = 16;
const vec2 POISSON[16] = vec2[16](
  vec2(0.0, 0.0),        vec2(0.54, 0.17),   vec2(-0.42, 0.31),  vec2(0.13, -0.58),
  vec2(-0.18, -0.29),    vec2(0.87, -0.27),  vec2(-0.79, -0.19), vec2(0.33, 0.68),
  vec2(-0.51, 0.74),     vec2(0.62, -0.71),  vec2(-0.93, 0.35),  vec2(0.21, 0.21),
  vec2(-0.16, 0.55),     vec2(0.44, -0.28),  vec2(-0.61, -0.56), vec2(0.95, 0.31)
);

vec3 bgAt(vec2 uv) {
  return mix(uBgBottom, uBgTop, clamp(uv.y, 0.0, 1.0));
}

// how much of this pixel is NOT pristine photograph
float syntheticness(vec4 aux) {
  float conf = (uDepthDecode > 0.0) ? aux.b : aux.g;
  float baseShare = (uDepthDecode > 0.0) ? aux.a : aux.b;
  return clamp(max(1.0 - baseShare, 1.0 - conf), 0.0, 1.0);
}

float hash13(vec3 p) {
  p = fract(p * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

/**
 * Sensor noise for invented pixels.
 *
 * Grain is the strongest tell there is: a region with no noise reads as
 * plastic long before anyone questions the geometry. It cannot be baked into
 * the anchor textures, because the raymarch then resamples it bilinearly and
 * generated regions get progressively CLEANER as the camera moves — the exact
 * opposite of a photograph. So it lives here, in screen space, at the very end
 * of the pipeline, after the defocus blur, where real sensor noise happens.
 *
 * Static in screen space by design: it does not crawl frame to frame, and the
 * eye does not track grain the way it tracks features. Amplitude follows
 * sqrt(luma) because photon shot noise does.
 */
vec3 grainAt(vec2 frag, float luma) {
  vec3 a = vec3(hash13(vec3(frag, 1.0)), hash13(vec3(frag, 2.0)), hash13(vec3(frag, 3.0)));
  vec3 b = vec3(hash13(vec3(frag, 4.0)), hash13(vec3(frag, 5.0)), hash13(vec3(frag, 6.0)));
  vec3 n = (a + b - 1.0) * 2.449;   // ~unit variance
  return n * uGrainSigma * sqrt(clamp(luma, 0.02, 1.0) / 0.5);
}

float depthFrom(vec4 aux) {
  return (uDepthDecode > 0.0)
    ? (aux.r * 255.0 * 256.0 + aux.g * 255.0) / 65535.0 * uDepthDecode
    : aux.r;
}

vec4 resolveAt(vec2 uv, out float depth, out float coverage) {
  vec4 c = texture(uTexColor, uv);
  vec4 aux = texture(uTexAux, uv);
  coverage = c.a;
  depth = (coverage > 1e-4) ? depthFrom(aux) : 1e6;
  return vec4(c.rgb + bgAt(uv) * (1.0 - c.a), 1.0);
}

float cocOf(float depth, float coverage) {
  float invZ = (coverage > 0.01) ? 1.0 / max(depth, 1e-3) : 0.0;
  return clamp(uDofStrength * abs(invZ - 1.0 / max(uFocusDist, 1e-3)), 0.0, uMaxCoC);
}

void main() {
  float depth, cov;
  vec4 base = resolveAt(vTex, depth, cov);
  float syn = syntheticness(texture(uTexAux, vTex));

  if (uDofStrength <= 0.0) {
    oColor = base;
    if (syn > 0.002) {
      oColor.rgb += grainAt(gl_FragCoord.xy,
        dot(oColor.rgb, vec3(0.2126, 0.7152, 0.0722))) * syn;
    }
    return;
  }

  float coc = cocOf(depth, cov);
  float ang = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  float ca = cos(ang), sa = sin(ang);
  mat2 rot = mat2(ca, sa, -sa, ca);

  float gatherR = max(coc, 1.5);
  vec3 acc = base.rgb;
  float wacc = 1.0;
  for (int i = 1; i < NTAPS; i++) {
    vec2 offs = rot * POISSON[i] * gatherR;
    vec2 uv = vTex + offs / uViewport;
    float td, tc;
    vec4 tap = resolveAt(uv, td, tc);
    float tapCoC = cocOf(td, tc);
    float w = clamp((max(tapCoC, coc) - length(offs)) + 1.0, 0.0, 1.0);
    acc += tap.rgb * w;
    wacc += w;
  }
  oColor = vec4(acc / wacc, 1.0);
  // after the blur: sensor noise is downstream of the optics
  if (syn > 0.002) {
    oColor.rgb += grainAt(gl_FragCoord.xy,
      dot(oColor.rgb, vec3(0.2126, 0.7152, 0.0722))) * syn;
  }
}
`;
