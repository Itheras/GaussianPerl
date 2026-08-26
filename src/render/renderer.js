// WebGL2 multi-anchor heightfield renderer (M9).
// Pass 1: fullscreen raymarch over anchor 0 (the photo, two layers) plus the
//         generated anchors resident in texture arrays -> offscreen MRT
//         (premultiplied colour + depth/confidence). No geometry, no sorting.
// Pass 2: composite over the background with optional depth-of-field gather.
//
// The renderer owns the anchor store: main.js commits generated anchors, the
// renderer ranks them against the novel camera each frame and marches the
// closest MAX_GEN_ANCHORS of them.

import { RAYMARCH_VS, RAYMARCH_FS, COMPOSITE_VS, COMPOSITE_FS, MAX_GEN_ANCHORS } from './shaders.js';
import { M3, relativePose, poseDistance } from './pose.js';
import { toHalfFloat } from '../util/imageops.js';

const DEPTH_RANGE_8BIT = 64.0; // world units encodable in the packed depth

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    throw new Error(`Shader compile failed: ${log}\n---\n${src.split('\n').map((l, i) => `${i + 1}: ${l}`).join('\n')}`);
  }
  return sh;
}

function link(gl, vsSrc, fsSrc) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`Program link failed: ${gl.getProgramInfoLog(prog)}`);
  }
  return prog;
}

function uniforms(gl, prog, names) {
  const u = {};
  for (const n of names) u[n] = gl.getUniformLocation(prog, n);
  return u;
}

const IDENTITY3 = M3.identity();

export class LayerRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: true, preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 not available');
    this.gl = gl;
    this.contextLost = false;
    this.onContextLost = null;
    this.onContextRestored = null;

    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      this.contextLost = true;
      if (this.onContextLost) this.onContextLost();
    });
    canvas.addEventListener('webglcontextrestored', () => {
      this.contextLost = false;
      this._initGL();
      if (this.onContextRestored) this.onContextRestored(); // re-upload everything
    });

    // anchor store (generated views); anchor 0 lives in the base textures
    this.anchorCapacity = 0;
    this.anchors = [];        // {slot, R, C, K, dMin, dMax, weight, born}
    this.genDims = null;      // {cw, ch, dw, dh}
    this._marched = [];       // anchors marched last frame (debug/tests)

    this._initGL();
  }

  _initGL() {
    const gl = this.gl;
    if (gl.getExtension('EXT_color_buffer_float')) {
      this.rtFormat = gl.RGBA16F; this.rtType = gl.HALF_FLOAT; this.depthEncoded = false;
    } else if (gl.getExtension('EXT_color_buffer_half_float')) {
      this.rtFormat = gl.RGBA16F; this.rtType = gl.HALF_FLOAT; this.depthEncoded = false;
    } else {
      this.rtFormat = gl.RGBA8; this.rtType = gl.UNSIGNED_BYTE; this.depthEncoded = true;
    }

    // Half-float disparity rows are pdw*2 bytes. WebGL's default 4-byte row
    // alignment then declares the upload buffer "too small" whenever that is
    // not a multiple of 4 — INVALID_OPERATION, texture silently stays zero,
    // the march never hits, every pixel reads as far shell. Width 1187 failed;
    // 918 and 1124 happened to work, which is why no test caught it.
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);

    this.progMarch = link(gl, RAYMARCH_VS, RAYMARCH_FS);
    this.uMarch = uniforms(gl, this.progMarch, [
      'uColor0', 'uDisp0', 'uColor1', 'uDisp1', 'uGenColor', 'uGenDisp',
      'uCropScale', 'uCropOff', 'uCropScaleD', 'uCropOffD', 'uBaseM', 'uBaseC', 'uBaseK', 'uBaseRange',
      'uBaseValid', 'uBaseMargin', 'uBaseTexel', 'uGenTexel', 'uStretchRange', 'uSmearJump',
      'uGenM[0]', 'uGenC[0]', 'uGenK[0]', 'uGenRange[0]', 'uGenLayer[0]',
      'uGenWeight[0]', 'uGenValid[0]', 'uNumGen',
      'uFitScale', 'uFitOff', 'uNovelK', 'uDSub', 'uDFloor',
      'uSteps', 'uStepsGen', 'uTrustBase', 'uDepthEncode',
    ]);
    this.progComp = link(gl, COMPOSITE_VS, COMPOSITE_FS);
    this.uComp = uniforms(gl, this.progComp, [
      'uTexColor', 'uTexAux', 'uViewport', 'uBgTop', 'uBgBottom',
      'uFocusDist', 'uDofStrength', 'uMaxCoC', 'uDepthDecode', 'uGrainSigma',
    ]);

    this.vaoEmpty = gl.createVertexArray();

    // scratch uniform buffers for the generated-anchor arrays
    this._uM = new Float32Array(9 * MAX_GEN_ANCHORS);
    this._uC = new Float32Array(3 * MAX_GEN_ANCHORS);
    this._uK = new Float32Array(2 * MAX_GEN_ANCHORS);
    this._uRange = new Float32Array(2 * MAX_GEN_ANCHORS);
    this._uLayer = new Float32Array(MAX_GEN_ANCHORS);
    this._uWeight = new Float32Array(MAX_GEN_ANCHORS);
    this._uValid = new Float32Array(MAX_GEN_ANCHORS);

    // after a context loss every old GL object is invalid — drop references
    this.texColor0 = null; this.texDisp0 = null;
    this.texColor1 = null; this.texDisp1 = null;
    this.texGenColor = null; this.texGenDisp = null;
    this.layers = null;
    this.fbo = null; this.fboColor = null; this.fboAux = null;
    this.fboW = 0; this.fboH = 0;
    this.capFbo = null; this.capColor = null; this.capAux = null;
    this.capW = 0; this.capH = 0;
    this._probeFbo = null; this._probeColor = null; this._probeAux = null;
  }

  _colorTexture(w, h, rgba) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE,
      rgba instanceof Uint8Array ? rgba : new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength));
    return tex;
  }

  _dispTexture(w, h, f32) {
    // R16F + LINEAR: half-float filtering is core WebGL2 everywhere
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R16F, w, h);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RED, gl.HALF_FLOAT, toHalfFloat(f32));
    return tex;
  }

  /**
   * layers: {color0, disp0, color1, disp1, pw, ph, pdw, pdh, w, h, padPx, padD}
   * (w, h = interior image dims; pads relate texture->interior mapping)
   */
  setLayers(layers) {
    const gl = this.gl;
    if (this.contextLost || gl.isContextLost()) { this.layers = layers; return; }
    for (const t of [this.texColor0, this.texDisp0, this.texColor1, this.texDisp1]) {
      if (t) gl.deleteTexture(t);
    }
    this.layers = layers;
    this.texColor0 = this._colorTexture(layers.pw, layers.ph, layers.color0);
    this.texColor1 = this._colorTexture(layers.pw, layers.ph, layers.color1);
    this.texDisp0 = this._dispTexture(layers.pdw, layers.pdh, layers.disp0);
    this.texDisp1 = this._dispTexture(layers.pdw, layers.pdh, layers.disp1);
  }

  // ---------------------------------------------------------------- anchors

  /** Allocate the generated-anchor texture arrays. Drops any existing ones. */
  initAnchorStore({ cw, ch, dw, dh, capacity }) {
    const gl = this.gl;
    this.clearAnchors();
    if (this.texGenColor) gl.deleteTexture(this.texGenColor);
    if (this.texGenDisp) gl.deleteTexture(this.texGenDisp);
    this.genDims = { cw, ch, dw, dh };
    this.anchorCapacity = capacity;
    if (this.contextLost || gl.isContextLost()) return;

    const mk = (target, fmt, w, h, n) => {
      const t = gl.createTexture();
      gl.bindTexture(target, t);
      gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texStorage3D(target, 1, fmt, w, h, n);
      return t;
    };
    this.texGenColor = mk(gl.TEXTURE_2D_ARRAY, gl.RGBA8, cw, ch, capacity);
    this.texGenDisp = mk(gl.TEXTURE_2D_ARRAY, gl.R16F, dw, dh, capacity);
  }

  clearAnchors() {
    this.anchors = [];
    this._marched = [];
  }

  /**
   * Commit a generated anchor.
   * a: {slot?, R, C, K, dMin, dMax, color (RGBA at cw x ch, alpha = trust),
   *     disp (Float32 at dw x dh), weight}
   * Returns the slot used, or -1 if the store is not ready.
   */
  addAnchor(a) {
    const gl = this.gl;
    if (!this.genDims || !this.texGenColor) return -1;
    const { cw, ch, dw, dh } = this.genDims;
    let slot = a.slot;
    if (slot === undefined || slot < 0) {
      const used = new Set(this.anchors.map((x) => x.slot));
      slot = -1;
      for (let i = 0; i < this.anchorCapacity; i++) {
        if (!used.has(i)) { slot = i; break; }
      }
      if (slot < 0) return -1; // caller must evict first
    }
    if (!this.contextLost && !gl.isContextLost()) {
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texGenColor);
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, slot, cw, ch, 1,
        gl.RGBA, gl.UNSIGNED_BYTE,
        a.color instanceof Uint8Array ? a.color
          : new Uint8Array(a.color.buffer, a.color.byteOffset, a.color.byteLength));
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texGenDisp);
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, slot, dw, dh, 1,
        gl.RED, gl.HALF_FLOAT, toHalfFloat(a.disp));
    }
    const rec = {
      slot,
      R: Float32Array.from(a.R),
      C: Float32Array.from(a.C),
      K: Float32Array.from(a.K),
      dMin: a.dMin, dMax: a.dMax,
      weight: a.weight ?? 1,
    };
    const existing = this.anchors.findIndex((x) => x.slot === slot);
    if (existing >= 0) this.anchors[existing] = rec; else this.anchors.push(rec);
    return slot;
  }

  removeAnchor(slot) {
    this.anchors = this.anchors.filter((a) => a.slot !== slot);
  }

  /** Anchors ranked by how well they match this camera (best first). */
  rankAnchors(cam, limit = MAX_GEN_ANCHORS) {
    return this.anchors
      .map((a) => ({ a, d: poseDistance(a, cam) }))
      .sort((x, y) => x.d - y.d)
      .slice(0, limit)
      .map((x) => x.a);
  }

  // ------------------------------------------------------------- rendering

  _ensureFbo(w, h) {
    const gl = this.gl;
    if (this.fbo && this.fboW === w && this.fboH === h) return;
    if (this.fbo) {
      gl.deleteFramebuffer(this.fbo);
      gl.deleteTexture(this.fboColor);
      gl.deleteTexture(this.fboAux);
    }
    this.fboW = w; this.fboH = h;
    const mk = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texStorage2D(gl.TEXTURE_2D, 1, this.rtFormat, w, h);
      return t;
    };
    this.fboColor = mk();
    this.fboAux = mk();
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboColor, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.fboAux, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      if (this.rtFormat !== gl.RGBA8) {
        this.rtFormat = gl.RGBA8; this.rtType = gl.UNSIGNED_BYTE; this.depthEncoded = true;
        this.fboW = 0;
        this._ensureFbo(w, h);
        return;
      }
      throw new Error(`FBO incomplete: 0x${status.toString(16)}`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  resize(pixelW, pixelH) {
    if (this.canvas.width !== pixelW || this.canvas.height !== pixelH) {
      this.canvas.width = pixelW;
      this.canvas.height = pixelH;
    }
  }

  /**
   * Bind everything pass 1 needs.
   * state: {cam:{R,C,K}, dSub, dMin, dMax, dFloor, steps, stepsGen,
   *         fit:{scale,off}, novelK, trustBase, encode}
   */
  _bindMarch(state, encode) {
    const gl = this.gl;
    const L = this.layers;
    gl.useProgram(this.progMarch);
    gl.bindVertexArray(this.vaoEmpty);

    const bind2d = (unit, tex, loc) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
    };
    bind2d(0, this.texColor0, this.uMarch.uColor0);
    bind2d(1, this.texDisp0, this.uMarch.uDisp0);
    bind2d(2, this.texColor1, this.uMarch.uColor1);
    bind2d(3, this.texDisp1, this.uMarch.uDisp1);
    gl.activeTexture(gl.TEXTURE0 + 6);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texGenColor);
    gl.uniform1i(this.uMarch.uGenColor, 6);
    gl.activeTexture(gl.TEXTURE0 + 7);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.texGenDisp);
    gl.uniform1i(this.uMarch.uGenDisp, 7);

    // interior image uv -> padded texture uv, colour and depth separately
    gl.uniform2f(this.uMarch.uCropScale, L.w / L.pw, L.h / L.ph);
    gl.uniform2f(this.uMarch.uCropOff, L.padPx / L.pw, L.padPx / L.ph);
    gl.uniform2f(this.uMarch.uCropScaleD, L.dw / L.pdw, L.dh / L.pdh);
    gl.uniform2f(this.uMarch.uCropOffD, L.padD / L.pdw, L.padD / L.pdh);

    // base anchor: world frame, so its pose is the identity pose
    const cam = state.cam;
    const base = { R: IDENTITY3, C: [0, 0, 0] };
    const rel = relativePose(base, cam);
    gl.uniformMatrix3fv(this.uMarch.uBaseM, false, rel.m);
    gl.uniform3fv(this.uMarch.uBaseC, rel.c);
    gl.uniform2fv(this.uMarch.uBaseK, state.baseK);
    gl.uniform2f(this.uMarch.uBaseRange, state.dMin, state.dMax);
    // valid rect = the padded plate, expressed in interior uv
    const px = L.padPx / L.w, py = L.padPx / L.h;
    gl.uniform4f(this.uMarch.uBaseValid, -px, -py, 1 + px, 1 + py);
    // feather inside the RING only, so the photo itself stays fully confident
    gl.uniform1f(this.uMarch.uBaseMargin, Math.max(0.5 * Math.min(px, py), 0.002));
    // one disparity texel, in INTERIOR uv
    gl.uniform2f(this.uMarch.uBaseTexel, 1 / L.dw, 1 / L.dh);
    const gd = this.genDims;
    gl.uniform2f(this.uMarch.uGenTexel, gd ? 1 / gd.dw : 1, gd ? 1 / gd.dh : 1);
    // (clean, hopeless) source-texel stretch. The old [2.2, 9] rated a 3x
    // stretched silhouette rim 96% confident, so it was captured into the
    // first generated anchor as a real surface and rendered — stretched
    // further — from every later pose: a translucent second arm beside a
    // person at 10 degrees. A 2x stretch is already missing data.
    // Past ~3x a sample is rubber; below 1.5x it is a surface seen a little
    // obliquely. The cliff gate above decides WHERE this test applies.
    // With the wide cliff gate restricting this test to real silhouettes,
    // the range can be strict: a 1.5x-stretched wall texel is already a
    // smear (at [1.5, 3.0] it scored 1.0 and was captured into generated
    // anchors as known content — the opaque strip beside a person).
    gl.uniform2fv(this.uMarch.uStretchRange, state.stretchRange || [1.15, 1.8]);
    // The cliff gate is the real wall/relief discriminator: a silhouette wall
    // spans the whole subject-to-background jump (0.3-0.7) over three texels;
    // a nose, a braid or a shoulder spans < 0.1. At 0.03 the stretch test
    // reached the relief and a tightened range then masked a man's head out
    // of his own photo (the model obligingly painted a different man).
    gl.uniform1f(this.uMarch.uSmearJump, state.smearJump ?? 0.12);

    // Generated anchors, ranked against this camera. Confidence decays with
    // distance from the pose the anchor was CAPTURED at: its pixels are a
    // resampling of the scene from there, so marching it from far away smears
    // a smear — and because the result then looks like data, nothing asks for
    // it to be regenerated. Letting it fade returns those pixels to the hole
    // budget, and a fresh anchor gets made where the camera actually is.
    const picked = this.rankAnchors(cam, MAX_GEN_ANCHORS);
    this._marched = picked;
    // Tight on purpose. A generated anchor is a single RGB-D view: it is
    // excellent at the pose it was captured from and smears as you march away
    // from it, exactly like the photograph does. Measured on a real photo, the
    // useful radius is small — so let it fade fast, which turns the smear back
    // into a HOLE and gets a fresh anchor made where the camera actually is,
    // instead of leaving confident-looking rubber on screen.
    // Widened once disocclusions are clamped to background depth. That fix
    // removes the ramp geometry that made a stale anchor smear, so a generated
    // view stays usable over a much larger baseline and the PER-PIXEL stretch
    // term can do the discriminating — which it does correctly, unlike a blunt
    // pose-distance fade that cannot tell a well-behaved flat background from
    // a rubber sheet.
    const near = state.anchorNear ?? 0.10;
    const far = state.anchorFar ?? 0.34;
    for (let i = 0; i < picked.length; i++) {
      const a = picked[i];
      const r = relativePose(a, cam);
      const t = (poseDistance(a, cam) - near) / Math.max(far - near, 1e-4);
      const fade = 1 - Math.min(Math.max(t, 0), 1);
      this._uM.set(r.m, i * 9);
      this._uC.set(r.c, i * 3);
      this._uK.set(a.K, i * 2);
      this._uRange[i * 2] = a.dMin;
      this._uRange[i * 2 + 1] = a.dMax;
      this._uLayer[i] = a.slot;
      const sm = fade * fade * (3 - 2 * fade);
      this._uWeight[i] = a.weight * sm;
      this._uValid[i] = sm;
    }
    if (picked.length) {
      gl.uniformMatrix3fv(this.uMarch['uGenM[0]'], false, this._uM);
      gl.uniform3fv(this.uMarch['uGenC[0]'], this._uC);
      gl.uniform2fv(this.uMarch['uGenK[0]'], this._uK);
      gl.uniform2fv(this.uMarch['uGenRange[0]'], this._uRange);
      gl.uniform1fv(this.uMarch['uGenLayer[0]'], this._uLayer);
      gl.uniform1fv(this.uMarch['uGenWeight[0]'], this._uWeight);
      gl.uniform1fv(this.uMarch['uGenValid[0]'], this._uValid);
    }
    gl.uniform1i(this.uMarch.uNumGen, picked.length);

    gl.uniform2fv(this.uMarch.uFitScale, state.fitScale);
    gl.uniform2fv(this.uMarch.uFitOff, state.fitOff);
    gl.uniform2fv(this.uMarch.uNovelK, state.novelK);
    gl.uniform1f(this.uMarch.uDSub, state.dSub);
    gl.uniform1f(this.uMarch.uDFloor, state.dFloor ?? 0.04);
    gl.uniform1i(this.uMarch.uSteps, state.steps ?? 48);
    gl.uniform1i(this.uMarch.uStepsGen, state.stepsGen ?? 32);
    gl.uniform1i(this.uMarch.uTrustBase, state.trustBase ? 1 : 0);
    gl.uniform1f(this.uMarch.uDepthEncode, encode ? 1 / DEPTH_RANGE_8BIT : 0);
  }

  /** contain-fit of the photo frame inside a viewport of w x h. */
  fitFor(w, h) {
    const L = this.layers;
    const imgAspect = L.w / L.h;
    const vpAspect = w / h;
    let fsx = 1, fsy = 1;
    if (vpAspect > imgAspect) fsx = vpAspect / imgAspect;
    else fsy = imgAspect / vpAspect;
    return { scale: [fsx, fsy], off: [(1 - fsx) / 2, (1 - fsy) / 2] };
  }

  _renderInto(state, targetFbo, w, h) {
    const gl = this.gl;
    this._ensureFbo(w, h);
    const fit = this.fitFor(w, h);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    this._bindMarch({
      ...state,
      fitScale: fit.scale, fitOff: fit.off,
      novelK: state.cam.K,
    }, this.depthEncoded);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // ---- pass 2: composite + DoF ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFbo);
    if (targetFbo) gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, w, h);
    gl.useProgram(this.progComp);
    gl.bindVertexArray(this.vaoEmpty);
    gl.activeTexture(gl.TEXTURE0 + 4);
    gl.bindTexture(gl.TEXTURE_2D, this.fboColor);
    gl.uniform1i(this.uComp.uTexColor, 4);
    gl.activeTexture(gl.TEXTURE0 + 5);
    gl.bindTexture(gl.TEXTURE_2D, this.fboAux);
    gl.uniform1i(this.uComp.uTexAux, 5);
    gl.uniform2f(this.uComp.uViewport, w, h);
    gl.uniform3fv(this.uComp.uBgTop, state.bgTop ?? [0.06, 0.07, 0.09]);
    gl.uniform3fv(this.uComp.uBgBottom, state.bgBottom ?? [0.02, 0.02, 0.03]);
    gl.uniform1f(this.uComp.uFocusDist, state.focusDist ?? 1);
    gl.uniform1f(this.uComp.uDofStrength, state.dofStrength ?? 0);
    gl.uniform1f(this.uComp.uMaxCoC, state.maxCoC ?? 22);
    gl.uniform1f(this.uComp.uDepthDecode, this.depthEncoded ? DEPTH_RANGE_8BIT : 0);
    gl.uniform3fv(this.uComp.uGrainSigma, state.grainSigma || [0, 0, 0]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  render(state) {
    if (this.contextLost || !this.layers) return;
    this._renderInto(state, null, this.canvas.width, this.canvas.height);
  }

  /** Offscreen composite render at up to `scale`x canvas size -> {pixels,width,height}. */
  capture(state, scale = 2) {
    if (this.contextLost || !this.layers) return null;
    const gl = this.gl;
    const maxDim = 4096;
    let w = Math.round(this.canvas.width * scale);
    let h = Math.round(this.canvas.height * scale);
    const over = Math.max(w, h) / maxDim;
    if (over > 1) { w = Math.round(w / over); h = Math.round(h / over); }

    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

    const keepW = this.fboW, keepH = this.fboH;
    const eff = w / this.canvas.width;
    const st = ((state.dofStrength ?? 0) > 0 && eff !== 1)
      ? { ...state, dofStrength: state.dofStrength * eff, maxCoC: (state.maxCoC ?? 22) * eff }
      : state;
    this._renderInto(st, fbo, w, h);

    const pixels = new Uint8ClampedArray(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(pixels.buffer));
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(tex);

    if (keepW && (keepW !== this.fboW || keepH !== this.fboH)) this._ensureFbo(keepW, keepH);
    flipRows(pixels, w, h);
    return { pixels, width: w, height: h };
  }

  _ensureCapFbo(w, h) {
    const gl = this.gl;
    if (this.capFbo && this.capW === w && this.capH === h) return;
    if (this.capFbo) {
      gl.deleteFramebuffer(this.capFbo);
      gl.deleteTexture(this.capColor);
      gl.deleteTexture(this.capAux);
    }
    this.capW = w; this.capH = h;
    const mk = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
      return t;
    };
    this.capColor = mk();
    this.capAux = mk();
    this.capFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.capFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.capColor, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.capAux, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error('anchor capture FBO incomplete');
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Render a novel view AS AN ANCHOR FRAME: no contain-fit, no DoF, raw march
   * output plus the per-pixel reference depth and confidence, read back for
   * the generative completion pass.
   * Returns {rgba, depth (novel-frame), conf, baseShare, w, h}, all Float32
   * and in image order (row 0 = top). baseShare is how much of each pixel is
   * still the original photograph rather than earlier generated anchors.
   */
  captureAnchorFrame(state, w, h) {
    if (this.contextLost || !this.layers) return null;
    const gl = this.gl;
    this._ensureCapFbo(w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.capFbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    this._bindMarch({
      ...state,
      fitScale: [1, 1], fitOff: [0, 0],
      novelK: state.cam.K,
      trustBase: false,
    }, true); // always packed: the capture target is RGBA8
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const rgba = new Uint8ClampedArray(w * h * 4);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(rgba.buffer));
    const aux = new Uint8Array(w * h * 4);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, aux);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);

    flipRows(rgba, w, h);
    flipRows(aux, w, h);
    const depth = new Float32Array(w * h);
    const conf = new Float32Array(w * h);
    const baseShare = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      const hi = aux[i * 4], lo = aux[i * 4 + 1];
      depth[i] = ((hi * 256 + lo) / 65535) * DEPTH_RANGE_8BIT;
      conf[i] = aux[i * 4 + 2] / 255;
      baseShare[i] = aux[i * 4 + 3] / 255;
    }
    return { rgba, depth, conf, baseShare, w, h };
  }

  /**
   * Ray probe: march ONE pixel of the current view and read back what it hit.
   * Used for tap-to-pivot / tap-to-focus. Renders a 1x1 anchor-style frame
   * whose whole extent is the requested image uv, so it costs one fragment and
   * needs no float readback path.
   * Returns {depth, conf} in novel-frame units, or null.
   */
  probeAt(state, u, v) {
    if (this.contextLost || !this.layers) return null;
    const gl = this.gl;
    if (!this._probeFbo) {
      const mk = () => {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, 1, 1);
        return t;
      };
      this._probeColor = mk();
      this._probeAux = mk();
      this._probeFbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this._probeFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._probeColor, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this._probeAux, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        this._probeFbo = null;
        return null;
      }
    }
    const eps = 1e-4;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._probeFbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, 1, 1);
    gl.disable(gl.BLEND);
    // vUv is (0.5,0.5) at the only fragment; the shader flips v, so pre-flip it
    this._bindMarch({
      ...state,
      fitScale: [eps, eps],
      fitOff: [u - 0.5 * eps, v - 0.5 * eps],
      novelK: state.cam.K,
      trustBase: false,
    }, true);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const px = new Uint8Array(4);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
    if (px[3] === 0) return null;
    return {
      depth: ((px[0] * 256 + px[1]) / 65535) * DEPTH_RANGE_8BIT,
      conf: px[2] / 255,
    };
  }

  dispose() {
    const gl = this.gl;
    for (const t of [this.texColor0, this.texDisp0, this.texColor1, this.texDisp1,
      this.fboColor, this.fboAux, this.capColor, this.capAux,
      this.texGenColor, this.texGenDisp]) {
      if (t) gl.deleteTexture(t);
    }
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
    if (this.capFbo) gl.deleteFramebuffer(this.capFbo);
    if (this._probeFbo) gl.deleteFramebuffer(this._probeFbo);
  }
}

/** GL reads bottom-up; flip to image order. */
function flipRows(buf, w, h) {
  const row = w * 4;
  const tmp = new (buf.constructor)(row);
  for (let y = 0; y < (h >> 1); y++) {
    const a = y * row, b = (h - 1 - y) * row;
    tmp.set(buf.subarray(a, a + row));
    buf.copyWithin(a, b, b + row);
    buf.set(tmp, b);
  }
}
