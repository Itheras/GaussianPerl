// WebGL2 layered-heightfield renderer (M8).
// Pass 1: fullscreen two-layer inverse-depth raymarch -> offscreen MRT
//         (premult color + composited depth). No geometry, no sorting.
// Pass 2: composite over background with optional depth-of-field gather.

import { RAYMARCH_VS, RAYMARCH_FS, COMPOSITE_VS, COMPOSITE_FS } from './shaders.js';
import { toHalfFloat } from '../util/imageops.js';

const DEPTH_RANGE_8BIT = 64.0; // world units encodable in RGBA8 fallback mode

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
      if (this.onContextRestored) this.onContextRestored(); // re-upload layers
    });

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

    this.progMarch = link(gl, RAYMARCH_VS, RAYMARCH_FS);
    this.uMarch = uniforms(gl, this.progMarch, [
      'uColor0', 'uDisp0', 'uColor1', 'uDisp1',
      'uCropScale', 'uCropOff', 'uFitScale', 'uFitOff',
      'uKxy', 'uKz', 'uDConv', 'uDMin', 'uDMax', 'uDSub', 'uDFloor',
      'uSteps', 'uDepthEncode',
    ]);
    this.progComp = link(gl, COMPOSITE_VS, COMPOSITE_FS);
    this.uComp = uniforms(gl, this.progComp, [
      'uTexColor', 'uTexDepth', 'uViewport', 'uBgTop', 'uBgBottom',
      'uFocusDist', 'uDofStrength', 'uMaxCoC', 'uDepthDecode',
    ]);

    this.vaoEmpty = gl.createVertexArray();

    // after a context loss every old GL object is invalid — drop references
    this.texColor0 = null; this.texDisp0 = null;
    this.texColor1 = null; this.texDisp1 = null;
    this.layers = null;
    this.fbo = null; this.fboColor = null; this.fboDepthT = null;
    this.fboW = 0; this.fboH = 0;
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

  _ensureFbo(w, h) {
    const gl = this.gl;
    if (this.fbo && this.fboW === w && this.fboH === h) return;
    if (this.fbo) {
      gl.deleteFramebuffer(this.fbo);
      gl.deleteTexture(this.fboColor);
      gl.deleteTexture(this.fboDepthT);
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
    this.fboDepthT = mk();
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboColor, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.fboDepthT, 0);
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
   * state: {eye:[ex,ey,ez], dConv, dSub, dMin, dMax, dFloor, fPx, steps,
   *         bgTop, bgBottom, focusDist, dofStrength, maxCoC}
   */
  _renderInto(state, targetFbo, w, h) {
    const gl = this.gl;
    const L = this.layers;
    this._ensureFbo(w, h);

    // contain-fit the image window in the viewport (bars show the bg gradient)
    const imgAspect = L.w / L.h;
    const vpAspect = w / h;
    let fsx = 1, fsy = 1;
    if (vpAspect > imgAspect) fsx = vpAspect / imgAspect;
    else fsy = imgAspect / vpAspect;
    const fox = (1 - fsx) / 2, foy = (1 - fsy) / 2;

    // interior image uv -> padded texture uv
    const csx = L.w / L.pw, csy = L.h / L.ph;
    const cox = L.padPx / L.pw, coy = L.padPx / L.ph;

    const [ex, ey, ez] = state.eye;
    const dSub = state.dSub;

    // ---- pass 1: raymarch -> MRT ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    gl.useProgram(this.progMarch);
    gl.bindVertexArray(this.vaoEmpty);
    const bind = (unit, tex, loc) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
    };
    bind(0, this.texColor0, this.uMarch.uColor0);
    bind(1, this.texDisp0, this.uMarch.uDisp0);
    bind(2, this.texColor1, this.uMarch.uColor1);
    bind(3, this.texDisp1, this.uMarch.uDisp1);
    gl.uniform2f(this.uMarch.uCropScale, csx, csy);
    gl.uniform2f(this.uMarch.uCropOff, cox, coy);
    gl.uniform2f(this.uMarch.uFitScale, fsx, fsy);
    gl.uniform2f(this.uMarch.uFitOff, fox, foy);
    // image v grows DOWN while world y grows up: flip the y term
    gl.uniform2f(this.uMarch.uKxy,
      state.fPx * ex / (L.w * dSub),
      -state.fPx * ey / (L.h * dSub));
    gl.uniform1f(this.uMarch.uKz, -ez / dSub);
    gl.uniform1f(this.uMarch.uDConv, state.dConv);
    gl.uniform1f(this.uMarch.uDMin, state.dMin);
    gl.uniform1f(this.uMarch.uDMax, state.dMax);
    gl.uniform1f(this.uMarch.uDSub, dSub);
    gl.uniform1f(this.uMarch.uDFloor, state.dFloor ?? 0.04);
    gl.uniform1i(this.uMarch.uSteps, state.steps ?? 40);
    gl.uniform1f(this.uMarch.uDepthEncode, this.depthEncoded ? 1 / DEPTH_RANGE_8BIT : 0);
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
    gl.bindTexture(gl.TEXTURE_2D, this.fboDepthT);
    gl.uniform1i(this.uComp.uTexDepth, 5);
    gl.uniform2f(this.uComp.uViewport, w, h);
    gl.uniform3fv(this.uComp.uBgTop, state.bgTop ?? [0.06, 0.07, 0.09]);
    gl.uniform3fv(this.uComp.uBgBottom, state.bgBottom ?? [0.02, 0.02, 0.03]);
    gl.uniform1f(this.uComp.uFocusDist, state.focusDist ?? 1);
    gl.uniform1f(this.uComp.uDofStrength, state.dofStrength ?? 0);
    gl.uniform1f(this.uComp.uMaxCoC, state.maxCoC ?? 22);
    gl.uniform1f(this.uComp.uDepthDecode, this.depthEncoded ? DEPTH_RANGE_8BIT : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  render(state) {
    if (this.contextLost || !this.layers) return;
    this._renderInto(state, null, this.canvas.width, this.canvas.height);
  }

  /** Offscreen render at up to `scale`x canvas size -> {pixels,width,height} (top-down), or null. */
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
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(pixels.buffer));
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(tex);

    if (keepW && (keepW !== this.fboW || keepH !== this.fboH)) this._ensureFbo(keepW, keepH);

    // flip vertically (GL reads bottom-up)
    const row = w * 4;
    const tmp = new Uint8ClampedArray(row);
    for (let y = 0; y < (h >> 1); y++) {
      const a = y * row, b = (h - 1 - y) * row;
      tmp.set(pixels.subarray(a, a + row));
      pixels.copyWithin(a, b, b + row);
      pixels.set(tmp, b);
    }
    return { pixels, width: w, height: h };
  }

  dispose() {
    const gl = this.gl;
    for (const t of [this.texColor0, this.texDisp0, this.texColor1, this.texDisp1,
      this.fboColor, this.fboDepthT]) {
      if (t) gl.deleteTexture(t);
    }
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
  }
}
