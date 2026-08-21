// WebGL2 gaussian splat renderer.
// Pass 1: instanced splat quads -> offscreen MRT (premult color + composited depth).
// Pass 2: composite over background with optional depth-of-field gather.

import { SPLAT_VS, SPLAT_FS, COMPOSITE_VS, COMPOSITE_FS, TEX_WIDTH } from './shaders.js';

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

export class SplatRenderer {
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

    // iOS Safari sheds GL contexts aggressively (backgrounding, memory
    // pressure). preventDefault is REQUIRED or the context never restores.
    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault();
      this.contextLost = true;
      if (this.onContextLost) this.onContextLost();
    });
    canvas.addEventListener('webglcontextrestored', () => {
      this.contextLost = false;
      this._initGL();
      if (this.onContextRestored) this.onContextRestored(); // re-upload cloud
    });

    this._initGL();
  }

  _initGL() {
    const gl = this.gl;
    // Float render targets: full float > half float > rgba8 (encoded depth).
    // Extensions must be re-queried after a context restore.
    if (gl.getExtension('EXT_color_buffer_float')) {
      this.rtFormat = gl.RGBA16F; this.rtType = gl.HALF_FLOAT; this.depthEncoded = false;
    } else if (gl.getExtension('EXT_color_buffer_half_float')) {
      this.rtFormat = gl.RGBA16F; this.rtType = gl.HALF_FLOAT; this.depthEncoded = false;
    } else {
      this.rtFormat = gl.RGBA8; this.rtType = gl.UNSIGNED_BYTE; this.depthEncoded = true;
    }

    this.progSplat = link(gl, SPLAT_VS, SPLAT_FS);
    this.uSplat = uniforms(gl, this.progSplat, [
      'uTexPos', 'uTexCovA', 'uTexCovB', 'uTexColor', 'uView', 'uProj',
      'uFocal', 'uViewport', 'uSplatScale', 'uDepthEncode',
    ]);
    this.progComp = link(gl, COMPOSITE_VS, COMPOSITE_FS);
    this.uComp = uniforms(gl, this.progComp, [
      'uTexColor', 'uTexDepth', 'uViewport', 'uBgTop', 'uBgBottom',
      'uFocusDist', 'uDofStrength', 'uMaxCoC', 'uDepthDecode',
    ]);

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.indexBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribIPointer(0, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(0, 1);
    gl.bindVertexArray(null);

    this.vaoEmpty = gl.createVertexArray(); // composite pass: no attribs

    // after a context loss every old GL object is invalid — drop references
    this.texPos = null; this.texCovA = null; this.texCovB = null; this.texColor = null;
    this.count = 0;
    this.instanceCount = 0;
    this.fbo = null; this.fboColor = null; this.fboDepthT = null;
    this.fboW = 0; this.fboH = 0;
  }

  _makeDataTexture(internal, format, type, w, h, data) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texStorage2D(gl.TEXTURE_2D, 1, internal, w, h);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, format, type, data);
    return tex;
  }

  /** cloud: {count, positions f32x3, cov f32x6, colors u8x4} */
  setCloud(cloud) {
    const gl = this.gl;
    // a 'built' can land while the context is lost (widened by the two-phase
    // build) — no-op is correct: onContextRestored re-uploads app.cloud
    if (this.contextLost || gl.isContextLost()) return;
    for (const t of [this.texPos, this.texCovA, this.texCovB, this.texColor]) {
      if (t) gl.deleteTexture(t);
    }
    const n = cloud.count;
    this.count = n;
    const texH = Math.max(1, Math.ceil(n / TEX_WIDTH));
    if (texH > gl.getParameter(gl.MAX_TEXTURE_SIZE)) throw new Error('too many splats');
    const texels = TEX_WIDTH * texH;

    const pos = new Float32Array(texels * 4);
    const covA = new Float32Array(texels * 4);
    const covB = new Float32Array(texels * 2);
    const col = new Uint8Array(texels * 4);
    for (let i = 0; i < n; i++) {
      pos[i * 4] = cloud.positions[i * 3];
      pos[i * 4 + 1] = cloud.positions[i * 3 + 1];
      pos[i * 4 + 2] = cloud.positions[i * 3 + 2];
      covA[i * 4] = cloud.cov[i * 6];
      covA[i * 4 + 1] = cloud.cov[i * 6 + 1];
      covA[i * 4 + 2] = cloud.cov[i * 6 + 2];
      covA[i * 4 + 3] = cloud.cov[i * 6 + 3];
      covB[i * 2] = cloud.cov[i * 6 + 4];
      covB[i * 2 + 1] = cloud.cov[i * 6 + 5];
    }
    col.set(cloud.colors.subarray(0, n * 4));

    this.texPos = this._makeDataTexture(gl.RGBA32F, gl.RGBA, gl.FLOAT, TEX_WIDTH, texH, pos);
    this.texCovA = this._makeDataTexture(gl.RGBA32F, gl.RGBA, gl.FLOAT, TEX_WIDTH, texH, covA);
    this.texCovB = this._makeDataTexture(gl.RG32F, gl.RG, gl.FLOAT, TEX_WIDTH, texH, covB);
    this.texColor = this._makeDataTexture(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, TEX_WIDTH, texH, col);

    // identity order until first sort arrives
    const idx = new Uint32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    this.setSortedIndices(idx);
  }

  setSortedIndices(indices) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuf);
    gl.bufferData(gl.ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);
    this.instanceCount = indices.length;
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
      // last-resort fallback to encoded RGBA8 targets
      if (this.rtFormat !== gl.RGBA8) {
        this.rtFormat = gl.RGBA8; this.rtType = gl.UNSIGNED_BYTE; this.depthEncoded = true;
        this.fboW = 0; // force realloc
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
   * state: {view, proj, bgTop, bgBottom, focusDist, dofStrength, maxCoC, splatScale}
   * Renders to targetFbo (null = canvas) at (w, h).
   */
  _renderInto(state, targetFbo, w, h) {
    const gl = this.gl;
    this._ensureFbo(w, h);

    // ---- pass 1: splats -> MRT ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    if (this.count > 0) {
      gl.useProgram(this.progSplat);
      gl.bindVertexArray(this.vao);
      const texBind = (unit, tex, loc) => {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.uniform1i(loc, unit);
      };
      texBind(0, this.texPos, this.uSplat.uTexPos);
      texBind(1, this.texCovA, this.uSplat.uTexCovA);
      texBind(2, this.texCovB, this.uSplat.uTexCovB);
      texBind(3, this.texColor, this.uSplat.uTexColor);
      gl.uniformMatrix4fv(this.uSplat.uView, false, state.view);
      gl.uniformMatrix4fv(this.uSplat.uProj, false, state.proj);
      const fx = state.proj[0] * w / 2;
      const fy = state.proj[5] * h / 2;
      gl.uniform2f(this.uSplat.uFocal, fx, fy);
      gl.uniform2f(this.uSplat.uViewport, w, h);
      gl.uniform1f(this.uSplat.uSplatScale, state.splatScale ?? 1);
      gl.uniform1f(this.uSplat.uDepthEncode, this.depthEncoded ? 1 / DEPTH_RANGE_8BIT : 0);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, Math.min(this.instanceCount, this.count));
      gl.bindVertexArray(null);
    }

    // ---- pass 2: composite + DoF ----
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFbo);
    if (targetFbo) gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.BLEND);
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
    gl.uniform1f(this.uComp.uFocusDist, state.focusDist ?? 2);
    gl.uniform1f(this.uComp.uDofStrength, state.dofStrength ?? 0);
    gl.uniform1f(this.uComp.uMaxCoC, state.maxCoC ?? 22);
    gl.uniform1f(this.uComp.uDepthDecode, this.depthEncoded ? DEPTH_RANGE_8BIT : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  render(state) {
    if (this.contextLost) return;
    this._renderInto(state, null, this.canvas.width, this.canvas.height);
  }

  /** Renders offscreen at up to `scale`x canvas size; returns {pixels, width, height} (top-down rows), or null if the GL context is lost. */
  capture(state, scale = 2) {
    if (this.contextLost) return null;
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
    // CoC is computed in render-target pixels: scale DoF to capture resolution
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

    // restore FBO size for interactive rendering
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
    for (const t of [this.texPos, this.texCovA, this.texCovB, this.texColor, this.fboColor, this.fboDepthT]) {
      if (t) gl.deleteTexture(t);
    }
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
  }
}
