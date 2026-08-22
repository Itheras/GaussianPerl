// Window-anchored TRANSLATION-ONLY camera (M8). No rotation, ever: rotation
// converts focal-length/scale error into shear — the "faces swim" mechanism.
// Translation-only novel views are pixel-exact images of the true scene under
// ANY focal/scale error (it only rescales the baseline), and production
// single-photo 3D (Facebook, Google Cinematic, Immersity) all ship exactly
// this envelope. Eye offsets are in SUBJECT units (Z_subject = 1):
//   ex, ey — lateral, clamped to the per-photo envelope (holes + parallax caps)
//   ez     — dolly, positive = toward the scene (opens almost no holes)
// Pointer events: 1-finger/mouse drag = lateral; wheel / pinch / two-finger =
// dolly; double-tap = re-pivot (onPick). Idle "wiggle" is a slow translation
// ellipse advertising parallax until first interaction.

export class WindowCam {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.onChange = opts.onChange || (() => {});
    this.onPick = opts.onPick || null;

    this.ex = 0; this.ey = 0; this.ez = 0;
    this.velX = 0; this.velY = 0;

    // envelope (subject units); setEnvelope() tightens per photo
    this.exyMax = 0.05;
    this.ezIn = 0.12;   // dolly toward the scene
    this.ezOut = 0.05;  // dolly away

    this.wiggle = true;
    this._wigglePhase = 0;
    this._idleTime = 10;
    this._everInteracted = false;

    this._pointers = new Map();
    this._pinchDist = 0;
    this._lastTap = { t: -1e9, x: 0, y: 0 };
    this._gestureScale = 1;

    this._bind();
  }

  /** per-photo clamp from fill coverage + parallax budget (subject units) */
  setEnvelope(exyMax, ezIn = this.ezIn, ezOut = this.ezOut) {
    this.exyMax = exyMax;
    this.ezIn = ezIn;
    this.ezOut = ezOut;
    this._clamp();
  }

  reset() {
    this.ex = 0; this.ey = 0; this.ez = 0;
    this.velX = 0; this.velY = 0;
    this.onChange();
  }

  home() {
    this.reset();
    this._idleTime = this._everInteracted ? 0 : 10;
  }

  eye() { return [this.ex, this.ey, this.ez]; }

  _soft(v, lim) {
    if (v > lim) return lim + (v - lim) * 0.35;
    if (v < -lim) return -lim + (v + lim) * 0.35;
    return v;
  }

  _clamp() {
    this.ex = this._soft(this.ex, this.exyMax);
    this.ey = this._soft(this.ey, this.exyMax);
    this.ez = Math.min(Math.max(this.ez, -this.ezOut), this.ezIn);
  }

  /** advance inertia + wiggle; returns true if the camera moved */
  update(dt) {
    dt = Math.min(dt, 0.05);
    let moved = false;
    this._idleTime += dt;

    if (this._pointers.size === 0) {
      if (Math.abs(this.velX) > 1e-5 || Math.abs(this.velY) > 1e-5) {
        this.ex += this.velX * dt;
        this.ey += this.velY * dt;
        const damp = Math.exp(-4.5 * dt);
        this.velX *= damp; this.velY *= damp;
        this._clamp();
        moved = true;
      }
      // spring back inside the soft envelope
      const ox = this.ex - Math.min(Math.max(this.ex, -this.exyMax), this.exyMax);
      const oy = this.ey - Math.min(Math.max(this.ey, -this.exyMax), this.exyMax);
      if (Math.abs(ox) > 1e-6 || Math.abs(oy) > 1e-6) {
        this.ex -= ox * Math.min(10 * dt, 1);
        this.ey -= oy * Math.min(10 * dt, 1);
        moved = true;
      }
    }

    if (this.wiggle && !this._everInteracted && this._idleTime > 1.2) {
      this._wigglePhase += dt;
      const t = this._wigglePhase;
      const amp = Math.min((this._idleTime - 1.2) / 2, 1) * this.exyMax * 0.7;
      this.ex = Math.sin(t * 0.7) * amp;
      this.ey = Math.cos(t * 0.7) * amp * 0.55;
      moved = true;
    }
    return moved;
  }

  _interact() {
    this._everInteracted = true;
    this._idleTime = 0;
  }

  _applyDrag(dx, dy) {
    // full half-canvas drag sweeps the whole envelope; content follows the finger
    const k = (2.2 * this.exyMax) / Math.max(this.canvas.clientHeight, 1);
    this.ex = this._soft(this.ex - dx * k, this.exyMax);
    this.ey = this._soft(this.ey + dy * k, this.exyMax);
    this.velX = -dx * k * 60 * 0.35;
    this.velY = dy * k * 60 * 0.35;
  }

  _applyDolly(factor) {
    // factor > 1 = dolly in
    this.ez = Math.min(Math.max(this.ez + Math.log(factor) * 0.25, -this.ezOut), this.ezIn);
  }

  _bind() {
    const c = this.canvas;
    c.style.touchAction = 'none';

    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      this._pointers.set(e.pointerId, {
        x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY,
      });
      this._interact();
      this.velX = 0; this.velY = 0;
      if (this._pointers.size === 2) {
        const [a, b] = [...this._pointers.values()];
        this._pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
        this._lastTap.t = -1e9; // multi-touch is never a tap
      }
      if (this._pointers.size === 1) {
        const now = performance.now();
        const lt = this._lastTap;
        if (now - lt.t < 320 && Math.hypot(e.clientX - lt.x, e.clientY - lt.y) < 30) {
          if (this.onPick) this.onPick(e.clientX, e.clientY);
          this._lastTap.t = -1e9;
        } else {
          this._lastTap = { t: now, x: e.clientX, y: e.clientY };
        }
      }
      e.preventDefault();
    });

    c.addEventListener('pointermove', (e) => {
      const p = this._pointers.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x, dy = e.clientY - p.y;
      if (Math.hypot(e.clientX - p.x0, e.clientY - p.y0) > 12) this._lastTap.t = -1e9;

      if (this._pointers.size === 1) {
        this._applyDrag(dx, dy);
      } else if (this._pointers.size === 2) {
        p.x = e.clientX; p.y = e.clientY;
        const [a, b] = [...this._pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (this._pinchDist > 0 && d > 0) this._applyDolly(d / this._pinchDist);
        this._pinchDist = d;
        this.onChange();
        return;
      }
      p.x = e.clientX; p.y = e.clientY;
      this.onChange();
    });

    const release = (e) => {
      this._pointers.delete(e.pointerId);
      this._pinchDist = 0;
    };
    c.addEventListener('pointerup', release);
    c.addEventListener('pointercancel', release);

    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this._interact();
      const k = e.ctrlKey ? 0.01 : (e.deltaMode === 1 ? 0.05 : 0.0015);
      this._applyDolly(Math.exp(-e.deltaY * k));
      this.onChange();
    }, { passive: false });

    c.addEventListener('contextmenu', (e) => e.preventDefault());

    // Safari trackpad/iPadOS pinch arrives as proprietary GestureEvents
    c.addEventListener('gesturestart', (e) => {
      e.preventDefault();
      this._interact();
      this._gestureScale = e.scale || 1;
    });
    c.addEventListener('gesturechange', (e) => {
      e.preventDefault();
      if (this._pointers.size >= 2) return;
      if (e.scale > 0 && this._gestureScale > 0) {
        this._applyDolly(e.scale / this._gestureScale);
        this.onChange();
      }
      this._gestureScale = e.scale || this._gestureScale;
    });
    c.addEventListener('gestureend', (e) => {
      e.preventDefault();
      this._gestureScale = 1;
    });
    for (const ev of ['gesturestart', 'gesturechange', 'gestureend']) {
      document.addEventListener(ev, (e) => e.preventDefault());
    }
    c.addEventListener('dblclick', (e) => e.preventDefault());
  }
}
