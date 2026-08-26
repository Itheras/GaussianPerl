// 6-DoF free camera (M9). Replaces M8's translation-only window camera, whose
// ~3%-of-frame envelope was so tight the scene read as flat.
//
// State is orbit-centric because it is the only parameterisation that stays
// well-conditioned for every gesture:
//   pivot  the point being looked at (world)
//   yaw/pitch  orientation (no roll — roll makes a photo read as broken)
//   dist   distance from pivot along the view axis
//   => C = pivot + dist * back(yaw, pitch),  looking at the pivot.
// Home is exactly the photo's own camera: pivot (0,0,-1), yaw=pitch=0, dist=1,
// so C = origin — the base anchor's pose, and the render is the photograph.
//
// Gestures
//   drag / 1 finger     orbit around the pivot   (parallax: the 3D sell)
//   shift-drag, right-drag, 2 fingers   pan (pivot + camera translate)
//   wheel / pinch       dolly (distance to pivot)
//   W A S D / arrows    fly (camera-relative translation, pivot follows)
//   Q E / R F           fly down / up
//   Shift               3x fly speed
//   double tap/click    re-pivot on the surface under the pointer (onPick)
//   look mode           drag rotates in place (pivot follows the view axis)
//
// Limits are HARD but generous, and there is no spring-back: being dragged
// backwards is what "I cannot move it" feels like. `freeRoam` removes them.

import { camRotation, cameraBasis } from '../render/pose.js';

const TAU = Math.PI * 2;

export class FreeCam {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.onChange = opts.onChange || (() => {});
    this.onPick = opts.onPick || null;

    // pose
    this.pivot = [0, 0, -1];
    this.yaw = 0;
    this.pitch = 0;
    this.dist = 1;
    this.K = [1, 1]; // (f/W, f/H) of the photo; set by the app

    // Limits (subject units / radians). Generous — an order of magnitude more
    // motion than M8's 3%-of-frame envelope — but not unbounded: past roughly
    // 30 degrees a single photo is more invention than photograph. `freeRoam`
    // lifts them for anyone who wants to go there anyway.
    // Measured, not guessed: on a real photo the completion loop holds up
    // beautifully to ~6 degrees of orbit, is still convincing near 11, and
    // past that a 28 MB GAN is inventing more than it is recalling. These are
    // roughly 7x M8's effective envelope; `freeRoam` lifts them entirely for
    // anyone who would rather see the failure than be stopped.
    this.limits = {
      yaw: 0.34,        // ~19.5 deg
      pitch: 0.22,      // ~12.6 deg
      distMin: 0.32,
      distMax: 2.8,
      pan: 1.0,         // pivot displacement from home
    };
    this.freeRoam = false;
    this.lookMode = false;

    // feel
    this.orbitSpeed = 1.1;   // radians per full canvas-height drag (~63 deg)
    this.flySpeed = 0.9;     // subject units per second at dist = 1
    this.maxSpin = 1.6;      // rad/s cap on flick inertia
    this.velYaw = 0; this.velPitch = 0;

    this.wiggle = true;
    this._wigglePhase = 0;
    this._idleTime = 10;
    this._everInteracted = false;
    this._wiggleBase = null;

    this._keys = new Set();
    this._pointers = new Map();
    this._pinchDist = 0;
    this._lastTap = { t: -1e9, x: 0, y: 0 };
    this._gestureScale = 1;
    this._home = this.snapshot();

    this._bind();
  }

  // ---------------------------------------------------------------- pose

  basis() { return cameraBasis(this.yaw, this.pitch); }

  /** Camera centre in world. */
  center() {
    const { back } = this.basis();
    return [
      this.pivot[0] + back[0] * this.dist,
      this.pivot[1] + back[1] * this.dist,
      this.pivot[2] + back[2] * this.dist,
    ];
  }

  /** {R, C, K} — everything the renderer needs. */
  pose() {
    return { R: camRotation(this.yaw, this.pitch), C: this.center(), K: this.K };
  }

  snapshot() {
    return { pivot: this.pivot.slice(), yaw: this.yaw, pitch: this.pitch, dist: this.dist };
  }

  restore(s) {
    this.pivot = s.pivot.slice();
    this.yaw = s.yaw; this.pitch = s.pitch; this.dist = s.dist;
    this.velYaw = 0; this.velPitch = 0;
    this._clamp();
  }

  /** Set the home pose (called once the photo's subject distance is known). */
  setHome({ pivotZ = -1, dist = 1 } = {}) {
    this._home = { pivot: [0, 0, pivotZ], yaw: 0, pitch: 0, dist };
    this.restore(this._home);
  }

  reset() {
    this.restore(this._home);
    this._wiggleBase = null;
    this.onChange();
  }

  home() {
    this.reset();
    this._idleTime = this._everInteracted ? 0 : 10;
  }

  /** Move the orbit centre to a world point, keeping the camera where it is. */
  setPivot(p) {
    const C = this.center();
    this.pivot = [p[0], p[1], p[2]];
    const dx = C[0] - p[0], dy = C[1] - p[1], dz = C[2] - p[2];
    this.dist = Math.max(Math.hypot(dx, dy, dz), this.limits.distMin);
    // re-aim at the new pivot so it stays centred while orbiting
    const inv = 1 / this.dist;
    const bx = dx * inv, by = dy * inv, bz = dz * inv;
    this.pitch = Math.asin(Math.max(-1, Math.min(1, -by)));
    this.yaw = Math.atan2(-bx, bz);
    this._clamp();
    this.onChange();
  }

  /** How far the camera has strayed from home — drives quality decisions. */
  displacement() {
    const C = this.center();
    return Math.hypot(C[0], C[1], C[2] - 0) + Math.abs(this.yaw) + Math.abs(this.pitch);
  }

  atHome() {
    const h = this._home;
    return Math.abs(this.yaw - h.yaw) < 1e-4 && Math.abs(this.pitch - h.pitch) < 1e-4
      && Math.abs(this.dist - h.dist) < 1e-4
      && Math.abs(this.pivot[0] - h.pivot[0]) < 1e-4
      && Math.abs(this.pivot[1] - h.pivot[1]) < 1e-4
      && Math.abs(this.pivot[2] - h.pivot[2]) < 1e-4;
  }

  _clamp() {
    const L = this.limits;
    this.dist = Math.min(Math.max(this.dist, L.distMin), this.freeRoam ? 60 : L.distMax);
    if (this.freeRoam) {
      // keep angles bounded to a sane range, but let the user go all the way round
      this.yaw = ((this.yaw + Math.PI) % TAU + TAU) % TAU - Math.PI;
      this.pitch = Math.min(Math.max(this.pitch, -1.45), 1.45);
      return;
    }
    this.yaw = Math.min(Math.max(this.yaw, -L.yaw), L.yaw);
    this.pitch = Math.min(Math.max(this.pitch, -L.pitch), L.pitch);
    const h = this._home.pivot;
    for (let i = 0; i < 3; i++) {
      this.pivot[i] = Math.min(Math.max(this.pivot[i], h[i] - L.pan), h[i] + L.pan);
    }
  }

  // ------------------------------------------------------------ gestures

  orbit(dYaw, dPitch) {
    this.yaw += dYaw;
    this.pitch += dPitch;
    this._clamp();
  }

  look(dYaw, dPitch) {
    // rotate in place: keep C, move the pivot onto the new view axis
    const C = this.center();
    this.yaw += dYaw;
    this.pitch += dPitch;
    this._clampAngles();
    const { forward } = this.basis();
    this.pivot = [
      C[0] + forward[0] * this.dist,
      C[1] + forward[1] * this.dist,
      C[2] + forward[2] * this.dist,
    ];
    this._clamp();
  }

  _clampAngles() {
    if (this.freeRoam) {
      this.yaw = ((this.yaw + Math.PI) % TAU + TAU) % TAU - Math.PI;
      this.pitch = Math.min(Math.max(this.pitch, -1.45), 1.45);
    } else {
      this.yaw = Math.min(Math.max(this.yaw, -this.limits.yaw), this.limits.yaw);
      this.pitch = Math.min(Math.max(this.pitch, -this.limits.pitch), this.limits.pitch);
    }
  }

  /** Translate camera + pivot together, in camera-relative units. */
  translate(dRight, dUp, dForward) {
    const { right, up, forward } = this.basis();
    for (let i = 0; i < 3; i++) {
      this.pivot[i] += right[i] * dRight + up[i] * dUp + forward[i] * dForward;
    }
    this._clamp();
  }

  dolly(factor) {
    this.dist /= factor;
    this._clamp();
  }

  /**
   * Back away along the view axis until the surface ahead is `amount` further
   * off. The app calls this when a probe says the camera is about to end up
   * inside the scene — flying through a wall reads as a bug, not as freedom.
   */
  backOff(amount) {
    this.dist = Math.min(this.dist + amount, this.freeRoam ? 60 : this.limits.distMax);
    this.velYaw *= 0.3; this.velPitch *= 0.3;
  }

  // ------------------------------------------------------------- per frame

  /** Advance inertia, keyboard fly and idle wiggle. True if the camera moved. */
  update(dt) {
    dt = Math.min(dt, 0.05);
    let moved = false;
    this._idleTime += dt;

    if (this._pointers.size === 0 &&
        (Math.abs(this.velYaw) > 1e-5 || Math.abs(this.velPitch) > 1e-5)) {
      this.orbit(this.velYaw * dt, this.velPitch * dt);
      const damp = Math.exp(-5.5 * dt);
      this.velYaw *= damp; this.velPitch *= damp;
      moved = true;
    }

    if (this._keys.size) {
      const fast = this._keys.has('shift') ? 3 : 1;
      const sp = this.flySpeed * fast * dt * Math.max(this.dist, 0.35);
      let r = 0, u = 0, f = 0;
      if (this._keys.has('w') || this._keys.has('arrowup')) f += sp;
      if (this._keys.has('s') || this._keys.has('arrowdown')) f -= sp;
      if (this._keys.has('a') || this._keys.has('arrowleft')) r -= sp;
      if (this._keys.has('d') || this._keys.has('arrowright')) r += sp;
      if (this._keys.has('e') || this._keys.has('r')) u += sp;
      if (this._keys.has('q') || this._keys.has('f')) u -= sp;
      if (r || u || f) {
        this._interact();
        this.translate(r, u, f);
        moved = true;
      }
    }

    if (this.wiggle && !this._everInteracted && this._idleTime > 1.2) {
      if (!this._wiggleBase) this._wiggleBase = this.snapshot();
      this._wigglePhase += dt;
      const t = this._wigglePhase;
      const amp = Math.min((this._idleTime - 1.2) / 2, 1) * 0.16;
      const b = this._wiggleBase;
      this.yaw = b.yaw + Math.sin(t * 0.7) * amp;
      this.pitch = b.pitch + Math.cos(t * 0.7) * amp * 0.45;
      this._clamp();
      moved = true;
    }
    return moved;
  }

  _interact() {
    if (!this._everInteracted && this._wiggleBase) {
      this.restore(this._wiggleBase);
      this._wiggleBase = null;
    }
    this._everInteracted = true;
    this._idleTime = 0;
  }

  // ---------------------------------------------------------------- input

  _panScale() {
    // one pointer pixel == one world pixel at the pivot plane
    const h = Math.max(this.canvas.clientHeight, 1);
    return (this.dist / Math.max(this.K[1], 1e-3)) / h;
  }

  _applyOrbitDrag(dx, dy, dt) {
    const k = this.orbitSpeed / Math.max(this.canvas.clientHeight, 1);
    const dYaw = dx * k, dPitch = -dy * k;
    if (this.lookMode) this.look(dYaw, dPitch);
    else this.orbit(dYaw, dPitch);
    // Flick inertia from ANGULAR VELOCITY, never from the raw event delta: a
    // coarse pointer stream (or a synthetic one) delivers the whole gesture in
    // a single event, and delta*60 then launches the camera into the clamps.
    if (dt > 1e-4) {
      const cap = this.maxSpin;
      const sy = dYaw / dt * 0.28, sp = dPitch / dt * 0.28;
      this.velYaw = Math.min(Math.max(sy, -cap), cap);
      this.velPitch = Math.min(Math.max(sp, -cap), cap);
    }
  }

  _applyPanDrag(dx, dy) {
    const s = this._panScale();
    this.translate(-dx * s, dy * s, 0);
    this.velYaw = 0; this.velPitch = 0;
  }

  _bind() {
    const c = this.canvas;
    c.style.touchAction = 'none';

    const isPan = (e) => e.shiftKey || e.button === 2 || e.buttons === 2 ||
      this._keys.has('shift') || this._panToggle;

    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      this._pointers.set(e.pointerId, {
        x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY,
        pan: isPan(e), t: performance.now(),
      });
      this._interact();
      this.velYaw = 0; this.velPitch = 0;
      if (this._pointers.size === 2) {
        const [a, b] = [...this._pointers.values()];
        this._pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
        this._lastTap.t = -1e9;
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

      const now = performance.now();
      const dt = (now - p.t) / 1000;
      p.t = now;
      if (this._pointers.size === 1) {
        if (p.pan) this._applyPanDrag(dx, dy);
        else this._applyOrbitDrag(dx, dy, dt);
      } else if (this._pointers.size === 2) {
        p.x = e.clientX; p.y = e.clientY;
        const [a, b] = [...this._pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        // two fingers: pinch dollies, drag pans
        const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
        if (this._twoCentre) {
          this._applyPanDrag(cx - this._twoCentre[0], cy - this._twoCentre[1]);
        }
        this._twoCentre = [cx, cy];
        if (this._pinchDist > 0 && d > 0) this.dolly(d / this._pinchDist);
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
      if (this._pointers.size < 2) this._twoCentre = null;
    };
    c.addEventListener('pointerup', release);
    c.addEventListener('pointercancel', release);

    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this._interact();
      const k = e.ctrlKey ? 0.01 : (e.deltaMode === 1 ? 0.05 : 0.0015);
      this.dolly(Math.exp(-e.deltaY * k));
      this.onChange();
    }, { passive: false });

    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('dblclick', (e) => e.preventDefault());

    // Safari trackpad / iPadOS pinch arrives as proprietary GestureEvents
    c.addEventListener('gesturestart', (e) => {
      e.preventDefault();
      this._interact();
      this._gestureScale = e.scale || 1;
    });
    c.addEventListener('gesturechange', (e) => {
      e.preventDefault();
      if (this._pointers.size >= 2) return;
      if (e.scale > 0 && this._gestureScale > 0) {
        this.dolly(e.scale / this._gestureScale);
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

    const typing = (e) => {
      const t = e.target;
      return t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA');
    };
    window.addEventListener('keydown', (e) => {
      if (typing(e)) return;
      const k = e.key.toLowerCase();
      if (k === 'shift') { this._keys.add('shift'); return; }
      if (!'wasdqerf'.includes(k) && !k.startsWith('arrow')) return;
      e.preventDefault();
      this._keys.add(k);
      this._interact();
    });
    window.addEventListener('keyup', (e) => {
      this._keys.delete(e.key.toLowerCase());
    });
    window.addEventListener('blur', () => this._keys.clear());
  }
}
