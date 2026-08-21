// Orbit / pan / dolly camera around a target point, driven by Pointer Events
// (mouse + touch + pen), with inertia, soft limits, and an idle "wiggle" that
// advertises parallax until the first interaction.

import { M4, V3, clamp } from '../util/math3d.js';

export class OrbitControls {
  /**
   * canvas: element to attach to
   * opts: {onChange, onPick(clientX, clientY)} — onPick fires on double-tap/click
   */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.onChange = opts.onChange || (() => {});
    this.onPick = opts.onPick || null;

    // camera state
    this.target = V3.make(0, 0, -2);
    this.homeTarget = V3.make(0, 0, -2);
    this.distance = 2;
    this.homeDistance = 2;
    this.yaw = 0;
    this.pitch = 0;
    this.fovY = 55 * Math.PI / 180;

    // limits (soft: rubber-band past these) — single-image splats fall apart
    // beyond ~30 degrees, so keep the reveal honest
    this.yawLimit = 0.5;
    this.pitchLimit = 0.38;
    this.minDistFactor = 0.3;
    this.maxDistFactor = 3.0;

    // motion
    this.velYaw = 0; this.velPitch = 0;
    this.wiggle = true;
    this._wigglePhase = 0;
    this._idleTime = 10; // start "idle" so wiggle runs immediately
    this._everInteracted = false;

    this._pointers = new Map();
    this._pinchDist = 0;
    this._lastTap = { t: -1e9, x: 0, y: 0 };
    this._eye = V3.make();
    this._view = new Float32Array(16);

    this._bind();
  }

  setHome(targetZ, distance) {
    this.target = V3.make(0, 0, targetZ);
    this.homeTarget = V3.make(0, 0, targetZ);
    this.distance = distance;
    this.homeDistance = distance;
    this.yaw = 0; this.pitch = 0;
    this.velYaw = 0; this.velPitch = 0;
    this._idleTime = this._everInteracted ? 0 : 10;
    this.onChange();
  }

  reset() {
    this.target = Float32Array.from(this.homeTarget);
    this.distance = this.homeDistance;
    this.yaw = 0; this.pitch = 0;
    this.velYaw = 0; this.velPitch = 0;
    this.onChange();
  }

  eye(out = this._eye) {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    // yaw=pitch=0 -> camera at target + (0,0,+dist), i.e. original photo viewpoint
    out[0] = this.target[0] + this.distance * sy * cp;
    out[1] = this.target[1] + this.distance * sp;
    out[2] = this.target[2] + this.distance * cy * cp;
    return out;
  }

  viewMatrix(out = this._view) {
    return M4.lookAt(this.eye(), this.target, V3.make(0, 1, 0), out);
  }

  /** advance inertia + wiggle; returns true if the camera moved */
  update(dt) {
    dt = Math.min(dt, 0.05);
    let moved = false;
    this._idleTime += dt;

    if (this._pointers.size === 0) {
      if (Math.abs(this.velYaw) > 1e-4 || Math.abs(this.velPitch) > 1e-4) {
        this.yaw += this.velYaw * dt;
        this.pitch += this.velPitch * dt;
        const damp = Math.exp(-4.5 * dt);
        this.velYaw *= damp; this.velPitch *= damp;
        moved = true;
      }
      // spring back inside soft limits
      const yovr = this.yaw - clamp(this.yaw, -this.yawLimit, this.yawLimit);
      const povr = this.pitch - clamp(this.pitch, -this.pitchLimit, this.pitchLimit);
      if (Math.abs(yovr) > 1e-5 || Math.abs(povr) > 1e-5) {
        this.yaw -= yovr * Math.min(10 * dt, 1);
        this.pitch -= povr * Math.min(10 * dt, 1);
        moved = true;
      }
    }

    if (this.wiggle && !this._everInteracted && this._idleTime > 1.2) {
      this._wigglePhase += dt;
      const t = this._wigglePhase;
      const amp = Math.min((this._idleTime - 1.2) / 2, 1) * 0.045;
      this.yaw = Math.sin(t * 0.9) * amp;
      this.pitch = Math.sin(t * 0.53 + 1.2) * amp * 0.5;
      moved = true;
    }
    return moved;
  }

  _interact() {
    this._everInteracted = true;
    this._idleTime = 0;
  }

  _applyOrbit(dx, dy) {
    const k = 2.2 / Math.max(this.canvas.clientHeight, 1);
    const ny = this.yaw - dx * k;
    const np = this.pitch + dy * k;
    // rubber-band outside limits
    const soft = (v, lim) => {
      if (v > lim) return lim + (v - lim) * 0.35;
      if (v < -lim) return -lim + (v + lim) * 0.35;
      return v;
    };
    this.yaw = soft(ny, this.yawLimit);
    this.pitch = soft(np, this.pitchLimit);
    this.velYaw = -dx * k * 60 * 0.35;
    this.velPitch = dy * k * 60 * 0.35;
  }

  _applyPan(dx, dy) {
    // move target in camera plane; scale so a pixel of drag ~ a pixel on screen at target depth
    const h = Math.max(this.canvas.clientHeight, 1);
    const worldPerPx = 2 * this.distance * Math.tan(this.fovY / 2) / h;
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    // camera basis
    const right = [cy, 0, -sy];
    const up = [-sy * sp, cp, -cy * sp];
    const limit = this.homeDistance * 0.9;
    for (let i = 0; i < 3; i++) {
      this.target[i] = clamp(
        this.target[i] + (-dx * right[i] + dy * up[i]) * worldPerPx,
        this.homeTarget[i] - limit, this.homeTarget[i] + limit);
    }
  }

  _applyDolly(factor) {
    this.distance = clamp(this.distance * factor,
      this.homeDistance * this.minDistFactor, this.homeDistance * this.maxDistFactor);
  }

  _bind() {
    const c = this.canvas;
    c.style.touchAction = 'none';

    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      this._pointers.set(e.pointerId, {
        x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY,
        button: e.button, shift: e.shiftKey,
      });
      this._interact();
      this.velYaw = 0; this.velPitch = 0;
      if (this._pointers.size === 2) {
        const [a, b] = [...this._pointers.values()];
        this._pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
        this._lastTap.t = -1e9; // multi-touch is never a tap
      }
      // double-tap / double-click detection
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
      // a drag is not a tap: a flick-flick must not fire an accidental pick
      if (Math.hypot(e.clientX - p.x0, e.clientY - p.y0) > 12) this._lastTap.t = -1e9;

      if (this._pointers.size === 1) {
        const pan = p.button === 2 || p.shift;
        if (pan) this._applyPan(dx, dy);
        else this._applyOrbit(dx, dy);
      } else if (this._pointers.size === 2) {
        p.x = e.clientX; p.y = e.clientY;
        const [a, b] = [...this._pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (this._pinchDist > 0 && d > 0) {
          this._applyDolly(this._pinchDist / d);
        }
        this._pinchDist = d;
        // two-finger drag pans (use this pointer's delta halved)
        this._applyPan(dx * 0.5, dy * 0.5);
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
      // trackpad pinch arrives as wheel+ctrlKey; both dolly
      const k = e.ctrlKey ? 0.01 : (e.deltaMode === 1 ? 0.05 : 0.0015);
      this._applyDolly(Math.exp(e.deltaY * k));
      this.onChange();
    }, { passive: false });

    c.addEventListener('contextmenu', (e) => e.preventDefault());

    // Safari (macOS trackpad / iPadOS) reports pinch as proprietary
    // GestureEvents with e.scale — it does NOT synthesize wheel+ctrlKey like
    // Chromium/Firefox. Map them to dolly; skip while a real two-pointer
    // pinch is active (iPadOS direct touch can fire both).
    this._gestureScale = 1;
    c.addEventListener('gesturestart', (e) => {
      e.preventDefault();
      this._interact();
      this._gestureScale = e.scale || 1;
    });
    c.addEventListener('gesturechange', (e) => {
      e.preventDefault();
      if (this._pointers.size >= 2) return;
      if (e.scale > 0 && this._gestureScale > 0) {
        this._applyDolly(this._gestureScale / e.scale);
        this.onChange();
      }
      this._gestureScale = e.scale || this._gestureScale;
    });
    c.addEventListener('gestureend', (e) => {
      e.preventDefault();
      this._gestureScale = 1;
    });
    // pinches starting on the UI overlay must not zoom the page either
    for (const ev of ['gesturestart', 'gesturechange', 'gestureend']) {
      document.addEventListener(ev, (e) => e.preventDefault());
    }
    c.addEventListener('dblclick', (e) => e.preventDefault());
  }
}
