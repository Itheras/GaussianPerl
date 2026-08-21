# GaussianPerl — Session Scratchpad

> Working notes for AI/dev sessions. Keep this updated at every milestone so future
> sessions can pick up context fast. Newest entries at the top of each section.

## Mission

Turn a **single photo** into an interactive **3D Gaussian splat**, using every trick
available — including a **local AI depth model** — running on **Mac M-series and
iPhones**. Mouse *and* touch navigation (orbit / pan / dolly), refocus (depth of
field with tap-to-focus), then **save a normal PNG** of the view you like.

## Core architecture decisions (do not re-litigate without reason)

- **Zero-build static web app** (ES modules, no bundler, no npm install needed to run).
  Rationale: one codebase covers macOS *and* iOS Safari; `python3 -m http.server`
  is the whole dev loop; works on GitHub Pages.
- **Rendering: WebGL2** (Safari 15+, universal). WebGPU is *not* required for
  rendering; it is used opportunistically only for AI inference.
- **Local AI depth: Depth Anything V2 Small** via `@huggingface/transformers` (v3,
  CDN + browser-cached). Device: `webgpu` when available (Safari 26+/Chrome),
  else `wasm` with q8 quantization. Weights download once (~25–50 MB) and are
  cached by the library in browser storage; inference is fully on-device.
- **No-model fallbacks** so the app always works offline:
  1. bundled sample image ships with **ground-truth depth PNG** (generated
     procedurally together with the image),
  2. heuristic depth (vertical gradient + center prior, edge-aware smoothed)
     for arbitrary images when the model can't load.
- **Heavy CPU work in workers**: splat building in `pipeline-worker.js`,
  depth sorting in `sort-worker.js` (16-bit counting sort, antimatter15-style).
  No SharedArrayBuffer (avoids COOP/COEP headaches on static hosts).
- **Camera convention**: right-handed, camera at origin looking down **−Z**,
  Y up. Pixel (u,v) at depth z>0 → position ((u−cx)/f·z, −(v−cy)/f·z, −z).
- **Splat storage**: SoA buffers — `positions` f32×3, `cov3d` f32×6 (upper
  triangle of Σ, precomputed CPU-side as Σ = Σᵢ sᵢ² aᵢaᵢᵀ), `colors` u8×4.
  Renderer stores them in RGBA32F/RGBA8 textures, draws instanced quads,
  fetches by sorted index via `texelFetch`.

## The bag of tricks (single image → convincing splat)

- AI monocular depth (relative disparity) → parameterized disparity-to-depth mapping.
- Joint bilateral refinement of disparity guided by image color (snaps depth edges
  to color edges; kills halos).
- Per-splat **normals from depth**, tangent-oriented anisotropic discs; stretch along
  the tilt direction by 1/cosθ (capped 3×) so slanted surfaces stay closed.
- **Depth-discontinuity mask** → edge splats stay isotropic + alpha-feathered
  (no rubber-sheet streaks between fg/bg).
- **Layered disocclusion fill**: behind foreground silhouettes, synthesize a
  background layer — color+depth inpainted by multi-directional background pull +
  smoothing + variance-matched noise. These splats are invisible until parallax
  reveals them. (The "fill missing data" part; classical, fast, on-device.)
- **Border skirt outpainting**: mirror-extended fading rim beyond image borders so
  small camera moves don't reveal a hard cut.
- **Multi-scale underlayer**: ×4-downsampled large splats beneath the fine layer to
  fill any cracks when dollying in.
- DoF post-pass (poisson gather, CoC from per-pixel composited depth), tap/dbl-click
  to focus. Aperture 0 ⇒ pass-through.
- Soft-clamped orbit (single-image splats break down past ~±40°), inertia, idle
  auto-wiggle to advertise parallax (stops on first interaction).

## Milestone log

### M0 — Skeleton (DONE)
- Empty repo → branch `claude/gaussian-splat-single-image-0iwx6y`, skeleton,
  this scratchpad, README, .gitignore.

### M1 — Sample assets + math utils (planned)
- `tools/gen-sample.mjs`: node script, no deps — procedural scene → `assets/sample.png`
  + `assets/sample_depth.png` (16-bit-ish GT disparity in 8-bit PNG), minimal PNG
  encoder via zlib.
- `src/util/math3d.js`, unit-testable in node.

### M2 — Renderer + controls (planned)
- WebGL2 instanced splatting, sort worker, orbit controls (mouse+touch), DoF composite.

### M3 — Pipeline (planned)
- depth-ai (transformers.js), depthproc, inpaint, splat-build, pipeline worker.

### M4 — UI + save (planned)
- Minimal overlay UI, quality presets, PNG capture (offscreen FBO + readPixels →
  2D canvas → toBlob; iOS share sheet), `.splat` export (eigendecomp of Σ).

### M5 — Tests + hardening (planned)
- Node unit tests (math, sort, PNG encoder, eigendecomp round-trip).
- Playwright headless e2e: serve, load `?demo=1&nomodel=1`, assert non-empty render,
  drag changes pixels, capture works.
- Multi-lens adversarial review (Safari/iOS compat, GL math, memory, UX) → fixes.

## Known platform gotchas (learned/anticipated)

- iOS Safari memory: cap base-layer splats ≈ 0.7 M on phones (UA + deviceMemory
  heuristic), ≈ 1.4 M desktop. RGBA32F textures: ~48 B/splat GPU side.
- `EXT_color_buffer_float` needed for HDR+depth MRT; fall back to
  `EXT_color_buffer_half_float`, then RGBA8 (depth encoded 8-bit) — DoF degrades
  gracefully.
- WebGL2 has one blend func for all MRT attachments — depth accumulation target
  must use the same premultiplied-alpha blend as color; that's fine
  (alpha-composited mean depth is what DoF wants).
- Module workers (`type:"module"`) OK on Safari 15+; SharedArrayBuffer NOT used.
- `canvas.toBlob` after explicit re-render in same task = reliable without
  preserveDrawingBuffer; we capture via offscreen FBO readPixels instead (safer).
- iOS `100vh` lies; use `100dvh` + `visualViewport`. `touch-action:none` on canvas.
- HEIC photos: iOS Safari decodes natively via `<img>`/createImageBitmap — just
  don't try to parse bytes ourselves.
- transformers.js: import from CDN inside try/catch — app must still boot offline.

## Open questions / next steps

- (none yet — fill in as they appear)
