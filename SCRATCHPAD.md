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

### M5 — Tests + hardening (DONE)
- 25 node unit tests (`node tests/run.mjs`): math, imageops, depthproc, inpaint,
  splat-build, .splat export, PNG encoder, sort-worker (simulated via
  `globalThis.self` shim).
- Headless e2e (`node e2e/run-e2e.mjs`): Playwright + SwiftShader
  (`--enable-unsafe-swiftshader`), serves repo, loads
  `?demo=1&nomodel=1&nowiggle=1&quality=low&maxpx=140000`, asserts: build,
  all four layers present, non-empty render with structure, drag orbits +
  changes pixels, wheel dollies, dbl-click refocuses near→far, DoF changes
  pixels, PNG encodes, .splat is count*32 bytes. Record shots in `e2e/out/`.
- **Artifact hunt (important learnings, see gotchas below)** — fixed in order:
  1. `[hidden]` elements resurrected by author `display:flex` → invisible
     full-screen overlays swallowed all pointer events. Fix: `[hidden]{display:none!important}`.
  2. Bilinear-downsampled disparity ⇒ mixed-depth "streak" splats along
     silhouettes. Fix: `snapDepthEdges` decision filter (3×3 min/max snap).
  3. Disocclusion fill averaged near+far march hits ⇒ mid-air "veil" sheets.
     Fix: plain-median cluster selection (dominant background surface).
  4. Ground's own smooth gradient passed the "farther than me" march test ⇒
     phantom fills under open ground. Fix: hit only counts if the march
     **crossed a discontinuity** (single-step drop > 0.8·jump).
  5. Camera-facing fill discs opened into a dashed lattice at grazing angles.
     Fix: orient fills by the bg depth-field gradient (shared `orientedCov`),
     shade ×0.94, band auto-scales (~11% short side, clamp 12..72).

### M4 — UI + save (DONE)
- index.html/styles.css/main.js: toolbar, settings panel, welcome, status chip,
  drag/paste, focus ring, safe-area + dvh. Save = offscreen 2× FBO render →
  readPixels → 2D canvas → toBlob PNG (DoF scaled to capture res inside
  renderer.capture). `.splat` export runs in the pipeline worker (eigendecomp).
- URL params: `demo`, `nomodel`, `nowiggle`, `quality=low|medium|high`,
  `maxpx=N` (test override). `window.__gp` exposes app/controls/renderer/captureNow.

### M3 — Pipeline (DONE)
- depth-ai.js: transformers.js v3 from jsdelivr CDN, Depth Anything V2 small,
  webgpu/fp16 → wasm/q8 fallback chain, progress callback, null on failure.
- pipeline-worker: normalize (percentile 1.5/98.5) → jointBilateral refine
  (AI only) → snapDepthEdges → edgeMask/fgBoundary → synthesizeBackground →
  buildSplats. Also handles 'export'.

### M2 — Renderer + controls (DONE)
- Instanced quads, data in RGBA32F/RG32F/RGBA8 textures fetched by sorted
  index (uint attrib, divisor 1). EWA: Σ2D = (J·R)Σ(J·R)ᵀ + 0.3px lowpass,
  2.5σ quads. MRT: premult color + (depth·α, α) with shared blend
  ONE/ONE_MINUS_SRC_ALPHA. Composite pass: bg gradient + poisson-16 DoF
  gather (scatter-as-gather weights, IGN rotation). RGBA16F targets with
  RGBA8+encoded-depth fallback.
- sort-worker: 65536-bucket counting sort, ping-pong index buffer transfer.
- OrbitControls: Pointer Events (1-finger orbit / 2-finger pinch+pan /
  right-or-shift-drag pan / wheel+ctrlKey trackpad pinch), inertia,
  soft limits (yaw 0.5, pitch 0.38 — beyond that the illusion dies),
  double-tap pick callback, idle wiggle until first interaction.

### M1 — Sample assets + math utils (DONE)
- Procedural golden-hour standing-stones scene, 1024×768, with exact GT
  disparity (16-bit in R/G) — the offline/no-model demo path.
- math3d (mat4 column-major, eigenSym3 Jacobi, matToQuat), imageops
  (resize, box blur, joint bilateral, gradients, dilate, histogram percentile),
  dep-free PNG encoder (None/Sub/Up filter heuristic).

### M0 — Skeleton (DONE)
- Empty repo → branch `claude/gaussian-splat-single-image-0iwx6y`, skeleton,
  this scratchpad, README, .gitignore.

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
- `[hidden]` + author `display:` rules: UA `[hidden]{display:none}` LOSES to any
  author display rule. Keep the `[hidden]{display:none!important}` reset.
- Headless e2e needs `--enable-unsafe-swiftshader` (Chromium ≥ 137) for WebGL2;
  software rasterizing 0.5M splats is seconds/frame → e2e uses `maxpx=140000`.
- GL screenshots: don't trust page.screenshot / toDataURL on the WebGL canvas
  (no preserveDrawingBuffer); use `__gp.captureNow()` (offscreen FBO+readPixels).
- Debug trick that found the lattice bug: re-upload the cloud with all layers
  except one alpha-zeroed (`scratchpad debug-layers.mjs` pattern), render each
  layer in isolation.

## Open questions / next steps

- Mountains/horizon band on the sample still reads as terraced cardboard at
  extreme yaw (>0.4) — inherent to giant far-depth cliffs; acceptable within
  soft limits. Could compress far-field disparity nonlinearly if it bothers.
- Fill brightness can mismatch surroundings slightly (haze-lightened pulls);
  shading ×0.94 hides most of it.
- transformers.js AI path is untested in the headless env (no model download
  in CI) — verify on a real device when possible; the wasm-q8 fallback chain
  is the risk area.
