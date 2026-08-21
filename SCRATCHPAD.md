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

### M7 — Generative fill: "the missing data must look real" (DONE)
Goal: revealed/outpainted areas indistinguishable from the photo. Research
(6-agent web workflow, findings verified with curl/graph inspection) → design →
implement → 35-agent adversarial review (27 confirmed findings applied/triaged).
- **Fill model: MI-GAN-512-Places2 full pipeline ONNX** (28.1 MB,
  `andraniksargsyan/migan/migan_pipeline_v2.onnx`, revision-pinned, sha256-gated,
  jsdelivr `migan-onnx@1.0.0` as hash-gated mirror; GitHub release assets send
  NO CORS header — not a mirror option). I/O: uint8 `image` [1,3,H,W] dynamic,
  uint8 `mask` [1,1,H,W] **255=known, 0=hole**, output `result` uint8; crop
  around mask bbox +128px → 512 → feathered ScatterND blend ALL in-graph, known
  pixels bit-exact. Runs under **onnxruntime-web 1.27.0** INSIDE the pipeline
  worker. Verified live: webgpu EP works on Chrome (fast), wasm elsewhere.
- **ORT bundle choice is a SAFETY matter**: WebKit gets `ort.wasm.bundle.min.mjs`
  (plain build) ONLY — jsep/asyncify builds melt WebKit 26.2's JIT (ORT #26827,
  unfixed); webgpu EP broken on iPhone (#26480). Non-Safari+adapter probe OK →
  `ort.webgpu.bundle.min.mjs` with run-time fallback to a fresh wasm session.
  isSafariEngine(): iOS-family check FIRST (CriOS/EdgiOS are WebKit!); Chrome on
  iPadOS wears the desktop-Mac UA → main thread forwards `webkitHint`
  (Macintosh + maxTouchPoints>2) into the worker.
- **Depth moved into the pipeline worker + transformers.js v4.2.0** (v3 ships
  only the JSEP wasm → exposed on iOS 26.2; v4 pins Safari to the plain build
  but no longer proxies wasm off-thread → main-thread inference would jank).
  Tiering: DA V2 **base q4f16** on desktop webgpu, small fp16 webgpu, small q8
  wasm. MUST probe `navigator.gpu.requestAdapter()` before ANY webgpu attempt:
  gpu-object-exists-but-no-adapter environments fail AND a failed webgpu
  attempt can poison the in-context wasm fallback (found via e2e, probe-fixed).
  `device:'wasm'` alone works fine in v4 when attempted first.
- **Fill algorithm**: classical `synthesizeBackground` keeps supplying GEOMETRY
  (bgDisp/bgMask — Kopf 2020: heuristic depth diffusion is adequate); the model
  replaces COLOR. Hole mask = bgMask ∪ fg **collar** (surface-following BFS
  from fgBoundary, ~2% short side — the occluder must be INSIDE the mask or
  the GAN continues it into the hole) ∪ 2px mixed-pixel rim. Per-cluster calls
  (32px-grid CC → merge <48px → split >512px w/ 96 overlap → coarsen ×1.6 until
  ≤6 calls desktop / 4 mobile; clusters with <24 *consumable* px keep classical)
  chained sequentially so later calls see earlier fills as context. Border
  outpaint: mirror-padded plate, ring mask, ≤6 tile calls at 768px, chained
  after interior. Prefill holes with classical colors so cross-tile context
  looks like background.
- **The two realism fixes that mattered (found by layer-isolation in browser)**:
  1. **Low-frequency anchoring**: GAN fills hallucinate at scale (dark blobs in
     grass, invented structures behind the horizon band). Fix: masked-box-blur
     correction `ai += blur((ref−ai)·mask)/blur(mask)` (r≈1.5% short side) —
     large-scale color pinned to the classical estimate, AI texture survives.
     Same treatment for the ring vs the exact plate init the model saw.
  2. **Far-field disparity compression** (`compressFarField`, knee 0.16 keep
     0.25, BEFORE snap/edge detection): real far fields don't parallax; the
     normalized-disparity range exaggerated sky/mountain separation into a
     "slab curtain" + made the horizon a giant hallucination-prone
     disocclusion edge. Compression turns it into one coherent backdrop and
     deletes those fills wholesale. (Sample GT: sky .02, ridges .055-.16,
     ground-foot .30 — banded worst case.)
- **Progressive build**: classical preview 'built' ships at classical speed →
  fill (download progress, per-call n/m) → 'final' built. Final is PARKED
  (`app.pendingCloud`) until its own sort lands — first visible frame fully
  sorted, no popping, no camera/focus yank (`setupId` guard). Superseded builds
  abort between model calls (macrotask checkpoint + AbortError — microtask-only
  chains starve onmessage) but a COMPLETED fill is cached even when superseded.
  Stage-cache: disp/edges/bg/fill keyed by sourceId; fill cache-hit requires
  plate.padPx ≥ current skirtPx AND bgColor when bg wanted; `fillFailed` flag
  stops per-rebuild churn (retry on next source).
- **Robustness net** (review findings): est.estimate() try/catch → heuristic
  (OOM on phones must not kill the build); model-session mutexes (worker
  re-entry during await); main-side 150s watchdog fed by ANY worker message
  (stale-id progress included!) → if preview stands, only `aiFillBroken` (cloud
  kept), else full respawn + `aiBroken`; worker onerror stops the watchdog
  (no respawn loop) + `workerDead`; export button re-enabled on respawn;
  renderer.setCloud no-ops on lost context; alpha preserved through the model
  round-trip (transparent-source guard stays alive); no-Content-Length
  downloads still post progress; crypto.subtle absent (http-over-LAN dev) →
  author-URL-only unverified, mirror disabled; skipped-cluster bg pixels keep
  classical shade ×0.94 + single grain; discarded collar output excluded from
  the plate init/context/anchor; buildSplats returns cap-sized buffers +
  count (no slice copies, all consumers honor count).
- **UI**: 'AI fill' toggle; status flow 'Enhancing hidden areas with AI…' →
  '✨ AI fill applied · N splats'; modelInfo 'depth: DA V2 small · wasm · fill:
  MI-GAN · webgpu'; URL params: `nomodel` (no AI at all), `nofill`,
  `fillep=wasm` (e2e/SwiftShader). bg shade 1.0 for AI fills (0.94 classical),
  plate skirt fade exp 1.2 α≤240, ring disparity smoothed (blur + ramp) —
  raw replicate depth made the horizon cliff a staircase of slabs in the ring.
- **Deferred (recorded, with review's blessing-by-triage)**: [8] nested model
  worker + terminate for iOS memory reclaim (wasm heaps never shrink; both
  runtimes+stage cache resident ≈ low-hundreds MB — revisit if jetsam shows
  up); [9] per-call crop packing (pack is 1-2% of a call vs inference — not
  worth contract risk); [15] webgpu-poisoning respawn protocol for depth
  (adapter probe covers the common case); [21B] preview/final segment reuse;
  [23B] sub-rect blurs. LaMa fp16 webgpu HQ tier (g-ronimo export, 106MB,
  0.25s/call) documented as a possible desktop upgrade.

### M6 — Adversarial review + fixes (DONE)
- Ran a 28-agent multi-lens review (6 finders: safari-ios / gl-correctness /
  math-pipeline / workers-async / touch-ux / memory-perf → dedupe → 1
  adversarial verifier per finding). 22 findings → 21 confirmed, 1 refuted
  (lowp-sampler; added `precision highp sampler2D` anyway, zero cost).
- **Fixed (all verified by re-running unit + e2e suites):**
  - `<img>`-first decode: Safari's createImageBitmap(blob) ignores EXIF
    (WebKit bug 237895) and *succeeds*, so try/catch can't catch it — iPhone
    portrait photos built sideways splats. `<img>`+decode() is correct
    everywhere; ImageBitmap kept only as fallback for exotic formats.
  - Safari GestureEvent pinch → dolly (macOS trackpad/iPadOS fire gesture*
    with e.scale, NOT wheel+ctrlKey); document-level gesture preventDefault
    so UI-chrome pinches don't zoom the page.
  - webglcontextlost/restored: renderer._initGL() refactor + reinit;
    main re-uploads cloud and re-sorts on restore (iOS sheds GL contexts).
  - Sort races: onBuilt bumps sortGen + drops spareIdx (in-flight sort of the
    OLD cloud must never index the new textures); sorted handler also checks
    indices.length === cloud.count.
  - openBlob/openSample load-token guard (concurrent opens mixed imageData
    from one photo with disparity from another); ensureEstimator memoized.
  - Export errors routed by msg.id==='export' before the buildId filter
    (errors used to strand the Export button disabled forever).
  - Worker onerror/onmessageerror wired; hint chip pointer-events:none
    (it swallowed bottom-center taps for 8 s); double-tap invalidated by
    drag >12px and by multi-touch; Lens slider now updates controls.fovY.
  - DoF strength AND cap both scale with dpr (blur used to differ 1x vs 2x).
  - snapDepthEdges threshold aligned to `jump` exactly (0.8j–1.0j steps were
    steepened but then missed by every silhouette consumer).
  - Memory/perf: onBuilt uses transferred buffers directly (was deep-copying
    ~110 MB at 'high'); bg layer capacity counted exactly from bgMask (was
    worst-case w*h ≈ +50 MB); pipeline worker stage-cache keyed by sourceId
    (Depth slider now skips refine+inpaint, straight to buildSplats);
    jointBilateral color weight via 2048-entry LUT (was ~35M Math.exp).
- **Deferred (recorded, not bugs):** underlayer overdraw (~1–2 ms GPU,
  inherent to the crack-filling design); inpaint at working res (one-time
  cost, now cached); far-field terracing on the sample at extreme yaw.

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

- **ORT-web on WebKit: plain wasm bundle ONLY** (`ort.wasm.bundle.min.mjs`).
  The jsep/asyncify builds melt WebKit 26.2's JIT (ORT #26827); webgpu EP
  fails on iPhones (#26480). Every iOS browser is WebKit (CriOS/EdgiOS too);
  Chrome-on-iPadOS wears the desktop-Mac UA → needs main-thread webkitHint.
- **Probe `navigator.gpu.requestAdapter()` before ANY webgpu attempt** (ORT or
  transformers.js): gpu object can exist with no adapter (headless, blocklists),
  and a failed webgpu attempt can poison the same-context wasm fallback.
- transformers.js v4: no wasm proxying (inference blocks the calling thread —
  run it in a worker); v3 ships ONLY the JSEP wasm binary (iOS 26.2 hazard).
- Wasm model calls block the worker thread: nothing async can preempt them.
  Yield a MACROTASK between calls or queued onmessage (rebuilds!) starves;
  main-side watchdog must treat ANY worker message as liveness (stale ids too).
- HF `/resolve/` URLs + jsdelivr: CORS `*` verified. GitHub release assets:
  NO ACAO header — never a browser model mirror.
- CacheStorage works in workers; crypto.subtle needs a secure context (absent
  on http-over-LAN iPhone dev — hash-gating policy degrades to author-URL-only).

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

- **Verify on real devices**: iPhone Safari (wasm fill timing 2-5s/call ×
  2-4 calls, memory headroom, webkitHint path) and desktop Safari 26. The
  e2e-ai suite covers real model downloads + wasm inference in Chromium only.
- iOS memory reclaim (deferred review finding [8]): both AI runtimes stay
  resident in the pipeline worker. If iPhone jetsam appears, move models into
  a terminable child worker (Safari ≥15.5 supports nested workers).
- Optional HQ fill tier for desktop webgpu: g-ronimo/lama `lama_512_fp16.onnx`
  (106.6 MB, Apache-2.0, matmul-FFT rewrite, ~0.25 s/call webgpu; NEVER wasm —
  52-61 s). Manual crop/512/composite adapter needed (fixed input, float32
  [1,4,512,512], mask channel 1=hole — polarity OPPOSITE to MI-GAN).
- Far-field knee (0.16/0.25) tuned on the synthetic sample; sanity-check on
  real photos (indoor scenes: a far wall at normalized disp <0.16 loses a bit
  of true parallax — acceptable trade so far).
- The e2e stale-JS-eval quirk in the in-app browser pane (javascript_exec can
  hit a pre-navigation context right after navigate) — screenshots are ground
  truth; re-eval after the page settles.
