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

### M11 (research) — "the point cloud as an LLM": next-point prediction
User idea: make the representation itself GENERATIVE — committed scene as
context, a model guessing the missing points like next-token prediction.
4-lane research sweep + adversarial verification (2026-08-22), all lanes held.

**Headline: the idea exists, published, and is brand new.** `GaussianGPT`
(nicolasvonluetzow/GaussianGPT, ECCV 2026 Oral, Niessner group, pushed
2026-08-21): sparse-3D-CNN VQ-VAE turns per-voxel Gaussians into DISCRETE
TOKENS; a ~350M causal GPT with 3D RoPE predicts them in xyz voxel order;
natively does COMPLETION/OUTPAINTING conditioned on a partial Gaussian scene;
real sampling (temperature exists). MIT, ~5.2 GB open checkpoints. Gaps vs our
need: NO image conditioning (grep-verified), trained ONLY on synthetic
interiors (3D-FRONT/ASE), voxelized fidelity, CUDA stack (MinkowskiEngine — a
CPU/OpenMP build exists for one-shot encode/decode; flash-attn -> SDPA is easy).

**Taxonomy of "next-X for 3D"** (all verified): next-TOKEN (MeshGPT lineage,
GaussianGPT, OccWorld; Copilot4D is paper-only), next-VIEW (CUT3R: 768
persistent state tokens, queryable at VIRTUAL poses via raymap — but readout is
deterministic regression = blurry means, and CC BY-NC-SA contaminated;
STream3R = literal KV-cache causal transformer over frames, ICLR 2026),
next-FRAME (world models: Genie 3 closed; Matrix-Game 3.0 Apache-2.0 open 5B
w/ frustum-overlap memory; HY-WorldPlay proves naive few-step distillation
DESTROYS memory; Gen3C = point-cloud cache rendered into a video model,
~43 GB CUDA; HunyuanWorld-Voyager commits generated RGB-D back into its cache
— our loop with a 7B generator — but 60+ GB CUDA + territory-excluded licence),
next-SCALE (VAR -> PointNSP MIT: order SCALES, not points — the clean answer
to permutation invariance; MAR = continuous tokens without VQ).

**Verified negatives (the frontier gap):** nobody does photo-conditioned,
photoreal, scene-scale next-splat AR with provenance. Nobody combines a
LEARNED queryable state with a SAMPLING readout. Zero Apple-Silicon presence
in the entire field. And the big one: Matrix-Game 3.0 / WorldMem / VMem /
Gen3C all REINVENTED our architecture (posed-view memory + frustum retrieval)
as learned approximations — none has byte-exact home pose or per-pixel
provenance with a refusal guard. We are not behind on memory; we lack only the
strong guesser.

**Local prep landed:** `src/pipeline/points.js` — anchorToPoints (world-space
serialisation, the guesser's context) + crossViewConsistency (matched/occluded
/floating buckets). Measured on analytic GT: self-exact (100% matched, 0
floaters); wide-baseline side view 94.8% matched, 0.16% floating with the
silhouette-cliff filter (3.6x fewer floaters than unfiltered); vs a view from
BEHIND: zero floaters (front points correctly occluded, never contradicting).
3 new tests; 57 total green.

**Recommended experiment (not started):** GaussianGPT transfer test. Local-first
order: (1) VQ round-trip our photo-derived Gaussians through the CPU/OpenMP
encoder build — if prompt tokens don't round-trip at PSNR >= 30 dB the
synthetic-domain tokenizer kills the idea regardless of the GPT; (2) GPT is a
plain 350M transformer -> SDPA/MPS port for completion sampling; (3) only rent
CUDA if the local path stalls (user has said local-only for the PRODUCT; a
dev-time CUDA box is their call). Fallback: render-conditioned inpainting
upgrade (GEN3C's pattern at local scale) — captureAnchorFrame already emits
exactly that conditioning.

**M11 build log — the strong guesser, tested until it was right (2026-08-22/23)**
Decision (user): highest chance of success = keep the anchor loop, replace the
guesser. Built `sidecar/server.py` (loopback HTTP, bearer token, origin
allowlist, framed binary protocol) + `src/backend/native-fill.js` (same
`fill()` contract as MI-GAN) + worker/main wiring (`?sidecar=PORT:TOKEN`,
`globalThis.__GP_SIDECAR__` for the shell). uv venv, torch 2.13 MPS,
diffusers 0.40.0. Findings, each one measured:
- **FLUX.2 klein-4B is NOT an inpainter.** Its diffusers inpaint pipeline
  passes the whole frame as a reference (`ref_images=[image_latents_encoded]`),
  so whatever sits inside the mask is reproduced: seed in -> seed out; NOISE in
  -> noise out; grey in -> a giant hallucinated bald head in the centre and grey
  kept at the border. 51-67 s per fill. Dropped. Code path kept as `--model klein`.
- **SDXL-inpainting (openrail++) is a true inpainter** (9-channel UNet, masked
  region explicitly unknown): loads in 7 s, **27-31 s per 768x1024 fill at 20
  steps on M2 Max**, fp16 + madebyollin fp16-fix VAE (MIT). From a flat grey
  init at strength 0.99 it read the frame edge as a WHITE WALL; from the
  classical seed at strength 0.9 it continues beach/crowd/sky. Defaults now:
  steps 20, strength 0.9, guidance 7, init 'seed', scene prompt via ?prompt=.
- Small hole components are invisible to a diffusion model (8x latent on a
  ~1 MP frame): `splitHolesByArea` routes only large components to the
  generator; pinholes keep the classical fill.
- **THE real cause of "people turn to mush", found by a CPU ray trace:** the
  orbit PIVOT. `subjectDisparity` took the centre-box median; on the beach photo
  that is the WATER (0.141) and the couple (0.96) sit 7x nearer than the pivot.
  A 5-degree orbit sweeps their sample path 64% of the frame and off screen —
  the "blob" was the empty space where they stood. New rule: pivot on the
  median of the nearest 15% of the frame (floor band excluded) when it stands
  >1.6x in front of the centre depth. Result at the same 5 degrees: visible
  holes 20.6% -> 4.5%, photo share 0.82 -> 0.99, couple rendered intact.
- Also fixed on the way: out-of-plate samples are MISSES, not edge clamps (a
  subject touching the frame edge had its disparity replicated into the ring,
  and rays hit that phantom first with conf 0); the epipolar smear test is
  gated on a real cliff (|dd| > 0.03 over 3 texels) so body relief is not
  "grazing"; the silhouette COLLAR is masked for the model but DISCARDED on
  composite (it was repainting 18 px of a face); low-frequency anchoring is
  skipped for the diffusion backend (it erased the invented structure);
  the worker URL is cache-busted in dev (stale worker code burned an hour).
- MAD gate was 0.05·range and rejected every generated anchor (measured MADs
  0.048 / 0.096 / 0.090 — a monocular model is never affine to ours); the
  residual field keeps known pixels exact regardless, so the gate only needs
  to catch garbage: now 0.15·range. Invented people get relief.
- Generated anchors had NO far shell: a ray reaching their dMin without a hit
  returned nothing and the sky they painted was unreachable (black top-right
  at 15 degrees). Same fallback as the base now (dMin, 0.45 trust).
- **Result, beach photo, orbit about the couple:** 5.2° -> holes 4.5%, photo
  share 0.99; 10.3° -> 14.6%; 15.5° -> 24.2% before generation, 7.7% after
  three committed anchors, couple intact and recognisable throughout, sky and
  mountain sharp, background crowd/beach invented (~30 s per anchor). Open:
  cross-anchor ghosting (each anchor invents its own crowd; colours differ
  between an anchor's own invention and a later anchor's re-render of it) —
  the "one world instead of N opinions" fusion from the M10 plan is next; and
  the invented far side of a head is a guess (a sunhat appeared).

**M11 test log, Khomami portrait (2026-08-23) — "show me going around the subject"**
Setup: 3072x2047 AVIF (converted via sips), no EXIF -> default 53 deg, pivot on
the man (dSub 0.875, the near-subject rule), SDXL sidecar, background-only
prompt. Measured: 5.2 deg -> 10.9% holes, 10.3 deg -> 24.5%; both close to 0
after one generated anchor each (~30 s). Subject hole fraction at 10 deg: 0.9%
(the man is never masked).
FOUND AND FIXED ON THE WAY (each verified by test or probe):
- WebGL UNPACK_ALIGNMENT: half-float disparity rows not a multiple of 4 bytes
  failed texSubImage2D silently (INVALID_OPERATION); the texture stayed zero
  and EVERY pixel read as far shell at home. Hit any photo with an odd padded
  width (1187 here; 918/1124 had passed by luck). `pixelStorei(UNPACK_ALIGNMENT,1)`
  + an odd-width e2e regression.
- Smear test: gate on a real CLIFF (|dd| > 0.12 over 3 AND 12 texels) so body
  relief is never "grazing" (a tightened range alone masked the man's head);
  range [1.15, 1.8] once the gate restricts it to silhouettes.
- Fallbacks never blend with a confident candidate (a stretched wall nearer
  than the front bucket was a 25% ghost arm).
- Collar: discarded on composite AND kept at subject depth (clamping it to
  background made every anchor carry the rim at far depth).
- Seed from the far side only, with the WHOLE occluder excluded (push-pull's
  pyramid reach let a torso dictate a sky band's colour); rim background level
  from the 10th percentile (the collar puts half the rim on the subject).
- Review findings applied: sidecar deadline + abort on supersession, MI-GAN
  fallback on sidecar failure (and a scoping bug in it, caught by test),
  per-job watchdog, expansion reset on build error, header-first validation,
  clamped options, bounded body, generic 500s, load-failure cleanup + loadError
  in health, busy (503) instead of queueing, MPS cache release, private dump
  dir, generated token in the README, Host check.
- Sidecar-aware guards: HOLE_CEILING 0.45, MIN_BASE_SHARE 0.25 with a sidecar.
- Dev hygiene: EXPAND_VERSION stamped into anchor stats to detect stale workers.
THE OPEN ARTIFACT: a skin-coloured strip beside the subject at >= 10 deg.
Provenance (probed, not guessed): it lives in the anchor generated AT that
pose, alpha 1.0 (generated), far-shell depth — i.e. the diffusion model
paints the subject's continuation into the displaced-silhouette hole beside
him. Reproduced with: subject visible + scene prompt (a second man), subject
visible + background prompt + people negative (still skin), subject MASKED
(a tree / a man in the man-shaped mask), subject replaced by smooth seed in
context (smooth blobs), replaced by row-mirror (streaks), classical seed only
in a band beside the subject (flat sky-blue even below the horizon). The one
thing not yet tried and most likely to work: a structure-continuation
inpainter (LaMa-class: g-ronimo/lama ONNX, Apache-2.0, ~0.4 s) for
disocclusion bands, with the diffusion model reserved for beyond-frame
regions. SDXL's prior WANTS to extend the adjacent person; LaMa has no such
prior and is strong precisely at "continue the background texture".

### M10 (in progress) — toward a local Mac app: "move like a real camera"
User decision, FIXED: generation runs LOCALLY, no cloud ever; the product
becomes a NATIVE MAC APP; target Apple Silicon, 32 GB unified memory minimum.
Goal in their words: "the camera should be able to move around the image like a
person would in the real world with a camera looking around, even behind the
couple."

**Settled first: the renderer is not the limit.** `tests/scene3d.mjs` builds an
analytic 3D scene with exact ground truth and a CPU mirror of the shader march.
Results, now permanent tests: rest pose reproduces the source view with mean
abs error EXACTLY 0; a camera behind the subject with only the photo's anchor
reports coverage 0 (it invents nothing); adding ONE generated anchor from
behind makes that side renderable at coverage 0.68 and 1.6% mean colour error,
from a camera 0.55 units and 9 degrees away from the generated pose. So
"see a person's back" is a GENERATOR problem. Do not rewrite the renderer.

Landed so far (all generator-agnostic, all verified):
- **Disocclusions are clamped to background depth** (`clampHolesToBackground`).
  THE fix for the dominant artifact. A monocular model asked for the depth of
  an invented band beside an occluder answers "somewhere between" — a RAMP. A
  ramp is a surface, so colour stretches across it, it looks locally plausible,
  no confidence term flags it, and the next anchor bakes it in. Label each
  hole, and if its rim straddles a real depth cliff, clamp the hole to the far
  side; rims with no cliff (frame-border extensions) are left alone. Measured
  on a real photo: 138k px clamped over 8 components on the first anchor, and
  7.4 degrees of orbit went from an unrecognisable streak to sharp subjects.
  Usable range roughly doubled, which let anchorNear/anchorFar widen from
  0.05/0.16 to 0.10/0.34.
- **Grain moved to screen space, weighted by provenance.** `addGrain` baked
  noise into anchor textures, which the raymarch then resamples bilinearly — so
  invented regions got progressively CLEANER as the camera moved, the exact
  opposite of a photograph. That is the plastic look. Now: measure the source's
  own per-channel sigma once (Immerkaer kernel + median, validated to ~2% over
  a 4x range and exactly 0 on a clean image), and apply grain in COMPOSITE_FS
  after the DoF blur, scaled by sqrt(luma), weighted by
  max(1-baseShare, 1-conf). Measured result: median noise sigma in generated
  tiles 0.00193 vs 0.00193 in photograph tiles. At home the weight is exactly
  zero, so the photo is returned untouched — now asserted in e2e (conf 1.000,
  baseShare 1.000).
- **Per-pixel provenance** (`baseShare`) is emitted from the march: how much of
  each pixel is still the photograph rather than earlier generated anchors. It
  drives the grain weight, the drift guard, and the scene-adaptive frontier.
- **Front-surface election bug.** qFront was an unweighted max over confident
  candidates, so ONE over-near generated depth sample could elect itself the
  surface and the Gaussian then suppressed the photograph entirely — the colour
  prior could not save it because the weight was already ~0. A generated anchor
  must now be a clear band nearer than the photo to displace it.
- **A live non-commercial dependency.** `MODEL.idHQ` was
  `onnx-community/depth-anything-v2-base` = cc-by-nc-4.0 (verified on the HF
  API), tried FIRST on every desktop WebGPU machine. Default is now the
  apache-2.0 `-small`; `?hq=1` opts in knowingly.
- **`tools/devserver.mjs`**: static dev server with `Cache-Control: no-store`.
  `python3 -m http.server` sends no cache headers, so browsers heuristically
  cache ES modules AND the pipeline worker — editing a module and reloading
  silently keeps running the OLD code. This cost real time and produced one
  wrong measurement before it was noticed. If stale code is ever suspected
  again, change the PORT: a new origin is the only reliable way to drop entries
  a previous server already poisoned.

Calibration measured on a hard photo (two people filling the near field):
baseShare 0.88 and 0.64 -> generated view sharp, subjects survive; 0.50 -> the
fill model has replaced them with mush. MIN_BASE_SHARE is therefore 0.55, the
conservative side of the last known-good measurement, because a committed
anchor is PERMANENT pollution — there is no undo.

**The local-Mac plan (researched + adversarially verified ON an M2 Max / 32 GiB,
2026-08-22). Measured ceiling: `recommendedMaxWorkingSetSize` = 21.33 GiB, and
`maxBufferLength` = 16 GiB per single allocation.**

- **Camera-conditioned VIDEO generation is dead locally, and this is settled.**
  `Wan2.1-Fun-V1.1-1.3B-Control-Camera` (apache-2.0, DiT only 3.23 GB) FITS
  easily — memory was never the constraint. It measured **24.52 s per forward
  pass** at 49 frames on M2 Max; 50 steps x CFG = 100 passes = **41 min/clip**.
  The 4-step distill that would fix it DOES NOT EXIST for these checkpoints:
  Self-Forcing is a full 5.7 GB checkpoint with NO declared licence (not a
  LoRA), CausVid is cc-by-nc-4.0, and lightx2v/Wan2.1-Distill-Models contains
  no 1.3B model at all. Also: ComfyUI core's `WanCameraEmbedding` only exposes
  9 hardcoded preset trajectories. Do not re-propose this without a distill.
- **The substitute is better for our architecture anyway: generate a 3D ASSET,
  not novel views.** A committed mesh is cross-view consistent BY CONSTRUCTION
  and feeds the anchor loop we already have — skipping inpaint AND depth
  estimation, because the mesh gives exact colour and depth from every pose.
  `microsoft/TRELLIS.2-4B` (MIT, ungated, 512 pipeline 11.07 GB) via
  `shivampkumar/trellis-mac` (MIT, PyTorch MPS). **Peak 18 GB measured by the
  port author** — not the 3-5 GB a naive read of staged weights suggests, so
  the renderer MUST be leased down while it runs. Swap its default RMBG-2.0
  (non-commercial + gated) for `ZhengPeng7/BiRefNet` (MIT) on day one.
- **No human-specific 3D model is usable.** Every one fails on licence
  (LHM++ ships LICENSE_WEIGHT = CC BY-NC behind an Apache README badge; SiTH,
  ECON, ICON all NC) or on CUDA-only ops (nvdiffrast, kaolin, spconv,
  diff_gaussian_rasterization — none has a Metal path). "SMPLX-FREE" drops the
  pose ESTIMATOR, not the body model. So the couple gets reconstructed by a
  GENERIC OBJECT model, with no body prior — expect fused fingers and melted
  faces at grazing angles. That is the price of the only thing that runs.
- **Inpainting upgrade: `black-forest-labs/FLUX.2-klein-4B`** (apache-2.0,
  ungated, 15.96 GB) via diffusers `Flux2KleinInpaintPipeline`. CONFLICT TO
  RESOLVE BEFORE BUILDING: the plan says that pipeline is in released 0.40.0;
  the verifier says main-branch only. Check before committing. mflux CANNOT do
  FLUX.2 masked fill — its only fill path is bound to FLUX.1-Fill-dev, which is
  gated, non-commercial AND 33.9 GB.
- **Depth upgrade with zero new runtime: `onnx-community/depth-anything-v3-base`**
  (412.7 MB, apache-2.0, ungated). DA3-LARGE-1.1 is apache-2.0 in HF tags but
  CC BY-NC in the upstream README — two official sources disagree, do not ship.
- **Shell: Tauri v2.** Decisive measurement: WKWebView passes the renderer's
  FULL requirements (webgl2, EXT_color_buffer_float, RGBA16F AND RGBA32F FBO
  COMPLETE, MAX_TEXTURE_SIZE 16384, module workers) with zero JS errors. BUT
  **`navigator.gpu === false` in WKWebView** and the only lever is a private,
  App-Store-rejectable API. That is a ONE-WAY DOOR: it is acceptable only
  because all inference leaves the webview. The "Tauri adds ~0 MB" argument is
  FALSE (measured ~204 MB across four XPC processes); pick it for the 3-15 MB
  bundle, codesign/notarytool integration and sidecars instead.
- **CoreML EP is an 11.6x REGRESSION on MI-GAN** (2121 ms vs 183 ms CPU,
  measured): the graph is fully dynamic, ORT supports 11 partitions of 559
  nodes. Pin `CPUExecutionProvider` and re-benchmark per model, never assume.
- Sidecar runtime installed on first launch by bundled `uv` (18.5 MB) into
  Application Support, NOT PyInstaller. Models go in `app_data_dir()`, never
  `~/Library/Caches` (the OS may purge a 12 GB model mid-session). HF `resolve`
  URLs 302 to a pre-signed CDN URL — re-request the resolve URL on every
  resume rather than caching the redirect, or long pauses break exactly the
  resume they were built for.

Landed for it today: the PIXEL-EXACT GATE in e2e — at the home pose the render
is the photograph with max per-channel delta **0**, asserted fresh, after an
anchor commit, AND after a real `WEBGL_lose_context` loss/restore (which also
proves generated anchors survive a context loss, 1/1). Every later stage must
pass this or it does not ship. Plus `src/backend/capabilities.js`: one accessor
for WebGPU probing, replacing three independent `navigator.gpu` sites — the
third, in `inpaint-ai.js`, picks the ONNX execution provider and is the one
that would otherwise leave a WebGPU path alive inside a WebGPU-less WKWebView.

### M9 — Free camera + progressive generative completion (DONE)
User verdict on M8: "I can't move it at all, the 3D is now gone." Correct — the
M8 envelope was min(0.05·boost, 3%-of-frame parallax, 1.75× fill coverage),
which on a real photo resolves to ~2% of subject distance. The user asked for a
free camera that GENERATES the missing data as it moves, "the same way we
generate ai video". Full cutover.

- **The load-bearing identity** (`src/render/pose.js`): for a novel ray and an
  anchor at arbitrary relative pose, the sample position in the anchor's image
  is EXACTLY AFFINE in that anchor's disparity:
      v = (R_a R_b^T) dir_b,  C = R_a (C_b − C_a),  invVz = 1/v.z
      E = C − v(C.z·invVz)  (E.z = 0),  F = −v·invVz  (F.z = −1)
      u(d) = (0.5 + Ka.x·F.x) + d·(Ka.x·E.x/dSub)
      v(d) = (0.5 − Ka.y·F.y) + d·(−Ka.y·E.y/dSub)
  and s(d) = sBias + sScale/d IS the novel-frame depth (because dir_b.z is
  exactly −1 — never normalise dir_b, and never normalise v either: scaling v
  leaves uv alone but rescales sBias/sScale and silently turns s into radial
  distance). Verified numerically against a fully independent brute-force path
  over uniform-SO(3) poses and mismatched intrinsics: max relative error 1.7e-12
  in float64, 1.1e-5 in float32. So M8's "rotation shears faces" note was about
  focal-length ERROR under the old approximate model, not about rotation: with
  the true mapping, a 6-DoF camera costs exactly what translation cost — one
  constant uv delta per march step.
- **Renderer**: N anchors. Anchor 0 = the photo (M8's two layers, unchanged);
  generated anchors live in TEXTURE_2D_ARRAY (RGBA8 colour + R16F disparity), 4
  resident desktop / 2 mobile, marched nearest-first. `captureAnchorFrame()`
  renders a novel view as an anchor frame and reads back colour + novel-frame
  depth + confidence + provenance; `probeAt()` marches a single pixel for
  tap-to-pivot without any float-readback path.
- **THE bug that mattered** — M8's gap test (`surf − dHit > gapThresh`) can
  NEVER fire on a bilinear heightfield: the binary refinement always lands
  exactly on the ramp a silhouette becomes, so a stretched rubber sheet scored
  conf 1.0. Under a 3% envelope that ramp was one pixel wide; at 12° of orbit
  it is the whole subject, and it made the completion loop structurally blind
  to the exact artifact it exists to repair. Fix: differentiate the hit
  condition. Footprint scale along the epipolar direction is 1/(1 − G) with
  G = ∇disp·slope; `smear = 1 − smoothstep(2.2, 9, max(k, 1/k))`, k = |1 − G|.
  Exactly 1 at rest for any camera, so the photograph stays pixel-exact.
  (An independent adversarial review found the same defect and derived the same
  correction.)
- **Selection**: order-independent front-bucket softmax, not argmax. Pass 1
  takes qFront = max disparity among confident candidates; pass 2 weights each
  by conf·prio·exp(−behind²/2band²). Argmax draws a one-pixel seam along the
  locus where the winner flips, and that locus SLIDES as the camera moves —
  worse than the double image it avoids. Band is absolute+relative in disparity
  (0.012·range + 0.035·q, capped at half a silhouette): a relative depth margin
  is ~0.002 at the far shell (pure noise, the whole sky flips every frame) and
  ~0.05 at the near plane (a real thin object loses). prio = 1.0 base / 0.9 gen,
  so the photograph owns any pixel it can explain.
- **Completion loop** (`expand.js` + `novel-view.js`): holes → MI-GAN over a
  RELAXED push-pull seed (a directional smear as seed makes MI-GAN repeat the
  streak) → anchor low frequencies to the seed → grain → Depth Anything on the
  COMPLETED frame → full silhouette stack → robust affine align → commit.
  Alignment: trimmed LSQ with per-disparity-bin de-biasing (known pixels are
  near/mid-biased by construction — holes ARE the silhouettes and the borders),
  residual field extended by bilinear push-pull so the result equals the
  reference EXACTLY at known pixels and continues smoothly through the hole.
  MAD above 5% of the scene range ⇒ reject the geometry, keep the colour, fall
  back to push-pull rather than bake a warp in forever.
- **Guards that make it converge** (each one fixes a real failure seen in
  testing): hole threshold at or BELOW the validity threshold, or a permanent
  annulus regenerates forever · trust for re-rendered content ABOVE CONF_OK, or
  it can never occlude and always reads as a hole · grazing gate at cos −0.1,
  not −1e-3 (source step per unit disparity is hyperbolic below that; measured
  worst case 5.6e4 uv/disparity) · anchor confidence decays with distance from
  its own capture pose, or you smear a smear and nothing asks for a repair ·
  per-pixel provenance (baseShare) so the loop refuses to build an anchor out
  of earlier anchors' output · hysteresis + dwell + a post-commit efficacy
  check that blacklists poses where generating provably did not help.
- **The frontier is the real constraint.** Lateral motion runs off the EDGE of
  the photo long before it opens disocclusions: the source sample shifts by
  d·K·e/dSub, so near-field content leaves the padded plate after ~2-3% of
  subject distance. That is why M8's envelope was 3% — not timidity, geometry.
  M9 gets past it because every pass outpaints a 12%-wider frame, so coverage
  GROWS as you explore. Measured on a real photo: jumping straight to 9° gives
  37% holes and MI-GAN returns mush; walking there in 2° steps gives
  7.9% → 2.3% → 0.8% → 4.4% → 0.8%, three anchors, and it looks right. Hence
  HOLE_CEILING 0.25 plus a rubber-band that eases the camera back at the
  frontier instead of committing a frame of invention.
- **Limits, measured not guessed**: ±11.5° yaw, ±8° pitch, dolly 0.35–2.5×,
  ~7× M8's envelope. Excellent at ~3°, still convincing near 11°, and past that
  a 28 MB GAN is inventing more than it recalls. `Free roam` lifts them.
- **Camera**: orbit-centric (pivot/yaw/pitch/dist) — the only parameterisation
  well-conditioned for every gesture; home is exactly the photo's own camera.
  Flick inertia comes from angular VELOCITY, never delta·60: a coarse or
  synthetic pointer stream delivers a whole gesture in one event and launched
  the camera into the clamps (found immediately in browser testing). Clearance
  guard: throttled one-fragment probe down the view axis backs the camera off
  rather than letting it fly through a wall.
- **Results**: rest pose = the photograph, hole fraction exactly 0. Orbit opens
  detectable holes (0.17) and one generated anchor closes them (→0.000) — both
  asserted in e2e. 50 unit tests, full e2e green.
- **Known ceiling / next**: MI-GAN is the binding constraint above ~10°, and
  outpainting (one-sided extension) is its weakest case. Researched desktop
  sidecar tiers and they verify badly — FLUX.1-Fill-dev is gated, non-commercial
  and 58 GB; sd.cpp's server has no auth and reflects Origin; Safari cannot
  reach http://127.0.0.1 from an https page at all (WebKit #171934) and Chrome
  gates it behind Local Network Access. The in-browser upgrade that DOES verify:
  `g-ronimo/lama` `lama_512_fp16.onnx` (106.6 MB, Apache-2.0, ~0.43 s/512² on
  WebGPU) for ENCLOSED disocclusion clusters, keeping MI-GAN for the one-sided
  border clusters. That is the next quality step, and it needs no desktop app.

### M8 — Layered heightfield: "indistinguishable under camera motion" (DONE)
User verdict on M7.5 with a real 12MP photo: splat confetti at angles, severe
face distortion under motion. 6-agent research pass (Kopf One-Shot 3D
Photography, Shih LDI, SLIDE/3D-Moments, Immersity/DepthFlow shader tech, FGS/
WLS depth filtering, EXIF geometry, frontier NVS) → decision: the artifacts are
STRUCTURAL to point splats; production single-photo 3D ships layered textured
representations with translation-only cameras. Full renderer cutover:
- **Renderer: two-layer full-res LDI heightfield raymarch** (fullscreen FS,
  the Immersity technique): layer 0 = photo + outpainted ring as TEXTURES
  (color at working res, disparity at ≤1.75MP depth res, R16F linear — core
  WebGL2), layer 1 = band-limited AI-filled background (feathered alpha).
  Per-pixel inverse-depth march (48 steps + 7 binary refinements; both
  parallax terms LINEAR in candidate disparity), gap detector (jump across the
  last coarse step > 0.02·range·140/N) switches to the layer-1 march;
  stretched layer-0 wall sample = never-void fallback. One bilinear tap of the
  full-res photo per pixel: photo-native sharpness, NO sorting, NO popping,
  and a heightfield CANNOT fragment — confetti is dead by construction.
  Rest pose renders the photo pixel-exactly. sort-worker/splat-build/orbit/
  .splat export deleted.
- **Camera: translation-only window camera** (window-cam.js) — no rotation
  ever (rotation converts focal error into (γ−1)·θ shear = the face-swimming
  mechanism; translation-only novel views are pixel-exact under ANY focal
  error). Subject-plane convergence (dConv; double-tap re-pivots): the face
  stays pixel-locked (e2e asserts <2px shift across the FULL envelope), the
  background does the moving. Dolly-dominant envelope (ez in +0.12/−0.05·Zs;
  dolly opens almost no holes and carries the wow — verified: dolly-in
  magnifies faces with zero distortion). Per-photo lateral envelope =
  min(0.05·boost, 3%-of-frame parallax cap, 1.75× fill-coverage cap).
- **EXIF intrinsics** (io/exif.js, exifr 7.1.3 CDN, JPEG+HEIC): render FoV =
  capture FoV. FocalLengthIn35mmFormat clamped [10,250] (iOS garbage-value
  bug), DigitalZoomRatio applied only on known base lenses, fallback
  fPx=max(W,H) (53.13° long side). Test photo: iPhone XR 26mm → 67.3° (the
  old assumed 55° was badly wrong).
- **Depth stack** (depth-filter.js) replaced joint-bilateral refine (imprints
  texture into geometry!) + 3×3 snap: (1) Fast Global Smoother — exact
  tridiagonal Thomas solves H/V, T=3, λ schedule 1.5λ4^(T−t)/(4^T−1), gradient
  conductance exp(−|dI|/7) — flattens interior noise with NO texture imprint;
  (2) Kopf 5×5 weighted median w/ edge-sample rejection (ramps → 1px steps);
  (3) floater CC merge (<20px·(min/384)² debris into largest-contact
  neighbor — kills crowd-depth speckle); (4) gated edge relocation
  (disparity-domain bilateral median at discontinuities, mutual-structure
  gate: nearby image edge with orientation agreement <30°, gradient argmax
  propagated over ~3px so offset boundaries still qualify).
  Depth res ≤1.75MP END TO END (smooth field; sharpness comes from color).
- **Mapping: subject-anchored reciprocal** Z = dSub/max(d, 0.04), Zs = 1
  (subject units); dSub = center-box median disparity. depthStrength died;
  the "3D boost" slider is a pure envelope gain, never geometry.
- **Fill machinery unchanged** (MI-GAN, classical synth at depth res, anchor,
  grain) — destination is now textures. CRITICAL lesson: the AI plate's
  INTERIOR carries fill colors at bgMask pixels — layer 0 must stamp the
  pristine photo over the interior (only the ring comes from the plate);
  using the raw plate as color0 painted fills OVER the subjects (found by
  JS-simulating the march against real worker buffers when the first browser
  render showed ghost stones).
- **Results** (12MP test photo, ultra): rest pose = the photo, exactly;
  envelope corner = clean parallax, zero tearing/voids; dolly-in = faces
  magnify undistorted. e2e rewritten for layers (pivot-lock assertion, pan/
  dolly/refocus/DoF/PNG); 39 unit tests. Perf: fullscreen march ~96 taps/px
  worst case — Immersity ships 5 layers×40 steps to phones; steps 48 desktop
  / 28 mobile.
- **Deferred to M9**: MoGe-2/DA3 geometry tier; person-matte planarization;
  gyro (needs metric anchor); Distill-Any-Depth HQ swap (license + edges);
  ring coverage for frame-edge-cut near subjects (residual stretch there);
  dolly-zoom preset; envelope-edge fade.

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

- **Fill quality is now the binding constraint** (see M9). Route enclosed
  disocclusion clusters to Big-LaMa (`g-ronimo/lama`, Apache-2.0, verified) and
  keep MI-GAN for border/outpaint clusters, which is LaMa's worst case.
- Whether a generated anchor should also carry a second layer (it currently has
  none, so a ray that falls through a silhouette in a generated view knows
  nothing there and drops to 0.25 trust).
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
