# GaussianPerl 🫧

Turn a **single photo** into a **3D scene you can move through** — right on your
device. No server, no upload.

Point a free camera anywhere; whenever it comes to rest somewhere the
photograph cannot explain, the app **generates that view**: local AI paints in
what was behind the foreground and beyond the frame, gives the invented content
depth that agrees with the rest of the scene, and commits it as a permanent part
of the scene. Walk away and back and it is still there, unchanged. At the home
pose the render is the original photograph, pixel for pixel.

Built for **Mac (M-series)** and **iPhone** — everything is plain WebGL2 + ES
modules, one codebase, mouse *and* touch.

## Quick start

```sh
# any static server works; no build step, no npm install
python3 -m http.server 8000
# open http://localhost:8000  (Safari, Chrome, Edge, Firefox)
```

On iPhone: serve from your Mac (`python3 -m http.server`) and open
`http://<your-mac>.local:8000` in Safari, or host the folder anywhere static
(GitHub Pages works).

## Using it

1. **Open** a photo (button, drag-drop, or paste) — or hit **Sample**.
2. The depth model downloads once (~25 MB, cached), then everything runs locally.
3. Navigate:
   - **Mouse**: drag = orbit · shift-drag or right-drag = pan · wheel = dolly
   - **Keyboard**: W A S D / arrows = fly · Q E = down/up · Shift = faster
   - **Touch**: one finger = orbit · two fingers = pan · pinch = dolly
   - **Double-click / double-tap**: re-centre the orbit on that surface, and
     focus there
4. Move somewhere new and pause. **Auto-fill** notices what is missing and
   generates it (or press **✨ Fill view**). The panel shows how many views
   have been generated.
5. Tune **Focus**, **Aperture** and camera **Speed** in the panel. **Free roam**
   removes the camera limits; **Look mode** turns drag into rotate-in-place.
6. **Save PNG** captures exactly what you see (2× supersampled).

## How it works (the bag of tricks)

| Stage | Trick |
|---|---|
| Camera | The photo's own EXIF focal length; a 6-DoF orbit/pan/dolly/fly camera. Rotation is **exact**, not approximated: for any relative pose, the sample position in an anchor's image is *affine in that anchor's disparity*, so rotating costs the marcher nothing |
| Scene | A growing set of **anchors** — RGB-D views of the world. Anchor 0 is the photograph; the rest are generated as you explore, and persist |
| Growth | Render → find pixels nothing can explain → inpaint them into a complete synthetic photo of that viewpoint → estimate depth on it → robustly align that depth to the scene → commit. The frontier moves outward as you go |
| Depth | Depth Anything V2 via transformers.js, on-device (WebGPU→WASM; *base* on desktop WebGPU) |
| Depth cleanup | Fast Global Smoother (no texture imprint) + weighted-median edge snapping + floater merge + image-edge relocation |
| Far field | Disparity tail compression: distant content becomes a coherent backdrop (real far fields don't parallax) |
| Render | Two-layer heightfield raymarch: one bilinear tap of the full-res photo per pixel — photo-native sharpness, structurally incapable of point-splat tearing |
| Hidden data | **Generative fill**: MI-GAN inpainting (28 MB ONNX, onnxruntime-web) paints both the pre-baked background layer and every novel view the camera asks for |
| Smear detection | A hit is only trusted if the source is not being stretched: differentiating the hit condition gives a footprint scale of exactly `1/(1 − ∇disp·slope)` along the epipolar direction. It is 1 at rest for any camera, and it is what makes a silhouette's rubber sheet register as *missing data* instead of as geometry |
| Anchor blending | Order-independent front-bucket softmax: anchors that agree on the surface cross-dissolve, anything behind is occluded. Argmax would draw a seam that slides across the frame as you move |
| Depth agreement | Robust bin-de-biased affine fit in disparity, residual field extended by push-pull, so a generated view matches the scene **exactly** at known pixels and continues smoothly through the holes. A bad fit is rejected rather than baked in |
| Drift control | Per-pixel provenance: the renderer reports how much of each pixel is still the photograph, and the loop refuses to build an anchor out of earlier anchors' output |
| Motion budget | Generous and scene-aware, with a frontier rubber-band: go past what the photo can support and the camera eases back rather than committing a frame of invention |
| Focus | DoF post-pass from the marched depth, double-tap to refocus AND re-pivot |
| Instant preview | Classical fill ships the first scene immediately; the AI layers swap in when ready |
| Offline | Everything degrades gracefully: classical fill, mirrored ring, heuristic depth — no network required |

Details and per-milestone notes: [`SCRATCHPAD.md`](SCRATCHPAD.md). The verified
2026 research review, supplied-photo measurements, and camera-conditioned point
continuation plan are in
[`docs/camera-conditioned-point-continuation.md`](docs/camera-conditioned-point-continuation.md).

## The strong guesser (native sidecar)

The in-browser fill (MI-GAN, 28 MB) handles thin disocclusion bands. To invent
what a real camera move reveals — a person's far side, a crowd behind them —
the app can talk to a **local inference sidecar** that runs SDXL-inpainting
(openrail++) on Apple Silicon. Nothing leaves the machine; see
[`sidecar/README.md`](sidecar/README.md). Measured on an M2 Max: ~30 s per
fill at 768×1024, 20 steps. Without a reachable sidecar the app silently uses
the in-browser path.

## Project layout

```
index.html            app shell
styles.css
src/main.js           bootstrap + UI wiring
src/config.js         quality presets, tunables
src/util/             math3d, imageops (pure, node-testable)
src/pipeline/         depth-ai (transformers.js), inpaint-ai (MI-GAN via
                      onnxruntime-web), depth-filter (FGS/median/relocation),
                      fill-plan (masks/clusters/anchoring), layer-build,
                      depthproc, inpaint (classical), novel-view (depth
                      alignment + push-pull), expand (novel view -> anchor),
                      pipeline-worker
src/render/           WebGL2 multi-anchor heightfield raymarch renderer,
                      pose.js (the affine-in-disparity march algebra)
src/controls/         6-DoF free camera (mouse + touch + keyboard)
src/backend/          capability probing; native sidecar client
sidecar/              local inference server (Python, diffusers on MPS)
src/io/               image load / EXIF intrinsics / PNG save
assets/               generated sample image + ground-truth depth
tools/gen-sample.mjs  regenerates the sample assets (node, no deps)
tests/                node unit tests        (node tests/run.mjs)
e2e/                  headless browser test  (node e2e/run-e2e.mjs)
```

## License

MIT
