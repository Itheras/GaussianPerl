# GaussianPerl 🫧

Turn a **single photo** into an interactive **3D photo** — right on your
device. No server, no upload: local AI models run in your browser (WebGPU or
WASM) — one estimates depth, another **generatively paints the areas the camera
never saw** (behind foreground objects and beyond the frame) — and a layered
heightfield renderer replays the scene at the photo's own resolution as you
move the camera, refocus, and capture.

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
   - **Mouse**: drag = look around (parallax) · wheel = dolly
   - **Touch**: one finger = look around · pinch = dolly
   - **Double-click / double-tap**: set the focus distance (depth of field)
4. Tune **Focus** and **Aperture** for the depth-of-field look, and **3D
   boost** for how far the camera may wander, in the panel.
5. **Save PNG** captures exactly what you see (2× supersampled).

## How it works (the bag of tricks)

| Stage | Trick |
|---|---|
| Camera | The photo's own EXIF focal length; translation-only window camera (rotation is what shears faces); subject-plane convergence so the subject stays pixel-locked |
| Depth | Depth Anything V2 via transformers.js, on-device (WebGPU→WASM; *base* on desktop WebGPU) |
| Depth cleanup | Fast Global Smoother (no texture imprint) + weighted-median edge snapping + floater merge + image-edge relocation |
| Far field | Disparity tail compression: distant content becomes a coherent backdrop (real far fields don't parallax) |
| Render | Two-layer heightfield raymarch: one bilinear tap of the full-res photo per pixel — photo-native sharpness, structurally incapable of point-splat tearing |
| Hidden data | **Generative fill**: MI-GAN inpainting (28 MB ONNX, onnxruntime-web) paints the background layer revealed by parallax; low frequencies anchored to a classical estimate |
| Image borders | **Generative outpaint**: the same model extends the scene beyond the frame onto the padded layer ring |
| Motion budget | Per-photo envelope: parallax ≤ ~3% of frame and never far past the fill coverage — dolly-dominant, the production recipe |
| Focus | DoF post-pass from the marched depth, double-tap to refocus AND re-pivot |
| Instant preview | Classical fill ships the first scene immediately; the AI layers swap in when ready |
| Offline | Everything degrades gracefully: classical fill, mirrored ring, heuristic depth — no network required |

Details and per-milestone notes: [`SCRATCHPAD.md`](SCRATCHPAD.md).

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
                      depthproc, inpaint (classical), pipeline-worker
src/render/           WebGL2 two-layer heightfield raymarch renderer
src/controls/         translation-only window camera (mouse + touch)
src/io/               image load / EXIF intrinsics / PNG save
assets/               generated sample image + ground-truth depth
tools/gen-sample.mjs  regenerates the sample assets (node, no deps)
tests/                node unit tests        (node tests/run.mjs)
e2e/                  headless browser test  (node e2e/run-e2e.mjs)
```

## License

MIT
