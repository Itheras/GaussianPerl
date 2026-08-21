# GaussianPerl 🫧

Turn a **single photo** into an interactive **3D Gaussian splat** — right on your
device. No server, no upload: a local AI depth model runs in your browser
(WebGPU or WASM), the missing parallax data is synthesized on-device, and the
result renders as a real gaussian splat you can orbit, refocus, and capture.

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
   - **Mouse**: drag = orbit · right-drag / shift-drag = pan · wheel = dolly
   - **Touch**: one finger = orbit · pinch = dolly · two-finger drag = pan
   - **Double-click / double-tap**: set the focus distance (depth of field)
4. Tune **Focus** and **Aperture** for the depth-of-field look, plus depth
   strength, splat size, and field of view in the panel.
5. **Save PNG** captures exactly what you see (2× supersampled). You can also
   export a standard `.splat` file for other viewers.

## How it works (the bag of tricks)

| Stage | Trick |
|---|---|
| Depth | Depth Anything V2 (small) via transformers.js, on-device (WebGPU→WASM) |
| Depth cleanup | Joint bilateral filter guided by image color — edges snap to color edges |
| Splats | Per-pixel anisotropic gaussians oriented by depth-normals, slant-stretched |
| Silhouettes | Depth-discontinuity mask → isotropic, alpha-feathered edge splats |
| Hidden data | Layered disocclusion fill: background color+depth inpainted behind foreground silhouettes — revealed by parallax |
| Image borders | Mirror-extended fading “skirt” so the frame edge doesn’t cut hard |
| Cracks | ×4-downsampled large-splat underlayer |
| Render | Instanced EWA splatting, worker counting-sort, premultiplied back-to-front |
| Focus | DoF post-pass from per-pixel composited depth, tap-to-focus |
| Offline | Bundled sample ships ground-truth depth; heuristic depth if the model can’t load |

Details and per-milestone notes: [`SCRATCHPAD.md`](SCRATCHPAD.md).

## Project layout

```
index.html            app shell
styles.css
src/main.js           bootstrap + UI wiring
src/config.js         quality presets, tunables
src/util/             math3d, imageops (pure, node-testable)
src/pipeline/         depth-ai (transformers.js), depthproc, inpaint,
                      splat-build, pipeline-worker
src/render/           WebGL2 renderer, shaders, sort-worker
src/controls/         orbit controls (mouse + touch + inertia)
src/io/               image load / PNG save / .splat export
assets/               generated sample image + ground-truth depth
tools/gen-sample.mjs  regenerates the sample assets (node, no deps)
tests/                node unit tests        (node tests/run.mjs)
e2e/                  headless browser test  (node e2e/run-e2e.mjs)
```

## License

MIT
