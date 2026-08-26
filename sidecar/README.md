# GaussianPerl inference sidecar

A loopback HTTP server that owns the heavy generative model (FLUX.2 klein-4B,
Apache-2.0, ~16 GB) so the renderer never has to. This is the "guess what is in
the missing region, given the scene so far" step of the anchor loop.

```sh
# one-time
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision "diffusers==0.40.0" transformers accelerate safetensors pillow numpy huggingface_hub
uvx --from huggingface_hub hf download black-forest-labs/FLUX.2-klein-4B --include "transformer/*" "text_encoder/*" "vae/*" "tokenizer/*" "scheduler/*" "*.json"

# run — the token is GENERATED and printed as a JSON line:
#   {"event":"listening","port":8970,"token":"<random>","url":"http://127.0.0.1:8970"}
.venv/bin/python sidecar/server.py --preload --port 8970
```

Then open the app with `?sidecar=8970:<that token>`. Never use a fixed or
short token: the sidecar is reachable by every process on the machine and the
token is the only real access control (CORS is a browser courtesy). Without a
reachable sidecar the app silently uses the in-browser MI-GAN path.

`--dump DIR` writes every request's input/mask/output PNGs (your photo) to a
private directory — debugging only.

Security: 127.0.0.1 only, bearer token on every request, CORS answered only
for an explicit origin allowlist (`ALLOWED_ORIGINS` in `server.py`).
