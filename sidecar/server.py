"""GaussianPerl local inference sidecar (M11).

A loopback HTTP server that owns the heavy generative model so the renderer
never has to. One job for now: semantic inpainting of a rendered novel view —
the "guess what is in the holes, given the scene so far" step of the anchor
loop — using FLUX.2 klein-4B (Apache-2.0) through diffusers on Apple Silicon.

Security: binds 127.0.0.1 only, requires a per-launch bearer token on every
request, and answers CORS only for an explicit origin allowlist. A loopback
server that runs a model is reachable by every process and every web page on
the machine; without the token any site you visit could drive your GPU.

Wire format (both directions):  [u32 LE header_len][JSON header][binary...]
  POST /v1/inpaint  body   = header{w,h,prompt?,strength?,steps?,guidance?,
                             seed?,crop_pad?} + RGBA u8[w*h*4] + mask u8[w*h]
                                                    (mask: nonzero = repaint)
                    reply  = header{w,h,ms,steps,...} + RGBA u8[w*h*4]
  GET  /v1/health   reply  = JSON {ok, model, loaded, device}

Run:  .venv/bin/python sidecar/server.py [--port N] [--token T]
The token is printed as a JSON line on stdout so a shell can read it.
"""
import argparse
import gc
import io
import json
import os
import secrets
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

MAX_SIDE = 4096
MAX_PIXELS = 16_000_000
MAX_BODY = 100 * 1024 * 1024


class BusyError(Exception):
    pass


def clamp_opts(h):
    """Bound every knob a client can turn; an authenticated client must still
    not be able to ask for a 4K frame at 500 steps."""
    import math
    o = dict(h)
    def num(k, lo, hi, default):
        try:
            v = float(o.get(k, default))
        except (TypeError, ValueError):
            v = default
        if not math.isfinite(v):
            v = default
        return min(max(v, lo), hi)
    o["steps"] = int(num("steps", 1, 50, 20))
    o["res"] = int(num("res", 256, 2048, 1024))
    o["strength"] = num("strength", 0.05, 1.0, 0.9)
    o["guidance"] = num("guidance", 0.0, 30.0, 7.0)
    o["seed"] = int(num("seed", 0, 2**31 - 1, 0))
    if o.get("crop_pad") is not None:
        o["crop_pad"] = int(num("crop_pad", 0, 1024, 0))
    for k in ("prompt", "negative", "init"):
        if k in o and not isinstance(o[k], str):
            o.pop(k)
        elif k in o:
            o[k] = o[k][:2000]
    return o


MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_PROMPT = ("a photograph, the same scene continued naturally, "
                  "consistent lighting, consistent perspective, realistic")
ALLOWED_ORIGINS = {
    "http://localhost:8944", "http://127.0.0.1:8944",
    "http://localhost:8931", "http://127.0.0.1:8931",
    "tauri://localhost", "http://tauri.localhost",
}


class Engine:
    """Lazily-loaded pipeline + a cache of prompt embeddings.

    The text encoder (Qwen3, ~8 GB bf16) is only needed to turn a prompt into
    embeddings. On unified memory "move it to CPU" frees nothing, so after the
    prompts we use are encoded it is DELETED; a new prompt reloads it. That
    keeps the resident set at transformer + VAE + activations.
    """

    def __init__(self, device):
        self.device = device
        self.pipe = None
        self.loaded = False
        self.lock = threading.Lock()
        self.embeds = {}
        self.load_s = None

    load_error = None

    def _release(self):
        gc.collect()
        if getattr(self, "torch", None) is not None and self.device == "mps":
            try:
                self.torch.mps.empty_cache()
            except Exception:
                pass

    def load(self):
        with self.lock:
            if self.loaded:
                return
            if self.load_error:
                raise RuntimeError("model failed to load earlier: " + self.load_error)
            try:
                self._load()
            except Exception as e:
                # never keep a half-loaded pipeline referenced: a retry would
                # double-allocate and take the whole machine down
                self.pipe = None
                self.load_error = f"{type(e).__name__}: {e}"
                self._release()
                print(json.dumps({"event": "load-failed", "error": self.load_error}), flush=True)
                raise

    def _load(self):
        import torch
        from diffusers import Flux2KleinInpaintPipeline
        t0 = time.time()
        self.torch = torch
        self.pipe = Flux2KleinInpaintPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16)
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self._encode(DEFAULT_PROMPT)
        self._drop_text_encoder()
        self.load_s = time.time() - t0
        self.loaded = True
        print(json.dumps({"event": "loaded", "seconds": round(self.load_s, 1)}), flush=True)

    def _encode(self, prompt):
        if prompt in self.embeds:
            return self.embeds[prompt]
        if self.pipe.text_encoder is None:
            from transformers import AutoModel
            self.pipe.text_encoder = AutoModel.from_pretrained(
                MODEL_ID, subfolder="text_encoder", torch_dtype=self.torch.bfloat16
            ).to(self.device)
        with self.torch.no_grad():
            out = self.pipe.encode_prompt(prompt=prompt, device=self.device, num_images_per_prompt=1)
        # encode_prompt returns (prompt_embeds, ...) across diffusers versions
        emb = out[0] if isinstance(out, (tuple, list)) else out
        self.embeds[prompt] = emb
        return emb

    def _drop_text_encoder(self):
        if self.pipe.text_encoder is not None:
            self.pipe.text_encoder = None
            gc.collect()
            if self.device == "mps":
                self.torch.mps.empty_cache()

    def inpaint(self, rgba, mask, w, h, opts):
        from PIL import Image
        self.load()
        prompt = opts.get("prompt") or DEFAULT_PROMPT
        with self.lock:
            emb = self._encode(prompt)
            if len(self.embeds) > 1:
                self._drop_text_encoder()
            # the model wants dimensions that are multiples of 16
            W = max(16, (w // 16) * 16)
            H = max(16, (h // 16) * 16)
            rgb = rgba.reshape(h, w, 4)[:, :, :3].copy()
            hole = mask.reshape(h, w) > 0
            # What the model SEES inside the mask matters: klein is an editing
            # model and the whole frame is a conditioning reference (diffusers
            # ref_images = [image_latents_encoded]), so a smooth seed inside the
            # mask is faithfully reproduced as a smooth blob. `init` chooses
            # the fill the masked region carries into the model.
            init = opts.get("init", "gray")
            if init == "gray":
                rgb[hole] = 128
            elif init == "black":
                rgb[hole] = 0
            elif init == "noise":
                rng = np.random.default_rng(int(opts.get("seed", 0)))
                rgb[hole] = rng.integers(0, 256, size=(int(hole.sum()), 3), dtype=np.uint8)
            elif init == "blur":
                from PIL import ImageFilter
                blurred = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(24)))
                rgb[hole] = blurred[hole]
            # "seed": leave whatever the caller sent
            img = Image.fromarray(rgb, "RGB")
            msk = Image.fromarray(np.where(hole, 255, 0).astype(np.uint8), "L")
            if (W, H) != (w, h):
                img = img.resize((W, H), Image.BICUBIC)
                msk = msk.resize((W, H), Image.NEAREST)
            seed = int(opts.get("seed", 0))
            gen = self.torch.Generator(device="cpu").manual_seed(seed)
            crop_pad = opts.get("crop_pad")
            t0 = time.time()
            with self.torch.no_grad():
                out = self.pipe(
                    prompt=None,
                    prompt_embeds=emb,
                    image=img,
                    mask_image=msk,
                    height=H, width=W,
                    strength=float(opts.get("strength", 1.0)),
                    num_inference_steps=int(opts.get("steps", 4)),
                    guidance_scale=float(opts.get("guidance", 1.0)),
                    padding_mask_crop=int(crop_pad) if crop_pad else None,
                    generator=gen,
                    output_type="pil",
                ).images[0]
            ms = (time.time() - t0) * 1000
            dump_images(img, msk, out)
            if out.size != (w, h):
                out = out.resize((w, h), Image.BICUBIC)
            res = np.asarray(out.convert("RGB"), dtype=np.uint8)
            full = rgba.reshape(h, w, 4).copy()
            # only the repaint region changes; known pixels stay byte-identical
            m = mask.reshape(h, w) > 0
            full[m, :3] = res[m]
            full[:, :, 3] = 255
            return full.reshape(-1), {"ms": round(ms), "W": W, "H": H, "init": init,
                                     "steps": int(opts.get("steps", 4))}


class SdxlEngine(Engine):
    """SDXL-inpainting: a TRUE inpainter. Its UNet takes 9 channels — noisy
    latents + the mask + the VAE-encoded image with the masked region zeroed —
    so the masked area is explicitly unknown to the model. That is the
    property klein's editing pipeline lacks: there, the whole frame is a
    reference and whatever sits inside the mask gets reproduced (measured:
    noise in, noise out). Here, grey in, beach out."""

    MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"   # openrail++
    VAE = "madebyollin/sdxl-vae-fp16-fix"                         # mit

    def _load(self):
        import torch
        from diffusers import AutoPipelineForInpainting, AutoencoderKL
        t0 = time.time()
        self.torch = torch
        vae = AutoencoderKL.from_pretrained(self.VAE, torch_dtype=torch.float16)
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            self.MODEL, vae=vae, torch_dtype=torch.float16, variant="fp16")
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.load_s = time.time() - t0
        self.loaded = True
        print(json.dumps({"event": "loaded", "model": self.MODEL,
                          "seconds": round(self.load_s, 1)}), flush=True)

    def inpaint(self, rgba, mask, w, h, opts):
        from PIL import Image
        self.load()
        prompt = opts.get("prompt") or DEFAULT_PROMPT
        negative = opts.get("negative") or ("blurry, smeared, distorted, deformed, "
                                            "extra limbs, extra heads, hat, cap, headwear, "
                                            "text, watermark, frame, border")
        # one job at a time, and a second caller learns that immediately
        # instead of queueing behind the GPU for a minute
        if not self.lock.acquire(timeout=0.5):
            raise BusyError()
        try:
            rgb = rgba.reshape(h, w, 4)[:, :, :3].copy()
            hole = mask.reshape(h, w) > 0
            init = opts.get("init", "gray")
            if init == "gray":
                rgb[hole] = 128
            elif init == "noise":
                rng = np.random.default_rng(int(opts.get("seed", 0)))
                rgb[hole] = rng.integers(0, 256, size=(int(hole.sum()), 3), dtype=np.uint8)
            # work at <= `res` on the long side (SDXL is a 1024-class model),
            # dimensions multiples of 8
            res = int(opts.get("res", 1024))
            sc = min(1.0, res / max(w, h))
            W = max(64, int(round(w * sc / 8)) * 8)
            H = max(64, int(round(h * sc / 8)) * 8)
            img = Image.fromarray(rgb, "RGB").resize((W, H), Image.BICUBIC)
            msk = Image.fromarray(np.where(hole, 255, 0).astype(np.uint8), "L").resize((W, H), Image.NEAREST)
            seed = int(opts.get("seed", 0))
            gen = self.torch.Generator(device="cpu").manual_seed(seed)
            steps = int(opts.get("steps", 20))
            t0 = time.time()
            with self.torch.no_grad():
                out = self.pipe(
                    prompt=prompt, negative_prompt=negative,
                    image=img, mask_image=msk, height=H, width=W,
                    strength=float(opts.get("strength", 0.99)),
                    num_inference_steps=steps,
                    guidance_scale=float(opts.get("guidance", 7.0)),
                    padding_mask_crop=(int(opts["crop_pad"]) if opts.get("crop_pad") else None),
                    generator=gen,
                ).images[0]
            ms = (time.time() - t0) * 1000
            dump_images(img, msk, out)
            if out.size != (w, h):
                out = out.resize((w, h), Image.BICUBIC)
            res_px = np.asarray(out.convert("RGB"), dtype=np.uint8)
            full = rgba.reshape(h, w, 4).copy()
            full[hole, :3] = res_px[hole]
            full[:, :, 3] = 255
            return full.reshape(-1), {"ms": round(ms), "W": W, "H": H, "init": init,
                                     "steps": steps, "model": "sdxl-inpaint"}
        finally:
            self.lock.release()
            self._release()


_DUMP_DIR = None
_dump_n = 0


def dump_images(img, msk, out):
    """Debug only (--dump DIR): writes the user's photo to disk, so the dir
    is created private and announced loudly at startup."""
    global _dump_n
    if not _DUMP_DIR:
        return
    _dump_n += 1
    tag = f"{time.strftime('%Y%m%d-%H%M%S')}-{_dump_n:04d}"
    try:  # a debug write must never fail a fill (the dir may have been cleared)
        os.makedirs(_DUMP_DIR, mode=0o700, exist_ok=True)
        img.save(os.path.join(_DUMP_DIR, f"{tag}-in.png"))
        msk.save(os.path.join(_DUMP_DIR, f"{tag}-mask.png"))
        out.save(os.path.join(_DUMP_DIR, f"{tag}-out.png"))
    except OSError as e:
        print(json.dumps({"event": "dump-failed", "error": str(e)}), flush=True)


def frame(header: dict, payload: bytes) -> bytes:
    hb = json.dumps(header).encode()
    return struct.pack("<I", len(hb)) + hb + payload


def unframe(body: bytes):
    (n,) = struct.unpack("<I", body[:4])
    header = json.loads(body[4:4 + n].decode())
    return header, body[4 + n:]


def make_handler(engine: Engine, token: str):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30   # an idle pre-auth connection is closed, not held forever

        def log_message(self, *a):  # quiet
            pass

        def _cors(self):
            origin = self.headers.get("Origin")
            if origin and origin in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Max-Age", "600")

        def _reject(self, code, msg):
            body = json.dumps({"error": msg}).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self):
            auth = self.headers.get("Authorization", "") or ""
            # bytes compare: a non-latin header must not raise outside try/except
            return secrets.compare_digest(auth.encode("latin-1", "replace"),
                                          f"Bearer {token}".encode())

        def do_OPTIONS(self):
            origin = self.headers.get("Origin")
            if not origin or origin not in ALLOWED_ORIGINS:
                self._reject(403, "origin not allowed")
                return
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            origin = self.headers.get("Origin")
            if origin and origin not in ALLOWED_ORIGINS:
                self._reject(403, "origin not allowed")
                return
            if self.path != "/v1/health":
                self._reject(404, "not found")
                return
            if not self._authed():
                self._reject(401, "bad token")
                return
            body = json.dumps({
                "ok": True, "model": getattr(engine, "MODEL", MODEL_ID), "loaded": engine.loaded,
                "device": engine.device, "loadSeconds": engine.load_s,
                "loadError": engine.load_error, "busy": engine.lock.locked(),
            }).encode()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self):
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin not in ALLOWED_ORIGINS:
                self._reject(403, "origin not allowed")
                return
            if not self._host_ok():
                self._reject(403, "bad host")
                return
            if not self._authed():
                self._reject(401, "bad token")
                return
            if self.path != "/v1/inpaint":
                self._reject(404, "not found")
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._reject(400, "bad length")
                return
            if n <= 4 or n > MAX_BODY:
                self._reject(413, "bad length")
                return
            try:
                # header FIRST, then exactly the bytes it promises — never a
                # bulk read of whatever the client claims
                (hn,) = struct.unpack("<I", self.rfile.read(4))
                if hn <= 0 or hn > 64 * 1024:
                    self._reject(400, "bad header")
                    return
                header = json.loads(self.rfile.read(hn).decode())
                w, h = int(header.get("w", 0)), int(header.get("h", 0))
                if not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE and w * h <= MAX_PIXELS):
                    self._reject(400, "bad dimensions")
                    return
                need = w * h * 5
                if n != 4 + hn + need:
                    self._reject(400, "payload size mismatch")
                    return
                payload = self.rfile.read(need)
                if len(payload) != need:
                    self._reject(400, "short body")
                    return
                rgba = np.frombuffer(payload[: w * h * 4], dtype=np.uint8)
                mask = np.frombuffer(payload[w * h * 4:], dtype=np.uint8)
                out, info = engine.inpaint(rgba, mask, w, h, clamp_opts(header))
                reply = frame({"w": w, "h": h, **info}, out.tobytes())
                self.send_response(200)
                self._cors()
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)
            except BusyError:
                self._reject(503, "busy")
            except Exception:  # log server-side, never echo paths/stats to clients
                import traceback
                traceback.print_exc()
                self._reject(500, "inference failed")

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("GP_SIDECAR_PORT", "0")))
    ap.add_argument("--token", default=os.environ.get("GP_SIDECAR_TOKEN") or secrets.token_urlsafe(24))
    ap.add_argument("--preload", action="store_true", help="load the model at startup")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model", default=os.environ.get("GP_SIDECAR_MODEL", "sdxl"),
                    choices=["sdxl", "klein"])
    ap.add_argument("--dump", default=os.environ.get("GP_SIDECAR_DUMP"),
                    help="DEBUG: write every request's input/mask/output PNGs here")
    args = ap.parse_args()

    if len(args.token) < 16:
        print(json.dumps({"event": "warning", "message":
                          "token shorter than 16 chars: any local process can guess it"}), flush=True)
    global _DUMP_DIR
    if args.dump:
        _DUMP_DIR = os.path.abspath(args.dump)
        os.makedirs(_DUMP_DIR, mode=0o700, exist_ok=True)
        print(json.dumps({"event": "dump-enabled", "dir": _DUMP_DIR}), flush=True)

    engine = SdxlEngine(args.device) if args.model == "sdxl" else Engine(args.device)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(engine, args.token))
    port = srv.server_address[1]
    print(json.dumps({"event": "listening", "port": port, "token": args.token,
                      "url": f"http://127.0.0.1:{port}"}), flush=True)
    if args.preload:
        threading.Thread(target=engine.load, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
