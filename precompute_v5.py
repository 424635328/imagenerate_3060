"""precompute_v5.py — latent/text cache builder for both SD1.5 and SDXL.

Upgrades over the V4 builder (`archive/v4/precompute_v4.py`):
  * architecture-aware: SDXL (dual text encoders → prompt_embeds 77x2048 + pooled
    1280) or SD1.5 (single CLIP-L), with the matching VAE scaling factor;
  * SDXL text embeddings are cached, so training never keeps the 2 text encoders
    in VRAM (that is what makes SDXL LoRA fit in 6 GB);
  * K random square crops per source image + hflip (same augmentation as V4), so
    the cache format stays compatible with the trainer.

Output cache keys: latents, text_ids, texts, caps, test_latents, test_texts,
empty_text, res, crops (+ prompt_embeds / pooled_embeds for SDXL).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
LANCZOS = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.Resampling.LANCZOS
SCALING = {"sd15": 0.18215, "sdxl": 0.13025}


def rand_crop(im, size, scale=(0.55, 1.0), ratio=(0.75, 1.34)):
    w, h = im.size
    for _ in range(12):
        area = w * h * random.uniform(*scale)
        ar = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw, ch = int(round(math.sqrt(area * ar))), int(round(math.sqrt(area / ar)))
        if 8 <= cw <= w and 8 <= ch <= h:
            x, y = random.randint(0, w - cw), random.randint(0, h - ch)
            return im.crop((x, y, x + cw, y + ch)).resize((size, size), LANCZOS)
    side = min(w, h)
    x, y = (w - side) // 2, (h - side) // 2
    return im.crop((x, y, x + side, y + side)).resize((size, size), LANCZOS)


def center(im, size):
    w, h = im.size
    side = min(w, h)
    x, y = (w - side) // 2, (h - side) // 2
    return im.crop((x, y, x + side, y + side)).resize((size, size), LANCZOS)


@torch.no_grad()
def to_latent(im, vae, scaling: float, device) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    tensor = ((tensor - 0.5) / 0.5).unsqueeze(0).to(device).float()
    return (vae.encode(tensor).latent_dist.sample() * scaling).squeeze(0).half().cpu()


@torch.no_grad()
def sd15_embed(cap, tokenizer, text_encoder, device):
    tokens = tokenizer(cap, padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="pt").to(device)
    return text_encoder(tokens.input_ids).last_hidden_state.squeeze(0).half().cpu(), None


@torch.no_grad()
def sdxl_embed(cap, tok, tok2, te, te2, device):
    """Returns (prompt_embeds[77,2048], pooled[1280]) exactly as SDXL expects."""
    t1 = tok(cap, padding="max_length", max_length=tok.model_max_length, truncation=True,
             return_tensors="pt").to(device)
    t2 = tok2(cap, padding="max_length", max_length=tok2.model_max_length, truncation=True,
              return_tensors="pt").to(device)
    out1 = te(t1.input_ids, output_hidden_states=True)
    out2 = te2(t2.input_ids, output_hidden_states=True)
    # diffusers SDXL convention: penultimate hidden state of encoder 1, concat with encoder 2
    prompt = torch.cat([out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1)
    pooled = out2.text_embeds
    return prompt.squeeze(0).half().cpu(), pooled.squeeze(0).half().cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "dataset1024"))
    ap.add_argument("--out", default=str(ROOT / "dataset1024" / "cache_v5_640.pt"))
    ap.add_argument("--base", default="SG161222/Realistic_Vision_V6.0_B1_noVAE")
    ap.add_argument("--arch", default="auto", choices=["auto", "sd15", "sdxl"])
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N train images")
    ap.add_argument("--test-limit", type=int, default=0, help="debug: only N test images (validation cache)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu is for shape/convention validation without touching the GPU")
    ap.add_argument("--cache_dir", default=str(ROOT / "models" / "hf_cache"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", args.cache_dir + "/hub")

    from diffusers import AutoencoderKL
    device = args.device
    arch = args.arch
    # 基座解析到本地目录并全程离线：HF Hub id 会让 transformers/diffusers 联网取文件，
    # 在代理环境下会以 SSL 错误失败（2026-09-13 SDXL 主线被误判中止的根因）。
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    from sdxl_base import offline_kwargs, resolve_base
    resolved = resolve_base(args.base)
    if resolved != args.base:
        print(f"base resolved: {args.base} -> {resolved}")
    args.base = resolved
    offline = offline_kwargs(args.base)
    if arch == "auto":
        if Path(args.base).is_dir():
            index = json.loads((Path(args.base) / "model_index.json").read_text(encoding="utf-8"))
        else:
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(args.base, cache_dir=args.cache_dir + "/hub",
                                          allow_patterns=["model_index.json", "*.json"]))
            index = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
        arch = "sdxl" if "text_encoder_2" in index else "sd15"
    print(f"arch={arch} res={args.res} crops={args.crops} base={args.base} device={device}")
    # fp16 kernels are patchy on CPU; the validation path stays fp32
    dtype = torch.float16 if device == "cuda" else torch.float32

    if arch == "sdxl":
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
        tok = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer", **offline)
        tok2 = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer_2", **offline)
        te = CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder",
                                           torch_dtype=dtype, **offline).to(device).eval()
        te2 = CLIPTextModelWithProjection.from_pretrained(args.base, subfolder="text_encoder_2",
                                                          torch_dtype=dtype, **offline).to(device).eval()
        embed = lambda cap: sdxl_embed(cap, tok, tok2, te, te2, device)   # noqa: E731
    else:
        from transformers import CLIPTokenizer, CLIPTextModel
        tok = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer", **offline)
        te = CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder",
                                           torch_dtype=dtype, **offline).to(device).eval()
        embed = lambda cap: sd15_embed(cap, tok, te, device)              # noqa: E731

    vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae", torch_dtype=torch.float32,
                                        **offline)
    vae = vae.to(device).eval()
    scaling = SCALING[arch]

    manifest = json.loads((Path(args.data_dir) / "manifest.json").read_text(encoding="utf-8"))
    train_items = [m for m in manifest if m["split"] == "train"]
    test_items = [m for m in manifest if m["split"] == "test"]
    if args.limit:
        train_items = train_items[: args.limit]
    if args.test_limit:
        test_items = test_items[: args.test_limit]
    print(f"train={len(train_items)} test={len(test_items)}")

    texts, caps, latents, text_ids, prompts, pooled = [], [], [], [], [], []
    started = time.time()
    data_dir = Path(args.data_dir)

    def item_path(entry: dict) -> Path:
        """manifest 里的 file 可以是绝对路径，也可以是相对 --data_dir 的相对路径
        （后者才是可移植的写法：协作者换机器无需改 manifest）。"""
        raw = Path(entry["file"])
        return raw if raw.is_absolute() else data_dir / raw

    for index, item in enumerate(train_items):
        cap = item.get("caption") or "a scenic landscape"
        image = Image.open(item_path(item)).convert("RGB")
        ti = len(texts)
        prompt_vec, pooled_vec = embed(cap)
        texts.append(prompt_vec)                    # SD1.5: [77,768]; SDXL: [77,2048]
        if arch == "sdxl":
            pooled.append(pooled_vec)               # [1280]
        caps.append(cap)
        for _ in range(args.crops):
            crop = rand_crop(image, args.res)
            if random.random() < 0.5:
                crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
            latents.append(to_latent(crop, vae, scaling, device))
            text_ids.append(ti)
        if (index + 1) % 200 == 0:
            print(f"  {index + 1}/{len(train_items)} latents={len(latents)} {time.time() - started:.0f}s", flush=True)

    test_lat, test_txt, test_pooled = [], [], []
    for item in test_items:
        image = Image.open(item_path(item)).convert("RGB")
        test_lat.append(to_latent(center(image, args.res), vae, scaling, device))
        prompt_vec, pooled_vec = embed(item.get("caption") or "a scenic landscape")
        test_txt.append(prompt_vec)
        if arch == "sdxl":
            test_pooled.append(pooled_vec)
    empty_prompt, empty_pooled = embed("")

    payload = {
        "latents": latents, "text_ids": text_ids, "texts": texts, "caps": caps,
        "test_latents": test_lat, "test_texts": test_txt,
        "empty_text": empty_prompt, "res": args.res, "crops": args.crops, "arch": arch,
    }
    if arch == "sdxl":
        payload["prompt_embeds"] = texts            # [N, 77, 2048] fp16, indexed like text_ids
        payload["pooled_embeds"] = pooled           # [N, 1280] fp16
        payload["test_pooled"] = test_pooled
        payload["empty_pooled"] = empty_pooled
    del prompts  # texts already holds the SDXL prompt embeddings
    torch.save(payload, args.out)
    size_mb = Path(args.out).stat().st_size / 1024 ** 2
    print(f"saved {args.out} ({size_mb:.0f} MB) train_latents={len(latents)} texts={len(texts)} test={len(test_lat)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
