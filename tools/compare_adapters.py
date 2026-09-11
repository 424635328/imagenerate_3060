"""compare_adapters.py — 客观对比多个 LoRA adapter 的出图质量。

同一组 prompt × 同一组 seed × 同一采样参数下分别出图，记录可量化指标，并拼一张
左右对比图供人工判断：

  * sharpness   —— Laplacian 方差（细节/锐度）
  * saturation  —— HSV 饱和度均值（过饱和是本项目历史上的典型退化）
  * contrast    —— 灰度标准差
  * clip_proxy  —— 图像与 prompt 的 CLIP 相似度（需 --clip，会额外下载 ViT-L/14）

用法:
    python tools/compare_adapters.py --adapters "V4:models/v4_640/adapter_best" "V5:models/v5_lora/adapter_best" \
        --out research/compare_v5 --steps 24 --res 512
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
BASE_LOCAL = os.environ.get("BASE_MODEL", str(ROOT / "models" / "base_rv6"))

PROMPTS = [
    "a dramatic mountain valley at sunrise, mist, alpine lake, golden light",
    "a tranquil alpine lake reflecting snow-capped peaks, morning fog",
    "aurora borealis over a snowy landscape, winter night, stars, green glow",
    "a powerful waterfall plunging into a teal pool in a lush mossy gorge",
    "golden sand dunes with long shadows at sunset, rippled texture",
    "an autumn forest with golden and red maple leaves, misty path, soft light",
]
SEEDS = [101, 202]


def laplacian_variance(gray: np.ndarray) -> float:
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    h, w = gray.shape
    out = np.zeros((h - 2, w - 2), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            weight = kernel[dy, dx]
            if weight:
                out += weight * gray[dy:dy + h - 2, dx:dx + w - 2]
    return float(out.var())


def metrics(image: Image.Image) -> dict:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = array @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    maximum = array.max(axis=2)
    minimum = array.min(axis=2)
    saturation = np.where(maximum > 0, (maximum - minimum) / np.maximum(maximum, 1e-6), 0)
    return {
        "sharpness": round(laplacian_variance(gray), 2),
        "saturation": round(float(saturation.mean()), 4),
        "contrast": round(float(gray.std()), 4),
        "mean_luma": round(float(gray.mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True, help='labels like "V4:models/v4_640/adapter_best"')
    ap.add_argument("--out", default=str(ROOT / "research" / "compare_v5"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--sampler", default="dpmpp2m_karras")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import StableDiffusionPipeline
    print(f"pipeline base: {BASE_LOCAL}")
    pipe = StableDiffusionPipeline.from_pretrained(BASE_LOCAL, torch_dtype=torch.float16,
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    if args.sampler == "dpmpp2m_karras":
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,
                                                                 use_karras_sigmas=True, algorithm_type="dpmsolver++")
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    rows = []
    grids = []
    for spec in args.adapters:
        label, _, path = spec.partition(":")
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        if path:
            pipe.load_lora_weights(path)
            print(f"[{label}] loaded {path}")
        else:
            print(f"[{label}] base model (no adapter)")
        images = []
        for prompt in PROMPTS:
            for seed in SEEDS:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.cfg,
                             width=args.res, height=args.res, generator=generator).images[0]
                stat = metrics(image)
                row = {"adapter": label, "prompt": prompt[:46], "seed": seed, **stat}
                rows.append(row)
                images.append(image)
                print(f"  [{label}] seed {seed} sharpness {stat['sharpness']:8.1f} "
                      f"sat {stat['saturation']:.3f} contrast {stat['contrast']:.3f}")
        grids.append((label, images))

    # side-by-side sheet: rows = prompt×seed, columns = adapters
    tile = 256
    cols = len(grids)
    rows_n = len(grids[0][1])
    sheet = Image.new("RGB", (tile * cols, tile * rows_n), "#0b1020")
    for column, (_, images) in enumerate(grids):
        for row_index, image in enumerate(images):
            sheet.paste(image.resize((tile, tile), Image.LANCZOS), (column * tile, row_index * tile))
    draw = ImageDraw.Draw(sheet)
    for column, (label, _) in enumerate(grids):
        draw.rectangle([column * tile, 0, column * tile + 90, 18], fill="#000000cc")
        draw.text((column * tile + 4, 4), label, fill="#ffffff")
    sheet_path = out_dir / "compare_sheet.png"
    sheet.save(sheet_path)

    csv_path = out_dir / "compare_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== mean metrics (higher sharpness = more detail) ===")
    for label, _ in grids:
        subset = [r for r in rows if r["adapter"] == label]
        mean_sharp = sum(r["sharpness"] for r in subset) / len(subset)
        mean_sat = sum(r["saturation"] for r in subset) / len(subset)
        mean_contrast = sum(r["contrast"] for r in subset) / len(subset)
        print(f"  {label:8s} sharpness {mean_sharp:8.1f}  saturation {mean_sat:.4f}  contrast {mean_contrast:.4f}")
    print(f"\nsheet: {sheet_path}\ncsv:   {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
