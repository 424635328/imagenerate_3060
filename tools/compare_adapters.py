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
import gc
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
        # 0–255 量纲：与常见的 Laplacian-variance 清晰度指标一致，否则数值小到看不出差别
        "sharpness": round(laplacian_variance(gray * 255.0), 2),
        "saturation": round(float(saturation.mean()), 4),
        "contrast": round(float(gray.std() * 255.0), 4),
        "mean_luma": round(float(gray.mean() * 255.0), 4),
    }


# ------------------------------------------------------------ adapter loading

def load_adapter_into(pipe, adapter_dir: str | None):
    """把 adapter 装进 pipeline —— **必须与线上 app.py 同路径**。

    为什么不用 `pipe.load_lora_weights(dir)`：我们的 adapter 是 peft 格式
    （`adapter_config.json` + `adapter_model.safetensors`），diffusers 的 LoRA 加载器
    找不到 `unet.` 前缀的键，会打印 "No LoRA keys associated ... This is safe to ignore"
    然后**什么都不加载**——出图与基座完全一致（实测踩过：两个模型指标一模一样）。
    线上用的是 peft：`get_peft_model` + `set_peft_model_state_dict`。
    """
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    unet = UNet2DConditionModel.from_pretrained(BASE_LOCAL, subfolder="unet", torch_dtype=torch.float16)
    if adapter_dir:
        config = LoraConfig.from_pretrained(adapter_dir)
        unet = get_peft_model(unet, config)
        set_peft_model_state_dict(unet, load_file(Path(adapter_dir) / "adapter_model.safetensors"))
    # 留在 CPU：由调用方决定"常驻显存"还是"CPU offload"。
    # 顺序很关键 —— diffusers 的 offload 是给**当前**模块挂 hook，先 offload 再换 unet 会让 hook 失效。
    pipe.unet = unet.eval()
    return adapter_dir is not None


def place_pipeline(pipe, force_offload: bool, min_free_gb: float) -> None:
    """按当前可用显存决定把管线放哪里（AGENTS.md 规范 §4：显存不足要自适应而不是硬上）。"""
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
    if force_offload or free_gb < min_free_gb:
        pipe.enable_model_cpu_offload()      # offload 要求所有组件在 CPU，由 hook 按需搬
        print(f"[mem] 可用 {free_gb:.2f} GB < {min_free_gb} GB → CPU offload（慢但不会卡死）", flush=True)
    else:
        pipe.to("cuda")                      # 整条管线迁移；只移 unet 会导致文本编码器设备不匹配
        pipe.enable_attention_slicing()
        print(f"[mem] 可用 {free_gb:.2f} GB → 常驻显存", flush=True)


def release_pipeline(pipe) -> None:
    """换 adapter 前把上一份 UNet 与 hook 彻底放掉，否则 8 个 adapter 会累积占满 6GB。"""
    try:
        pipe.remove_all_hooks()
    except Exception:                                        # noqa: BLE001
        pass
    del pipe.unet
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- CLIP score

CLIP_REPO = os.environ.get("CLIP_MODEL", "openai/clip-vit-large-patch14")


def load_clip():
    """CLIP ViT-L/14 for text-image alignment.  Returns (model, processor, device)
    or (None, None, None) when it cannot be fetched (offline / no disk).

    先试 `local_files_only=True`：模型已在 HF 缓存里时，联网做元数据检查只会带来
    SSL 重试与几十秒到几分钟的卡顿（实测踩过），离线加载更稳。
    """
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        print("[clip] transformers missing")
        return None, None, None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for local_only in (True, False):
        try:
            model = CLIPModel.from_pretrained(CLIP_REPO, torch_dtype=torch.float16,
                                              local_files_only=local_only).to(device).eval()
            processor = CLIPProcessor.from_pretrained(CLIP_REPO, local_files_only=local_only)
            print(f"[clip] {CLIP_REPO} on {device}" + ("" if local_only else " (downloaded)"))
            return model, processor, device
        except Exception as error:
            if local_only:
                continue          # 未缓存 → 再走一次在线加载
            print(f"[clip] load failed: {type(error).__name__}: {str(error)[:120]}")
    return None, None, None


_BASE_TEXT_ENCODER = None


def apply_text_encoder(pipe, adapter_path: str) -> bool:
    """Load the adapter's fine-tuned text encoder when it exists, else restore the base one.

    V4 的线上形态是「UNet LoRA + 微调 CLIP」（app.py 会加载 `<adapter>_text_encoder.pt`）。
    比较部署形态时若不还原这个差异，就等于把 V4 的手绑起来跟 V5 打。
    """
    global _BASE_TEXT_ENCODER
    if _BASE_TEXT_ENCODER is None:
        _BASE_TEXT_ENCODER = {k: v.detach().clone() for k, v in pipe.text_encoder.state_dict().items()}
    tuned = Path(str(adapter_path) + "_text_encoder.pt") if adapter_path else None
    if tuned and tuned.exists():
        pipe.text_encoder.load_state_dict(torch.load(str(tuned), map_location="cpu"))
        return True
    pipe.text_encoder.load_state_dict(_BASE_TEXT_ENCODER)
    return False


@torch.no_grad()
def clip_similarity(model, processor, image: Image.Image, prompt: str, device: str) -> float:
    """Cosine similarity between the image and its prompt embedding."""
    inputs = processor(text=[prompt], images=[image], return_tensors="pt",
                       padding=True, truncation=True).to(device)
    output = model(**inputs)
    image_embed = output.image_embeds / output.image_embeds.norm(dim=-1, keepdim=True)
    text_embed = output.text_embeds / output.text_embeds.norm(dim=-1, keepdim=True)
    return float((image_embed * text_embed).sum(dim=-1).item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True, help='labels like "V4:models/v4_640/adapter_best"')
    ap.add_argument("--out", default=str(ROOT / "research" / "compare_v5"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--sampler", default="dpmpp2m_karras")
    ap.add_argument("--no-clip", action="store_true",
                    help="skip the CLIP text-image alignment score (avoids a ~1.7 GB one-time download)")
    ap.add_argument("--offload", action="store_true", help="强制 CPU offload（最省显存，速度慢）")
    ap.add_argument("--min-free-gb", type=float, default=3.5,
                    help="低于该可用显存自动切 CPU offload；避免 WDDM 抖动导致 0 产出")
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
    # 显存自适应（AGENTS.md 规范 §4）：VAE 分块+切片常开 —— VAE 解码是最大的瞬时尖峰。
    # 具体"常驻显存 / CPU offload"的决策放在每个 adapter 装载之后（见 place_pipeline）。
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)

    rows = []
    grids = []
    for spec in args.adapters:
        label, _, path = spec.partition(":")
        load_adapter_into(pipe, path or None)
        if path:
            used_tuned_te = apply_text_encoder(pipe, path)
            print(f"[{label}] loaded {path} via peft"
                  + ("  (+ fine-tuned text encoder)" if used_tuned_te else "  (base text encoder)"))
        else:
            apply_text_encoder(pipe, "")
            print(f"[{label}] base model (no adapter)")
        # 装载 adapter 之后再决定放显存还是 offload（hook 必须挂在当前模块上）
        place_pipeline(pipe, args.offload, args.min_free_gb)
        images = []
        for prompt in PROMPTS:
            for seed in SEEDS:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.cfg,
                             width=args.res, height=args.res, generator=generator).images[0]
                stat = metrics(image)
                row = {"adapter": label, "prompt": prompt[:46], "seed": seed,
                       "_image": image, "_prompt": prompt, **stat}
                rows.append(row)
                images.append(image)
                print(f"  [{label}] seed {seed} sharpness {stat['sharpness']:8.1f} "
                      f"sat {stat['saturation']:.3f} contrast {stat['contrast']:.3f}", flush=True)
        grids.append((label, images))
        release_pipeline(pipe)          # 释放本轮的 UNet/hook，避免多 adapter 累积占满显存

    # ---- CLIP text-image alignment (prompt 跟随度) ----
    # Generated images live in RAM (a few MB), so the SD pipeline can be released
    # before CLIP loads — the 6 GB card never has to hold both.
    if not args.no_clip:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        model, processor, device = load_clip()
        if model is None:
            print("[clip] unavailable — skipping prompt-alignment score")
        else:
            for row in rows:
                row["clip_score"] = round(clip_similarity(model, processor, row["_image"],
                                                          row["_prompt"], device), 4)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    for row in rows:
        row.pop("_image", None)
        row.pop("_prompt", None)

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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    has_clip = "clip_score" in rows[0]
    print("\n=== mean metrics (sharpness ↑ 细节 / clip_score ↑ prompt 跟随) ===")
    print(f"  {'adapter':10s} {'sharpness':>10s} {'saturation':>11s} {'contrast':>9s}"
          + (f" {'clip_score':>11s}" if has_clip else ""))
    for label, _ in grids:
        subset = [r for r in rows if r["adapter"] == label]
        mean_sharp = sum(r["sharpness"] for r in subset) / len(subset)
        mean_sat = sum(r["saturation"] for r in subset) / len(subset)
        mean_contrast = sum(r["contrast"] for r in subset) / len(subset)
        line = f"  {label:10s} {mean_sharp:10.1f} {mean_sat:11.4f} {mean_contrast:9.4f}"
        if has_clip:
            line += f" {sum(r['clip_score'] for r in subset) / len(subset):11.4f}"
        print(line)
    print(f"\nsheet: {sheet_path}\ncsv:   {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
