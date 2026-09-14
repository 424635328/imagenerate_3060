"""eval_fid.py — 用**留出 prompt + 分布距离**判断"画质是否真的更好"，而不是训练损失。

为什么需要它（2026-09-13 的方法论修正）：固定协议 val（去噪 MSE）显示 V5b 0.18583 < V5 0.18589
< V4 0.18644，配对检验 t=17 也"显著"。但那是**在训练同一数据集的 test split 上**测的：
  * 效应量只有 0.3%，而画质指标（锐度/CLIP）根本区分不开；
  * 训练步数越多这个数必然越低 —— 它奖励的是"拟合得更狠"，未必是"画得更好"。
所以判定质量必须换判据。本工具用三条**与训练目标无关**的证据：

  1. **KID / CLIP-FID**：生成图与**真实风景照片**（dataset1024/test 的留出集）在 CLIP 特征空间里的
     分布距离（KID 是无偏 MMD，小样本下比 FID 可靠；FID 因协方差秩不足只作参考）；
     ↓ 越低越好，衡量"像不像这个领域的真实照片"。
  2. **留出 prompt 的 CLIP 一致性**：用**自拟的 24 条提示词**（不在数据集 caption 里）测 prompt 跟随度，
     衡量泛化而不是记忆；↑ 越高越好。
  3. **细节量/锐度**：Laplacian 方差（0-255 量纲），衡量是否只是"更平滑"。

所有候选使用**完全相同的 prompt、种子、步数、CFG、分辨率、采样器**，因此可以直接横向比较。

用法:
    python tools/eval_fid.py --adapters "BASE:" "V4:models/v4_640/adapter_best" \
        "V5:models/v5_lora/adapter_best" "V5b:models/v5b_lora/adapter_best" \
        --seeds 2 --steps 24 --res 512 --out research/fid
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_adapters import (BASE_LOCAL, ROOT as _ROOT, load_clip,  # noqa: E402
                              load_adapter_into, apply_text_encoder, metrics,
                              place_pipeline, release_pipeline)

# 留出提示词：全部**自拟**，与数据集 BLIP caption 无关（覆盖本项目的风格族）
HELD_OUT_PROMPTS = [
    "a snow-capped mountain range reflected in a still alpine lake at dawn, ultra detailed",
    "dense pine forest in heavy fog, soft god rays, moody green tones",
    "golden desert dunes with long shadows at sunset, minimal composition",
    "aurora borealis over a frozen lake, stars, deep blue and green, long exposure",
    "autumn valley with a winding river, warm red and orange foliage, aerial view",
    "dramatic sea cliffs at stormy sunset, crashing waves, spray and mist",
    "terraced rice fields on a hillside in morning mist, southeast asia",
    "lavender field under a dramatic summer sky, distant farmhouse",
    "winter village at dusk, warm window lights, falling snow, cozy atmosphere",
    "volcanic landscape with black sand beach and turquoise water, aerial",
    "bamboo grove path with dappled sunlight, serene and quiet",
    "canyon river winding through red rock walls, midday sun, wide angle",
    "misty hills layered in blue haze at sunrise, minimalist ink painting feel",
    "tropical island with palm trees and clear shallow water, bright sky",
    "rolling green hills with a lone oak tree under cumulus clouds",
    "glacier lagoon with floating icebergs, cold blue light, overcast",
    "waterfall in a lush rainforest, moss covered rocks, long exposure",
    "savanna grassland at golden hour with acacia trees and distant mountains",
    "coastal road along cliffs with ocean spray, late afternoon light",
    "snowy forest clearing with animal tracks, soft winter light",
    "salt flats with mirror reflection of the sky at sunset, ultra wide",
    "cherry blossom trees along a quiet riverbank, petals on the water",
    "dramatic thunderstorm over open plains, lightning in the distance",
    "ancient stone bridge in a misty valley, ivy covered, soft morning light",
]


def kid_mmd(real: np.ndarray, fake: np.ndarray, subsets: int = 100, seed: int = 1234) -> tuple[float, float]:
    """KID：多项式核的无偏 MMD 估计（小样本比 FID 稳），返回 (均值, 标准差)。"""
    dimension = real.shape[1]
    size = min(len(real), len(fake), 50)
    generator = np.random.default_rng(seed)

    def polynomial(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a @ b.T / dimension + 1.0) ** 3

    values = []
    for _ in range(subsets):
        pick_r = generator.choice(len(real), size, replace=False)
        pick_f = generator.choice(len(fake), size, replace=False)
        r, f = real[pick_r], fake[pick_f]
        k_rr, k_ff, k_rf = polynomial(r, r), polynomial(f, f), polynomial(r, f)
        count = size
        term = ((k_rr.sum() - np.trace(k_rr)) / (count * (count - 1))
                + (k_ff.sum() - np.trace(k_ff)) / (count * (count - 1))
                - 2 * k_rf.mean())
        values.append(float(term))
    return float(np.mean(values)), float(np.std(values))


def frechet(real: np.ndarray, fake: np.ndarray) -> float:
    """CLIP-FID（参考值）：协方差秩不足时数值不稳，仅作对照。"""
    mu_r, mu_f = real.mean(0), fake.mean(0)
    cov_r = np.cov(real, rowvar=False) + np.eye(real.shape[1]) * 1e-6
    cov_f = np.cov(fake, rowvar=False) + np.eye(fake.shape[1]) * 1e-6
    difference = mu_r - mu_f
    from scipy.linalg import sqrtm
    covmean = sqrtm(cov_r @ cov_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(difference @ difference + np.trace(cov_r + cov_f - 2 * covmean))


def image_features(model, inputs) -> torch.Tensor:
    """取图像嵌入，兼容两种 transformers 返回形态。

    transformers 5.x 的 `get_image_features` 返回 `BaseModelOutputWithPooling` 而不是裸张量
    （2026-09-14 实测：`AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'float'`）。
    裸张量、`image_embeds`、`pooler_output` 三种都接住，避免"换个版本就崩"。
    """
    output = model.get_image_features(**inputs)
    if isinstance(output, torch.Tensor):
        return output
    for attribute in ("image_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, attribute, None)
        if isinstance(value, torch.Tensor):
            if attribute == "last_hidden_state":            # 兜底：对序列维取均值
                return value.mean(dim=1)
            return value
    raise TypeError(f"无法从 {type(output).__name__} 取到图像嵌入")


def embed_images(model, processor, images, device: str) -> np.ndarray:
    vectors = []
    with torch.no_grad():
        for start in range(0, len(images), 16):
            batch = images[start:start + 16]
            inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            vectors.append(image_features(model, inputs).float().cpu().numpy())
    return np.concatenate(vectors, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "research" / "fid"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=2, help="每条 prompt 出几张")
    ap.add_argument("--prompts", type=int, default=0, help="只用前 N 条提示词（0=全部）")
    ap.add_argument("--sampler", default="dpmpp2m_karras")
    ap.add_argument("--real-limit", type=int, default=144, help="真实参考图数量（test split）")
    ap.add_argument("--arch", default="sd15", choices=["sd15", "sdxl"],
                    help="sdxl 时走 SDXL 推理路径（RealVisXL 基座 + 可选 LoRA），用于跨基座对比")
    ap.add_argument("--sdxl-base", default=os.environ.get("SDXL_BASE", "SG161222/RealVisXL_V5.0"))
    ap.add_argument("--mode", default="offload", choices=["offload", "int8", "fp16"],
                    help="SDXL 加载策略（offload 最省显存）")
    ap.add_argument("--lightning", action="store_true",
                    help="SDXL 走 SDXL-Lightning 4 步 LoRA（guidance=0）：服务形态，且显著缩短边缘停留时间")
    ap.add_argument("--offload", action="store_true", help="强制 CPU offload（最省显存，速度慢）")
    ap.add_argument("--min-free-gb", type=float, default=3.5,
                    help="低于该可用显存自动切 CPU offload，避免 WDDM 抖动导致 0 张产出")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = HELD_OUT_PROMPTS[: args.prompts] if args.prompts else HELD_OUT_PROMPTS
    seeds = [101 + 7 * index for index in range(args.seeds)]

    from PIL import Image
    real_files = sorted((ROOT / "dataset1024" / "test").glob("*.jpg"))[: args.real_limit]
    if not real_files:
        print("[错误] 找不到 dataset1024/test 的真实照片，无法计算分布距离")
        return 2
    real_images = [Image.open(path).convert("RGB").resize((args.res, args.res), Image.LANCZOS)
                   for path in real_files]
    print(f"提示词 {len(prompts)} 条 × 种子 {len(seeds)} = 每个候选 {len(prompts) * len(seeds)} 张；"
          f"真实参考 {len(real_images)} 张；{args.res}px/{args.steps} 步")

    if args.arch == "sdxl":
        # SDXL 路线只在**推理侧**可行（训练实测 125+ 小时，见 docs/TRAINING.md §10.3），
        # 但"换基座"是本机唯一还有真实质量空间的动作，所以判据必须也覆盖它。
        from compare_sdxl import attach_adapter, build_pipe, resolve_base
        resolved = resolve_base(args.sdxl_base)
        print(f"SDXL 基座: {resolved}\n策略: {args.mode}")
        pipe = build_pipe(resolved, args.mode)
        if args.lightning:
            lightning = ROOT / "models" / "lightning" / "sdxl_lightning_4step_lora.safetensors"
            if not lightning.exists():
                print(f"[错误] 找不到 {lightning}（`python tools/fetch_models.py lightning`）")
                return 2
            pipe.load_lora_weights(str(lightning))
            args.steps = 4
            args.cfg = 0.0
            print("Lightning 4 步 LoRA 已加载：steps=4, guidance=0")
    else:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        pipe = StableDiffusionPipeline.from_pretrained(BASE_LOCAL, torch_dtype=torch.float16,
                                                       safety_checker=None, requires_safety_checker=False)
        if args.sampler == "dpmpp2m_karras":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        # 与 compare_adapters 同一套显存自适应：VAE 分块/切片常开，余量不足切 CPU offload。
        # 6GB 共享卡上 `pipe.to("cuda")` + VAE 解码尖峰会把显存顶穿 → WDDM 抖动 → 0 张产出且不报错。
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
        if args.offload or free_gb < args.min_free_gb:
            pipe.enable_model_cpu_offload()
            print(f"[mem] 可用 {free_gb:.2f} GB < {args.min_free_gb} GB → CPU offload", flush=True)
        else:
            pipe.to("cuda")
            pipe.enable_attention_slicing()
            print(f"[mem] 可用 {free_gb:.2f} GB → 常驻显存", flush=True)
    pipe.set_progress_bar_config(disable=True)

    rows = []
    generated: dict[str, list] = {}
    for spec in args.adapters:
        label, _, path = spec.partition(":")
        if args.arch == "sdxl":
            # SDXL 没有"微调过的文本编码器文件"，也不该走 SD1.5 的 peft 装载路径
            if path:
                attach_adapter(pipe, path)
            used_te = False
        else:
            load_adapter_into(pipe, path or None)
            used_te = apply_text_encoder(pipe, path) if path else apply_text_encoder(pipe, "")
            # adapter 装载完成后再决定放显存还是 offload（hook 必须挂在当前模块上）
            place_pipeline(pipe, args.offload, args.min_free_gb)
        images, sharpness, alignments = [], [], []
        for prompt in prompts:
            for seed in seeds:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.cfg,
                             width=args.res, height=args.res, generator=generator).images[0]
                images.append(image)
                sharpness.append(metrics(image)["sharpness"])
                alignments.append((image, prompt))
        folder = out_dir / label.replace("+", "_")
        folder.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images):
            image.save(folder / f"{index:04d}.png")
        generated[label] = images
        rows.append({"adapter": label, "path": path or "(base)", "images": len(images),
                     "sharpness": round(float(np.mean(sharpness)), 2),
                     "te": "fine-tuned" if used_te else "base",
                     "_alignment": alignments})
        print(f"  [{label}] 生成 {len(images)} 张，锐度 {np.mean(sharpness):.1f}"
              f"{'（含微调 TE）' if used_te else ''}", flush=True)
        release_pipeline(pipe)          # 换下一个候选前释放 UNet 与 hook，避免累积占满显存

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    model, processor, device = load_clip()
    if model is None:
        print("[错误] CLIP 不可用，无法计算分布距离与一致性")
        return 2
    real_features = embed_images(model, processor, real_images, device)
    real_features /= np.linalg.norm(real_features, axis=1, keepdims=True) + 1e-8

    for row in rows:
        images = generated[row["adapter"]]
        features = embed_images(model, processor, images, device)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        kid_mean, kid_std = kid_mmd(real_features, features)
        row["kid"] = round(kid_mean, 6)
        row["kid_std"] = round(kid_std, 6)
        try:
            row["clip_fid"] = round(frechet(real_features, features), 4)
        except Exception:                                   # noqa: BLE001
            row["clip_fid"] = ""
        scores = []
        for image, prompt in row["_alignment"]:
            inputs = processor(text=[prompt], images=[image], return_tensors="pt",
                               padding=True, truncation=True).to(device)
            output = model(**inputs)
            image_embed = output.image_embeds / output.image_embeds.norm(dim=-1, keepdim=True)
            text_embed = output.text_embeds / output.text_embeds.norm(dim=-1, keepdim=True)
            scores.append(float((image_embed * text_embed).sum(dim=-1).item()))
        row["clip_score"] = round(float(np.mean(scores)), 4)
        del row["_alignment"]

    del model
    gc.collect()
    torch.cuda.empty_cache()

    csv_path = out_dir / "fid_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "fid_metrics.json").write_text(
        json.dumps({"prompts": prompts, "seeds": seeds, "steps": args.steps,
                    "cfg": args.cfg, "res": args.res, "real_images": len(real_images),
                    "results": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8", newline="\n")

    print("\n=== 与训练目标无关的三条判据（越低/越高见列头）===")
    print(f"  {'候选':14s} {'KID ↓':>10s} {'CLIP-FID↓':>11s} {'CLIP一致性↑':>12s} {'锐度↑':>9s}  TE")
    for row in sorted(rows, key=lambda r: r["kid"]):
        print(f"  {row['adapter']:14s} {row['kid']:10.6f} {str(row['clip_fid']):>11s} "
              f"{row['clip_score']:12.4f} {row['sharpness']:9.1f}  {row['te']}")
    print(f"\ncsv:   {csv_path}\njson:  {out_dir / 'fid_metrics.json'}"
          f"\n图片:  {out_dir}/<候选>/")
    print("判读：KID/CLIP-FID 越低=越像真实风景照片；CLIP 一致性越高=越听 prompt。"
          "\n      注意 KID 的标准差列在 CSV 里，差异小于标准差时不能声称更好。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
