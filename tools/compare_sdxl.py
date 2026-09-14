"""compare_sdxl.py — SDXL 线的「前后对比」证据：同一 prompt x seed 下 base 与 LoRA 的出图对比。

为什么单独一个工具：`tools/compare_adapters.py` 走的是 SD1.5 `StableDiffusionPipeline`，
而 SDXL 在 6GB 卡上必须换一套加载策略（fp16 UNet 4.9GB + 双文本编码器 1.5GB + VAE 0.16GB
= 6.6GB > 6GB）。这里采用 `tools/probe_sdxl_infer.py` 已经实测过的三条策略：

    offload   fp16 UNet + enable_model_cpu_offload（默认，保真度最高）
    int8      用 train_v5.quantize_unet_8bit 量化基座后常驻显存（与训练/部署同形态）
    fp16      全部常驻显存（仅在显存确实够用时）

**指标与 SD1.5 侧完全同口径**（直接 import `compare_adapters` 的 PROMPTS / SEEDS /
metrics / CLIP 实现），因此两边的数字可以并排看；但 SD1.5 与 SDXL 是**不同基座**，
跨架构只能作为产品级对比，不能当成同一协议的性能排名——报告里必须分开列。

用法:
    python tools/compare_sdxl.py --dry-run                       # 只校验路径与配置（不加载模型）
    python tools/compare_sdxl.py --adapters "SDXL-BASE:" "SDXL:models/v5_sdxl/adapter_best" \
        --out research/compare_sdxl --steps 24 --res 1024
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import numpy as np  # noqa: E402
from compare_adapters import (PROMPTS, SEEDS, ROOT, clip_similarity,  # noqa: E402
                              load_clip, metrics)
from PIL import Image, ImageDraw  # noqa: E402

DEFAULT_BASE = os.environ.get("SDXL_BASE", "SG161222/RealVisXL_V5.0")


def resolve_base(base: str) -> str:
    """本地优先：项目内目录 > HF 缓存里的完整快照 > 原样交给 diffusers（可能联网）。"""
    candidate = Path(base)
    if candidate.is_dir() and (candidate / "model_index.json").exists():
        return str(candidate)
    cache = Path(os.environ.get("HF_HOME") or (ROOT / "models" / "hf_cache"))
    hub = cache / "hub" / ("models--" + base.replace("/", "--")) / "snapshots"
    if hub.is_dir():
        for snapshot in sorted(hub.iterdir(), reverse=True):
            if (snapshot / "model_index.json").exists() and (snapshot / "unet").exists():
                return str(snapshot)
    return base


def build_pipe(base: str, mode: str):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16,
                                                     add_watermarker=False,
                                                     local_files_only=Path(base).is_dir())
    if mode == "int8":
        from train_v5 import quantize_unet_8bit
        pipe.unet = pipe.unet.to("cpu")
        replaced = quantize_unet_8bit(pipe.unet)
        print(f"   量化 {replaced} 个 Linear 层 -> int8")
    # 1024 出图的 VAE 解码会瞬时吃 1-2GB，分块+切片是 6GB 卡的必要设置
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    if mode == "fp16":
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def attach_adapter(pipe, adapter_dir: str):
    """peft 生产路径加载（与 app.py / eval 工具一致）；SDXL 无微调文本编码器。"""
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    config = LoraConfig.from_pretrained(adapter_dir)
    unet = get_peft_model(pipe.unet, config)
    missing, unexpected = set_peft_model_state_dict(
        unet, load_file(str(Path(adapter_dir) / "adapter_model.safetensors")))
    pipe.unet = unet
    return missing, unexpected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+",
                    default=["SDXL-BASE:", "SDXL:models/v5_sdxl/adapter_best"],
                    help='"标签:adapter目录"；空路径表示不挂 adapter 的 SDXL 基座')
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=str(ROOT / "research" / "compare_sdxl"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--mode", default="offload", choices=["offload", "int8", "fp16"])
    ap.add_argument("--no-clip", action="store_true", help="跳过 CLIP 图文一致性（省一次加载）")
    ap.add_argument("--dry-run", action="store_true", help="只校验路径与配置，不加载模型、不用 GPU")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    out_dir = Path(args.out)
    resolved = resolve_base(args.base)
    print(f"SDXL 基座: {args.base}\n  解析为: {resolved}")
    print(f"策略: {args.mode}  步数: {args.steps}  CFG: {args.cfg}  分辨率: {args.res}")

    specs = []
    problems = []
    for spec in args.adapters:
        label, _, path = spec.partition(":")
        if path and not (Path(path) / "adapter_model.safetensors").exists():
            problems.append(f"{label}: 找不到 {path}/adapter_model.safetensors")
        specs.append((label, path))
        print(f"  {label:12s} {'(基座，无 adapter)' if not path else path}")
    if problems and not args.dry_run:
        for problem in problems:
            print(f"[错误] {problem}")
        return 2

    if args.dry_run:
        print(f"\n[dry-run] 将输出：{out_dir}/compare_metrics.csv、compare_sheet.png、sdxl_probe.json")
        print(f"[dry-run] 计划出图：{len(specs)} 个模型 x {len(PROMPTS)} prompt x {len(SEEDS)} seed "
              f"= {len(specs) * len(PROMPTS) * len(SEEDS)} 张 {args.res}x{args.res}")
        if problems:
            print("[dry-run] 待解决：" + "; ".join(problems))
        print("[dry-run] 未加载任何模型，未使用 GPU。")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        print("[错误] 需要 CUDA")
        return 2
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU: {torch.cuda.get_device_name(0)}  显存 {total_memory:.2f}GB\n")

    rows = []
    grids = []
    probe = {"base": args.base, "resolved_base": resolved, "mode": args.mode,
             "steps": args.steps, "cfg": args.cfg, "resolution": args.res,
             "gpu_total_gb": round(total_memory, 2), "arms": []}
    reference: dict[tuple[int, int], Image.Image] = {}

    for label, path in specs:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        print(f"[{label}] 加载 SDXL ({args.mode}) ...")
        pipe = build_pipe(resolved, args.mode)
        if path:
            missing, unexpected = attach_adapter(pipe, path)
            if missing:
                print(f"  [警告] {len(missing)} 个参数未匹配（前几个：{list(missing)[:3]}）")
            if unexpected:
                print(f"  [警告] {len(unexpected)} 个权重未被使用（前几个：{list(unexpected)[:3]}）")
            print(f"  已挂载 adapter（peft）：{path}")
        images = []
        deltas = []
        for index, prompt in enumerate(PROMPTS):
            for seed in SEEDS:
                generator = torch.Generator("cpu").manual_seed(seed)
                image = pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.cfg,
                             width=args.res, height=args.res, generator=generator).images[0]
                stat = metrics(image)
                key = (index, seed)
                if not path:
                    reference.setdefault(key, image)          # 基座臂建立参照
                elif key in reference:
                    left = np.asarray(reference[key], dtype=np.float32)
                    right = np.asarray(image, dtype=np.float32)
                    deltas.append(float(abs(left - right).mean()))
                rows.append({"adapter": label, "prompt": prompt[:46], "seed": seed,
                             "_image": image, "_prompt": prompt, **stat})
                images.append(image)
                print(f"  [{label}] seed {seed} 锐度 {stat['sharpness']:8.1f} "
                      f"饱和 {stat['saturation']:.3f} 对比 {stat['contrast']:.3f}")
        elapsed = time.time() - started
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        count = len(images)
        probe["arms"].append({"label": label, "adapter": path or None, "peak_gb": round(peak, 2),
                              "seconds_total": round(elapsed, 1),
                              "seconds_per_image": round(elapsed / max(1, count), 2),
                              "images": count})
        print(f"[{label}] 峰值显存 {peak:.2f}GB  总耗时 {elapsed:.1f}s  "
              f"({elapsed / max(1, count):.1f}s/图)")
        if deltas:
            mean_delta = sum(deltas) / len(deltas)
            flag = "  <== 与基座几乎一致，adapter 可能没生效！" if mean_delta < 1.0 else ""
            print(f"[{label}] 与基座同 seed 的平均像素差 {mean_delta:.2f} / 255{flag}")
            probe["arms"][-1]["mean_abs_delta_vs_base"] = round(mean_delta, 3)
        grids.append((label, images))
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # ---- CLIP 一致性：先释放 SDXL，再加载 CLIP（6GB 卡不同时驻留两套模型）----
    if not args.no_clip:
        model, processor, device = load_clip()
        if model is None:
            print("[clip] 不可用，跳过图文一致性")
        else:
            for row in rows:
                row["clip_score"] = round(clip_similarity(model, processor, row["_image"],
                                                          row["_prompt"], device), 4)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    csv_path = out_dir / "compare_metrics.csv"
    fieldnames = [key for key in rows[0].keys() if not key.startswith("_")]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    # 拼版：行 = prompt x seed，列 = 模型
    tile = 384
    columns = len(grids)
    line_count = len(grids[0][1])
    sheet = Image.new("RGB", (tile * columns, tile * line_count + 28), "#0b1020")
    for column, (label, images) in enumerate(grids):
        for row_index, image in enumerate(images):
            sheet.paste(image.resize((tile, tile), Image.LANCZOS), (column * tile, row_index * tile))
    draw = ImageDraw.Draw(sheet)
    for column, (label, _) in enumerate(grids):
        draw.text((column * tile + 8, tile * line_count + 6), label, fill="#e8eefc")
    sheet_path = out_dir / "compare_sheet.png"
    sheet.save(sheet_path)

    probe_path = out_dir / "sdxl_probe.json"
    probe_path.write_text(json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print("\n=== 汇总（同一 prompt x seed，同口径指标） ===")
    for label, _ in grids:
        subset = [row for row in rows if row["adapter"] == label]
        sharp = sum(row["sharpness"] for row in subset) / len(subset)
        sat = sum(row["saturation"] for row in subset) / len(subset)
        contrast = sum(row["contrast"] for row in subset) / len(subset)
        line = f"  {label:12s} 锐度 {sharp:8.1f}  饱和 {sat:.3f}  对比 {contrast:.3f}"
        if subset and subset[0].get("clip_score"):
            clip = sum(row["clip_score"] for row in subset) / len(subset)
            line += f"  CLIP {clip:.4f}"
        print(line)
    print(f"\nsheet: {sheet_path}\ncsv:   {csv_path}\nprobe: {probe_path}")
    print("注意：SDXL 与 SD1.5 是不同基座，跨架构数字只能作产品级对比，报告需分开列。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
