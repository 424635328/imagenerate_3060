"""probe_sdxl_infer.py — 6GB 上「SDXL 到底能不能用、有多快」的实测探针。

训练出 SDXL LoRA 只是一半；能不能在这张卡上**跑起来**是另一半。SDXL 的困难在于
三个模型都要驻留：UNet fp16 ≈ 4.9 GB + 双文本编码器 ≈ 1.5 GB + VAE ≈ 0.16 GB
= 6.6 GB，已经超过 6GB 显存。本探针逐档实测下列策略的真实峰值与耗时：

  fp16-offload      fp16 UNet + enable_model_cpu_offload（按需换入显存）
  int8-cuda         int8 UNet（train_v5.quantize_unet_8bit）+ 全部常驻显存
  int8-offload      int8 UNet + CPU offload（最省显存）
  每种策略分别测 25 步标准采样 与 4 步 SDXL-Lightning（guidance=0）

用法:
    python tools/probe_sdxl_infer.py                       # 全部策略
    python tools/probe_sdxl_infer.py --modes int8-cuda --steps 4
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))
from train_v5 import quantize_unet_8bit  # noqa: E402 — 与训练侧同一套量化实现

BASE = os.environ.get("SDXL_BASE", "SG161222/RealVisXL_V5.0")
LIGHTNING = ROOT / "models" / "lightning" / "sdxl_lightning_4step_lora.safetensors"
PROMPT = "a dramatic mountain valley at sunrise, mist, alpine lake, golden light"


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def build_pipe(quantized: bool, offload: bool):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(BASE, torch_dtype=torch.float16, add_watermarker=False)
    if quantized:
        print("   quantizing UNet to int8 ...")
        pipe.unet = pipe.unet.to("cpu")
        replaced = quantize_unet_8bit(pipe.unet)
        print(f"   {replaced} Linear layers -> int8")
    # VAE 解码 1024 图会瞬时吃 1-2 GB，分块+切片是 6GB 卡的必要设置
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def run_once(pipe, steps: int, lightning: bool, resolution: int):
    if lightning:
        if not LIGHTNING.exists():
            return None, "lightning lora missing (python tools/fetch_models.py lightning)"
        pipe.load_lora_weights(str(LIGHTNING))
        guidance = 0.0
    else:
        guidance = 7.0
    generator = torch.Generator("cpu").manual_seed(1234)
    torch.cuda.synchronize()
    started = time.time()
    image = pipe(PROMPT, num_inference_steps=steps, guidance_scale=guidance,
                 width=resolution, height=resolution, generator=generator).images[0]
    torch.cuda.synchronize()
    seconds = time.time() - started
    if lightning:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
    return (seconds, torch.cuda.max_memory_allocated() / 1024 ** 3, image.size), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="fp16-offload,int8-cuda,int8-offload")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--lightning", action="store_true", help="额外测 4 步 Lightning（guidance=0）")
    args = ap.parse_args()
    if args.lightning:
        args.steps = 4

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    print(f"base={BASE}\ngpu={torch.cuda.get_device_name(0)} "
          f"total={torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB")
    print(f"steps={args.steps} res={args.resolution} lightning={args.lightning}\n")

    rows = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        quantized = mode.startswith("int8")
        offload = mode.endswith("offload")
        free()
        try:
            pipe = build_pipe(quantized, offload)
            # 一次热身（分配器增长 + 首次 kernel 编译），再测一次真实耗时
            warm, error = run_once(pipe, args.steps, args.lightning, args.resolution)
            if error:
                print(f"{mode:14s} SKIP {error}")
                rows.append((mode, "skip", 0.0, 0.0))
                del pipe
                continue
            result, error = run_once(pipe, args.steps, args.lightning, args.resolution)
            seconds, peak, size = result
            print(f"{mode:14s} OK   peak={peak:5.2f}GB  {seconds:6.2f}s/image  {size[0]}x{size[1]}")
            rows.append((mode, "ok", peak, seconds))
            del pipe
        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"{mode:14s} OOM  peak={peak:5.2f}GB")
            rows.append((mode, "oom", peak, 0.0))
        except Exception as error:
            print(f"{mode:14s} FAIL {type(error).__name__}: {str(error)[:120]}")
            rows.append((mode, "fail", 0.0, 0.0))
        finally:
            free()

    print("\n=== summary ===")
    for mode, status, peak, seconds in rows:
        print(f"  {mode:14s} {status:5s} peak={peak:5.2f}GB {seconds:6.2f}s/image")
    feasible = [r for r in rows if r[1] == "ok"]
    if feasible:
        best = min(feasible, key=lambda r: r[3])
        print(f"\n最快可行配置: {best[0]}  ({best[3]:.2f}s/image, 峰值 {best[2]:.2f}GB)")
    else:
        print("\n没有任何配置跑通 —— SDXL 需要云端 GPU（server_cloud.py 已预留）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
