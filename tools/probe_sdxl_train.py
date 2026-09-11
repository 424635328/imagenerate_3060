"""probe_sdxl_train.py — can this 6 GB card actually *train* an SDXL LoRA?

Loads the SDXL UNet under several memory strategies and runs one real
forward+backward with cached latents and pre-computed text embeddings (the same
setup train_v5.py uses), reporting peak VRAM and seconds per step.

Configurations tested, in order of increasing memory cost:
  8bit-512 / 8bit-768 / 8bit-1024   bnb int8 base UNet + fp16 LoRA (QLoRA style)
  fp16-640  / fp16-768 / fp16-1024  plain fp16 base UNet + fp16 LoRA

Usage:
    python tools/probe_sdxl_train.py [--res 1024] [--modes 8bit-1024,fp16-768]
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
from train_v5 import quantize_unet_8bit  # noqa: E402  — same code path as training
BASE = os.environ.get("BASE_MODEL", "SG161222/RealVisXL_V5.0")
CACHE_DIR = os.environ.get("HF_CACHE", str(ROOT / "models" / "hf_cache"))


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def build_unet(base: str, quantized: bool):
    """Mirrors train_v5.py exactly: manual int8 replacement (validated by
    tools/test_qlora_path.py).  diffusers' own quantizer does not cover plain
    UNet2DConditionModel, so the probe must not rely on it."""
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=torch.float16)
    if quantized:
        print(f"  quantized {quantize_unet_8bit(unet)} Linear layers to int8")
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    return unet


def attach_lora(unet, rank: int = 32):
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(unet, LoraConfig(
        r=rank, lora_alpha=rank, lora_dropout=0.0, bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    ))
    for name, param in model.named_parameters():
        param.requires_grad_(any(token in name for token in ("lora_", "LoRA", "magnitude")))
    return model


def one_step(unet, res: int, device: str = "cuda"):
    """One forward+backward with the exact tensor shapes SDXL training needs."""
    batch = 1
    latent = torch.randn(batch, 4, res // 8, res // 8, device=device, dtype=torch.float16)
    noise = torch.randn_like(latent)
    ts = torch.randint(0, 1000, (batch,), device=device)
    prompt = torch.randn(batch, 77, 2048, device=device, dtype=torch.float16)
    pooled = torch.randn(batch, 1280, device=device, dtype=torch.float16)
    time_ids = torch.tensor([[res, res, 0, 0, res, res]], device=device, dtype=torch.float16)

    torch.cuda.synchronize()
    started = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        pred = unet(latent, ts, encoder_hidden_states=prompt,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
        loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
    loss.backward()
    torch.cuda.synchronize()
    seconds = time.time() - started
    params = [p for p in unet.parameters() if p.requires_grad]
    for param in params:
        param.grad = None
    return seconds, loss.item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--modes", default="8bit-512,8bit-768,8bit-1024,fp16-640,fp16-768,fp16-1024")
    ap.add_argument("--rank", type=int, default=32)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", CACHE_DIR)
    os.environ.setdefault("HF_HUB_CACHE", CACHE_DIR + "/hub")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"base={args.base}")
    print(f"gpu={torch.cuda.get_device_name(0)} total={torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB")
    results = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        quantized, res = mode.split("-")
        res = int(res)
        free()
        try:
            unet = build_unet(args.base, quantized == "8bit")
            unet = attach_lora(unet, args.rank).to("cuda")
            load_gb = torch.cuda.memory_allocated() / 1024 ** 3
            # warm-up step (allocator growth) then a measured step
            one_step(unet, res)
            seconds, loss = one_step(unet, res)
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"{mode:12s} OK   peak={peak:5.2f}GB weights={load_gb:5.2f}GB "
                  f"{seconds:5.2f}s/step loss={loss:.4f}")
            results.append((mode, True, peak, seconds))
            del unet
        except torch.cuda.OutOfMemoryError:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"{mode:12s} OOM  peak={peak:5.2f}GB")
            results.append((mode, False, peak, 0.0))
        except Exception as error:
            print(f"{mode:12s} FAIL {type(error).__name__}: {str(error)[:160]}")
            results.append((mode, False, 0.0, 0.0))
        finally:
            free()

    print("\n=== summary ===")
    for mode, ok, peak, seconds in results:
        print(f"  {mode:12s} {'feasible' if ok else 'not feasible':12s} peak={peak:5.2f}GB {seconds:5.2f}s/step")
    feasible = [r for r in results if r[1]]
    print(f"\n{len(feasible)}/{len(results)} configurations fit in 6 GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
