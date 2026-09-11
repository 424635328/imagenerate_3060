"""test_qlora_path.py — unit-test the int8-base + fp16-LoRA training path.

Loading the real 4.9 GB SDXL UNet just to find out whether quantization and
backprop work would cost minutes and gigabytes.  This builds a *tiny* UNet with
the same SDXL shapes (4-channel latents, 2048-dim cross attention, pooled
text_embeds + 6-dim time_ids) and runs the exact code path train_v5.py uses:

    load fp16 -> quantize_unet_8bit -> peft LoRA -> forward -> backward -> grads

Usage: python tools/test_qlora_path.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

from train_v5 import quantize_unet_8bit  # noqa: E402  (needs ROOT on sys.path)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig, get_peft_model

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    started = time.time()
    config = dict(sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
                  block_out_channels=(32, 64), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                  up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                  cross_attention_dim=2048, attention_head_dim=8, norm_num_groups=8)
    unet = UNet2DConditionModel(**config).half()
    check("tiny SDXL-shaped UNet builds", True)

    replaced = quantize_unet_8bit(unet)
    check("quantize_unet_8bit replaces attention/linear layers", replaced > 0, f"{replaced} layers")

    import bitsandbytes as bnb
    quantized_count = sum(1 for m in unet.modules() if isinstance(m, bnb.nn.Linear8bitLt))
    plain_count = sum(1 for m in unet.modules() if type(m) is torch.nn.Linear)
    check("no plain nn.Linear left in the base", plain_count == 0, f"{quantized_count} int8 / {plain_count} plain")

    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    unet = get_peft_model(unet, LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, bias="none",
                                           target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    for name, param in unet.named_parameters():
        param.requires_grad_(any(token in name for token in ("lora_", "LoRA", "magnitude")))
    trainable = [p for p in unet.parameters() if p.requires_grad]
    check("peft attached LoRA onto int8 layers", len(trainable) > 0, f"{len(trainable)} tensors")

    unet = unet.to("cuda")
    batch, res = 1, 128
    latent = torch.randn(batch, 4, res // 8, res // 8, device="cuda", dtype=torch.float16)
    noise = torch.randn_like(latent)
    timesteps = torch.randint(0, 1000, (batch,), device="cuda")
    prompt = torch.randn(batch, 77, 2048, device="cuda", dtype=torch.float16)
    pooled = torch.randn(batch, 1280, device="cuda", dtype=torch.float16)
    time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device="cuda", dtype=torch.float16)

    try:
        # Two steps: peft initialises lora_B to zero, so lora_A legitimately has
        # zero gradient on the very first backward.  After one optimizer step
        # every adapter must receive gradient.
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = unet(latent, timesteps, encoder_hidden_states=prompt,
                            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
                loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
            loss.backward()
            optimizer.step()
        finite = bool(torch.isfinite(loss).all())
        grad_norm = sum(float(p.grad.norm()) for p in trainable if p.grad is not None)
        with_grad = sum(1 for p in trainable if p.grad is not None and p.grad.abs().sum() > 0)
        check("forward+backward through int8 base", finite and grad_norm > 0,
              f"loss={loss.item():.4f} grad_norm={grad_norm:.4f}")
        check("every adapter received gradients after 2 steps", with_grad == len(trainable),
              f"{with_grad}/{len(trainable)}")
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        check("tiny model stays small", peak < 512, f"peak {peak:.0f} MB")
    except Exception as error:
        check("forward+backward through int8 base", False, f"{type(error).__name__}: {str(error)[:160]}")

    print()
    failed = 0
    for name, ok, detail in checks:
        if not ok:
            failed += 1
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok or not detail else f'  [{detail}]'}")
        if ok and detail:
            print(f"        {detail}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed in {time.time() - started:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
