"""test_adapter_load.py — 部署门禁：确认 adapter **真的生效**，而不只是"没报错"。

为什么要写这么重：曾用 `pipe.load_lora_weights(dir)` 加载 peft 格式的 adapter，
diffusers 只打印 "No LoRA keys associated to UNet2DConditionModel found ... This is safe
to ignore"，然后**什么都不加载**；旧版门禁仅检查"无异常"，于是给出假阳性，
连带出图对比里 V4 与 V5 的指标一模一样（出的其实都是基座图）。

本门禁的做法：
  1. 按**线上 app.py 的同一路径**加载（peft：LoraConfig + get_peft_model + set_peft_model_state_dict）；
  2. 固定 seed 出图两次：基座 vs 挂 adapter；
  3. 断言两者**确实不同**（平均像素差超过阈值），否则判定该 adapter 未生效；
  4. 若随 adapter 一起微调过文本编码器（`<adapter>_text_encoder.pt`），同样校验其权重确有变化。

用法: python tools/test_adapter_load.py <adapter_dir> [<adapter_dir> ...]
退出码：0 = 全部生效；1 = 有 adapter 未生效或加载失败。
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
BASE = os.environ.get("BASE_MODEL", str(ROOT / "models" / "base_rv6"))
PROMPT = "a misty alpine lake at dawn, dramatic clouds"
SEED = 777
STEPS = 8
MIN_MEAN_DIFF = 1.0          # 平均像素差（0–255）：生效的 adapter 远高于此阈值


def generate(pipe) -> np.ndarray:
    generator = torch.Generator("cpu").manual_seed(SEED)
    image = pipe(PROMPT, num_inference_steps=STEPS, guidance_scale=5.0,
                 width=256, height=256, generator=generator).images[0]
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def with_adapter(pipe, adapter_dir: Path) -> int:
    """与 app.py 完全一致的加载方式（peft 路径），返回载入的张量数。"""
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", torch_dtype=torch.float16)
    config = LoraConfig.from_pretrained(str(adapter_dir))
    unet = get_peft_model(unet, config)
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    set_peft_model_state_dict(unet, state)
    pipe.unet = unet.to("cuda").eval()
    return len(state)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python tools/test_adapter_load.py <adapter_dir> [...]")
        return 2
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    from diffusers import StableDiffusionPipeline

    print(f"base pipeline: {BASE}")
    pipe = StableDiffusionPipeline.from_pretrained(BASE, torch_dtype=torch.float16,
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    baseline = generate(pipe)
    print(f"baseline image: mean {baseline.mean():.1f}, std {baseline.std():.1f}\n")

    targets = sys.argv[1:]
    failures = 0
    for target in targets:
        path = Path(target)
        label = f"{path.parent.name}/{path.name}"
        if not (path / "adapter_model.safetensors").exists():
            print(f"SKIP {label}: no adapter_model.safetensors")
            continue
        try:
            tensors = with_adapter(pipe, path)
            image = generate(pipe)
            diff = float(np.abs(image - baseline).mean())
            ok = diff >= MIN_MEAN_DIFF
            # 文本编码器（若随 adapter 一起微调过）
            te_note = ""
            te_file = Path(str(path) + "_text_encoder.pt")
            if te_file.exists():
                base_state = copy.deepcopy(pipe.text_encoder.state_dict())
                pipe.text_encoder.load_state_dict(torch.load(str(te_file), map_location="cpu"))
                current = pipe.text_encoder.state_dict()
                te_diff = float(max((current[k].float() - base_state[k].float()).abs().max().item()
                                    for k in base_state))
                pipe.text_encoder.load_state_dict(base_state)
                te_note = f" | text_encoder max|delta|={te_diff:.5f}" + (" APPLIED" if te_diff > 0 else " NOT-APPLIED")
                if te_diff == 0:
                    ok = False
            if not ok:
                failures += 1
            print(f"{'PASS' if ok else 'FAIL'}  {label}: {tensors} tensors, mean|Δ|={diff:.2f}{te_note}")
        except Exception as error:
            failures += 1
            print(f"FAIL  {label}: {type(error).__name__}: {str(error)[:140]}")
    print(f"\n{len(targets) - failures}/{len(targets)} adapters verified effective "
          f"(阈值 mean|Δ| ≥ {MIN_MEAN_DIFF})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
