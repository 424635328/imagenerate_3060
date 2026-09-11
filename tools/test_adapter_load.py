"""test_adapter_load.py — 判断某个导出的 adapter 能否被现有推理栈加载。

背景：diffusers 0.40 的 LoRA 加载器没有 DoRA 支持（源码中无 use_dora 引用），
而 peft 0.20 能训练 DoRA。若 DoRA 权重无法加载，生产训练必须用纯 LoRA。

用法: python tools/test_adapter_load.py <adapter_dir> [<adapter_dir> ...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
BASE = os.environ.get("BASE_MODEL", str(ROOT / "models" / "base_rv6"))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python tools/test_adapter_load.py <adapter_dir> [...]")
        return 2
    import torch
    from diffusers import StableDiffusionPipeline

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    print(f"base pipeline: {BASE}")
    pipe = StableDiffusionPipeline.from_pretrained(BASE, torch_dtype=torch.float16,
                                                   safety_checker=None, requires_safety_checker=False)
    pipe = pipe.to("cuda")
    failures = 0
    for target in sys.argv[1:]:
        path = Path(target)
        label = path.name
        if not path.exists():
            print(f"SKIP {label}: not found")
            continue
        config = path / "adapter_config.json"
        if config.exists():
            print(f"  {label} adapter_config: {config.read_text(encoding='utf-8')[:220]}")
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass
        try:
            pipe.load_lora_weights(str(path))
            # a real forward pass proves the weights are wired, not just parsed
            _ = pipe("a misty alpine lake at dawn", num_inference_steps=2, guidance_scale=1.0,
                     width=256, height=256, output_type="np")
            print(f"LOAD OK   {label}")
        except Exception as error:
            failures += 1
            print(f"LOAD FAIL {label}: {type(error).__name__}: {str(error)[:200]}")
    print(f"\n{len(sys.argv) - 1 - failures}/{len(sys.argv) - 1} adapters load in the inference stack")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
