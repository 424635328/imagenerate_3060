"""fetch_models.py — 一条命令拉齐「不入库」的模型资产。

仓库只保存源码/配置/文档/站点资源；权重与数据集靠脚本重建（见 docs/REPRODUCE.md）。
本脚本把外部依赖拉到本地约定路径，让协作者 clone 后能直接跑训练与推理。

    python tools/fetch_models.py all      # SD1.5 基座 + SDXL 基座 + LCM-LoRA + 超分权重
    python tools/fetch_models.py sd15     # SG161222/Realistic_Vision_V6.0_B1_noVAE
    python tools/fetch_models.py sdxl     # SG161222/RealVisXL_V5.0（仅 fp16，约 6.5 GB）
    python tools/fetch_models.py lcm      # LCM-LoRA → models/lcm_lora/
    python tools/fetch_models.py sr       # 4x-UltraSharp → models/sr/
    python tools/fetch_models.py realesrgan   # RealESRGAN_x4plus.pth（官方 release）

路径约定（与 config.py 一致）：
    HF_HOME=<root>/models/hf_cache
    models/lcm_lora/pytorch_lora_weights.safetensors
    models/sr/{4x-UltraSharp.pth, RealESRGAN_x4plus.pth}

关于基座目录：config.py 默认 `BASE_MODEL=models/base_rv6`（本地 diffusers 目录）。
协作者不必自己转换——把环境变量指向 HF 仓库 id 即可：
    $env:BASE_MODEL="SG161222/Realistic_Vision_V6.0_B1_noVAE"
"""
from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
CACHE = Path(os.environ.get("HF_CACHE", str(ROOT / "models" / "hf_cache")))
MODELS = ROOT / "models"

# repo_id -> (允许的文件模式, {快照内文件名: 需要落到本地的目录})
HF_TARGETS = {
    "sd15": ("SG161222/Realistic_Vision_V6.0_B1_noVAE", [
        "model_index.json", "*.json", "*.txt",
        "unet/*", "text_encoder/*", "vae/*", "tokenizer/*", "scheduler/*",
    ], {}),
    "sdxl": ("SG161222/RealVisXL_V5.0", [
        "model_index.json", "*.json", "*.txt",
        "unet/*.json", "unet/*fp16*",
        "text_encoder/*.json", "text_encoder/*fp16*",
        "text_encoder_2/*.json", "text_encoder_2/*fp16*",
        "vae/*.json", "vae/*fp16*",
        "tokenizer/*", "tokenizer_2/*", "scheduler/*",
    ], {}),
    "lcm": ("latent-consistency/lcm-lora-sdv1-5", ["pytorch_lora_weights.safetensors"], {
        "pytorch_lora_weights.safetensors": MODELS / "lcm_lora",
    }),
    "sr": ("Kim2091/UltraSharp", ["4x-UltraSharp.pth"], {
        "4x-UltraSharp.pth": MODELS / "sr",
    }),
}

# RealESRGAN_x4plus.pth 只在官方 release 上（HF 镜像里叫 RealESRGAN_x4.pth，并非同一文件）
DIRECT_FILES = {
    "realesrgan": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        MODELS / "sr" / "RealESRGAN_x4plus.pth",
    ),
}


def human(path: Path) -> str:
    if not path.exists():
        return "missing"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / 1024 ** 3:.2f} GB" if total > 1024 ** 3 else f"{total / 1024 ** 2:.0f} MB"


def fetch_hf(target: str) -> None:
    repo, patterns, mapping = HF_TARGETS[target]
    os.environ.setdefault("HF_HOME", str(CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(CACHE / "hub"))
    from huggingface_hub import snapshot_download

    print(f"[{target}] downloading {repo} ...")
    snapshot = Path(snapshot_download(repo_id=repo, cache_dir=str(CACHE / "hub"),
                                      allow_patterns=patterns, max_workers=4))
    for name, destination in mapping.items():
        source = next((p for p in snapshot.rglob(name) if p.is_file()), None)
        if not source:
            print(f"[{target}] !! {name} not found in snapshot")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        local = destination / name
        if local.exists() and local.stat().st_size == source.stat().st_size:
            print(f"[{target}] {local.relative_to(ROOT)} already present")
            continue
        try:
            os.link(source, local)              # hardlink when on the same volume
        except OSError:
            shutil.copy2(source, local)
        print(f"[{target}] -> {local.relative_to(ROOT)}  ({human(destination)})")
    print(f"[{target}] snapshot: {snapshot}")


def fetch_direct(target: str) -> None:
    url, destination = DIRECT_FILES[target]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1024:
        print(f"[{target}] {destination.relative_to(ROOT)} already present")
        return
    print(f"[{target}] downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"[{target}] -> {destination.relative_to(ROOT)}")


def main() -> int:
    targets = sys.argv[1:] or ["all"]
    if "all" in targets:
        targets = ["sd15", "sdxl", "lcm", "sr", "realesrgan"]
    for target in targets:
        if target in HF_TARGETS:
            fetch_hf(target)
        elif target in DIRECT_FILES:
            fetch_direct(target)
        else:
            print(f"unknown target: {target}")
            print("available: " + " | ".join(list(HF_TARGETS) + list(DIRECT_FILES) + ["all"]))
            return 2
    print("\n=== 本地资产现状 ===")
    for path in [MODELS / "hf_cache", MODELS / "lcm_lora", MODELS / "sr", MODELS / "base_rv6"]:
        print(f"  {path.relative_to(ROOT)}: {human(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
