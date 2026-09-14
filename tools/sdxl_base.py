"""sdxl_base.py — 把 SDXL 基座解析成**本地路径**并强制离线加载。

为什么必须有这个模块（2026-09-13 事故）：`tools/probe_sdxl_train.py` 用 HF Hub id
`SG161222/RealVisXL_V5.0` 直接 `from_pretrained`，diffusers 于是认为需要"取文件"，
在代理环境下卡了 5 分 22 秒后抛出

    ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol

五个显存模式**全部** FAIL，`train_pipeline.py` 把它读成"6GB 跑不了 SDXL"并中止主线 ——
实际上本地 HF 缓存里那份快照是完整的（unet / text_encoder / text_encoder_2 / vae /
两个 tokenizer / scheduler / model_index.json 全在）。结论很清楚：**基座必须走本地目录 +
`local_files_only=True`**，既避免联网等待，也避免把基础设施故障误判成硬件容量不足。

用法:
    from sdxl_base import resolve_base, offline_kwargs
    base = resolve_base(os.environ.get("SDXL_BASE", "SG161222/RealVisXL_V5.0"))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, **offline_kwargs(base), torch_dtype=...)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])


def hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME") or (ROOT / "models" / "hf_cache"))


def snapshot_for(repo_id: str) -> Path | None:
    """在 HF 缓存里找该仓库**可用的**快照（必须含 model_index.json 与 unet）。

    取最新的一个；快照目录缺失关键组件时继续往前找 —— 半个快照比没有更糟，
    它会让人以为"模型没问题"，然后在加载 VAE 时才炸。
    """
    repo = hf_cache_root() / "hub" / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    if not repo.is_dir():
        return None
    for snapshot in sorted(repo.iterdir(), reverse=True):
        if (snapshot / "model_index.json").exists() and (snapshot / "unet").exists():
            return snapshot
    return None


def resolve_base(value: str) -> str:
    """本地目录 > HF 缓存里的完整快照 > 原样返回（保持旧行为，便于报错时看清原值）。"""
    if not value:
        return value
    candidate = Path(value)
    if candidate.is_dir() and (candidate / "model_index.json").exists():
        return str(candidate)
    if "/" in value and not candidate.exists():
        snapshot = snapshot_for(value)
        if snapshot is not None:
            return str(snapshot)
    return value


def offline_kwargs(base: str) -> dict:
    """本地目录一律要求离线：不联网、不检查元数据、不下载。"""
    return {"local_files_only": True} if Path(base).is_dir() else {}


def is_local(base: str) -> bool:
    return Path(base).is_dir()


def describe(base: str) -> str:
    """给日志用的一句话说明，避免"到底加载的是哪份权重"成为谜。"""
    if is_local(base):
        return f"{base}  (本地目录，离线加载)"
    return f"{base}  (非本地：可能联网，代理环境下会卡住)"
