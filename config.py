"""Landscape·Art shared configuration.

All paths are derived from LANDSCAPE_ROOT or this file's location.  Do not add
machine-specific absolute paths here; use environment variables instead.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT", Path(__file__).resolve().parent)).resolve()
MODELS_DIR = ROOT / "models"
BASE_MODEL_DIR = Path(os.environ.get("BASE_MODEL", str(MODELS_DIR / "base_rv6")))
ADAPTER_DIR = Path(os.environ.get("LORA_ADAPTER", str(MODELS_DIR / "v4_640" / "adapter_best")))
LCM_DIR = Path(os.environ.get("LCM_LORA", str(MODELS_DIR / "lcm_lora")))
TCD_DIR = Path(os.environ.get("TCD_LORA", str(MODELS_DIR / "tcd_lora")))
SR_DIR = MODELS_DIR / "sr"
HF_CACHE = Path(os.environ.get("HF_CACHE", str(MODELS_DIR / "hf_cache")))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(ROOT / "results")))

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8001"))
API_TOKEN = os.environ.get("API_TOKEN", "change-me")

# ---------------------------------------------------------------------------
# 可选权重（"模型版本"）：**服务端唯一入口**
#
# 前端要能选版本，就不能让请求里的字符串直接变成文件路径 —— 那是路径穿越。
# 所以这里是**白名单**：slug → (相对 models/ 的目录, 诚实标签, 备注)。
# 前端只传 slug，服务端用 resolve_adapter() 解析；未知 slug 一律拒绝。
# 备注里的数字来自本项目评测（research/ 与 docs/HUMAN_VERDICT.md），不是宣传语。
# ---------------------------------------------------------------------------
ADAPTER_CHOICES: dict[str, dict] = {
    "v4": {"dir": "v4_640/adapter_best", "label": "V4 基线", "slogan": "部署基线（抗过拟合倍增）",
           "note": "当前线上权重；固定协议 val 0.186437、留出集 KID 最低 0.000359。"},
    "v5": {"dir": "v5_lora/adapter_best", "label": "V5", "slogan": "min-SNR-γ + Prodigy",
           "note": "只改优化器与损失加权；文本编码器未微调。val 0.185891，KID 0.000404。"},
    "v5b": {"dir": "v5b_lora/adapter_best", "label": "V5b", "slogan": "V5 + 微调 CLIP",
            "note": "固定协议 val 最低 0.185834、KID 0.000377；text encoder 真的被微调过。"},
    "v6q": {"dir": "v6_qwen/adapter_best", "label": "V6q", "slogan": "Qwen2-VL 重写 caption",
            "note": "最后训练的版本；conditioning 不同 ⇒ val 与其它候选不可比，KID 0.000396。"},
    "merge-v4v5-linear": {"dir": "merged/linear_0.50_0.50", "label": "融合 V4⊕V5 线性",
                          "slogan": "免训练权重平均", "note": "UNet 增量 0.5/0.5 线性融合，val 0.185969。"},
    "merge-v4v5-slerp": {"dir": "merged/slerp_0.50_0.50", "label": "融合 V4⊕V5 SLERP",
                         "slogan": "球面插值", "note": "val 0.185901。"},
    "merge-v4v5-ties": {"dir": "merged/ties_0.50_0.50", "label": "融合 V4⊕V5 TIES",
                        "slogan": "符号一致裁剪", "note": "val 0.186027。"},
    "merge-v4v5-dare": {"dir": "merged/dare_ties_0.50_0.50", "label": "融合 V4⊕V5 DARE-TIES",
                        "slogan": "随机丢弃再裁剪", "note": "val 0.186041。"},
    "merge-v4v5b-linear": {"dir": "merged_v4v5b/linear_0.50_0.50", "label": "融合 V4⊕V5b 线性",
                           "slogan": "免训练权重平均", "note": "val 0.185963。"},
    "merge-v4v5b-slerp": {"dir": "merged_v4v5b/slerp_0.50_0.50", "label": "融合 V4⊕V5b SLERP",
                          "slogan": "球面插值", "note": "val 0.185939。"},
}
# 前端首次打开时的默认版本（要"最新"就改这里，或设环境变量 DEFAULT_ADAPTER）
DEFAULT_ADAPTER = os.environ.get("DEFAULT_ADAPTER", "v5b")


def adapter_dir(slug: str) -> Path:
    """slug → 绝对目录（不在白名单里就抛 ValueError，绝不拼接客户端给的路径）。"""
    entry = ADAPTER_CHOICES.get(str(slug or "").strip())
    if entry is None:
        raise ValueError(f"unknown adapter: {slug!r}")
    return (MODELS_DIR / entry["dir"]).resolve()


def adapter_slugs() -> list[str]:
    """白名单里的全部 slug（错误信息与前端提示用同一份来源）。"""
    return sorted(ADAPTER_CHOICES)


def adapter_catalogue(only_existing: bool = True) -> list[dict]:
    """给 /models 与前端用的目录（含 exists 与是否有微调过的文本编码器）。"""
    rows = []
    for slug, entry in ADAPTER_CHOICES.items():
        path = (MODELS_DIR / entry["dir"]).resolve()
        has_config = (path / "adapter_config.json").exists()
        if only_existing and not has_config:
            continue
        rows.append({
            "id": slug, "label": entry["label"], "slogan": entry["slogan"], "note": entry["note"],
            "dir": entry["dir"], "text_encoder": Path(f"{path}_text_encoder.pt").exists(),
            "default": slug == DEFAULT_ADAPTER,
        })
    return rows


ALLOW_ORIGINS = [x.strip() for x in os.environ.get("ALLOW_ORIGINS", "*").split(",") if x.strip()]
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "20"))
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "8"))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))
MAX_JOBS = int(os.environ.get("MAX_JOBS", "500"))
RESULTS_QUOTA_MB = int(os.environ.get("RESULTS_QUOTA_MB", "2048"))
RESULTS_QUOTA_BYTES = max(0, RESULTS_QUOTA_MB) * 1024 * 1024

# Generation safety/performance limits.  500 steps is not a quality setting for
# this SD1.5 setup: it wastes time after convergence and increases oversaturation.
MIN_STEPS = int(os.environ.get("MIN_STEPS", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "60"))
MAX_FAST_STEPS = int(os.environ.get("MAX_FAST_STEPS", "12"))
MIN_CFG = float(os.environ.get("MIN_CFG", "1.0"))
MAX_CFG = float(os.environ.get("MAX_CFG", "15.0"))

# Memory/GC governance.  IDLE_UNLOAD_SEC releases the pipeline (VRAM + host RAM)
# when nobody generates for a while; JANITOR_SEC drives the periodic sweep that
# enforces the result quota and removes orphan files.
IDLE_UNLOAD_SEC = int(os.environ.get("IDLE_UNLOAD_SEC", "900"))
JANITOR_SEC = max(15, int(os.environ.get("JANITOR_SEC", "60")))

# Uploaded reference images (img2img).  The browser downsizes and re-encodes to
# WebP before upload, so the server only needs a small safety ceiling.
ALLOW_UPLOAD = os.environ.get("ALLOW_UPLOAD", "1").lower() not in {"0", "false", "no"}
MAX_UPLOAD_BYTES = max(64 * 1024, int(os.environ.get("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024))))
MAX_INIT_EDGE = max(256, min(1536, int(os.environ.get("MAX_INIT_EDGE", "1024"))))
# Upload hardening: pixel budget (decompression-bomb guard) and the only formats
# the browser is allowed to send.  Anything else is rejected before decoding.
MAX_INIT_PIXELS = max(1_000_000, int(os.environ.get("MAX_INIT_PIXELS", str(24 * 1024 * 1024))))
ALLOWED_UPLOAD_FORMATS = {"JPEG", "PNG", "WEBP"}

# Progressive preview: decode a cheap latent thumbnail every N denoise steps so
# the browser can show something long before the final image exists.
PREVIEW_EVERY = max(0, int(os.environ.get("PREVIEW_EVERY", "4")))

OUT_FORMAT = os.environ.get("OUT_FORMAT", "webp").lower()
if OUT_FORMAT not in {"webp", "jpeg"}:
    OUT_FORMAT = "webp"
OUT_QUALITY = max(1, min(100, int(os.environ.get("OUT_QUALITY", "88"))))
PREVIEW_MAX = max(128, int(os.environ.get("PREVIEW_MAX", "512")))
CHUNK_SIZE = max(256 * 1024, min(4 * 1024 * 1024, int(os.environ.get("CHUNK_SIZE", "2000000"))))


def clamp_steps(value: int, fast: bool = False) -> int:
    """Clamp user input to a useful, bounded denoising range."""
    upper = MAX_FAST_STEPS if fast else MAX_STEPS
    return max(MIN_STEPS if not fast else 2, min(int(value), upper))


def clamp_cfg(value: float, fast: bool = False) -> float:
    """Clamp CFG; LCM/TCD need low CFG to remain stable."""
    upper = 2.5 if fast else MAX_CFG
    return max(1.0 if fast else MIN_CFG, min(float(value), upper))


def clamp_strength(value: float) -> float:
    """Clamp img2img strength to a range that keeps structure and stays useful."""
    return max(0.05, min(float(value), 0.95))


def output_suffix() -> str:
    return ".webp" if OUT_FORMAT == "webp" else ".jpg"


def output_mime() -> str:
    return "image/webp" if OUT_FORMAT == "webp" else "image/jpeg"


def ensure_runtime_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
