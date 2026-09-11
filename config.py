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
