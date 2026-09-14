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
# 版本台账（"模型版本"的唯一真相）
#
# 设计见 docs/VERSION_ROUTING.md。要点：
#   · **版本是数据，不是代码**：上线 = 翻 channels.default 指针，不是改 Python 字典；
#   · **不可变 + 可校验**：promoted 之后权重 sha256 冻结，加载前校验，不符即拒绝加载
#     （本项目两次被"静默 no-op"咬过：键与形状全对、出图却和基座逐位相同）；
#   · **归档是墓碑**：权重移走、台账留条目与 hash ⇒ 历史/画廊仍能追溯出处；
#   · **台账读不到时不静默退回硬编码**（那会静默用错权重），而是按磁盘重建并**明确告警**。
#
# 落地位置说明：设计稿里写的是 models/registry.json，但 models/ 被 .gitignore 排除，
# 台账必须能跨克隆存活 —— 因此放在受版本控制的 registry/versions.json（可用环境变量
# VERSION_REGISTRY 覆盖）。台账里只有相对路径与哈希，不含任何机器绝对路径。
# ---------------------------------------------------------------------------
REGISTRY_PATH = Path(os.environ.get("VERSION_REGISTRY", str(ROOT / "registry" / "versions.json")))
# 扫描磁盘时认得的布局：models/<...>/adapter_best 与 models/merged*/<name>
ADAPTER_SCAN_GLOBS = ("*/adapter_best", "merged*/*")
# 已知版本的人工信息（诚实标签与备注；台账里已有则以台账为准）。新版本没登记时，
# scan 会以 auto-<dir> 形式收进来并标 needs_review=True —— 猜标签不如让人补。
KNOWN_VERSIONS: dict[str, dict] = {
    "v4": {"dir": "v4_640/adapter_best", "label": "V4 基线", "slogan": "部署基线（抗过拟合倍增）",
           "note": "当前线上基线；固定协议 val 0.186437、留出集 KID 最低 0.000359。"},
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
# 台账缺失时的引导默认（环境变量优先）。**只在没有任何台账时使用**，用完会明确告警。
BOOTSTRAP_DEFAULT = os.environ.get("DEFAULT_ADAPTER", "v5b")
VALID_STATES = ("candidate", "promoted", "superseded", "archived")


def sha256_file(path: Path, cache: dict | None = None) -> str:
    """流式 sha256；按 (路径, 大小, mtime) 记忆，避免每次请求都重算 150 MB。"""
    import hashlib
    store = _HASH_CACHE if cache is None else cache
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    hit = store.get(key)
    if hit:
        return hit
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    store.clear()                      # 只留最新一批，别让缓存无限长
    store[key] = digest.hexdigest()
    return store[key]


_HASH_CACHE: dict = {}


def _adapter_entry(path: Path) -> dict:
    """权重条目：目录 + 主文件哈希 + 可选微调文本编码器。"""
    main = path / "adapter_model.safetensors"
    te = Path(f"{path}_text_encoder.pt")
    return {
        "dir": _relative_to_models(path),
        "sha256": sha256_file(main) if main.exists() else None,
        "bytes": main.stat().st_size if main.exists() else 0,
        "text_encoder": ({"file": te.name, "sha256": sha256_file(te), "bytes": te.stat().st_size}
                         if te.exists() else None),
    }


def _relative_to_models(path: Path) -> str:
    try:
        return path.resolve().relative_to(MODELS_DIR.resolve()).as_posix()
    except ValueError:
        raise ValueError(f"版本目录必须在 models/ 之内：{path}")


def scan_versions() -> dict[str, dict]:
    """扫磁盘上认得的权重目录 → {slug: 条目}（不保留旧状态，纯事实）。"""
    found: dict[str, dict] = {}
    known_by_dir = {entry["dir"]: slug for slug, entry in KNOWN_VERSIONS.items()}
    for pattern in ADAPTER_SCAN_GLOBS:
        for path in sorted(MODELS_DIR.glob(pattern)):
            if not (path / "adapter_config.json").exists():
                continue
            rel = path.relative_to(MODELS_DIR).as_posix()
            slug = known_by_dir.get(rel) or f"auto-{rel.replace('/', '-')}"
            found[slug] = _adapter_entry(path)
    return found


def build_registry(existing: dict | None = None) -> dict:
    """按磁盘重建台账：保留已有条目的状态/评测/标签，新目录以 candidate 收进来。"""
    previous = (existing or {}).get("versions", {})
    versions: dict[str, dict] = {}
    for slug, entry in scan_versions().items():
        old = previous.get(slug, {})
        meta = KNOWN_VERSIONS.get(slug, {})
        versions[slug] = {
            "label": old.get("label") or meta.get("label") or slug,
            "slogan": old.get("slogan") or meta.get("slogan") or "（未登记，请在 registry 里补说明）",
            "note": old.get("note") or meta.get("note") or "未登记版本：先补 label/note 与评测再上线。",
            "state": old.get("state") or "candidate",
            "arch": old.get("arch", "sd15"),
            "adapter": entry,
            "recipe": old.get("recipe", {}),
            "eval": old.get("eval", {}),
            "created_at": old.get("created_at"),
            "promoted_at": old.get("promoted_at"),
            "needs_review": old.get("needs_review", slug.startswith("auto-")),
        }
    channels = dict((existing or {}).get("channels", {}))
    channels.setdefault("default", BOOTSTRAP_DEFAULT)
    channels.setdefault("previous", None)
    channels.setdefault("staging", None)
    # 通道指针必须指向真实存在的版本（磁盘上刚被删掉的版本要立刻从通道里摘掉）
    for name in ("default", "previous", "staging"):
        if channels.get(name) and channels[name] not in versions:
            channels[name] = None
    if not channels["default"]:
        channels["default"] = next(iter(versions), "")
    if channels["default"] in versions:
        versions[channels["default"]]["state"] = "promoted"
    if channels.get("previous") in versions:
        versions[channels["previous"]]["state"] = "superseded"
    return {"schema": 1, "channels": channels, "versions": versions}


def load_registry(force: bool = False) -> dict:
    """读台账（带 mtime 热重载）。缺失 → 按磁盘重建 + 明确告警（绝不静默用硬编码）。"""
    import json
    try:
        stat = REGISTRY_PATH.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None
    cached = _REGISTRY_CACHE.get("data")
    if cached is not None and _REGISTRY_CACHE.get("stamp") == stamp and not force:
        return cached
    if stamp is None:
        data = build_registry()
        _REGISTRY_CACHE.update({"data": data, "stamp": None, "bootstrapped": True})
        print(f"[registry] 台账不存在（{REGISTRY_PATH}）—— 已按磁盘重建，"
              f"default={data['channels']['default']}。建议执行：python tools/registry.py scan",
              flush=True)
        return data
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    _REGISTRY_CACHE.update({"data": data, "stamp": stamp, "bootstrapped": False})
    return data


_REGISTRY_CACHE: dict = {}


def save_registry(data: dict) -> Path:
    """原子写回台账（.tmp + os.replace），行尾 LF。"""
    import json
    import os as _os
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8", newline="\n")
    _os.replace(tmp, REGISTRY_PATH)
    load_registry(force=True)
    return REGISTRY_PATH


def registry_versions() -> dict[str, dict]:
    return load_registry().get("versions", {})


def default_adapter() -> str:
    """当前默认版本（台账里的 channels.default；每次读取，热重载即时生效）。"""
    return (load_registry().get("channels", {}) or {}).get("default") or BOOTSTRAP_DEFAULT


def previous_adapter() -> str | None:
    return (load_registry().get("channels", {}) or {}).get("previous")


def adapter_slugs(include_archived: bool = True) -> list[str]:
    versions = registry_versions()
    if include_archived:
        return sorted(versions)
    return sorted(slug for slug, entry in versions.items() if entry.get("state") != "archived")


def adapter_dir(slug: str) -> Path:
    """slug → 绝对目录。**只认台账里的 slug**，客户端字符串永不参与路径拼接。"""
    entry = registry_versions().get(str(slug or "").strip())
    if entry is None:
        raise ValueError(f"unknown adapter: {slug!r}")
    return (MODELS_DIR / entry["adapter"]["dir"]).resolve()


def verify_adapter(slug: str) -> tuple[bool, str]:
    """加载前校验：权重文件必须与台账里的 sha256 一致。

    返回 (ok, 说明)。不一致时调用方必须**拒绝加载**，而不是换个文件凑合。
    """
    versions = registry_versions()
    entry = versions.get(slug)
    if entry is None:
        return False, f"台账里没有这个版本：{slug}"
    path = MODELS_DIR / entry["adapter"]["dir"] / "adapter_model.safetensors"
    if not path.exists():
        return False, f"权重文件不在盘上：{entry['adapter']['dir']}"
    expected = entry["adapter"].get("sha256")
    actual = sha256_file(path)
    if expected and actual != expected:
        return False, f"权重哈希与台账不符（期望 {expected[:12]}… 实际 {actual[:12]}…）"
    return True, "ok"


def slug_for_dir(path: str | Path) -> str | None:
    """目录 → slug（给"拿到路径但要查台账"的调用方用，例如管线加载前校验）。"""
    target = Path(path).resolve()
    for slug, entry in registry_versions().items():
        if (MODELS_DIR / entry["adapter"]["dir"]).resolve() == target:
            return slug
    return None


def verify_adapter_path(path: str | Path) -> tuple[bool, str, str | None]:
    """按目录校验（返回 ok / 说明 / slug）。台账里没有这个目录时 slug=None（未登记路径）。"""
    slug = slug_for_dir(path)
    if slug is None:
        return True, "未登记在台账里的路径（跳过哈希校验）", None
    ok, reason = verify_adapter(slug)
    return ok, reason, slug


def adapter_catalogue(only_existing: bool = True) -> list[dict]:
    """给 /models 与前端用的清单（含状态、哈希前 12 位、TE 标记、是否已校验）。"""
    data = load_registry()
    versions = data.get("versions", {})
    current = data.get("channels", {}).get("default")
    rows = []
    for slug, entry in sorted(versions.items()):
        path = MODELS_DIR / entry["adapter"]["dir"]
        if only_existing and not (path / "adapter_config.json").exists():
            continue
        ok, reason = (True, "ok") if entry.get("state") == "archived" else verify_adapter(slug)
        rows.append({
            "id": slug, "label": entry.get("label", slug), "slogan": entry.get("slogan", ""),
            "note": entry.get("note", ""), "dir": entry["adapter"]["dir"],
            "state": entry.get("state", "candidate"),
            "sha256": (entry["adapter"].get("sha256") or "")[:12],
            "bytes": entry["adapter"].get("bytes", 0),
            "verified": ok, "verify_note": reason if not ok else "",
            "text_encoder": bool(entry["adapter"].get("text_encoder")),
            "eval": entry.get("eval", {}),
            "needs_review": bool(entry.get("needs_review")),
            "default": slug == current,
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
