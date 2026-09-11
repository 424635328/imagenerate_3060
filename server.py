"""Landscape·Art local GPU backend.

Compatible with the previous API (all old fields still accepted) and adds:

* bounded quality/fast samplers (DPM++ / Euler / DDIM / LCM / TCD)
* real step progress + progressive latent preview + server-side ETA
* ``init_image`` img2img from browser uploads (WebP, size checked)
* ``/warmup`` to move the cold-start cost off the first click
* parallel-safe byte chunks for large results
* GC governance: pipeline idle unload, job-count limit, disk quota, janitor loop
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from config import (
    ALLOW_ORIGINS, ALLOW_UPLOAD, ALLOWED_UPLOAD_FORMATS, API_TOKEN, CHUNK_SIZE,
    IDLE_UNLOAD_SEC, JANITOR_SEC, JOB_TIMEOUT, MAX_INIT_EDGE, MAX_INIT_PIXELS,
    MAX_JOBS, MAX_QUEUE, MAX_UPLOAD_BYTES, OUT_FORMAT, OUT_QUALITY, PREVIEW_EVERY,
    PREVIEW_MAX, RATE_PER_MIN, RESULTS_DIR, RESULTS_QUOTA_BYTES, clamp_cfg,
    clamp_steps, clamp_strength, ensure_runtime_dirs, output_mime, output_suffix,
)
from runtime import DurationStats, IdleGovernor, ResultStore, collect_gpu, memory_snapshot
from app import _gen_one, _size, build_prompt, current_mode, get_i2i, unload_pipes, warmup

LOG = logging.getLogger("landscape.server")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())
DEFAULT_TOKEN_WARN = API_TOKEN == "change-me"

_jobs: dict[str, dict] = {}
_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=MAX_QUEUE)
_rate: dict[str, deque[float]] = defaultdict(deque)
_total_submitted = 0

_store = ResultStore(RESULTS_DIR, quota_bytes=RESULTS_QUOTA_BYTES, max_jobs=MAX_JOBS)
_idle = IdleGovernor(idle_seconds=IDLE_UNLOAD_SEC, unload=unload_pipes)
_timings = DurationStats()

# --------------------------------------------------------------------------
# Result cache: identical requests (same prompt/style/seed/sampler/steps/…)
# short-circuit the GPU entirely.  Cache entries live in their own directory so
# job pruning can never delete them, and every job owns a hard link (cheap, no
# extra bytes) into the normal results directory.
# --------------------------------------------------------------------------
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "1").lower() not in {"0", "false", "no"}
CACHE_MAX_FILES = max(0, int(os.environ.get("CACHE_MAX_FILES", "200")))
CACHE_DIR = RESULTS_DIR / "cache"
_cache_index: dict[str, dict] = {}


def _cache_load() -> None:
    global _cache_index
    if not CACHE_ENABLED or CACHE_MAX_FILES <= 0:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index_path = CACHE_DIR / "index.json"
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            _cache_index = {k: v for k, v in raw.items() if isinstance(v, dict)}
    except FileNotFoundError:
        _cache_index = {}
    except Exception:
        LOG.warning("cache index unreadable; starting empty")
        _cache_index = {}
    # Drop entries whose payload vanished (manual cleanup, disk reset, …).
    for key in [k for k, v in _cache_index.items() if not (CACHE_DIR / v.get("name", "")).exists()]:
        _cache_index.pop(key, None)


def _cache_save() -> None:
    if not CACHE_ENABLED or CACHE_MAX_FILES <= 0:
        return
    try:
        tmp = CACHE_DIR / "index.json.tmp"
        tmp.write_text(json.dumps(_cache_index, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, CACHE_DIR / "index.json")
    except Exception:
        LOG.warning("could not persist cache index")


def _cache_key(req: "GenerateReq", seed: int, steps: int, cfg: float, sampler: str,
               fast: bool, init_hash: str) -> str:
    """Stable fingerprint of everything that can change the pixels."""
    parts = [
        req.prompt.strip(), req.style, req.res, req.aspect, str(steps), f"{cfg:.3f}",
        str(seed), req.neg.strip(), sampler, "1" if fast else "0",
        str(int(req.highres)), str(int(req.enhance)), req.sr_model,
        f"{float(req.enhance_strength):.3f}", str(int(req.enhance_steps)),
        "1" if req.upscale else "0", init_hash, f"{float(req.strength):.3f}",
    ]
    return hashlib.sha1("\u0001".join(parts).encode("utf-8")).hexdigest()[:20]


def _cache_lookup(key: str) -> Path | None:
    entry = _cache_index.get(key)
    if not entry:
        return None
    path = CACHE_DIR / entry.get("name", "")
    if not path.exists():
        _cache_index.pop(key, None)
        return None
    entry["hits"] = int(entry.get("hits", 0)) + 1
    entry["last"] = time.time()
    _cache_save()
    return path


def _cache_store(key: str, path: Path) -> None:
    """Copy a finished result into the cache and trim it to CACHE_MAX_FILES."""
    if not CACHE_ENABLED or CACHE_MAX_FILES <= 0:
        return
    target = CACHE_DIR / f"{key}{path.suffix}"
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    except Exception:
        LOG.warning("cache store failed for %s", key)
        return
    _cache_index[key] = {"name": target.name, "created": time.time(), "last": time.time(), "hits": 0}
    if len(_cache_index) > CACHE_MAX_FILES:
        for old in sorted(_cache_index, key=lambda k: _cache_index[k].get("last", 0))[: len(_cache_index) - CACHE_MAX_FILES]:
            entry = _cache_index.pop(old, None)
            if entry:
                (CACHE_DIR / entry.get("name", "")).unlink(missing_ok=True)
    _cache_save()


def _cache_link(key: str, job_id: str) -> Path | None:
    """Give a cache hit its own results path via a hard link (fallback: copy)."""
    source = _cache_lookup(key)
    if source is None:
        return None
    target = RESULTS_DIR / f"{job_id}{source.suffix}"
    try:
        target.unlink(missing_ok=True)
        os.link(source, target)
    except Exception:
        try:
            shutil.copy2(source, target)
        except Exception:
            LOG.warning("cache link failed for %s", job_id)
            return None
    return target



class GenerateReq(BaseModel):
    prompt: str = Field(default="", max_length=2000)
    style: str = Field(default="写实摄影", max_length=40)
    res: str = "512"
    aspect: str = "方"
    steps: int = Field(default=24, ge=2, le=500)
    cfg: float = Field(default=7.5, ge=1.0, le=30.0)
    seed: int = Field(default=-1, ge=-1, le=2**31 - 1)
    neg: str = Field(default="blurry, low quality, watermark, text, distorted, oversaturated, bad anatomy", max_length=2000)
    upscale: bool = False
    highres: int = Field(default=0, ge=0, le=4096)
    enhance: int = Field(default=0, ge=0, le=4096)
    sr_model: str = Field(default="ultrasharp", max_length=200)
    enhance_strength: float = Field(default=0.30, ge=0.0, le=0.95)
    enhance_steps: int = Field(default=0, ge=0, le=500)
    fast: bool = False
    sampler: str = Field(default="dpmpp2m_karras", max_length=32)
    # Server-side batch: one queue slot, one poll, N images (seeds seed..seed+N-1).
    batch: int = Field(default=1, ge=1, le=4)
    # img2img from the browser: compressed WebP/JPEG data URL or raw base64.
    init_image: str = Field(default="", max_length=MAX_UPLOAD_BYTES * 2)
    strength: float = Field(default=0.55, ge=0.05, le=0.95)


class WarmupReq(BaseModel):
    fast: bool = False
    sampler: str = Field(default="dpmpp2m_karras", max_length=32)


def _check_key(x_key: str = Header(default="", alias="X-API-Key")):
    if x_key != API_TOKEN:
        raise HTTPException(401, "invalid API key")


def _rate_ok(ip: str) -> bool:
    now = time.time()
    bucket = _rate[ip]
    while bucket and bucket[0] < now - 60:
        bucket.popleft()
    if len(bucket) >= RATE_PER_MIN:
        return False
    bucket.append(now)
    # Bound the map itself, not just each bucket: without this the per-IP keys
    # of a long-running server only ever grow.
    if len(_rate) > 512:
        for stale in [key for key, seen in _rate.items() if not seen or seen[-1] < now - 120]:
            _rate.pop(stale, None)
    return True


def _job_files(job: dict) -> list[Path]:
    paths = [Path(p) for p in job.get("files", []) if p]
    for key in ("file", "preview"):
        if job.get(key):
            paths.append(Path(job[key]))
    seen: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen


def _live_files() -> set[Path]:
    live: set[Path] = set()
    for job in _jobs.values():
        for path in _job_files(job):
            try:
                if path.exists():
                    live.add(path.resolve())
            except OSError:
                continue
    return live


def _prune_jobs() -> None:
    """Keep only the newest MAX_JOBS jobs; drop their files with the record."""
    if len(_jobs) <= MAX_JOBS:
        return
    finished = sorted(
        (j for j in _jobs.values() if j["status"] not in {"queued", "running"}),
        key=lambda j: j["created"],
    )
    for job in finished[: len(_jobs) - MAX_JOBS]:
        _jobs.pop(job["id"], None)
        for path in _job_files(job):
            _store.remove(path)


def _forget_file(resolved: Path) -> None:
    """Mark jobs whose file was reclaimed so the frontend can show 已过期."""
    for job in _jobs.values():
        for key in ("file", "preview"):
            value = job.get(key)
            if not value:
                continue
            try:
                same = Path(value).resolve() == resolved
            except OSError:
                continue
            if not same:
                continue
            job[key] = None
            if key == "file" and job["status"] == "done":
                job["status"] = "expired"


def _prune_storage() -> None:
    protected = {p.resolve() for job in _jobs.values()
                 if job["status"] in {"queued", "running"}
                 for p in _job_files(job) if p.exists()}
    _store.enforce_quota(protected, on_delete=_forget_file)


def _save_image(img: Image.Image, path: Path, max_edge: int | None = None) -> None:
    """Write WebP/JPEG atomically so pollers never read a half-written preview."""
    img = img.convert("RGB")
    if max_edge and max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    if path.suffix.lower() == ".webp":
        img.save(tmp, "WEBP", quality=OUT_QUALITY, method=6)
    else:
        img.save(tmp, "JPEG", quality=OUT_QUALITY, optimize=True, progressive=True)
    os.replace(tmp, path)


def _safe_error(exc: Exception) -> str:
    # Never return local paths, tokens, or a full traceback to the browser.
    message = str(exc).replace("\\", "/")
    message = message.rsplit("/", 1)[-1]
    return f"{type(exc).__name__}: {message[:240]}"


# Magic-byte signatures: never trust the data-URL MIME type or a filename.
_MAGIC = ((b"\xff\xd8\xff", "JPEG"), (b"\x89PNG\r\n\x1a\n", "PNG"))
_DATA_URL_RE = re.compile(r"^data:image/(png|jpe?g|webp);base64,", re.I)


def _sniff_format(raw: bytes) -> str | None:
    """Identify the container from its first bytes (SVG/HTML/EXE ⇒ None)."""
    for signature, name in _MAGIC:
        if raw.startswith(signature):
            return name
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "WEBP"
    return None


def _strip_metadata(img: Image.Image) -> Image.Image:
    """Rebuild the bitmap in a fresh canvas: EXIF/XMP/IPTC/ICC/GPS cannot survive."""
    oriented = ImageOps.exif_transpose(img) or img      # honour orientation first
    clean = Image.new("RGB", oriented.size)
    clean.paste(oriented.convert("RGB"))
    return clean


def _sanitize_upload_bytes(raw: bytes) -> Image.Image:
    """Upload hardening pipeline (see AGENTS.md → Image Upload).

    magic bytes → decode → pixel/size budget → metadata strip → re-encode by the
    normal save path.  The uploaded bytes are never written to disk, so no
    original (with GPS/EXIF) is retained anywhere.
    """
    sniffed = _sniff_format(raw)
    if sniffed is None:
        raise HTTPException(400, "unsupported file: only PNG/JPEG/WebP are accepted")
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            detected = (probe.format or "").upper()
            size = probe.size
    except Image.DecompressionBombError:
        raise HTTPException(413, "image is too large to decode")
    except Exception:
        raise HTTPException(400, "reference image could not be decoded")
    if detected not in ALLOWED_UPLOAD_FORMATS:
        raise HTTPException(400, "reference image must be WebP/JPEG/PNG")
    if detected != sniffed:
        raise HTTPException(400, "file content does not match a supported image format")
    pixels = size[0] * size[1]
    if pixels > MAX_INIT_PIXELS:
        raise HTTPException(413, f"image has too many pixels ({pixels:,} > {MAX_INIT_PIXELS:,})")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(400, "reference image could not be decoded")
    clean = _strip_metadata(img)
    if max(clean.size) > MAX_INIT_EDGE:
        clean.thumbnail((MAX_INIT_EDGE, MAX_INIT_EDGE), Image.Resampling.LANCZOS)
    return clean


def _decode_init_image(payload: str) -> Image.Image | None:
    """Decode a browser-uploaded reference image through the sanitization pipeline."""
    if not payload:
        return None
    if not ALLOW_UPLOAD:
        raise HTTPException(400, "uploads are disabled on this backend")
    if payload.startswith("data:"):
        if not _DATA_URL_RE.match(payload):
            raise HTTPException(400, "only base64 PNG/JPEG/WebP data URLs are accepted")
        data = payload.split(",", 1)[1]
    else:
        data = payload
    if len(data) > MAX_UPLOAD_BYTES * 2:
        raise HTTPException(413, "reference image too large")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "reference image is not valid base64")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "reference image too large")
    return _sanitize_upload_bytes(raw)


def _set_stage(job: dict, stage: str, done: int = 0, total: int = 0) -> None:
    job["stage"] = stage
    if total:
        job["steps_done"] = done
        job["steps_total"] = total
    job["updated"] = time.time()


def generate_sync(req: GenerateReq, job_id: str, init_img: Image.Image | None) -> list[str]:
    """Run one bounded job on the single GPU worker: preview first, then final.

    Server-side batching: ``req.batch`` images (consecutive seeds) are produced by
    one queue slot with a single poll, which removes N-1 submissions, N-1 queue
    entries and N-1 polling streams compared with the client looping.
    """
    from app import DEFAULT_NEG
    base_seed = req.seed if req.seed >= 0 else random.randint(0, 2**31 - 1)
    sampler = req.sampler.lower()
    fast = bool(req.fast or sampler in {"lcm", "tcd"})
    steps = clamp_steps(req.steps, fast=fast)
    cfg = clamp_cfg(req.cfg, fast=fast)
    strength = clamp_strength(req.strength)
    count = max(1, min(int(req.batch or 1), 4))
    seeds = [base_seed + i for i in range(count)]
    job = _jobs[job_id]
    job.update(seed=base_seed, seeds=seeds, count=count, steps=steps, cfg=cfg, sampler=sampler,
               fast=fast, started=time.time(), strength=strength if init_img is not None else None)
    prompt = build_prompt(req.prompt, req.style, False) or "a scenic landscape"
    neg = req.neg or DEFAULT_NEG
    w, h = _size(req.res, req.aspect)
    preview_path = RESULTS_DIR / f"{job_id}.preview.webp"
    job["preview"] = str(preview_path)

    def on_progress(done: int, total: int) -> None:
        _set_stage(job, "denoise", done, total)

    def on_preview(img: Image.Image, done: int, total: int) -> None:
        _save_image(img, preview_path, PREVIEW_MAX)
        job["preview_ready"] = True
        job["preview_rev"] = job.get("preview_rev", 0) + 1
        job["preview_kind"] = "latent"

    files: list[str] = []
    for index, seed in enumerate(seeds):
        job["batch_index"] = index
        _set_stage(job, "denoise", 0, steps)
        img = _gen_one(prompt, w, h, steps, cfg, seed, neg, init_img, strength, 1.0, None,
                       fast=fast, sampler=sampler, progress_cb=on_progress,
                       preview_cb=on_preview, preview_every=PREVIEW_EVERY if index == 0 else 0)

        # A real (VAE-decoded) preview is written before any optional high-res
        # work, so the browser can show the finished composition immediately.
        if index == 0:
            _save_image(img, preview_path, PREVIEW_MAX)
            job["preview_ready"] = True
            job["preview_rev"] = job.get("preview_rev", 0) + 1
            job["preview_kind"] = "base"
            job["base_seconds"] = round(time.time() - job["started"], 2)

        if req.highres > 0 and req.highres > max(w, h):
            _set_stage(job, "highres")
            img = get_i2i(fast, sampler)(
                prompt=prompt, negative_prompt=neg, image=img.resize((req.highres, req.highres)),
                strength=0.45,
                num_inference_steps=(max(4, steps) if fast else max(12, int(steps * 0.6))),
                guidance_scale=cfg,
            ).images[0]
        if req.upscale:
            _set_stage(job, "upscale")
            from app import _sr_up
            img = _sr_up(img)
        if req.enhance > 0 and req.enhance > max(img.size):
            _set_stage(job, "enhance")
            import enhance as enhancer
            refine_steps = req.enhance_steps or (max(6, steps) if fast else max(16, int(steps * 0.5)))
            img = enhancer.enhance(
                img, req.enhance, enhancer.get_sr(req.sr_model), get_i2i(fast, sampler),
                prompt, neg, strength=req.enhance_strength, steps=clamp_steps(refine_steps, fast),
                cfg=cfg, tile=512, overlap=96, seed=seed, sharpen=True,
            )
        _set_stage(job, "encode")
        suffix = output_suffix()
        final_path = RESULTS_DIR / (f"{job_id}{suffix}" if index == 0 else f"{job_id}_{index}{suffix}")
        _save_image(img, final_path)
        files.append(str(final_path))
        job["files"] = list(files)
        if index == 0:
            job["format"] = OUT_FORMAT
            job["width"], job["height"] = img.size
    _timings.record(DurationStats.key(req.res, sampler, req.enhance),
                    max(0.05, time.time() - job["started"]), steps * count)
    _set_stage(job, "done", steps, steps)
    return files


async def worker() -> None:
    while True:
        job_id = await _queue.get()
        job = _jobs.get(job_id)
        if not job:
            _queue.task_done()
            continue
        if job.get("cancelled"):
            # Cancelled while queued: never touch the GPU.
            job["status"] = "cancelled"
            job["updated"] = time.time()
            _queue.task_done()
            continue
        job["status"] = "running"
        _idle.mark_active()
        try:
            produced = await asyncio.wait_for(
                asyncio.to_thread(generate_sync, job["req"], job_id, job.pop("init_img", None)),
                timeout=JOB_TIMEOUT)
            job["files"] = [str(p) for p in produced]
            job["file"] = job["files"][0] if job["files"] else None
            job["status"] = "done"
            job["seconds"] = round(time.time() - job.get("started", job["created"]), 2)
            # Cache single-image, non-img2img results so an identical request is instant.
            if (job["files"] and job.get("cache_key") and job["count"] == 1
                    and not job.get("has_init")):
                _cache_store(job["cache_key"], Path(job["files"][0]))
        except asyncio.TimeoutError:
            job["status"] = "failed"; job["error"] = "timeout"
        except Exception as exc:
            LOG.exception("job %s failed", job_id)
            job["status"] = "failed"; job["error"] = _safe_error(exc)
        finally:
            if job["status"] == "failed":
                for path in _job_files(job):
                    _store.remove(path)
                job["file"] = None; job["files"] = []; job["preview"] = None
            _idle.mark_active()
            _prune_jobs(); _prune_storage(); collect_gpu(); _queue.task_done()


async def janitor() -> None:
    """Periodic housekeeping: quota, orphan files, result cache, idle VRAM release."""
    while True:
        await asyncio.sleep(JANITOR_SEC)
        try:
            _prune_jobs()
            _prune_storage()
            _cache_trim()
            if _queue.empty() and not any(j["status"] == "running" for j in _jobs.values()):
                _store.sweep_orphans(_live_files(), older_than=max(JOB_TIMEOUT, 600))
                if _idle.maybe_unload():
                    LOG.info("pipeline unloaded after idle period")
        except Exception:  # pragma: no cover - housekeeping must never die
            LOG.exception("janitor iteration failed")


def _cache_trim() -> None:
    """Evict least-recently-used cache entries beyond CACHE_MAX_FILES."""
    if not CACHE_ENABLED or CACHE_MAX_FILES <= 0:
        return
    changed = False
    for key in [k for k, v in _cache_index.items() if not (CACHE_DIR / v.get("name", "")).exists()]:
        _cache_index.pop(key, None); changed = True
    overflow = len(_cache_index) - CACHE_MAX_FILES
    if overflow > 0:
        for old in sorted(_cache_index, key=lambda k: _cache_index[k].get("last", 0))[:overflow]:
            entry = _cache_index.pop(old, None)
            if entry:
                (CACHE_DIR / entry.get("name", "")).unlink(missing_ok=True)
            changed = True
    if changed:
        _cache_save()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_runtime_dirs()
    _store.ensure()
    _store.sweep_orphans(set(), older_than=0)
    _cache_load()
    LOG.info("result cache: %d entries (enabled=%s, max=%d)",
             len(_cache_index), CACHE_ENABLED, CACHE_MAX_FILES)
    tasks = [asyncio.create_task(worker()), asyncio.create_task(janitor())]
    yield
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
    unload_pipes()


app = FastAPI(title="LANDSCAPE·ART 推理后端", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOW_ORIGINS,
                   allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])


@app.get("/health")
def health():
    statuses = [j["status"] for j in _jobs.values()]
    return {
        "ok": True, "backend": "local-sd15", "format": OUT_FORMAT,
        "total_submitted": _total_submitted, "queued": statuses.count("queued"),
        "running": statuses.count("running"), "completed": statuses.count("done"),
        "failed": statuses.count("failed"), "queue": _queue.qsize(),
        "jobs": _total_submitted, "tracked_jobs": len(_jobs),
        "max_queue": MAX_QUEUE, "upload": ALLOW_UPLOAD,
        "max_upload_mb": round(MAX_UPLOAD_BYTES / (1024 * 1024), 2),
        "preview_every": PREVIEW_EVERY,
        "cache": {
            "enabled": bool(CACHE_ENABLED and CACHE_MAX_FILES > 0),
            "entries": len(_cache_index), "max": CACHE_MAX_FILES,
            "hits": sum(int(v.get("hits", 0)) for v in _cache_index.values()),
        },
        "memory": memory_snapshot(), "storage": _store.stats(),
        "pipeline": {**current_mode(), **_idle.stats()},
        "sec_per_step": _timings.stats(),
    }


@app.post("/warmup", dependencies=[Depends(_check_key)])
async def do_warmup(req: WarmupReq):
    """Load the pipeline now so the first generate does not pay cold start."""
    info = await asyncio.to_thread(warmup, req.fast, req.sampler)
    _idle.mark_active()
    return info


@app.post("/gc", dependencies=[Depends(_check_key)])
async def do_gc(unload: bool = False):
    """Manual housekeeping hook: quota sweep, optional pipeline unload."""
    if unload:
        await asyncio.to_thread(unload_pipes)
        _idle.loaded = False
    _prune_jobs(); _prune_storage()
    _store.sweep_orphans(_live_files(), older_than=0)
    collect_gpu(deep=True)
    return {"ok": True, "memory": memory_snapshot(), "storage": _store.stats(),
            "pipeline": {**current_mode(), **_idle.stats()}}


@app.post("/generate", dependencies=[Depends(_check_key)])
async def gen(req: GenerateReq, request: Request):
    global _total_submitted
    if not _rate_ok(request.client.host if request.client else "unknown"):
        raise HTTPException(429, "rate limited")
    if _queue.full():
        raise HTTPException(429, "queue full, try later")
    init_img = _decode_init_image(req.init_image)
    job_id = uuid.uuid4().hex[:12]
    _total_submitted += 1
    fast = bool(req.fast or req.sampler.lower() in {"lcm", "tcd"})
    steps = clamp_steps(req.steps, fast=fast)
    cfg = clamp_cfg(req.cfg, fast=fast)
    count = max(1, min(int(req.batch or 1), 4))
    sampler = req.sampler.lower()
    # Only a fully specified request can be cached: a random seed is new every time.
    cache_key = None
    if CACHE_ENABLED and req.seed >= 0 and count == 1 and init_img is None:
        cache_key = _cache_key(req, req.seed, steps, cfg, sampler, fast, "none")

    # ---- cache fast path: identical request → no queue, no GPU work ----
    if cache_key:
        hit = _cache_link(cache_key, job_id)
        if hit is not None:
            _jobs[job_id] = {
                "id": job_id, "status": "done", "cached": True,
                "req": req.model_copy(update={"init_image": ""}),
                "created": time.time(), "updated": time.time(),
                "file": str(hit), "files": [str(hit)], "preview": None,
                "preview_ready": False, "preview_rev": 0, "preview_kind": None, "error": None,
                "stage": "done", "steps_done": steps, "steps_total": steps,
                "seed": req.seed, "seeds": [req.seed], "count": 1, "batch_index": 0,
                "steps": steps, "cfg": cfg, "sampler": sampler, "fast": fast,
                "width": None, "height": None, "format": OUT_FORMAT,
                "has_init": False, "seconds": 0.0, "cache_key": cache_key,
                "eta": 0,
            }
            _prune_jobs()
            return {"job_id": job_id, "status": "done", "cached": True, "queue_position": 0,
                    "eta": 0, "count": 1, "seeds": [req.seed]}

    _jobs[job_id] = {
        "id": job_id, "status": "queued", "req": req.model_copy(update={"init_image": ""}),
        "created": time.time(), "updated": time.time(), "file": None, "files": [],
        "preview": None, "preview_ready": False, "preview_rev": 0, "preview_kind": None,
        "error": None, "stage": "queued", "steps_done": 0, "steps_total": steps,
        "init_img": init_img, "has_init": init_img is not None,
        "count": count,
        "seeds": [req.seed + i for i in range(count)] if req.seed >= 0 else [],
        "cache_key": cache_key, "cached": False,
        "eta": _timings.estimate(DurationStats.key(req.res, sampler, req.enhance), steps * count),
    }
    _prune_jobs()
    await _queue.put(job_id)
    return {"job_id": job_id, "status": "queued", "queue_position": _queue.qsize(),
            "eta": _jobs[job_id]["eta"], "count": count,
            "seeds": _jobs[job_id]["seeds"]}


def _job_meta(job: dict) -> dict:
    listed = [Path(p) for p in job.get("files", []) if p]
    if not listed and job.get("file"):
        listed = [Path(job["file"])]
    sizes = [p.stat().st_size if p.exists() else 0 for p in listed]
    preview = Path(job["preview"]) if job.get("preview") else None
    req = job["req"]
    total = max(1, job.get("steps_total", req.steps))
    done = min(job.get("steps_done", 0), total)
    elapsed = time.time() - job.get("started", job["created"])
    eta = job.get("eta", 0)
    if job["status"] == "running" and done:
        eta = max(0.0, round(elapsed / done * (total - done), 1))
    elif job["status"] in {"done", "failed", "expired"}:
        eta = 0
    return {
        "id": job["id"], "status": job["status"], "error": job.get("error"),
        "prompt": req.prompt, "style": req.style, "res": req.res, "aspect": req.aspect,
        "seed": job.get("seed", req.seed), "steps": job.get("steps", req.steps),
        "cfg": job.get("cfg", req.cfg), "sampler": job.get("sampler", req.sampler),
        "fast": job.get("fast", req.fast), "upscale": req.upscale,
        "highres": req.highres, "enhance": req.enhance, "has_init": job.get("has_init", False),
        "strength": job.get("strength"), "stage": job.get("stage", "queued"),
        "steps_done": done, "steps_total": total,
        "progress": round(done / total, 3) if job["status"] != "done" else 1.0,
        "eta": eta, "elapsed": round(elapsed, 1) if job.get("started") else 0,
        "seconds": job.get("seconds"), "base_seconds": job.get("base_seconds"),
        "width": job.get("width"), "height": job.get("height"),
        "format": job.get("format", OUT_FORMAT),
        "bytes": sizes[0] if sizes else 0,
        # --- batch / cache additions (additive: old clients keep working) ---
        "count": job.get("count", 1), "seeds": job.get("seeds", [job.get("seed", req.seed)]),
        "files": len(sizes), "bytes_each": sizes, "cached": bool(job.get("cached")),
        "cancelled": bool(job.get("cancelled")), "batch_index": job.get("batch_index", 0),
        "preview_ready": bool(preview and preview.exists()),
        "preview_rev": job.get("preview_rev", 0), "preview_kind": job.get("preview_kind"),
        "preview_bytes": preview.stat().st_size if preview and preview.exists() else 0,
    }


@app.delete("/jobs/{job_id}", dependencies=[Depends(_check_key)])
def cancel_job(job_id: str):
    """Cancel a job that is still queued.

    Running GPU work is deliberately NOT interrupted: killing a CUDA graph
    mid-step risks VRAM corruption, and the pipeline is shared.  Queued jobs are
    cancelled instantly and never touch the GPU.
    """
    found = _jobs.get(job_id)
    if not found:
        raise HTTPException(404, "not found")
    if found["status"] == "queued":
        found["cancelled"] = True
        found["status"] = "cancelled"
        found["stage"] = "cancelled"
        found["updated"] = time.time()
        _prune_jobs()
        return {"ok": True, "id": job_id, "status": "cancelled"}
    if found["status"] == "running":
        raise HTTPException(409, "job is running on the GPU and cannot be interrupted safely")
    return {"ok": True, "id": job_id, "status": found["status"], "note": "already settled"}


@app.get("/jobs/{job_id}")
def job(job_id: str):
    found = _jobs.get(job_id)
    if not found:
        raise HTTPException(404, "not found")
    return _job_meta(found)


@app.get("/jobs")
def jobs(limit: int = 20):
    limit = max(1, min(int(limit), 100))
    recent = sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)[:limit]
    return {"jobs": [_job_meta(j) for j in recent], "queue": _queue.qsize()}


@app.get("/preview/{job_id}")
def preview(job_id: str):
    found = _jobs.get(job_id)
    path = Path(found["preview"]) if found and found.get("preview") else None
    if not path or not path.exists():
        raise HTTPException(404, "preview not ready")
    # Previews are revised in place while denoising, so they must not be cached.
    return FileResponse(path, media_type="image/webp", headers={
        "Cache-Control": "no-store", "X-Preview-Rev": str(found.get("preview_rev", 0)),
        "X-Preview-Kind": str(found.get("preview_kind") or "")})


@app.get("/result/{job_id}")
def result(job_id: str, i: int = 0):
    found = _jobs.get(job_id)
    if not found:
        raise HTTPException(404, "not found")
    files = [Path(p) for p in found.get("files", []) if p]
    if not files and found.get("file"):
        files = [Path(found["file"])]
    if found["status"] != "done" or not files:
        return JSONResponse({"error": "result not ready"}, status_code=409)
    index = max(0, min(int(i), len(files) - 1))
    path = files[index]
    if not path.exists():
        return JSONResponse({"error": "result not ready"}, status_code=409)
    return FileResponse(path, media_type=output_mime(), headers={
        "Cache-Control": "public, max-age=31536000, immutable",
        "X-Job-Count": str(len(files)), "X-Job-Index": str(index)})


@app.get("/chunk/{job_id}")
def chunk(job_id: str, off: int = 0, length: int = CHUNK_SIZE, i: int = 0):
    found = _jobs.get(job_id)
    files = [Path(p) for p in (found or {}).get("files", []) if p]
    if not files and found and found.get("file"):
        files = [Path(found["file"])]
    if not found or found["status"] != "done" or not files:
        raise HTTPException(404, "not ready")
    index = max(0, min(int(i), len(files) - 1))
    path = files[index]
    if not path.exists():
        raise HTTPException(404, "not ready")
    total = path.stat().st_size
    off = max(0, min(int(off), total))
    length = max(1, min(int(length), 4 * 1024 * 1024, total - off))
    with path.open("rb") as stream:
        stream.seek(off)
        data = stream.read(length)
    return Response(content=data, media_type="application/octet-stream", headers={
        "X-Total": str(total), "X-Offset": str(off), "X-Length": str(len(data)),
        "X-Job-Count": str(len(files)), "X-Job-Index": str(index),
        "Cache-Control": "public, max-age=31536000, immutable", "Accept-Ranges": "bytes"})


if __name__ == "__main__":
    import uvicorn
    if DEFAULT_TOKEN_WARN:
        print("WARNING: API_TOKEN is still the placeholder; set a strong environment variable.")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8001"))
    print(f"LANDSCAPE·ART backend on http://{host}:{port} format={OUT_FORMAT} "
          f"idle_unload={IDLE_UNLOAD_SEC}s quota={RESULTS_QUOTA_BYTES // (1024 * 1024)}MB")
    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("LOG_LEVEL", "warning"))
