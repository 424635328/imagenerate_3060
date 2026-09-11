"""Runtime governance for the local GPU backend.

One place for everything that used to be scattered across server.py:

* ``collect_gpu``      - release Python/CUDA garbage without touching the pipeline.
* ``memory_snapshot``  - VRAM/host numbers reported by ``/health``.
* ``ResultStore``      - result-file retention: count limit, disk quota, orphans.
* ``IdleGovernor``     - unload the pipeline after an idle period.
* ``DurationStats``    - rolling generation timings used for ETA.

Paths come from ``config`` only; nothing here knows a machine-specific location.
"""
from __future__ import annotations

import gc
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

LOG = logging.getLogger("landscape.runtime")


def collect_gpu(deep: bool = False) -> None:
    """Drop dead references and cached CUDA blocks.

    ``deep=True`` additionally synchronizes and resets peak stats; use it after
    unloading a pipeline, not between two queued jobs (it costs a few ms).
    """
    gc.collect()
    try:
        import torch
    except ImportError:  # torch is optional for tooling/tests
        return
    if not torch.cuda.is_available():
        return
    if deep:
        try:
            torch.cuda.synchronize()
        except Exception:  # pragma: no cover - driver hiccup must not kill a job
            LOG.debug("cuda synchronize failed", exc_info=True)
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover
        LOG.debug("ipc_collect failed", exc_info=True)
    if deep:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # pragma: no cover
            pass


def memory_snapshot() -> dict:
    """Best-effort memory report; never raises, never leaks paths."""
    info: dict = {"cuda": False, "gc_objects": len(gc.get_objects())}
    try:
        import torch
    except ImportError:
        return info
    if not torch.cuda.is_available():
        return info
    free = total = 0
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:  # pragma: no cover
        pass
    mb = 1024 * 1024
    info.update(
        cuda=True,
        allocated_mb=round(torch.cuda.memory_allocated() / mb, 1),
        reserved_mb=round(torch.cuda.memory_reserved() / mb, 1),
        peak_mb=round(torch.cuda.max_memory_allocated() / mb, 1),
        free_mb=round(free / mb, 1),
        total_mb=round(total / mb, 1),
    )
    return info


@dataclass
class ResultStore:
    """Retention for generated files: newest-N jobs plus a hard disk quota."""

    directory: Path
    quota_bytes: int = 0
    max_jobs: int = 500
    removed_files: int = 0
    removed_bytes: int = 0

    def ensure(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def remove(self, path: str | Path | None) -> None:
        if not path:
            return
        target = Path(path)
        try:
            size = target.stat().st_size if target.exists() else 0
            target.unlink(missing_ok=True)
            self.removed_files += 1
            self.removed_bytes += size
        except OSError:
            LOG.warning("could not remove a result file")

    def dir_bytes(self) -> int:
        total = 0
        for path in self.directory.glob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    def sweep_orphans(self, keep: set[Path], older_than: float) -> int:
        """Delete files not referenced by any live job (e.g. after a restart)."""
        removed = 0
        now = time.time()
        for path in self.directory.glob("*"):
            try:
                if not path.is_file() or path.resolve() in keep:
                    continue
                if now - path.stat().st_mtime < older_than:
                    continue
            except OSError:
                continue
            self.remove(path)
            removed += 1
        return removed

    def enforce_quota(self, protected: set[Path], on_delete=None) -> int:
        """Delete oldest files until the directory fits the quota (LRU by mtime)."""
        if self.quota_bytes <= 0:
            return 0
        try:
            files = sorted(
                (p for p in self.directory.glob("*") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return 0
        total = sum(p.stat().st_size for p in files if p.exists())
        removed = 0
        for path in files:
            if total <= self.quota_bytes:
                break
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in protected:
                continue
            size = path.stat().st_size if path.exists() else 0
            self.remove(path)
            total -= size
            removed += 1
            if on_delete:
                on_delete(resolved)
        return removed

    def stats(self) -> dict:
        used = self.dir_bytes()
        return {
            "results_mb": round(used / (1024 * 1024), 1),
            "quota_mb": round(self.quota_bytes / (1024 * 1024), 1) if self.quota_bytes else 0,
            "usage_pct": round(used * 100 / self.quota_bytes, 1) if self.quota_bytes else 0,
            "reclaimed_files": self.removed_files,
            "reclaimed_mb": round(self.removed_bytes / (1024 * 1024), 1),
        }


@dataclass
class IdleGovernor:
    """Unload the GPU pipeline when the backend has been idle long enough."""

    idle_seconds: int
    unload: callable
    last_active: float = field(default_factory=time.time)
    unloads: int = 0
    loaded: bool = False

    def mark_active(self, loaded: bool = True) -> None:
        self.last_active = time.time()
        self.loaded = loaded

    def idle_for(self) -> float:
        return time.time() - self.last_active

    def maybe_unload(self) -> bool:
        if self.idle_seconds <= 0 or not self.loaded:
            return False
        if self.idle_for() < self.idle_seconds:
            return False
        try:
            self.unload()
        except Exception:  # pragma: no cover - unload must never crash the loop
            LOG.exception("idle unload failed")
            return False
        self.loaded = False
        self.unloads += 1
        collect_gpu(deep=True)
        return True

    def stats(self) -> dict:
        return {
            "model_loaded": self.loaded,
            "idle_sec": round(self.idle_for(), 1),
            "idle_unload_sec": self.idle_seconds,
            "idle_unloads": self.unloads,
        }


class DurationStats:
    """Rolling per-workload timings so the frontend can show a real ETA."""

    def __init__(self, window: int = 24) -> None:
        self._per_step: dict[str, deque[float]] = {}
        self._window = window

    @staticmethod
    def key(res: str | int, sampler: str, enhance: int = 0) -> str:
        return f"{res}:{(sampler or 'default').lower()}:{1 if enhance else 0}"

    def record(self, key: str, seconds: float, steps: int) -> None:
        steps = max(1, int(steps))
        bucket = self._per_step.setdefault(key, deque(maxlen=self._window))
        bucket.append(max(0.02, seconds / steps))

    def estimate(self, key: str, steps: int, fallback_per_step: float = 0.42) -> float:
        bucket = self._per_step.get(key)
        per_step = sum(bucket) / len(bucket) if bucket else fallback_per_step
        return round(per_step * max(1, int(steps)), 2)

    def stats(self) -> dict:
        return {k: round(sum(v) / len(v), 3) for k, v in self._per_step.items() if v}
