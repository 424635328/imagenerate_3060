"""End-to-end API check against a running backend.

Reads the target from the environment so no host, tunnel domain or token is ever
hard-coded:

    BACKEND_URL (default http://127.0.0.1:8001)   API_TOKEN (required)
    E2E_STEPS   (default 8)                       E2E_RES   (default 512)
    E2E_SKIP_UPLOAD=1 to skip the img2img leg

Checks: /health shape, /warmup, progress + progressive preview, final bytes,
chunk assembly integrity, uploaded reference image path, and /gc reclaim.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time

import requests

BASE = os.environ.get("BACKEND_URL", "http://127.0.0.1:8001").rstrip("/")
TOKEN = os.environ.get("API_TOKEN", "")
STEPS = int(os.environ.get("E2E_STEPS", "8"))
RES = os.environ.get("E2E_RES", "512")
HEAD = {"X-API-Key": TOKEN}
TIMEOUT = int(os.environ.get("E2E_TIMEOUT", "600"))

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'} {label}{f' :: {detail}' if detail else ''}", flush=True)
    if not ok:
        failures.append(label)
    return ok


def wait_for_job(job_id: str) -> dict:
    """Poll a job, recording whether progress and previews actually appeared."""
    seen_progress = False
    preview_revs: set[int] = set()
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        meta = requests.get(f"{BASE}/jobs/{job_id}", timeout=30).json()
        if 0 < meta.get("progress", 0) < 1:
            seen_progress = True
        if meta.get("preview_ready"):
            preview_revs.add(meta.get("preview_rev", 0))
        if meta["status"] in {"done", "failed", "expired"}:
            meta["_seen_progress"] = seen_progress
            meta["_preview_revs"] = sorted(preview_revs)
            return meta
        time.sleep(0.5)
    raise TimeoutError("job did not settle")


def make_reference_png() -> str:
    from PIL import Image
    img = Image.new("RGB", (768, 512))
    for x in range(768):
        for y in range(0, 512, 64):
            img.putpixel((x, y), (x % 255, y % 255, 128))
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=80)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    if not TOKEN:
        print("FAIL API_TOKEN is not set in the environment")
        return 2

    health = requests.get(f"{BASE}/health", timeout=30).json()
    check("health ok", health.get("ok") is True, health.get("format", ""))
    for key in ("memory", "storage", "pipeline", "sec_per_step"):
        check(f"health.{key} present", key in health)

    warm = requests.post(f"{BASE}/warmup", json={"fast": False, "sampler": "dpmpp2m_karras"},
                         headers=HEAD, timeout=900).json()
    check("warmup loaded", warm.get("loaded") is True, f"{warm.get('seconds')}s")
    check("pipeline reported as loaded",
          requests.get(f"{BASE}/health", timeout=30).json()["pipeline"]["model_loaded"] is True)

    body = {"prompt": "a misty alpine lake at dawn", "res": RES, "steps": STEPS,
            "cfg": 7.5, "seed": 12345, "sampler": "dpmpp2m_karras"}
    submitted = requests.post(f"{BASE}/generate", json=body, headers=HEAD, timeout=60).json()
    check("generate accepted", "job_id" in submitted, str(submitted.get("eta")))
    meta = wait_for_job(submitted["job_id"])
    check("job done", meta["status"] == "done", meta.get("error") or "")
    check("progress reported during denoise", meta["_seen_progress"])
    check("progressive previews written", len(meta["_preview_revs"]) >= 1, str(meta["_preview_revs"]))
    check("result has bytes", meta["bytes"] > 0, f"{meta['bytes']} B")
    check("seed honoured", meta["seed"] == 12345)
    check("steps clamped into range", 2 <= meta["steps"] <= 60, str(meta["steps"]))

    preview = requests.get(f"{BASE}/preview/{submitted['job_id']}", timeout=30)
    check("preview served as webp", preview.headers.get("content-type") == "image/webp",
          f"{len(preview.content)} B")
    check("preview is not cached", preview.headers.get("cache-control") == "no-store")

    result = requests.get(f"{BASE}/result/{submitted['job_id']}", timeout=60)
    check("result matches reported size", len(result.content) == meta["bytes"])

    # Chunked transfer must reassemble byte-for-byte identically.
    assembled = bytearray()
    off, span = 0, max(1024, meta["bytes"] // 3)
    while off < meta["bytes"]:
        part = requests.get(f"{BASE}/chunk/{submitted['job_id']}",
                            params={"off": off, "length": span}, timeout=60)
        assembled += part.content
        off += len(part.content)
    check("chunk assembly identical", bytes(assembled) == result.content,
          f"{len(assembled)}/{meta['bytes']} bytes")

    if os.environ.get("E2E_SKIP_UPLOAD") != "1":
        upload_body = dict(body, seed=999, init_image=make_reference_png(), strength=0.45)
        up = requests.post(f"{BASE}/generate", json=upload_body, headers=HEAD, timeout=120).json()
        check("upload job accepted", "job_id" in up, str(up.get("eta")))
        up_meta = wait_for_job(up["job_id"])
        check("img2img job done", up_meta["status"] == "done", up_meta.get("error") or "")
        check("backend recorded reference image", up_meta.get("has_init") is True)
        check("img2img reports strength", bool(up_meta.get("strength")), str(up_meta.get("strength")))

        oversize = "data:image/webp;base64," + base64.b64encode(b"\x00" * (8 * 1024 * 1024)).decode()
        bad = requests.post(f"{BASE}/generate", json=dict(body, init_image=oversize), headers=HEAD, timeout=60)
        check("oversized upload rejected", bad.status_code in {400, 413}, str(bad.status_code))

    gc = requests.post(f"{BASE}/gc", headers=HEAD, timeout=120).json()
    check("gc responds with storage stats", "storage" in gc and "memory" in gc)

    unauthorized = requests.post(f"{BASE}/generate", json=body, timeout=30)
    check("missing key rejected", unauthorized.status_code == 401, str(unauthorized.status_code))

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
