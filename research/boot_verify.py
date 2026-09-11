"""Optional live smoke test; endpoint comes from LANDSCAPE_PROXY_URL."""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

ROOT = Path(os.environ.get("LANDSCAPE_ROOT", Path(__file__).resolve().parents[1])).resolve()
PROXY = os.environ.get("LANDSCAPE_PROXY_URL", "").rstrip("/")
if not PROXY:
    raise SystemExit("Set LANDSCAPE_PROXY_URL to a private proxy URL before running this live test.")

def get(url, **kwargs):
    return requests.get(url, timeout=60, **kwargs)

def post(url, **kwargs):
    return requests.post(url, timeout=60, **kwargs)

r = get(PROXY + "?op=health")
print("proxy health:", r.status_code, r.text[:160])
r.raise_for_status()
body = {"prompt": "a dramatic mountain valley at sunrise, mist, alpine lake, golden light", "style": "写实摄影", "res": "512", "steps": 20, "seed": 2026, "enhance": 0}
jid = post(PROXY + "?op=generate", json=body).json()["job_id"]
print("job submitted:", jid)
for _ in range(40):
    time.sleep(5)
    job = get(PROXY + "?op=job&id=" + jid).json()
    if job["status"] in ("done", "failed"):
        break
print("final:", job["status"], "bytes:", job.get("bytes"), "err:", job.get("error"))
if job["status"] == "done":
    image = get(PROXY + "?op=image&id=" + jid)
    print("image:", image.status_code, image.headers.get("Content-Type"), len(image.content), "bytes")
    out = ROOT / "outputs" / "boot_live_test.webp"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(image.content)
    print("saved", out.relative_to(ROOT))
