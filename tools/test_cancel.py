"""test_cancel.py — 验证「取消排队任务」：提交 2 个任务，取消排队中的第 2 个。

运行前先启动本机后端（python server.py）。
"""
import os
import sys
import time
import requests

BASE = os.environ.get("BACKEND", "http://127.0.0.1:8001")
H = {"X-API-Key": os.environ.get("API_TOKEN", "change-me"), "Content-Type": "application/json"}


def wait(job_id, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = requests.get(f"{BASE}/jobs/{job_id}", headers=H, timeout=30).json()
        if m["status"] in ("done", "failed", "expired", "cancelled"):
            return m
        time.sleep(1)
    return {"status": "timeout"}


body = {"prompt": "a calm lake at dusk", "style": "写实摄影", "res": "512",
        "steps": 20, "sampler": "dpmpp2m_karras"}

a = requests.post(f"{BASE}/generate", json=body | {"seed": 11}, headers=H, timeout=60).json()
b = requests.post(f"{BASE}/generate", json=body | {"seed": 22}, headers=H, timeout=60).json()
print("submitted:", a["job_id"], a["status"], "|", b["job_id"], b["status"])

first = requests.get(f"{BASE}/jobs/{a['job_id']}", headers=H, timeout=30).json()
second = requests.get(f"{BASE}/jobs/{b['job_id']}", headers=H, timeout=30).json()
running_id = a["job_id"] if first["status"] == "running" else b["job_id"]
queued_id = b["job_id"] if running_id == a["job_id"] else a["job_id"]
print(f"running={running_id}  queued={queued_id}")

r = requests.delete(f"{BASE}/jobs/{queued_id}", headers=H, timeout=30)
print("DELETE queued ->", r.status_code, r.text[:120])
ok_cancel = r.status_code == 200 and r.json().get("status") == "cancelled"

r2 = requests.delete(f"{BASE}/jobs/{running_id}", headers=H, timeout=30)
print("DELETE running ->", r2.status_code, r2.text[:120])
ok_guard = r2.status_code == 409

m = requests.get(f"{BASE}/jobs/{queued_id}", headers=H, timeout=30).json()
print("cancelled job meta:", {k: m[k] for k in ("status", "cancelled", "files")})
ok_state = m["status"] == "cancelled" and m["cancelled"] is True and m["files"] == 0

final = wait(running_id)
print("running job finished as:", final["status"])

print("\nRESULT:", "ALL PASS" if (ok_cancel and ok_guard and ok_state) else "SOME FAIL")
sys.exit(0 if (ok_cancel and ok_guard and ok_state) else 1)
