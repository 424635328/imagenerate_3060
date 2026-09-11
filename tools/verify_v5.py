"""verify_v5.py — 验证 v5：服务端批量、结果缓存、索引起图、immutable 头。

用法：先启动本机后端，再运行本脚本（令牌取自 API_TOKEN 环境变量）。
"""
import os, time
import requests

BASE = os.environ.get("BACKEND", "http://127.0.0.1:8001")
TOKEN = os.environ.get("API_TOKEN", "change-me")
H = {"X-API-Key": TOKEN, "Content-Type": "application/json"}


def post(path, body):
    r = requests.post(f"{BASE}{path}", json=body, headers=H, timeout=60)
    r.raise_for_status()
    return r.json()


def wait(job_id, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = requests.get(f"{BASE}/jobs/{job_id}", headers=H, timeout=30).json()
        if m["status"] in ("done", "failed", "expired"):
            return m, time.time() - t0
        time.sleep(1)
    raise TimeoutError(job_id)


print("== 1) 批内多图 (batch=2) ==")
body = {"prompt": "a tranquil alpine lake reflecting snow-capped peaks", "style": "写实摄影",
        "res": "512", "steps": 12, "seed": 4321, "sampler": "dpmpp2m_karras", "batch": 2}
sub = post("/generate", body)
print("   submit ->", sub)
meta, secs = wait(sub["job_id"])
print(f"   status={meta['status']} count={meta['count']} seeds={meta['seeds']} "
      f"files={meta['files']} bytes_each={meta['bytes_each']} {secs:.1f}s")
ok_batch = meta["status"] == "done" and meta["count"] == 2 and meta["files"] == 2
for i in range(2):
    r = requests.get(f"{BASE}/result/{sub['job_id']}?i={i}", headers=H, timeout=60)
    print(f"   GET /result?i={i} -> HTTP {r.status_code} {r.headers.get('content-type')} "
          f"{len(r.content)} B  X-Job-Index={r.headers.get('x-job-index')} "
          f"cache={r.headers.get('cache-control')}")
    ok_batch = ok_batch and r.status_code == 200 and len(r.content) > 1000
print("   batch =>", "PASS" if ok_batch else "FAIL")

print("\n== 2) 结果缓存 (同一单图请求跑两次) ==")
single = {"prompt": "a misty forest at dawn with light rays", "style": "写实摄影",
          "res": "512", "steps": 12, "seed": 98765, "sampler": "dpmpp2m_karras", "batch": 1}
first = post("/generate", single)
meta1, secs1 = wait(first["job_id"])
print(f"   1st: status={meta1['status']} {secs1:.1f}s cached={meta1.get('cached')}")
again = post("/generate", single)
print("   2nd submit ->", again)
hit_ok = False
if again.get("cached") and again.get("status") == "done":
    t0 = time.time()
    hit = requests.get(f"{BASE}/result/{again['job_id']}", headers=H, timeout=60)
    hit_ok = hit.status_code == 200 and len(hit.content) > 1000
    print(f"   cache hit -> HTTP {hit.status_code} {len(hit.content)} B in {time.time()-t0:.3f}s "
          f"cache={hit.headers.get('cache-control')}")
else:
    meta2, secs2 = wait(again["job_id"])
    print(f"   2nd: status={meta2['status']} {secs2:.1f}s cached={meta2.get('cached')}  (未命中缓存)")
ok_cache = bool(again.get("cached")) and hit_ok
print("   cache =>", "PASS" if ok_cache else "FAIL")

print("\n== 3) health.cache ==")
h = requests.get(f"{BASE}/health", headers=H, timeout=30).json()
print("   ", h.get("cache"))

print("\nRESULT:", "ALL PASS" if (ok_batch and ok_cache) else "SOME FAIL")
