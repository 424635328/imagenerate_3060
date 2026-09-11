"""verify_live_v5.py — 经线上代理验证 v5：健康、批量取图(i=0/1)、缓存秒回。"""
import time
import requests

P = "https://landscape-art-demo.netlify.app/.netlify/functions/proxy"


def get(u, **k):
    for i in range(5):
        try:
            return requests.get(u, timeout=60, **k)
        except Exception:
            if i == 4:
                raise
            time.sleep(3)


def post(u, body):
    for i in range(5):
        try:
            return requests.post(u, json=body, timeout=60)
        except Exception:
            if i == 4:
                raise
            time.sleep(3)


r = get(P + "?op=health")
print("proxy health:", r.status_code, r.text[:140])

print("\n== 批量 batch=2 ==")
base = {"prompt": "a serene turquoise tropical coastline, palm trees", "style": "写实摄影",
        "res": "512", "steps": 12, "seed": 24680, "sampler": "dpmpp2m_karras"}
sub = post(P + "?op=generate", base | {"batch": 2, "neg": ""}).json()
print("submit ->", sub)
jid = sub["job_id"]
for _ in range(60):
    time.sleep(3)
    m = get(P + f"?op=job&id={jid}").json()
    if m["status"] in ("done", "failed"):
        break
print(f"status={m['status']} count={m['count']} seeds={m['seeds']} bytes_each={m['bytes_each']}")
for i in range(m.get("count", 1)):
    ir = get(P + f"?op=image&id={jid}&i={i}")
    print(f"  op=image&i={i} -> HTTP {ir.status_code} {ir.headers.get('content-type')} "
          f"{len(ir.content)} B  idx={ir.headers.get('x-job-index')}/{ir.headers.get('x-job-count')}")

print("\n== 缓存秒回（同一单图请求两次）==")
one = base | {"batch": 1, "seed": 13579, "neg": ""}
first = post(P + "?op=generate", one).json()
while True:
    time.sleep(3)
    m1 = get(P + f"?op=job&id={first['job_id']}").json()
    if m1["status"] in ("done", "failed"):
        break
print(f"1st: {m1['status']} {m1.get('seconds')}s cached={m1.get('cached')}")
second = post(P + "?op=generate", one).json()
print("2nd submit ->", second)
if second.get("cached"):
    t0 = time.time()
    ir = get(P + f"?op=image&id={second['job_id']}")
    print(f"  cache hit: HTTP {ir.status_code} {len(ir.content)} B in {time.time()-t0:.2f}s")
    print("\nRESULT: ALL PASS")
else:
    print("\nRESULT: cache MISS")
