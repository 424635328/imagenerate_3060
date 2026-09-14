"""真机验证：切换模型版本**确实改变了像素**（不是静默 no-op）。

本项目被"adapter 看起来加载了、其实没生效"咬过两次（`load_lora_weights` 不加载 peft 权重；
PEFT `add_weighted_adapter` 产出全零 lora_B），两次的键集合与形状都完全正常。
所以这条验证不看"接口返回 200"，只看**出图是否真的不同**：
  同一 prompt、同一 seed、同一步数/CFG/采样器，分别用 v4 与 v5b 出图 →
  两张图必须不同（sha256 + 平均像素差），且各自与"两者之间的差异"可比。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageChops  # noqa: E402

BASE = "http://127.0.0.1:8001"
TOKEN = os.environ.get("API_TOKEN", "")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # 绕开本机代理
PROMPT = "a snow-capped mountain range reflected in a still alpine lake at dawn, ultra detailed"
SEED = 424242


def call(path: str, payload: dict | None = None, timeout: int = 240) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(f"{BASE}{path}", data=data,
                                     headers={"Content-Type": "application/json",
                                              "X-API-Key": TOKEN})
    with OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch(path: str) -> bytes:
    request = urllib.request.Request(f"{BASE}{path}", headers={"X-API-Key": TOKEN})
    with OPENER.open(request, timeout=120) as response:
        return response.read()


def generate(adapter: str) -> bytes:
    job = call("/generate", {
        "prompt": PROMPT, "steps": 20, "cfg": 7.5, "seed": SEED, "res": "512",
        "sampler": "dpmpp2m_karras", "batch": 1, "adapter": adapter,
    })
    job_id = job["job_id"]
    for _ in range(120):
        time.sleep(2)
        status = call(f"/jobs/{job_id}")
        if status.get("status") in ("done", "failed", "error"):
            break
    if status.get("status") != "done":
        raise SystemExit(f"[失败] adapter={adapter} 任务未完成：{status.get('status')}")
    return fetch(f"/result/{job_id}")


def main() -> int:
    health = call("/health")
    print(f"后端就绪：pipeline.loaded={health['pipeline']['loaded']} "
          f"now={health['pipeline'].get('adapter')}")

    images = {}
    for slug in ("v4", "v5b"):
        started = time.time()
        raw = generate(slug)
        images[slug] = raw
        size = Image.open(io.BytesIO(raw)).size
        print(f"  {slug:5s} → {len(raw)} B, {size}, 耗时 {time.time() - started:.1f}s, "
              f"sha256 {hashlib.sha256(raw).hexdigest()[:16]}")

    same_bytes = images["v4"] == images["v5b"]
    a = Image.open(io.BytesIO(images["v4"])).convert("RGB")
    b = Image.open(io.BytesIO(images["v5b"])).convert("RGB")
    diff = ImageChops.difference(a, b)
    mean = sum(sum(pixel) for pixel in diff.getdata()) / (a.width * a.height * 3)

    print(f"\n两版差异：逐字节相同={same_bytes}，平均像素差={mean:.2f}/255")
    ok = (not same_bytes) and mean > 1.0
    print("结论：" + ("✅ 切换版本确实改变了像素（不是 no-op）" if ok
                     else "❌ 两版出图几乎一致 —— 版本切换没有真正生效"))
    # 顺带确认后端现在驻留的是最后请求的版本
    after = call("/health")["pipeline"].get("adapter")
    print(f"后端当前驻留：{after}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
