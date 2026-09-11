import os
"""bench_steps.py — 速度/质量对比：质量模式(40/24 步) vs 极速模式(LCM 6/8 步)。同 prompt 同 seed。"""
import time
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

OUT = Path(ROOT + "/outputs/bench"); OUT.mkdir(parents=True, exist_ok=True)
H = {"X-API-Key": os.environ.get("API_TOKEN", "change-me"), "Content-Type": "application/json"}
PROMPT = "a dramatic mountain valley at sunrise, mist, alpine lake, golden light"
SEED = 4242

CASES = [
    ("quality 40步 DPM++", dict(fast=False, steps=40, cfg=7.5)),
    ("quality 24步 DPM++", dict(fast=False, steps=24, cfg=7.5)),
    ("fast 8步 LCM",      dict(fast=True,  steps=8,  cfg=2.0)),
    ("fast 6步 LCM",      dict(fast=True,  steps=6,  cfg=1.5)),
    ("fast 4步 LCM",      dict(fast=True,  steps=4,  cfg=1.5)),
]

def run(case):
    body = {"prompt": PROMPT, "style": "写实摄影", "res": "512", "seed": SEED, "enhance": 0, **case[1]}
    t0 = time.time()
    jid = requests.post("http://127.0.0.1:8001/generate", json=body, headers=H, timeout=30).json()["job_id"]
    while True:
        time.sleep(1)
        j = requests.get(f"http://127.0.0.1:8001/jobs/{jid}", timeout=20).json()
        if j["status"] in ("done", "failed"): break
    el = time.time() - t0
    if j["status"] != "done": raise RuntimeError(j["error"])
    img = requests.get(f"http://127.0.0.1:8001/result/{jid}", timeout=60).content
    fp = OUT / f"bench_{case[0].replace(' ','_')}.jpg"; fp.write_bytes(img)
    print(f"{case[0]:22s} {el:6.1f}s   {len(img)/1024:6.1f} KB", flush=True)
    return case[0], el, Image.open(fp).convert("RGB")

rows = [run(c) for c in CASES]
cell, pad, top = 320, 10, 78
W = pad + len(rows)*(cell+pad); Hh = top + cell + 34
canvas = Image.new("RGB", (W, Hh), (12, 15, 24)); d = ImageDraw.Draw(canvas)
def font(sz, b=False):
    try: return ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc" if b else r"C:\Windows\Fonts\msyh.ttc", sz)
    except Exception: return ImageFont.load_default()
d.text((pad, 12), "速度 × 质量对比（同 prompt 同 seed=4242，512×512）", font=font(19, True), fill=(232, 237, 245))
d.text((pad, 42), "500 步 ≈ 200s（你实测 ~4min）—— 步数远超收敛点，既慢又易过饱和", font=font(13), fill=(255, 190, 120))
for i, (name, el, im) in enumerate(rows):
    x = pad + i*(cell+pad)
    canvas.paste(im.resize((cell, cell), Image.LANCZOS), (x, top))
    d.text((x+4, top+cell+4), f"{name}  {el:.1f}s", font=font(14, True), fill=(255, 220, 150))
canvas.save(OUT/"bench_compare.png")
print("montage:", OUT/"bench_compare.png")
