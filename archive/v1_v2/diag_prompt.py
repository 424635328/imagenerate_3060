import os
"""diag_prompt.py — 诊断：为什么某些 prompt 全出沙漠（同 seed 对照）。"""
import requests, time, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))
OUT = Path(ROOT + "/outputs/diag"); OUT.mkdir(parents=True, exist_ok=True)
H = {"X-API-Key": os.environ.get("API_TOKEN", "change-me"), "Content-Type": "application/json"}
CASES = [
    ("A 原始", "an autumn forest with golden and red maple leaves, misty path, soft light", ""),
    ("B 去掉path/light", "an autumn forest with golden and red maple leaves", ""),
    ("C 换词", "a dense forest with red and orange maple trees, autumn", ""),
    ("D 朴素描述", "a misty forest path in autumn with colorful leaves", ""),
    ("E 绿色森林对照", "a green forest with fog, tall trees", ""),
    ("F 沙漠对照", "golden sand dunes at sunset", ""),
    ("G 原始+反沙漠", "an autumn forest with golden and red maple leaves, misty path, soft light",
     "desert, sand, dunes, arid, barren, canyon, rocky"),
]
def gen(prompt, neg, seed=777):
    body = {"prompt": prompt, "style": "写实摄影", "res": "512", "steps": 22, "seed": seed,
            "neg": neg or "blurry, low quality, watermark, text, distorted, oversaturated, bad anatomy",
            "enhance": 0}
    jid = requests.post("http://127.0.0.1:8001/generate", json=body, headers=H, timeout=30).json()["job_id"]
    for _ in range(60):
        time.sleep(4)
        j = requests.get(f"http://127.0.0.1:8001/jobs/{jid}", timeout=20).json()
        if j["status"] in ("done", "failed"):
            if j["status"] == "failed": raise RuntimeError(j["error"])
            break
    return requests.get(f"http://127.0.0.1:8001/result/{jid}", timeout=60).content
imgs = []
for name, p, neg in CASES:
    t = time.time()
    data = gen(p, neg)
    fp = OUT / f"diag_{name.split()[0]}.jpg"; fp.write_bytes(data)
    imgs.append((name, p, Image.open(fp).convert("RGB")))
    print(f"{name}: {time.time()-t:.0f}s -> {fp.name}", flush=True)

cell, pad, top = 300, 10, 78
cols = 4; rows = (len(imgs)+cols-1)//cols
W = pad + cols*(cell+pad); Hh = top + rows*(cell+34) + pad
canvas = Image.new("RGB", (W, Hh), (12, 15, 24)); d = ImageDraw.Draw(canvas)
def font(sz, b=False):
    for p in ([r"C:\Windows\Fonts\msyhbd.ttc"] if b else [r"C:\Windows\Fonts\msyh.ttc"]):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()
d.text((pad, 12), "prompt 诊断（同 seed=777）  A原始 / B去path / C换词 / D朴素 / E绿森对照 / F沙漠对照 / G加反沙漠", font=font(18, True), fill=(232,237,245))
for i, (name, p, im) in enumerate(imgs):
    r, c = divmod(i, cols); x = pad + c*(cell+pad); y = top + r*(cell+34)
    canvas.paste(im.resize((cell, cell), Image.LANCZOS), (x, y))
    d.text((x, y-24), name, font=font(15, True), fill=(255,220,150))
    d.text((x, y+cell+4), p[:44], font=font(11), fill=(160,175,195))
canvas.save(OUT/"diag_montage.png")
print("montage:", OUT/"diag_montage.png", canvas.size)
