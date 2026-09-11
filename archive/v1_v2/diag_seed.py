import os
"""diag_seed.py — 同 prompt 多随机 seed，复现用户看到的"全是沙漠"。"""
import random, time
import requests
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))
OUT = Path(ROOT + "/outputs/diag"); OUT.mkdir(parents=True, exist_ok=True)
H = {"X-API-Key": os.environ.get("API_TOKEN", "change-me"), "Content-Type": "application/json"}
PROMPT = "an autumn forest with golden and red maple leaves, misty path, soft light"
def gen(prompt, seed, steps=40, cfg=7.5):
    body = {"prompt": prompt, "style": "写实摄影", "res": "512", "steps": steps, "seed": seed,
            "enhance": 0, "highres": 0}
    jid = requests.post("http://127.0.0.1:8001/generate", json=body, headers=H, timeout=30).json()["job_id"]
    for _ in range(90):
        time.sleep(4)
        j = requests.get(f"http://127.0.0.1:8001/jobs/{jid}", timeout=20).json()
        if j["status"] in ("done", "failed"):
            if j["status"] == "failed": raise RuntimeError(j["error"])
            break
    return requests.get(f"http://127.0.0.1:8001/result/{jid}", timeout=60).content
seeds = [random.randint(0, 2**31-1) for _ in range(4)]
imgs = []
for sd in seeds:
    t = time.time(); data = gen(PROMPT, sd)
    fp = OUT / f"seed_{sd}.jpg"; fp.write_bytes(data)
    imgs.append((str(sd), Image.open(fp).convert("RGB")))
    print(f"seed {sd}: {time.time()-t:.0f}s", flush=True)
cell, pad, top = 320, 10, 70
W = pad + 4*(cell+pad); Hh = top + cell + 30
canvas = Image.new("RGB", (W, Hh), (12, 15, 24)); d = ImageDraw.Draw(canvas)
def font(sz, b=False):
    try: return ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc" if b else r"C:\Windows\Fonts\msyh.ttc", sz)
    except Exception: return ImageFont.load_default()
d.text((pad, 10), "同 prompt（秋日枫林）+ 4 个随机 seed，看是否出现沙漠", font=font(18, True), fill=(232,237,245))
d.text((pad, 36), PROMPT, font=font(12), fill=(150,170,195))
for i, (sd, im) in enumerate(imgs):
    x = pad + i*(cell+pad)
    canvas.paste(im.resize((cell, cell), Image.LANCZOS), (x, top))
    d.text((x+4, top+cell+4), "seed="+sd, font=font(13), fill=(255,220,150))
canvas.save(OUT/"diag_seeds.png"); print("montage:", OUT/"diag_seeds.png")
