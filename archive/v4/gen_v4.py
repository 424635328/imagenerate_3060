"""gen_v4.py — 用 V4 权重 + 微调 CLIP 生成样张（可配分辨率 -> 两段式 img2img 放大到 1024）。

环境变量: LORA_ADAPTER(默认 v4_lora/adapter_best) GEN_RES(默认512) OUT_DIR PREFIX
"""
import os, sys
os.environ.setdefault("LORA_ADAPTER", ROOT + "/models/v4_lora/adapter_best")
sys.path.insert(0, ROOT)
import torch, time
from pathlib import Path
import app
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

OUT = Path(os.environ.get("OUT_DIR", ROOT + "/outputs/gallery_v4")); OUT.mkdir(parents=True, exist_ok=True)
PREFIX = os.environ.get("PREFIX", "v4")
RES = int(os.environ.get("GEN_RES", "512"))
PROMPTS = [
    ("mountain", "a dramatic mountain valley at sunrise, mist, alpine lake, golden light"),
    ("rainforest", "a lush green rainforest waterfall, mossy rocks, sun rays through the canopy"),
    ("desert", "an expansive golden desert with rippled dunes under a vivid blue sky"),
    ("coast", "a serene turquoise tropical coastline, palm trees, calm sea, golden hour"),
    ("waterfall", "a majestic waterfall in a mossy gorge, flowing water, teal pool"),
    ("lake", "a tranquil alpine lake reflecting snow-capped peaks, morning fog"),
]
NEG = app.DEFAULT_NEG
print("adapter:", app.ADAPTER, flush=True)
pipe = app.get_pipe()
i2i = app.get_i2i()
print("pipe ready", flush=True)
for i, (name, p) in enumerate(PROMPTS):
    t = time.time()
    g = torch.Generator(device="cuda").manual_seed(100 + i)
    img = pipe(prompt=p, negative_prompt=NEG, num_inference_steps=40, guidance_scale=7.5,
               width=RES, height=RES, generator=g).images[0]
    img.save(OUT / f"{PREFIX}_{name}_{RES}.png")
    hr = i2i(prompt=p, negative_prompt=NEG, image=img.resize((1024, 1024)), strength=0.45,
             num_inference_steps=24, guidance_scale=7.5,
             generator=torch.Generator(device="cuda").manual_seed(100 + i)).images[0]
    hr.save(OUT / f"{PREFIX}_{name}_1024.png")
    print(f"[{i+1}/{len(PROMPTS)}] {name} {time.time()-t:.1f}s -> {OUT}/{PREFIX}_{name}_1024.png", flush=True)
print("done")
