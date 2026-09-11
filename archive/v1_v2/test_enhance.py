import os, sys, time
os.environ.setdefault("LORA_ADAPTER", ROOT + "/models/v4_640/adapter_best")
sys.path.insert(0, ROOT)
import torch
from pathlib import Path
import app, enhance
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

OUT = Path(ROOT + "/outputs"); OUT.mkdir(parents=True, exist_ok=True)
P = "a dramatic mountain valley at sunrise, mist, alpine lake, golden light"
NEG = app.DEFAULT_NEG
print("loading model...", flush=True)
pipe = app.get_pipe(); i2i = app.get_i2i(); sr = app.get_sr()
print("generating 512...", flush=True)
t = time.time()
img = pipe(prompt=P, negative_prompt=NEG, num_inference_steps=40, guidance_scale=7.5,
           width=512, height=512, generator=torch.Generator(device="cuda").manual_seed(7)).images[0]
img.save(OUT / "cmp_base_512.png"); print(f"base 512 {time.time()-t:.1f}s", flush=True)

t = time.time()
sr_only = enhance.sr_to(img, 2048, sr)
sr_only.save(OUT / "cmp_esrgan_2048.png"); print(f"esrgan 2048 {time.time()-t:.1f}s", flush=True)

t = time.time()
big = enhance.enhance(img, 2048, sr, i2i, P, NEG, strength=0.30, steps=20, cfg=7.5,
                      tile=512, overlap=96, seed=7, sharpen=True)
big.save(OUT / "cmp_ultimate_2048.png"); print(f"ultimate 2048 {big.size} {time.time()-t:.1f}s", flush=True)
print("done")
