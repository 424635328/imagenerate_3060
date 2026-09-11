"""upscale.py — 用 Real-ESRGAN（经 spandrel 加载）对风景图做超分：512/768 -> 2K/3K。

用法:
  python upscale.py IN.png OUT.png [--scale 4]
"""
import argparse, os
import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

def get_upscaler(model_path):
    model = ModelLoader().load_from_file(model_path)
    if hasattr(model, "scale"):
        _ = model.scale
    return model

def upscale(img, model, scale=4):
    model = model.eval().to("cuda")
    x = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to("cuda")
    with torch.no_grad():
        out = model(x)
    out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = (out * 255).astype(np.uint8)
    return Image.fromarray(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()
    model_path = ROOT + "/models/sr/RealESRGAN_x4plus.pth"
    m = get_upscaler(model_path)
    res = upscale(Image.open(args.inp), m, args.scale)
    res.save(args.out)
    print(f"saved {args.out}  size={res.size}")

if __name__ == "__main__":
    main()
