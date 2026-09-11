"""upscale_batch.py — 复用 SR 模型批量超分，输出压到 target_max 边长。"""
import argparse, os
import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

SR = ROOT + "/models/sr/RealESRGAN_x4plus.pth"

def load():
    m = ModelLoader().load_from_file(SR)
    return m.eval().to("cuda")

def up(img, m, target_max):
    x = np.asarray(img.convert("RGB")).astype(np.float32)/255.0
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to("cuda")
    with torch.no_grad():
        out = m(x)
    out = out.clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()
    r = Image.fromarray((out*255).astype(np.uint8))
    if target_max and max(r.size) > target_max:
        r.thumbnail((target_max, target_max), Image.LANCZOS)
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srcs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_max", type=int, default=2560)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    m = None
    for i, s in enumerate(args.srcs):
        im = Image.open(s)
        if args.dry:
            print(f"{i}: {s} -> {im.size}"); continue
        if m is None:
            m = load()
        r = up(im, m, args.target_max)
        out = os.path.join(args.out_dir, f"sr_{i}.png")
        r.save(out)
        print(f"{i}: {s} -> {out}  {r.size}", flush=True)

if __name__ == "__main__":
    main()
