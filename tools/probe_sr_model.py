"""probe_sr_model.py — 超分模型对比：证明"放大后确实保留了更多细节"。

对同一批测试图做 ×4 放大，横向比较：
  * **bicubic**（纯插值基线：像素变大但细节没增加）
  * **RealESRGAN_x4plus**（通用模型）
  * **本项目训练的风景 GAN**（领域微调）
  * 可选 4x-UltraSharp

指标分两类，缺一不可：
  * 参考指标 PSNR / SSIM —— 与 HR 的接近程度（**过高反而可能是过度平滑**）
  * 无参考细节指标 Laplacian 方差 / 梯度能量 —— "细节量"，插值放大在这项上不会提升
    真正的 GAN 超分应当在 PSNR 不塌的前提下把细节指标明显拉高。

模型通过 **spandrel** 加载 —— 与 enhance.py 同一条路径，因此本脚本同时验证了**可部署性**。

用法:
    python tools/probe_sr_model.py --images 8
    python tools/probe_sr_model.py --models "bicubic:" "RealESRGAN:models/sr/RealESRGAN_x4plus.pth" "Ours:models/sr/landscape_gan_x4_best.pth"
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))
from train_sr_gan import psnr, ssim  # noqa: E402  与训练期同一套实现

DEFAULT_MODELS = [
    ("bicubic", ""),
    ("RealESRGAN", str(ROOT / "models" / "sr" / "RealESRGAN_x4plus.pth")),
    ("UltraSharp", str(ROOT / "models" / "sr" / "4x-UltraSharp.pth")),
    ("Ours", str(ROOT / "models" / "sr" / "landscape_gan_x4_best.pth")),
]
SUBSET_SEED = 20240912


@torch.no_grad()
def detail_metrics(image: torch.Tensor) -> tuple[float, float]:
    """无参考细节量：Laplacian 方差（锐度）与 Sobel 梯度能量。0–255 量纲。"""
    gray = (image * 255.0).mean(dim=0, keepdim=True).unsqueeze(0)
    lap = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]], device=gray.device).view(1, 1, 3, 3)
    laplacian = F.conv2d(gray, lap, padding=1)
    sobel_x = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=gray.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    return float(laplacian.var().item()), float((gx ** 2 + gy ** 2).mean().sqrt().item())


def load_model(path: str):
    if not path:
        return None
    from spandrel import ModelLoader
    model = ModelLoader().load_from_file(path)
    return model.model.eval().to("cuda")


@torch.no_grad()
def upscale(model, lr: torch.Tensor) -> torch.Tensor:
    if model is None:      # bicubic 基线
        return F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False).clamp(0, 1)
    return model(lr).clamp(0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[f"{name}:{path}" for name, path in DEFAULT_MODELS],
                    help='形如 "Ours:models/sr/landscape_gan_x4_best.pth"；空路径表示 bicubic')
    ap.add_argument("--images", type=int, default=8, help="测试图数量")
    ap.add_argument("--crop", type=int, default=512, help="HR 中心裁剪边长（LR 为其 1/4）")
    ap.add_argument("--out", default=str(ROOT / "research" / "sr_probe"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_files = sorted((ROOT / "dataset1024" / "test").glob("*.jpg"))
    order = torch.randperm(len(test_files), generator=torch.Generator().manual_seed(SUBSET_SEED))[
        : args.images].tolist()
    print(f"test images: {len(test_files)}  使用 {len(order)} 张，HR 裁剪 {args.crop}^2 → LR {args.crop // 4}^2\n")

    specs = []
    for spec in args.models:
        label, _, path = spec.partition(":")
        if path and not Path(path).exists():
            print(f"  [skip] {label}: 未找到 {path}")
            continue
        specs.append((label, path))

    # HR 参考 + LR 输入
    samples = []
    for index in order:
        image = Image.open(test_files[index]).convert("RGB")
        width, height = image.size
        side = min(args.crop, width, height)
        x, y = (width - side) // 2, (height - side) // 2
        hr = np.asarray(image.crop((x, y, x + side, y + side)), dtype=np.float32) / 255.0
        hr_t = torch.from_numpy(hr).permute(2, 0, 1)
        lr_t = F.interpolate(hr_t[None], scale_factor=1 / 4, mode="bicubic",
                             align_corners=False, antialias=True)[0]
        samples.append((hr_t, lr_t))

    rows = []
    tiles: dict[str, list[Image.Image]] = {}
    for label, path in specs:
        model = load_model(path)
        entries = []
        for hr_t, lr_t in samples:
            sr = upscale(model, lr_t[None].to("cuda"))[0].float().cpu()
            sharp, edges = detail_metrics(sr)
            entries.append({
                "psnr": psnr(sr, hr_t), "ssim": ssim(sr, hr_t),
                "sharpness": sharp, "edges": edges,
            })
        del model
        gc.collect()
        torch.cuda.empty_cache()
        mean = {key: float(np.mean([e[key] for e in entries])) for key in entries[0]}
        rows.append((label, mean))
        tiles[label] = [Image.fromarray((s[1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for s in samples]
        print(f"  {label:12s} PSNR {mean['psnr']:6.2f} dB  SSIM {mean['ssim']:.4f}  "
              f"Laplacian {mean['sharpness']:8.2f}  梯度 {mean['edges']:6.2f}")

    # 参考列（HR）
    tiles = {"HR(参考)": [Image.fromarray((s[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for s in samples],
             **tiles}
    columns = list(tiles.keys())
    tile = 192
    sheet = Image.new("RGB", (tile * len(columns), tile * len(samples)), "#0b1020")
    for column, label in enumerate(columns):
        for row_index, image in enumerate(tiles[label]):
            sheet.paste(image.resize((tile, tile), Image.LANCZOS), (column * tile, row_index * tile))
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(columns):
        draw.rectangle([column * tile, 0, column * tile + 96, 16], fill="#000000cc")
        draw.text((column * tile + 3, 3), label, fill="#ffffff")
    sheet_path = out_dir / "sr_compare_sheet.png"
    sheet.save(sheet_path)

    print("\n=== 结论 ===")
    base = next((m for l, m in rows if l == "bicubic"), None)
    for label, mean in rows:
        if base and label != "bicubic":
            print(f"  {label:12s} 相对 bicubic：PSNR {mean['psnr'] - base['psnr']:+.2f} dB  "
                  f"细节量 {mean['sharpness'] / max(1e-6, base['sharpness']):.2f}×")
    print(f"\n对比拼版：{sheet_path}")
    print("判读要点：细节量明显 >1.0× 且 PSNR 不塌（不低于 bicubic 太多）才是真正的细节增强；")
    print("          PSNR 很高但细节量与 bicubic 接近，说明只是更平滑的插值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
