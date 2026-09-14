"""enhance.py — 高清晰度流水线（ultimate upscale）：
  ① 神经超分放大到目标尺寸 → ② 分块(tile) img2img 低强度重绘补细节 → ③ 轻锐化。

分块重绘让每块都在 512（SD1.5 原生）内，6GB 显存可安全产出 2K/3K。

超分模型可选：
  * `ultrasharp` / `realesrgan`：社区权重（4x-UltraSharp、RealESRGAN_x4plus）
  * `ours` / `ours_final`：**本项目自训**的风景领域 x4 模型（Real-ESRGAN 高阶退化微调，4000 迭代）。
    客观证据（`research/sr_probe_final/`、docs/TRAINING.md §8）：相对 bicubic 细节量
    **6.53×（EMA）/ 8.69×（final）**，PSNR −1.45/−1.56 dB；RealESRGAN 7.09×、UltraSharp 8.03×
    —— 自训模型与两个商用权重同档，且在本地风景分布上训练。
"""
import math
import numpy as np
import torch
from PIL import Image, ImageFilter
from config import SR_DIR
SR_MODELS = {
    "ultrasharp": str(SR_DIR / "4x-UltraSharp.pth"),
    "realesrgan": str(SR_DIR / "RealESRGAN_x4plus.pth"),
    "ours": str(SR_DIR / "landscape_gan_x4_best.pth"),      # EMA 权重（验证集最优）
    "ours_final": str(SR_DIR / "landscape_gan_x4.pth"),     # 末轮权重（细节量更高、PSNR 略低）
}
_SR_CACHE = {}

def sr_model_names():
    """允许的模型名（供 API 校验与前端下拉使用）。"""
    return sorted(SR_MODELS)

def resolve_sr_path(name):
    """把 `sr_model` 解析成**受控目录内**的模型路径。

    这是外部输入校验：`sr_model` 直接来自 HTTP 请求体，原实现 `SR_MODELS.get(name, name)`
    等于允许客户端传任意路径。现在只接受已注册的名字，或位于 `config.SR_DIR` 内的相对文件名。
    """
    key = str(name or "").strip()
    if key.lower() in SR_MODELS:
        return SR_MODELS[key.lower()]
    candidate = (SR_DIR / key).resolve() if key else None
    if candidate is not None and candidate.is_file() and SR_DIR.resolve() in candidate.parents:
        return str(candidate)
    raise ValueError(f"unknown sr_model {name!r}; allowed: {', '.join(sr_model_names())}")

def get_sr(name="ultrasharp"):
    """按名称或受控路径加载超分模型（缓存）。"""
    path = resolve_sr_path(name)
    if path not in _SR_CACHE:
        from spandrel import ModelLoader
        _SR_CACHE[path] = ModelLoader().load_from_file(path)
    return _SR_CACHE[path]

def sr_to(img, target, sr_model, max_passes=1, device=None):
    """用神经超分做至多 max_passes 轮放大，其余用 LANCZOS 补齐（避免多轮超分极慢）。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = sr_model.eval().to(device)
    passes = 0
    while max(img.size) < target and passes < max_passes:
        x = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            o = m(x).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray((o * 255).astype(np.uint8))
        passes += 1
    if max(img.size) != target:                      # 补齐到目标尺寸
        img = img.resize((target, target), Image.LANCZOS)
    return img

def _feather(size, overlap):
    """余弦羽化窗（用于分块无缝融合）。"""
    w = np.ones((size,), np.float32)
    if overlap > 0:
        ramp = (1 - np.cos(np.linspace(0, math.pi, overlap))) / 2
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return np.outer(w, w)[..., None]

def tile_refine(img, i2i, prompt, neg, strength=0.30, steps=20, cfg=7.5,
                tile=512, overlap=96, seed=0):
    """把大图分块做 img2img 低强度重绘，羽化融合（每块 512，显存安全）。"""
    W, H = img.size
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    acc = np.zeros_like(arr); wsum = np.zeros((H, W, 1), np.float32)
    stride = tile - overlap
    xs = list(range(0, max(1, W - tile) + 1, stride)) or [0]
    ys = list(range(0, max(1, H - tile) + 1, stride)) or [0]
    if xs[-1] != W - tile: xs.append(max(0, W - tile))
    if ys[-1] != H - tile: ys.append(max(0, H - tile))
    fw = _feather(tile, overlap)
    n = 0
    for y in ys:
        for x in xs:
            crop = img.crop((x, y, min(x + tile, W), min(y + tile, H)))
            if crop.size != (tile, tile): crop = crop.resize((tile, tile), Image.LANCZOS)
            gen = i2i(prompt=prompt, negative_prompt=neg, image=crop, strength=strength,
                      num_inference_steps=steps, guidance_scale=cfg,
                      generator=torch.Generator(device="cuda").manual_seed(seed + n)).images[0]
            tw = np.asarray(gen.convert("RGB")).astype(np.float32)
            h = min(tile, H - y); w = min(tile, W - x)
            acc[y:y+h, x:x+w] += tw[:h, :w] * fw[:h, :w]
            wsum[y:y+h, x:x+w] += fw[:h, :w]
            n += 1
    out = acc / np.maximum(wsum, 1e-6)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))

def unsharp(img, radius=2.0, percent=70, threshold=3):
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

def enhance(img, target, sr_model, i2i, prompt, neg, strength=0.30, steps=20, cfg=7.5,
            tile=512, overlap=96, seed=0, sharpen=True):
    """完整高清晰度流水线：超分 -> 分块重绘 -> 锐化。sr_model 可为名称或已加载模型。"""
    if isinstance(sr_model, str): sr_model = get_sr(sr_model)
    up = sr_to(img, target, sr_model)
    if strength > 0:
        up = tile_refine(up, i2i, prompt, neg, strength=strength, steps=steps, cfg=cfg,
                         tile=tile, overlap=overlap, seed=seed)
    if sharpen:
        up = unsharp(up)
    return up
