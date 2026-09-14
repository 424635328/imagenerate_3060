"""train_sr_gan.py — 训练风景专用超分 GAN（细节化放大），6GB 可用。

目标：把「放大」从插值（bicubic 只是把像素摊平）变成**重建**——用 GAN 学出丢失的高频细节，
放大后更锐利、更少涂抹。这正是 Real-ESRGAN（Wang et al., arXiv:2107.10833）的核心思路：
**高阶退化建模** + 感知损失 + 对抗损失。本脚本在其之上做**领域微调**：用本项目 1511 张
风景图（1024^2）继续训练，使模型对风景纹理（草叶、水波、岩层、云层）更敏感。

关键设计
  * 生成器：RRDBNet（Real-ESRGAN x4plus 结构，16.7M 参数），默认从
    `models/sr/RealESRGAN_x4plus.pth` 初始化 —— 从通用模型出发做领域微调，远快于从零训练；
  * 判别器：U-Net + 谱归一化（UNetDiscriminatorSN），逐像素真伪判别；
  * 损失：L1（像素）×1.0 + VGG19 感知 ×1.0 + 对抗 ×0.1 + 判别器特征 L1 ×0.01（Real-ESRGAN 配方）；
  * 退化：blur → resize → noise（`--degradation real` 再叠一层，模拟真实低质来源）；
  * 训练用 fp32 权重 + autocast fp16（**不要**把可训练权重放 fp16：GradScaler 会拒绝反缩放 fp16 梯度）；
  * 断点续训 + 停滞看门狗 + 显存预检；导出 `params_ema`（spandrel/enhance.py 直接可读）。

用法:
    python train_sr_gan.py --iters 3000 --hr-size 256 --batch 4          # 领域微调（约 30-60 分钟）
    python train_sr_gan.py --iters 200 --hr-size 192 --batch 2 --smoke  # 快速冒烟
产物:
    models/sr/landscape_gan_x4.pth      可直接给 enhance.py 用（sr_model=<path>）
    models/sr/landscape_gan_x4_best.pth 验证集 PSNR 最优
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
DEFAULTS = dict(
    data_dir=str(ROOT / "dataset1024"),
    out_dir=str(ROOT / "models" / "sr"),
    init_generator=str(ROOT / "models" / "sr" / "RealESRGAN_x4plus.pth"),
    scale=4,
    hr_size=256,
    batch=4,
    iters=3000,
    lr_g=1e-4,
    lr_d=1e-4,
    pixel_weight=1.0,
    perceptual_weight=1.0,
    gan_weight=0.1,
    disc_feature_weight=0.01,
    ema_decay=0.999,
    degradation="real",
    val_every=250,
    val_images=24,
    save_every=500,
    keep_checkpoints=2,
    # Windows 下每个 worker 都会重新 import torch 并各占 300-500MB RAM；
    # 本机 RAM 已被 DSH/浏览器占用不少，默认 1 个 worker 足够（图像解码不是瓶颈）。
    num_workers=1,
    seed=42,
    resume="",
)
EVAL_SUBSET_SEED = 20240912


# --------------------------------------------------------------------- 生成器
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN x4plus 生成器（与其官方权重逐层对应）。"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ------------------------------------------------------------------- 判别器
class UNetDiscriminatorSN(nn.Module):
    """Real-ESRGAN 的 U-Net 判别器（与官方 discriminator_arch.py 逐层一致）。

    结构是 **3 次下采样 + 3 次上采样**，跳连的两端通道必须对齐：
        x3(8C) → up → conv4 → x4(4C) + x2(4C) → up → conv5 → x5(2C) + x1(2C)
        → up → conv6 → x6(C) + x0(C) → conv7/8 → conv9 → 1 通道 logits
    第一版我把下采样写成 8 层、跳连通道也没对齐，结果报
    "The size of tensor a (64) must match the size of tensor b (512)"。
    另外 `spectral_norm` 只能包卷积（它需要 weight 参数），套在 LeakyReLU 上会 KeyError。
    """

    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = nn.utils.spectral_norm
        self.conv0 = norm(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        x3 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        return self.conv9(out)


class VGGPerceptual(nn.Module):
    """VGG19 前 35 层的感知损失（ESRGAN 配方）。"""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.available = False
        try:
            from torchvision import models
            weights = models.VGG19_Weights.IMAGENET1K_V1
            vgg = models.vgg19(weights=weights).features[:35].eval()
            for param in vgg.parameters():
                param.requires_grad_(False)
            self.vgg = vgg.to(device).half()
            self.available = True
            # ImageNet 归一化
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1).half())
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1).half())
        except Exception as error:
            print(f"[perceptual] VGG19 unavailable ({type(error).__name__}: {str(error)[:80]}) — 退化为纯 L1+GAN")

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.vgg(x)


# --------------------------------------------------------------------- 退化
def gaussian_kernel(size: int, sigma: float, device, dtype):
    coords = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.to(dtype)


def degrade(hr: torch.Tensor, scale: int, mode: str = "real") -> torch.Tensor:
    """HR [B,3,H,W] (0-1) → LR [B,3,H/scale,W/scale]。

    「高阶退化」是 Real-ESRGAN 相对早期 SR 模型的关键：只训练 bicubic 下采样，模型学到的
    是插值的逆运算；加上模糊/缩放/噪声后，模型才被迫**重建**细节。
    """
    out = hr
    batch, channels, height, width = out.shape
    # 一阶：模糊
    if random.random() < 0.8:
        size = random.choice([7, 9, 11, 13])
        sigma = random.uniform(0.2, 2.4)
        kernel1d = gaussian_kernel(size, sigma, out.device, out.dtype)
        kernel2d = (kernel1d[:, None] * kernel1d[None, :])[None, None].repeat(channels, 1, 1, 1)
        out = F.conv2d(F.pad(out, (size // 2,) * 4, mode="reflect"), kernel2d, groups=channels)
    # 一阶：缩放（先下后上，制造信息损失）
    if random.random() < 0.8:
        factor = random.uniform(0.6, 1.0)
        small = F.interpolate(out, scale_factor=factor, mode=random.choice(["bilinear", "area"]),
                              align_corners=False if factor > 0 else None) if False else F.interpolate(
            out, scale_factor=factor, mode="bilinear", align_corners=False)
        out = F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
    # 二阶（仅 real）：再来一轮，模拟"多次压缩/翻拍"
    if mode == "real" and random.random() < 0.5:
        size = random.choice([7, 9])
        sigma = random.uniform(0.1, 1.2)
        kernel1d = gaussian_kernel(size, sigma, out.device, out.dtype)
        kernel2d = (kernel1d[:, None] * kernel1d[None, :])[None, None].repeat(channels, 1, 1, 1)
        out = F.conv2d(F.pad(out, (size // 2,) * 4, mode="reflect"), kernel2d, groups=channels)
        factor = random.uniform(0.7, 1.0)
        small = F.interpolate(out, scale_factor=factor, mode="bilinear", align_corners=False)
        out = F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)
    # 噪声
    if random.random() < 0.8:
        sigma = random.uniform(0.0, 0.05)
        out = (out + torch.randn_like(out) * sigma).clamp(0, 1)
    out = F.interpolate(out, size=(height // scale, width // scale), mode="area")
    return out.clamp(0, 1)


# ----------------------------------------------------------------------- 数据
class LandscapeHR(Dataset):
    def __init__(self, files: list[Path], hr_size: int):
        self.files = files
        self.hr_size = hr_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        from PIL import Image
        path = self.files[index % len(self.files)]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        size = self.hr_size
        if width < size or height < size:
            image = image.resize((max(size, width), max(size, height)), Image.BICUBIC)
            width, height = image.size
        x = random.randint(0, width - size)
        y = random.randint(0, height - size)
        crop = image.crop((x, y, x + size, y + size))
        if random.random() < 0.5:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        array = np.asarray(crop, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


# ------------------------------------------------------------------ 指标
def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.clamp(0, 1), b.clamp(0, 1)).item()
    return 99.0 if mse <= 1e-10 else 10 * math.log10(1.0 / mse)


def _gaussian_window(size: int = 11, sigma: float = 1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    return (g[:, None] * g[None, :])[None, None].to(dtype)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """标准 SSIM（11×11 高斯窗），避免额外依赖 scikit-image。

    channels 取 `shape[-3]`：既支持 [C,H,W] 也支持 [B,C,H,W]。
    第一版写成 `shape[1]`，对 [C,H,W] 会把高度 512 当通道数，报
    "expected input[1, 3, 512, 512] to have 512 channels"。
    """
    channels = a.shape[-3]
    window = _gaussian_window(device=a.device, dtype=a.dtype).repeat(channels, 1, 1, 1)
    a4 = a.unsqueeze(0) if a.dim() == 3 else a
    b4 = b.unsqueeze(0) if b.dim() == 3 else b
    mu_a = F.conv2d(a4, window, groups=channels)
    mu_b = F.conv2d(b4, window, groups=channels)
    sigma_a = F.conv2d(a4 * a4, window, groups=channels) - mu_a ** 2
    sigma_b = F.conv2d(b4 * b4, window, groups=channels) - mu_b ** 2
    sigma_ab = F.conv2d(a4 * b4, window, groups=channels) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / \
            ((mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a + sigma_b + c2))
    return float(score.mean().item())


# ---------------------------------------------------------------------- 训练
def load_generator_state(path: Path):
    if not path.exists():
        return None
    state = torch.load(str(path), map_location="cpu")
    for key in ("params_ema", "params", "state_dict"):
        if key in state:
            state = state[key]
            break
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    return cleaned


def main() -> int:
    ap = argparse.ArgumentParser()
    for key, value in DEFAULTS.items():
        # 同时接受 --hr_size 与 --hr-size：配置文件用下划线，命令行习惯用连字符
        ap.add_argument("--" + key, "--" + key.replace("_", "-"), dest=key,
                        type=type(value) if isinstance(value, (int, float, str)) else str, default=value)
    ap.add_argument("--smoke", action="store_true", help="极小规模冒烟（少量迭代 + 小尺寸）")
    args = ap.parse_args()
    cfg = vars(args)

    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    # 判别器只做 3 次下采样，HR 边长需 ≥ 8（即 2^3）才能保证最后一层卷积有输入
    if cfg["hr_size"] < 8:
        print(f"[error] --hr_size 至少 8（当前 {cfg['hr_size']}）：判别器有 3 次下采样")
        return 2
    device = "cuda"
    free_gb, total_gb = [v / 1024 ** 3 for v in torch.cuda.mem_get_info()]
    print(f"VRAM free {free_gb:.2f}/{total_gb:.2f} GB | iters={cfg['iters']} hr={cfg['hr_size']} "
          f"batch={cfg['batch']} scale={cfg['scale']} degradation={cfg['degradation']}")
    if free_gb < 2.0:
        print("[warn] 空闲显存 < 2GB：训练会变慢甚至 OOM，建议先关掉其它占用 GPU 的程序")

    data_dir = Path(cfg["data_dir"])
    train_files = sorted((data_dir / "train").glob("*.jpg")) + sorted((data_dir / "train").glob("*.png"))
    test_files = sorted((data_dir / "test").glob("*.jpg")) + sorted((data_dir / "test").glob("*.png"))
    if cfg["smoke"]:
        train_files = train_files[:16]
        test_files = test_files[:4]
        cfg["iters"] = min(cfg["iters"], 6)
        cfg["val_images"] = min(cfg["val_images"], 2)
    print(f"train images={len(train_files)} test images={len(test_files)}")

    generator = RRDBNet().to(device)
    state = load_generator_state(Path(cfg["init_generator"]))
    if state:
        missing, unexpected = generator.load_state_dict(state, strict=False)
        print(f"init generator from {Path(cfg['init_generator']).name}: "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    discriminator = UNetDiscriminatorSN().to(device)
    perceptual = VGGPerceptual(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg["lr_g"], betas=(0.9, 0.99))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg["lr_d"], betas=(0.9, 0.99))
    scaler_g = torch.amp.GradScaler("cuda", enabled=True)
    scaler_d = torch.amp.GradScaler("cuda", enabled=True)
    ema = {k: v.detach().clone().float() for k, v in generator.state_dict().items()}

    loader = DataLoader(LandscapeHR(train_files, cfg["hr_size"]), batch_size=cfg["batch"],
                        shuffle=True, num_workers=cfg["num_workers"], drop_last=True,
                        persistent_workers=cfg["num_workers"] > 0)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "landscape_gan_params.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    ck_dir = out_dir / "gan_checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "landscape_gan_metrics.csv"
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle, lineterminator="\n").writerow(
                ["iter", "loss_g", "loss_d", "pixel", "perceptual", "gan", "psnr_val", "ssim_val", "elapsed"])

    start_iter = 0
    if cfg["resume"]:
        resume_path = Path(cfg["resume"])
        if cfg["resume"] == "latest":
            candidates = sorted(ck_dir.glob("iter_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
            resume_path = candidates[-1] if candidates else Path("")
        if resume_path and resume_path.exists():
            ck = torch.load(str(resume_path), map_location="cpu")
            generator.load_state_dict(ck["generator"])
            discriminator.load_state_dict(ck["discriminator"])
            opt_g.load_state_dict(ck["opt_g"])
            opt_d.load_state_dict(ck["opt_d"])
            ema = ck["ema"]
            start_iter = ck.get("iter", 0)
            print(f"resumed at iter {start_iter}")

    @torch.no_grad()
    def validate() -> tuple[float, float]:
        generator.eval()
        order = torch.randperm(len(test_files), generator=torch.Generator().manual_seed(EVAL_SUBSET_SEED))[
            : cfg["val_images"]].tolist()
        psnrs, ssims = [], []
        for index in order:
            from PIL import Image
            image = Image.open(test_files[index]).convert("RGB")
            width, height = image.size
            side = min(512, width, height)
            x = (width - side) // 2
            y = (height - side) // 2
            hr = np.asarray(image.crop((x, y, x + side, y + side)), dtype=np.float32) / 255.0
            hr_t = torch.from_numpy(hr).permute(2, 0, 1)[None].to(device)
            # 验证统一用 bicubic 下采样（固定的、可比的退化）
            lr_t = F.interpolate(hr_t, scale_factor=1 / cfg["scale"], mode="bicubic",
                                 align_corners=False, antialias=True)
            with torch.autocast("cuda", dtype=torch.float16):
                sr_t = generator(lr_t)
            psnrs.append(psnr(sr_t.float()[0], hr_t[0]))
            ssims.append(ssim(sr_t.float()[0], hr_t[0]))
        generator.train()
        return (sum(psnrs) / len(psnrs), sum(ssims) / len(ssims))

    def export(best: bool = False):
        payload = {"params_ema": {k: v.half() for k, v in ema.items()}, "scale": cfg["scale"]}
        name = "landscape_gan_x4_best.pth" if best else "landscape_gan_x4.pth"
        torch.save(payload, out_dir / name)
        return name

    def save_checkpoint(iteration: int):
        payload = {"iter": iteration, "generator": generator.state_dict(),
                   "discriminator": discriminator.state_dict(), "opt_g": opt_g.state_dict(),
                   "opt_d": opt_d.state_dict(), "ema": ema, "cfg": cfg}
        tmp = ck_dir / f"iter_{iteration}.pt.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, ck_dir / f"iter_{iteration}.pt")
        for old in sorted(ck_dir.glob("iter_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[
                   : -cfg["keep_checkpoints"]]:
            old.unlink(missing_ok=True)

    started = time.time()
    last_beat = started
    best_psnr = -1.0
    iteration = start_iter
    print("training ... (L1 + perceptual + GAN)")
    generator.train()
    discriminator.train()
    while iteration < cfg["iters"]:
        for hr in loader:
            if iteration >= cfg["iters"]:
                break
            hr = hr.to(device, non_blocking=True)
            lr = degrade(hr, cfg["scale"], cfg["degradation"])
            try:
                # ---- 判别器 ----
                with torch.autocast("cuda", dtype=torch.float16):
                    sr_detached = generator(lr).detach()
                    real_logits = discriminator(hr)
                    fake_logits = discriminator(sr_detached)
                    loss_d = 0.5 * (F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean())
                opt_d.zero_grad(set_to_none=True)
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()

                # ---- 生成器 ----
                with torch.autocast("cuda", dtype=torch.float16):
                    sr = generator(lr)
                    pixel = F.l1_loss(sr, hr)
                    perceptual_loss = torch.tensor(0.0, device=device)
                    if perceptual.available:
                        perceptual_loss = F.l1_loss(perceptual(sr), perceptual(hr).detach())
                    fake_logits = discriminator(sr)
                    gan_loss = F.softplus(-fake_logits).mean()
                    loss_g = (cfg["pixel_weight"] * pixel
                              + cfg["perceptual_weight"] * perceptual_loss
                              + cfg["gan_weight"] * gan_loss)
                opt_g.zero_grad(set_to_none=True)
                scaler_g.scale(loss_g).backward()
                scaler_g.step(opt_g)
                scaler_g.update()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                opt_g.zero_grad(set_to_none=True)
                opt_d.zero_grad(set_to_none=True)
                print(f"[oom] skipped at iter {iteration} (try --hr-size {cfg['hr_size'] // 2})", flush=True)
                continue

            with torch.no_grad():
                decay = cfg["ema_decay"]
                for key, value in generator.state_dict().items():
                    ema[key].mul_(decay).add_(value.detach().float(), alpha=1 - decay)
            iteration += 1

            now = time.time()
            if now - last_beat > 240:
                free_gb, _ = [v / 1024 ** 3 for v in torch.cuda.mem_get_info()]
                print(f"[stall] 240s 无迭代完成（free {free_gb:.2f} GB）—— 另一个进程在抢 GPU", flush=True)
                last_beat = now
            if iteration % 20 == 0 or iteration == cfg["iters"]:
                elapsed = now - started
                speed = elapsed / max(1, iteration - start_iter)
                print(f"iter {iteration}/{cfg['iters']} "
                      f"G {loss_g.item():.4f} (pix {pixel.item():.4f} per {perceptual_loss.item():.4f} "
                      f"gan {gan_loss.item():.4f}) D {loss_d.item():.4f} "
                      f"{speed:.2f}s/it vram {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB", flush=True)
                last_beat = now
            if iteration % cfg["save_every"] == 0:
                save_checkpoint(iteration)
            if iteration % cfg["val_every"] == 0 or iteration == cfg["iters"]:
                val_psnr, val_ssim = validate()
                improved = val_psnr > best_psnr
                if improved:
                    best_psnr = val_psnr
                    export(best=True)
                with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
                    csv.writer(handle, lineterminator="\n").writerow(
                        [iteration, f"{loss_g.item():.5f}", f"{loss_d.item():.5f}", f"{pixel.item():.5f}",
                         f"{perceptual_loss.item():.5f}", f"{gan_loss.item():.5f}",
                         f"{val_psnr:.4f}", f"{val_ssim:.5f}", f"{time.time() - started:.1f}"])
                print(f"  [val] iter {iteration} PSNR {val_psnr:.3f} dB  SSIM {val_ssim:.4f}"
                      + ("  <- best" if improved else ""), flush=True)

    save_checkpoint(iteration)
    exported = export(best=False)
    print(f"done iter={iteration} best_psnr={best_psnr:.3f} elapsed={(time.time() - started) / 60:.1f}min")
    print(f"exported: {out_dir / exported}")
    print(f"用法: enhance.py 的 sr_model 指向该文件，或用 python tools/probe_sr_model.py 对比质量")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
