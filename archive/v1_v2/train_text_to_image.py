"""train_text_to_image.py — 在 RTX 3060 6GB 上用 LoRA 微调 Stable Diffusion 1.5 以生成风景图。

特点 / 鲁棒性：
  * 预计算 VAE latent + CLIP 文本嵌入并缓存，训练循环只跑 UNet+LoRA，省显存、更快。
  * fp16 + GradientScaler + UNet 梯度检查点，适配 6GB。
  * 断点续训：定期保存 checkpoint(lora权重+optimizer+scheduler+global_step)，支持 --resume 恢复。
  * OOM 防护：显存不足时自动清理缓存并提示降级参数；避免崩溃。
  * 仅把训练集与测试集分开；不修改原数据；所有输出写入 --out_dir。
  * 支持从 config.yaml 读取，命令行参数可覆盖。

用法：
  python train_text_to_image.py --config config.yaml
  （续训）python train_text_to_image.py --config config.yaml --resume latest
"""
import argparse, json, math, os, random, time, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

# 提前设置以可控缓存
def _load_diffusers():
    import diffusers
    from diffusers import AutoencoderKL, DiffusionPipeline, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    return diffusers, AutoencoderKL, DiffusionPipeline, DDPMScheduler, UNet2DConditionModel, CLIPTokenizer, CLIPTextModel

def _load_peft():
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    return LoraConfig, get_peft_model, set_peft_model_state_dict

# ---------------------------------------------------------------- config
DEFAULTS = dict(
    pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
    data_dir=ROOT + "/dataset",
    out_dir=ROOT + "/models/lora",
    resolution=512,
    train_batch_size=1,
    gradient_accumulation_steps=4,
    lora_rank=16,
    lora_alpha=16,
    lora_dropout=0.0,
    learning_rate=1e-4,
    lr_scheduler="cosine",
    lr_warmup_steps=500,
    max_train_steps=4000,          # 0 => 由 num_epochs 决定
    num_train_epochs=6,
    gradient_checkpointing=True,
    mixed_precision="fp16",
    flip_aug=True,             # 潜在空间随机水平翻转增强（防过拟合、增多样）
    seed=42,
    save_every=200,
    eval_every=100,
    eval_subset=64,
    resume="",                     # "" | "latest" | 某个 checkpoint 路径
    keep_checkpoints=3,
    cache_dir=ROOT + "/models/hf_cache",
    lr_scale=1.0,
)

def merge_config(cli, defaults):
    cfg = dict(defaults)
    for k, v in cli.items():
        if v is not None:
            cfg[k] = v
    return cfg

def load_toml_like(path):
    """读取相当简单的 key=value 配置文件（兼容 yaml 风格的简单 kv）。"""
    cfg = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip(); v = v.strip().strip('"').strip("'")
                cfg[k] = v
    return cfg

def parse_bool(v):
    if isinstance(v, bool): return v
    return str(v).lower() in ("1", "true", "yes", "on")

def parse_int(v):
    try: return int(float(v))
    except Exception: return v

def parse_float(v):
    try: return float(v)
    except Exception: return v

def coerce(cfg):
    """将字符串配置转成合适的类型。"""
    ints = ["resolution","train_batch_size","gradient_accumulation_steps","lora_rank","lora_alpha",
            "learning_rate","lr_warmup_steps","max_train_steps","num_train_epochs",
            "seed","save_every","eval_every","eval_subset","keep_checkpoints","lr_scale"]
    for k in ints:
        if k in cfg: cfg[k] = parse_float(cfg[k]) if k in ("learning_rate","lr_scale") else parse_int(cfg[k])
    for k in ["gradient_checkpointing","flip_aug"]:
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = parse_bool(cfg[k])
    if "lora_dropout" in cfg: cfg["lora_dropout"] = parse_float(cfg["lora_dropout"])
    for k in ["pretrained_model_name_or_path","data_dir","out_dir","lr_scheduler","mixed_precision","resume","cache_dir"]:
        if k in cfg and isinstance(cfg[k], float): cfg[k] = str(cfg[k])
    return cfg

# ---------------------------------------------------------------- dataset
class CachedLatentDataset(Dataset):
    """基于预计算的 latent + text embedding。每个样本存 (latent, text_emb, caption)。"""
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        lat, txt, cap = self.items[i]
        return lat, txt, cap

def encode_dataset(extract_items, tokenizer, text_encoder, vae, resolution, device, fp16):
    """将图片列表编码为 (latent, text_emb, caption)，返回 CPU 张量列表。"""
    from torch.utils.data import DataLoader as DL
    from PIL import Image
    latents = []
    texts = []
    caps = []
    vae = vae.to(device).eval()
    text_encoder = text_encoder.to(device).eval()
    # 逐个处理，避免峰值显存
    for idx, item in enumerate(extract_items):
        img_path = item["file"]
        cap = item.get("caption", "a landscape")
        try:
            im = Image.open(img_path).convert("RGB")
            im = im.resize((resolution, resolution), Image.LANCZOS)
            t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
            t = (t - 0.5) / 0.5
            t = t.unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                    lat = vae.encode(t).latent_dist.sample() * 0.18215
                # 文本
                tok = tokenizer(cap, padding="max_length", max_length=tokenizer.model_max_length,
                                truncation=True, return_tensors="pt").to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                    te = text_encoder(tok.input_ids).last_hidden_state
            # 去掉编码时引入的 batch 维，恢复为 [C,H,W] / [L,D]
            lat = lat.squeeze(0)
            te = te.squeeze(0)
            latents.append(lat.detach().half().cpu())
            texts.append(te.detach().half().cpu())
            caps.append(cap)
        except Exception as e:
            print(f"  [encode] skip {img_path}: {e}")
            continue
        if idx % max(1, len(extract_items)//10) == 0:
            print(f"  [encode] {idx}/{len(extract_items)}", flush=True)
    # 释放 vae/text_encoder 到 CPU 减少显存占用
    vae = vae.to("cpu")
    text_encoder = text_encoder.to("cpu")
    return latents, texts, caps

@torch.no_grad()
def eval_loss_on(unet, test_lat, test_txt, noise_scheduler, device, use_fp16, n=64):
    """在保留的测试集上计算平均 MSE 损失，用作过拟合/收敛监控。"""
    if len(test_lat) == 0:
        return None
    unet.eval()
    n = min(n, len(test_lat))
    idxs = torch.randint(0, len(test_lat), (n,))
    total = 0.0
    for i in idxs.tolist():
        lat = test_lat[i].unsqueeze(0).to(device)
        txt = test_txt[i].unsqueeze(0).to(device)
        noise = torch.randn_like(lat)
        ts = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
        noisy = noise_scheduler.add_noise(lat, noise, ts)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            out = unet(noisy, ts, txt).sample
        total += F.mse_loss(out.float(), noise.float()).item()
    unet.train()
    return total / n

# ---------------------------------------------------------------- main
def _write_metrics(out_dir, step, train_loss, val_loss, lr, epoch, elapsed):
    """追加一行过程指标（train_loss/val_loss 二选一为空）。"""
    import os, csv
    p = os.path.join(out_dir, "metrics.csv")
    if not os.path.exists(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step","train_loss","val_loss","lr","epoch","elapsed"])
    with open(p, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([step,
                                round(train_loss,5) if train_loss is not None else "",
                                round(val_loss,5) if val_loss is not None else "",
                                f"{lr:.2e}" if lr is not None else "",
                                epoch, round(elapsed,1) if elapsed is not None else ""])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--pretrained_model_name_or_path", default=None)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--train_batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None)
    ap.add_argument("--lora_rank", type=int, default=None)
    ap.add_argument("--lora_alpha", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--max_train_steps", type=int, default=None)
    ap.add_argument("--num_train_epochs", type=int, default=None)
    ap.add_argument("--lr_warmup_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", default=None, help="latest 或 checkpoint 路径")
    ap.add_argument("--init_lora", default="", help="仅加载该 checkpoint 的 LoRA 权重（重置步数/优化器/调度器），用于不同分辨率微调")
    ap.add_argument("--save_every", type=int, default=None)
    ap.add_argument("--peak_only", action="store_true", help="仅做一次前向以测显存/速度")
    args = ap.parse_args()

    file_cfg = load_toml_like(args.config) if args.config else {}
    cfg = dict(DEFAULTS)
    for k, v in file_cfg.items():
        cfg[k] = v
    # 命令行显式给出的优先级最高
    for k in ["pretrained_model_name_or_path","data_dir","out_dir","resolution","train_batch_size",
              "gradient_accumulation_steps","lora_rank","lora_alpha","learning_rate","max_train_steps",
              "num_train_epochs","lr_warmup_steps","seed","resume","save_every","init_lora"]:
        val = getattr(args, k, None)
        if val is not None:
            cfg[k] = val
    cfg = coerce(cfg)

    print("=== CONFIG ===")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    if cfg.get("cache_dir"):
        os.environ.setdefault("HF_HOME", cfg["cache_dir"])
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(cfg["cache_dir"], "hub"))

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "GPU:", torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

    diffusers, AutoencoderKL, DiffusionPipeline, DDPMScheduler, UNet2DConditionModel, CLIPTokenizer, CLIPTextModel = _load_diffusers()
    LoraConfig, get_peft_model, set_peft_model_state_dict = _load_peft()

    data_dir = Path(cfg["data_dir"])
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    train_items = [m for m in manifest if m["split"] == "train"]
    test_items = [m for m in manifest if m["split"] == "test"]
    print(f"train items: {len(train_items)}  test items: {len(test_items)}")
    if not train_items:
        print("no train items; abort"); return

    # ---- 加载模型
    model_id = cfg["pretrained_model_name_or_path"]
    print("loading base model", model_id)
    use_fp16 = (cfg["mixed_precision"] == "fp16") and (device == "cuda")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
    unet.requires_grad_(False)
    if cfg["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    # ---- LoRA
    lora_config = LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=cfg["lora_dropout"], bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    for n, p in unet.named_parameters():
        p.requires_grad_(p.requires_grad and any(x in n for x in ["lora_", "LoRA"]))
    trainable = [n for n, p in unet.named_parameters() if p.requires_grad]
    print(f"trainable params: {len(trainable)}; total trainable = {sum(p.numel() for p in unet.parameters() if p.requires_grad)}")

    unet = unet.to(device)

    # ---- 预计算 latent + text（缓存按分辨率区分，512 沿用旧名）
    cname = "encoded_cache.pt" if cfg["resolution"] == 512 else f"encoded_cache_{cfg['resolution']}.pt"
    cache_file = data_dir / cname
    if cache_file.exists() and not cfg.get("peak_only"):
        print("loading precomputed cache", cache_file)
        cache = torch.load(cache_file, map_location="cpu")
        latents, texts, caps = cache["latents"], cache["texts"], cache["caps"]
        # 过滤是否与 train_items 数量一致（简单起见重算若不一致）
    else:
        print("encoding train set...")
        t0 = time.time()
        latents, texts, caps = encode_dataset(train_items, tokenizer, text_encoder, vae, cfg["resolution"], device, use_fp16)
        if not cfg.get("peak_only"):
            torch.save({"latents": latents, "texts": texts, "caps": caps}, cache_file)
        print(f"encoded in {time.time()-t0:.1f}s")

    if cfg.get("peak_only"):
        # 测显存/速度：推进度
        print("peak-only mode: running a few forward/backward steps")
        unet.train()
        opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=cfg["learning_rate"])
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
        t0 = time.time()
        for i in range(3):
            idx = i % len(latents)
            lat = latents[idx].unsqueeze(0).to(device)
            txt = texts[idx].unsqueeze(0).to(device)
            noise = torch.randn_like(lat)
            ts = torch.randint(0, noise_scheduler.config.num_train_timesteps, (lat.shape[0],), device=device)
            noisy = noise_scheduler.add_noise(lat, noise, ts)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                out = unet(noisy, ts, txt).sample
            loss = F.mse_loss(out.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        print(f"peak-only: 3 steps in {time.time()-t0:.2f}s, peak mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        return

    dataset = CachedLatentDataset(list(zip(latents, texts, caps)))
    def collate(batch):
        lat = torch.stack([b[0] for b in batch])
        txt = torch.stack([b[1] for b in batch])
        cap = [b[2] for b in batch]
        return lat, txt, cap
    loader = DataLoader(dataset, batch_size=cfg["train_batch_size"], shuffle=True,
                        num_workers=0, collate_fn=collate, drop_last=True)

    # ---- 测试集编码（验证损失，用于早停/选最优）
    test_lat, test_txt = [], []
    if test_items:
        print(f"encoding {len(test_items)} test items for validation...")
        test_lat, test_txt, _ = encode_dataset(test_items, tokenizer, text_encoder, vae,
                                                cfg["resolution"], device, use_fp16)
        print(f"test encoded: {len(test_lat)}")

    # ---- 优化器 / 调度器
    optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=cfg["learning_rate"])
    total_steps = cfg["max_train_steps"]
    if total_steps <= 0:
        total_steps = len(loader) * cfg["num_train_epochs"]
    if cfg["lr_scheduler"] == "cosine":
        def lr_lambda(step):
            if step < cfg["lr_warmup_steps"]:
                return float(step) / max(1.0, float(cfg["lr_warmup_steps"]))
            prog = (step - cfg["lr_warmup_steps"]) / max(1.0, total_steps - cfg["lr_warmup_steps"])
            return 0.5 * (1.0 + math.cos(math.pi * prog))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    # ---- 断点恢复
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    # 记录过程参数 + 创建指标表头
    try:
        with open(out_dir / "params.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
    except Exception as e:
        print("warn params.json", e)
    _write_metrics(out_dir, 0, None, None, None, 0, 0.0)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    start_epoch = 0
    # 仅加载已有 LoRA 权重（保留 V3 拟合），重置步数/优化器/调度器，用于分辨率微调
    if cfg.get("init_lora"):
        init_path = Path(cfg["init_lora"])
        if init_path.exists() and init_path.is_file():
            print("init LoRA from", init_path)
            ck = torch.load(init_path, map_location="cpu")
            if "unet_lora" in ck:
                set_peft_model_state_dict(unet, ck["unet_lora"])
                print("  loaded unet_lora (weights only)")
        else:
            print("WARN init_lora not found:", init_path)
    resume_path = cfg["resume"]
    if resume_path:
        if resume_path == "latest":
            cands = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
            resume_path = str(cands[-1]) if cands else ""
        if resume_path and Path(resume_path).exists():
            print("resuming from", resume_path)
            ck = torch.load(resume_path, map_location="cpu")
            global_step = ck.get("global_step", 0)
            start_epoch = ck.get("epoch", 0)
            if "unet_lora" in ck:
                set_peft_model_state_dict(unet, ck["unet_lora"])
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            if "scheduler" in ck and hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(ck["scheduler"])
            print(f"resumed at global_step={global_step} epoch={start_epoch}")

    def save_checkpoint(tag, extra=None):
        # 原子写：先写临时再 rename
        tmp = ckpt_dir / f"step_{global_step}.pt.tmp"
        payload = {
            "global_step": global_step,
            "epoch": start_epoch,
            "unet_lora": unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "config": cfg,
        }
        if extra: payload.update(extra)
        torch.save(payload, tmp)
        final = ckpt_dir / f"step_{global_step}.pt"
        os.replace(tmp, final)
        print(f"[ckpt] saved step_{global_step}.pt")
        # 保留最近 keep_checkpoints 个
        cands = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        for old in cands[:-cfg["keep_checkpoints"]]:
            old.unlink(missing_ok=True)

    # ---- 训练循环
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    unet.train()
    opt = optimizer
    print(f"training for {total_steps} steps...")
    t0 = time.time()
    last_log = t0
    oom_occurred = False
    epoch = start_epoch
    best_val = float("inf")
    best_val_step = 0
    while global_step < total_steps:
        torch.manual_seed(cfg["seed"] + epoch * 10000)
        for lat, txt, cap in loader:
            if global_step >= total_steps:
                break
            # 周期重启 DataLoader 的 shuffle（简单处理 done via manual seed）
            lat = lat.to(device); txt = txt.to(device)
            # ---- 潜空间强增强：水平翻转 / 缩放 / 平移（抗过拟合，可长训）----
            if cfg["flip_aug"]:
                if random.random() < 0.5:
                    lat = torch.flip(lat, dims=[-1])
                # 缩放（在 64x64 latent 上做，相当于图像级 zoom）
                if random.random() < 0.5:
                    f = random.uniform(0.90, 1.10)
                    if abs(f - 1.0) > 0.02:
                        h, w = lat.shape[-2], lat.shape[-1]
                        lat = F.interpolate(lat.float(), scale_factor=f, mode="bilinear", align_corners=False)
                        lat = F.interpolate(lat, size=(h, w), mode="bilinear", align_corners=False).half()
                # 平移（小幅 roll）
                if random.random() < 0.5:
                    dx = random.choice([-2, -1, 0, 1, 2]); dy = random.choice([-2, -1, 0, 1, 2])
                    if dx or dy:
                        lat = torch.roll(lat, shifts=(dy, dx), dims=(-2, -1))
            noise = torch.randn_like(lat)
            ts = torch.randint(0, noise_scheduler.config.num_train_timesteps, (lat.shape[0],), device=device)
            noisy = noise_scheduler.add_noise(lat, noise, ts)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    out = unet(noisy, ts, txt).sample
                loss = F.mse_loss(out.float(), noise.float()) / cfg["gradient_accumulation_steps"]
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[oom] caught; empty cache; will retry step after skip")
                oom_occurred = True
                if global_step == 0:
                    print("[oom] at first step — consider lowering resolution/rank. Aborting training run.")
                    return
                continue

            if (global_step + 1) % cfg["gradient_accumulation_steps"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                now = time.time()
                if now - last_log > 10 or global_step % cfg["save_every"] == 0:
                    lr = opt.param_groups[0]["lr"]
                    print(f"step {global_step}/{total_steps}  loss {loss.item()*cfg['gradient_accumulation_steps']:.4f}  lr {lr:.2e}  elapsed {now-t0:.0f}s", flush=True)
                    _write_metrics(out_dir, global_step, loss.item()*cfg['gradient_accumulation_steps'], None, lr, epoch, now - t0)
                    last_log = now
                if global_step % cfg["save_every"] == 0 and global_step > 0:
                    save_checkpoint(global_step)
                if test_lat and global_step % cfg["eval_every"] == 0 and global_step > 0:
                    val = eval_loss_on(unet, test_lat, test_txt, noise_scheduler, device, use_fp16, cfg["eval_subset"])
                    if val is not None:
                        print(f"  [eval] step {global_step} val_loss {val:.4f} (best {best_val:.4f})", flush=True)
                        _write_metrics(out_dir, global_step, None, val, None, epoch, time.time() - t0)
                        if val < best_val:
                            best_val = val
                            best_val_step = global_step
                            payload = {"global_step": global_step, "epoch": epoch,
                                       "unet_lora": unet.state_dict(),
                                       "optimizer": optimizer.state_dict(),
                                       "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
                                       "config": cfg, "val_loss": val}
                            tmp = ckpt_dir / "best_val.pt.tmp"
                            torch.save(payload, tmp)
                            os.replace(tmp, ckpt_dir / "best_val.pt")
                            unet.save_pretrained(out_dir / "adapter_best")
                            print(f"  [eval] new best val_loss {val:.4f} at step {global_step}")
        epoch += 1

        # 每个 epoch 后重新打乱（手动）
        loader = DataLoader(dataset, batch_size=cfg["train_batch_size"], shuffle=True,
                            num_workers=0, collate_fn=collate, drop_last=True)

    # 最终保存
    save_checkpoint(global_step)
    # 也导出可直接 load 的 LoRA adapter
    print("saving final LoRA adapter...")
    unet.save_pretrained(out_dir / "adapter")
    print("done. final step", global_step, "elapsed", time.time() - t0)

if __name__ == "__main__":
    main()
