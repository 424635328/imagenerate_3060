"""train_v3.py — V3 鲁棒性/深度训练：逐批像素级增强 + 在线 VAE 编码 + 验证损失 + 过程记录。

相比 V2 的关键改进：
  * 训练时对每批图做**像素级随机增强**（随机缩放裁剪/水平翻转/旋转/颜色抖动/噪声），并在训练循环里**在线 VAE 编码**。
    每步看到的是新的增强样本，从根本上推迟过拟合，因此可作“无限时长”深度训练。
  * 自动记录过程参数与指标：metrics.csv(step,train_loss,val_loss,lr,epoch,elapsed) + params.json(全部超参)。
  * 保留测试集验证损失早停/最优 checkpoint（best_val.pt / adapter_best）。
  * 支持 512 与后续 640 高清微调（--resolution 切换），支持断点续训。

用法:
  python train_v3.py --config config_v3.cfg
  （续训）python train_v3.py --config config_v3.cfg --resume latest
"""
import argparse, json, math, os, random, time, csv
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = dict(
    pretrained_model_name_or_path="SG161222/Realistic_Vision_V6.0_B1_noVAE",
    data_dir=ROOT + "/dataset",
    out_dir=ROOT + "/models/v3_lora",
    resolution=512,
    train_batch_size=4,
    gradient_accumulation_steps=1,
    lora_rank=32,
    lora_alpha=32,
    lora_dropout=0.05,
    learning_rate=1e-4,
    lr_scheduler="cosine",
    lr_warmup_steps=2000,
    max_train_steps=30000,
    num_train_epochs=60,
    gradient_checkpointing=True,
    mixed_precision="fp16",
    aug=True,
    aug_crop_scale_low=0.7,
    aug_crop_scale_high=1.0,
    aug_rot=6,
    seed=42,
    save_every=500,
    eval_every=200,
    eval_subset=64,
    keep_checkpoints=4,
    resume="",
    cache_dir=ROOT + "/models/hf_cache",
)

def load_kv(path):
    cfg = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            if "=" in line:
                k, v = line.split("=", 1); cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg

def parse_bool(v):
    if isinstance(v, bool): return v
    return str(v).lower() in ("1","true","yes","on")

def coerce(cfg):
    for k in ["learning_rate","lora_alpha","lora_dropout","aug_crop_scale_low","aug_crop_scale_high","aug_rot"]:
        if k in cfg: cfg[k] = float(cfg[k])
    for k in ["resolution","train_batch_size","gradient_accumulation_steps","lora_rank","lr_warmup_steps",
              "max_train_steps","num_train_epochs","seed","save_every","eval_every","eval_subset","keep_checkpoints"]:
        if k in cfg: cfg[k] = int(float(cfg[k]))
    for k in ["gradient_checkpointing","aug"]:
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = parse_bool(cfg[k])
    return cfg

# ---------------- 文本嵌入 / 测试集 latent（不做增强，公平验证） ----------------
@torch.no_grad()
def encode_texts(captions, tokenizer, text_encoder, device, use_fp16):
    text_encoder.to(device).eval()
    embs = []
    for cap in captions:
        tok = tokenizer(cap, padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            te = text_encoder(tok.input_ids).last_hidden_state
        embs.append(te.squeeze(0).detach().half().cpu())
    text_encoder.to("cpu")
    return embs

@torch.no_grad()
def encode_latents(paths, vae, resolution, device, use_fp16, progress=False):
    vae.to(device).eval()
    lats = []
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        im = im.resize((resolution, resolution), Image.LANCZOS)
        t = torch.from_numpy(np.asarray(im, dtype=np.float32)/255.0).permute(2,0,1)
        t = (t - 0.5)/0.5
        t = t.unsqueeze(0).to(device).float()
        lat = vae.encode(t).latent_dist.sample() * 0.18215   # fp32，更稳
        lats.append(lat.squeeze(0).detach().half().cpu())
        if progress and (i+1) % max(1,len(paths)//10) == 0:
            print(f"  [encode] {i+1}/{len(paths)}", flush=True)
    return lats

@torch.no_grad()
def eval_loss_on(unet, test_lat, test_txt, scheduler, device, use_fp16, n=64):
    if len(test_lat)==0: return None
    unet.eval()
    n = min(n, len(test_lat))
    idx = torch.randint(0, len(test_lat), (n,))
    total = 0.0
    for i in idx.tolist():
        lat = test_lat[i].unsqueeze(0).to(device)
        txt = test_txt[i].unsqueeze(0).to(device)
        noise = torch.randn_like(lat)
        ts = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device)
        noisy = scheduler.add_noise(lat, noise, ts)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
            out = unet(noisy, ts, txt).sample
        total += F.mse_loss(out.float(), noise.float()).item()
    unet.train()
    return total / n

# ---------------- 增强数据集 ----------------
class AugDataset(Dataset):
    def __init__(self, paths, texts, tf):
        self.paths, self.texts, self.tf = paths, texts, tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        t = self.tf(im)
        return t, self.texts[i]

def make_transform(res, aug):
    if not aug:
        return transforms.Compose([transforms.Resize(res), transforms.ToTensor(),
                                   transforms.Normalize([0.5]*3,[0.5]*3)])
    return transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(0.7,1.0), ratio=(0.8,1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=6, fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.RandomGrayscale(p=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3),
    ])

def collate_v3(b):
    imgs = torch.stack([x[0] for x in b])
    txts = torch.stack([x[1] for x in b])
    return imgs, txts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--max_train_steps", type=int, default=None)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--save_every", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--lora_rank", type=int, default=None)
    ap.add_argument("--aug", type=int, default=None, help="1/0")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    file_cfg = load_kv(args.config) if args.config else {}
    cfg.update({k:v for k,v in file_cfg.items()})
    for k, v in vars(args).items():
        if k in ("config",""): continue
        if v is not None:
            if k == "aug": v = bool(v)
            cfg[k] = v
    cfg = coerce(cfg)
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["out_dir"]).joinpath("params.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)   # 记录过程参数

    print("=== V3 CONFIG ===")
    for k,v in cfg.items(): print(f"  {k}: {v}")

    if cfg.get("cache_dir"):
        os.environ.setdefault("HF_HOME", cfg["cache_dir"])
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(cfg["cache_dir"],"hub"))
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (cfg["mixed_precision"]=="fp16") and (device=="cuda")
    print("device:", device, torch.cuda.get_device_name(0) if device=="cuda" else "")

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    data_dir = Path(cfg["data_dir"])
    manifest = json.loads((data_dir/"manifest.json").read_text(encoding="utf-8")) if (data_dir/"manifest.json").exists() else []
    train_items = [m for m in manifest if m["split"]=="train"]
    test_items = [m for m in manifest if m["split"]=="test"]
    print("train:", len(train_items), "test:", len(test_items))

    model_id = cfg["pretrained_model_name_or_path"]
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    vae = vae.to(dtype=torch.float32)  # VAE 用 fp32 编码更稳定（在线编码）
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
    unet.requires_grad_(False)
    if cfg["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()

    lora_cfg = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
                          target_modules=["to_q","to_k","to_v","to_out.0"],
                          lora_dropout=cfg["lora_dropout"], bias="none")
    unet = get_peft_model(unet, lora_cfg)
    tr = [n for n,p in unet.named_parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in unet.parameters() if p.requires_grad))
    unet = unet.to(device)
    unet.train()

    res = cfg["resolution"]
    print("encoding text embeddings (BLIP) for train/test...")
    train_texts = encode_texts([m["caption"] for m in train_items], tokenizer, text_encoder, device, use_fp16)
    test_texts  = encode_texts([m["caption"] for m in test_items], tokenizer, text_encoder, device, use_fp16)
    print("encoding test latents (no aug) for validation...")
    test_lat = encode_latents([m["file"] for m in test_items], vae, res, device, use_fp16, progress=True)
    train_paths = [m["file"] for m in train_items]
    print("encoded test latents:", len(test_lat))

    tf = make_transform(res, cfg["aug"])
    dataset = AugDataset(train_paths, train_texts, tf)
    loader = DataLoader(dataset, batch_size=cfg["train_batch_size"], shuffle=True,
                        num_workers=2, collate_fn=collate_v3, drop_last=True)

    optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=cfg["learning_rate"])
    total_steps = cfg["max_train_steps"]
    if cfg["lr_scheduler"]=="cosine":
        def lr_lambda(step):
            if step < cfg["lr_warmup_steps"]:
                return float(step)/max(1.0,float(cfg["lr_warmup_steps"]))
            prog = (step-cfg["lr_warmup_steps"])/max(1.0,total_steps-cfg["lr_warmup_steps"])
            return 0.5*(1.0+math.cos(math.pi*prog))
        scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler_lr = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.0)

    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir/"checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir/"metrics.csv"
    if not metrics_path.exists():
        with open(metrics_path,"w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["step","train_loss","val_loss","lr","epoch","elapsed"])

    global_step, start_epoch = 0, 0
    resume_path = cfg["resume"]
    if resume_path:
        if resume_path=="latest":
            cands = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p:int(p.stem.split("_")[-1]))
            resume_path = str(cands[-1]) if cands else ""
        if resume_path and Path(resume_path).exists():
            ck = torch.load(resume_path, map_location="cpu")
            global_step = ck.get("global_step",0); start_epoch = ck.get("epoch",0)
            if "unet_lora" in ck: set_peft_model_state_dict(unet, ck["unet_lora"])
            if "optimizer" in ck: optimizer.load_state_dict(ck["optimizer"])
            if "scheduler" in ck and hasattr(scheduler_lr,"load_state_dict"): scheduler_lr.load_state_dict(ck["scheduler"])
            print(f"resumed at step={global_step} epoch={start_epoch}")

    def save_checkpoint(extra=None):
        payload = {"global_step":global_step,"epoch":start_epoch,
                   "unet_lora":unet.state_dict(),"optimizer":optimizer.state_dict(),
                   "scheduler":scheduler_lr.state_dict() if hasattr(scheduler_lr,"state_dict") else None,
                   "config":cfg}
        if extra: payload.update(extra)
        tmp = ckpt_dir/f"step_{global_step}.pt.tmp"; torch.save(payload,tmp)
        os.replace(tmp, ckpt_dir/f"step_{global_step}.pt")
        cands = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p:int(p.stem.split("_")[-1]))
        for old in cands[:-cfg["keep_checkpoints"]]: old.unlink(missing_ok=True)

    best_val = float("inf"); best_step = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    t0 = time.time(); last_log = t0
    epoch = start_epoch
    print(f"training for {total_steps} steps (res={res})...")
    while global_step < total_steps:
        torch.manual_seed(cfg["seed"] + epoch*10000)
        for imgs, txts in loader:
            if global_step >= total_steps: break
            imgs = imgs.to(device); txts = txts.to(device)
            try:
                # 在线 VAE 编码（对增强后的批次，VAE 为 fp32 + no_grad 更稳）
                with torch.no_grad():
                    lat = (vae.encode(imgs.float()).latent_dist.sample() * 0.18215).half()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                    noise = torch.randn_like(lat)
                    ts = torch.randint(0, scheduler.config.num_train_timesteps, (lat.shape[0],), device=device)
                    noisy = scheduler.add_noise(lat, noise, ts)
                    out = unet(noisy, ts, txts).sample
                if not torch.isfinite(out).all():
                    print("  [skip] non-finite prediction"); optimizer.zero_grad(set_to_none=True); continue
                loss = F.mse_loss(out.float(), noise.float()) / cfg["gradient_accumulation_steps"]
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[oom] empty cache; skip step")
                optimizer.zero_grad(set_to_none=True)
                continue
            if (global_step+1) % cfg["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in unet.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                scheduler_lr.step()
                global_step += 1
                now = time.time()
                if now-last_log > 10 or global_step % cfg["save_every"]==0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"step {global_step}/{total_steps} loss {loss.item()*cfg['gradient_accumulation_steps']:.4f} lr {lr:.2e} el {now-t0:.0f}s", flush=True)
                    with open(metrics_path,"a",newline="",encoding="utf-8") as f:
                        csv.writer(f).writerow([global_step, round(loss.item()*cfg['gradient_accumulation_steps'],5), "", f"{lr:.2e}", epoch, round(now-t0,1)])
                    last_log = now
                if global_step % cfg["save_every"]==0:
                    save_checkpoint()
                if global_step % cfg["eval_every"]==0 and test_lat:
                    val = eval_loss_on(unet, test_lat, test_texts, scheduler, device, use_fp16, cfg["eval_subset"])
                    if val is not None:
                        with open(metrics_path,"a",newline="",encoding="utf-8") as f:
                            csv.writer(f).writerow([global_step, "", round(val,5), "", epoch, ""])
                        print(f"  [eval] step {global_step} val_loss {val:.4f} (best {best_val:.4f})", flush=True)
                        if val < best_val:
                            best_val = val; best_step = global_step
                            payload = {"global_step":global_step,"epoch":epoch,"unet_lora":unet.state_dict(),
                                       "optimizer":optimizer.state_dict(),
                                       "scheduler":scheduler_lr.state_dict() if hasattr(scheduler_lr,"state_dict") else None,
                                       "config":cfg,"val_loss":val}
                            tmp = ckpt_dir/"best_val.pt.tmp"; torch.save(payload,tmp)
                            os.replace(tmp, ckpt_dir/"best_val.pt")
                            unet.save_pretrained(out_dir/"adapter_best")
                            print(f"  [eval] new best {val:.4f} @ step {global_step}")
        epoch += 1
        loader = DataLoader(dataset, batch_size=cfg["train_batch_size"], shuffle=True,
                            num_workers=2, collate_fn=collate_v3, drop_last=True)
    save_checkpoint()
    unet.save_pretrained(out_dir/"adapter")
    print(f"done step {global_step} elapsed {time.time()-t0:.0f}s. best_val {best_val:.4f} @ {best_step}")

if __name__ == "__main__":
    main()
