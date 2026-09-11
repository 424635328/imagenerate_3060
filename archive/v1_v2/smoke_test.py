"""smoke_test.py — 单独验证 LoRA 微调 SD1.5 的前向/反向可跑通、显存与速度。

只生成随机 latents/text，不依赖数据集，用于在正式训练前校准配置。
用法: conda run -n ldm python smoke_test.py [--res 512 --rank 16 --batch 1 --steps 4]
"""
import argparse, time, os
import torch, torch.nn as nn, torch.nn.functional as F
from config import ROOT
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
ap.add_argument("--res", type=int, default=512)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=16)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--grad-ckpt", type=int, default=1)
ap.add_argument("--cache", default=str(ROOT / "models" / "hf_cache"))
args = ap.parse_args()
os.environ["HF_HOME"] = args.cache
os.environ["HF_HUB_CACHE"] = os.path.join(args.cache, "hub")

device = "cuda"
print("loading model...", args.model, flush=True)
model_id = args.model
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
unet.requires_grad_(False)
if args.grad_ckpt:
    unet.enable_gradient_checkpointing()
print("grad checkpointing:", args.grad_ckpt == 1, flush=True)

lora_config = LoraConfig(r=args.rank, lora_alpha=args.alpha,
                         target_modules=["to_q", "to_k", "to_v", "to_out.0"], lora_dropout=0.0, bias="none")
unet = get_peft_model(unet, lora_config)
tr = [n for n, p in unet.named_parameters() if p.requires_grad]
print("trainable layers:", len(tr), "total trainable params:",
      sum(p.numel() for p in unet.parameters() if p.requires_grad), flush=True)
unet = unet.to(device)
for n, p in unet.named_parameters():
    if "lora_" not in n:
        p.requires_grad_(False)
    else:
        p.requires_grad_(True)
vae = vae.to(device).eval()
text_encoder = text_encoder.to(device).eval()

print("=== forward/backward test ===", flush=True)
opt = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=True)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad():
    # encode a real-ish latent to sanity-check VAE/CLIP
    dummy = torch.randn(3, args.res, args.res, device=device)
    dummy = (dummy - dummy.mean()) / (dummy.std() + 1e-6)
    lat = vae.encode(dummy.unsqueeze(0).half()).latent_dist.sample() * 0.18215
    tok = tokenizer("a scenic landscape", padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").to(device)
    te = text_encoder(tok.input_ids).last_hidden_state
print("prep done", flush=True)
unet.train()
for i in range(args.steps):
    lat2 = lat.clone().repeat(args.batch, 1, 1, 1)
    te2 = te.clone().repeat(args.batch, 1, 1)
    noise = torch.randn_like(lat2)
    ts = torch.randint(0, scheduler.config.num_train_timesteps, (lat2.shape[0],), device=device)
    noisy = scheduler.add_noise(lat2, noise, ts)
    if i == 0:
        torch.cuda.synchronize(); t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        out = unet(noisy, ts, te2).sample
    loss = F.mse_loss(out.float(), noise.float())
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update()
torch.cuda.synchronize()
dt = time.time() - t0
print(f"steps={args.steps} res={args.res} batch={args.batch} rank={args.rank} grad_ckpt={args.grad_ckpt}")
print(f"avg step: {dt/args.steps:.3f}s  peak_mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB  reserved: {torch.cuda.max_memory_reserved()/1e9:.2f} GB", flush=True)
print("memory left:", round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())/1e9, 2), "GB")
print("OK")
