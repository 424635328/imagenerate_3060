"""train_v4.py — V4 训练器（在 6GB/SD1.5 上顶格质量）。

关键特性：
  * 多随机裁剪 latent（cache_v4）→ 有效数据倍增，抗过拟合
  * LoRA rank64 + dropout + weight decay
  * caption 随机丢弃（提升 prompt 跟随）
  * EMA（对 LoRA 与可选 CLIP 微调参数做滑动平均；验证与导出均用 EMA 权重）
  * 可选 CLIP 文本编码低 LR 微调（每次按 caption 在线编码）
  * 长余弦 + 每 eval_every 在测试集验证并保存最优 checkpoint；支持断点续训 + metrics/params 记录
用法: python train_v4.py --config config_v4.cfg
"""
import argparse, json, math, os, random, time, csv
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

DEFAULTS = dict(
    pretrained_model_name_or_path="SG161222/Realistic_Vision_V6.0_B1_noVAE",
    data_dir=ROOT + "/dataset1024",
    cache=ROOT + "/dataset1024/cache_v4.pt",
    out_dir=ROOT + "/models/v4_lora",
    resolution=512,
    train_batch_size=2,
    gradient_accumulation_steps=2,
    lora_rank=64, lora_alpha=64, lora_dropout=0.1,
    learning_rate=1e-4, text_lr=1e-5, weight_decay=0.01,
    caption_dropout=0.05, finetune_text=True,
    ema_decay=0.9995,
    lr_scheduler="cosine", lr_warmup_steps=1000, max_train_steps=20000,
    gradient_checkpointing=True, mixed_precision="fp16",
    seed=42, save_every=1000, eval_every=200, eval_subset=64, keep_checkpoints=3,
    resume="", cache_dir=ROOT + "/models/hf_cache",
    init_lora="",
)

def load_kv(p):
    cfg={}; p=Path(p)
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln=ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k,v=ln.split("=",1); cfg[k.strip()]=v.strip().strip('"').strip("'")
    return cfg
def pb(v): return v if isinstance(v,bool) else str(v).lower() in ("1","true","yes","on")
def coerce(c):
    for k in ["learning_rate","text_lr","weight_decay","caption_dropout","ema_decay","lora_dropout","lora_alpha"]:
        if k in c: c[k]=float(c[k])
    for k in ["resolution","train_batch_size","gradient_accumulation_steps","lora_rank","lr_warmup_steps","max_train_steps","seed","save_every","eval_every","eval_subset","keep_checkpoints"]:
        if k in c: c[k]=int(float(c[k]))
    for k in ["gradient_checkpointing","finetune_text"]:
        if k in c and isinstance(c[k],str): c[k]=pb(c[k])
    return c

class CachedDS(Dataset):
    def __init__(self, latents, caps): self.latents, self.caps = latents, caps
    def __len__(self): return len(self.latents)
    def __getitem__(self, i): return self.latents[i], self.caps[i]
def collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b]

@torch.no_grad()
def eval_loss(unet, text_encoder, tokenizer, test_lat, test_caps, sched, dev, fp16, n=64):
    if not test_lat: return None
    unet.eval(); text_encoder.eval()
    n=min(n,len(test_lat)); idx=torch.randint(0,len(test_lat),(n,))
    tot=0.0
    for i in idx.tolist():
        lat=test_lat[i].unsqueeze(0).to(dev)
        cap=test_caps[i] if i < len(test_caps) else "a scenic landscape"
        tk=tokenizer([cap], padding="max_length", max_length=tokenizer.model_max_length,
                     truncation=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda",dtype=torch.float16,enabled=fp16):
            txt=text_encoder(tk.input_ids).last_hidden_state
            noise=torch.randn_like(lat); ts=torch.randint(0,sched.config.num_train_timesteps,(1,),device=dev)
            noisy=sched.add_noise(lat,noise,ts)
            out=unet(noisy,ts,txt).sample
        tot+=F.mse_loss(out.float(),noise.float()).item()
    return tot/n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    for k in list(DEFAULTS.keys()):
        t = type(DEFAULTS[k])
        ap.add_argument("--"+k, type=t if t in (int,float,str) else str, default=None)
    args=ap.parse_args()

    cfg=dict(DEFAULTS); cfg.update(load_kv(args.config) if args.config else {})
    for k in DEFAULTS:
        v=getattr(args,k,None)
        if v is not None: cfg[k]=v
    cfg=coerce(cfg)
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["out_dir"])/"params.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=1),encoding="utf-8")
    print("=== V4 CONFIG ==="); [print(f"  {k}: {v}") for k,v in cfg.items()]

    os.environ.setdefault("HF_HOME", cfg["cache_dir"]); os.environ.setdefault("HF_HUB_CACHE", cfg["cache_dir"]+"/hub")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    dev="cuda"; fp16=(cfg["mixed_precision"]=="fp16")

    from diffusers import DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    mid=cfg["pretrained_model_name_or_path"]
    tok=CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer")
    sched=DDPMScheduler.from_pretrained(mid, subfolder="scheduler")
    te=CLIPTextModel.from_pretrained(mid, subfolder="text_encoder", torch_dtype=torch.float16).to(dev)
    unet=UNet2DConditionModel.from_pretrained(mid, subfolder="unet", torch_dtype=torch.float16)
    unet.requires_grad_(False)
    if cfg["gradient_checkpointing"]: unet.enable_gradient_checkpointing()
    unet=get_peft_model(unet, LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        target_modules=["to_q","to_k","to_v","to_out.0"], lora_dropout=cfg["lora_dropout"], bias="none"))
    for n,p in unet.named_parameters(): p.requires_grad_(any(x in n for x in ["lora_","LoRA"]))
    unet=unet.to(dev)
    te.requires_grad_(bool(cfg["finetune_text"])); te.train() if cfg["finetune_text"] else te.eval()
    print("trainable unet:", sum(p.numel() for p in unet.parameters() if p.requires_grad),
          "| text frozen" if not cfg["finetune_text"] else "| text trainable")

    # 从已有权重初始化（支持 adapter 目录 或 checkpoint .pt）
    if cfg["init_lora"]:
        ip=Path(cfg["init_lora"]); from safetensors.torch import load_file
        if ip.is_dir() and (ip/"adapter_model.safetensors").exists():
            set_peft_model_state_dict(unet, load_file(str(ip/"adapter_model.safetensors")))
            tef=Path(str(ip)+"_text_encoder.pt")
            if tef.exists() and cfg["finetune_text"]:
                te.load_state_dict(torch.load(tef, map_location="cpu"))
            print("init from adapter dir:", ip)
        elif ip.is_file():
            ck=torch.load(ip, map_location="cpu")
            if "unet_lora" in ck: set_peft_model_state_dict(unet, ck["unet_lora"])
            if cfg["finetune_text"] and "text_state" in ck: te.load_state_dict(ck["text_state"])
            print("init from checkpoint:", ip)

    cache=torch.load(cfg["cache"], map_location="cpu")
    latents=cache["latents"]; caps=cache["caps"]; text_ids=cache["text_ids"]
    caps_per=[caps[text_ids[i]] if text_ids[i] < len(caps) else "a scenic landscape" for i in range(len(latents))]
    test_lat=cache["test_latents"]; test_txt=cache["test_texts"]
    man=json.loads((Path(cfg["data_dir"])/"manifest.json").read_text(encoding="utf-8"))
    test_caps=[(m.get("caption") or "a scenic landscape") for m in man if m["split"]=="test"]
    empty_txt=cache["empty_text"]
    print(f"train latents={len(latents)} test={len(test_lat)}")

    ds=CachedDS(latents, caps_per)
    loader=DataLoader(ds, batch_size=cfg["train_batch_size"], shuffle=True, num_workers=0, collate_fn=collate, drop_last=True)

    # optimizer（unet lora + 可选 text）
    groups=[{"params":[p for p in unet.parameters() if p.requires_grad], "lr":cfg["learning_rate"]}]
    if cfg["finetune_text"]: groups.append({"params":[p for p in te.parameters() if p.requires_grad], "lr":cfg["text_lr"]})
    opt=torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    total=cfg["max_train_steps"]
    def lam(step):
        if step<cfg["lr_warmup_steps"]: return float(step)/max(1,cfg["lr_warmup_steps"])
        prog=(step-cfg["lr_warmup_steps"])/max(1,total-cfg["lr_warmup_steps"]); return 0.5*(1+math.cos(math.pi*prog))
    sch=torch.optim.lr_scheduler.LambdaLR(opt, lam)

    # EMA
    names_u=[(n,p) for n,p in unet.named_parameters() if p.requires_grad]
    names_t=[(n,p) for n,p in te.named_parameters() if p.requires_grad] if cfg["finetune_text"] else []
    ema={("u",n):p.detach().clone() for n,p in names_u}
    ema.update({("t",n):p.detach().clone() for n,p in names_t})
    def ema_step():
        d=cfg["ema_decay"]
        with torch.no_grad():
            for n,p in names_u: ema[("u",n)].mul_(d).add_(p.detach(), alpha=1-d)
            for n,p in names_t: ema[("t",n)].mul_(d).add_(p.detach(), alpha=1-d)
    def with_ema(fn):
        bu={n:p.detach().clone() for n,p in names_u}; bt={n:p.detach().clone() for n,p in names_t}
        with torch.no_grad():
            for n,p in names_u: p.copy_(ema[("u",n)])
            for n,p in names_t: p.copy_(ema[("t",n)])
        try: return fn()
        finally:
            with torch.no_grad():
                for n,p in names_u: p.copy_(bu[n])
                for n,p in names_t: p.copy_(bt[n])

    out=Path(cfg["out_dir"]); ck_dir=out/"checkpoints"; ck_dir.mkdir(parents=True, exist_ok=True)
    metrics=out/"metrics.csv"
    if not metrics.exists():
        with open(metrics,"w",newline="",encoding="utf-8") as f: csv.writer(f).writerow(["step","train_loss","val_loss","lr","elapsed"])
    def wm(step,tr,va,lr,el):
        with open(metrics,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow([step, round(tr,5) if tr is not None else "", round(va,5) if va is not None else "", f"{lr:.2e}" if lr else "", round(el,1)])

    gstep=0
    if cfg["resume"]:
        rp=cfg["resume"]
        if rp=="latest":
            cands=sorted(ck_dir.glob("step_*.pt"), key=lambda p:int(p.stem.split("_")[-1])); rp=str(cands[-1]) if cands else ""
        if rp and Path(rp).exists():
            ck=torch.load(rp, map_location="cpu"); gstep=ck.get("global_step",0)
            if "unet_lora" in ck: set_peft_model_state_dict(unet, ck["unet_lora"])
            if cfg["finetune_text"] and "text_state" in ck: te.load_state_dict(ck["text_state"])
            if "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
            if "scheduler" in ck: sch.load_state_dict(ck["scheduler"])
            if "ema" in ck:
                for k, v in ck["ema"].items():
                    if k in ema: ema[k].copy_(v.to(ema[k].device))   # 搬回 GPU，避免设备不一致
            print("resumed at", gstep)

    def save_ckpt():
        payload={"global_step":gstep,"unet_lora":unet.state_dict(),"optimizer":opt.state_dict(),
                 "scheduler":sch.state_dict(),"ema":ema,"cfg":cfg}
        if cfg["finetune_text"]: payload["text_state"]=te.state_dict()
        tmp=ck_dir/f"step_{gstep}.pt.tmp"; torch.save(payload,tmp); os.replace(tmp, ck_dir/f"step_{gstep}.pt")
        cands=sorted(ck_dir.glob("step_*.pt"), key=lambda p:int(p.stem.split("_")[-1]))
        for old in cands[:-cfg["keep_checkpoints"]]: old.unlink(missing_ok=True)

    def export(dirname):
        def _s():
            unet.save_pretrained(out/dirname)
            if cfg["finetune_text"]: torch.save(te.state_dict(), out/(dirname+"_text_encoder.pt"))
        with_ema(_s)

    scaler=torch.amp.GradScaler("cuda", enabled=fp16)
    best=float("inf"); t0=time.time(); last=t0; micro=0
    print(f"training {total} steps ...")
    while gstep<total:
        for lat,cap in loader:
            if gstep>=total: break
            lat=lat.to(dev)
            caps_b=[("" if random.random()<cfg["caption_dropout"] else c) for c in cap]
            with torch.no_grad():
                tk=tok(caps_b, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").to(dev)
                with torch.autocast("cuda",dtype=torch.float16,enabled=fp16):
                    txt=te(tk.input_ids).last_hidden_state
            # 水平翻转增强（latent 层面）
            if random.random()<0.5: lat=torch.flip(lat,dims=[-1])
            noise=torch.randn_like(lat); ts=torch.randint(0,sched.config.num_train_timesteps,(lat.shape[0],),device=dev)
            noisy=sched.add_noise(lat,noise,ts)
            try:
                with torch.autocast("cuda",dtype=torch.float16,enabled=fp16):
                    pred=unet(noisy,ts,txt).sample
                loss=F.mse_loss(pred.float(),noise.float())/cfg["gradient_accumulation_steps"]
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); opt.zero_grad(set_to_none=True); print("[oom] skip"); continue
            micro+=1
            if micro%cfg["gradient_accumulation_steps"]==0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step(); gstep+=1; ema_step()
                now=time.time()
                if now-last>10 or gstep%cfg["save_every"]==0:
                    print(f"step {gstep}/{total} loss {loss.item()*cfg['gradient_accumulation_steps']:.4f} lr {opt.param_groups[0]['lr']:.2e} el {now-t0:.0f}s", flush=True)
                    wm(gstep, loss.item()*cfg['gradient_accumulation_steps'], None, opt.param_groups[0]['lr'], now-t0); last=now
                if gstep%cfg["save_every"]==0: save_ckpt()
                if gstep%cfg["eval_every"]==0 and test_lat:
                    val=with_ema(lambda: eval_loss(unet, te, tok, test_lat, test_caps, sched, dev, fp16, cfg["eval_subset"]))
                    unet.train(); te.train() if cfg["finetune_text"] else te.eval()
                    if val is not None:
                        wm(gstep, None, val, None, time.time()-t0)
                        print(f"  [eval] step {gstep} val {val:.4f} (best {best:.4f})", flush=True)
                        if val<best:
                            best=val; export("adapter_best")
                            print(f"  [eval] new best {val:.4f} @ {gstep} (EMA) -> adapter_best")
    save_ckpt(); export("adapter")
    print(f"done step {gstep} best {best:.4f} elapsed {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
