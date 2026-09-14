"""train_v5.py — 现代方法训练器（SD1.5 与 SDXL 共用，6GB 可用）。

相比 V4（`archive/v4/train_v4.py`）的升级点（均为 2023–2025 的方法，不改推理侧权重格式）：

  * **min-SNR-γ 损失加权**（Hang et al., arXiv:2303.09556）：按信噪比对每个
    timestep 的 MSE 加权，收敛更快、细节更稳；
  * **Prodigy 优化器**（Mishchenko & Defazio, arXiv:2306.06101）：自适应估计
    更新尺度，LoRA 上通常优于手调 AdamW；也支持 AdamW-8bit（bitsandbytes）省显存；
  * **DoRA**（Liu et al., arXiv:2402.09353 / WACV 2025）：把权重更新分解为幅度+
    方向，同秩下表达力更强（peft `use_dora=True`）；
  * **缓存文本嵌入**：SDXL 双文本编码器不再进显存（这是 6GB 能训 SDXL 的关键）；
  * 保留 V4 的 EMA、验证集选优、断点续训、caption dropout、latent 缓存。

用法:
    python train_v5.py --config config_v5.cfg
    python train_v5.py --cache dataset1024/cache_v4_640.pt --max_train_steps 120 --out_dir models/v5_smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

# Deterministic validation protocol (see eval_loss): same subset, same noise,
# same timesteps on every eval so the curve reflects the model, not the dice.
EVAL_SUBSET_SEED = 1234
EVAL_NOISE_SEED = 4321
EVAL_TIMESTEPS = [50, 250, 450, 650, 850]

DEFAULTS = dict(
    pretrained_model_name_or_path="SG161222/Realistic_Vision_V6.0_B1_noVAE",
    arch="auto",                     # auto | sd15 | sdxl
    cache=ROOT + "/dataset1024/cache_v4_640.pt",
    out_dir=ROOT + "/models/v5_lora",
    resolution=640,
    train_batch_size=1,
    gradient_accumulation_steps=2,
    lora_rank=64, lora_alpha=64, lora_dropout=0.08,
    use_dora=False,
    base_8bit=False,                 # QLoRA-style: int8 base UNet (SDXL on 6 GB)
    optimizer="prodigy",             # prodigy | adamw | adamw8bit
    learning_rate=1.0,               # Prodigy: 1.0 (=d_coef); AdamW: tune
    weight_decay=0.01,
    min_snr_gamma=5.0,               # 0 disables the weighting
    finetune_text=False, text_lr=5e-6,
    caption_dropout=0.05,
    ema_decay=0.9995,
    lr_scheduler="constant",         # constant | cosine (Prodigy wants constant)
    lr_warmup_steps=0,               # Prodigy self-warms (safeguard_warmup); AdamW: set ~200
    max_train_steps=6000,
    gradient_checkpointing=True,
    mixed_precision="fp16",
    min_free_gb=2.0,                  # preflight warning threshold (shared 6 GB laptop card)
    seed=42, save_every=1000, eval_every=200, eval_subset=48, keep_checkpoints=3,
    resume="", init_lora="", cache_dir=ROOT + "/models/hf_cache",
)


def load_kv(path: str) -> dict:
    cfg = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                # Inline comments are common in these .cfg files ("lr=0.5  # note").
                # Only cut at a '#' that follows whitespace, so values like
                # "https://host/path#frag" survive intact.
                value = re.split(r"\s+#", value, maxsplit=1)[0]
                cfg[key.strip()] = value.strip().strip('"').strip("'")
    return cfg


def as_bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")


def coerce(cfg: dict) -> dict:
    for key in ("learning_rate", "text_lr", "weight_decay", "caption_dropout",
                "ema_decay", "lora_dropout", "lora_alpha", "min_snr_gamma", "min_free_gb"):
        if key in cfg:
            cfg[key] = float(cfg[key])
    for key in ("resolution", "train_batch_size", "gradient_accumulation_steps", "lora_rank",
                "lr_warmup_steps", "max_train_steps", "seed", "save_every", "eval_every",
                "eval_subset", "keep_checkpoints"):
        if key in cfg:
            cfg[key] = int(float(cfg[key]))
    for key in ("gradient_checkpointing", "finetune_text", "use_dora", "base_8bit"):
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = as_bool(cfg[key])
    return cfg


def detect_arch(base: str, cache_dir: str) -> str:
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download(base, cache_dir=cache_dir + "/hub",
                                  allow_patterns=["model_index.json", "*.json"]))
    index = json.loads((path / "model_index.json").read_text(encoding="utf-8"))
    return "sdxl" if "text_encoder_2" in index else "sd15"


def min_snr_weights(scheduler, timesteps, gamma: float) -> torch.Tensor:
    """Hang et al. arXiv:2303.09556 — w_t = min(SNR_t, γ) / SNR_t."""
    alphas = scheduler.alphas_cumprod.to(timesteps.device)[timesteps]
    snr = alphas / (1.0 - alphas)
    return torch.stack([snr, gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr


class CachedDS(Dataset):
    def __init__(self, latents, caps):
        self.latents, self.caps = latents, caps

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, i):
        return self.latents[i], self.caps[i]


def quantize_unet_8bit(unet):
    """QLoRA-style base: swap every nn.Linear for bitsandbytes int8.

    Why not `quantization_config=BitsAndBytesConfig(...)`: diffusers 0.40 only
    routes that path through its own quantizers, and support for plain
    `UNet2DConditionModel` is not guaranteed.  Replacing the layers by hand is
    explicit, works with peft's LoRA on `Linear8bitLt`, and keeps the adapters
    in fp16 — which is what makes SDXL LoRA fit a 6 GB card.
    """
    import bitsandbytes as bnb

    replaced = 0
    for _, module in list(unet.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, torch.nn.Linear):
                continue
            new = bnb.nn.Linear8bitLt(child.in_features, child.out_features,
                                      bias=child.bias is not None,
                                      has_fp16_weights=False, threshold=6.0)
            new.weight = bnb.nn.Int8Params(child.weight.data.contiguous(),
                                           requires_grad=False, has_fp16_weights=False)
            if child.bias is not None:
                new.bias = torch.nn.Parameter(child.bias.data.clone())
            setattr(module, child_name, new)
            replaced += 1
    return replaced


def collate(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]


def plan(cfg: dict) -> int:
    """`--plan`：零显存、零 GPU 地回答"要跑多久 / 从哪续 / 缓存对不对"。

    规范要求训练脚本必须能在不加载模型的前提下自证工作量（AGENTS.md §0）。这里只做
    三件事：读缓存元数据、列检查点、算步数与等效批次；**不 import diffusers、不碰 CUDA**。
    """
    print("=== PLAN（不加载模型、不使用 GPU）===")
    print(f"  架构            : {cfg['arch']}")
    print(f"  基座            : {cfg['pretrained_model_name_or_path']}")
    print(f"  输出目录        : {cfg['out_dir']}")
    print(f"  优化器/精度     : {cfg['optimizer']} / {cfg['mixed_precision']}"
          f"{' + int8 基座' if cfg['base_8bit'] else ''}")
    print(f"  LoRA            : r={cfg['lora_rank']} alpha={cfg['lora_alpha']}"
          f"{' + DoRA' if cfg['use_dora'] else ''}")
    print(f"  文本编码器      : {'微调 (text_lr=' + str(cfg['text_lr']) + ')' if cfg['finetune_text'] else '冻结'}")

    cache_path = Path(cfg["cache"])
    if cache_path.exists():
        try:
            cache = torch.load(cache_path, map_location="cpu", mmap=True)
        except (TypeError, RuntimeError):
            cache = torch.load(cache_path, map_location="cpu")
        train_count = len(cache.get("latents", []))
        test_count = len(cache.get("test_latents", []))
        resolution = cache.get("res", cfg["resolution"])
        print(f"  缓存            : {cache_path.name}  res={resolution}  "
              f"train={train_count}  test={test_count}  crops={cache.get('crops')}  "
              f"arch={cache.get('arch', 'sd15')}")
        if int(resolution) != int(cfg["resolution"]):
            print(f"    [注意] 缓存分辨率 {resolution} 与 config 的 resolution={cfg['resolution']} 不一致"
                  f"（实际按缓存的 latent 尺寸训练）")
        if cfg["finetune_text"] and not cache.get("caps"):
            print("    [错误] finetune_text=true 需要缓存的 captions，但缓存里没有 caps")
        steps = int(cfg["max_train_steps"])
        accum = max(1, int(cfg["gradient_accumulation_steps"]))
        batch = max(1, int(cfg["train_batch_size"]))
        rewinds = 0 if train_count == 0 else -(-steps * accum * batch // train_count)
        print(f"  步数            : {steps} 步 × 累积 {accum} × batch {batch} "
              f"= 等效 batch {accum * batch}")
        print(f"  数据量          : {train_count} 个 latent ⇒ 约 {rewinds} 个 epoch")
        if test_count:
            print(f"  验证            : 每 {cfg['eval_every']} 步，子集 {cfg['eval_subset']}/{test_count}"
                  f"，固定时间步 {EVAL_TIMESTEPS}")
    else:
        print(f"  缓存            : [缺失] {cache_path} —— 先跑 precompute_v5.py")

    ck_dir = Path(cfg["out_dir"]) / "checkpoints"
    candidates = sorted(ck_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1])) \
        if ck_dir.is_dir() else []
    if candidates:
        newest = candidates[-1]
        size_mb = newest.stat().st_size / 1024 ** 2
        print(f"  检查点          : {len(candidates)} 个，最新 {newest.name}（{size_mb:.0f} MB）"
              f"，保留 {cfg['keep_checkpoints']} 个")
        if cfg["resume"]:
            print(f"  → 续训将从    : {newest.name}"
                  f"{'（--resume latest）' if cfg['resume'] == 'latest' else ''}")
        else:
            print("  → 本次为全新训练（未开 --resume）")
    else:
        print(f"  检查点          : 无（{ck_dir}）⇒ 从 0 开始")

    metrics_path = Path(cfg["out_dir"]) / "metrics.csv"
    if metrics_path.exists():
        rows = metrics_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        if len(rows) > 1:
            header, last = rows[0].split(","), rows[-1].split(",")
            observed = dict(zip(header, last))
            print(f"  上次实测        : step={observed.get('step')} "
                  f"elapsed={observed.get('elapsed')}s "
                  f"peak={observed.get('vram_gb', observed.get('vram_gb'))}GB "
                  f"free={observed.get('free_gb', 'n/a')}GB")
            try:
                step = int(float(observed.get("step", 0)))
                elapsed = float(observed.get("elapsed", 0))
                if step > 0 and elapsed > 0:
                    remaining = max(0, int(cfg["max_train_steps"]) - step)
                    print(f"  预估剩余        : {remaining} 步 × {elapsed / step:.2f} s/step "
                          f"≈ {remaining * elapsed / step / 60:.1f} 分钟（按上次实测速率）")
            except (TypeError, ValueError):
                pass

    print(f"  落盘            : 每 {cfg['save_every']} 步（原子写 .tmp → os.replace）")
    print(f"  显存门槛        : 启动前要求 ≥ {cfg['min_free_gb']} GB 空闲")
    print("=== PLAN END（去掉 --plan 即开始训练）===")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--plan", action="store_true",
                    help="只打印工作量/缓存/检查点/预估耗时，不加载模型、不使用 GPU")
    for key, value in DEFAULTS.items():
        ap.add_argument("--" + key, type=type(value) if isinstance(value, (int, float, str)) else str, default=None)
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update(load_kv(args.config) if args.config else {})
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    cfg = coerce(cfg)

    if args.plan:
        return plan(cfg)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "params.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print("=== V5 CONFIG ===")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

    os.environ.setdefault("HF_HOME", cfg["cache_dir"])
    os.environ.setdefault("HF_HUB_CACHE", cfg["cache_dir"] + "/hub")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = "cuda"
    fp16 = cfg["mixed_precision"] == "fp16"
    free_gb, total_gb = [v / 1024 ** 3 for v in torch.cuda.mem_get_info()]
    print(f"VRAM free {free_gb:.2f} GB / {total_gb:.2f} GB")
    if free_gb < float(cfg["min_free_gb"]):
        print(f"[warn] only {free_gb:.2f} GB free (< min_free_gb={cfg['min_free_gb']}): "
              "close other GPU apps (browser/LLM server/game) or training will crawl or OOM")
    arch = cfg["arch"] if cfg["arch"] != "auto" else detect_arch(cfg["pretrained_model_name_or_path"], cfg["cache_dir"])
    print(f"arch={arch}")

    from diffusers import DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    base = cfg["pretrained_model_name_or_path"]
    # 基座一律解析到本地目录（HF Hub id 会触发联网取文件，在代理环境下以 SSL 错误失败，
    # 而且失败信息完全看不出是网络问题 —— 2026-09-13 SDXL 主线就是这样被误判中止的）。
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
    from sdxl_base import describe as describe_base, offline_kwargs, resolve_base
    base = resolve_base(base)
    if base != cfg["pretrained_model_name_or_path"]:
        print(f"base resolved: {cfg['pretrained_model_name_or_path']} -> {describe_base(base)}")
    scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler", **offline_kwargs(base))
    if cfg["base_8bit"]:
        # SDXL's fp16 UNet alone is 4.9 GB; an int8 base leaves room for LoRA
        # training on a 6 GB card.  Quantized weights are frozen — only the fp16
        # adapters train (QLoRA).
        raw = UNet2DConditionModel.from_pretrained(base, subfolder="unet",
                                                   torch_dtype=torch.float16, **offline_kwargs(base))
        replaced = quantize_unet_8bit(raw)
        print(f"base UNet quantized to int8: {replaced} Linear layers replaced")
        unet = raw.to(device)
        quantized = True
    else:
        unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet",
                                                   torch_dtype=torch.float16, **offline_kwargs(base))
        quantized = False
    unet.requires_grad_(False)
    if cfg["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    unet = get_peft_model(unet, LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=["to_q", "to_k", "to_v", "to_out.0"], bias="none",
        use_dora=bool(cfg["use_dora"]),
    ))
    for name, param in unet.named_parameters():
        param.requires_grad_(any(token in name for token in ("lora_", "LoRA", "magnitude")))
    if not quantized:
        unet.to(device)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"trainable unet params: {trainable:,} ({'DoRA' if cfg['use_dora'] else 'LoRA'} r={cfg['lora_rank']})")

    text_encoder = None
    tokenizer = None
    if cfg["finetune_text"]:
        if arch == "sdxl":
            raise SystemExit("SDXL 文本编码器微调超出 6GB 预算，请保持 finetune_text=false")
        from transformers import CLIPTextModel, CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        # 可训练参数必须是 **fp32 主权重**：GradScaler 拒绝反缩放 fp16 梯度
        # （实测报 "Attempting to unscale FP16 gradients."）。历史上 V4 的 TE 阶段
        # 从未真正跑过——它把文本编码放在 no_grad 里，因此这个不兼容一直没暴露。
        text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder",
                                                     torch_dtype=torch.float32).to(device)
        text_encoder.requires_grad_(True)
        text_encoder.train()
        print(f"text encoder: trainable in fp32 (online encoding, "
              f"{(sum(p.numel() for p in text_encoder.parameters()) / 1e6):.0f}M params)")

    if cfg["init_lora"]:
        from safetensors.torch import load_file
        init_path = Path(cfg["init_lora"])
        try:
            if init_path.is_dir() and (init_path / "adapter_model.safetensors").exists():
                set_peft_model_state_dict(unet, load_file(str(init_path / "adapter_model.safetensors")))
                print("init from adapter:", init_path)
            elif init_path.is_file():
                ck = torch.load(init_path, map_location="cpu")
                if "unet_lora" in ck:
                    set_peft_model_state_dict(unet, ck["unet_lora"])
                print("init from checkpoint:", init_path)
        except Exception as error:      # e.g. plain-LoRA weights into a DoRA model
            print(f"[warn] init_lora ignored ({type(error).__name__}: {str(error)[:120]})")

    # mmap=True 让缓存按需分页而不是整份读进物理内存：SDXL 缓存约 1.1GB、
    # SD1.5 640 缓存约 419MB，训练本身并不需要它们常驻 RAM。
    try:
        cache = torch.load(cfg["cache"], map_location="cpu", mmap=True)
    except (TypeError, RuntimeError):
        cache = torch.load(cfg["cache"], map_location="cpu")
    cache_arch = cache.get("arch", "sd15")
    if cache_arch != arch:
        raise SystemExit(f"cache arch={cache_arch} but model arch={arch}; rebuild the cache")
    latents = cache["latents"]
    caps = cache["caps"]
    text_ids = cache["text_ids"]
    texts = cache["texts"]
    caps_per = [caps[text_ids[i]] if text_ids[i] < len(caps) else "a scenic landscape" for i in range(len(latents))]
    test_lat = cache["test_latents"]
    test_txt = cache["test_texts"]
    empty_text = cache["empty_text"]
    prompt_embeds = cache.get("prompt_embeds")
    pooled_embeds = cache.get("pooled_embeds")
    test_pooled = cache.get("test_pooled")
    empty_pooled = cache.get("empty_pooled")
    # cached text embeddings are only usable while the text encoder stays frozen
    use_cached_text = not cfg["finetune_text"] and texts is not None and len(texts) == len(caps)
    print(f"train latents={len(latents)} test={len(test_lat)} cached_text={use_cached_text}")

    time_ids = None
    if arch == "sdxl":
        res = int(cache.get("res", cfg["resolution"]))
        time_ids = torch.tensor([[res, res, 0, 0, res, res]], dtype=torch.float16, device=device)

    ds = CachedDS(latents, caps_per)

    def deterministic_batch(step: int, micro_index: int):
        """第 step 步的第 micro_index 个 micro-batch —— 只由 (seed, step, micro) 决定。

        为什么不用 `DataLoader(shuffle=True)`：它的排列是**每个 epoch 抽一次**，续训后从 epoch
        头部重新开始，数据顺序与"不中断地跑"不同。2026-09-13 的续训一致性测试因此失败
        （kill→resume 与一口气跑完的权重差 1.5e-3，而换 seed 只差 1.1e-1 → 差异确实来自顺序）。
        改成 (step, micro) 的纯函数后，第 k 步看到的数据与是否重启无关，续训才真的可复现
        （AGENTS.md 训练脚本规范 §2「不要用跳过前 N 个 batch 续训」的落地）。
        """
        generator = torch.Generator().manual_seed(cfg["seed"] * 1000003 + step * 131 + micro_index)
        indices = torch.randperm(len(ds), generator=generator)[: cfg["train_batch_size"]].tolist()
        return collate([ds[index] for index in indices])

    def batch_stream():
        """无限批次流：第 i 个 micro-batch 属于 step = start_step + i // accum，micro = i % accum。"""
        start_step = global_step
        index = 0
        while True:
            yield deterministic_batch(start_step + index // cfg["gradient_accumulation_steps"],
                                      index % cfg["gradient_accumulation_steps"])
            index += 1

    groups = [{"params": [p for p in unet.parameters() if p.requires_grad], "lr": cfg["learning_rate"]}]
    if text_encoder is not None:
        groups.append({"params": [p for p in text_encoder.parameters() if p.requires_grad],
                       "lr": cfg["text_lr"]})
    opt_name = cfg["optimizer"].lower()
    if opt_name == "prodigy" and len(groups) > 1 and cfg["text_lr"] != cfg["learning_rate"]:
        # Prodigy 用单一全局 d 估计，不支持参数组间不同 lr（底层会抛
        # "Setting different lr values in different parameter groups is only supported
        #  for values of 0"）。TE 与 UNet 需要不同步长时请改用 adamw/adamw8bit。
        raise SystemExit(
            "optimizer=prodigy 不支持 text_lr != learning_rate 的分组学习率；"
            "请把 optimizer 改为 adamw（或 adamw8bit）后再微调文本编码器")
    if opt_name == "prodigy":
        from prodigyopt import Prodigy
        optimizer = Prodigy(groups, decouple=True, use_bias_correction=True, safeguard_warmup=True,
                            weight_decay=cfg["weight_decay"])
    elif opt_name == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(groups, weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    print(f"optimizer={opt_name}")

    total = cfg["max_train_steps"]

    def lr_lambda(step: int) -> float:
        if step < cfg["lr_warmup_steps"]:
            return float(step) / max(1, cfg["lr_warmup_steps"])
        if cfg["lr_scheduler"] != "cosine":
            return 1.0
        progress = (step - cfg["lr_warmup_steps"]) / max(1, total - cfg["lr_warmup_steps"])
        return 0.5 * (1 + math.cos(math.pi * progress))

    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    unet_params = [(n, p) for n, p in unet.named_parameters() if p.requires_grad]
    text_params = ([(n, p) for n, p in text_encoder.named_parameters() if p.requires_grad]
                   if text_encoder is not None else [])
    ema = {("u", n): p.detach().clone() for n, p in unet_params}
    ema.update({("t", n): p.detach().clone() for n, p in text_params})

    def ema_step():
        decay = cfg["ema_decay"]
        with torch.no_grad():
            for name, param in unet_params:
                ema[("u", name)].mul_(decay).add_(param.detach(), alpha=1 - decay)
            for name, param in text_params:
                ema[("t", name)].mul_(decay).add_(param.detach(), alpha=1 - decay)

    def with_ema(fn):
        backup_u = {n: p.detach().clone() for n, p in unet_params}
        backup_t = {n: p.detach().clone() for n, p in text_params}
        with torch.no_grad():
            for name, param in unet_params:
                param.copy_(ema[("u", name)])
            for name, param in text_params:
                param.copy_(ema[("t", name)])
        try:
            return fn()
        finally:
            with torch.no_grad():
                for name, param in unet_params:
                    param.copy_(backup_u[name])
                for name, param in text_params:
                    param.copy_(backup_t[name])

    def encode(captions):
        """Batch text conditioning: cached embeddings or online CLIP encoding."""
        if use_cached_text and arch == "sd15":
            idx = []
            for caption in captions:
                try:
                    idx.append(caps.index(caption))
                except ValueError:
                    idx.append(0)
            emb = torch.stack([texts[i] for i in idx]).to(device)
            return emb, None
        if text_encoder is None:
            raise SystemExit("cache has no usable text embeddings and finetune_text=false")
        tokens = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length,
                           truncation=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            return text_encoder(tokens.input_ids).last_hidden_state, None

    def cached_conditioning(caption_indices):
        """SDXL: gather precomputed [77,2048] + pooled [1280] for the batch."""
        emb = torch.stack([prompt_embeds[i] for i in caption_indices]).to(device)
        pooled = torch.stack([pooled_embeds[i] for i in caption_indices]).to(device)
        return emb, pooled

    caption_to_index = {caption: i for i, caption in enumerate(caps)}

    @torch.no_grad()
    def eval_loss(subset: int):
        """Plain unweighted MSE on a *deterministic* protocol.

        The V4 trainer sampled a random subset and a random timestep每 eval, which
        made the curve jitter by ±0.02 — enough to hide a real improvement.  Here
        the subset, the noise and the timestep per sample are all fixed, so the
        curve only moves when the model does.  Cross-model ranking still belongs
        to tools/eval_val_mse.py (it also controls for the text encoder).
        """
        if not test_lat:
            return None
        unet.eval()
        count = min(subset, len(test_lat))
        order = torch.randperm(len(test_lat), generator=torch.Generator().manual_seed(EVAL_SUBSET_SEED))[:count].tolist()
        total_loss = 0.0
        for position, i in enumerate(order):
            lat = test_lat[i].unsqueeze(0).to(device)
            noise = torch.randn(lat.shape, generator=torch.Generator().manual_seed(EVAL_NOISE_SEED + i)).to(device, dtype=lat.dtype)
            timestep = EVAL_TIMESTEPS[position % len(EVAL_TIMESTEPS)]
            ts = torch.tensor([timestep], device=device)
            noisy = scheduler.add_noise(lat, noise, ts)
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                if arch == "sdxl":
                    pooled = test_pooled[i].unsqueeze(0).to(device) if test_pooled else empty_pooled.unsqueeze(0).to(device)
                    out = unet(noisy, ts, encoder_hidden_states=test_txt[i].unsqueeze(0).to(device),
                               added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
                else:
                    out = unet(noisy, ts, test_txt[i].unsqueeze(0).to(device)).sample
            total_loss += F.mse_loss(out.float(), noise.float()).item()
        return total_loss / count

    ck_dir = out_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    METRIC_HEADER = ["step", "train_loss", "val_loss", "lr", "elapsed",
                     "vram_gb", "reserved_gb", "free_gb"]
    # 列升级后旧文件会造成"参差不齐的行"，无法被解析；保留旧文件改名，另起一份
    if metrics_path.exists():
        with open(metrics_path, newline="", encoding="utf-8") as handle:
            current = next(csv.reader(handle), [])
        if current != METRIC_HEADER:
            metrics_path.replace(out_dir / "metrics_prev.csv")
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(METRIC_HEADER)

    def vram_snapshot() -> dict:
        """显存四元组：只看 allocated 会漏掉碎片，只看 reserved 会漏掉"别人占着"。

        6GB 卡上"还能不能跑"由 free 决定（含其它进程占用），而"我们有没有泄漏"
        由 allocated/reserved 的差距决定。卡死（WDDM 超配）时这两个数看起来都正常，
        所以还必须配合心跳看门狗（见 tools/train_driver.py --stall-seconds）。
        """
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024 ** 3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024 ** 3,
            "peak_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "free_gb": free_bytes / 1024 ** 3,
            "total_gb": total_bytes / 1024 ** 3,
        }

    def write_metric(step, train_loss, val_loss, elapsed):
        snapshot = vram_snapshot()
        with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                step,
                round(train_loss, 5) if train_loss is not None else "",
                round(val_loss, 5) if val_loss is not None else "",
                f"{optimizer.param_groups[0]['lr']:.3e}",
                round(elapsed, 1),
                round(snapshot["peak_gb"], 2),
                round(snapshot["reserved_gb"], 2),
                round(snapshot["free_gb"], 2),
            ])

    global_step = 0
    def rng_state() -> dict:
        """续训可复现的关键：不存 RNG，恢复后看到的数据顺序与噪声都变了。

        numpy 的 state 带 ndarray，而 torch 2.6 起 `torch.load` 默认 weights_only=True
        不接受 ndarray —— 所以这里把 numpy state 拆成纯 python 基本类型再存。
        """
        state = {"python": random.getstate(), "torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        try:
            name, keys, pos, has_gauss, cached = np.random.get_state()
            state["numpy"] = {"name": name, "keys": [int(k) for k in keys],
                              "pos": int(pos), "has_gauss": int(has_gauss),
                              "cached_gaussian": float(cached)}
        except (ValueError, TypeError):
            pass                                  # 非 MT19937 后端就不存，不阻塞训练
        return state

    def restore_rng(state: dict) -> None:
        if not state:
            return
        if "python" in state:
            random.setstate(state["python"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        numpy_state = state.get("numpy")
        if numpy_state:
            np.random.set_state((numpy_state["name"],
                                 np.array(numpy_state["keys"], dtype=np.uint32),
                                 numpy_state["pos"], numpy_state["has_gauss"],
                                 numpy_state["cached_gaussian"]))

    def load_checkpoint(path: Path) -> dict:
        """优先 weights_only=True（更安全）；含非张量对象（cfg/RNG 元组）时回退。"""
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(path, map_location="cpu", weights_only=False)

    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    if cfg["resume"]:
        resume_path = cfg["resume"]
        if resume_path == "latest":
            candidates = sorted(ck_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
            resume_path = str(candidates[-1]) if candidates else ""
        if resume_path and Path(resume_path).exists():
            ck = load_checkpoint(Path(resume_path))
            global_step = ck.get("global_step", 0)
            if "unet_lora" in ck:
                set_peft_model_state_dict(unet, ck["unet_lora"])
            if text_encoder is not None and "text_state" in ck:
                text_encoder.load_state_dict(ck["text_state"])
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            if "scheduler" in ck:
                schedule.load_state_dict(ck["scheduler"])
            if "ema" in ck:
                for key, value in ck["ema"].items():
                    if key in ema:
                        ema[key].copy_(value.to(ema[key].device))
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            restore_rng(ck.get("rng") or {})
            print(f"resumed at step {global_step}（RNG 状态已恢复: {'rng' in ck}）", flush=True)
        else:
            print(f"[resume] 没有可用检查点（{resume_path or '目录为空'}）——从 0 开始", flush=True)

    def save_checkpoint():
        # Only trainable adapters + optimizer + EMA are stored.  A full
        # unet.state_dict() drags the 2.3 GB frozen base along, which makes
        # frequent checkpoints — the thing that actually saves work after an
        # interrupted run — too expensive to take often.
        adapter_state = {name: tensor for name, tensor in unet.state_dict().items()
                         if "lora_" in name or "magnitude" in name}
        payload = {"global_step": global_step, "unet_lora": adapter_state,
                   "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
                   "ema": ema, "cfg": cfg, "arch": arch,
                   "rng": rng_state(),                       # 续训可复现
                   "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                   "protocol": {"eval_timesteps": EVAL_TIMESTEPS,
                                "eval_subset_seed": EVAL_SUBSET_SEED,
                                "eval_noise_seed": EVAL_NOISE_SEED}}
        if text_encoder is not None:
            payload["text_state"] = text_encoder.state_dict()
        tmp = ck_dir / f"step_{global_step}.pt.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, ck_dir / f"step_{global_step}.pt")     # 同目录原子替换
        candidates = sorted(ck_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
        for old in candidates[: -cfg["keep_checkpoints"]]:     # 只删旧的，永不删最新
            old.unlink(missing_ok=True)

    def export(dirname: str):
        def _save():
            unet.save_pretrained(out_dir / dirname)
            if text_encoder is not None:
                torch.save(text_encoder.state_dict(), out_dir / f"{dirname}_text_encoder.pt")
        with_ema(_save)

    best = float("inf")
    started = time.time()
    last_log = started
    last_step_at = started
    micro = 0
    consecutive_oom = 0
    stall_reported = False
    print(f"training {total} steps ...")
    while global_step < total:
        for lat, cap_batch in batch_stream():
            if global_step >= total:
                break
            # Shared-card watchdog: a laptop GPU is often busy with other apps.
            # Silent 40x slowdowns are worse than a loud warning.
            if not stall_reported and time.time() - last_step_at > 240:
                free_gb, _ = [v / 1024 ** 3 for v in torch.cuda.mem_get_info()]
                print(f"[stall] no optimizer step for {int(time.time() - last_step_at)}s "
                      f"(free {free_gb:.2f} GB) — another process is competing for the GPU", flush=True)
                stall_reported = True
            lat = lat.to(device)
            captions = [("" if random.random() < cfg["caption_dropout"] else c) for c in cap_batch]
            if arch == "sdxl" and use_cached_text:
                indices = [caption_to_index.get(c, caption_to_index.get("", 0)) for c in captions]
                prompt, pooled = cached_conditioning(indices)
                for position, caption in enumerate(captions):
                    if caption == "":
                        prompt[position] = empty_text.to(device)
                        pooled[position] = empty_pooled.to(device)
                extra = {"text_embeds": pooled, "time_ids": time_ids.expand(pooled.shape[0], -1)}
            else:
                prompt, _ = encode(captions)
                for position, caption in enumerate(captions):
                    if caption == "" and not cfg["finetune_text"]:
                        prompt[position] = empty_text.to(device)
                extra = None

            noise = torch.randn_like(lat)
            ts = torch.randint(0, scheduler.config.num_train_timesteps, (lat.shape[0],), device=device)
            noisy = scheduler.add_noise(lat, noise, ts)
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                    if arch == "sdxl":
                        pred = unet(noisy, ts, encoder_hidden_states=prompt, added_cond_kwargs=extra).sample
                    else:
                        pred = unet(noisy, ts, encoder_hidden_states=prompt).sample
                per_sample = F.mse_loss(pred.float(), noise.float(), reduction="none").mean(dim=[1, 2, 3])
                if cfg["min_snr_gamma"] > 0:
                    per_sample = per_sample * min_snr_weights(scheduler, ts, cfg["min_snr_gamma"])
                loss = per_sample.mean() / cfg["gradient_accumulation_steps"]
                scaler.scale(loss).backward()
                consecutive_oom = 0
            except torch.cuda.OutOfMemoryError:
                # Another process on this shared 6 GB card grew its footprint.
                # Back off instead of spinning: retry, and give up loudly.
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                consecutive_oom += 1
                free_gb, _ = [v / 1024 ** 3 for v in torch.cuda.mem_get_info()]
                print(f"[oom] retry {consecutive_oom}/5 (free {free_gb:.2f} GB)", flush=True)
                if consecutive_oom >= 5:
                    print("[oom] giving up: free the GPU (other apps hold memory) and "
                          f"resume with --resume latest", flush=True)
                    save_checkpoint()
                    return 1
                time.sleep(20)
                continue

            micro += 1
            if micro % cfg["gradient_accumulation_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group["params"]], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                schedule.step()
                global_step += 1
                last_step_at = time.time()
                stall_reported = False
                ema_step()
                now = time.time()
                if now - last_log > 10 or global_step % cfg["save_every"] == 0:
                    print(f"step {global_step}/{total} loss {loss.item() * cfg['gradient_accumulation_steps']:.4f} "
                          f"lr {optimizer.param_groups[0]['lr']:.2e} el {now - started:.0f}s "
                          f"vram {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB", flush=True)
                    write_metric(global_step, loss.item() * cfg["gradient_accumulation_steps"], None, now - started)
                    last_log = now
                if global_step % cfg["save_every"] == 0:
                    save_checkpoint()
                if global_step % cfg["eval_every"] == 0 and test_lat:
                    val = with_ema(lambda: eval_loss(cfg["eval_subset"]))
                    unet.train()
                    if text_encoder is not None:
                        text_encoder.train()
                    if val is not None:
                        write_metric(global_step, None, val, time.time() - started)
                        print(f"  [eval] step {global_step} val {val:.4f} (best {best:.4f})", flush=True)
                        if val < best:
                            best = val
                            export("adapter_best")
                            print(f"  [eval] new best {val:.4f} @ {global_step} (EMA) -> adapter_best")

    save_checkpoint()
    export("adapter")
    print(f"done step {global_step} best {best:.4f} elapsed {time.time() - started:.0f}s "
          f"peak_vram {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
