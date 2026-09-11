"""inference.py — 用训练好的 LoRA 生成风景图。

输入：
  --prompt  风格描述 prompt。为空 => 随机生成（从内置风格模板随机选一个，或纯无条件）。
  --seed    -1 表示随机。
输出：
  保存到 --out_dir 下的新图；默认调用感知哈希与训练/测试集比对，若过于接近则自动重新采样。

用法：
  python inference.py --base stable-diffusion-v1-5/stable-diffusion-v1-5 \
      --lora <ROOT>/models/lora/adapter \
      --prompt "a misty alpine lake at dawn, dramatic clouds"
  随机生成：  python inference.py --lora .../adapter
"""
import argparse, json, os, random, sys, time
from pathlib import Path
from PIL import Image
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

def ahash(img, hash_size=8):
    from PIL import Image
    g = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    px = list(g.getdata())
    avg = sum(px) / len(px)
    return "".join("1" if p >= avg else "0" for p in px)

def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

# 内置风格模板（随机生成时的候选池）
RANDOM_TEMPLATES = [
    "a scenic landscape, golden hour light, dramatic clouds",
    "a rugged mountain valley at sunrise, mist, alpine lake",
    "a tropical island coastline, turquoise sea, palm trees",
    "a lush green rainforest river, mossy rocks, sunlight rays",
    "a vast golden desert with dunes under a clear blue sky",
    "a serene lake with mountains reflected, morning fog",
    "a dramatic waterfall in a mossy gorge, long exposure",
    "a wildflower meadow under a towering peak, spring",
    "a coastal cliff at sunset, waves crashing, warm light",
    "an aerial view of winding river through green valleys",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--lora", default=ROOT + "/models/lora/adapter")
    ap.add_argument("--prompt", default="", help="风格描述；空 => 随机")
    ap.add_argument("--negative", default="blurry, low quality, watermark, text, distorted, oversaturated, bad anatomy, poorly drawn")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--scheduler", default="dpmpp2m",
                    help="dpmpp2m | dpmpp2m_karras | euler_a | euler | ddim | pndm; 空则用基座默认")
    ap.add_argument("--highres", type=int, default=0,
                    help=">0 时按该边长做 img2img 高清修复(如 768 或 1024), 0 关闭")
    ap.add_argument("--out_dir", default=ROOT + "/outputs")
    ap.add_argument("--manifest", default=ROOT + "/dataset/manifest.json")
    ap.add_argument("--random-uncond", action="store_true",
                    help="prompt 为空时使用 empty prompt + cfg=1.0（完全无条件），而不是随机模板")
    ap.add_argument("--dedup-threshold", type=int, default=6, help="感知哈希海明距离阈值，低于则重新采样")
    ap.add_argument("--max-retry", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 推理")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    dataset_hashes = []
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            dataset_hashes = [m["ahash"] for m in data if m.get("ahash")]
            print(f"loaded {len(dataset_hashes)} dataset perceptual hashes for dedup")
        except Exception as e:
            print("warn: cannot load manifest for dedup", e)

    from diffusers import DiffusionPipeline
    import torch
    if args.device == "auto":
        device = "cuda" if (not args.cpu and torch.cuda.is_available()) else "cpu"
    else:
        device = "cpu" if args.cpu else args.device
    print("loading base + lora...")
    pipe = DiffusionPipeline.from_pretrained(args.base, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                                             safety_checker=None, requires_safety_checker=False)
    if args.lora and Path(args.lora).exists():
        # 用 peft 包装并注入权重（与训练时同一结构，确保 LoRA 真正生效）
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        lora_cfg = LoraConfig.from_pretrained(args.lora)
        pipe.unet = get_peft_model(pipe.unet, lora_cfg)
        sd = load_file(str(Path(args.lora) / "adapter_model.safetensors"))
        set_peft_model_state_dict(pipe.unet, sd)
        print("LoRA applied (peft):", args.lora)
    else:
        print("WARN: LoRA not found at", args.lora, "- using base model only")
    pipe = pipe.to(device)
    if args.scheduler:
        from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler,
                               EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                               UniPCMultistepScheduler)
        sched_name = args.scheduler.lower()
        sched_cls = {"dpmpp2m": DPMSolverMultistepScheduler,
                     "dpmpp2m_karras": DPMSolverMultistepScheduler,
                     "euler_a": EulerAncestralDiscreteScheduler,
                     "euler": EulerDiscreteScheduler,
                     "ddim": DDIMScheduler,
                     "unipc": UniPCMultistepScheduler}.get(sched_name)
        if sched_cls:
            if sched_name == "dpmpp2m_karras":
                kwargs = dict(use_karras_sigmas=True)
            else:
                kwargs = {}
            pipe.scheduler = sched_cls.from_config(pipe.scheduler.config, **kwargs)
            print("scheduler set:", sched_name)
    if device == "cuda":
        try:
            pipe.enable_attention_slicing()
        except Exception as e:
            print("attention slicing skipped:", e)

    prompts = []
    for _ in range(args.count):
        if args.prompt:
            p = args.prompt
        else:
            if args.random_uncond:
                p = ""   # 完全无条件
            else:
                p = random.choice(RANDOM_TEMPLATES)
        prompts.append(p)

    results = []
    for i, p in enumerate(prompts):
        seed = args.seed if args.seed >= 0 else random.randint(0, 2**31 - 1)
        generator = torch.Generator(device=device).manual_seed(seed)
        # 重新采样直到与数据集不重叠
        attempt = 0
        while True:
            g = pipe(
                prompt=p if p else "",   # 空串而非 None：SD 管线不支持 prompt=None
                negative_prompt=args.negative if p else None,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg if p else 1.0,
                width=args.width, height=args.height,
                generator=generator,
            ).images[0]
            attempt += 1
            h = ahash(g)
            near = False
            if dataset_hashes:
                m = min(hamming(h, dh) for dh in dataset_hashes)
                near = m <= args.dedup_threshold
            # 无条件随机时不做去重强制（说明为完全随机），但仍建议保留差异
            if (not p) or (not near) or attempt >= args.max_retry:
                break
            seed = random.randint(0, 2**31 - 1)
            generator = torch.Generator(device=device).manual_seed(seed)
            print("  [dedup] regenerating (too close to dataset)")
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"gen_{ts}_{i}_{'uncond' if not p else 'prompt'}.png"
        # 高清修复（img2img 放大补细节）
        if args.highres:
            try:
                from diffusers import StableDiffusionImg2ImgPipeline
                i2i = getattr(pipe, "_i2i", None)
                if i2i is None:
                    i2i = StableDiffusionImg2ImgPipeline(
                        vae=pipe.vae, text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer,
                        unet=pipe.unet, scheduler=pipe.scheduler,
                        feature_extractor=getattr(pipe, "feature_extractor", None),
                        safety_checker=None, requires_safety_checker=False
                    ).to(device)
                    pipe._i2i = i2i
                init = g.resize((args.highres, args.highres), Image.LANCZOS)
                g = i2i(
                    prompt=p if p else None,
                    negative_prompt=args.negative if p else None,
                    image=init, strength=0.45,
                    num_inference_steps=max(10, int(args.steps * 0.6)),
                    guidance_scale=args.cfg if p else 1.0,
                    generator=torch.Generator(device=device).manual_seed(seed),
                ).images[0]
                print(f"  [highres] upscaled to {args.highres}x{args.highres}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("  [highres] OOM - keeping base resolution")
            except Exception as e:
                print("  [highres] skipped:", e)
        fpath = out_dir / fname
        g.save(fpath)
        results.append({"file": str(fpath), "prompt": p, "seed": seed,
                        "n_attempts": attempt, "min_hamming": m if dataset_hashes else None})
        print(f"saved {fpath}  prompt={p!r}  seed={seed}  min_hamming={m if dataset_hashes else 'n/a'}")

    meta = out_dir / "last_generation.json"
    meta.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("metadata ->", meta)
    return results

if __name__ == "__main__":
    main()
