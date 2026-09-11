"""generate_v2_gallery.py — 单次加载 V2 管线，批量生成展示画廊（可选高清修复 / 随机模式）。

用法:
  python generate_v2_gallery.py \
      --base SG161222/Realistic_Vision_V6.0_B1_noVAE \
      --adapter <ROOT>/models/v2_lora/adapter_best \
      --prompts "a misty alpine lake at dawn,dramatic clouds|a lush rainforest waterfall|..." \
      --out_dir <ROOT>/outputs/gallery_v2 \
      --width 768 --height 768 --highres 1024 --start-seed 100 --steps 45 --cfg 7.5
  （--prompts 用 | 分隔；缺省则每个用内置随机模板。）
"""
import argparse, json, os, random, time
from pathlib import Path
from PIL import Image
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

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
    "a misty pine forest at dawn, soft light, fog",
    "a tranquil rice terrace landscape, morning mist",
]

def ahash(img, hash_size=8):
    g = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    px = list(g.getdata())
    avg = sum(px) / len(px)
    return "".join("1" if p >= avg else "0" for p in px)

def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="SG161222/Realistic_Vision_V6.0_B1_noVAE")
    ap.add_argument("--adapter", default=ROOT + "/models/v2_lora/adapter_best")
    ap.add_argument("--prompts", default="", help="| 分隔的提示词列表")
    ap.add_argument("--prompts-file", default="", help="每行一个 prompt 的文件（覆盖 --prompts）")
    ap.add_argument("--style", default="", help="附加风格后缀（如 'oil painting, impasto'）")
    ap.add_argument("--out_dir", default=ROOT + "/outputs/gallery_v2")
    ap.add_argument("--manifest", default=ROOT + "/dataset/manifest.json")
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--highres", type=int, default=0)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--negative", default="blurry, low quality, watermark, text, distorted, oversaturated, bad anatomy")
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--start-seed", type=int, default=100)
    ap.add_argument("--scheduler", default="dpmpp2m_karras")
    ap.add_argument("--count", type=int, default=1, help="随机模式下生成的张数")
    ap.add_argument("--dedup-threshold", type=int, default=6)
    ap.add_argument("--max-retry", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dataset_hashes = []
    if Path(args.manifest).exists():
        try:
            data = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            dataset_hashes = [m["ahash"] for m in data if m.get("ahash")]
            print("loaded", len(dataset_hashes), "dataset hashes")
        except Exception as e:
            print("warn manifest", e)

    import torch
    from diffusers import (DiffusionPipeline, DPMSolverMultistepScheduler,
                           EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                           DDIMScheduler, StableDiffusionImg2ImgPipeline)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print("loading base:", args.base, "on", device)
    pipe = DiffusionPipeline.from_pretrained(args.base,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None, requires_safety_checker=False)
    # VAE 半精度修复（可选）
    if device == "cuda":
        try:
            pipe.vae.config.force_upcast = False
        except Exception:
            pass
    if args.adapter and Path(args.adapter).exists():
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        pipe.unet = get_peft_model(pipe.unet, LoraConfig.from_pretrained(args.adapter))
        set_peft_model_state_dict(pipe.unet, load_file(str(Path(args.adapter) / "adapter_model.safetensors")))
        print("LoRA applied:", args.adapter)
    pipe = pipe.to(device)
    # scheduler
    cls = {"dpmpp2m": DPMSolverMultistepScheduler, "dpmpp2m_karras": DPMSolverMultistepScheduler,
           "euler_a": EulerAncestralDiscreteScheduler, "euler": EulerDiscreteScheduler,
           "ddim": DDIMScheduler}.get(args.scheduler.lower())
    if cls:
        pipe.scheduler = cls.from_config(pipe.scheduler.config, **({"use_karras_sigmas": True} if "karras" in args.scheduler else {}))
        print("scheduler:", args.scheduler)
    if device == "cuda":
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()   # 降低高分辨率解码显存
            pipe.enable_vae_tiling()
        except Exception as e:
            print("memory opts skip:", e)

    # prompts
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split("|") if p.strip()]
    elif args.prompts_file and Path(args.prompts_file).exists():
        prompts = [ln.strip() for ln in Path(args.prompts_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        prompts = [random.choice(RANDOM_TEMPLATES) for _ in range(args.count)]
    if args.style and prompts:
        prompts = [f"{p}, {args.style}" for p in prompts]

    # img2img for highres
    i2i = None
    if args.highres:
        i2i = StableDiffusionImg2ImgPipeline(vae=pipe.vae, text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer, unet=pipe.unet, scheduler=pipe.scheduler,
                feature_extractor=getattr(pipe, "feature_extractor", None),
                safety_checker=None, requires_safety_checker=False).to(device)

    results = []
    seed = args.seed if args.seed >= 0 else args.start_seed
    for idx, p in enumerate(prompts):
        base_seed = seed + idx
        gen = torch.Generator(device=device).manual_seed(base_seed)
        attempt = 0
        while True:
            g = pipe(prompt=p, negative_prompt=args.negative, num_inference_steps=args.steps,
                     guidance_scale=args.cfg, width=args.width, height=args.height,
                     generator=gen).images[0]
            attempt += 1
            h = ahash(g)
            near = (min(hamming(h, d) for d in dataset_hashes) <= args.dedup_threshold) if dataset_hashes else False
            if (not near) or attempt >= args.max_retry:
                break
            base_seed = seed + idx + attempt * 1000
            gen = torch.Generator(device=device).manual_seed(base_seed)
            print("  [dedup] regen")
        if i2i:
            try:
                init = g.resize((args.highres, args.highres), Image.LANCZOS)
                g = i2i(prompt=p, negative_prompt=args.negative, image=init, strength=0.45,
                        num_inference_steps=max(12, int(args.steps * 0.6)), guidance_scale=args.cfg,
                        generator=torch.Generator(device=device).manual_seed(base_seed)).images[0]
                print(f"  [highres] -> {args.highres}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); print("  [highres] OOM kept base")
            except Exception as e:
                print("  [highres] skipped", e)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"v2_{ts}_{idx}_{'rand' if not args.prompts else 'prompt'}.png"
        fpath = out_dir / fname
        g.save(fpath)
        results.append({"file": str(fpath), "prompt": p, "seed": base_seed, "n_attempts": attempt})
        print(f"saved {fpath}")
        seed += 1
    (out_dir / "gallery.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done. ->", out_dir)

if __name__ == "__main__":
    main()
