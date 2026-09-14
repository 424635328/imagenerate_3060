"""vlm_judge.py — 自研的**盲测成对偏好裁判**：用本地 VLM 判断"哪张图更好"，而不是比训练损失。

为什么要自研这个：固定协议 val（去噪 MSE）与训练目标同源，训练越久越低，它奖励"拟合更狠"
而不是"画得更好"（2026-09-13：V5b val 0.18583 最低，但锐度/CLIP 与 V4 区分不开，效应量仅 0.3%）。
KID/CLIP-FID 衡量"像不像真实照片"，但仍然不看**提示词是否被满足**、也不看**构图是否合理**。
人类评审最贴近"质量"，但不能自动化。于是用本地 VLM 做裁判 —— 这是可复现、可扩展、
且与训练目标完全无关的判据（VLM-as-a-judge，思路同 ImageReward / PickScore 一类偏好模型）。

算法设计（三个防作弊要点，缺一个结论就不可信）：
  1. **盲测**：把两个候选的图**随机左右交换**（由固定种子决定），裁判不知道哪边是谁；
     这样"位置偏好"（VLM 常偏向某侧）会被抵消。
  2. **强制选择 + 平局允许**：只允许 A/B/TIE 三种回答，解析失败的样本单独统计（不静默丢弃）。
  3. **统计口径**：胜率带 Wilson 95% 区间；**区间跨过 50% 就是"无法区分"**，不许拿 52% 说更好。

裁判只看到 prompt + 两张图，被要求从三个具体维度比较并给出最终选择：
  主体与场景是否符合提示词 / 细节与清晰度 / 整体观感（构图、光线、是否有明显畸变）。

用法:
    python tools/vlm_judge.py --a "V4:models/v4_640/adapter_best" --b "V6:models/v6_qwen/adapter_best" \
        --prompts 24 --seeds 2 --steps 24 --res 512 --out research/judge_v4_vs_v6
    python tools/vlm_judge.py --dry-run          # 打印题目与用法，不加载任何模型
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_adapters import (BASE_LOCAL, apply_text_encoder, load_adapter_into,  # noqa: E402
                              metrics, place_pipeline, release_pipeline)
from eval_fid import HELD_OUT_PROMPTS  # noqa: E402  与 FID 评测共用同一组留出提示词

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
INSTRUCTION = (
    "You are judging two AI-generated landscape images for the SAME text prompt.\n"
    "Prompt: {prompt}\n"
    "Compare them on: (1) does the scene match the prompt, (2) detail and clarity, "
    "(3) overall look (composition, lighting, no obvious artifacts).\n"
    "Answer with exactly one line: 'WINNER: A' or 'WINNER: B' or 'WINNER: TIE'."
)


def wilson(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% 区间：小样本胜率必须给区间，否则 52% 会被误读成"更好"。"""
    if total == 0:
        return 0.0, 1.0
    phat = wins / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def parse_verdict(text: str) -> str:
    """把裁判输出解析成 'A' / 'B' / 'TIE' / 'UNPARSED'（不静默丢弃失败样本）。"""
    upper = (text or "").upper()
    for token, label in (("WINNER: A", "A"), ("WINNER: B", "B"), ("WINNER: TIE", "TIE"),
                         ("WINNER:TIE", "TIE"), ("WINNER:A", "A"), ("WINNER:B", "B")):
        if token in upper:
            return label
    if "TIE" in upper or "SAME" in upper or "EQUAL" in upper:
        return "TIE"
    return "UNPARSED"


def generate_images(adapters: list[tuple[str, str]], prompts: list[str], seeds: list[int],
                    steps: int, cfg_scale: float, res: int, out_dir: Path,
                    offload: bool, min_free_gb: float) -> dict[str, list]:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    pipe = StableDiffusionPipeline.from_pretrained(BASE_LOCAL, torch_dtype=torch.float16,
                                                   safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)

    produced: dict[str, list] = {}
    for label, path in adapters:
        load_adapter_into(pipe, path or None)
        used_te = apply_text_encoder(pipe, path) if path else apply_text_encoder(pipe, "")
        place_pipeline(pipe, offload, min_free_gb)
        images = []
        folder = out_dir / "images" / label
        folder.mkdir(parents=True, exist_ok=True)
        for index, prompt in enumerate(prompts):
            for seed in seeds:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                image = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg_scale,
                             width=res, height=res, generator=generator).images[0]
                image.save(folder / f"p{index:02d}_s{seed}.png")
                images.append((prompt, seed, image))
        produced[label] = images
        sharp = sum(metrics(item[2])["sharpness"] for item in images) / len(images)
        print(f"  [{label}] {len(images)} 张，锐度 {sharp:.1f}{'（含微调 TE）' if used_te else ''}",
              flush=True)
        release_pipeline(pipe)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return produced


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=False, help='候选 A，形如 "V4:models/v4_640/adapter_best"')
    ap.add_argument("--b", required=False, help='候选 B，形如 "V6:models/v6_qwen/adapter_best"')
    ap.add_argument("--c", default="", help="可选第三个候选（做三方两两比较）")
    ap.add_argument("--prompts", type=int, default=24, help="留出提示词条数（0=全部 24 条）")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--first-seed", type=int, default=101)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT / "research" / "judge"))
    ap.add_argument("--swap-seed", type=int, default=20240913, help="左右交换的随机种子（盲测）")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=3.5)
    ap.add_argument("--reuse-images", action="store_true",
                    help="若 <out>/images/<label> 已有图则直接复用（不重新生成）")
    ap.add_argument("--images-a", default="",
                    help="直接指定候选 A 的图片目录（跳过生成；用于跨基座比较，如 SD1.5 的 V4 vs SDXL 基座）")
    ap.add_argument("--images-b", default="")
    ap.add_argument("--images-c", default="")
    ap.add_argument("--label-a", default="")
    ap.add_argument("--label-b", default="")
    ap.add_argument("--label-c", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prompts = HELD_OUT_PROMPTS[: args.prompts] if args.prompts else HELD_OUT_PROMPTS
    seeds = [args.first_seed + 7 * index for index in range(args.seeds)]
    use_dirs = bool(args.images_a and args.images_b)
    if args.dry_run or (not use_dirs and (not args.a or not args.b)):
        print("盲测成对偏好裁判（自研）：")
        print(f"  留出提示词 {len(prompts)} 条 × 种子 {len(seeds)} = 每个候选 {len(prompts) * len(seeds)} 张")
        print("  题目示例：", prompts[0][:90])
        print("  判决格式：WINNER: A / WINNER: B / WINNER: TIE（左右随机交换，裁判不知道身份）")
        print("  统计口径：Wilson 95% 区间；区间跨 50% 即“无法区分”")
        print("\n用法：python tools/vlm_judge.py --a \"V4:models/v4_640/adapter_best\" "
              "--b \"V6:models/v6_qwen/adapter_best\" --out research/judge_v4_vs_v6")
        return 0

    specs = []
    for spec in (args.a, args.b, args.c):
        if spec:
            label, _, path = spec.partition(":")
            specs.append((label, path))
    pairs = [(specs[0], specs[1])]
    if len(specs) > 2:
        pairs += [(specs[0], specs[2]), (specs[1], specs[2])]

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in specs]

    if use_dirs:
        # 直接判"已有图片目录"：跨基座比较必须这样走 —— SD1.5 与 SDXL 的图像由各自管线生成
        # （分辨率、VAE、装载方式都不同），裁判只负责"看两个目录里的图，盲选哪张更好"。
        from PIL import Image
        directories = [args.images_a, args.images_b, args.images_c]
        produced = {}
        for slot, directory in enumerate(directories):
            if not directory:
                continue
            label = [args.label_a, args.label_b, args.label_c][slot] or Path(directory).name
            folder = Path(directory)
            files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
            expected = len(prompts) * len(seeds)
            if len(files) < expected:
                print(f"[错误] {folder} 只有 {len(files)} 张图，需要 {expected} 张"
                      f"（顺序必须与 --prompts/--seeds 一致）")
                return 2
            items = []
            cursor = 0
            for prompt in prompts:
                for seed in seeds:
                    items.append((prompt, seed, Image.open(files[cursor]).convert("RGB")))
                    cursor += 1
            produced[label] = items
            print(f"  [{label}] 读入 {len(items)} 张（{folder}）", flush=True)
        names = list(produced)
        if len(names) < 2:
            print("[错误] --images-a/--images-b 至少各给一个目录")
            return 2
        # 两两比较：2 个候选比 1 对，3 个候选比 3 对（judging 循环按 spec[0] 取标签）
        pairs = [(name, None) for name in names]
        pairs = [(pairs[0], pairs[1])] if len(pairs) == 2 else [
            (pairs[0], pairs[1]), (pairs[0], pairs[2]), (pairs[1], pairs[2])]
    elif args.reuse_images and all((out_dir / "images" / label).is_dir() for label in labels):
        from PIL import Image
        produced = {}
        for label in labels:
            folder = out_dir / "images" / label
            items = []
            for index, prompt in enumerate(prompts):
                for seed in seeds:
                    path = folder / f"p{index:02d}_s{seed}.png"
                    if path.exists():
                        items.append((prompt, seed, Image.open(path).convert("RGB")))
            produced[label] = items
        print(f"[reuse] 复用已有图片：{ {k: len(v) for k, v in produced.items()} }")
    else:
        produced = generate_images(specs, prompts, seeds, args.steps, args.cfg, args.res,
                                   out_dir, args.offload, args.min_free_gb)

    print(f"\n加载裁判 {JUDGE_MODEL}（4-bit）...", flush=True)
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    judge = Qwen2VLForConditionalGeneration.from_pretrained(
        JUDGE_MODEL, quantization_config=quant, torch_dtype=torch.float16, device_map={"": 0})
    judge.eval()
    processor = AutoProcessor.from_pretrained(JUDGE_MODEL)

    rng = random.Random(args.swap_seed)
    rows = []
    for (left_spec, right_spec) in pairs:
        left_label, right_label = left_spec[0], right_spec[0]
        wins_left = wins_right = ties = unparsed = 0
        for index, ((prompt, seed, image_left), (_, _, image_right)) in enumerate(
                zip(produced[left_label], produced[right_label])):
            # 盲测：随机决定左右，裁判永远只看到 A/B
            swap = rng.random() < 0.5
            image_a, image_b = (image_right, image_left) if swap else (image_left, image_right)
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "image"},
                {"type": "text", "text": INSTRUCTION.format(prompt=prompt)}]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image_a, image_b],
                               return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = judge.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            answer = processor.batch_decode(
                [chunk[len(one):] for one, chunk in zip(inputs.input_ids, output)],
                skip_special_tokens=True)[0]
            verdict = parse_verdict(answer)
            if verdict == "TIE":
                ties += 1
            elif verdict == "UNPARSED":
                unparsed += 1
            elif (verdict == "A") != swap:           # 还原身份
                wins_left += 1
            else:
                wins_right += 1
            rows.append({"pair": f"{left_label} vs {right_label}", "prompt": prompt[:60], "seed": seed,
                         "swapped": swap, "verdict": verdict, "raw": answer.strip()[:40]})
            if (index + 1) % 12 == 0:
                print(f"  [{left_label} vs {right_label}] {index + 1} 题："
                      f"{left_label} {wins_left} / {right_label} {wins_right} / tie {ties}", flush=True)

        decided = wins_left + wins_right
        rate = wins_left / decided if decided else 0.0
        low, high = wilson(wins_left, decided)
        margin = "显著更好" if low > 0.5 else ("显著更差" if high < 0.5 else "**无法区分**（区间跨 50%）")
        print(f"\n=== {left_label} vs {right_label} ===")
        print(f"  {left_label}: {wins_left} 胜 ｜ {right_label}: {wins_right} 胜 ｜ 平局 {ties} "
              f"｜ 解析失败 {unparsed}")
        print(f"  {left_label} 胜率 {rate * 100:.1f}%（已决 {decided} 题，Wilson 95% 区间 "
              f"{low * 100:.1f}%–{high * 100:.1f}%）→ {margin}")
        rows.append({"pair": f"{left_label} vs {right_label}", "summary": True,
                     "prompt": "", "seed": "", "swapped": "", "verdict": "",
                     "raw": f"{wins_left}/{wins_right}/{ties}/{unparsed} rate={rate:.3f} "
                            f"CI=[{low:.3f},{high:.3f}]"})

    del judge
    gc.collect()
    torch.cuda.empty_cache()

    csv_path = out_dir / "judge_verdicts.csv"
    # 汇总行多一个 summary 键，必须并进 fieldnames（否则 DictWriter 直接抛 ValueError：
    # 2026-09-14 实测三轮判完却在写 CSV 时崩溃，结果只能从日志里捞）
    fieldnames = list(dict.fromkeys(list(rows[0].keys()) + ["summary"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "judge_meta.json").write_text(json.dumps({
        "judge_model": JUDGE_MODEL, "a": args.a, "b": args.b, "c": args.c,
        "prompts": prompts, "seeds": seeds, "steps": args.steps, "cfg": args.cfg,
        "res": args.res, "swap_seed": args.swap_seed}, ensure_ascii=False, indent=1),
        encoding="utf-8", newline="\n")
    print(f"\ncsv:  {csv_path}\nmeta: {out_dir / 'judge_meta.json'}\n图:   {out_dir / 'images'}")
    print("判读：胜率区间跨 50% = 无法区分；只看点数不看区间就是在自欺。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
