"""recaption.py — 用现代 VLM 重写数据集 caption（BLIP-base → Qwen2-VL-2B）。

为什么：训练数据的文本监督质量通常比换优化器更能决定上限。现有 caption 来自
BLIP-base，句式模板化、主体信息稀薄（"a view of the ocean from the top of a hill"）。
本脚本用 Qwen2-VL-2B-Instruct（原生支持，无需 trust_remote_code），4-bit 量化后可
在 6GB 卡上与其它任务共存，输出结构化、逗号分隔的英文描述。

安全设计：
  * 明确禁止臆造不存在的主体（错误 caption 比笼统 caption 更有害）；
  * 校验输出（必须英文、长度合理、非空），不合格则保留原 caption；
  * 原 caption 备份到 `caption_blip` 字段，可随时回退；manifest 另存 `.bak`。

用法:
    python recaption.py --manifest dataset1024/manifest.json --limit 5 --dry-run
    python recaption.py --manifest dataset1024/manifest.json --out dataset1024/captions_qwen.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])

INSTRUCTION = (
    "Describe this landscape photo as a training caption. "
    "Output ONLY comma-separated English noun phrases — exactly 10 to 18 of them, nothing else. "
    "No sentences, no verbs (is/are/stands/stretches/overlooks), no explanation, no quotes. "
    "Cover in order: main subject, scene, terrain, weather, lighting, time of day, colour, mood, camera style. "
    "Never invent people, buildings or events. "
    "Example: snow-capped mountain range, still alpine lake, dawn mist, golden light, cold blue tones, "
    "mirror reflection, wide angle landscape photograph"
)
# 被拒时的加强版指令（成本只花在失败样本上，而不是整批）
INSTRUCTION_RETRY = (
    "Look at the photo. Write 12 comma-separated English noun phrases describing what is visible. "
    "Do not write sentences. Do not explain. Start directly with the first phrase. "
    "Example: rocky coastline, turquoise water, white surf, sea stacks, overcast sky, soft light, "
    "cool tones, long exposure, wide angle landscape photograph"
)
STYLE_TAG = "professional landscape photography"

# 目标形状：以逗号短语为主（SD1.5 的文本域就是标签式短语，长散文会挤占 77 token 预算）
MAX_PHRASES = 20
MIN_PHRASES = 5


def phrases(text: str) -> list[str]:
    return [part.strip(" .") for part in (text or "").split(",") if part.strip(" .")]


def normalize(text: str) -> str:
    """把模型输出压成"逗号短语"形状：去掉句子残余、超长则在短语边界截断。

    为什么不直接丢弃不合格输出：实测 58% 的样本因为写成散文/超长而被 `acceptable()` 拒掉，
    GPU 时间白烧。截断到短语边界能保住大部分信息，又不会把长句塞进 CLIP 的 token 预算。
    """
    cleaned = clean(text)
    cleaned = re.sub(r"\b(is|are|was|were|stands|stretches|overlooks|surrounded by|bathed in)\b",
                     "", cleaned, flags=re.I)
    kept = phrases(cleaned)[:MAX_PHRASES]
    return ", ".join(kept)


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = text.strip('"\'` ')
    text = re.sub(r"^(the image (shows|depicts)|this is|a photograph of)\s+", "", text, flags=re.I)
    return text.rstrip(".")


def acceptable(text: str) -> bool:
    """接纳判据：短语数够、明显是英文、长度在 CLIP 77 token 预算内（≈ 55 词以内）。"""
    if not text or len(text) < 12 or len(text) > 320:
        return False
    ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(1, len(text))
    if ascii_ratio <= 0.9:
        return False
    if not (MIN_PHRASES <= len(phrases(text)) <= MAX_PHRASES + 4):
        return False
    return 5 <= len(text.split()) <= 55


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "dataset1024" / "manifest.json"))
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--out", default=str(ROOT / "dataset1024" / "captions_qwen.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-edge", type=int, default=768, help="resize before captioning (saves VRAM)")
    ap.add_argument("--max-new-tokens", type=int, default=90)
    ap.add_argument("--force", action="store_true", help="re-caption even if already done by this model")
    ap.add_argument("--dry-run", action="store_true", help="caption a few samples, write nothing")
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="每 N 张就把 manifest/captions 落盘一次，被中止后可续跑（0=只在结束时写）")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    import torch
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    path = Path(args.manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    todo = [m for m in data
            if args.force or "recaption_model" not in m]
    if args.limit:
        todo = todo[: args.limit]
    print(f"entries={len(data)} to caption={len(todo)} model={args.model}")
    if not todo:
        return 0

    print("loading captioner (4-bit)...")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model, quantization_config=quant, torch_dtype=torch.float16, device_map={"": 0})
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)

    results = {}
    kept = 0
    failed = 0
    started = time.time()

    def flush() -> None:
        """增量落盘：manifest 与 captions 都写。

        为什么必须有：2026-09-13 的一次运行在 15 分钟处被中止，因为脚本只在**跑完时**才写文件，
        已完成的部分全部作废。现在每 `--checkpoint-every` 张就落盘一次，重启时 todo 过滤器会
        跳过已带 recaption_model 的条目，续跑代价降到"最多丢 N 张"。
        `.bak` 只在第一次落盘时创建（保留原始 BLIP caption），避免后续覆盖。
        """
        backup = path.with_suffix(".json.bak")
        if not backup.exists():
            backup.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        os.replace(tmp, path)                       # 原子替换，避免中断留下半截 JSON
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

    for index, item in enumerate(todo):
        try:
            image = Image.open(item["file"]).convert("RGB")
            if max(image.size) > args.max_edge:
                scale = args.max_edge / max(image.size)
                image = image.resize((max(64, int(image.width * scale)), max(64, int(image.height * scale))),
                                     Image.LANCZOS)

            def ask(instruction: str) -> str:
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ]}]
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                         do_sample=False)
                return processor.batch_decode(
                    [chunk[len(one):] for one, chunk in zip(inputs.input_ids, out)],
                    skip_special_tokens=True)[0]

            # 先归一成"逗号短语"形状，再判接纳；不合格就用加强版指令**重试一次**
            # （成本只花在失败样本上：实测首轮接纳率约 33%，重试后显著提高）
            caption = normalize(ask(INSTRUCTION))
            retried = False
            if not acceptable(caption):
                caption = normalize(ask(INSTRUCTION_RETRY))
                retried = True
            if acceptable(caption):
                if STYLE_TAG not in caption.lower():
                    caption = f"{caption}, {STYLE_TAG}"
                item.setdefault("caption_blip", item.get("caption", ""))
                item["caption"] = caption
                item["recaption_model"] = args.model
                item["recaption_retried"] = retried
                results[item.get("entry", item["file"])] = caption
            else:
                kept += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            failed += 1
            print(f"  [oom] {item.get('entry', index)}", flush=True)
        except Exception as error:
            failed += 1
            item.setdefault("caption_error", f"{type(error).__name__}: {str(error)[:120]}")
        if (index + 1) % 50 == 0:
            rate = (time.time() - started) / (index + 1)
            print(f"  {index + 1}/{len(todo)} kept={kept} failed={failed} "
                  f"{rate:.2f}s/img eta={(len(todo) - index - 1) * rate / 60:.1f}min", flush=True)
        if not args.dry_run and args.checkpoint_every and (index + 1) % args.checkpoint_every == 0:
            flush()
            print(f"  [ckpt] 已落盘 {index + 1}/{len(todo)}", flush=True)

    print(f"\ndone: recaptioned={len(results)} kept_original={kept} failed={failed} "
          f"elapsed={(time.time() - started) / 60:.1f}min")
    for entry, caption in list(results.items())[:5]:
        print(f"  {entry[:34]:34s} {caption[:96]}")
    if args.dry_run:
        print("dry-run: nothing written")
        return 0

    flush()
    print(f"manifest updated: {path}\nbackup: {path.with_suffix('.json.bak')}\ncaptions: {args.out}")
    print("NOTE: latent 缓存里的文本嵌入已过期，重新训练前请重建缓存：")
    print("  python precompute_v5.py --base SG161222/Realistic_Vision_V6.0_B1_noVAE --arch sd15 \\")
    print("      --res 640 --crops 3 --out dataset1024/cache_v6_qwen640.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
