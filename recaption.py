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
    "你是一位风景摄影图库的编辑，正在为文生图模型准备训练描述。"
    "用英文、逗号分隔的短语写一句描述（最多 45 个词），依次覆盖："
    "画面主体与场景、环境与地貌、天气与光线、时间或季节、色调与氛围、"
    "以及可能的摄影风格（如 long exposure、aerial view）。"
    "只描述画面中确实存在的内容，绝对不要臆造人物、建筑或事件；不要写完整句子。"
)
STYLE_TAG = "professional landscape photography"


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = text.strip('"\'` ')
    text = re.sub(r"^(the image (shows|depicts)|this is)\s+", "", text, flags=re.I)
    if text and not text.endswith("."):
        pass
    return text.rstrip(".")


def acceptable(text: str) -> bool:
    if not text or len(text) < 12 or len(text) > 320:
        return False
    ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(1, len(text))
    return ascii_ratio > 0.9 and 3 <= len(text.split()) <= 60


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
    for index, item in enumerate(todo):
        try:
            image = Image.open(item["file"]).convert("RGB")
            if max(image.size) > args.max_edge:
                scale = args.max_edge / max(image.size)
                image = image.resize((max(64, int(image.width * scale)), max(64, int(image.height * scale))),
                                     Image.LANCZOS)
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION},
            ]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            generated = processor.batch_decode(
                [chunk[len(one):] for one, chunk in zip(inputs.input_ids, out)],
                skip_special_tokens=True)[0]
            caption = clean(generated)
            if acceptable(caption):
                if STYLE_TAG not in caption.lower():
                    caption = f"{caption}, {STYLE_TAG}"
                item.setdefault("caption_blip", item.get("caption", ""))
                item["caption"] = caption
                item["recaption_model"] = args.model
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

    print(f"\ndone: recaptioned={len(results)} kept_original={kept} failed={failed} "
          f"elapsed={(time.time() - started) / 60:.1f}min")
    for entry, caption in list(results.items())[:5]:
        print(f"  {entry[:34]:34s} {caption[:96]}")
    if args.dry_run:
        print("dry-run: nothing written")
        return 0

    path.with_suffix(".json.bak").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"manifest updated: {path}\nbackup: {path.with_suffix('.json.bak')}\ncaptions: {args.out}")
    print("NOTE: latent 缓存里的文本嵌入已过期，重新训练前请重建缓存：")
    print("  python precompute_v5.py --res 640 --out dataset1024/cache_v6_640.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
