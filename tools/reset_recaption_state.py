"""临时脚本：重置 manifest 的重标注状态 + 单测新的归一化/接纳规则（不加载模型）。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recaption import acceptable, normalize, phrases  # noqa: E402

CASES = [
    ("散文（上一轮的真实输出）",
     "A vast expanse of calm blue ocean stretches out towards the horizon, where a distant "
     "mountain range rises against a backdrop of fluffy white clouds."),
    ("理想输出（标签式）",
     "snow-capped mountain range, still alpine lake, dawn mist, golden light, cold blue tones, "
     "wide angle landscape photograph"),
    ("带前言",
     "The image shows a small stream in the middle of the jungle, lush rainforest, lush greenery."),
    ("中文跑偏", "这是一个中文描述，说明模型跑偏了，需要重新生成。"),
    ("过短", "too short, ok"),
]

print("=== 归一化 / 接纳规则 ===")
for name, text in CASES:
    normalized = normalize(text)
    verdict = "接纳" if acceptable(normalized) else "拒绝"
    print(f"  [{name}] 短语 {len(phrases(normalized))} 个 / 词 {len(normalized.split())} 个 -> {verdict}")
    print(f"      {normalized[:110]}")

print("\n=== 重置 manifest 的重标注状态 ===")
path = ROOT / "dataset1024" / "manifest.json"
data = json.loads(path.read_text(encoding="utf-8"))
cleared = 0
for item in data:
    if "recaption_model" in item or "caption_error" in item:
        if item.get("caption_blip"):
            item["caption"] = item.pop("caption_blip")
        item.pop("recaption_model", None)
        item.pop("caption_error", None)
        cleared += 1
path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
remaining = sum(1 for item in data if "recaption_model" in item)
print(f"  清理 {cleared} 条；剩余带 recaption_model 的条目 {remaining}")
print(f"  样例（已还原为 BLIP 原文）: {data[0].get('caption', '')[:100]}")
