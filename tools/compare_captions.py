"""临时脚本：对比新 caption（Qwen2-VL）与旧 caption（BLIP）的形状与内容。"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "dataset1024" / "manifest.json").read_text(encoding="utf-8"))

new = [item for item in data if "recaption_model" in item]
old = [item for item in data if "caption_blip" in item]
missing = [item for item in data if "recaption_model" not in item]

print(f"重标注 {len(new)} / {len(data)}　未完成 {len(missing)}")
new_words = [len(item["caption"].split()) for item in new]
old_words = [len(item.get("caption_blip", "").split()) for item in old]
new_phrases = [len([p for p in item["caption"].split(",") if p.strip()]) for item in new]
print(f"  词数：新 中位 {statistics.median(new_words):.0f} / 均值 {statistics.mean(new_words):.1f}"
      f"　旧 中位 {statistics.median(old_words):.0f} / 均值 {statistics.mean(old_words):.1f}")
print(f"  新 caption 短语数中位 {statistics.median(new_phrases):.0f}"
      f"　（旧 caption 是单句 + 模板尾巴）")
tagged = sum(1 for item in new if "professional landscape photography" in item["caption"])
print(f"  仍带旧模板尾巴的 {tagged}/{len(new)}　重试过的 {sum(1 for i in new if i.get('recaption_retried'))}")

# 词汇多样性：新 caption 里出现最多的短语
counter = {}
for item in new:
    for phrase in item["caption"].split(","):
        phrase = phrase.strip().lower()
        if phrase and phrase != "professional landscape photography":
            counter[phrase] = counter.get(phrase, 0) + 1
top = sorted(counter.items(), key=lambda kv: -kv[1])[:12]
print("  最高频短语：" + "、".join(f"{name}({count})" for name, count in top))

# 关键检查：新 caption 是真的"更有信息量"，还是只把"地点模板"换成了"情绪模板"？
# 做法：把出现在 >15% 图片里的短语视为"套话"，统计每张图剩下的**具体内容短语**数量。
overused = {name for name, count in counter.items() if count > len(new) * 0.15}
concrete_counts = []
for item in new:
    phrases = [p.strip().lower() for p in item["caption"].split(",") if p.strip()]
    concrete = [p for p in phrases
                if p not in overused and p != "professional landscape photography"]
    concrete_counts.append(len(concrete))
print(f"  视为套话的短语 {len(overused)} 个（出现在 >15% 的图里）")
print(f"  每张图的**具体内容短语**数：中位 {statistics.median(concrete_counts):.0f} "
      f"均值 {statistics.mean(concrete_counts):.1f} 最少 {min(concrete_counts)}")
low_info = sum(1 for count in concrete_counts if count < 4)
print(f"  具体内容 < 4 个短语的图：{low_info}/{len(new)}（越低越好）")

# 旧 caption 的同口径对照：BLIP 句子去掉模板尾巴后的实词数
old_concrete = []
for item in old:
    text = (item.get("caption_blip") or "").lower()
    for filler in ("professional landscape photography", "scenic trail", "tropical bay",
                   "lush rainforest", "lush greenery"):
        text = text.replace(filler, "")
    old_concrete.append(len([w for w in text.replace(",", " ").split() if len(w) > 3]))
if old_concrete:
    print(f"  旧 BLIP 同口径的实词数：中位 {statistics.median(old_concrete):.0f} "
          f"均值 {statistics.mean(old_concrete):.1f}")

print("\n同一张图的前 4 组对照：")
for item in new[:4]:
    print(f"  [{item.get('entry', '?')[:26]}]")
    print(f"    旧: {(item.get('caption_blip') or '')[:100]}")
    print(f"    新: {item['caption'][:100]}")

if missing:
    print("\n未完成的样例：" + "、".join(item.get("entry", "?")[:24] for item in missing[:6]))
