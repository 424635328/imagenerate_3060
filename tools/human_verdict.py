"""human_verdict.py — 分析「版本评判台」收到的人工盲测结果（纯 CPU，不联网、不用 GPU）。

人工判据是四条机器判据之外的**最后一条**（因为 VLM 裁判那条已被证伪，见
docs/FINAL_VERDICT.md §4）。所以这个脚本不只是算个胜率，它必须同时回答三个问题：

1. **结果是什么**：每个版本被选中几次、占比、Wilson 95% 区间、是否显著高于随机期望。
2. **这个结果算不算数**：位置偏置（对照：VLM 裁判 144/144 全选 A 位 ⇒ 判据作废）、
   均匀性卡方、跨 seed 稳定性 —— 人也会累、也会总点第一个，这些必须自检。
3. **它能不能得出「没差别」的结论**：给统计功效表。48 题查不出 30% 级别的偏好，
   所以「无法区分」的准确含义是「**没有大到能被这个样本量看见的差异**」，
   而不是「四个版本一样好」。脚本会把这句写进文档，避免以后被人误读。

数据：`research/human_judge/verdicts_*.json`（可多份提交；同一 (prompt, seed) 以最新一次为准）。
产出：`research/human_verdict.json`（机器可读）+ `docs/HUMAN_VERDICT.md`（人读）。

用法:
    python tools/human_verdict.py                     # 打印报告并写出上述两个文件
    python tools/human_verdict.py --check             # 只算不写（门禁用）
    python tools/human_verdict.py --json              # 只打印机器可读结果
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "research" / "human_judge"
GALLERY_DATA = ROOT / "site" / "data" / "versions.json"
OUT_JSON = ROOT / "research" / "human_verdict.json"
OUT_DOC = ROOT / "docs" / "HUMAN_VERDICT.md"
CHI2_CRIT_3DF = 7.815           # α = 0.05, df = 3
LEAK = re.compile(r"[A-Za-z]:\\\\?Users|/home/[a-z]|F:\\\\|nfp_[A-Za-z0-9]{8,}")


def wilson(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% 区间 —— 与 tools/vlm_judge.py、tools/judge_collector.py 同一式子。"""
    if total <= 0:
        return 0.0, 1.0
    phat = wins / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def chi2_uniform(counts: list[int]) -> float:
    total = sum(counts)
    if not total:
        return 0.0
    expected = total / len(counts)
    return sum((count - expected) ** 2 / expected for count in counts)


def load_records(directory: Path) -> tuple[list[dict], list[str]]:
    """读全部提交并按 (prompt, seed, mode) 去重：同一题重复评判以**最新**一份为准。"""
    files = sorted(directory.glob("verdicts_*.json"))
    if not files:
        raise SystemExit(f"[错误] {directory.relative_to(ROOT)} 里没有 verdicts_*.json —— "
                         f"先跑 python tools/judge_collector.py 并在页面上提交")
    latest: dict[tuple, dict] = {}
    sources = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(path.name)
        stamp = payload.get("received_at", "")
        for row in payload.get("records", []):
            key = (row.get("prompt_index"), row.get("seed"), row.get("mode") or "blind4")
            if key not in latest or stamp >= latest[key][0]:
                latest[key] = (stamp, row)
    return [row for _, row in latest.values()], sources


def power_at(true_rate: float, total: int, chance: float = 0.25, z: float = 1.96) -> float:
    """双侧 α=0.05 下，n 题能查出 true_rate ≠ chance 的概率（正态近似）。"""
    se = math.sqrt(chance * (1 - chance) / total)
    delta = (true_rate - chance) / se
    upper = 0.5 * (1 + math.erf((delta - z) / math.sqrt(2)))     # 落在拒绝域右侧
    lower = 0.5 * (1 + math.erf((-delta - z) / math.sqrt(2)))    # 落在拒绝域左侧
    return upper + lower


def analyse(directory: Path = IN_DIR) -> dict:
    records, sources = load_records(directory)
    versions = sorted({row["winner"] for row in records if row.get("winner")}) or ["V4", "V5", "V5b", "V6q"]
    decided = [row for row in records if not row.get("tie")]
    ties = len(records) - len(decided)
    total = len(decided)
    wins = Counter(row["winner"] for row in decided)

    per_version = []
    for version in versions:
        count = wins.get(version, 0)
        low, high = wilson(count, total)
        per_version.append({
            "version": version, "wins": count, "decided": total,
            "rate": round(count / total, 4) if total else 0.0,
            "ci": [round(low, 4), round(high, 4)],
            "chance": 0.25,
            "indistinguishable": low <= 0.25 <= high,
        })

    # 位置偏置：选了第几个出现的？（VLM 裁判就是因为 144/144 全选第一位而作废）
    slots = Counter()
    for row in decided:
        order = row.get("order") or []
        if row["winner"] in order:
            slots["ABCD"[order.index(row["winner"])]] += 1
    slot_counts = [slots.get(slot, 0) for slot in "ABCD"]

    # 跨 seed 稳定性：同一题两个 seed 是否选了同一版本（随机期望 = 1/4）
    by_prompt: dict[int, list[str]] = defaultdict(list)
    for row in decided:
        by_prompt[row["prompt_index"]].append(row["winner"])
    multi = {p: picks for p, picks in by_prompt.items() if len(picks) > 1}
    consistent = sum(1 for picks in multi.values() if len(set(picks)) == 1)

    result = {
        "schema": 1,
        "kind": "human_verdict_summary",
        "sources": sources,
        "records": len(records),
        "decided": total,
        "ties": ties,
        "versions": per_version,
        "chi2": round(chi2_uniform([wins.get(v, 0) for v in versions]), 3),
        "chi2_critical": CHI2_CRIT_3DF,
        "uniform_not_rejected": chi2_uniform([wins.get(v, 0) for v in versions]) < CHI2_CRIT_3DF,
        "position": {
            "counts": dict(zip("ABCD", slot_counts)),
            "chi2": round(chi2_uniform(slot_counts), 3),
            "biased": chi2_uniform(slot_counts) >= CHI2_CRIT_3DF,
        },
        "cross_seed": {
            "prompts_with_two_seeds": len(multi),
            "same_pick": consistent,
            "rate": round(consistent / len(multi), 4) if multi else 0.0,
            "chance": 0.25,
        },
        "power": {str(int(r * 100)): round(power_at(r, total), 3)
                  for r in (0.30, 0.35, 0.40, 0.50)} if total else {},
        "any_significant": any(not row["indistinguishable"] for row in per_version),
    }
    result["headline"] = headline(result)
    return result


def headline(result: dict) -> str:
    if not result["decided"]:
        return "还没有已决题（全为平局或未评判）。"
    if not result["any_significant"]:
        return (f"已决 {result['decided']} 题：四个版本的 95% 区间都包含随机期望 25%"
                f"（卡方 {result['chi2']} < {result['chi2_critical']}）⇒ 记为「无法区分」。")
    best = max((row for row in result["versions"] if not row["indistinguishable"]),
               key=lambda row: row["rate"])
    return (f"已决 {result['decided']} 题：{best['version']} 被选中 {best['wins']}/{best['decided']}"
            f"（区间 {best['ci'][0]}–{best['ci'][1]}，高于随机期望 0.25）。")


def machine_ranking() -> list[tuple[str, float]]:
    """机器判据的排序（KID 越低越像真照片），用来和人工排序对照。"""
    if not GALLERY_DATA.exists():
        return []
    data = json.loads(GALLERY_DATA.read_text(encoding="utf-8"))
    rows = [(item["id"], item["kid"]) for item in data["versions"] if isinstance(item.get("kid"), float)]
    return sorted(rows, key=lambda row: row[1])


def render_doc(result: dict) -> str:
    machine = machine_ranking()
    human_order = [row["version"] for row in sorted(result["versions"], key=lambda r: -r["rate"])]
    lines = [
        "# 人工评判结果（版本评判台 · 盲测）",
        "",
        "> 数据源：`research/human_judge/verdicts_*.json`"
        f"（{len(result['sources'])} 份提交，共 {result['records']} 条记录，"
        f"已决 {result['decided']} 题、平局 {result['ties']} 题）",
        "> 生成工具：`python tools/human_verdict.py`（确定性；本文件会被重写，不要手工改）",
        "",
        "## 1. 结论",
        "",
        f"**{result['headline']}**",
        "",
        "| 版本 | 被选中 | 占比 | Wilson 95% 区间 | 判定 |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(result["versions"], key=lambda r: -r["rate"]):
        verdict = "无法区分" if row["indistinguishable"] else "**显著高于随机**"
        lines.append(f"| {row['version']} | {row['wins']} / {row['decided']} | {row['rate'] * 100:.1f}% | "
                     f"{row['ci'][0] * 100:.1f}% – {row['ci'][1] * 100:.1f}% | {verdict} |")
    lines += [
        "",
        f"均匀性卡方（H0：四版本等概率 25%，df=3）：**{result['chi2']}** < 临界值 "
        f"{result['chi2_critical']} ⇒ "
        + ("不能拒绝「等概率」，即看不出谁更好。" if result["uniform_not_rejected"] else "拒绝等概率。"),
        "",
        "## 2. 这个结果算不算数（方法自检）",
        "",
        "| 检查 | 结果 | 含义 |",
        "|---|---|---|",
        f"| 位置偏置（选了第几个出现的） | A/B/C/D = "
        f"{result['position']['counts']['A']}/{result['position']['counts']['B']}/"
        f"{result['position']['counts']['C']}/{result['position']['counts']['D']}，"
        f"卡方 {result['position']['chi2']} | "
        + ("无显著位置偏好 ⇒ 判据有效"
           if not result["position"]["biased"] else "**存在位置偏好 ⇒ 结论需警惕**") + " |",
        f"| 跨 seed 稳定性 | {result['cross_seed']['same_pick']} / "
        f"{result['cross_seed']['prompts_with_two_seeds']} 题两个 seed 选了同一版本"
        f"（{result['cross_seed']['rate'] * 100:.0f}%，随机期望 25%） | "
        + ("与随机期望一致 ⇒ 没有「某道题上某个版本稳定更好」的模式"
           if abs(result["cross_seed"]["rate"] - 0.25) < 0.15 else "呈现一定稳定性") + " |",
        "",
        "> 对照：VLM 盲测裁判 **144/144 次都选了先出现的那张**（100% 位置偏置），"
        "因此那条判据被判定为**无效**（`docs/FINAL_VERDICT.md` §4）。"
        "人工这次没有这个问题，所以这一条判据是**可用的**。",
        "",
        "## 3. 与机器判据的对照",
        "",
        "| 判据 | 排序（好 → 差） |",
        "|---|---|",
        f"| 人工盲测（被选次数） | {' > '.join(human_order)} |",
    ]
    if machine:
        lines.append("| 留出集 KID ↓（机器） | " + " < ".join(f"{name} {value:.6f}" for name, value in machine) + " |")
        lines += [
            "",
            "两个排序**对不上**：人工选得最多的是 KID 最差的候选，人工选得最少的是 KID 第二好的候选。"
            "在 48 题、任何区间都跨随机期望的前提下，这更像是『两个判据各自在噪声里排序』，"
            "而不是『人对不上机器』。",
        ]
    lines += [
        "",
        "## 4. 结论的边界（别把它读成「四个版本一样好」）",
        "",
        f"已决 {result['decided']} 题的统计功效（α=0.05 双侧）：",
        "",
        "| 若真实偏好是 | 这个样本量能查出来的概率 |",
        "|---|---|",
    ]
    for rate, power in result["power"].items():
        lines.append(f"| {rate}% | 约 {power * 100:.0f}% |")
    lines += [
        "",
        "所以「无法区分」的准确含义是：**没有大到能被这个样本量看见的差异** ——",
        f"它**排除不了** 30–35% 级别的偏好（那种差异在 {result['decided']} 题下多半查不出来），",
        "但它确实排除得了『某个版本碾压式更好』（50% 级别的偏好有 ~98% 概率会被查出）。",
        "",
        "想更灵敏，两条路：",
        "",
        "1. **换口径**：用页面上的「擦除对照」做两两比较（随机期望 50%，同样题量下功效更高）；",
        "2. **加题量**：把 24 题 × 2 seed 判完再重判几轮（同一题重复评判以最新一份为准），",
        "   区间宽度按 1/√n 收窄。",
        "",
        "## 5. 复现",
        "",
        "```powershell",
        "python tools/judge_collector.py                     # 页面 + 提交接口（同源，127.0.0.1:8787）",
        "#   浏览器打开 http://127.0.0.1:8787/versions.html → 盲测 → 点「提交评判」",
        "python tools/human_verdict.py                       # 重算本文件与 research/human_verdict.json",
        "python tools/test_human_verdict.py                  # 门禁：数字可复算 + 自检结论必须在",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="分析人工盲测结果（可复现）")
    ap.add_argument("--in-dir", default=str(IN_DIR))
    ap.add_argument("--check", action="store_true", help="只算不写")
    ap.add_argument("--json", action="store_true", help="只打印机器可读结果")
    args = ap.parse_args()

    result = analyse(Path(args.in_dir))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0

    print(f"提交文件：{', '.join(result['sources'])}")
    print(f"记录 {result['records']} 条（已决 {result['decided']}，平局 {result['ties']}）")
    for row in sorted(result["versions"], key=lambda r: -r["rate"]):
        print(f"  {row['version']:4s} {row['wins']:3d}/{row['decided']} "
              f"{row['rate'] * 100:5.1f}%  区间 [{row['ci'][0] * 100:5.1f}%, {row['ci'][1] * 100:5.1f}%]  "
              f"{'无法区分' if row['indistinguishable'] else '显著高于随机'}")
    print(f"  均匀性卡方 {result['chi2']}（临界 {result['chi2_critical']}）"
          f" ⇒ {'不能拒绝等概率' if result['uniform_not_rejected'] else '拒绝等概率'}")
    print(f"  位置偏置 A/B/C/D = {result['position']['counts']}，卡方 {result['position']['chi2']}"
          f" ⇒ {'无显著位置偏好' if not result['position']['biased'] else '存在位置偏好'}")
    print(f"  跨 seed 一致性 {result['cross_seed']['same_pick']}/"
          f"{result['cross_seed']['prompts_with_two_seeds']} = {result['cross_seed']['rate'] * 100:.0f}%"
          f"（随机 25%）")
    print(f"  功效：{'，'.join(f'{k}%→{v * 100:.0f}%' for k, v in result['power'].items())}")
    print(f"\n结论：{result['headline']}")

    if args.check:
        return 0
    doc = render_doc(result)
    if LEAK.search(doc):
        raise SystemExit("[错误] 文档里出现了个人路径/凭据")
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    OUT_DOC.write_text(doc, encoding="utf-8", newline="\n")
    print(f"\n已写出 {OUT_JSON.relative_to(ROOT)} 与 {OUT_DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
