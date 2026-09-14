"""test_human_verdict.py — 人工评判结果的门禁（纯 CPU）。

人工判据现在只剩这一条还站得住（VLM 裁判那条已被证伪），所以它的"可算、可核、不自夸"
必须由门禁保证：

1. **数据可用**：提交文件 schema 正确、每条都有洗牌顺序（否则"盲"不成立）、(prompt, seed) 不重复、
   没有 User-Agent 之类的可识别信息、没有个人绝对路径。
2. **数字可复算**：Wilson 区间与 `tools/vlm_judge.py` 的公式逐位一致；卡方、位置偏置、
   跨 seed 一致性都能被独立重算（第三条实现，前两条是页面与服务端）。
3. **不自夸**：没有任何版本显著高于随机时，文档里**不许**出现"更好/超过/优于"这类正面结论；
   同时必须写明统计功效边界（"无法区分"≠"一样好"）。
4. **可重生成**：文档是确定性的（连跑两次逐字节相同），且与 `research/human_verdict.json` 一致。

用法: python tools/test_human_verdict.py     （退出码 0 = 全部通过）
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import human_verdict as hv  # noqa: E402

PASSED = 0
FAILED: list[str] = []
LEAK = re.compile(r"[A-Za-z]:\\\\?Users|/home/[a-z]|user_agent|nfp_[A-Za-z0-9]{8,}")


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  [ok]   {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}  {detail}")


def wilson_from_judge():
    """从 tools/vlm_judge.py 原文抽取 wilson()：口径只允许有一处定义。"""
    source = (ROOT / "tools" / "vlm_judge.py").read_text(encoding="utf-8")
    match = re.search(r"^def wilson\(.*?\n(?=^def |\Z)", source, re.S | re.M)
    namespace = {"math": math}
    exec(match.group(0), namespace)          # noqa: S102 —— 只跑本项目自己的纯函数
    return namespace["wilson"]


def main() -> int:
    files = sorted((ROOT / "research" / "human_judge").glob("verdicts_*.json"))
    print("提交文件")
    check("research/human_judge/ 里至少有一份提交", bool(files),
          "先跑 python tools/judge_collector.py 并在页面上提交")
    if not files:
        print(f"\n结果：{PASSED} 项通过，{len(FAILED)} 项失败")
        return 1

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    records = [row for payload in payloads for row in payload["records"]]
    check("每份都是 schema=1 / kind=human_verdicts",
          all(p.get("schema") == 1 and p.get("kind") == "human_verdicts" for p in payloads))
    check("服务端写入了独立的 summary（不依赖页面自算）",
          all(isinstance(p.get("summary"), dict) and p["summary"].get("headline") for p in payloads))
    check("没有把 User-Agent 或本机路径带进来",
          not LEAK.search(json.dumps(payloads, ensure_ascii=False)),
          str(LEAK.search(json.dumps(payloads, ensure_ascii=False))))
    keys = [(row["prompt_index"], row["seed"], row.get("mode")) for row in records]
    check("(prompt, seed, mode) 没有重复（重复评判会被去重后再算）",
          len(keys) == len(set(keys)), f"{len(keys)} → {len(set(keys))}")
    versions = payloads[0]["versions"]
    check("每条记录都带 4 个槽位的洗牌顺序（盲测成立的前提）",
          all(isinstance(row.get("order"), list) and len(row["order"]) == 4
              and set(row["order"]) == set(versions) for row in records))
    check("winner 一定在当时的顺序里（选择与呈现一致）",
          all(row.get("tie") or row["winner"] in row["order"] for row in records))

    result = hv.analyse()
    print("\n数字可复算（第三条独立实现：页面 / 服务端 / 本测试）")
    wilson = wilson_from_judge()
    decided = [row for row in records if not row.get("tie")]
    wins = Counter(row["winner"] for row in decided)
    check("已决题数一致", result["decided"] == len(decided), f"{result['decided']} vs {len(decided)}")
    ok = True
    detail = []
    for row in result["versions"]:
        low, high = wilson(wins.get(row["version"], 0), len(decided))
        if abs(row["ci"][0] - low) > 1e-3 or abs(row["ci"][1] - high) > 1e-3:
            ok = False
            detail.append(f"{row['version']}: {row['ci']} vs [{low:.4f}, {high:.4f}]")
    check("每个版本的 Wilson 区间可复算", ok, "; ".join(detail))
    counts = [wins.get(v, 0) for v in versions]
    expected = len(decided) / len(versions)
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    check("均匀性卡方可复算", abs(result["chi2"] - round(chi2, 3)) < 1e-9,
          f"{result['chi2']} vs {chi2:.3f}")
    slots = Counter()
    for row in decided:
        order = row.get("order") or []
        if row["winner"] in order:
            slots["ABCD"[order.index(row["winner"])]] += 1
    slot_chi2 = sum((slots.get(s, 0) - len(decided) / 4) ** 2 / (len(decided) / 4) for s in "ABCD")
    check("位置偏置卡方可复算（这条决定结论算不算数）",
          abs(result["position"]["chi2"] - round(slot_chi2, 3)) < 1e-9,
          f"{result['position']['chi2']} vs {slot_chi2:.3f}")
    by_prompt = defaultdict(list)
    for row in decided:
        by_prompt[row["prompt_index"]].append(row["winner"])
    multi = {p: picks for p, picks in by_prompt.items() if len(picks) > 1}
    same = sum(1 for picks in multi.values() if len(set(picks)) == 1)
    check("跨 seed 一致性可复算",
          result["cross_seed"]["same_pick"] == same
          and result["cross_seed"]["prompts_with_two_seeds"] == len(multi),
          f"{result['cross_seed']} vs {same}/{len(multi)}")

    print("\n不自夸（结论必须与显著性一致）")
    doc = (ROOT / "docs" / "HUMAN_VERDICT.md").read_text(encoding="utf-8")
    if not result["any_significant"]:
        # 判据不是"出现'更好'两个字"，而是"**某个具体版本**被说成更好"：
        # "看不出谁更好"、"某个版本稳定更好"（假设语气）都是合法的否定/假设表述。
        claims = re.findall(
            r"[^。；\n]*(?:V4|V5b|V5|V6q)[^。；\n]{0,12}(?:更好|更优|优于|超过|胜出|碾压)[^。；\n]*",
            doc)
        claims = [c for c in claims if "看不出" not in c and "某个" not in c and "排除" not in c]
        check("四版本都无法区分时，文档没有把任何具体版本说成更好",
              not claims, "; ".join(claims)[:160])
        check("文档明确写出「无法区分」", "无法区分" in doc)
    check("文档写了统计功效边界（防止把「无法区分」读成「一样好」）",
          "统计功效" in doc and "排除不了" in doc)
    check("文档记录了位置偏置自检与 VLM 对照",
          "位置偏置" in doc and "144/144" in doc)
    check("文档给了复现命令", "tools/judge_collector.py" in doc and "tools/human_verdict.py" in doc)
    check("文档无个人路径/可识别信息", not LEAK.search(doc))

    print("\n可重生成（确定性）")
    # 先让工具按当前证据重算一遍（文档/JSON 是派生产物，像 lockfile 一样要跟上数据），
    # 再断言"再跑一次逐字节相同" —— 这样文档过期只会给出明确提示，不会误报成不确定。
    first = subprocess.run([sys.executable, str(ROOT / "tools" / "human_verdict.py")],
                           cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    check("tools/human_verdict.py 按当前证据重算成功", first.returncode == 0,
          (first.stderr or "")[-200:])
    summary_path = ROOT / "research" / "human_verdict.json"
    check("research/human_verdict.json 与本次分析一致",
          summary_path.exists()
          and json.loads(summary_path.read_text(encoding="utf-8"))["headline"] == result["headline"],
          str(json.loads(summary_path.read_text(encoding="utf-8"))["headline"]) if summary_path.exists() else "")
    before = (ROOT / "docs" / "HUMAN_VERDICT.md").read_text(encoding="utf-8")
    second = subprocess.run([sys.executable, str(ROOT / "tools" / "human_verdict.py")],
                            cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace",
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    after = (ROOT / "docs" / "HUMAN_VERDICT.md").read_text(encoding="utf-8")
    check("重跑一次得到逐字节相同的文档", second.returncode == 0 and before == after,
          (second.stderr or "")[-200:])
    check("文档行尾是 LF", "\r\n" not in after)

    print(f"\n结果：{PASSED} 项通过，{len(FAILED)} 项失败")
    for name in FAILED:
        print(f"  - {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
