"""final_verdict.py — 把所有证据汇成一份可复核的判决书（`docs/FINAL_VERDICT.md`）。

目标要求"产出比现有 V4 质量更高的权重，并附带客观评测与前后对比证据"。本脚本把四条**互相独立**的
判据读在一起，逐候选给出结论，并**明确标注每条判据的适用边界**：

  1. 固定协议 val（`research/eval_val.csv`）+ 配对检验（`research/eval_val_paired.csv`）
     —— 衡量"对数据集分布的拟合"；与训练目标同源，训练越久越低，**不能单独作为质量判据**。
  2. 出图指标与 CLIP 一致性（`research/compare_*/compare_metrics.csv`）
     —— 部署形态下的锐度/饱和度/对比度/prompt 跟随。
  3. 留存分布距离 KID / CLIP-FID（`research/fid*/fid_metrics.csv`）
     —— 与真实风景照片的分布距离，与训练目标无关。
  4. 盲测成对偏好裁判（`research/judge*/judge_verdicts.csv`）
     —— 本地 VLM 盲测胜率 + Wilson 95% 区间；区间跨 50% 即"无法区分"。

判据冲突时的处理写在同一份文件里（例：val 更低但 KID 更差 = 拟合更狠而非画质更好）。

用法:
    python tools/final_verdict.py                       # 写出 docs/FINAL_VERDICT.md
    python tools/final_verdict.py --baseline V4          # 指定基线标签
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def aggregate_images(rows: list[dict]) -> dict[str, dict]:
    """把逐图指标按候选聚合（锐度/饱和/对比/CLIP）。"""
    buckets: dict[str, dict] = {}
    for row in rows:
        label = row.get("adapter") or ""
        if not label:
            continue
        bucket = buckets.setdefault(label, {"n": 0, "sharpness": 0.0, "saturation": 0.0,
                                            "contrast": 0.0, "clip": 0.0, "clip_n": 0})
        bucket["n"] += 1
        bucket["sharpness"] += float(row.get("sharpness") or 0)
        bucket["saturation"] += float(row.get("saturation") or 0)
        bucket["contrast"] += float(row.get("contrast") or 0)
        if row.get("clip_score"):
            bucket["clip"] += float(row["clip_score"])
            bucket["clip_n"] += 1
    for bucket in buckets.values():
        count = max(1, bucket["n"])
        bucket["sharpness"] /= count
        bucket["saturation"] /= count
        bucket["contrast"] /= count
        bucket["clip"] = bucket["clip"] / bucket["clip_n"] if bucket["clip_n"] else 0.0
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="V4")
    ap.add_argument("--out", default=str(ROOT / "docs" / "FINAL_VERDICT.md"))
    args = ap.parse_args()
    baseline = args.baseline

    val_rows = read_csv(ROOT / "research" / "eval_val.csv")
    paired_rows = [row for row in read_csv(ROOT / "research" / "eval_val_paired.csv")
                   if row.get("adapter")]
    image_dirs = [ROOT / "research" / "compare_final", ROOT / "research" / "compare_v6",
                  ROOT / "research" / "compare_sdxl"]
    image_rows: list[dict] = []
    for directory in image_dirs:
        image_rows += read_csv(directory / "compare_metrics.csv")
    fid_dirs = [ROOT / "research" / "fid_v6", ROOT / "research" / "fid"]
    fid_rows: list[dict] = []
    for directory in fid_dirs:
        fid_rows += read_csv(directory / "fid_metrics.csv")
    judge_rows = []
    judge_detail = []          # 逐题记录：位置偏置只能从这里看出来
    for directory in [ROOT / "research" / "judge_final", ROOT / "research" / "judge_v4_v5b"]:
        csv_path = directory / "judge_verdicts.csv"
        if not csv_path.exists():
            continue
        for row in read_csv(csv_path):
            if row.get("summary"):
                judge_rows.append(row)
            elif row.get("prompt"):
                judge_detail.append(row)
    val = {row["adapter"]: float(row["mean_mse"]) for row in val_rows if row.get("mean_mse")}
    images = aggregate_images(image_rows)
    fid = {row["adapter"]: row for row in fid_rows if row.get("adapter")}
    labels = [label for label in dict.fromkeys(
        list(val) + list(images) + list(fid) + [row["pair"].split(" vs ")[0] for row in judge_rows])]

    lines = ["# 最终判决书（自动生成）", "",
             f"> 生成工具：`tools/final_verdict.py`　基线：**{baseline}**　"
             f"数据源：`research/eval_val*.csv`、`research/compare_*/`、`research/fid*/`、`research/judge*/`",
             "",
             "本文件把四条**互相独立**的判据并列，任何单条都不足以判定「质量更好」：", "",
             "| 判据 | 衡量什么 | 适用边界 |", "|---|---|---|",
             "| 固定协议 val（+配对检验） | 对数据集分布的拟合 | 与训练目标同源，训练越久越低，**不能单独当质量判据** |",
             "| 出图指标 / CLIP | 部署形态下的锐度与 prompt 跟随 | 单张图的统计量，受 prompt 选择影响 |",
             "| KID / CLIP-FID | 与**真实风景照**的分布距离 | 参考集只有 144 张，KID 给了标准差 |",
             "| 盲测裁判胜率 | VLM 盲选「哪张更好」 | 裁判本身有偏，故随机左右交换 + 给区间；**本项目实测已失效（见 §4）** |", ""]

    lines += ["## 1. 固定协议 val（144 样本 × 5 时间步，同协议可比）", ""]
    if val:
        base_val = val.get(baseline)
        lines += ["| 候选 | mean MSE ↓ | Δ vs 基线 |", "|---|---|---|"]
        for label, value in sorted(val.items(), key=lambda kv: kv[1]):
            delta = f"{value - base_val:+.5f}" if base_val is not None else "—"
            lines.append(f"| {label} | {value:.5f} | {delta} |")
    else:
        lines.append("_缺失_")

    if paired_rows:
        lines += ["", "### 1.1 配对差异检验（同一 (样本, 时间步) 逐对相减）", "",
                  "| 候选 | Δ 均值（正=更好） | 标准误 | t | 配对数 |", "|---|---|---|---|---|"]
        for row in paired_rows:
            lines.append(f"| {row['adapter']} | {float(row['mean_delta']):+.5f} | "
                         f"{float(row['stderr']):.5f} | {float(row['t']):.2f} | {row['n']} |")
        lines += ["", "> 注意：t 值只说明「在这个协议下差异可重复」，**不等于**「画质更好」 —— "
                      "效应量不足 1% 时，画质判据（下面三节）才是决定性的。"]

    lines += ["", "## 2. 出图指标（部署形态，固定 prompt × seed）", ""]
    if images:
        base_img = images.get(baseline)
        has_clip = any(bucket["clip_n"] for bucket in images.values())
        header = "| 候选 | 图数 | 锐度 ↑ | 饱和度 | 对比度 |" + (" CLIP ↑ |" if has_clip else "")
        lines += [header, "|---" * (6 if has_clip else 5) + "|"]
        for label, bucket in sorted(images.items(), key=lambda kv: -kv[1]["sharpness"]):
            row = (f"| {label} | {bucket['n']} | {bucket['sharpness']:.1f} | "
                   f"{bucket['saturation']:.4f} | {bucket['contrast']:.4f} |")
            if has_clip:
                row += f" {bucket['clip']:.4f} |"
            lines.append(row)
        if base_img:
            lines += ["", f"> 基线 {baseline}：锐度 {base_img['sharpness']:.1f}、"
                          f"CLIP {base_img['clip']:.4f}"]
    else:
        lines.append("_缺失_")

    lines += ["", "## 3. 分布距离（留出 prompt，与真实风景照比较）", ""]
    if fid:
        lines += ["| 候选 | KID ↓ | KID 标准差 | CLIP-FID ↓ | 留出 prompt CLIP ↑ | 锐度 | 文本编码器 |",
                  "|---|---|---|---|---|---|---|"]
        for label, row in sorted(fid.items(), key=lambda kv: float(kv[1].get("kid") or 9e9)):
            lines.append(f"| {label} | {row.get('kid')} | {row.get('kid_std')} | "
                         f"{row.get('clip_fid') or '—'} | {row.get('clip_score')} | "
                         f"{row.get('sharpness')} | {row.get('te')} |")
    else:
        lines.append("_缺失（`tools/eval_fid.py` 尚未成功产出：需要留出 prompt 的生成结果）_")

    lines += ["", "## 4. 盲测成对偏好裁判（VLM 当裁判，随机左右交换）", ""]
    if judge_rows:
        # 位置偏置检测：裁判如果几乎总是选"先出现的那张"（A 位），随机左右交换
        # 这条防线就失效了 —— 还原后的胜率会退化成"未交换行的占比"，与图像无关。
        # 所以先算偏置，再决定这一节是"判据"还是"作废记录"。
        detail = judge_detail
        a_answers = sum(1 for row in detail if row.get("verdict") == "A")
        biased = bool(detail) and a_answers / len(detail) >= 0.98
        per_pair = {}
        for row in detail:
            bucket = per_pair.setdefault(row["pair"], [0, 0])
            bucket[0] += 1 if row.get("verdict") == "A" else 0
            bucket[1] += 1
        lines += ["| 对比 | 汇总（左/右/平/未解析）| 胜率与 Wilson 95% 区间 |", "|---|---|---|"]
        for row in judge_rows:
            lines.append(f"| {row['pair']} | {row['raw']} | 见 `research/judge_final/judge_verdicts.csv` |")
        verdict_rows = []
        for row in judge_rows:
            try:
                left, right, ties, unparsed = [int(x) for x in row["raw"].split()[0].split("/")]
                decided = left + right
                if decided:
                    import math
                    phat = left / decided
                    z = 1.96
                    denom = 1 + z * z / decided
                    centre = (phat + z * z / (2 * decided)) / denom
                    half = z * math.sqrt(phat * (1 - phat) / decided + z * z / (4 * decided * decided)) / denom
                    low, high = max(0.0, centre - half), min(1.0, centre + half)
                    verdict = ("左侧显著更好" if low > 0.5 else
                               "右侧显著更好" if high < 0.5 else "**无法区分**")
                    if biased:
                        verdict = "**判据无效**"
                    verdict_rows.append((row["pair"], phat, low, high, verdict))
            except (ValueError, IndexError):
                continue
        if verdict_rows:
            if biased:
                lines += [
                    "> ### ⚠️ 本节判据已被证伪：数字不携带质量信息",
                    ">",
                    f"> 逐题复核 `judge_verdicts.csv`：{len(detail)} 条记录里 **{a_answers} 条"
                    f"（{a_answers / max(1, len(detail)) * 100:.0f}%）都选了『先出现的那张』（A 位）**"
                    + ("；分组：" + "、".join(f"{pair} {a}/{n}" for pair, (a, n) in sorted(per_pair.items()))
                       if per_pair else "") + "。",
                    "> `tools/vlm_judge.py` 的还原逻辑本身正确（`(verdict == \"A\") != swap` ⇒ 按真实身份计数），",
                    "> 但在『永远选 A』的前提下，还原后的胜率退化成**随机交换排程里未交换行的占比**，",
                    "> 与图像内容无关 —— 因此下面的胜率与区间**不能作为任何一方的证据**，",
                    "> 也不能解读为『接近打平』。复现：`python tools/test_version_gallery.py`。",
                    ">",
                    "> 结论：四条判据实际只剩 val（非画质）、KID、CLIP 一致性，加上**人眼**。",
                    "> 人眼判据的入口：`site/versions.html`（版本评判台，含盲测与 Wilson 统计）。",
                    "",
                ]
            lines += ["", "| 对比 | 左侧胜率 | 95% 区间 | 判读 |", "|---|---|---|---|"]
            for pair, phat, low, high, verdict in verdict_rows:
                shown = f"~~{phat * 100:.1f}%~~" if biased else f"{phat * 100:.1f}%"
                lines.append(f"| {pair} | {shown} | {low * 100:.1f}%–{high * 100:.1f}% | {verdict} |")
    else:
        lines.append("_缺失（`tools/vlm_judge.py` 尚未产出）_")

    lines += ["", "## 5. 结论", ""]
    lines += [
        "判定原则（写在前面，避免事后解释）：",
        "",
        "1. **只有画质类判据（第 2–4 节）一致占优，才能说「质量更好」**；val 更低只能说明「拟合更狠」。",
        "2. 任何判据的差异如果小于它的不确定度（KID 的标准差、裁判胜率的区间），一律记为**无法区分**。",
        "3. 判据互相冲突时（例如 val 更好但 KID 更差），结论写「未证实」，不写「更好」。",
        "",
        f"基线 `{baseline}` 的参照值：val "
        f"{val.get(baseline, float('nan')):.5f}"
        + (f"、锐度 {images[baseline]['sharpness']:.1f}" if baseline in images else "")
        + (f"、CLIP {images[baseline]['clip']:.4f}" if baseline in images and images[baseline]["clip_n"] else "")
        + (f"、KID {fid[baseline].get('kid')}" if baseline in fid else "") + "。",
        "",
        "详见：`docs/EVAL_REPORT.md`（自动汇总表）、`docs/VISUAL_REVIEW.md`（视觉逐格判读）、"
        "`docs/TRAINING.md`（方法与路线决策）、`AGENTS.md`（训练脚本规范）。",
        ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print(f"verdict: {out.relative_to(ROOT)}")
    print(f"  读到：val {len(val)} 行、配对 {len(paired_rows)} 行、出图 {len(images)} 候选、"
          f"KID {len(fid)} 候选、裁判 {len(judge_rows)} 个对比")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
