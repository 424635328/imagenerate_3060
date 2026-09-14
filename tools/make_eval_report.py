"""make_eval_report.py — 汇总所有可用模型，生成 docs/EVAL_REPORT.md。

目标要求的「客观评测与前后对比证据」= 这个脚本的产物。它把三层评测拼成一份可复核的报告：

  1. **跨模型 val MSE**（tools/eval_val_mse.py，固定协议）→ 判断"拟合得更好吗"
  2. **出图指标 + CLIP 图文一致性**（tools/compare_adapters.py）→ 判断"画质与 prompt 跟随更好吗"
  3. 方法与论文依据表（写死在报告模板里，便于协作者复核）

用法:
    python tools/make_eval_report.py                 # 只汇总已有结果（缺的标 not available）
    python tools/make_eval_report.py --with-images   # 另跑出图对比（需要 GPU，约 5-10 分钟）
    python tools/make_eval_report.py --run-val       # 另跑 val 评测（需要 GPU，约 2 分钟）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import compat_key  # noqa: E402  纯逻辑，不导入 torch

PY = sys.executable
SD15_CACHE = ROOT / "dataset1024" / "cache_v4_640.pt"
SDXL_CACHE = ROOT / "dataset1024" / "cache_v5_sdxl1024.pt"
SDXL_VAL_CSV = ROOT / "research" / "eval_val_sdxl.csv"
REPORT = ROOT / "docs" / "EVAL_REPORT.md"

# label -> adapter 目录（None = 未挂 adapter 的基座）
# 融合产物也在这里列出：它们的 val 已经与 V4/V5 同协议可比，出图指标同样要进报告
# （否则"融合是否更好"只有损失数字、没有画质与 prompt 跟随的证据）。
CANDIDATES = [
    ("BASE", None),
    ("V4", "models/v4_640/adapter_best"),
    ("V5", "models/v5_lora/adapter_best"),
    ("V5b", "models/v5b_lora/adapter_best"),
    ("slerp", "models/merged/slerp_0.50_0.50"),
    ("linear", "models/merged/linear_0.50_0.50"),
    ("ties", "models/merged/ties_0.50_0.50"),
    ("dare_ties", "models/merged/dare_ties_0.50_0.50"),
    ("SDXL", "models/v5_sdxl/adapter_best"),
]

METHODS = [
    ("min-SNR-γ 损失加权", "Hang et al., arXiv:2303.09556",
     "train_v5.py::min_snr_weights，γ=5"),
    ("Prodigy 优化器", "Mishchenko & Defazio, arXiv:2306.06101",
     "自适应更新尺度，LoRA 免手调 LR"),
    ("DoRA", "Liu et al., arXiv:2402.09353 (WACV 2025)",
     "权重分解为幅度+方向；已实测可被 diffusers 0.40 加载"),
    ("QLoRA（int8 基座）", "Dettmers et al., arXiv:2305.14314",
     "SDXL UNet 4.9GB → int8 2.5GB，使 6GB 卡可训"),
    ("LoRA", "Hu et al., arXiv:2106.09685", "推理侧权重格式不变，部署零改动"),
    ("SDXL + Lightning 4 步", "Podell et al., arXiv:2307.01952；ByteDance SDXL-Lightning",
     "推理质量跃升 + 少步加速（models/lightning/）"),
    ("多随机裁剪 latent 缓存", "数据集增强（本项目自研，见 README §5.2）",
     "1520 图 → 4560 latent，抗过拟合"),
]


def run(command: list[str]) -> int:
    print("run: " + " ".join(str(c) for c in command[1:]))
    return subprocess.run([str(c) for c in command], cwd=str(ROOT)).returncode


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def available() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    sd15, sdxl = [], []
    for label, relative in CANDIDATES:
        if relative is None:
            sd15.append((label, ""))
            continue
        if not (ROOT / relative / "adapter_model.safetensors").exists():
            continue
        (sdxl if label == "SDXL" else sd15).append((label, relative))
    return sd15, sdxl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-val", action="store_true", help="跑 val 评测（需要 GPU）")
    ap.add_argument("--with-images", action="store_true", help="跑出图对比（需要 GPU）")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()

    sd15, sdxl = available()
    val_csv = ROOT / "research" / "eval_val.csv"
    img_dir = ROOT / "research" / "compare_final"
    sdxl_img_dir = ROOT / "research" / "compare_sdxl"

    if args.run_val:
        specs = [f"{label}:{path}" if path else f"{label}:" for label, path in sd15]
        command = [PY, str(ROOT / "tools" / "eval_val_mse.py"), "--adapters", *specs,
                   "--cache", str(SD15_CACHE), "--csv", str(val_csv)]
        run(command)
        if sdxl and SDXL_CACHE.exists():
            # SDXL 是**另一个基座**，必须带自己的基座锚点行，否则"有没有变好"无从判断。
            sdxl_specs = ["SDXL-BASE:"] + [f"{label}:{path}" for label, path in sdxl]
            run([PY, str(ROOT / "tools" / "eval_val_mse.py"), "--adapters", *sdxl_specs,
                 "--cache", str(SDXL_CACHE), "--device", "cuda",
                 "--csv", str(SDXL_VAL_CSV)])
        elif sdxl:
            print(f"[val] SDXL adapter 存在但缓存缺失（{SDXL_CACHE.name}）—— 先跑训练流水线")

    if args.with_images and len(sd15) >= 2:
        specs = [f"{label}:{path}" for label, path in sd15 if path]
        run([PY, str(ROOT / "tools" / "compare_adapters.py"), "--adapters", *specs,
             "--out", str(img_dir), "--steps", str(args.steps), "--res", str(args.res)])

    # SDXL 出图对比：best-effort。它比 SD1.5 侧慢得多（CPU offload），失败也不该毁掉报告。
    if args.with_images and sdxl:
        sdxl_specs = ["SDXL-BASE:"] + [f"{label}:{path}" for label, path in sdxl]
        code = run([PY, str(ROOT / "tools" / "compare_sdxl.py"), "--adapters", *sdxl_specs,
                    "--out", str(sdxl_img_dir), "--steps", str(args.steps), "--res", "1024"])
        if code != 0:
            print(f"[images] SDXL 出图对比失败（exit={code}），其余证据照常生成")

    val_rows = read_csv(val_csv)
    merged_rows = read_csv(ROOT / "research" / "eval_merged.csv")
    img_rows = read_csv(img_dir / "compare_metrics.csv")

    lines = ["# 模型质量评测报告（自动生成）", "",
             f"> 生成工具：`tools/make_eval_report.py`　数据来源：`tools/eval_val_mse.py`（固定协议 val）"
             f" + `tools/compare_adapters.py`（出图指标与 CLIP 一致性）",
             "", "## 1. 方法依据", "", "| 方法 | 论文 | 本项目落点 |", "|---|---|---|"]
    lines += [f"| {name} | {paper} | {where} |" for name, paper, where in METHODS]

    lines += ["", "## 2. 参与评测的权重", "", "| 标签 | 路径 | 状态 |", "|---|---|---|"]
    for label, relative in sd15 + sdxl:
        status = "已评测" if any(r.get("adapter") == label for r in val_rows) else "可用（未评测）"
        lines.append(f"| {label} | `{relative or '(base)'}` | {status} |")

    lines += ["", "## 3. 验证损失（固定协议 val）", ""]
    if val_rows:
        base = next((float(r["mean_mse"]) for r in val_rows if r["adapter"] == "BASE"), None)
        v4 = next((float(r["mean_mse"]) for r in val_rows if r["adapter"] == "V4"), None)
        steps = (val_rows[0].get("timesteps") or "50|250|450|650|850").replace("|", " × ")
        canonical_key = compat_key(val_rows[0])
        sample_count = canonical_key[3] if canonical_key else (val_rows[0].get("samples") or "?")
        device = val_rows[0].get("device") or "cuda（旧格式行未记录）"
        lines += [f"> 协议：固定排列的前 {sample_count} 个测试样本 × 时间步 {steps}（device={device}）。",
                  "> 同一 (样本, 时间步) 用固定种子噪声，文本条件统一用 base CLIP 缓存嵌入"
                  "（刻意排除文本编码器差异，只衡量 UNet 侧适配质量）。", ""]

        # 合并 adapter 的结果写在独立 CSV 里；只有可比较键一致的行才允许并进下表，
        # 否则报告会把两种协议的均值并列，那正是本项目反复强调不能做的事。
        # 注意：`eval_merged.csv` 里可能预置了 BASE/V4/V5 锚点行（为了复用评测结果），
        # 这些标签已经在 §3 的主表里，必须去重，否则同一个模型会出现两行。
        seen_labels = {row["adapter"] for row in val_rows}
        mergeable = [row for row in merged_rows
                     if compat_key(row) == canonical_key and row["adapter"] not in seen_labels]
        in_comparable = [row for row in merged_rows if compat_key(row) != canonical_key]
        if merged_rows and not mergeable:
            lines += [f"> 合并 adapter 的结果（`research/eval_merged.csv`）与上表协议不同，"
                      f"**未并入**，需按同协议重跑后再比较。", ""]
        elif mergeable:
            names = ", ".join(f"`{row['adapter']}`" for row in mergeable)
            lines += [f"> 并入的合并 adapter（来自 `research/eval_merged.csv`，可比较键一致）：{names}", ""]
        if in_comparable:
            lines += [f"> 另有 {len(in_comparable)} 行因协议不一致被排除："
                      + ", ".join(f"`{row['adapter']}`" for row in in_comparable), ""]

        lines += ["| 模型 | mean MSE ↓ | Δ vs BASE | Δ vs V4 | 样本数 × 时间步 |", "|---|---|---|---|---|"]
        for row in val_rows + mergeable:
            mean = float(row["mean_mse"])
            d_base = f"{mean - base:+.5f}" if base is not None else "—"
            d_v4 = f"{mean - v4:+.5f}" if v4 is not None else "—"
            key = compat_key(row)
            shape = f"{key[3]} × {len(key[4])}" if key else f"{row['samples']} × ?"
            lines.append(f"| {row['adapter']} | {mean:.5f} | {d_base} | {d_v4} | {shape} |")

        # 配对检验：均值的绝对差只有 5e-4 量级，必须给"这个差是不是真的"一个统计口径
        paired_rows = read_csv(ROOT / "research" / "eval_val_paired.csv")
        if paired_rows:
            baseline_label = paired_rows[0].get("baseline", "V4")
            lines += ["", f"**配对差异检验（同一 (样本, 时间步) 上逐对相减，基准 = {baseline_label}）**", "",
                      "| 模型 | Δ 均值（正=更好） | 标准误 | t 值 | 配对数 | 判读 |", "|---|---|---|---|---|---|"]
            for row in sorted(paired_rows, key=lambda r: -float(r["mean_delta"])):
                t_value = float(row["t"])
                if abs(t_value) >= 3:
                    verdict = "显著"
                elif abs(t_value) > 1.96:
                    verdict = "边缘"
                else:
                    verdict = "不显著"
                direction = "更好" if float(row["mean_delta"]) > 0 else "更差"
                lines.append(f"| {row['adapter']} | {float(row['mean_delta']):+.5f} | "
                             f"{float(row['stderr']):.5f} | {t_value:.2f} | {row['n']} | {direction}，{verdict} |")
            lines += ["",
                      "> 为什么必须配对：本协议样本间标准差约 0.19，只看均值的标准误 ≈ 0.007，"
                      "而候选之间的差异只有 5e-4 —— 非配对口径判不出任何差别。逐对相减后样本间方差抵消，"
                      "剩下的才是真实差异（见 `tools/eval_common.py::paired_stats`）。"]
    else:
        lines.append("_尚未运行：`python tools/make_eval_report.py --run-val`_")

    # ---- 3.1 SDXL 线：另一个基座、另一份缓存，必须单独列，不能与上表并列比较 ----
    sdxl_rows = read_csv(SDXL_VAL_CSV)
    lines += ["", "### 3.1 SDXL 线（不同基座与缓存，独立协议；**不可与上表并列比较**）", ""]
    if sdxl_rows:
        anchor = next((r for r in sdxl_rows
                       if r.get("adapter") in ("SDXL-BASE", "BASE")), None)
        anchor_mean = float(anchor["mean_mse"]) if anchor else None
        key = compat_key(sdxl_rows[0]) if sdxl_rows else None
        if key:
            lines += [f"> 协议：{key[0]}｜{key[1]}｜{key[2]}｜前 {key[3]} 个测试样本 × "
                      f"{len(key[4])} 个时间步。", ""]
        lines += ["| 模型 | mean MSE ↓ | Δ vs SDXL-BASE | 样本数 × 时间步 |", "|---|---|---|---|"]
        for row in sdxl_rows:
            mean = float(row["mean_mse"])
            delta = f"{mean - anchor_mean:+.5f}" if anchor_mean is not None else "—"
            row_key = compat_key(row)
            shape = f"{row_key[3]} × {len(row_key[4])}" if row_key else f"{row.get('samples')} × ?"
            lines.append(f"| {row['adapter']} | {mean:.5f} | {delta} | {shape} |")
        if anchor_mean is None:
            lines.append("\n> **缺少 SDXL 基座锚点行**：没有它就无法判断 LoRA 是否真的变好，"
                         "请用 `--run-val` 重跑。")
    else:
        lines.append("_尚未运行：SDXL 训练完成后用 `python tools/make_eval_report.py --run-val` 生成_")

    lines += ["", "## 4. 出图指标与 prompt 跟随（固定 prompt × seed）", ""]
    if img_rows:
        labels = []
        for row in img_rows:
            if row["adapter"] not in labels:
                labels.append(row["adapter"])
        has_clip = "clip_score" in img_rows[0] and img_rows[0]["clip_score"]
        header = "| 模型 | 锐度 ↑ | 饱和度 | 对比度 |" + (" CLIP 一致性 ↑ |" if has_clip else "")
        lines += [header, "|---" * (5 if has_clip else 4) + "|"]
        for label in labels:
            subset = [r for r in img_rows if r["adapter"] == label]
            n = len(subset)

            def mean_of(key):
                return sum(float(r[key]) for r in subset if r.get(key)) / n

            row = f"| {label} | {mean_of('sharpness'):.1f} | {mean_of('saturation'):.4f} | {mean_of('contrast'):.4f} |"
            if has_clip:
                row += f" {mean_of('clip_score'):.4f} |"
            lines.append(row)
        sheet = img_dir / "compare_sheet.png"
        if sheet.exists():
            lines.append(f"\n对比拼版：`{sheet.relative_to(ROOT)}`（左→右按上表顺序）")
    else:
        lines.append("_尚未运行：`python tools/make_eval_report.py --with-images`_")

    # ---- 4.1 SDXL 出图对比（同 prompt/seed/指标口径，但基座不同）----
    sdxl_img_rows = read_csv(sdxl_img_dir / "compare_metrics.csv")
    lines += ["", "### 4.1 SDXL 出图对比（1024，同 prompt × seed；基座与第 4 节不同）", ""]
    if sdxl_img_rows:
        has_clip = "clip_score" in sdxl_img_rows[0] and sdxl_img_rows[0]["clip_score"]
        header = "| 模型 | 锐度 ↑ | 饱和度 | 对比度 |" + (" CLIP 一致性 ↑ |" if has_clip else "")
        lines += [header, "|---" * (5 if has_clip else 4) + "|"]
        ordered = []
        for row in sdxl_img_rows:
            if row["adapter"] not in ordered:
                ordered.append(row["adapter"])
        for label in ordered:
            subset = [r for r in sdxl_img_rows if r["adapter"] == label]
            n = len(subset)

            def mean_sdxl(key):
                return sum(float(r[key]) for r in subset if r.get(key)) / n

            row_text = (f"| {label} | {mean_sdxl('sharpness'):.1f} | {mean_sdxl('saturation'):.4f} | "
                        f"{mean_sdxl('contrast'):.4f} |")
            if has_clip:
                row_text += f" {mean_sdxl('clip_score'):.4f} |"
            lines.append(row_text)
        sdxl_sheet = sdxl_img_dir / "compare_sheet.png"
        probe_file = sdxl_img_dir / "sdxl_probe.json"
        if sdxl_sheet.exists():
            lines.append(f"\n对比拼版：`{sdxl_sheet.relative_to(ROOT)}`")
        if probe_file.exists():
            try:
                probe = json.loads(probe_file.read_text(encoding="utf-8"))
                arms = probe.get("arms", [])
                fastest = min(arms, key=lambda a: a.get("seconds_per_image", 1e9)) if arms else None
                if fastest:
                    lines.append(f"\n6GB 可跑性实测（`{probe_file.relative_to(ROOT)}`）：策略 "
                                 f"{probe.get('mode')}，单图 "
                                 f"{fastest.get('seconds_per_image')}s，峰值显存 "
                                 f"{fastest.get('peak_gb')}GB / {probe.get('gpu_total_gb')}GB")
            except (ValueError, KeyError):
                pass
    else:
        lines.append("_尚未运行（需要已训练好的 SDXL adapter）：`tools/compare_sdxl.py`_")

    lines += ["", "## 5. 结论与局限", ""]
    if val_rows and img_rows:
        lines.append("判定口径（对**部署基线 V4**）：val 更低且 CLIP 一致性不明显变差 → 视为超越；"
                     "只有一维变好则标注为混合；两维都不占优则未超越。")
        lines.append("")
        v4_val = next((float(r["mean_mse"]) for r in val_rows if r["adapter"] == "V4"), None)
        image_mean: dict[str, dict] = {}
        for row in img_rows:
            label = row["adapter"]
            bucket = image_mean.setdefault(label, {"sharpness": [], "clip_score": []})
            bucket["sharpness"].append(float(row["sharpness"]))
            if row.get("clip_score"):
                bucket["clip_score"].append(float(row["clip_score"]))
        v4_clip = None
        if "V4" in image_mean and image_mean["V4"]["clip_score"]:
            v4_clip = sum(image_mean["V4"]["clip_score"]) / len(image_mean["V4"]["clip_score"])
        lines += ["| 模型 | val vs V4 | CLIP vs V4 | 判定 |", "|---|---|---|---|"]
        val_by_label = {row["adapter"]: float(row["mean_mse"]) for row in val_rows}
        for label in sorted(set(val_by_label) | set(image_mean)):
            if label == "V4":
                continue
            val_delta = (val_by_label[label] - v4_val) if (label in val_by_label and v4_val is not None) else None
            clip_delta = None
            if label in image_mean and image_mean[label]["clip_score"] and v4_clip is not None:
                clip_mean = sum(image_mean[label]["clip_score"]) / len(image_mean[label]["clip_score"])
                clip_delta = clip_mean - v4_clip
            if val_delta is None:
                verdict = "只有出图数据（val 缺）"
            elif val_delta < -1e-5 and (clip_delta is None or clip_delta >= -0.002):
                verdict = "**超越基线**（val 更低，CLIP 不明显变差）"
            elif val_delta < -1e-5:
                verdict = "混合：拟合更好，但 CLIP 明显更低"
            elif clip_delta is not None and clip_delta > 0.002:
                verdict = "混合：CLIP 更好，但 val 未更低"
            else:
                verdict = "未超越（与 V4 不可区分）"
            lines.append(f"| {label} | "
                         f"{'—' if val_delta is None else f'{val_delta:+.5f}'} | "
                         f"{'—' if clip_delta is None else f'{clip_delta:+.4f}'} | {verdict} |")
    else:
        lines.append("_评测尚未跑全，暂无自动判定。_")
    if sdxl_rows:
        anchor = next((r for r in sdxl_rows if r.get("adapter") in ("SDXL-BASE", "BASE")), None)
        tuned = next((r for r in sdxl_rows if r.get("adapter") not in ("SDXL-BASE", "BASE")), None)
        if anchor and tuned:
            delta = float(tuned["mean_mse"]) - float(anchor["mean_mse"])
            verdict = "下降（该基座上更好）" if delta < 0 else "未下降"
            lines.append(f"SDXL 线判定：`{tuned['adapter']}` 相对 SDXL 基座 val "
                         f"{float(tuned['mean_mse']):.5f} vs {float(anchor['mean_mse']):.5f} "
                         f"（Δ {delta:+.5f}，{verdict}）。"
                         f"注意这是**另一个基座**上的结论，不能与 SD1.5 的 V4 直接比数值。")
    lines += [
        "",
        "- val MSE 衡量对数据集分布的拟合，**不等于画质**；两者必须同时看。",
        "- 第 3 节的 val 刻意统一使用 **base CLIP 文本嵌入**，只比较 UNet 侧适配质量；",
        "  第 4 节走**完整部署形态**（含 `<adapter>_text_encoder.pt`，若存在），代表线上真实观感。",
        "  文本编码器没被微调过的 adapter（例如 V4 的 TE 实测与 base 逐位相同）在两节里没有差别。",
        "- 测试集仅 144 张，均值已固定协议降噪，但结论仍应结合人工查看 `compare_sheet.png`。",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": Python 文本模式在 Windows 上会把 \n 翻成 CRLF，而仓库门禁要求 LF
    with open(REPORT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    print(f"\nreport: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
