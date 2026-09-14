"""make_version_gallery.py — 把训练实验的评测出图打包成前端可评判的静态画廊。

为什么需要它：四个版本（V4/V5/V5b/V6q）的对比图现在只存在于 `research/`（被 gitignore）
与几份 CSV 里，**人没法看**。而机器判据已经全部走到"无法区分"——包括自研盲测裁判被证明
失效（144/144 次都选"presented A"，见下）。这时候唯一还能推进的判据是**人的眼睛**，
所以要把语料变成一个能逐像素对照、能盲测、能记录选择的页面。

语料为什么成立（`tools/eval_fid.py` 的生成协议，已核对源码）：
  for prompt in HELD_OUT_PROMPTS:        # 24 条**未参与训练**的提示词
      for seed in (101, 108):            # torch.Generator(device).manual_seed(seed)
          同一基座 + 同一 prompt + 同一 seed ⇒ **初始噪声逐位相同**

因此固定 (prompt, seed) 的四张图是**同一构图、同一噪声**，差异只来自权重
（UNet LoRA，以及 V4/V5b 额外微调过的文本编码器）。可以直接叠图擦除式对照。

产出：
  site/img/versions/<候选>/p<NN>_s<SSS>.webp   512×512，WebP
  site/img/versions/sr/<列>_r<行>.webp          超分对比拼版的格子（192×192）
  site/data/versions.json                       版本/提示词/指标/裁判/结论/出处

用法:
  python tools/make_version_gallery.py            # 生成资源与 JSON
  python tools/make_version_gallery.py --check    # 只校验，不写（门禁用）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 候选顺序 = 前端展示顺序（V4 是部署基线，放最左，作为对照基准）
VERSIONS = [
    {
        "id": "V4",
        "label": "V4",
        "slogan": "部署基线",
        "src": "research/fid_v6/V4",
        "adapter": "models/v4_640/adapter_best",
        "recipe": "SD1.5 · 640 训练 · 抗过拟合倍增（当前线上权重）",
        "note": "本项目交付基线；所有候选都以它为准绳。",
    },
    {
        "id": "V5",
        "label": "V5",
        "slogan": "min-SNR-γ + Prodigy",
        "src": "research/fid_v6/V5",
        "adapter": "models/v5_lora/adapter_best",
        "recipe": "640 · r64 · Prodigy · min-SNR-γ=5 · EMA",
        "note": "只改优化器与损失加权（论文路线：arXiv:2303.09556 / 2306.06101）。",
    },
    {
        "id": "V5b",
        "label": "V5b",
        "slogan": "V5 + 微调 CLIP",
        "src": "research/fid_v6/V5b",
        "adapter": "models/v5b_lora/adapter_best",
        "recipe": "V5 续训 · 512 缓存 · 文本编码器 fp32/adamw8bit",
        "note": "唯一变量 = 文本编码器是否真的被微调（V4 的 TE 与基座逐位相同，等于没训）。",
    },
    {
        "id": "V6q",
        "label": "V6q",
        "slogan": "重写 caption",
        "src": "research/fid_v6/V6",
        "adapter": "models/v6_qwen/adapter_best",
        "recipe": "Qwen2-VL-2B 重标注 1649 条结构化 caption 后重训",
        "note": "唯一变量 = caption（配方与 V5 逐项相同）。目录名 V6 是当时的旧命名。",
    },
]

SEEDS = [101, 108]
SR_TILE = 192
SR_COLUMNS = [
    ("hr", "HR（参考）"),
    ("bicubic", "bicubic"),
    ("realesrgan", "RealESRGAN"),
    ("ultrasharp", "UltraSharp"),
    ("ours_ema", "Ours-EMA"),
    ("ours_final", "Ours-final"),
]
# 探针实测值（research/sr_probe_final/，2026-09-14）；该脚本未落 CSV，数值取自文档
SR_METRICS = {
    "source": "docs/TRAINING.md §10.1（tools/probe_sr_model.py 实测，research/sr_probe_final/）",
    "kind": "documented",
    "baseline": "bicubic",
    "rows": [
        {"key": "ours_ema", "label": "Ours-EMA", "detail_ratio": 6.53, "psnr_delta_db": -1.45},
        {"key": "ours_final", "label": "Ours-final", "detail_ratio": 8.69, "psnr_delta_db": -1.56},
        {"key": "realesrgan", "label": "RealESRGAN", "detail_ratio": 7.09, "psnr_delta_db": -1.81},
        {"key": "ultrasharp", "label": "UltraSharp", "detail_ratio": 8.03, "psnr_delta_db": -1.29},
    ],
}

COLUMNS = [
    {"key": "val_mse", "label": "固定协议 val ↓", "digits": 6,
     "means": "同源去噪 MSE（144 样本 × 5 时间步）",
     "caveat": "与训练目标同源，训练越久必然越低；**不是画质指标**"},
    {"key": "kid", "label": "留出集 KID ↓", "digits": 6,
     "means": "与真实照片分布的偏差（无偏 MMD，越小越像真照片）",
     "caveat": "σ≈3.5e-05，差异小于 σ 即在噪声内"},
    {"key": "clip_fid", "label": "CLIP-FID ↓", "digits": 4,
     "means": "CLIP 特征空间的分布距离",
     "caveat": "越大越『不自然』，但对分辨率/风格敏感"},
    {"key": "clip_score", "label": "CLIP 一致性 ↑", "digits": 4,
     "means": "prompt 跟随度（图与文是否对得上）",
     "caveat": "与『好看』无关，只测『听话』"},
    {"key": "sharpness", "label": "细节量 ↑", "digits": 1,
     "means": "Laplacian 方差（越大越锐）",
     "caveat": "过锐/噪点也会拉高，需人眼确认"},
    {"key": "text_encoder", "label": "文本编码器", "digits": 0,
     "means": "该权重是否真的微调过 CLIP",
     "caveat": "V4/V5b 微调过，V5/V6q 用基座 TE"},
]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_sources() -> dict:
    """读取全部元数据源（缺一即报错，不允许静默降级成"少几列"）。"""
    fid_csv = ROOT / "research" / "fid_v6" / "fid_metrics.csv"
    val_csv = ROOT / "research" / "eval_val.csv"
    judge_csv = ROOT / "research" / "judge_final" / "judge_verdicts.csv"
    meta_json = ROOT / "research" / "judge_final" / "judge_meta.json"
    sheet = ROOT / "research" / "sr_probe_final" / "sr_compare_sheet.png"
    missing = [str(p.relative_to(ROOT)) for p in (fid_csv, val_csv, judge_csv, meta_json, sheet)
               if not p.exists()]
    if missing:
        raise SystemExit(f"[错误] 缺少评测产物：{missing}\n"
                         f"       请先跑 tools/eval_fid.py / tools/vlm_judge.py / tools/probe_sr_model.py")

    fid_by_label = {}
    for row in read_csv(fid_csv):
        label = "V6q" if row["adapter"] == "V6" else row["adapter"]
        fid_by_label[label] = {
            "kid": float(row["kid"]),
            "kid_std": float(row["kid_std"]),
            "clip_fid": float(row["clip_fid"]),
            "clip_score": float(row["clip_score"]),
            "sharpness": float(row["sharpness"]),
            "text_encoder": row["te"],
            "images": int(row["images"]),
        }
    val_by_label = {}
    val_rows = read_csv(val_csv)
    for row in val_rows:
        key = {"v4v5b_linear": "v4v5b+linear", "v4v5b_slerp": "v4v5b+slerp"}.get(row["adapter"], row["adapter"])
        val_by_label[key] = {"val_mse": float(row["mean_mse"]), "val_std": float(row["std"]),
                             "samples": int(row["samples"]), "path": row["path"]}

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    return {"fid": fid_by_label, "val": val_by_label, "val_rows": val_rows,
            "judge_rows": read_csv(judge_csv), "meta": meta, "sheet": sheet}


def build_judge(rows: list[dict], meta: dict) -> dict:
    """汇总盲测裁判，并**独立重算**它自己的汇总行（不信任 CSV 里的结论行）。

    关键发现（本项目 2026-09-15 复核）：`vlm_judge.py` 的还原逻辑是对的
    （``(verdict == "A") != swap`` ⇒ 按真实身份计数），但 Qwen2-VL-2B 在这套
    提示格式下 **144/144 次都回答 "A"**，于是"胜率"退化成了随机交换排程里
    未交换行的占比（30/48、23/48、22/48）——不含任何质量信息。所以这个判据
    报 `valid: false`，而不是"接近打平"。
    """
    pairs: dict[str, dict] = {}
    for row in rows:
        if not row["prompt"]:                     # 汇总行
            continue
        pairs.setdefault(row["pair"], []).append(row)

    out = []
    total_rows = a_answers = 0
    for pair, items in pairs.items():
        left, _, right = pair.partition(" vs ")
        wins_left = wins_right = ties = unparsed = 0
        for row in items:
            verdict, swap = row["verdict"], row["swapped"] == "True"
            if verdict == "TIE":
                ties += 1
            elif verdict == "UNPARSED":
                unparsed += 1
            elif (verdict == "A") != swap:
                wins_left += 1
            else:
                wins_right += 1
            total_rows += 1
            a_answers += 1 if verdict == "A" else 0
        decided = wins_left + wins_right
        rate = wins_left / decided if decided else 0.0
        low, high = wilson(wins_left, decided)
        out.append({
            "pair": pair, "left": left, "right": right,
            "decided": decided, "wins_left": wins_left, "wins_right": wins_right,
            "ties": ties, "unparsed": unparsed,
            "rate_left": round(rate, 4), "ci": [round(low, 4), round(high, 4)],
            # 区间含 50% ⇒ 依本项目判据记为"无法区分"
            "indistinguishable": low <= 0.5 <= high,
            "presented_a_wins": sum(1 for row in items if row["verdict"] == "A"),
            "total": len(items),
        })
    out.sort(key=lambda item: item["pair"])
    return {
        "model": meta["judge_model"],
        "swap_seed": meta["swap_seed"],
        "pairs": out,
        "valid": False,
        "position_bias": {"presented_a_wins": a_answers, "total": total_rows,
                          "fraction": round(a_answers / total_rows, 4) if total_rows else 0.0},
        "invalid_reason": (
            f"裁判模型 {a_answers}/{total_rows} 次都选了『先出现的那张』（A 位）。"
            "交换左右后按真实身份还原的胜率恰好等于『未交换行的占比』"
            "（30/48、23/48、22/48），即结果完全由随机交换排程决定，"
            "**不携带任何质量信息**。该判据记为无效，而不是『接近打平』。"
        ),
    }


def wilson(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间与 tools/vlm_judge.py 同一实现（人评判的分页也用同一口径）。"""
    if total <= 0:
        return 0.0, 1.0
    phat = wins / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = z * ((phat * (1 - phat) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def measure_alignment(versions: list[dict]) -> dict:
    """实测"同 prompt 同 seed ⇒ 构图对齐"这个前提，而不是只在页面上断言它。

    做法：把每张图降采样成 8×8 灰度（64 维低频结构），去均值后算皮尔逊相关。
    参照组必须一起算，否则数字没有意义：
      · 跨版本（同 prompt 同 seed）—— 若初始噪声相同，应当很高；
      · 同版本不同 seed —— 噪声不同，应当接近 0；
      · 同版本不同 prompt —— 都是风景，会有一个"基线相关"，跨版本必须明显高于它。
    """
    import numpy as np
    from PIL import Image

    def low(path: Path) -> "np.ndarray":
        with Image.open(path) as image:
            array = np.asarray(image.convert("L").resize((8, 8), Image.BOX), dtype=np.float32).ravel()
        return array - array.mean()

    def corr(a, b) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    vectors = {(v["id"], c["prompt"], c["seed"]): low(ROOT / v["source"] / f"{c['prompt'] * len(SEEDS) + SEEDS.index(c['seed']):04d}.png")
               for v in versions for c in v["cells"]}
    ids = [v["id"] for v in versions]
    cross, cross_seed, cross_prompt = [], [], []
    for prompt in range(24):
        stack = {i: vectors[(i, prompt, SEEDS[0])] for i in ids}
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                cross.append(corr(stack[ids[a]], stack[ids[b]]))
        cross_seed.append(corr(vectors[(ids[0], prompt, SEEDS[0])], vectors[(ids[0], prompt, SEEDS[1])]))
        if prompt + 1 < 24:
            cross_prompt.append(corr(vectors[(ids[0], prompt, SEEDS[0])], vectors[(ids[0], prompt + 1, SEEDS[0])]))
    return {
        "method": "8×8 灰度低频结构的皮尔逊相关（64 维）",
        "cross_version_r": round(float(np.mean(cross)), 3),
        "cross_version_p10": round(float(np.percentile(cross, 10)), 3),
        "different_seed_r": round(float(np.mean(cross_seed)), 3),
        "different_prompt_r": round(float(np.mean(cross_prompt)), 3),
        "n_trials": len(cross),
    }


def build(args: argparse.Namespace) -> dict:
    from PIL import Image                      # 只在真正构建时导入

    src = load_sources()
    prompts = src["meta"]["prompts"]
    if len(prompts) != 24:
        raise SystemExit(f"[错误] 期望 24 条留出提示词，实际 {len(prompts)}")

    versions, problems = [], []
    cells = len(prompts) * len(SEEDS)
    for spec in VERSIONS:
        folder = ROOT / spec["src"]
        files = sorted(folder.glob("*.png"))
        if len(files) != cells:
            problems.append(f"{spec['id']}: 期望 {cells} 张，实际 {len(files)}（{spec['src']}）")
        metrics = src["fid"].get(spec["id"])
        if metrics is None:
            problems.append(f"{spec['id']}: fid_metrics.csv 里没有这一行")
        val = src["val"].get(spec["id"])
        # V6q 的 caption 改了 conditioning，它的 val 与其它行**不可比**，
        # 所以 eval_val.csv 里根本没有它 —— 这里记 null，绝不编一个数出来。
        val_note = None if val else "该候选的 conditioning 与其它候选不同，val 不可比（未纳入同一协议）"
        if problems:
            continue
        items = []
        for index, path in enumerate(files):
            # index = prompt_index * len(SEEDS) + seed_index（已与 judge_final 命名逐字节核对）
            prompt_index, seed_index = divmod(index, len(SEEDS))
            seed = SEEDS[seed_index]
            with Image.open(path) as image:
                if image.size != (512, 512):
                    problems.append(f"{path.name}: 期望 512×512，实际 {image.size}")
                    continue
                name = f"p{prompt_index:02d}_s{seed}.webp"
                target = Path(args.out) / spec["id"] / name
                if not args.check:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image.convert("RGB").save(target, "WEBP", quality=args.quality, method=6)
            items.append({"prompt": prompt_index, "seed": seed,
                          "file": f"img/versions/{spec['id']}/{name}",
                          "sha16": sha16(path)})
        versions.append({**{k: spec[k] for k in ("id", "label", "slogan", "adapter", "recipe", "note")},
                         "source": spec["src"], "cells": items,
                         **{k: metrics[k] for k in ("kid", "kid_std", "clip_fid", "clip_score",
                                                    "sharpness", "text_encoder")},
                         "val_mse": val["val_mse"] if val else None,
                         "val_std": val["val_std"] if val else None,
                         "val_samples": val["samples"] if val else 0,
                         "val_comparable": bool(val),
                         "val_note": val_note})

    # 超分拼版：6 列 × 8 行，每格 192×192（标签烧在图上，前端另配可读标签）
    sr_rows = 0
    if not args.check:
        with Image.open(src["sheet"]) as sheet:
            sr_rows = sheet.height // SR_TILE
            for col, (key, _) in enumerate(SR_COLUMNS):
                target = Path(args.out) / "sr"
                target.mkdir(parents=True, exist_ok=True)
                for row in range(sr_rows):
                    box = (col * SR_TILE, row * SR_TILE, (col + 1) * SR_TILE, (row + 1) * SR_TILE)
                    sheet.crop(box).convert("RGB").save(
                        target / f"{key}_r{row:02d}.webp", "WEBP", quality=88, method=6)

    ranked = sorted(versions, key=lambda v: v["kid"])
    alignment = measure_alignment(versions)
    # 固定协议 val 的完整榜（含融合产物与基座）：给页面当"同一把尺子"的上下文，
    # 但必须同时标注它不是画质指标，否则会误导读者。
    val_table = [
        {"id": row["adapter"], "path": row["path"], "val_mse": float(row["mean_mse"]),
         "val_std": float(row["std"]), "samples": int(row["samples"]),
         "in_gallery": row["adapter"] in {v["id"] for v in versions}}
        for row in sorted((r for r in src["val_rows"] if r.get("mean_mse")),
                          key=lambda r: float(r["mean_mse"]))
    ]
    payload = {
        "schema": 1,
        "built_by": "tools/make_version_gallery.py",
        "protocol": {
            "prompt_set": "留出提示词（未参与任何训练）",
            "prompt_count": len(prompts),
            "seeds": SEEDS,
            "res": src["meta"]["res"], "steps": src["meta"]["steps"], "cfg": src["meta"]["cfg"],
            "aligned": "同一 (prompt, seed) 下初始噪声逐位相同 ⇒ 可逐像素对照",
            "alignment": alignment,
            "sources": ["research/fid_v6/", "research/fid_v6/fid_metrics.csv",
                        "research/eval_val.csv", "research/judge_final/",
                        "research/sr_probe_final/"],
            "note": "全部图片来自同一次生成（research/judge_final/images 与 research/fid_v6 逐字节同源）。",
        },
        "columns": COLUMNS,
        "prompts": prompts,
        "versions": versions,
        "ranking_kid": [v["id"] for v in ranked],
        "val_table": val_table,
        "judge": build_judge(src["judge_rows"], src["meta"]),
        "sr": {"columns": [{"key": k, "label": l} for k, l in SR_COLUMNS], "rows": sr_rows or 8,
               "metrics": SR_METRICS},
        "conclusion": {
            "headline": "没有任何微调版本在有效判据上超过 V4",
            "points": [
                "固定协议 val：V5b 0.185834 < V5 0.185891 < 线性融合 0.185969 < V4 0.186437（差 ~0.3%），"
                "但该指标与训练目标同源、不是画质指标。",
                "留出集 KID：V4 0.000359（最低）< V5b 0.000377 < V6q 0.000396 < V5 0.000404，σ=3.5e-05 "
                "⇒ 差异在噪声内。",
                "CLIP 一致性：V5 0.2726 ≈ V5b 0.2720 ≈ V6q 0.2719 ≈ V4 0.2716（测『听话』，不测『好看』）。",
                "自研盲测裁判：**判据无效**（见 judge.invalid_reason），不能作为任何一方的证据。",
                "唯一被客观证据支持的质量提升来自**自训超分 GAN**（细节量 6.5–8.7× vs bicubic，"
                "与 RealESRGAN/UltraSharp 同档），现已作为三个入口的默认超分模型。",
            ],
            "ask": "机器判据已用尽（且其中一条被证伪），因此把你自己的判断作为最后一条判据——"
                   "建议先用盲测模式，不要提前知道哪张是 V4。",
        },
    }

    if problems:
        raise SystemExit("[错误] 语料不完整：\n  - " + "\n  - ".join(problems))

    if not args.check:
        data = Path(args.data)
        data.parent.mkdir(parents=True, exist_ok=True)
        # 行尾必须是 LF（本项目曾有 CRLF 触发门禁的事故）
        data.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
        total = sum(f.stat().st_size for f in Path(args.out).rglob("*.webp"))
        print(f"候选 {len(versions)} 个 × {cells} 格 = {len(versions) * cells} 张；"
              f"超分格 {len(SR_COLUMNS)}×{sr_rows}")
        print(f"图片总量 {total / 1024 / 1024:.2f} MB → {args.out}")
        print(f"数据 {data} ({data.stat().st_size / 1024:.0f} KB)")
    else:
        print(f"[check] 语料完整：{len(versions)} 个候选 × {cells} 格；JSON 结构可生成")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="打包版本评判画廊（research/ → site/）")
    ap.add_argument("--out", default=str(ROOT / "site" / "img" / "versions"))
    ap.add_argument("--data", default=str(ROOT / "site" / "data" / "versions.json"))
    ap.add_argument("--quality", type=int, default=78)
    ap.add_argument("--check", action="store_true", help="只校验语料，不写任何文件")
    build(ap.parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
