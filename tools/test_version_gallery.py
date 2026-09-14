"""test_version_gallery.py — 版本评判台的门禁（纯 CPU，不需要 GPU／模型）。

改这个页面时最容易犯的三类错误，本测试各钉一条：

1. **语料对不上**：JSON 说 48 格，磁盘上少几张，或者 (prompt, seed) 有重复/缺漏
   —— 于是某个版本在某一题上悄悄换成了别的题，人判出来的结论是错的。
2. **数字被手改**：表格里的 KID/CLIP/val 与 `research/` 里的原始 CSV 不一致（复制粘贴走形）。
3. **把失效判据当证据**：盲测裁判 144/144 次选 A 位，这个"位置偏置"必须一直写在数据里，
   否则页面会退化成"看起来接近打平"。谁把它删掉，谁就得先解释为什么。

另外校验：不泄露个人绝对路径、图片确实是 512×512 WebP、总体积不失控、
`site/versions.html` 引用的静态资源都存在。

用法: python tools/test_version_gallery.py     （退出码 0 = 全部通过）
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DATA = SITE / "data" / "versions.json"
PAGE = SITE / "versions.html"
PAGE_JS = SITE / "js" / "versions.js"
VERSIONS = ["V4", "V5", "V5b", "V6q"]
SIZE_BUDGET_MB = 12.0
# 个人绝对路径 / 凭据前缀（AGENTS.md 第 1、4 条）
LEAK = re.compile(r"[A-Za-z]:\\\\?Users|/home/[a-z]|F:\\\\|nfp_[A-Za-z0-9]{8,}|API_TOKEN\s*=")

PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  [ok]   {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}  {detail}")


def wilson_from_judge():
    """从 tools/vlm_judge.py **原文抽取** wilson()，保证口径只有一处定义。"""
    source = (ROOT / "tools" / "vlm_judge.py").read_text(encoding="utf-8")
    match = re.search(r"^def wilson\(.*?\n(?=^def |\Z)", source, re.S | re.M)
    if not match:
        raise SystemExit("[错误] 在 tools/vlm_judge.py 里找不到 wilson()")
    namespace: dict = {"math": __import__("math")}
    exec(match.group(0), namespace)          # noqa: S102 —— 只跑本项目自己的纯函数
    return namespace["wilson"]


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    if not DATA.exists():
        print(f"[FAIL] 缺少 {DATA.relative_to(ROOT)}（先跑 tools/make_version_gallery.py）")
        return 1
    raw = DATA.read_text(encoding="utf-8")
    data = json.loads(raw)
    wilson = wilson_from_judge()

    print("结构")
    check("schema = 1", data.get("schema") == 1)
    check("候选顺序 V4/V5/V5b/V6q", [v["id"] for v in data["versions"]] == VERSIONS)
    check("标注了生成协议（24 题 × 2 seed × 512px/24 步/CFG7.5）",
          data["protocol"]["prompt_count"] == 24 and data["protocol"]["seeds"] == [101, 108]
          and data["protocol"]["res"] == 512 and data["protocol"]["steps"] == 24
          and data["protocol"]["cfg"] == 7.5)
    print("\n构图对齐（数据里的 r 值必须能复算，且必须显著高于基线）")
    check("说明了「初始噪声逐位相同 ⇒ 可逐像素对照」", "逐位相同" in data["protocol"]["aligned"])
    align = data["protocol"].get("alignment")
    check("记录了实测相关（含方法、样本数与三组对照）",
          bool(align) and set(align) >= {"method", "cross_version_r", "different_seed_r",
                                         "different_prompt_r", "n_trials"})
    if align:
        import numpy as np
        from PIL import Image

        def low(version_id: str, prompt: int, seed: int):
            folder = "V6" if version_id == "V6q" else version_id
            index = prompt * len(data["protocol"]["seeds"]) + data["protocol"]["seeds"].index(seed)
            path = ROOT / "research" / "fid_v6" / folder / f"{index:04d}.png"
            array = np.asarray(Image.open(path).convert("L").resize((8, 8), Image.BOX), dtype=np.float32).ravel()
            return array - array.mean()

        def corr(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

        cross = [corr(low("V4", p, 101), low("V5b", p, 101)) for p in range(24)]
        seeds = [corr(low("V4", p, 101), low("V4", p, 108)) for p in range(24)]
        mine = float(np.mean(cross))
        check("跨版本相关可复算（|差| < 0.01）", abs(mine - align["cross_version_r"]) < 0.01,
              f"JSON {align['cross_version_r']} vs 复算 {mine:.3f}")
        check("跨版本相关显著高于 0.7（构图确实对齐）", mine > 0.7, f"{mine:.3f}")
        check("不同 seed 的相关接近 0（噪声确实不同）", abs(float(np.mean(seeds))) < 0.3,
              f"{float(np.mean(seeds)):.3f}")
        check("跨版本 > 不同 prompt 的基线（排除『风景图都差不多』这个解释）",
              mine > align["different_prompt_r"] + 0.2,
              f"{mine:.3f} vs {align['different_prompt_r']}")
    check("结论首句仍是「没有超过 V4」", "没有" in data["conclusion"]["headline"]
          and "V4" in data["conclusion"]["headline"])
    check("提示词 24 条且无重复", len(data["prompts"]) == 24
          and len(set(data["prompts"])) == 24)

    print("\n语料完整性（每格都要在磁盘上，且 (prompt, seed) 不重不漏）")
    expected = {(p, s) for p in range(24) for s in data["protocol"]["seeds"]}
    total_bytes = 0
    for version in data["versions"]:
        cells = version["cells"]
        keys = {(c["prompt"], c["seed"]) for c in cells}
        check(f"{version['id']}: 48 格且 (prompt, seed) 不重不漏",
              len(cells) == 48 and keys == expected,
              f"实际 {len(cells)} 格，缺 {sorted(expected - keys)[:3]} 多 {sorted(keys - expected)[:3]}")
        missing = [c["file"] for c in cells if not (SITE / c["file"]).exists()]
        check(f"{version['id']}: 48 个 WebP 全部存在", not missing, f"缺 {missing[:3]}")
        check(f"{version['id']}: 文件名为 p<NN>_s<SSS>.webp 且与格位一致",
              all(c["file"].endswith(f"p{c['prompt']:02d}_s{c['seed']}.webp") for c in cells))
        total_bytes += sum((SITE / c["file"]).stat().st_size for c in cells if (SITE / c["file"]).exists())

    from PIL import Image
    sample = data["versions"][0]["cells"]
    sizes = {Image.open(SITE / c["file"]).size for c in sample}
    check("图片统一 512×512（逐像素对照的前提）", sizes == {(512, 512)}, str(sizes))
    check("图片格式为 WebP",
          all(Image.open(SITE / c["file"]).format == "WEBP" for c in sample[:6]))

    print("\n数字与原始 CSV 必须一致（防手改表格）")
    fid = {row["adapter"]: row for row in csv_rows(ROOT / "research" / "fid_v6" / "fid_metrics.csv")}
    val = {row["adapter"]: row for row in csv_rows(ROOT / "research" / "eval_val.csv")}
    for version in data["versions"]:
        source = fid.get("V6" if version["id"] == "V6q" else version["id"])
        if source is None:
            check(f"{version['id']}: fid_metrics.csv 有对应行", False)
            continue
        ok = (abs(version["kid"] - float(source["kid"])) < 1e-12
              and abs(version["clip_score"] - float(source["clip_score"])) < 1e-12
              and abs(version["sharpness"] - float(source["sharpness"])) < 1e-9
              and version["text_encoder"] == source["te"])
        check(f"{version['id']}: KID/CLIP/细节量/TE 与 fid_metrics.csv 一致", ok)
        row = val.get(version["id"])
        if row is None:
            check(f"{version['id']}: val 记为「不可比」而不是编造数值",
                  version["val_mse"] is None and version["val_comparable"] is False
                  and bool(version["val_note"]))
        else:
            check(f"{version['id']}: val 与 eval_val.csv 一致",
                  abs(version["val_mse"] - float(row["mean_mse"])) < 1e-12
                  and version["val_comparable"] is True)

    print("\n盲测裁判：无效判定必须一直写在数据里")
    judge = data["judge"]
    rows = [row for row in csv_rows(ROOT / "research" / "judge_final" / "judge_verdicts.csv") if row["prompt"]]
    a_answers = sum(1 for row in rows if row["verdict"] == "A")
    check("原始 CSV 里确实是 144/144 都答 A（位置偏置成立）",
          a_answers == len(rows) == 144, f"{a_answers}/{len(rows)}")
    check("position_bias 记录为 100%", judge["position_bias"]["fraction"] == 1.0
          and judge["position_bias"]["total"] == 144)
    check("valid = False 且写了不可用原因", judge["valid"] is False
          and len(judge["invalid_reason"]) > 30)
    for pair in judge["pairs"]:
        group = [row for row in rows if row["pair"] == pair["pair"]]
        wins_left = sum(1 for row in group
                        if (row["verdict"] == "A") != (row["swapped"] == "True"))
        wins_right = len(group) - wins_left
        low, high = wilson(wins_left, wins_left + wins_right)
        ok = (pair["wins_left"] == wins_left and pair["wins_right"] == wins_right
              and abs(pair["ci"][0] - low) < 1e-3 and abs(pair["ci"][1] - high) < 1e-3)
        check(f"{pair['pair']}: 胜场与 Wilson 区间可独立复算（{wins_left}/{wins_right}）", ok,
              f"JSON {pair['wins_left']}/{pair['wins_right']} CI{pair['ci']} vs 复算 {wins_left}/{wins_right} CI[{low:.4f}, {high:.4f}]")
        check(f"{pair['pair']}: 区间含 50% ⇒ 记为无法区分",
              pair["indistinguishable"] is (low <= 0.5 <= high))

    print("\n判据口径说明（防止把 val 当画质）")
    columns = {c["key"]: c for c in data["columns"]}
    check("val 列标注了「不是画质指标」", "不是画质指标" in columns["val_mse"]["caveat"])
    check("KID 列标注了 σ 的量级", "σ" in columns["kid"]["caveat"] or "sigma" in columns["kid"]["caveat"])
    check("CLIP 列标注了「不测好看」", "好看" in columns["clip_score"]["caveat"])
    check("每列都有口径与陷阱两段说明",
          all(c.get("means") and c.get("caveat") for c in data["columns"]))

    print("\n超分（唯一被证据支持的提升）")
    sr = data["sr"]
    check("6 列 × 8 行", len(sr["columns"]) == 6 and sr["rows"] == 8)
    tiles = [SITE / "img" / "versions" / "sr" / f"{c['key']}_r{row:02d}.webp"
             for c in sr["columns"] for row in range(sr["rows"])]
    check("48 个格子文件全部存在", all(t.exists() for t in tiles),
          f"缺 {[t.name for t in tiles if not t.exists()][:3]}")
    total_bytes += sum(t.stat().st_size for t in tiles if t.exists())
    check("列出了与 bicubic 的对照数字", len(sr["metrics"]["rows"]) == 4
          and all(r["detail_ratio"] > 1 for r in sr["metrics"]["rows"]))
    check("标注了数字出处（脚本未落 CSV，属于文档值）",
          "source" in sr["metrics"] and sr["metrics"]["kind"] == "documented")

    print("\n体积与卫生")
    total_mb = total_bytes / 1024 / 1024
    check(f"WebP 总量 < {SIZE_BUDGET_MB:.0f} MB", total_mb < SIZE_BUDGET_MB, f"实际 {total_mb:.2f} MB")
    print(f"         实际 {total_mb:.2f} MB（{len(data['versions']) * 48 + 48} 张）")
    for path in (DATA, PAGE, PAGE_JS, SITE / "css" / "versions.css"):
        text = path.read_text(encoding="utf-8")
        check(f"{path.name} 无个人绝对路径/凭据", not LEAK.search(text),
              str(LEAK.search(text).group(0)) if LEAK.search(text) else "")

    print("\n页面接线")
    page = PAGE.read_text(encoding="utf-8")
    assets = re.findall(r'(?:href|src)="(/[A-Za-z0-9_./-]+)"', page)
    missing = [a for a in assets if not (SITE / a.lstrip("/")).exists()]
    check("versions.html 引用的静态资源都存在", not missing, f"缺 {missing}")
    check("versions.html 引用了 /data/versions.json 的脚本与页面自身 CSS",
          "/js/versions.js" in page and "/css/versions.css" in page)
    sw = (SITE / "sw.js").read_text(encoding="utf-8")
    for name in ("/versions.html", "/js/versions.js", "/css/versions.css", "/data/versions.json"):
        check(f"sw.js 预缓存 {name}", f"'{name}'" in sw)
    shell = (SITE / "index.html").read_text(encoding="utf-8")
    check("工作台首页有入口链接", 'href="/versions.html"' in shell)
    check("命令面板里有入口命令",
          "versions.html" in (SITE / "js" / "main.js").read_text(encoding="utf-8"))

    print("\n静态画廊（零 JS 保底路径：打不开交互版时必须还能看）")
    gallery = SITE / "gallery.html"
    check("site/gallery.html 存在", gallery.exists())
    if gallery.exists():
        html = gallery.read_text(encoding="utf-8")
        check("零脚本、零外部样式表（自包含）",
              "<script" not in html and "<link" not in html)
        check("图片全部用相对路径（file:// 双击也能看）",
              'src="img/versions' in html and 'src="/img/' not in html)
        check("四版本 × 24 题 × 2 seed = 192 张全在图里",
              html.count("<img") == 192 + 48, f"{html.count('<img')} 个 img")
        check("每张图都有版本与题号 alt",
              all(f'alt="{vid} 第 ' in html for vid in VERSIONS))
        check("带上了判据无效标记", "判据无效" in html)
        check("带上了对齐实测 r 值", f"{data['protocol']['alignment']['cross_version_r']:.2f}" in html)
        check("说明了 val 不是画质指标", "不是画质指标" in html)
        check("超分六列齐全", all(label in html for label in ("HR", "bicubic", "RealESRGAN",
                                                              "UltraSharp", "Ours-EMA", "Ours-final")))
        check("页面里给了交互版的入口", 'href="versions.html"' in html)
        check("体积合理（< 300 KB HTML）", gallery.stat().st_size < 300 * 1024,
              f"{gallery.stat().st_size / 1024:.0f} KB")
    check("交互版顶栏有静态画廊入口（相对链接，本地与线上都能开）",
          'href="gallery.html"' in PAGE.read_text(encoding="utf-8"))
    check("载入失败时给的是可诊断面板（不是转圈）",
          "renderLoadFailure" in PAGE_JS.read_text(encoding="utf-8"))
    sw_text = (SITE / "sw.js").read_text(encoding="utf-8")
    version_match = re.search(r"lsart-shell-v(\d+)", sw_text)
    check("sw.js 缓存版本 >= v7（离线回退按路径修好之后）",
          bool(version_match) and int(version_match.group(1)) >= 7,
          version_match.group(0) if version_match else "找不到版本号")
    check("sw.js 导航回退先找本条路径", "${path}.html" in sw_text)

    print(f"\n结果：{PASSED} 项通过，{len(FAILED)} 项失败")
    for name in FAILED:
        print(f"  - {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
