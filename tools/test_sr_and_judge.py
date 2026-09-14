"""test_sr_and_judge.py — 自训超分接入与自研裁判的纯逻辑门禁（不需要 GPU、不加载模型）。

覆盖两条"加入项目"的改动：
  1. **超分模型注册表与输入校验**：`sr_model` 是来自 HTTP 请求体、又会被用来打开文件的字段，
     必须只接受已注册名字或 `models/sr/` 内的文件名（路径越界一律拒绝）；
     Gradio(`app.py`) / API(`server.py`) / 前端三处必须指向同一注册表。
  2. **自研盲测裁判的统计与解析**：Wilson 区间（小样本胜率必须给区间）与判决解析
     （A/B/TIE/无法解析），以及"区间跨 50% 就是无法区分"这条判读规则。

用法: python tools/test_sr_and_judge.py     （退出码 0 = 全部通过）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

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


def main() -> int:
    from enhance import SR_MODELS, resolve_sr_path, sr_model_names

    print("== 超分模型注册表 ==")
    names = sr_model_names()
    check("自训权重已注册", "ours" in names and "ours_final" in names, str(names))
    check("商用权重仍在", "ultrasharp" in names and "realesrgan" in names)
    for name in names:
        check(f"{name} 指向存在的文件", Path(SR_MODELS[name]).is_file(), SR_MODELS[name])
    check("注册表里的文件都在 models/sr 内",
          all(Path(path).parent == Path(SR_MODELS["ours"]).parent for path in SR_MODELS.values()))

    print("== sr_model 输入校验（路径越界必须拒） ==")
    check("已注册名可用", resolve_sr_path("ours").endswith("landscape_gan_x4_best.pth"))
    check("大小写不敏感", resolve_sr_path("UltraSharp") == SR_MODELS["ultrasharp"])
    check("SR_DIR 内的文件名可用",
          resolve_sr_path("4x-UltraSharp.pth") == SR_MODELS["ultrasharp"])
    for bad in ["../../windows/system32/cmd.exe", "..\\config.py", "/etc/passwd",
                "nope", "", "subdir/model.pth"]:
        try:
            resolve_sr_path(bad)
            check(f"越界/未知输入被拒: {bad!r}", False, "竟然被接受")
        except ValueError:
            check(f"越界/未知输入被拒: {bad!r}", True)

    print("== API 与界面默认值一致 ==")
    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    from server import GenerateReq
    check("API 默认用自训权重", GenerateReq().sr_model == "ours", GenerateReq().sr_model)
    index = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    check("前端下拉包含自训权重", 'value="ours"' in index and 'value="ours_final"' in index)
    check("前端第一个选项就是自训权重", index.index('value="ours"') < index.index('value="ultrasharp"'))
    app_source = (ROOT / "app.py").read_text(encoding="utf-8")
    check("Gradio 入口走同一注册表",
          "enhance.get_sr(" in app_source and "SR_MODEL" in app_source)
    check("Gradio 不再硬编码商用权重路径",
          'SR = str(SR_DIR / "RealESRGAN_x4plus.pth")' not in app_source)

    print("== 自研裁判：Wilson 区间 ==")
    from vlm_judge import parse_verdict, wilson
    low, high = wilson(12, 24)
    check("12/24 区间跨 50%（无法区分）", low < 0.5 < high, f"{low:.3f}-{high:.3f}")
    low, high = wilson(20, 24)
    check("20/24 下界 > 50%（显著更好）", low > 0.5, f"{low:.3f}-{high:.3f}")
    low, high = wilson(4, 24)
    check("4/24 上界 < 50%（显著更差）", high < 0.5, f"{low:.3f}-{high:.3f}")
    check("0 题时区间为全域", wilson(0, 0) == (0.0, 1.0))
    check("全胜时上界为 1", wilson(10, 10)[1] == 1.0)

    print("== 自研裁判：判决解析 ==")
    check("标准格式", parse_verdict("WINNER: A") == "A" and parse_verdict("WINNER: B") == "B")
    check("大小写无关", parse_verdict("winner: b") == "B")
    check("平局", parse_verdict("WINNER: TIE") == "TIE")
    check("自然语言平局", parse_verdict("The two images look the same") == "TIE")
    check("无法解析单列统计（不静默丢弃）", parse_verdict("garbage output") == "UNPARSED")
    check("带解释也能取到判决", parse_verdict("A has more detail.\nWINNER: A") == "A")

    print(f"\n结果：{PASSED} 项通过，{len(FAILED)} 项失败")
    for name in FAILED:
        print(f"  - {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
