"""deploy_check.py — 部署前静态检查：线上（app.py / server.py）到底会加载哪份权重。

为什么需要：`app.py` 的部署形态是「UNet LoRA + `<ADAPTER_DIR>_text_encoder.pt`（若存在）」，
而 **TE 文件缺失时它会静默退回基座 CLIP** —— 这正是 V4 的情况：`v4_640/adapter_best_text_encoder.pt`
与 base CLIP 逐位相同（等于从未微调）。所以"权重训好了"与"线上拿到的是不是它"是两件事，
必须在切换 `LORA_ADAPTER` 之前用一条命令看清。

本脚本不加载模型、不用 GPU、不联网，只读文件与配置：
  * 解析 `LORA_ADAPTER`（或 config 默认值）指向的 adapter 目录；
  * 校验 PEFT 必需文件、adapter_config 的秩/目标模块、以及可选的文本编码器文件；
  * 报告"线上实际会加载什么"，并对每个缺失项给出后果（而不是只报"缺文件"）。

用法:
    python tools/deploy_check.py                                    # 检查默认/环境变量指向
    python tools/deploy_check.py --adapter models/v5b_lora/adapter_best
    python tools/deploy_check.py --adapter models/v5b_lora/adapter_best --expect-text-encoder
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.environ.get("LORA_ADAPTER", ""),
                    help="adapter 目录；默认取 LORA_ADAPTER 环境变量，再退回 config.py 的默认值")
    ap.add_argument("--expect-text-encoder", action="store_true",
                    help="要求存在 <adapter>_text_encoder.pt（部署形态必须含微调文本编码器时加）")
    args = ap.parse_args()

    adapter_value = args.adapter
    if not adapter_value:
        try:
            sys.path.insert(0, str(ROOT))
            from config import ADAPTER_DIR                    # noqa: E402
            adapter_value = str(ADAPTER_DIR)
            source = "config.py::ADAPTER_DIR（未设置 LORA_ADAPTER）"
        except Exception as error:                            # noqa: BLE001
            print(f"[错误] 无法确定 adapter：既没有 --adapter/LORA_ADAPTER，也读不到 config.py（{error}）")
            return 2
    else:
        source = "LORA_ADAPTER 环境变量" if os.environ.get("LORA_ADAPTER") else "--adapter 参数"

    adapter = resolve(adapter_value)
    print("=== 部署检查（静态，不加载模型）===")
    print(f"  adapter 来源 : {source}")
    print(f"  adapter 路径 : {adapter}")

    problems: list[str] = []

    if not adapter.is_dir():
        print(f"  [致命] 目录不存在 —— 线上会启动失败（不会静默退回基座）")
        return 2

    weights = adapter / "adapter_model.safetensors"
    config_path = adapter / "adapter_config.json"
    if weights.exists():
        size_mb = weights.stat().st_size / 1024 ** 2
        print(f"  UNet LoRA    : {weights.name}  {size_mb:.1f} MB")
    else:
        problems.append("缺少 adapter_model.safetensors：peft 无法加载（app.py 会直接报错）")
        print("  [致命] 缺少 adapter_model.safetensors")

    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rank = config.get("r")
        alpha = config.get("lora_alpha")
        scale = (float(alpha) / float(rank)) if rank else 0.0
        print(f"  adapter 配置 : r={rank} alpha={alpha} 缩放={scale:g} "
              f"dropout={config.get('lora_dropout')} dora={config.get('use_dora')}")
        print(f"  目标模块     : {config.get('target_modules')}")
        if not rank or scale == 0.0:
            problems.append("r 或 lora_alpha 为 0：LoRA 缩放系数为 0，适配器恒等于 no-op")
        if config.get("use_dora"):
            print("  [注意] DoRA 适配器需要支持 magnitude 向量的加载路径（本项目已实测 diffusers 0.40 可用）")
    else:
        problems.append("缺少 adapter_config.json：无法确定秩与目标模块")
        print("  [致命] 缺少 adapter_config.json")

    text_encoder = Path(str(adapter) + "_text_encoder.pt")
    if text_encoder.exists():
        size_mb = text_encoder.stat().st_size / 1024 ** 2
        print(f"  微调文本编码器: {text_encoder.name}  {size_mb:.0f} MB  → app.py 会加载它")
        print("      ⚠️ 文件存在 ≠ 微调有效：V4 的 TE 文件与 base CLIP **逐位相同**（等于没微调）。"
              "用 tools/test_adapter_load.py 校验 max|delta| > 0 才算数。")
    else:
        message = (f"没有 {text_encoder.name}：app.py 会**静默使用基座 CLIP**"
                   f"（若该 adapter 本应带微调 TE，这就是 V4 踩过的坑）")
        print(f"  {'[致命]' if args.expect_text_encoder else '[注意]'} {message}")
        if args.expect_text_encoder:
            problems.append(message)

    readme = adapter / "README.md"
    if readme.exists():
        first = readme.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[:1]
        if first:
            print(f"  README       : {first[0][:100]}")

    print("\n=== 线上加载结果（app.py::_build_pipe 的实际行为）===")
    print(f"  UNet         : peft 注入 + 加载 {adapter.name}/adapter_model.safetensors")
    print(f"  文本编码器   : {'加载 ' + text_encoder.name if text_encoder.exists() else '基座 CLIP（未微调）'}")
    print(f"  两种模式     : quality 与 fast/LCM 都会应用上述 adapter 与文本编码器")

    if problems:
        print(f"\n{len(problems)} 项问题：")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\n结论：部署形态自洽，可以切换 LORA_ADAPTER 指向该目录。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
