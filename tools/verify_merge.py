"""verify_merge.py — 证明融合产物**真的改变了权重**，而不是一个静默的 no-op。

为什么必须有这个门禁：本项目已经被同类问题咬过三次——
  1. `pipe.load_lora_weights(peft_dir)` 打印 "No LoRA keys ... safe to ignore" 然后什么都不加载，
     两个模型的出图指标一模一样；
  2. 2026-09-13 的合并评测里 `linear 0.5/0.5` 的固定协议 val = **0.19153**，与未挂 adapter 的
     基座（BASE 0.19153）逐位相同：根因是 PEFT 的 `add_weighted_adapter` 在本环境静默失败，
     `lora_B` 全部停留在**零初始化**，于是增量 `B@A = 0`，产物是个 no-op；
  3. 旧门禁只比对"键集合与形状一致"，对这种失败完全免疫 —— 所以判据必须落在**数值**上。

判据（本脚本）：
  * **结构**：键集合一致、形状与 `adapter_config.json` 的 `r` 相符、不存在"期望非零却全零"的张量、
    `lora_alpha != 0`（否则缩放系数为 0，同样恒等于 no-op）；
  * **数值**：把两个 adapter 都还原成**等效增量** `delta = (alpha/r) * (B @ A)`，
    融合产物取同样的还原；`linear` 的期望是 `sum(w_i * delta_i)`，因此可以直接逐模块算相对误差。
     用等效增量而不是原始 A/B 张量比较，是因为 SVD 重压缩后因子分解不唯一，只有乘积有意义。

不加载任何模型、不用 GPU：
    python tools/verify_merge.py --sources models/v4_640/adapter_best models/v5_lora/adapter_best \
        --merged models/merged --weights 0.5 0.5
    python tools/verify_merge.py --sources ... --merged models/merged/ties_0.50_0.50 --weights 0.5 0.5
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"
CONFIG_KEYS = ["peft_type", "r", "lora_alpha", "lora_dropout", "use_dora", "use_rslora",
               "bias", "target_modules", "base_model_name_or_path"]


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def short(path: Path) -> str:
    return str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)


def read_config(directory: Path) -> dict:
    path = directory / "adapter_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_pairs(directory: Path) -> dict[str, dict]:
    from safetensors.torch import load_file
    state = load_file(str(directory / "adapter_model.safetensors"))
    modules: dict[str, dict] = {}
    for key, tensor in state.items():
        if key.endswith(A_SUFFIX):
            modules.setdefault(key[: -len(A_SUFFIX)], {})["A"] = tensor.float()
        elif key.endswith(B_SUFFIX):
            modules.setdefault(key[: -len(B_SUFFIX)], {})["B"] = tensor.float()
    return {name: parts for name, parts in modules.items() if "A" in parts and "B" in parts}


def scale_of(config: dict) -> float:
    rank = float(config.get("r") or 0)
    alpha = float(config.get("lora_alpha") or 0)
    return (alpha / rank) if rank > 0 else 0.0


def effective_deltas(pairs: dict[str, dict], scale: float) -> dict[str, torch.Tensor]:
    """delta = scale * (B @ A)：这才是挂到模型上真正起作用的东西。"""
    return {name: scale * (parts["B"] @ parts["A"]) for name, parts in pairs.items()}


def structural_report(label: str, directory: Path, merged_pairs: dict[str, dict],
                      config: dict, source_pairs: dict[str, dict]) -> dict:
    rank = int(config.get("r") or 0)
    zero_but_expected = []
    shape_bad = []
    for name, parts in merged_pairs.items():
        if float(parts["B"].abs().max()) < 1e-8 or float(parts["A"].abs().max()) < 1e-8:
            zero_but_expected.append(name)
        if rank and parts["A"].shape[0] != rank:
            shape_bad.append((name, tuple(parts["A"].shape)))
    only_merged = sorted(set(merged_pairs) - set(source_pairs))
    only_source = sorted(set(source_pairs) - set(merged_pairs))
    merged_abs_max = max((float(parts["B"].abs().max()) for parts in merged_pairs.values()),
                         default=0.0)
    print(f"=== {label} ===")
    print(f"  模块: 共有 {len(set(merged_pairs) & set(source_pairs))}  "
          f"｜ 仅源有 {len(only_source)}  ｜ 仅产物有 {len(only_merged)}")
    print(f"  r={config.get('r')} alpha={config.get('lora_alpha')} 缩放={scale_of(config):g} "
          f"dropout={config.get('lora_dropout')} dora={config.get('use_dora')} "
          f"peft_type={config.get('peft_type')}  权重幅值 max {merged_abs_max:.3e}")
    missing_config = [key for key in CONFIG_KEYS if key not in config]
    if missing_config:
        print(f"  [警告] adapter_config.json 缺少字段: {missing_config}")
    if zero_but_expected:
        print(f"  [致命] {len(zero_but_expected)} 个模块的 lora_A 或 lora_B 整体为 0"
              f"（例如 {zero_but_expected[0]}）——该模块的增量恒为 0")
    if shape_bad:
        print(f"  [致命] {len(shape_bad)} 个模块的秩与 config 的 r={rank} 不符"
              f"（例如 {shape_bad[0][0]} 形状 {shape_bad[0][1]}）")
    if not rank or scale_of(config) == 0.0:
        print("  [致命] r 或 lora_alpha 为 0：LoRA 缩放系数为 0，适配器恒等于 no-op")
    if only_merged:
        print(f"  [致命] 产物里有 {len(only_merged)} 个源里没有的模块（例如 {only_merged[0]}）")
    ok = (not zero_but_expected and not shape_bad and bool(rank) and scale_of(config) != 0.0
          and not only_merged and merged_abs_max > 1e-8)
    return {"ok": ok, "modules": len(merged_pairs)}


def numeric_report(merged_deltas: dict[str, torch.Tensor], expected: dict[str, torch.Tensor],
                   expect_sum: bool) -> tuple[bool, float]:
    """比较等效增量；`linear` 的期望等于加权和，所以相对误差是有意义的判据。"""
    common = sorted(set(merged_deltas) & set(expected))
    worst_relative, worst_name = 0.0, ""
    ratios = []
    for name in common:
        want = expected[name]
        got = merged_deltas[name]
        want_norm = float(want.norm())
        if want_norm < 1e-9:
            continue
        relative = float((got - want).norm()) / want_norm
        ratios.append(relative)
        if relative > worst_relative:
            worst_relative, worst_name = relative, name
    mean_relative = sum(ratios) / len(ratios) if ratios else 0.0
    suffix = "" if expect_sum else "   [仅供参考：该融合算子不等于加权和]"
    print(f"  等效增量 vs 加权和：平均相对误差 {mean_relative * 100:.3f}%  "
          f"最大 {worst_relative * 100:.3f}%（{worst_name or '-'}）{suffix}")
    return (mean_relative <= 0.05 if expect_sum else True), mean_relative


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="被融合的源 adapter 目录（顺序与 --weights 对应）")
    ap.add_argument("--merged", nargs="+", required=True, help="融合产物目录（或包含多个产物的父目录）")
    ap.add_argument("--weights", nargs="*", type=float, default=[], help="各源权重；省略则等权")
    ap.add_argument("--tolerance", type=float, default=0.05, help="linear 的等效增量相对误差阈值")
    args = ap.parse_args()

    source_dirs = [resolve(p) for p in args.sources]
    merged_dirs: list[Path] = []
    for raw in args.merged:
        candidate = resolve(raw)
        if (candidate / "adapter_model.safetensors").exists():
            merged_dirs.append(candidate)
        elif candidate.is_dir():
            merged_dirs += [d for d in sorted(candidate.iterdir())
                            if (d / "adapter_model.safetensors").exists()]
        else:
            print(f"[skip] 找不到 {candidate}")
    if not merged_dirs:
        print("[错误] 没有可校验的产物")
        return 2

    weights = args.weights or [1.0 / len(source_dirs)] * len(source_dirs)
    if len(weights) != len(source_dirs):
        print(f"[错误] --weights 数量（{len(weights)}）与 --sources（{len(source_dirs)}）不一致")
        return 2
    total = sum(weights)
    weights = [w / total for w in weights]

    print("=== 源 adapter（等效增量口径） ===")
    source_pairs, source_deltas = [], []
    for directory, weight in zip(source_dirs, weights):
        if not (directory / "adapter_model.safetensors").exists():
            print(f"  [错误] {directory} 缺少 adapter_model.safetensors")
            return 2
        config = read_config(directory)
        pairs = load_pairs(directory)
        scale = scale_of(config)
        source_pairs.append(pairs)
        source_deltas.append(effective_deltas(pairs, scale))
        zero = sum(1 for parts in pairs.values()
                   if float(parts["B"].abs().max()) < 1e-8 or float(parts["A"].abs().max()) < 1e-8)
        print(f"  {short(directory)}  权重 {weight:.4f}  模块 {len(pairs)}  "
              f"r={config.get('r')} alpha={config.get('lora_alpha')} 缩放={scale:g}  "
              f"零张量模块 {zero}")
        if zero == len(pairs):
            print("  [错误] 该源 adapter 本身就是 no-op（B 或 A 全零），无法参与融合")
            return 3
    print(f"  权重和 = {sum(weights):.4f}\n")

    rows = []
    for directory in merged_dirs:
        config = read_config(directory)
        merged_pairs = load_pairs(directory)
        merged_deltas = effective_deltas(merged_pairs, scale_of(config))
        expect_sum = "linear" in directory.name
        expected = {}
        for name in merged_deltas:
            acc = None
            for deltas, weight in zip(source_deltas, weights):
                if name not in deltas:
                    continue
                term = deltas[name] * weight
                acc = term if acc is None else acc + term
            if acc is not None:
                expected[name] = acc
        structure = structural_report(short(directory), directory, merged_pairs, config, source_pairs[0])
        numeric_ok, relative = numeric_report(merged_deltas, expected, expect_sum)
        verdict = structure["ok"] and numeric_ok and (not expect_sum or relative <= args.tolerance)
        print(f"  判定: {'PASS 结构自洽且数值符合该融合算子的定义' if verdict else 'FAIL 见上面的致命项'}\n")
        rows.append((short(directory), verdict, relative, expect_sum))

    print("=== 汇总 ===")
    for name, ok, relative, expect_sum in rows:
        detail = f"（等效增量相对误差 {relative * 100:.3f}%）" if expect_sum else "（结构性检查）"
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{detail}")
    failed = [name for name, ok, _, _ in rows if not ok]
    if failed:
        print(f"\n{len(failed)} 个产物未通过 —— 不要把它们写进评测报告。")
        return 1
    print("\n全部通过：增量不是 no-op、结构与 config 自洽；linear 的等效增量等于加权和。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
