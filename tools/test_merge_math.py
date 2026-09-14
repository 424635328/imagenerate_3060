"""test_merge_math.py — 融合算术的回归测试（合成张量，纯 CPU，不加载模型、不用 GPU）。

为什么要有它：2026-09-13 的 `models/merged/*` 全是坏产物 —— PEFT 的 `add_weighted_adapter`
在本环境静默失败，`lora_B` 停留在零初始化，导致增量恒为 0，评测值与被测基座逐位相同
（linear/ties = 0.19153 = BASE），差点被当成"融合无用"写进报告。旧门禁只比键集合与形状，
对这种失败完全免疫。本测试把四条不变量钉死：

  1. `delta = (alpha/r) * (B @ A)`，且**零 B 必须被识别为零增量**（历史 bug 的直接复现）；
  2. `linear` 的等效增量等于 `sum(w_i * delta_i)`；
  3. `recompress` 的产物 `B @ A` 逼近目标矩阵，且 B 不为零（SVD 之后仍然有效）；
  4. TIES/DARE 的裁剪与随机丢弃符合论文定义（保留比例、符号一致性、可复现）。

用法: python tools/test_merge_math.py     （退出码 0 = 全部通过）
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_lora import (A_SUFFIX, B_SUFFIX, dare_ties, module_deltas,  # noqa: E402
                        module_pairs, recompress, source_scale, ties_merge)

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


def make_module(name: str, rank: int, in_features: int, out_features: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(rank, in_features, generator=generator) * 0.1
    b = torch.randn(out_features, rank, generator=generator) * 0.1
    return {name + A_SUFFIX: a, name + B_SUFFIX: b}


def pairs_from(state: dict) -> dict:
    return module_pairs(state)


def effective(pairs: dict, scale: float) -> dict:
    return module_deltas(pairs, scale)


def main() -> int:
    name = "base_model.model.down_blocks.0.attentions.0.to_q"
    state_a = make_module(name, rank=4, in_features=8, out_features=6, seed=1)
    state_b = make_module(name, rank=4, in_features=8, out_features=6, seed=2)
    pairs_a, pairs_b = pairs_from(state_a), pairs_from(state_b)

    print("== 增量还原 ==")
    delta_a = effective(pairs_a, 1.0)[name]
    manual = state_a[name + B_SUFFIX].float() @ state_a[name + A_SUFFIX].float()
    check("delta == B @ A", torch.allclose(delta_a, manual, atol=1e-6))
    scaled = effective(pairs_a, 0.5)[name]
    check("缩放系数作用于增量", torch.allclose(scaled, manual * 0.5, atol=1e-6))
    check("alpha/r 解析正确", abs(source_scale({"r": 64, "lora_alpha": 32}) - 0.5) < 1e-9)
    try:
        source_scale({"r": 0, "lora_alpha": 8})
        check("r=0 必须报错", False, "未抛异常")
    except ValueError:
        check("r=0 必须报错", True)

    print("== 历史 bug 复现：零 lora_B 就是 no-op ==")
    broken = dict(state_a)
    broken[name + B_SUFFIX] = torch.zeros_like(broken[name + B_SUFFIX])
    broken_delta = effective(pairs_from(broken), 1.0)[name]
    check("B 全零 -> 增量全零（旧门禁抓不到，数值门禁能抓到）",
          float(broken_delta.abs().max()) == 0.0)
    check("键集合与形状仍然完全一致（所以旧门禁才会漏）",
          set(broken) == set(state_a)
          and all(broken[key].shape == state_a[key].shape for key in state_a))

    print("== linear 融合 ==")
    combined = 0.5 * delta_a + 0.5 * effective(pairs_b, 1.0)[name]
    check("加权和等于手工计算",
          torch.allclose(combined, 0.5 * manual + 0.5 * (state_b[name + B_SUFFIX].float()
                                                          @ state_b[name + A_SUFFIX].float()),
                         atol=1e-6))

    print("== SVD 重压缩 ==")
    a_full, b_full, energy_full = recompress(combined, rank=8)
    rebuilt = b_full @ a_full
    relative = float((rebuilt - combined).norm()) / float(combined.norm())
    check("秩 8（= 源秩之和）时近乎无损", relative < 1e-4, f"相对误差 {relative:.2e}")
    check("能量保留 ~100%", energy_full > 0.9999, f"{energy_full:.6f}")
    a_low, b_low, energy_low = recompress(combined, rank=1)
    check("降秩必然丢失能量（0 < 保留 < 100%）", 0.0 < energy_low < 1.0, f"{energy_low:.4f}")
    check("降秩后 B 不为零（历史 bug 的直接守卫）", float(b_low.abs().max()) > 1e-6)
    low_relative = float(((b_low @ a_low) - combined).norm()) / float(combined.norm())
    check("降秩产物仍是最佳秩 1 逼近（相对误差 = sqrt(1-能量)）",
          abs(low_relative - (1 - energy_low) ** 0.5) < 1e-4,
          f"{low_relative:.4f} vs {(1 - energy_low) ** 0.5:.4f}")

    print("== TIES ==")
    weights = [0.5, 0.5]
    delta_b = effective(pairs_b, 1.0)[name]
    # 语义 1：不裁剪时，输出必须恰好等于"符号一致项才参与"的加权和
    elected = torch.sign(0.5 * delta_a + 0.5 * delta_b)
    manual_ties = ((torch.where(torch.sign(delta_a) == elected, delta_a, torch.zeros_like(delta_a)) * 0.5)
                   + (torch.where(torch.sign(delta_b) == elected, delta_b, torch.zeros_like(delta_b)) * 0.5))
    full = ties_merge([delta_a, delta_b], weights, 1.0, "total")
    check("density=1 时等于符号一致项的加权和", torch.allclose(full, manual_ties, atol=1e-6))
    # 语义 2：裁剪必然减少能量（保留越少，输出越小）
    trimmed = ties_merge([delta_a, delta_b], weights, 0.5, "total")
    check("density=0.5 的能量小于 density=1.0", float(trimmed.norm()) < float(full.norm()),
          f"{float(trimmed.norm()):.4f} vs {float(full.norm()):.4f}")
    check("裁剪后的非零项必然与选出的符号一致",
          bool(torch.all(torch.sign(trimmed[trimmed != 0]) == elected[trimmed != 0])))
    only_first = ties_merge([delta_a, delta_b], [1.0, 0.0], 1.0, "total")
    check("权重 (1,0) 且不裁剪时等于第一个增量（符号自洽）",
          torch.allclose(only_first, delta_a, atol=1e-6))
    frequency = ties_merge([delta_a, delta_b], weights, 1.0, "frequency")
    check("frequency 多数投票可用且非零", float(frequency.abs().max()) > 1e-6)

    print("== DARE ==")
    dropped, stats = dare_ties([delta_a, effective(pairs_b, 1.0)[name]], weights, 0.9, "total", seed=7)
    actual = sum(stats["drop_rate_actual"]) / len(stats["drop_rate_actual"])
    check("丢弃率 ≈ 1-density", abs(actual - 0.1) < 0.02, f"实际 {actual:.4f}")
    dropped_again, _ = dare_ties([delta_a, effective(pairs_b, 1.0)[name]], weights, 0.9, "total", seed=7)
    check("同种子可复现", all(torch.equal(x, y) for x, y in zip(dropped, dropped_again)))
    dropped_other, _ = dare_ties([delta_a, effective(pairs_b, 1.0)[name]], weights, 0.9, "total", seed=8)
    check("不同种子结果不同", not torch.equal(dropped[0], dropped_other[0]))

    print("== 端到端：真实 CLI 产出的 B 必须非零（本次事故的回归测试） ==")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = []
        for index, state in enumerate((state_a, state_b), start=1):
            directory = root / f"src{index}"
            directory.mkdir()
            from safetensors.torch import save_file
            save_file(state, str(directory / "adapter_model.safetensors"))
            (directory / "adapter_config.json").write_text(json.dumps({
                "peft_type": "LORA", "r": 4, "lora_alpha": 4, "lora_dropout": 0.0,
                "target_modules": [name.split(".")[-1]], "use_dora": False,
                "base_model_name_or_path": "synthetic"}), encoding="utf-8")
            sources.append(directory)
        out = root / "merged"
        result = subprocess.run([sys.executable, "-X", "utf8",
                                 str(Path(__file__).resolve().parent / "merge_lora.py"),
                                 "--adapters", *[str(p) for p in sources],
                                 "--methods", "linear", "ties", "dare_ties",
                                 "--weights", "0.5", "0.5", "--rank", "4", "--out", str(out)],
                                capture_output=True, text=True, encoding="utf-8")
        check("CLI 正常退出", result.returncode == 0, result.stderr[-300:])
        for method in ("linear", "ties", "dare_ties"):
            product = out / f"{method}_0.50_0.50"
            report = product / "adapter_model.safetensors"
            if not report.exists():
                check(f"{method} 产物存在", False, "缺少 adapter_model.safetensors")
                continue
            from safetensors.torch import load_file
            merged_state = load_file(str(report))
            b_tensors = [value for key, value in merged_state.items() if key.endswith(B_SUFFIX)]
            nonzero = all(float(value.float().abs().max()) > 1e-6 for value in b_tensors)
            check(f"{method} 的 lora_B 全部非零（回归守卫）", bool(b_tensors) and nonzero)
            config = json.loads((product / "adapter_config.json").read_text(encoding="utf-8"))
            check(f"{method} 的 config 缩放系数非零",
                  float(config["lora_alpha"]) != 0 and int(config["r"]) == 4)
            if method == "linear":
                merged_delta = (merged_state[name + B_SUFFIX].float()
                                @ merged_state[name + A_SUFFIX].float())
                want = 0.5 * delta_a + 0.5 * effective(pairs_b, 1.0)[name]
                err = float((merged_delta - want).norm()) / float(want.norm())
                # 两个秩 4 的增量之和秩最多 8，压到秩 4 必然有损；SVD 是最优逼近，
                # 因此正确的判据是"误差 <= sqrt(1 - 保留能量)"而不是一个拍脑袋的常数。
                _, _, energy = recompress(want, 4)
                bound = (1 - energy) ** 0.5
                check("linear 的等效增量在 SVD 最优界内", err <= bound + 0.01,
                      f"相对误差 {err:.4f} vs 界 {bound:.4f}")

    print(f"\n结果：{PASSED} 项通过，{len(FAILED)} 项失败")
    for name_ in FAILED:
        print(f"  - {name_}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
