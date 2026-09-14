"""merge_lora.py — 免训练模型融合：把多个 LoRA 合并成一个更好的（Model Soup / TIES / DARE / SLERP）。

为什么做这个：V4 与 V5 是**同一个基座、同一份数据、不同配方**训出的两个适配器，在验证集上打平
（V5 0.18589 vs V4 0.18644）。这类"同一盆地里的多个解"正是权重平均的适用场景：

  * **Model Soups**（Wortsman et al., ICML 2022, arXiv:2203.05482）：直接平均多个微调权重；
  * **TIES-Merging**（Yadav et al., NeurIPS 2023, arXiv:2306.01708）：裁剪小幅度参数 + 符号多数投票消解冲突；
  * **DARE**（Yu et al., 2023, arXiv:2311.03099）：随机丢弃增量并重缩放，再走 TIES；
  * **SLERP**（球面线性插值）：在两个方向之间沿球面插值，保持模长。

⚠️ 2026-09-13 事故（必须记住）：本工具原先调用 PEFT 的 `LoraModel.add_weighted_adapter`，
它在本环境里**静默地产出了坏权重**——`lora_B` 全部保持零初始化（LoRA 的 B 初值就是 0），
于是增量 `B@A = 0`，融合产物在评测里与"未挂 adapter 的基座"逐位相同（linear val 0.19153 = BASE），
差点被写成"打平"。旧门禁只检查"键集合/形状一致"，恰好对这种失败免疫。
现在改为**自己算 LoRA 算术**：每个目标模块的增量取 `delta = (alpha/r) * (B @ A)`，
在增量空间做加权/TIES/DARE，再用 SVD 重压缩回指定秩；每一步都有能量校验，
产物必须通过 `tools/verify_merge.py` 才算有效。

用法:
    python tools/merge_lora.py --adapters models/v4_640/adapter_best models/v5_lora/adapter_best \
        --methods linear slerp ties dare_ties --weights 0.5 0.5 --out models/merged
    python tools/verify_merge.py --sources models/v4_640/adapter_best models/v5_lora/adapter_best \
        --merged models/merged --weights 0.5 0.5          # 门禁：必须全 PASS

产物:
    models/merged/<method>_<w0>_<w1>/{adapter_model.safetensors,adapter_config.json}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])

A_SUFFIX = ".lora_A.weight"
B_SUFFIX = ".lora_B.weight"


def resolve(path: str) -> Path:
    """相对路径一律相对**项目根**解析（不要相对当前工作目录，否则 save/relative_to 会错位）。"""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def short(path: Path) -> str:
    """打印用路径：可能不在项目根下（例如测试传入临时目录），此时返回绝对路径而不是抛异常。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------- 读取与拆解

def load_source(directory: Path) -> tuple[dict, dict]:
    from safetensors.torch import load_file
    state = load_file(str(directory / "adapter_model.safetensors"))
    config_path = directory / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return state, config


def source_scale(config: dict) -> float:
    """PEFT 前向里的缩放系数：alpha / r（写错会让适配器整体失效，所以显式取出来）。"""
    rank = float(config.get("r") or 0)
    alpha = float(config.get("lora_alpha") or 0)
    if rank <= 0:
        raise ValueError("adapter_config.json 缺少有效的 r")
    return alpha / rank


def module_pairs(state: dict) -> dict[str, dict]:
    """把 state_dict 拆成 模块名 -> {'A': r×in, 'B': out×r}（一律 fp32 计算）。"""
    modules: dict[str, dict] = {}
    for key, tensor in state.items():
        if key.endswith(A_SUFFIX):
            modules.setdefault(key[: -len(A_SUFFIX)], {})["A"] = tensor.float()
        elif key.endswith(B_SUFFIX):
            modules.setdefault(key[: -len(B_SUFFIX)], {})["B"] = tensor.float()
    return {name: parts for name, parts in modules.items() if "A" in parts and "B" in parts}


def module_deltas(pairs: dict[str, dict], scale: float) -> dict[str, torch.Tensor]:
    """增量 delta = scale * (B @ A)，形状 out×in —— 融合真正应该作用的数学对象。"""
    return {name: scale * (parts["B"] @ parts["A"]) for name, parts in pairs.items()}


# ----------------------------------------------------------------- 融合算子

def ties_merge(deltas: list[torch.Tensor], weights: list[float], density: float,
               method: str) -> torch.Tensor:
    """TIES：逐增量裁剪到前 density 比例幅度 -> 符号多数投票 -> 只保留同号项后加权求和。"""
    trimmed = []
    for delta in deltas:
        flat = delta.flatten()
        keep = max(1, int(flat.numel() * density))
        threshold = flat.abs().kthvalue(max(1, flat.numel() - keep + 1)).values
        trimmed.append(torch.where(delta.abs() >= threshold, delta, torch.zeros_like(delta)))
    stacked = torch.stack(trimmed)
    weight_tensor = torch.tensor(weights, dtype=stacked.dtype).view(-1, *([1] * trimmed[0].dim()))
    if method == "frequency":
        votes = torch.sign(stacked)
        score = (votes * weight_tensor).sum(dim=0)
    else:                                   # total：按加权幅度决定符号
        score = (stacked * weight_tensor).sum(dim=0)
    elected = torch.sign(score)
    agreed = torch.where(torch.sign(stacked) == elected.unsqueeze(0), stacked,
                         torch.zeros_like(stacked))
    return (agreed * weight_tensor).sum(dim=0)


def dare_ties(deltas: list[torch.Tensor], weights: list[float], density: float,
              method: str, seed: int) -> tuple[list[torch.Tensor], dict]:
    """DARE：以 (1-density) 概率随机置零并重缩放 1/density，再交给 TIES。"""
    kept_probability = density
    dropped: list[torch.Tensor] = []
    stats = {"drop_rate_requested": round(1 - kept_probability, 4), "drop_rate_actual": []}
    for index, delta in enumerate(deltas):
        generator = torch.Generator().manual_seed(seed + index)
        mask = (torch.rand(delta.shape, generator=generator) < kept_probability).to(delta.dtype)
        stats["drop_rate_actual"].append(round(float(1 - mask.mean()), 4))
        dropped.append(delta * mask / kept_probability)
    return dropped, stats


# ------------------------------------------------------------ SVD 重压缩导出

def recompress(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """把 out×in 的增量重压缩成秩 rank 的 LoRA 因子 (A, B)：B @ A ≈ delta。

    返回 (A = r×in, B = out×r, 保留能量比例)。用 sqrt(S) 两侧对分，避免数值尺度失衡。
    """
    out_features, in_features = delta.shape
    rank = max(1, min(rank, out_features, in_features))
    left, singular, right = torch.linalg.svd(delta, full_matrices=False)
    kept = singular[:rank]
    root = kept.clamp(min=0).sqrt()
    b = left[:, :rank] * root.unsqueeze(0)
    a = root.unsqueeze(1) * right[:rank, :]
    total = float((singular ** 2).sum())
    energy = float((kept ** 2).sum()) / total if total > 0 else 1.0
    return a, b, energy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="与 adapters 一一对应的权重（默认均分）")
    ap.add_argument("--methods", nargs="+", default=["linear", "slerp", "ties", "dare_ties"])
    ap.add_argument("--rank", type=int, default=0,
                    help="产物秩（默认沿用源的秩；调大保真度更高、推理更慢）")
    ap.add_argument("--density", type=float, default=0.5, help="TIES/DARE 保留比例")
    ap.add_argument("--majority-sign-method", default="total", choices=["total", "frequency"])
    ap.add_argument("--seed", type=int, default=1234, help="DARE 随机丢弃种子（保证可复现）")
    ap.add_argument("--out", default=str(ROOT / "models" / "merged"))
    args = ap.parse_args()

    adapters = [resolve(p) for p in args.adapters]
    for path in adapters:
        if not (path / "adapter_model.safetensors").exists():
            print(f"missing adapter: {path}")
            return 2
    weights = args.weights or [1.0 / len(adapters)] * len(adapters)
    if len(weights) != len(adapters):
        print(f"weights 数量({len(weights)}) 与 adapters({len(adapters)}) 不一致")
        return 2
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    print("=== 源 adapter ===")
    sources = []
    for path, weight in zip(adapters, weights):
        state, config = load_source(path)
        scale = source_scale(config)
        pairs = module_pairs(state)
        sources.append({"path": path, "config": config, "pairs": pairs, "scale": scale})
        b_zero = sum(1 for parts in pairs.values() if float(parts["B"].abs().max()) < 1e-8)
        magnitudes = [key for key in state if "lora_magnitude_vector" in key]
        print(f"  {short(path)}  权重 {weight:.4f}  模块 {len(pairs)}  "
              f"r={config.get('r')} alpha={config.get('lora_alpha')} 缩放={scale:g}  "
              f"B 全零模块 {b_zero}")
        if magnitudes:
            print(f"  [警告] 该 adapter 带 DoRA 幅度向量（{len(magnitudes)} 个），本工具只融合 "
                  f"lora_A/lora_B，DoRA 部分会被忽略")
        if b_zero == len(pairs):
            print("  [错误] 该 adapter 的 lora_B 全为零 —— 它本身就是 no-op，无法参与融合")
            return 3
    print(f"  权重和 = {sum(weights):.4f}\n")

    names = list(sources[0]["pairs"].keys())
    for source in sources[1:]:
        if list(source["pairs"].keys()) != names:
            print("[错误] 各 adapter 的目标模块集合不一致，无法逐模块融合")
            return 3
    source_rank = int(sources[0]["config"].get("r") or 0)
    rank = args.rank or source_rank
    print(f"目标模块 {len(names)} 个；产物秩 {rank}（源秩 {source_rank}）\n")

    out_root = resolve(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for method in args.methods:
        tag = method + "_" + "_".join(f"{w:.2f}" for w in weights)
        target = out_root / tag
        target.mkdir(parents=True, exist_ok=True)

        if method == "slerp":
            if len(adapters) != 2:
                print(f"  [{method:10s}] 跳过：SLERP 只支持两个 adapter")
                continue
            try:
                from safetensors.torch import load_file, save_file
                state_a = load_file(str(adapters[0] / "adapter_model.safetensors"))
                state_b = load_file(str(adapters[1] / "adapter_model.safetensors"))
                t = weights[1]
                merged = {}
                for key, a in state_a.items():
                    b = state_b[key]
                    flat_a, flat_b = a.float().flatten(), b.float().flatten()
                    norm_a, norm_b = flat_a.norm(), flat_b.norm()
                    if norm_a < 1e-8 or norm_b < 1e-8:
                        out = (1 - t) * flat_a + t * flat_b
                    else:
                        unit_a, unit_b = flat_a / norm_a, flat_b / norm_b
                        cosine = float((unit_a * unit_b).sum().clamp(-1, 1))
                        if cosine > 0.9995:
                            out = (1 - t) * flat_a + t * flat_b
                        else:
                            theta = torch.acos(torch.tensor(cosine))
                            sin_theta = torch.sin(theta)
                            out = (torch.sin((1 - t) * theta) / sin_theta) * flat_a \
                                + (torch.sin(t * theta) / sin_theta) * flat_b
                    merged[key] = out.reshape(a.shape).to(a.dtype)
                save_file(merged, str(target / "adapter_model.safetensors"))
                shutil.copy2(adapters[0] / "adapter_config.json", target / "adapter_config.json")
                zero = sum(1 for key in merged if ".lora_B." in key
                           and float(merged[key].float().abs().max()) < 1e-8)
                print(f"  [{method:10s}] -> {short(target)}  r={source_rank}  "
                      f"B 全零张量 {zero}（必须为 0）")
                summary.append({"method": method, "weights": weights, "rank": source_rank,
                                "dir": short(target), "b_zero_tensors": zero})
            except Exception as error:
                print(f"  [{method:10s}] FAILED: {type(error).__name__}: {str(error)[:140]}")
                summary.append({"method": method, "error": f"{type(error).__name__}: {str(error)[:140]}"})
            continue

        # ---- 增量空间融合（linear / ties / dare_ties）----
        deltas = [module_deltas(source["pairs"], source["scale"]) for source in sources]
        merged_modules = {}
        energies = []
        drop_rates = []
        for module_index, name in enumerate(names):
            per_source = [per_module[name] for per_module in deltas]
            if method == "linear":
                combined = sum(w * delta for w, delta in zip(weights, per_source))
            elif method == "ties":
                combined = ties_merge(per_source, weights, args.density, args.majority_sign_method)
            elif method == "dare_ties":
                # DARE 的随机丢弃逐模块、逐源进行；种子固定（seed + 模块序号）保证可复现
                dropped, stats = dare_ties(per_source, weights, args.density,
                                           args.majority_sign_method, args.seed + module_index)
                drop_rates += stats["drop_rate_actual"]
                combined = ties_merge(dropped, weights, args.density, args.majority_sign_method)
            else:
                print(f"  [{method:10s}] 未知方法，跳过")
                combined = None
            if combined is None:
                break
            a, b, energy = recompress(combined, rank)
            merged_modules[name] = (a, b)
            energies.append(energy)

        if len(merged_modules) != len(names):
            continue

        # ---- 导出为标准 PEFT 目录（评测/部署零改动可用）----
        from safetensors.torch import save_file
        state = {}
        for name, (a, b) in merged_modules.items():
            # 存 fp32：SVD 把幅度分摊到两个因子上，再各自压到 fp16 会白丢精度（磁盘代价可接受）
            state[name + A_SUFFIX] = a.float().contiguous()
            state[name + B_SUFFIX] = b.float().contiguous()
        save_file(state, str(target / "adapter_model.safetensors"))

        config = dict(sources[0]["config"])
        config.update({"r": rank, "lora_alpha": rank,      # 保持 alpha/r = 1 的原始缩放
                       "lora_dropout": float(config.get("lora_dropout") or 0.0),
                       "use_dora": False, "peft_type": "LORA",
                       "inference_mode": True})
        (target / "adapter_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        size_mb = (target / "adapter_model.safetensors").stat().st_size / 1024 ** 2
        b_zero = sum(1 for key in state if B_SUFFIX in key
                     and float(state[key].float().abs().max()) < 1e-8)
        energy_mean = sum(energies) / len(energies)
        print(f"  [{method:10s}] -> {short(target)}  {size_mb:.1f}MB  r={rank}  "
              f"SVD 保留能量 {energy_mean * 100:.2f}%  B 全零张量 {b_zero}（必须为 0）")
        entry = {"method": method, "weights": weights, "rank": rank, "size_mb": round(size_mb, 1),
                 "svd_energy_kept": round(energy_mean, 5), "b_zero_tensors": b_zero,
                 "dir": short(target), "density": args.density}
        if drop_rates:
            entry["dare_drop_rate_mean"] = round(sum(drop_rates) / len(drop_rates), 4)
        summary.append(entry)

    (out_root / "merge_manifest.json").write_text(
        json.dumps({"adapters": [str(a) for a in adapters], "weights": weights, "rank": rank,
                    "density": args.density, "seed": args.seed, "results": summary},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nmanifest: {short(out_root / 'merge_manifest.json')}")
    print("\n下一步（门禁必须全 PASS，否则不要拿去评测）：")
    print("  python tools/verify_merge.py --sources " + " ".join(short(a) for a in adapters)
          + f" --merged {short(out_root)} --weights "
          + " ".join(f"{w:.2f}" for w in weights))
    print("  python tools/eval_val_mse.py --adapters \"V4:models/v4_640/adapter_best\" "
          "\"linear:models/merged/linear_0.50_0.50\" --cache dataset1024/cache_v4_640.pt "
          "--csv research/eval_merged.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
