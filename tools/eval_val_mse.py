"""eval_val_mse.py — 低方差、跨模型可比的验证损失（可计划、可断点续跑）。

为什么需要它：train_v4.py / train_v5.py 的在线验证每次**随机抽 48/144 个样本 + 随机时间步**，
噪声可达 ±0.02——用它判断"V5 是否超过 V4"并不可靠（V5 实测 0.1783 → 0.1648 → 0.1755 的起伏
就包含这份噪声）。本脚本把协议钉死：

  * 测试集：cache 中 test_latents 的固定随机排列前 N 个（默认全部 144）
  * 时间步：固定序列（默认 5 个），每个 (样本, 时间步) 的噪声由固定种子生成
  * 文本条件：统一用 base CLIP 的缓存嵌入 —— 刻意排除"谁额外微调了 CLIP"这一变量，
    只衡量 **UNet 侧的适配质量**；部署配置（含微调 TE）的完整对比用 tools/compare_adapters.py
  * 结果：原始 MSE 的均值 ± 标准差（越小越好），并给出相对基线的差值

工程约束（都是被真实事故逼出来的）：

  * **每算完一个 adapter 立刻写 CSV**（`tools/eval_common.py::save_rows`）。旧版本只在全部
    adapter 跑完后才写文件：2026-09-12 的后台评测跑了 10 分钟被终止，证据一个字节没留下。
  * **`--resume`（默认开）**跳过 CSV 里协议签名完全一致的 adapter，续跑不重算。
  * **`--plan`** 只打印将要执行的工作量与协议签名，不加载模型、不碰 GPU —— 先看代价再决定跑不跑。
  * **行缓冲输出**：重定向到日志文件时也逐行可见，避免"跑没跑起来"只能靠猜。
    仍建议用 `python -u` 或后台作业 + 日志文件。

用法:
    python tools/eval_val_mse.py --plan --adapters "V4:models/v4_640/adapter_best" "BASE:" \\
        --cache dataset1024/cache_v4_640.pt --limit 48 --timesteps 50,450     # 先看代价
    python tools/eval_val_mse.py --adapters "BASE:" "V4:models/v4_640/adapter_best" \\
        "V5:models/v5_lora/adapter_best" --cache dataset1024/cache_v4_640.pt \\
        --csv research/eval_val.csv                                          # 正式评测
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import (canonical_guard, format_eta, format_seconds,  # noqa: E402
                         load_rows, make_row, merge_rows, paired_stats,
                         parse_timesteps, plan_run, protocol_signature, save_rows)

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
BASE = os.environ.get("BASE_MODEL", str(ROOT / "models" / "base_rv6"))
DEFAULT_TIMESTEPS = [50, 250, 450, 650, 850]
SUBSET_SEED = 1234
NOISE_SEED = 4321


def line_buffer_stdout() -> None:
    """重定向到文件时也逐行刷新——否则日志会长时间空白，无法判断是否卡住。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass


def load_cache(path: str):
    try:
        return torch.load(path, map_location="cpu", mmap=True)   # 按需分页，不整份驻留
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu")


def load_unet(base: str, dtype):
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype)
    return unet.requires_grad_(False).eval()


def load_adapter(unet, adapter: str):
    """peft 会把 adapter 挂到已加载的 UNet 上；返回挂载后的模型。"""
    from peft import PeftModel
    return PeftModel.from_pretrained(unet, adapter, is_trainable=False)


@torch.no_grad()
def evaluate(unet, scheduler, cache: dict, device: str, arch: str, limit: int,
             timesteps: list[int]) -> tuple[float, float, int, int, list]:
    latents = cache["test_latents"]
    texts = cache["test_texts"]
    pooled = cache.get("test_pooled")
    count = min(limit, len(latents)) if limit else len(latents)
    order = torch.randperm(len(latents), generator=torch.Generator().manual_seed(SUBSET_SEED))[:count].tolist()
    resolution = int(cache.get("res", 640))
    time_ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]],
                            device=device, dtype=torch.float16)
    autocast_dtype = torch.float16 if device == "cuda" else torch.float32
    losses = []
    for item_index in order:
        latent = latents[item_index].unsqueeze(0).to(device, dtype=autocast_dtype)
        noise_generator = torch.Generator().manual_seed(NOISE_SEED + item_index)
        text = texts[item_index].unsqueeze(0).to(device, dtype=autocast_dtype)
        for timestep in timesteps:
            noise = torch.randn(latent.shape, generator=noise_generator).to(device, dtype=autocast_dtype)
            ts = torch.tensor([timestep], device=device)
            noisy = scheduler.add_noise(latent, noise, ts)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                if arch == "sdxl":
                    out = unet(noisy, ts, encoder_hidden_states=text,
                               added_cond_kwargs={
                                   "text_embeds": pooled[item_index].unsqueeze(0).to(device, dtype=autocast_dtype),
                                   "time_ids": time_ids}).sample
                else:
                    out = unet(noisy, ts, encoder_hidden_states=text).sample
            losses.append(F.mse_loss(out.float(), noise.float()).item())
    tensor = torch.tensor(losses)
    # 逐对损失一并返回：配对统计需要"同一 (样本, 时间步)"的原始值，聚合后的均值无法反推
    return float(tensor.mean()), float(tensor.std()), len(losses), count, losses


def main() -> int:
    line_buffer_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True,
                    help='如 "V4:models/v4_640/adapter_best"；用 "BASE:" 表示未挂 adapter 的基座')
    ap.add_argument("--cache", default=str(ROOT / "dataset1024" / "cache_v4_640.pt"))
    ap.add_argument("--limit", type=int, default=0, help="只评估固定排列的前 N 个样本（默认全部 144）")
    ap.add_argument("--timesteps", default="", help="逗号分隔的时间步；默认 50,250,450,650,850")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="cpu 用于验证脚本本身（慢但零显存占用，可与训练并行）")
    ap.add_argument("--csv", default="", help="结果 CSV（供 tools/make_eval_report.py 汇总）")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True,
                    help="跳过 CSV 中协议签名一致的 adapter（默认开）")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="全部重算")
    ap.add_argument("--plan", action="store_true",
                    help="只打印协议签名与工作量（前向次数、复用情况），不加载模型、不用 GPU")
    ap.add_argument("--allow-overwrite-canonical", dest="allow_canonical", action="store_true",
                    help="允许用非正式协议覆盖 research/eval_val.csv（默认拒绝，见 eval_common.canonical_guard）")
    ap.add_argument("--paired", action="store_true",
                    help="输出逐对损失差的配对统计（均值差 ± 标准误、t 值）：判断 5e-4 量级的差异是否真实")
    ap.add_argument("--paired-with", default="",
                    help="指定配对基准标签（默认取第一个有逐对数据的模型）；基准必须在本轮被计算")
    ap.add_argument("--paired-csv", default="",
                    help="配对统计写到哪里（默认 <csv 同目录>/eval_val_paired.csv；无 --csv 时仅打印）")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / "models" / "hf_cache"))
    device = args.device
    timesteps = parse_timesteps(args.timesteps, DEFAULT_TIMESTEPS)
    cache = load_cache(args.cache)
    arch = cache.get("arch", "sd15")
    total = len(cache["test_latents"])
    count = min(args.limit, total) if args.limit else total
    passes = count * len(timesteps)
    guard_error = canonical_guard(args.csv, count, timesteps, device, allow=args.allow_canonical)
    if guard_error:
        print(guard_error)
        return 2
    signature = protocol_signature(args.cache, arch, device, count, timesteps)
    rows = load_rows(args.csv) if args.csv else []
    legacy = {"cache": args.cache, "arch": arch, "device": device,
              "count": count, "timesteps": timesteps}
    plan = plan_run(args.adapters, rows, signature, resume=args.resume, legacy=legacy)

    print(f"cache: {args.cache}  arch={arch}  test={total}  res={cache.get('res')}  device={device}")
    print(f"协议签名: {signature}")
    print(f"工作量:   {count} 样本 x {len(timesteps)} 时间步 = {passes} 次前向/adapter")
    print(f"base:     {BASE}\n")

    if args.plan:
        todo = 0
        print(f"  {'adapter':14s} {'action':12s} {'样本':>5s} {'时间步':>7s} {'前向':>7s}  说明")
        for label, path, action, existing in plan:
            note = ""
            if action == "reuse":
                note = f"已完成 mean={existing.get('mean_mse')} ({existing.get('samples')} 样本, 时间步 {existing.get('timesteps')})"
            elif action == "reuse-legacy":
                note = (f"旧格式行 mean={existing.get('mean_mse')}，可核对项全部一致"
                        f"（cache/arch/timesteps/前向次数），按 cuda 推断复用")
            elif existing is not None:
                note = f"旧行协议不同({existing.get('protocol')})，将覆盖"
            elif path and not Path(path).exists():
                note = f"警告：adapter 路径不存在 {path}"
            todo += 0 if action.startswith("reuse") else 1
            print(f"  {label:14s} {action:12s} {count:5d} {len(timesteps):7d} {passes:7d}  {note}")
        print(f"\n合计需计算 {todo} 个 adapter = {todo * passes} 次前向；"
              f"GPU 实测速率会在第一个 adapter 完成后给出 ETA。")
        if args.csv:
            print(f"写入目标: {args.csv}（每完成一个 adapter 立即落盘，被终止也不丢已完成行）")
        print("去掉 --plan 即开始执行；加 --no-resume 可强制全部重算。")
        return 0

    from diffusers import DDPMScheduler
    scheduler = DDPMScheduler.from_pretrained(BASE, subfolder="scheduler")

    weight_dtype = torch.float16 if device == "cuda" else torch.float32
    baseline = None
    durations: list[float] = []
    updates: list[dict] = []
    results: list[tuple[str, float, str]] = []
    pair_losses: dict[str, list] = {}          # 仅 --paired 时收集
    started = time.time()
    remaining = sum(1 for _, _, action, _ in plan if not action.startswith("reuse"))

    for index, (label, path, action, existing) in enumerate(plan, start=1):
        if action.startswith("reuse"):
            mean = float(existing["mean_mse"])
            delta = "—" if baseline is None else f"{mean - baseline:+.5f}"
            origin = "协议一致" if action == "reuse" else "旧格式行，核对项一致"
            print(f"  [{index}/{len(plan)}] {label:14s} {mean:10.5f}  Δ {delta:>9s}  "
                  f"复用已完成行（{origin}），跳过")
            if baseline is None:
                baseline = mean
            results.append((label, mean, "复用" if action == "reuse" else "复用(旧格式)"))
            continue

        t0 = time.time()
        unet = load_unet(BASE, weight_dtype)
        if path:
            unet = load_adapter(unet, path)
        unet = unet.to(device).eval()
        mean, std, pass_count, used, losses = evaluate(unet, scheduler, cache, device, arch,
                                                       args.limit, timesteps)
        del unet
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elapsed = time.time() - t0
        durations.append(elapsed)
        remaining -= 1
        if args.paired:
            pair_losses[label] = losses

        if baseline is None:
            baseline = mean
        delta = f"{mean - baseline:+.5f}"
        eta = format_eta((sum(durations) / len(durations)) * remaining) if remaining else "—"
        print(f"  [{index}/{len(plan)}] {label:14s} {mean:10.5f} ± {std:.5f}  Δ {delta:>9s}  "
              f"{used} 样本 x {len(timesteps)} 步 = {pass_count} 次前向  用时 {format_seconds(elapsed)}  "
              f"ETA {eta}")

        row = make_row(label, path, args.cache, arch, device, signature, mean, std,
                       used, pass_count, timesteps, elapsed)
        updates.append(row)
        if args.csv:
            save_rows(args.csv, merge_rows(load_rows(args.csv), [row]))   # 立即落盘
        results.append((label, mean, "本次计算"))

    if args.csv and updates:
        print(f"\ncsv: {args.csv}（{len(updates)} 行本次写入，总 {len(load_rows(args.csv))} 行）")
    elif args.csv:
        print(f"\ncsv: {args.csv}（全部复用，未改动）")

    if results:
        print(f"\n按协议 {signature} 的结果（越小越好）：")
        for rank, (label, mean, origin) in enumerate(sorted(results, key=lambda item: item[1]), start=1):
            marker = "  <- 最低" if rank == 1 else ""
            print(f"  {rank}. {label:14s} {mean:.5f}  [{origin}]{marker}")

    # ---- 配对统计：判断 5e-4 量级的均值差是否真实（非配对口径下标准误约 0.007，判不出来）----
    if args.paired:
        order_labels = [label for label, _, _, _ in plan]
        if args.paired_with:
            reference = args.paired_with if args.paired_with in pair_losses else None
            if reference is None:
                print(f"\n[警告] 基准 {args.paired_with} 本轮没有逐对数据（被复用了？）——"
                      f"用 --no-resume 让它参与计算，或省略 --paired-with")
        else:
            reference = next((label for label in order_labels if label in pair_losses), None)
        if reference is None:
            print("  没有可用的配对基准 —— 跳过配对统计")
        elif len(pair_losses) < 2:
            print(f"\n配对差异（基准 = {reference}）：只有 1 个模型有逐对数据"
                  f"（其余是复用行，没有逐对损失）——用 --no-resume 重跑全部模型才能得到配对统计")
        else:
            print(f"\n配对差异（基准 = {reference}；正数表示该模型损失更低=更好）：")
            print(f"  {'模型':14s} {'Δ 均值':>10s} {'标准误':>9s} {'t 值':>8s} {'配对数':>6s}  判读")
            paired_rows = []
            for label, losses in pair_losses.items():
                if label == reference:
                    continue
                stats = paired_stats(pair_losses[reference], losses)
                if stats is None:
                    print(f"  {label:14s} 配对数不一致，跳过")
                    continue
                verdict = ("显著（|t| > 3）" if abs(stats["t"]) >= 3
                           else ("边缘（1.96 < |t| < 3）" if abs(stats["t"]) > 1.96 else "不显著"))
                print(f"  {label:14s} {stats['mean_delta']:+10.5f} {stats['stderr']:9.5f} "
                      f"{stats['t']:8.2f} {stats['n']:6d}  该模型{'更好' if stats['mean_delta'] > 0 else '更差'}，{verdict}")
                paired_rows.append({"baseline": reference, "adapter": label,
                                    "protocol": signature, "n": stats["n"],
                                    "mean_delta": round(stats["mean_delta"], 6),
                                    "stderr": round(stats["stderr"], 6),
                                    "t": round(stats["t"], 3),
                                    "baseline_mean": round(stats["baseline_mean"], 6),
                                    "other_mean": round(stats["other_mean"], 6)})
            if paired_rows:
                target = args.paired_csv or (str(Path(args.csv).with_name("eval_val_paired.csv"))
                                             if args.csv else "")
                if target:
                    import csv as _csv
                    path = Path(target)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w", newline="", encoding="utf-8") as handle:
                        writer = _csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()),
                                                 lineterminator="\n")
                        writer.writeheader()
                        writer.writerows(paired_rows)
                    print(f"  paired csv: {path}")
            print("  说明：配对=同一 (样本, 时间步) 上两个模型的损失差；样本间方差在相减时抵消，")
            print("        因此这里的标准误比「只看均值」小两个数量级，才是判断小差异的正确口径。")

    print(f"\n总用时 {format_seconds(time.time() - started)}")
    print("说明：同一测试集/时间步/噪声/文本条件下测得，可直接横向比较；")
    print("      均值越低=对数据集分布拟合越好，但不等于画质更高——画质见 tools/compare_adapters.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
