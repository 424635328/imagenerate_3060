"""test_resume_consistency.py — 证明"杀掉再续"与"一口气跑完"得到**同样的权重**。

为什么必须有它：AGENTS.md 的训练脚本规范把"可续训"列为底线，但"能续上"和"续得对"是两件事。
只恢复 model/optimizer 而**不恢复 RNG**，数据顺序与噪声就变了 —— 训练照样"跑得下去"，
loss 曲线看着也正常，但复现性已经破了，而且这种错误在任何日志里都看不出来。

判据（三条，缺一不可）：
  1. **A vs B 一致**：A = 一口气 6 步；B = 跑到 3 步停下，再从检查点续到 6 步。
     正确实现下 B 的第 4-6 步与 A 完全相同（同样的数据、同样的噪声、同样的优化器状态），
     因此 step_6 的 adapter 权重应当逐张量一致（容差内）。
  2. **检查点确实被用上**：B 的第二段日志必须出现 `resumed at step 3`，否则测的是"从头再跑"。
  3. **测试有牙齿**：C = 换一个 seed 跑 6 步，权重必须**明显不同** —— 否则"一致"可能只是因为
     比较方法失灵（例如张量根本没被写进检查点）。

用法（需要 GPU，约 2-3 分钟）: python tools/test_resume_consistency.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
PY = sys.executable
CACHE = "dataset1024/cache_v4.pt"          # 512 缓存：小、快，足以暴露顺序差异
OPTIMIZER = "adamw"                         # 由 --optimizer 覆盖

CONFIG_TEMPLATE = """\
# 由 tools/test_resume_consistency.py 生成的最小配方（不要手工复用）
pretrained_model_name_or_path=SG161222/Realistic_Vision_V6.0_B1_noVAE
arch=sd15
cache={cache}
out_dir={out_dir}
resolution=512
train_batch_size=1
gradient_accumulation_steps=1
lora_rank=8
lora_alpha=8
lora_dropout=0.0
use_dora=false
base_8bit=true
optimizer={optimizer}
learning_rate=0.0001
lr_scheduler=constant
lr_warmup_steps=0
min_snr_gamma=5.0
caption_dropout=0.0
ema_decay=0.999
finetune_text=false
gradient_checkpointing=true
mixed_precision=fp16
max_train_steps={steps}
save_every={save_every}
eval_every=100000
eval_subset=2
keep_checkpoints=5
seed={seed}
cache_dir=models/hf_cache
min_free_gb=1.0
"""


def write_config(path: Path, out_dir: Path, steps: int, save_every: int = 3, seed: int = 42) -> Path:
    path.write_text(CONFIG_TEMPLATE.format(cache=CACHE, out_dir=str(out_dir).replace("\\", "/"),
                                           steps=steps, save_every=save_every, seed=seed,
                                           optimizer=OPTIMIZER),
                    encoding="utf-8", newline="\n")
    return path


def run_trainer(config: Path, extra: list[str]) -> subprocess.CompletedProcess:
    environment = dict(os.environ, LANDSCAPE_ROOT=str(ROOT),
                       HF_HOME=str(ROOT / "models" / "hf_cache"), PYTHONIOENCODING="utf-8")
    return subprocess.run([PY, "-X", "utf8", str(ROOT / "train_v5.py"), "--config", str(config), *extra],
                          cwd=str(ROOT), env=environment, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def adapter_of(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return payload["unet_lora"]


def max_delta(left: dict, right: dict) -> float:
    worst = 0.0
    for key in left:
        if key not in right:
            return float("inf")
        worst = max(worst, float((left[key].float() - right[key].float()).abs().max()))
    return worst


def main() -> int:
    global OPTIMIZER
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adamw8bit", "prodigy"],
                    help="对照用：8-bit/Prodigy 的优化器状态能否精确往返")
    ap.add_argument("--tolerance", type=float, default=1e-6)
    cli = ap.parse_args()
    OPTIMIZER = cli.optimizer
    print(f"优化器 = {OPTIMIZER}（容差 {cli.tolerance:g}）\n")
    passed, failed = 0, []

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed
        if condition:
            passed += 1
            print(f"  [ok]   {name}")
        else:
            failed.append(name)
            print(f"  [FAIL] {name}  {detail}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        print("== A：一口气跑 6 步（参考） ==")
        config_a = write_config(root / "a.cfg", root / "runA", steps=6)
        result_a = run_trainer(config_a, [])
        ckpt_a = root / "runA" / "checkpoints" / "step_6.pt"
        check("A 跑完并落盘 step_6.pt", ckpt_a.exists(),
              (result_a.stderr or result_a.stdout)[-300:])

        print("== B：跑到 3 步 → 从检查点续到 6 步 ==")
        config_b = write_config(root / "b.cfg", root / "runB", steps=3)
        run_trainer(config_b, [])
        ckpt_b3 = root / "runB" / "checkpoints" / "step_3.pt"
        check("B 第一段落盘 step_3.pt", ckpt_b3.exists())
        config_b2 = write_config(root / "b2.cfg", root / "runB", steps=6)
        result_b2 = run_trainer(config_b2, ["--resume", "latest"])
        ckpt_b6 = root / "runB" / "checkpoints" / "step_6.pt"
        check("B 第二段落盘 step_6.pt", ckpt_b6.exists(),
              (result_b2.stderr or result_b2.stdout)[-300:])
        check("B 第二段确实从 step 3 续训（而不是从头开始）",
              "resumed at step 3" in (result_b2.stdout + result_b2.stderr),
              "日志里没有 'resumed at step 3'")

        print("== C：换 seed 跑 6 步（用来证明比较方法有牙齿） ==")
        config_c = write_config(root / "c.cfg", root / "runC", steps=6, seed=1234)
        run_trainer(config_c, [])
        ckpt_c = root / "runC" / "checkpoints" / "step_6.pt"
        check("C 跑完并落盘 step_6.pt", ckpt_c.exists())

        print("== D：同配置**重跑一遍**（对照：进程间可复现性） ==")
        config_d = write_config(root / "d.cfg", root / "runD", steps=6)
        run_trainer(config_d, [])
        ckpt_d = root / "runD" / "checkpoints" / "step_6.pt"
        check("D 跑完并落盘 step_6.pt", ckpt_d.exists())

        if ckpt_a.exists() and ckpt_b6.exists() and ckpt_c.exists() and ckpt_d.exists():
            adapter_a, adapter_b, adapter_c, adapter_d = (adapter_of(ckpt_a), adapter_of(ckpt_b6),
                                                          adapter_of(ckpt_c), adapter_of(ckpt_d))
            check("检查点里确实有 adapter 权重", bool(adapter_a))
            delta_ab = max_delta(adapter_a, adapter_b)
            delta_ac = max_delta(adapter_a, adapter_c)
            delta_ad = max_delta(adapter_a, adapter_d)
            print(f"       A vs B（续训）{delta_ab:.3e} ｜ A vs D（同配置重跑）{delta_ad:.3e} "
                  f"｜ A vs C（换 seed）{delta_ac:.3e}")
            check("A 与 C 明显不同（比较方法有效）", delta_ac > 1e-5,
                  f"差异 {delta_ac:.3e} —— 换 seed 都一样说明比较对象可能没变，测试不可信")
            # 关键判据：续训造成的差异不应**明显大于**同配置重跑造成的差异。
            # 若两者同量级，说明残差来自进程间的浮点不确定性（cuDNN/cuBLAS 选核），
            # 而不是"续训把轨迹搞坏了"；此时应报出这个事实，并给出严格复现的开关。
            check("续训差异不显著大于同配置重跑的差异", delta_ab <= max(delta_ad * 3, cli.tolerance),
                  f"续训 {delta_ab:.3e} vs 重跑 {delta_ad:.3e}")
            check("A 与 B 的权重一致（容差内）", delta_ab < cli.tolerance,
                  f"差异 {delta_ab:.3e}")
            if delta_ad > cli.tolerance:
                print("       [说明] 同配置重跑本身就有差异，说明该环境存在进程间浮点不确定性。"
                      "\n              要严格 bitwise 复现需设 CUBLAS_WORKSPACE_CONFIG=:4096:8 且 "
                      "torch.use_deterministic_algorithms(True)（会变慢）。")

    print(f"\n结果：{passed} 项通过，{len(failed)} 项失败")
    for name in failed:
        print(f"  - {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
