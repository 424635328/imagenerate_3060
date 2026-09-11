import os
"""plot_loss.py — 绘制 V3 训练/验证损失曲线。

读取某次训练输出的 metrics.csv（列: step,train_loss,val_loss,lr,epoch,elapsed），
绘制 train_loss（含滑动平均）与 val_loss 随 step 变化，保存 loss_curve.png。

用法:
  python plot_loss.py --metrics <ROOT>/models/v3_lora/metrics.csv \
      --out <ROOT>/models/v3_lora/loss_curve.png
"""
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=25, help="训练损失滑动平均窗口")
    args = ap.parse_args()

    steps, tr, va, lr, ep, el = [], [], [], [], [], []
    with open(args.metrics, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row or len(row) < 2: continue
            try:
                s = int(float(row[0]))
            except Exception:
                continue
            steps.append(s)
            tr.append(float(row[1]) if row[1] not in ("",) else None)
            va.append(float(row[2]) if row[2] not in ("",) else None)
            lr.append(float(row[3]) if len(row) > 3 and row[3] not in ("",) else None)
            ep.append(row[4] if len(row) > 4 else "")
            el.append(row[5] if len(row) > 5 else "")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax = axes[0]
    # train loss (drop None)
    ts = [s for s, v in zip(steps, tr) if v is not None]
    tv = [v for v in tr if v is not None]
    ax.plot(ts, tv, alpha=0.35, lw=0.8, color="tab:blue", label="train loss (raw)")
    if len(tv) >= args.smooth:
        import numpy as np
        k = min(args.smooth, len(tv))
        sm = np.convolve(tv, np.ones(k)/k, mode="valid")
        ax.plot(ts[:len(sm)], sm, lw=1.8, color="tab:blue", label=f"train loss (smooth {k})")
    vs_steps = [s for s, v in zip(steps, va) if v is not None]
    vvals = [v for v in va if v is not None]
    if vs_steps:
        ax.plot(vs_steps, vvals, "o-", lw=1.6, color="tab:red", label="val loss")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("V3 training vs validation loss")

    ax2 = axes[1]
    ls = [s for s, v in zip(steps, lr) if v is not None]
    lv = [v for v in lr if v is not None]
    if ls:
        ax2.plot(ls, lv, lw=1.6, color="tab:green")
    ax2.set_ylabel("learning rate"); ax2.set_xlabel("global step")
    ax2.set_yscale("log"); ax2.grid(alpha=0.3); ax2.set_title("LR schedule")

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140)
    print("saved", args.out)

if __name__ == "__main__":
    main()
