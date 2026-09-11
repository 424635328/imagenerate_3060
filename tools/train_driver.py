"""train_driver.py — run a training config to completion on a *shared* GPU.

A 6 GB laptop card is usually also driving a browser, an LLM server and whatever
else the owner is doing.  Instead of one long fragile run, this driver:

  * launches train_v5.py (fresh, or --resume latest) and waits;
  * if the trainer gives up because another process ate the VRAM, waits for the
    card to cool down (polling free memory) and resumes from the last checkpoint;
  * keeps a driver log so a multi-hour, multi-restart session is still readable.

Usage:
    python tools/train_driver.py --config config_v5.cfg [--attempts 12] [--min-free 2.0]
Any argument that is not a driver option is forwarded to train_v5.py.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])


def free_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.mem_get_info()[0] / 1024 ** 3
    except Exception:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=12)
    ap.add_argument("--min-free", type=float, default=2.0)
    ap.add_argument("--wait-for-gpu", type=float, default=900, help="seconds to wait for free VRAM before each attempt")
    ap.add_argument("--log", default=str(ROOT / "models" / "train_driver.log"))
    args, forwarded = ap.parse_known_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def say(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    resume = False
    for attempt in range(1, args.attempts + 1):
        # wait for the card to have room; other apps come and go
        deadline = time.time() + args.wait_for_gpu
        while free_gb() < args.min_free and time.time() < deadline:
            say(f"waiting for VRAM: {free_gb():.2f} GB free (< {args.min_free})")
            time.sleep(30)
        available = free_gb()
        command = [sys.executable, str(ROOT / "train_v5.py"), *forwarded]
        if resume:
            command += ["--resume", "latest"]
        say(f"attempt {attempt}/{args.attempts}: free {available:.2f} GB | {' '.join(command[1:])}")
        started = time.time()
        result = subprocess.run(command, cwd=str(ROOT))
        elapsed = time.time() - started
        if result.returncode == 0:
            say(f"training finished cleanly in {elapsed / 60:.1f} min")
            return 0
        say(f"attempt {attempt} exited rc={result.returncode} after {elapsed / 60:.1f} min; resuming from latest checkpoint")
        resume = True
        time.sleep(30)
    say(f"gave up after {args.attempts} attempts — free the GPU and re-run the driver")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
