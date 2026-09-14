"""train_driver.py — run a training config to completion on a *shared* GPU.

A 6 GB laptop card is usually also driving a browser, an LLM server and whatever
else the owner is doing.  Instead of one long fragile run, this driver:

  * launches train_v5.py (fresh, or --resume latest) and waits;
  * if the trainer gives up because another process ate the VRAM, waits for the
    card to cool down (polling free memory) and resumes from the last checkpoint;
  * **看门狗**：监控 `<out_dir>/metrics.csv` 心跳，长时间无更新即判定"卡死"并重启。
    WDDM 显存超配时进程既不退出也不报错（2026-09-13 V5b：180 秒 0 步推进、GPU 100%、
    CPU 空转、显存 5.0/6.0GB），没有看门狗就只能干等；
  * keeps a driver log so a multi-hour, multi-restart session is still readable.

Usage:
    python tools/train_driver.py --config config_v5.cfg [--attempts 12] [--min-free 2.0]
    python tools/train_driver.py --script train_v5.py --config config_v5b.cfg --stall-seconds 300
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


def resolve_watch(config_path: str) -> Path | None:
    """训练脚本每步都会写 `<out_dir>/metrics.csv`；用它当"还活着"的心跳。

    卡死判据必须落在**文件时间戳**上，因为 WDDM 超配时进程既没退出也不报错：
    2026-09-13 的 V5b 就是 180 秒 0 步推进、GPU 100%、CPU 空转，而 driver 只会傻等。
    """
    if not config_path:
        return None
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("out_dir"):
            _, _, value = line.partition("=")
            out_dir = Path(value.strip().strip('"').strip("'"))
            if not out_dir.is_absolute():
                out_dir = ROOT / out_dir
            return out_dir / "metrics.csv"
    return None


def monitor(command: list[str], watch: Path | None, stall_seconds: float, say) -> int | str:
    """跑训练并在"心跳文件长时间不更新"时终止它。返回退出码或 "stalled"。"""
    process = subprocess.Popen(command, cwd=str(ROOT))
    if watch is None:
        return process.wait()
    if not watch.exists():
        # 还没写出第一个指标：给它一个完整的宽限窗口（模型加载 + 首次量化可能好几分钟）
        say(f"watchdog: 等待心跳文件 {watch}")
    while True:
        try:
            code = process.wait(timeout=30)
            return code
        except subprocess.TimeoutExpired:
            pass
        if not watch.exists():
            continue
        idle = time.time() - watch.stat().st_mtime
        if idle > stall_seconds:
            say(f"watchdog: {watch} 已 {idle:.0f}s 未更新（> {stall_seconds:.0f}s）——判定卡死，终止并重试")
            process.kill()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass
            return "stalled"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="train_v5.py",
                    help="要守护的训练脚本（相对项目根），如 train_sr_gan.py")
    ap.add_argument("--attempts", type=int, default=12)
    ap.add_argument("--min-free", type=float, default=2.0)
    ap.add_argument("--wait-for-gpu", type=float, default=900, help="seconds to wait for free VRAM before each attempt")
    ap.add_argument("--log", default=str(ROOT / "models" / "train_driver.log"))
    ap.add_argument("--watch", default=None, help="心跳文件（默认从 --config 的 out_dir 推导出 metrics.csv）")
    ap.add_argument("--stall-seconds", type=float, default=300.0,
                    help="心跳文件多久不更新判定为卡死并重启；0 表示关闭看门狗")
    args, forwarded = ap.parse_known_args()

    if args.watch is None and args.stall_seconds > 0:
        config_arg = ""
        for index, token in enumerate(forwarded):
            if token == "--config" and index + 1 < len(forwarded):
                config_arg = forwarded[index + 1]
        args.watch = resolve_watch(config_arg)
    watch = Path(args.watch) if args.watch else None

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
        command = [sys.executable, str(ROOT / args.script), *forwarded]
        if resume:
            command += ["--resume", "latest"]
        say(f"attempt {attempt}/{args.attempts}: free {available:.2f} GB | {' '.join(command[1:])}")
        started = time.time()
        result = monitor(command, args.watch, args.stall_seconds, say)
        elapsed = time.time() - started
        if result == 0:
            say(f"training finished cleanly in {elapsed / 60:.1f} min")
            return 0
        if result == "stalled":
            say(f"attempt {attempt}: 训练进程**卡死**（{args.stall_seconds}s 无步进，典型原因是 WDDM "
                f"显存超配）；已终止，准备降配或续训")
        else:
            say(f"attempt {attempt} exited rc={result} after {elapsed / 60:.1f} min; "
                f"resuming from latest checkpoint")
        resume = True
        time.sleep(30)
    say(f"gave up after {args.attempts} attempts — free the GPU and re-run the driver")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
