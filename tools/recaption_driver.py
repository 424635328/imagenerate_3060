"""recaption_driver.py — 给 VLM 重标注加"心跳看门狗 + 断点续跑"，像 train_driver 守护训练那样。

为什么需要：重标注要在 6GB 共享卡上连跑 1-2 小时，而桌面上的 LLM/embedding 服务会周期性抢占显存。
显存一紧，WDDM 会让进程**静默卡死**（不报错、不退出、0 产出；2026-09-13/14 各踩过一次）。
`recaption.py` 已经做到每 N 张增量落盘，所以"卡死→重启→跳过已完成"的代价只有最多 N 张。

判据：`manifest.json` 的 **mtime**。它会随每 N 张重标注更新一次，
因此"mtime 长时间不动"就是卡死的可靠信号（进程存在与否、GPU 利用率都不可靠）。

用法:
    python tools/recaption_driver.py --manifest dataset1024/manifest.json \
        --out dataset1024/captions_qwen.json --checkpoint-every 25 \
        --stall-seconds 300 --attempts 12
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
PY = sys.executable


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _free_gb() -> float | None:
    """当前可用显存（GB）；不可用时返回 None（不阻塞流程）。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.mem_get_info()[0] / 1024 ** 3
    except Exception:                                        # noqa: BLE001
        return None


def done_count(manifest: Path) -> int:
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return -1
    return sum(1 for item in data if "recaption_model" in item)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "dataset1024" / "manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "dataset1024" / "captions_qwen.json"))
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--max-edge", type=int, default=512,
                    help="送入 VLM 前的最长边（越小越快越省显存：768→512 可省掉约一半视觉 token）")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--min-free-gb", type=float, default=2.5,
                    help="开跑前等待显存回到该阈值（桌面/ollama/llama-server 会周期性抢占）")
    ap.add_argument("--stall-seconds", type=float, default=300.0,
                    help="manifest mtime 多久不更新判定为卡死")
    ap.add_argument("--attempts", type=int, default=12)
    ap.add_argument("--startup-grace", type=float, default=420.0,
                    help="首次心跳前的宽限（模型加载 + 前 N 张）")
    ap.add_argument("--log", default=str(ROOT / "models" / "recaption.log"))
    args = ap.parse_args()

    manifest = Path(args.manifest)
    total = done_count(manifest)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    say(f"目标：把 manifest 里的条目全部重标注（当前已完成 {total}）")

    for attempt in range(1, args.attempts + 1):
        before = done_count(manifest)
        if 0 <= before and before >= len(json.loads(manifest.read_text(encoding="utf-8"))):
            say(f"全部完成（{before} 条），无需再跑")
            return 0

        # 等显存回到门槛：4-bit VLM 在 6GB 共享卡上对余量很敏感，
        # 余量不足时不会报错，而是变成"每张图十几秒"的隐性抖动（实测 4s/张 → 11s/张）。
        deadline = time.time() + 900
        while time.time() < deadline:
            free_gb = _free_gb()
            if free_gb is None or free_gb >= args.min_free_gb:
                break
            say(f"等待显存：当前 {free_gb:.2f} GB < {args.min_free_gb} GB")
            time.sleep(30)

        command = [PY, "-X", "utf8", str(ROOT / "recaption.py"),
                   "--manifest", str(manifest), "--out", args.out,
                   "--checkpoint-every", str(args.checkpoint_every),
                   "--max-edge", str(args.max_edge),
                   "--max-new-tokens", str(args.max_new_tokens)]
        say(f"attempt {attempt}/{args.attempts}：{before} 条已完成，启动重标注"
            f"（max-edge={args.max_edge}, free={_free_gb()}）")
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"\n=== {time.strftime('%H:%M:%S')} attempt {attempt} "
                         f"(done={before}) {' '.join(command)} ===\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=str(ROOT), stdout=handle,
                                       stderr=subprocess.STDOUT)
        started = time.time()
        last_mtime = manifest.stat().st_mtime if manifest.exists() else started
        while True:
            code = process.poll()
            if code is not None:
                after = done_count(manifest)
                if code == 0:
                    say(f"重标注正常结束（{after} 条）用时 {(time.time() - started) / 60:.1f} min")
                    return 0
                say(f"退出码 {code}（{after} 条已完成），准备续跑")
                break
            time.sleep(30)
            if manifest.exists():
                current = manifest.stat().st_mtime
                if current > last_mtime:
                    last_mtime = current
                    say(f"心跳：已完成 {done_count(manifest)} 条")
            grace = args.startup_grace if not manifest.exists() else 0
            idle = time.time() - last_mtime
            limit = args.stall_seconds + grace
            if idle > limit:
                say(f"**卡死**：manifest 已 {idle:.0f}s 未更新（> {limit:.0f}s）——终止并续跑")
                process.kill()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
                break
        time.sleep(20)

    say("达到最大尝试次数仍未完成；下次运行会从未完成的条目继续（增量落盘保证不重复劳动）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
