"""train_pipeline.py — unattended GPU queue: V5 (SD1.5) → SDXL QLoRA.

On a 6 GB card shared with the owner's other apps, a multi-hour chain has to run
itself.  This pipeline:

  1. waits for the running V5 driver to finish (final adapter written);
  2. probes which SDXL resolutions actually fit (int8 base + fp16 LoRA);
  3. builds the SDXL latent/text cache at the best feasible resolution
     (skipped when the cache already exists);
  4. starts the SDXL training driver, which itself auto-resumes across OOM.

Every stage appends to models/pipeline.log, so the whole night is one file.

Usage:
    python tools/train_pipeline.py --v5-out models/v5_lora --sdxl-config config_v5_sdxl.cfg
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
LOG = ROOT / "models" / "pipeline.log"


def say(message: str) -> None:
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def wait_for_v5(out_dir: Path, timeout_h: float = 6.0) -> bool:
    """V5 is done when the final (non-EMA-best) adapter is exported."""
    final_adapter = out_dir / "adapter" / "adapter_model.safetensors"
    metrics = out_dir / "metrics.csv"
    deadline = time.time() + timeout_h * 3600
    last_beat = 0.0
    while time.time() < deadline:
        if final_adapter.exists():
            say(f"V5 finished: {final_adapter}")
            return True
        if time.time() - last_beat > 600:      # heartbeat: an idle log looks like a dead pipeline
            step = "?"
            if metrics.exists():
                rows = metrics.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                if len(rows) > 1:
                    step = rows[-1].split(",")[0]
            say(f"waiting for V5 ... latest step={step}")
            last_beat = time.time()
        time.sleep(60)
    say(f"V5 still not finished after {timeout_h}h — continuing anyway")
    return False


def run(command: list[str], log_name: str) -> int:
    say("run: " + " ".join(command[1:]))
    log_path = ROOT / "models" / log_name
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n=== {time.strftime('%H:%M:%S')} {' '.join(command)} ===\n")
        handle.flush()
        result = subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
    say(f"exit {result.returncode} ({log_name})")
    return result.returncode


def parse_feasible_modes(text: str) -> list[str]:
    """Modes the probe reported as OK, highest resolution first.

    Kept pure (and unit-tested in tools/test_pipeline_logic.py) because the whole
    multi-hour SDXL run hangs off this decision.
    """
    modes = set(re.findall(r"^\s*(8bit-\d+|fp16-\d+)\s+OK\b", text, flags=re.M | re.I))
    return sorted(modes, key=lambda mode: int(mode.split("-")[1]), reverse=True)


def probe_failure_kind(text: str) -> str:
    """区分"探针基础设施故障"与"显存真的不够"——两者都不 OK，但结论完全不同。

    2026-09-13 事故：基座走 HF Hub id 导致联网取文件，代理返回
    `ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]`，五个模式全部 FAIL。
    旧代码把它汇成一句"no feasible SDXL mode"，读起来像"6GB 跑不了 SDXL"，
    于是整条主线被误判中止。判据：有 FAIL 但一个 OOM 都没有 → 先怀疑环境，
    而不是硬件。纯函数，单测覆盖（tools/test_pipeline_logic.py）。

    注意探针把显存不足也常打印成 `FAIL RuntimeError: CUDA out of memory`，
    所以"算作 OOM"要同时认 `OOM` 与 `out of memory`，否则会把正常的能力结论误判成环境故障。
    """
    oom = len(re.findall(r"^\s*(?:8bit-\d+|fp16-\d+)\s+.*(?:OOM|out of memory)",
                         text, flags=re.M | re.I))
    failed = len(re.findall(r"^\s*(?:8bit-\d+|fp16-\d+)\s+FAIL\b(?!.*out of memory)",
                            text, flags=re.M | re.I))
    if failed and not oom:
        return "infrastructure"
    if failed and oom:
        return "mixed"
    return "capacity"


def choose_resolution(modes: list[str], preference: list[int]) -> int | None:
    """First preferred resolution that the probe proved feasible, else the largest."""
    feasible = sorted({int(mode.split("-")[1]) for mode in modes}, reverse=True)
    if not feasible:
        return None
    for resolution in preference:
        if resolution in feasible:
            return resolution
    return feasible[0]


def probe(command: list[str], log_name: str) -> tuple[list[str], str]:
    """Run the VRAM probe; return (feasible modes, failure kind)."""
    log_path = ROOT / "models" / log_name
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n=== probe {time.strftime('%H:%M:%S')} ===\n")
        handle.flush()
        subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    modes = parse_feasible_modes(text)
    say(f"probe feasible modes: {modes or 'none'}")
    return modes, probe_failure_kind(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-out", default=str(ROOT / "models" / "v5_lora"))
    ap.add_argument("--sdxl-config", default=str(ROOT / "config_v5_sdxl.cfg"))
    ap.add_argument("--res-preference", default="1024,896,768")
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--wait-v5-hours", type=float, default=6.0)
    ap.add_argument("--skip-v5-wait", action="store_true")
    args = ap.parse_args()

    python = sys.executable
    if not args.skip_v5_wait:
        wait_for_v5(Path(args.v5_out), args.wait_v5_hours)

    # ---- 2. feasibility probe (int8 base first, fp16 as a long shot) ----
    modes, failure = probe([
        python, str(ROOT / "tools" / "probe_sdxl_train.py"),
        "--modes", "8bit-1024,8bit-896,8bit-768,fp16-768,fp16-640",
    ], "pipeline_probe.log")
    if not modes:
        if failure == "infrastructure":
            say("probe 全部 FAIL 且没有任何 OOM —— 这是**环境故障**（典型原因：基座走 HF Hub id "
                "导致联网取文件失败，见 models/pipeline_probe.log 里的 ConnectError/SSL），"
                "不是显存不足。先修环境（tools/sdxl_base.py 会解析到本地快照），再重跑。")
        else:
            say("no feasible SDXL mode — stopping pipeline (V5 remains the deliverable)")
        return 1

    preference = [int(x) for x in args.res_preference.split(",")]
    resolution = choose_resolution(modes, preference)
    if resolution is None:
        say("probe returned no usable resolution")
        return 1
    say(f"chosen SDXL resolution: {resolution}")

    # ---- 3. cache at that resolution ----
    cache = ROOT / "dataset1024" / f"cache_v5_sdxl{resolution}.pt"
    if not cache.exists():
        run([
            python, str(ROOT / "precompute_v5.py"),
            "--base", os.environ.get("SDXL_BASE", "SG161222/RealVisXL_V5.0"),
            "--arch", "sdxl", "--res", str(resolution), "--crops", str(args.crops),
            "--out", str(cache),
        ], "pipeline_cache.log")
    else:
        say(f"cache already present: {cache}")

    # ---- 3.5 smoke: prove the whole SDXL path on the real cache first ----
    # A 5-8 hour run must not be started on faith: 4 steps + one eval + one export
    # exercise quantization, cached text conditioning, validation, checkpointing
    # and adapter export.  Failure here costs ~3 minutes instead of a whole night.
    smoke_dir = ROOT / "models" / "v5_sdxl_smoke"
    smoke_code = run([
        python, str(ROOT / "train_v5.py"), "--config", args.sdxl_config, "--cache", str(cache),
        "--out_dir", str(smoke_dir), "--max_train_steps", "4", "--save_every", "2",
        "--eval_every", "2", "--eval_subset", "2", "--gradient_accumulation_steps", "1",
        "--keep_checkpoints", "1", "--min_free_gb", "1.5",
    ], "pipeline_sdxl_smoke.log")
    if smoke_code != 0:
        say("SDXL smoke FAILED — not starting the long run (see models/pipeline_sdxl_smoke.log)")
        return 1
    adapter = smoke_dir / "adapter" / "adapter_model.safetensors"
    if not adapter.exists():
        say("SDXL smoke ran but exported no adapter — not starting the long run")
        return 1
    say(f"SDXL smoke passed ({adapter.name} exported) — starting the real run")

    # ---- 4. SDXL training (driver auto-resumes across OOM) ----
    run([
        python, str(ROOT / "tools" / "train_driver.py"),
        "--config", args.sdxl_config, "--cache", str(cache),
        "--save_every", "250", "--min-free", "2.5", "--attempts", "20",
    ], "pipeline_sdxl_train.log")
    say("pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
