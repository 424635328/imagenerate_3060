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


def probe(command: list[str], log_name: str) -> list[str]:
    """Run the VRAM probe and return the modes that reported OK."""
    log_path = ROOT / "models" / log_name
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n=== probe {time.strftime('%H:%M:%S')} ===\n")
        handle.flush()
        subprocess.run(command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    ok = re.findall(r"^(8bit-\d+|fp16-\d+)\s+OK", text, flags=re.M)
    say(f"probe feasible modes: {ok or 'none'}")
    return ok


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
    modes = probe([
        python, str(ROOT / "tools" / "probe_sdxl_train.py"),
        "--modes", "8bit-1024,8bit-896,8bit-768,fp16-768,fp16-640",
    ], "pipeline_probe.log")
    if not modes:
        say("no feasible SDXL mode — stopping pipeline (V5 remains the deliverable)")
        return 1

    preference = [int(x) for x in args.res_preference.split(",")]
    feasible = sorted((int(m.split("-")[1]) for m in modes), reverse=True)
    resolution = next((r for r in preference if r in feasible), feasible[0])
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
