"""test_pipeline_logic.py — 单元测试流水线的两个关键纯函数。

为什么值得测：`tools/train_pipeline.py` 会据此决定**几个小时**的 SDXL 训练跑在哪个分辨率、
以及是否需要重建缓存。解析错一个字符就可能：把 1024 误判为不可行（白降画质）、
或在只剩 768 可行时挑了不可行的档位（白跑一夜）。

用法: python tools/test_pipeline_logic.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("LANDSCAPE_ROOT") or Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "tools"))

from train_pipeline import choose_resolution, parse_feasible_modes, probe_failure_kind  # noqa: E402

# 探针真实输出片段（tools/probe_sdxl_train.py 的打印格式）
SAMPLE = """
base=SG161222/RealVisXL_V5.0
  quantized 397 Linear layers to int8
8bit-1024    OK   peak= 5.41GB weights= 2.68GB  9.87s/step loss=0.9123
8bit-896     OK   peak= 5.02GB weights= 2.68GB  7.65s/step loss=0.9310
8bit-768     OK   peak= 4.71GB weights= 2.68GB  6.12s/step loss=0.9288
fp16-768     OOM  peak= 6.00GB
fp16-640     FAIL RuntimeError: CUDA out of memory

=== summary ===
  8bit-1024    feasible     peak= 5.41GB  9.87s/step
  8bit-896     feasible     peak= 5.02GB  7.65s/step
  8bit-768     feasible     peak= 4.71GB  6.12s/step
"""

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


modes = parse_feasible_modes(SAMPLE)
check("只挑出 OK 档位（不含 OOM/FAIL）", modes == ["8bit-1024", "8bit-896", "8bit-768"], str(modes))
check("summary 段落不会造成重复计数", len(modes) == len(set(modes)), str(len(modes)))

check("偏好 1024 且可行 → 选 1024", choose_resolution(modes, [1024, 896, 768]) == 1024)
check("1024 不可行 → 退到 896", choose_resolution(["8bit-896", "8bit-768"], [1024, 896, 768]) == 896)
check("只剩 768 → 选 768", choose_resolution(["8bit-768"], [1024, 896, 768]) == 768)
check("可行档位不在偏好里 → 取最大可行值", choose_resolution(["8bit-832", "8bit-512"], [1024, 896, 768]) == 832)
check("全 OOM → 返回 None（触发流水线停止而非瞎跑）",
      choose_resolution(parse_feasible_modes("8bit-1024    OOM  peak= 6.00GB\n"), [1024, 896, 768]) is None)
check("fp16 兜底档位也能被识别", parse_feasible_modes("fp16-640     OK   peak= 5.90GB") == ["fp16-640"])
check("缩进/多余空格容错", parse_feasible_modes("   8bit-768   OK  peak=1GB") == ["8bit-768"])
check("大小写（ok/OK）都能识别", parse_feasible_modes("8bit-768  ok\n") == ["8bit-768"] and
      parse_feasible_modes("8bit-768  OK\n") == ["8bit-768"])
check("非 OK 状态不应被当成可行", parse_feasible_modes("8bit-768  OOM  peak=6GB\n") == [] and
      parse_feasible_modes("8bit-768  FAIL RuntimeError\n") == [])

# 探针"全部 FAIL 且无 OOM"必须被识别为环境故障（2026-09-13 SDXL 主线被误判中止的根因）
SSL_FAILURES = """
8bit-1024    FAIL ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]
8bit-896     FAIL ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]
8bit-768     FAIL ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]
fp16-768     FAIL ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]
fp16-640     FAIL ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]
"""
check("全部 FAIL 且无 OOM → infrastructure（不是显存不足）",
      probe_failure_kind(SSL_FAILURES) == "infrastructure")
check("FAIL 与 OOM 混合 → mixed", probe_failure_kind("8bit-1024 OOM peak=6GB\n8bit-768 FAIL x\n") == "mixed")
check("只有 OOM → capacity", probe_failure_kind("8bit-1024 OOM peak=6GB\n") == "capacity")
check("探针把 OOM 打成 FAIL 时仍算容量问题（不误报环境故障）",
      probe_failure_kind("fp16-640  FAIL RuntimeError: CUDA out of memory\n") == "capacity")
check("混合样本（3 OK + 1 OOM + 1 OOM-FAIL）→ capacity", probe_failure_kind(SAMPLE) == "capacity")

failed = 0
for name, ok, detail in results:
    if not ok:
        failed += 1
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok or not detail else f'  [got {detail}]'}")
print(f"\n{len(results) - failed}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
