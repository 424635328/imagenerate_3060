"""eval_common.py — 评测协议签名、CSV 暂存与断点续跑的纯逻辑。

为什么单独一个模块：`tools/eval_val_mse.py` 需要"每完成一个 adapter 就落盘"，
这样进程被 kill（用户在跑训练时要求让出 GPU）也不会丢掉已算完的结果——
2026-09-12 那次后台评测跑了 10 分钟、被终止后 CSV 一个字节都没有，就是因为
旧脚本只在**全部** adapter 跑完后才写文件。

刻意**不 import torch**：报告脚本与回归测试（`tools/test_eval_plan.py`）可以直接
复用这些函数，不加载模型、不占显存、不引入数百 MB 的常驻内存。
"""
from __future__ import annotations

import csv
from pathlib import Path

# CSV 列顺序即对外契约：tools/make_eval_report.py 按列名读取，扩列保持向后兼容。
FIELDS = [
    "adapter", "path", "cache", "arch", "device", "protocol",
    "mean_mse", "std", "samples", "passes", "timesteps", "seconds",
]


def parse_timesteps(text, default):
    """解析 `--timesteps "50,450"`；空值回退默认序列，结果升序去重。

    校验放在这里而不是 argparse，是为了让错误信息说清"哪个值不合法"。
    """
    if text is None or str(text).strip() == "":
        return [int(t) for t in default]
    values: list[int] = []
    for chunk in str(text).replace(" ", "").split(","):
        if not chunk:
            continue
        try:
            value = int(chunk)
        except ValueError:
            raise SystemExit(f"--timesteps 需要逗号分隔的整数，收到 {chunk!r}")
        if not 0 <= value < 1000:
            raise SystemExit(f"--timesteps 必须在 0..999 之间（DDPM 训练时间步），收到 {value}")
        if value not in values:
            values.append(value)
    if not values:
        raise SystemExit("--timesteps 解析后为空")
    return sorted(values)


def protocol_signature(cache, arch, device, samples, timesteps) -> str:
    """同一签名 = 可以放进同一张表直接比较；任何一项不同都不可比。

    样本子集是"固定随机排列的前 N 个"，所以 limit=48 是 limit=144 的子集、
    时间步也按子集取用时仍然可比——但**必须**把这两项写进签名，否则报告里会
    混进不同协议的均值。cache 用完整路径（不是文件名）以免同名缓存互相冒充。
    """
    normalized = str(cache).replace("\\", "/")
    steps = ",".join(str(t) for t in timesteps)
    return f"{normalized}|{arch}|{device}|{int(samples)}|{steps}"


def load_rows(csv_path) -> list[dict]:
    """读已有 CSV；文件不存在或列不全都返回可用列表（缺列由 restval 兜底）。"""
    path = Path(csv_path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def save_rows(csv_path, rows) -> None:
    """整表重写并立即落盘（行数是个位数，代价可忽略）；LF 行尾。

    调用方每算完一个 adapter 调用一次——这就是断点续跑的数据来源。
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n",
                                extrasaction="ignore", restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_row(label, path, cache, arch, device, protocol, mean, std,
             samples, passes, timesteps, seconds) -> dict:
    row = {
        "adapter": label, "path": path or "(base)", "cache": str(cache), "arch": arch,
        "device": device, "protocol": protocol, "mean_mse": round(float(mean), 6),
        "std": round(float(std), 6), "samples": int(samples), "passes": int(passes),
        "timesteps": "|".join(str(t) for t in timesteps), "seconds": round(float(seconds), 1),
    }
    return row


def legacy_matches(row, cache, arch, device, count, timesteps) -> bool:
    """旧格式 CSV 行（没有 protocol 列）的**保守**一致性推断。

    2026-09-12 之前的 `eval_val.csv` 由旧脚本写入，没有 protocol/device/passes 列，
    但它的协议其实就是"全部测试样本 × 默认时间步、cuda"。如果不认这些行，续跑会把
    已经花过 GPU 时间的 BASE/V4/V5 全部重算。这里的每一个可核对项都必须一致才认，
    任何缺项/不一致一律返回 False（宁可重算，也不要混协议）。
    """
    if (row.get("protocol") or "").strip():
        return False                                  # 有协议列就按协议判断
    stored_cache = (row.get("cache") or "").replace("\\", "/")
    wanted_cache = str(cache).replace("\\", "/")
    if stored_cache != wanted_cache and Path(stored_cache).name != Path(wanted_cache).name:
        return False
    if (row.get("arch") or "sd15") != arch:
        return False
    if (row.get("device") or "cuda") != device:
        return False
    if (row.get("timesteps") or "").replace("|", ",") != ",".join(str(t) for t in timesteps):
        return False
    passes = row.get("passes") or row.get("samples") or 0     # 旧行把前向次数写在 samples
    try:
        return int(float(passes)) == int(count) * len(timesteps)
    except (TypeError, ValueError):
        return False


def plan_run(specs, rows, signature, resume=True, legacy=None):
    """决定每个 adapter 是 reuse / reuse-legacy 还是 compute。

    返回 `[(label, path, action, existing_row_or_None), ...]`，action ∈ {reuse, reuse-legacy, compute}。
    只有**协议签名完全一致**的行才允许复用；`legacy` 给定时（形如
    `{"cache":…, "arch":…, "device":…, "count":…, "timesteps":…}`）额外允许旧格式行参与复用，
    判断完全交给 `legacy_matches`。协议不同（样本数/时间步/设备变了）一律重算，
    避免把两种协议的均值混在一张表里。
    """
    by_label: dict[str, dict] = {}
    for row in rows:
        label = (row.get("adapter") or "").strip()
        if label:
            by_label[label] = row          # 同名只保留最后一行
    plan = []
    for spec in specs:
        label, _, path = str(spec).partition(":")
        existing = by_label.get(label)
        same = existing is not None and (existing.get("protocol") or "") == signature
        action = "compute"
        if resume and existing is not None:
            if same:
                action = "reuse"
            elif legacy is not None and legacy_matches(existing, **legacy):
                action = "reuse-legacy"
        plan.append((label, path, action, existing))
    return plan


def merge_rows(rows, updates) -> list[dict]:
    """按 adapter 标签合并：同标签整行替换（不新增重复标签），新标签追加在末尾。

    报告工具用 `next(... adapter == 'V4')` 取行，重复标签会让它读到旧协议的旧值，
    所以"一个标签只允许一行"是这里的硬约束。
    """
    order: list[str] = []
    table: dict[str, dict] = {}
    for row in rows:
        label = (row.get("adapter") or "").strip()
        if not label:
            continue
        if label not in table:
            order.append(label)
        table[label] = row
    for row in updates:
        label = row["adapter"]
        if label not in table:
            order.append(label)
        table[label] = row
    return [table[label] for label in order]


def compat_key(row):
    """把任意一行（含旧格式）归一成"可比较键"：`(cache文件名, arch, device, 样本数, 时间步)`。

    报告工具要判断"这一行能不能和下表并列"，而旧格式行的 `samples` 存的是**前向次数**
    （720 = 144 样本 x 5 步），现代行的 `samples` 是样本数、前向次数在 `passes`。
    这里把两者统一成样本数，避免报告把 720 和 144 当成两个不同协议。
    返回 None 表示这行信息不全，不能判定为可比。
    """
    raw_steps = (row.get("timesteps") or "").replace("|", ",")
    steps = tuple(int(part) for part in (p.strip() for p in raw_steps.split(",")) if part)
    if not steps:
        return None
    try:
        samples = int(float(row.get("samples")))
    except (TypeError, ValueError):
        return None
    if not str(row.get("passes") or "").strip():    # 旧格式：samples 即前向次数
        if samples % len(steps):
            return None
        samples //= len(steps)
    cache_name = Path((row.get("cache") or "").replace("\\", "/")).name
    return (cache_name, (row.get("arch") or "sd15"), (row.get("device") or "cuda"), samples, steps)


CANONICAL_CSV_NAME = "eval_val.csv"
CANONICAL_SAMPLES = 144
CANONICAL_TIMESTEPS = (50, 250, 450, 650, 850)


def canonical_guard(csv_path, count, timesteps, device, allow=False):
    """拦住"用便宜协议覆盖正式证据文件"这一脚枪。

    `research/eval_val.csv` 是 docs/EVAL_REPORT.md、docs/TRAINING.md 引用的正式协议
    （全部 144 个测试样本 x 5 个固定时间步、cuda）证据文件。用 48 样本 x 2 步去写它，
    会把文档里引用的数字悄悄换成另一个协议的数值——这正是本项目反复强调不可接受的事。
    便宜协议应写进单独的 CSV（如 research/eval_quick.csv）。

    返回 None 表示放行；返回字符串表示拒绝原因（调用方负责打印并退出）。
    """
    if allow or not csv_path:
        return None
    if Path(str(csv_path)).name != CANONICAL_CSV_NAME:
        return None
    if int(count) == CANONICAL_SAMPLES and tuple(timesteps) == CANONICAL_TIMESTEPS and device == "cuda":
        return None
    return (f"拒绝写入 {CANONICAL_CSV_NAME}：它是文档引用的正式协议文件"
            f"（{CANONICAL_SAMPLES} 样本 x {list(CANONICAL_TIMESTEPS)}，cuda），"
            f"当前是 {int(count)} 样本 x {list(timesteps)}，device={device}。\n"
            f"  → 便宜协议请换一个 --csv（例如 research/eval_quick.csv）；"
            f"确实要覆盖正式文件时显式加 --allow-overwrite-canonical。")


def paired_stats(baseline_losses, other_losses):
    """配对差异统计：同一 (样本, 时间步) 上两个 adapter 的损失差。

    为什么必须配对：本协议的样本间标准差约 0.19，只看均值的标准误 ≈ 0.19/sqrt(720) ≈ 0.007，
    而 V5 与 V4 的均值差只有 5e-4 —— 非配对口径根本判不出差别。但两者是在**同样的样本、
    同样的噪声、同样的时间步**上测的，逐对相减后样本间方差大部分抵消，剩下的才是真实差异。
    返回均值差（正数表示 other 更低=更好）、差值标准误、t 值与配对数；长度不一致时返回 None。
    """
    import math

    baseline = [float(value) for value in baseline_losses]
    other = [float(value) for value in other_losses]
    if not baseline or len(baseline) != len(other):
        return None
    diffs = [b - o for b, o in zip(baseline, other)]
    count = len(diffs)
    mean = sum(diffs) / count
    if count > 1:
        variance = sum((value - mean) ** 2 for value in diffs) / (count - 1)
        standard_error = math.sqrt(max(variance, 0.0) / count)
    else:
        standard_error = 0.0
    if standard_error > 0:
        t_value = mean / standard_error
    else:
        # 差值是常数：均值为 0 表示两组完全一致（无可判定），否则是"无限显著"
        t_value = 0.0 if abs(mean) < 1e-15 else float("inf")
    return {"n": count, "mean_delta": mean, "stderr": standard_error, "t": t_value,
            "baseline_mean": sum(baseline) / count, "other_mean": sum(other) / count}


def format_seconds(seconds: float) -> str:
    """给日志读的耗时；超过 60 秒换成 x分y秒。"""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}分{seconds % 60:04.1f}秒"


def format_eta(remaining_seconds: float) -> str:
    if remaining_seconds <= 0:
        return "—"
    minutes, seconds = divmod(int(remaining_seconds), 60)
    if minutes < 1:
        return f"{seconds}s"
    return f"{minutes}分{seconds:02d}秒"
