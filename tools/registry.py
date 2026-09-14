"""registry.py — 版本台账的维护入口（纯 CPU，不需要 GPU）。

台账是"模型版本"的唯一真相（设计见 `docs/VERSION_ROUTING.md`）。这个脚本是**唯一**允许
改台账的地方，四个动作对应四件事：

    scan            扫磁盘重建事实（算 sha256、认 TE、收进未登记的新版本）
    eval <id>       把现有评测产物（val / KID / CLIP / 人工盲测）写进该版本
    channel ...     翻通道指针（default / previous / staging）——**上线与回滚就是这一步**
    validate        校验四条不变式（默认必须 promoted+在盘+哈希对；previous 必须在盘；
                    archived 不得出现在通道里；台账里不出现机器绝对路径）
    show            打印台账（人看的）

为什么"上线"不放在这个脚本里而只放"翻指针"：预检、冒烟出图、预热属于 **P3 的 promote.py**
（见设计稿 §2.4），那一步会调用这里的 `channel`。这里保持"只改数据、可被门禁演练"的定位。

用法:
    python tools/registry.py scan                    # 扫磁盘 → 更新台账
    python tools/registry.py eval v5b                # 从评测产物写入 eval 字段
    python tools/registry.py channel set default v5b # 上线（翻指针）
    python tools/registry.py channel set previous v4 # 指定回滚目标
    python tools/registry.py validate                # 四条不变式
    python tools/registry.py show
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

# 评测产物里的候选名 ↔ 台账 slug（现有 CSV 用的是当年的旧命名）
VAL_LABEL = {
    "v4": "V4", "v5": "V5", "v5b": "V5b",
    "merge-v4v5-linear": "linear", "merge-v4v5-slerp": "slerp",
    "merge-v4v5-ties": "ties", "merge-v4v5-dare": "dare_ties",
    "merge-v4v5b-linear": "v4v5b_linear", "merge-v4v5b-slerp": "v4v5b_slerp",
}
FID_LABEL = {"v4": "V4", "v5": "V5", "v5b": "V5b", "v6q": "V6"}


def _display(path: Path) -> str:
    """能相对项目根就相对，否则给绝对路径（VERSION_REGISTRY 可能指向仓库外，测试也用临时目录）。"""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def collect_eval(slug: str) -> dict:
    """把分散的评测产物汇总成一条 eval 记录（不重跑评测，只复用产物）。"""
    out: dict = {}
    label = VAL_LABEL.get(slug)
    if label:
        for row in _rows(ROOT / "research" / "eval_val.csv"):
            if row.get("adapter") == label and row.get("mean_mse"):
                out["val_mse"] = float(row["mean_mse"])
                out["val_samples"] = int(row["samples"]) if row.get("samples") else None
    fid = FID_LABEL.get(slug)
    if fid:
        for row in _rows(ROOT / "research" / "fid_v6" / "fid_metrics.csv"):
            if row.get("adapter") == fid:
                out.update({"kid": float(row["kid"]), "kid_std": float(row["kid_std"]),
                            "clip_fid": float(row["clip_fid"]), "clip_score": float(row["clip_score"]),
                            "sharpness": float(row["sharpness"]), "text_encoder": row["te"]})
    human_path = ROOT / "research" / "human_verdict.json"
    if human_path.exists():
        human = json.loads(human_path.read_text(encoding="utf-8"))
        for row in human.get("versions", []):
            # 人工评判页用的标签是 V4/V5/V5b/V6q，与 slug 大小写不一致 ⇒ 统一小写比较
            if str(row.get("version", "")).lower() == slug.lower():
                out["human"] = {"decided": row["decided"], "wins": row["wins"],
                                "rate": row["rate"], "ci": row["ci"],
                                "verdict": "无法区分" if row["indistinguishable"] else "显著高于随机",
                                "source": "docs/HUMAN_VERDICT.md"}
    out["recorded_at"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc).isoformat(timespec="seconds")
    return out


def validate(data: dict) -> tuple[list[str], list[str]]:
    """校验不变式。返回 (problems, warnings)。

    problems = 设计稿 §2.1 的四条硬不变式（default promoted+在盘+哈希对；previous 在盘且未归档；
               archived 不得进通道；台账里不出现机器绝对路径）。
    warnings = 软规则（例如 promoted 但还没有评测记录）—— 上线前必须有证据，由 P3 的
               promote.py 强制（`--ack-indistinguishable`），但不该让日常 scan 变成失败。
    """
    problems: list[str] = []
    warnings: list[str] = []
    if data.get("schema") != 1:
        problems.append(f"schema 必须为 1（实际 {data.get('schema')!r}）")
    versions = data.get("versions", {})
    channels = data.get("channels", {})
    if not versions:
        problems.append("versions 为空：先跑 registry.py scan")
    for slug, entry in versions.items():
        if entry.get("state") not in config.VALID_STATES:
            problems.append(f"{slug}: 非法状态 {entry.get('state')!r}")
        rel = (entry.get("adapter") or {}).get("dir")
        if not rel:
            problems.append(f"{slug}: 缺 adapter.dir")
            continue
        if str(rel).startswith(("/", "\\")) or ":" in str(rel) or ".." in Path(str(rel)).parts:
            problems.append(f"{slug}: adapter.dir 必须是 models/ 内的相对路径（{rel}）")
            continue
        if entry.get("state") != "archived":
            path = config.MODELS_DIR / rel / "adapter_config.json"
            if not path.exists():
                problems.append(f"{slug}: 权重不在盘上（{rel}）但状态是 {entry.get('state')}")
        if entry.get("state") == "promoted" and not entry.get("eval"):
            warnings.append(f"{slug}: promoted 但还没有任何评测记录（上线前应补齐："
                            f"registry.py eval {slug}）")
        if entry.get("needs_review"):
            warnings.append(f"{slug}: 未登记版本（needs_review），标签/说明是占位内容")
    default = channels.get("default")
    if not default:
        problems.append("channels.default 为空")
    elif default not in versions:
        problems.append(f"channels.default={default} 不在 versions 里")
    else:
        if versions[default].get("state") != "promoted":
            problems.append(f"channels.default={default} 的状态不是 promoted"
                            f"（{versions[default].get('state')}）")
        ok, reason = config.verify_adapter(default)
        if not ok:
            problems.append(f"channels.default={default} 校验失败：{reason}")
    previous = channels.get("previous")
    if previous:
        if previous not in versions:
            problems.append(f"channels.previous={previous} 不在 versions 里（回滚会失败）")
        elif versions[previous].get("state") == "archived":
            problems.append(f"channels.previous={previous} 已被归档 —— previous 永不归档（回滚要用）")
        else:
            ok, reason = config.verify_adapter(previous)
            if not ok:
                problems.append(f"channels.previous={previous} 校验失败：{reason}")
    for name, slug in channels.items():
        if slug and versions.get(slug, {}).get("state") == "archived":
            problems.append(f"channels.{name}={slug} 指向已归档版本")
    return problems, warnings


def cmd_scan(args) -> int:
    existing = config.load_registry()
    before = existing.get("versions", {})
    rebuilt = config.build_registry(existing)

    added = sorted(set(rebuilt["versions"]) - set(before))
    removed = sorted(set(before) - set(rebuilt["versions"]))
    changed = []
    for slug in sorted(set(rebuilt["versions"]) & set(before)):
        old_hash = (before[slug].get("adapter") or {}).get("sha256")
        new_hash = (rebuilt["versions"][slug]["adapter"] or {}).get("sha256")
        if old_hash and new_hash and old_hash != new_hash:
            changed.append(slug)
    if added:
        print(f"新收进台账（candidate，需补说明与评测）：{', '.join(added)}")
    if removed:
        print(f"磁盘上已不存在，从台账移除：{', '.join(removed)}")
    for slug in changed:
        state = rebuilt["versions"][slug].get("state")
        flag = "⚠ 已上线版本的权重变了！" if state in ("promoted", "superseded") else "（候选版本权重更新）"
        print(f"哈希变化：{slug} {flag}")
    if args.dry_run:
        print("[dry-run] 未写盘")
        return 0
    path = config.save_registry(rebuilt)
    print(f"台账已更新：{_display(path)}（{len(rebuilt['versions'])} 个版本，"
          f"default={rebuilt['channels']['default']}，previous={rebuilt['channels']['previous']}）")
    problems, warnings = validate(config.load_registry())
    for warning in warnings:
        print(f"  [提示] {warning}")
    for problem in problems:
        print(f"  [校验] {problem}")
    return 0 if not problems else 1


def cmd_eval(args) -> int:
    data = config.load_registry()
    if args.id not in data["versions"]:
        print(f"[错误] 台账里没有版本 {args.id}；可选：{', '.join(sorted(data['versions']))}")
        return 2
    record = collect_eval(args.id)
    if not record:
        print(f"[错误] 没有为 {args.id} 找到任何评测产物（先跑 eval_val_mse.py / eval_fid.py）")
        return 2
    data["versions"][args.id]["eval"] = record
    config.save_registry(data)
    pretty = "、".join(f"{k}={v}" for k, v in record.items() if k != "recorded_at")
    print(f"{args.id} 的评测已写入台账：{pretty}")
    return 0


def cmd_channel(args) -> int:
    data = config.load_registry()
    versions = data["versions"]
    if args.action == "show":
        for name, slug in data["channels"].items():
            print(f"  {name:9s} = {slug or '(空)'}")
        return 0
    target = args.value
    if target in ("none", "null", "-"):
        data["channels"][args.name] = None
        config.save_registry(data)
        print(f"channels.{args.name} 已清空")
        return 0
    if target not in versions:
        print(f"[错误] 台账里没有版本 {target}；可选：{', '.join(sorted(versions))}")
        return 2
    if versions[target].get("state") == "archived":
        print(f"[错误] {target} 已归档；先 restore 再上线（归档版本不许进通道）")
        return 2
    if args.name == "default":
        old = data["channels"].get("default")
        if old and old != target and old in versions:
            if data["channels"].get("previous") and data["channels"]["previous"] != old:
                versions[data["channels"]["previous"]]["state"] = "superseded"
            data["channels"]["previous"] = old          # 旧默认自动成为回滚目标
            versions[old]["state"] = "superseded"
        versions[target]["state"] = "promoted"
        versions[target]["promoted_at"] = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds")
    else:
        data["channels"][args.name] = target
        # previous 是"上一版默认"，语义上属于 superseded（仍在盘上、可回滚）
        if args.name == "previous" and target != data["channels"].get("default"):
            versions[target]["state"] = "superseded"
    data["channels"][args.name] = target
    config.save_registry(data)
    print(f"channels.{args.name} = {target}（台账已更新；服务端读 mtime，无需重启）")
    problems, warnings = validate(config.load_registry())
    for warning in warnings:
        print(f"  [提示] {warning}")
    for problem in problems:
        print(f"  [校验] {problem}")
    return 0 if not problems else 1


def cmd_validate(args) -> int:
    data = config.load_registry()
    problems, warnings = validate(data)
    for warning in warnings:
        print(f"[提示] {warning}")
    if not problems:
        print(f"台账校验通过：{len(data['versions'])} 个版本，"
              f"default={data['channels']['default']}，previous={data['channels']['previous']}")
        return 0
    print(f"台账校验失败（{len(problems)} 项）：")
    for problem in problems:
        print(f"  - {problem}")
    return 1


def cmd_show(args) -> int:
    data = config.load_registry()
    channels = data["channels"]
    print(f"台账：{_display(config.REGISTRY_PATH)}")
    print(f"通道：default={channels.get('default')}  previous={channels.get('previous')}  "
          f"staging={channels.get('staging')}")
    print(f"{'slug':22s} {'状态':11s} {'哈希':13s} {'TE':4s} {'val':10s} {'KID':10s} 标签")
    for slug, entry in sorted(data["versions"].items()):
        ev = entry.get("eval", {})
        val = f"{ev['val_mse']:.6f}" if isinstance(ev.get("val_mse"), float) else "—"
        kid = f"{ev['kid']:.6f}" if isinstance(ev.get("kid"), float) else "—"
        print(f"{slug:22s} {entry.get('state', '?'):11s} "
              f"{(entry['adapter'].get('sha256') or '')[:12]:13s} "
              f"{'yes' if entry['adapter'].get('text_encoder') else '-':4s} "
              f"{val:10s} {kid:10s} {entry.get('label', '')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="版本台账维护（scan / eval / channel / validate / show）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_scan = sub.add_parser("scan", help="扫磁盘更新台账")
    p_scan.add_argument("--dry-run", action="store_true")
    p_scan.set_defaults(func=cmd_scan)
    p_eval = sub.add_parser("eval", help="把评测产物写进某版本")
    p_eval.add_argument("id")
    p_eval.set_defaults(func=cmd_eval)
    p_ch = sub.add_parser("channel", help="翻通道指针（上线/回滚就是这一步）")
    p_ch.add_argument("action", choices=["set", "show"])
    p_ch.add_argument("name", nargs="?", choices=["default", "previous", "staging"])
    p_ch.add_argument("value", nargs="?")
    p_ch.set_defaults(func=cmd_channel)
    sub.add_parser("validate", help="校验四条不变式").set_defaults(func=cmd_validate)
    sub.add_parser("show", help="打印台账").set_defaults(func=cmd_show)
    args = ap.parse_args()
    if args.cmd == "channel" and args.action == "set" and (not args.name or not args.value):
        ap.error("channel set 需要 <default|previous|staging> <id|none>")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
