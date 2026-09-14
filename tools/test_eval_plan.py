"""test_eval_plan.py — `tools/eval_common.py` 与评测驱动的纯逻辑回归（无 GPU、无 torch）。

覆盖 2026-09-12 事故后新增的三条保证：
  1. 每完成一个 adapter 就落盘 —— 被 kill 不丢已完成行；
  2. 只有**协议签名一致**的行才复用，样本数/时间步/设备变了必须重算；
  3. 一个 adapter 标签在 CSV 里只允许一行，否则报告工具会读到旧协议的旧值。

用法: python tools/test_eval_plan.py     （退出码 0 = 全部通过）
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import (FIELDS, canonical_guard, compat_key, format_eta,  # noqa: E402
                         format_seconds, legacy_matches, load_rows, make_row,
                         merge_rows, paired_stats, parse_timesteps, plan_run,
                         protocol_signature, save_rows)

PASSED = 0
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if condition:
        PASSED += 1
        print(f"  [ok]   {name}")
    else:
        FAILED.append(name)
        print(f"  [FAIL] {name}  {detail}")


def expect_system_exit(name: str, function, *args) -> None:
    try:
        function(*args)
    except SystemExit as error:
        check(name, True, str(error))
    else:
        check(name, False, "应当抛出 SystemExit 但没有")


def main() -> int:
    default = [50, 250, 450, 650, 850]
    print("== 时间步解析 ==")
    check("空值回退默认序列", parse_timesteps("", default) == default)
    check("None 回退默认序列", parse_timesteps(None, default) == default)
    check("升序排列", parse_timesteps("450,50", default) == [50, 450])
    check("去重", parse_timesteps("50,50,450", default) == [50, 450])
    check("容忍空格", parse_timesteps(" 50 , 450 ", default) == [50, 450])
    check("忽略空段", parse_timesteps("50,,450,", default) == [50, 450])
    expect_system_exit("非整数报错", parse_timesteps, "abc", default)
    expect_system_exit("超出 0..999 报错", parse_timesteps, "1200", default)
    expect_system_exit("全空段报错", parse_timesteps, ",,,", default)

    print("== 协议签名 ==")
    base_sig = protocol_signature("a/cache_v4.pt", "sd15", "cuda", 144, default)
    check("同输入同签名", protocol_signature("a/cache_v4.pt", "sd15", "cuda", 144, default) == base_sig)
    check("样本数不同 -> 不可比", protocol_signature("a/cache_v4.pt", "sd15", "cuda", 48, default) != base_sig)
    check("时间步不同 -> 不可比", protocol_signature("a/cache_v4.pt", "sd15", "cuda", 144, [50, 450]) != base_sig)
    check("设备不同 -> 不可比", protocol_signature("a/cache_v4.pt", "sd15", "cpu", 144, default) != base_sig)
    check("架构不同 -> 不可比", protocol_signature("a/cache_v4.pt", "sdxl", "cuda", 144, default) != base_sig)
    check("同名不同目录的 cache -> 不可比",
          protocol_signature("b/cache_v4.pt", "sd15", "cuda", 144, default) != base_sig)
    check("反斜杠路径归一化",
          protocol_signature("a\\cache_v4.pt", "sd15", "cuda", 144, default) == base_sig)

    print("== CSV 读写（断点续跑的数据来源） ==")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "nested" / "eval_test.csv"
        row_a = make_row("V4", "models/v4", "dataset1024/cache_v4_640.pt", "sd15", "cuda",
                         base_sig, 0.1864444, 0.0912345, 144, 720, default, 61.25)
        row_b = make_row("ties", "models/merged/ties", "dataset1024/cache_v4_640.pt", "sd15", "cuda",
                         base_sig, 0.1851234, 0.09, 144, 720, default, 60.0)
        save_rows(csv_path, [row_a])                      # 第 1 个 adapter 完成即落盘
        after_first = load_rows(csv_path)
        check("父目录自动创建", csv_path.exists())
        check("落盘 1 行", len(after_first) == 1, f"实际 {len(after_first)}")
        check("均值往返一致", abs(float(after_first[0]["mean_mse"]) - 0.186444) < 1e-9)
        check("列头顺序符合契约", list(after_first[0].keys()) == FIELDS, str(list(after_first[0].keys())))
        raw = csv_path.read_bytes()
        check("LF 行尾（无 CRLF）", b"\r\n" not in raw)
        check("样本数与前向次数分列", after_first[0]["samples"] == "144" and after_first[0]["passes"] == "720")
        save_rows(csv_path, [row_a, row_b])               # 第 2 个 adapter 完成后再落盘
        check("续写后 2 行（模拟被 kill 前已完成的部分）", len(load_rows(csv_path)) == 2)
        save_rows(csv_path, [{"adapter": "X", "mean_mse": 1.0}])   # 缺列不炸
        check("缺列用空串兜底", load_rows(csv_path)[0].get("path") == "")
        missing = load_rows(Path(tmp) / "不存在.csv")
        check("文件不存在返回空列表", missing == [])

    print("== 复用决策 ==")
    rows = [make_row("V4", "models/v4", "c.pt", "sd15", "cuda", base_sig, 0.1864, 0.09, 144, 720, default, 61.0)]
    specs = ["V4:models/v4", "BASE:", "V5:models/v5_lora/adapter_best"]
    same = plan_run(specs, rows, base_sig)
    check("协议一致 -> 复用", same[0][2] == "reuse")
    check("未评测标签 -> 计算", same[1][2] == "compute" and same[1][3] is None)
    check("复用行带回原值", float(same[0][3]["mean_mse"]) == 0.1864)
    other = protocol_signature("c.pt", "sd15", "cuda", 48, [50, 450])
    check("协议不同 -> 重算", plan_run(specs, rows, other)[0][2] == "compute")
    check("--no-resume -> 全部重算", all(item[2] == "compute" for item in plan_run(specs, rows, base_sig, resume=False)))
    dup = rows + [dict(rows[0], mean_mse="0.9")]
    check("重复标签取最后一行", float(plan_run(["V4:x"], dup, base_sig)[0][3]["mean_mse"]) == 0.9)

    print("== 合并写入（一个标签只留一行） ==")
    existing = [
        make_row("BASE", "", "c.pt", "sd15", "cuda", base_sig, 0.1915, 0.09, 144, 720, default, 60.0),
        make_row("V4", "models/v4", "c.pt", "sd15", "cuda", "old-protocol", 0.1999, 0.09, 48, 96, [50], 20.0),
    ]
    updated = make_row("V4", "models/v4", "c.pt", "sd15", "cuda", base_sig, 0.1864, 0.09, 144, 720, default, 61.0)
    merged = merge_rows(existing, [updated])
    check("标签数不增加", len(merged) == 2, f"实际 {len(merged)}")
    check("位置保持不变", [row["adapter"] for row in merged] == ["BASE", "V4"])
    check("旧协议行被新行整行替换", merged[1]["protocol"] == base_sig and merged[1]["mean_mse"] == 0.1864)
    check("未涉及行原样保留", merged[0]["mean_mse"] == 0.1915)
    appended = merge_rows(existing, [make_row("slerp", "models/merged/slerp", "c.pt", "sd15", "cuda",
                                              base_sig, 0.185, 0.09, 144, 720, default, 59.0)])
    check("新标签追加在末尾", [row["adapter"] for row in appended] == ["BASE", "V4", "slerp"])
    check("空输入可用", merge_rows([], [updated]) and len(merge_rows([], [updated])) == 1)

    print("== 旧格式 CSV 的保守复用 ==")
    legacy_kwargs = {"cache": "dataset1024/cache_v4_640.pt", "arch": "sd15", "device": "cuda",
                     "count": 144, "timesteps": default}
    # 旧脚本写出的真实行形状：没有 protocol/device/passes 列，samples 存的是前向次数
    old_row = {"adapter": "V4", "path": "models/v4_640/adapter_best",
               "cache": "dataset1024/cache_v4_640.pt", "arch": "sd15", "mean_mse": "0.18644",
               "std": "0.09", "samples": "720", "timesteps": "50|250|450|650|850"}
    check("旧格式行可复用", legacy_matches(old_row, **legacy_kwargs))
    check("前向次数不符 -> 不复用",
          not legacy_matches(dict(old_row, samples="96"), **legacy_kwargs))
    check("时间步不符 -> 不复用",
          not legacy_matches(dict(old_row, timesteps="50|450"), **legacy_kwargs))
    check("cache 不同名 -> 不复用",
          not legacy_matches(dict(old_row, cache="other/cache_other.pt"), **legacy_kwargs))
    check("带 protocol 列的现代行不走旧格式推断",
          not legacy_matches(dict(old_row, protocol=base_sig), **legacy_kwargs))
    check("残缺行（缺 timesteps）-> 不复用",
          not legacy_matches({"adapter": "V4", "samples": "720"}, **legacy_kwargs))
    legacy_plan = plan_run(["V4:models/v4_640/adapter_best"], [old_row], base_sig, legacy=legacy_kwargs)
    check("plan_run 标记 reuse-legacy", legacy_plan[0][2] == "reuse-legacy", str(legacy_plan[0][2]))
    check("不传 legacy 时旧格式行重算",
          plan_run(["V4:models/v4_640/adapter_best"], [old_row], base_sig)[0][2] == "compute")
    check("--no-resume 时旧格式行也重算",
          plan_run(["V4:models/v4_640/adapter_best"], [old_row], base_sig,
                   resume=False, legacy=legacy_kwargs)[0][2] == "compute")
    check("协议完全一致仍优先按 reuse 判定",
          plan_run(["V4:x"], [make_row("V4", "x", "c.pt", "sd15", "cuda", base_sig,
                                       0.18, 0.09, 144, 720, default, 61.0)],
                   base_sig, legacy=legacy_kwargs)[0][2] == "reuse")

    print("== 可比较键（报告用它决定能否并表） ==")
    legacy_row = {"adapter": "V4", "cache": "dataset1024/cache_v4_640.pt", "arch": "sd15",
                  "mean_mse": "0.18644", "samples": "720", "timesteps": "50|250|450|650|850"}
    modern_row = make_row("ties", "models/merged/ties", "dataset1024/cache_v4_640.pt", "sd15",
                          "cuda", base_sig, 0.185, 0.09, 144, 720, default, 59.0)
    check("旧格式与现代格式归一后相等", compat_key(legacy_row) == compat_key(modern_row),
          f"{compat_key(legacy_row)} vs {compat_key(modern_row)}")
    check("归一后样本数是 144（不是 720）", compat_key(legacy_row)[3] == 144)
    check("时间步数量不同 -> 键不同",
          compat_key(dict(legacy_row, samples="96", timesteps="50|450"))
          != compat_key(modern_row))
    check("缓存不同名 -> 键不同",
          compat_key(dict(modern_row, cache="other/cache_other.pt")) != compat_key(modern_row))
    check("缺时间步 -> None", compat_key({"adapter": "X", "samples": "10"}) is None)
    check("前向次数不能整除时间步数 -> None",
          compat_key(dict(legacy_row, samples="721")) is None)

    print("== 配对差异统计 ==")
    base_losses = [1.0, 2.0, 3.0, 4.0]
    check("完全相同的两组 -> Δ=0、t=0",
          (lambda s: abs(s["mean_delta"]) < 1e-12 and abs(s["t"]) < 1e-12)(paired_stats(base_losses, base_losses)))
    better = [0.9, 1.9, 2.9, 3.9]                      # other 每对都低 0.1
    stats = paired_stats(base_losses, better)
    check("逐对更好的模型给出正 Δ（正数=更好）", abs(stats["mean_delta"] - 0.1) < 1e-12, str(stats))
    check("配对后标准误为 0（差值是常数）", abs(stats["stderr"]) < 1e-12)
    noisy = [1.0, 1.5, 3.5, 4.0]                       # 差值有涨落
    stats2 = paired_stats(base_losses, noisy)
    check("有涨落时标准误 > 0 且 t 有限",
          stats2["stderr"] > 0 and abs(stats2["t"]) < 1e6, str(stats2))
    check("长度不一致返回 None（协议不同不可配对）", paired_stats([1.0, 2.0], [1.0]) is None)
    check("空输入返回 None", paired_stats([], []) is None)
    check("配对数正确", paired_stats(base_losses, noisy)["n"] == 4)

    print("== 正式证据文件的写保护 ==")
    check("正式协议 + cuda 放行",
          canonical_guard("research/eval_val.csv", 144, default, "cuda") is None)
    cheap = canonical_guard("research/eval_val.csv", 48, [50, 450], "cuda")
    check("便宜协议写正式文件被拒", isinstance(cheap, str) and "eval_quick" in cheap)
    check("cpu 设备写正式文件被拒",
          isinstance(canonical_guard("research/eval_val.csv", 144, default, "cpu"), str))
    check("样本数不足被拒",
          isinstance(canonical_guard("research/eval_val.csv", 72, default, "cuda"), str))
    check("其他 CSV 不受限",
          canonical_guard("research/eval_merged.csv", 48, [50, 450], "cuda") is None)
    check("显式放行开关生效",
          canonical_guard("research/eval_val.csv", 48, [50, 450], "cuda", allow=True) is None)
    check("未指定 --csv 时不拦", canonical_guard("", 48, [50, 450], "cuda") is None)

    print("== 报告并表（在临时 ROOT 上跑集成测试，不碰真实证据） ==")
    with tempfile.TemporaryDirectory() as tmp:
        temp_root = Path(tmp)
        (temp_root / "research").mkdir(parents=True, exist_ok=True)
        (temp_root / "docs").mkdir(parents=True, exist_ok=True)
        header = "adapter,path,cache,arch,mean_mse,std,samples,timesteps\n"
        (temp_root / "research" / "eval_val.csv").write_text(
            header
            + "BASE,,dataset1024/cache_v4_640.pt,sd15,0.191528,0.09,720,50|250|450|650|850\n"
            + "V4,models/v4_640/adapter_best,dataset1024/cache_v4_640.pt,sd15,0.186437,0.09,720,50|250|450|650|850\n",
            encoding="utf-8")
        merged_header = ("adapter,path,cache,arch,device,protocol,mean_mse,std,samples,passes,"
                         "timesteps,seconds\n")
        (temp_root / "research" / "eval_merged.csv").write_text(
            merged_header
            + "V4,models/v4_640/adapter_best,dataset1024/cache_v4_640.pt,sd15,cuda,sig,0.186437,"
              "0.09,144,720,50|250|450|650|850,61.0\n"          # 与 eval_val.csv 重复的锚点行
            + "ties,models/merged/ties_0.50_0.50,dataset1024/cache_v4_640.pt,sd15,cuda,sig,0.185000,"
              "0.09,144,720,50|250|450|650|850,59.0\n"
            + "odd,models/merged/odd,dataset1024/cache_v4_640.pt,sd15,cuda,sig2,0.180000,"
              "0.09,48,96,50|450,12.0\n",
            encoding="utf-8")
        # SDXL 线：另一个基座 + 另一份缓存，报告必须单独列（§3.1 / §4.1）
        (temp_root / "research" / "eval_val_sdxl.csv").write_text(
            merged_header
            + "SDXL-BASE,,dataset1024/cache_v5_sdxl1024.pt,sdxl,cuda,sigx,0.220000,0.10,48,96,"
              "50|450,20.0\n"
            + "SDXL,models/v5_sdxl/adapter_best,dataset1024/cache_v5_sdxl1024.pt,sdxl,cuda,sigx,"
              "0.205000,0.10,48,96,50|450,20.0\n",
            encoding="utf-8")
        (temp_root / "research" / "compare_sdxl").mkdir(parents=True, exist_ok=True)
        (temp_root / "research" / "compare_sdxl" / "compare_metrics.csv").write_text(
            "adapter,prompt,seed,sharpness,saturation,contrast,mean_luma,clip_score\n"
            "SDXL-BASE,a valley,101,500.0,0.4000,55.0000,120.0000,0.2700\n"
            "SDXL,a valley,101,560.0,0.4500,58.0000,122.0000,0.2750\n",
            encoding="utf-8")
        (temp_root / "research" / "compare_sdxl" / "sdxl_probe.json").write_text(
            '{"mode": "offload", "gpu_total_gb": 6.0, "arms": [{"label": "SDXL", "peak_gb": 5.4, '
            '"seconds_per_image": 12.5}]}', encoding="utf-8")
        environment = dict(os.environ, LANDSCAPE_ROOT=str(temp_root))
        result = subprocess.run([sys.executable, "-X", "utf8",
                                 str(Path(__file__).resolve().parent / "make_eval_report.py")],
                                cwd=str(temp_root), env=environment,
                                capture_output=True, text=True, encoding="utf-8")
        report = (temp_root / "docs" / "EVAL_REPORT.md")
        text = report.read_text(encoding="utf-8") if report.exists() else ""
        check("报告生成成功", result.returncode == 0 and bool(text), result.stderr[-400:])
        check("可比较键一致的合并行并入表格", "| ties |" in text and "并入的合并 adapter" in text)
        check("协议不同的行被排除并说明", "因协议不一致被排除" in text and "`odd`" in text)
        check("被排除的行不进入表格", "| odd |" not in text)
        check("表头标注样本数 x 时间步", "样本数 × 时间步" in text and "| 144 × 5 |" in text)
        check("旧格式行的样本数被正确还原为 144", "| 720 ×" not in text)
        check("SDXL 单独成节（§3.1）", "### 3.1 SDXL 线" in text and "不可与上表并列比较" in text)
        check("SDXL 表含基座锚点与差值",
              "| SDXL-BASE | 0.22000 |" in text and "| SDXL | 0.20500 | -0.01500 |" in text)
        check("SDXL 协议行来自它自己的缓存",
              "cache_v5_sdxl1024.pt" in text and "48 × 2" in text)
        check("结论节给出 SDXL 判定", "SDXL 线判定" in text and "-0.01500" in text)
        check("SDXL 出图对比单独成节（§4.1）",
              "### 4.1 SDXL 出图对比" in text and "| SDXL | 560.0 |" in text)
        check("SDXL 出图带 CLIP 一致性列", "CLIP 一致性" in text and "0.2750" in text)
        check("6GB 可跑性实测被引用",
              "6GB 可跑性实测" in text and "12.5s" in text and "5.4GB" in text)
        check("重复的锚点行不会在表里出现两遍", text.count("| V4 |") == 1,
              f"V4 行出现 {text.count('| V4 |')} 次")
        check("主表模型数量正确（BASE/V4 + ties）", text.count("| 0.18500 |") == 1
              and "| BASE |" in text)

    print("== 日志格式 ==")
    check("秒级格式", format_seconds(9.44) == "9.4s")
    check("分钟级格式", format_seconds(61.25).startswith("1分"))
    check("ETA 小于 1 分钟", format_eta(45).endswith("s"))
    check("ETA 为 0 时占位", format_eta(0) == "—")

    print("== 静态检查：驱动脚本真的在循环内落盘 ==")
    source = (Path(__file__).resolve().parent / "eval_val_mse.py").read_text(encoding="utf-8")
    loop_at = source.find("for index, (label, path, action, existing) in enumerate(plan")
    flush_at = source.find("save_rows(args.csv, merge_rows(load_rows(args.csv), [row]))")
    check("存在主循环", loop_at > 0)
    check("落盘写在主循环内部（不是循环之后）", flush_at > loop_at and loop_at > 0,
          f"loop@{loop_at} flush@{flush_at}")
    check("声明了行缓冲输出", "line_buffering=True" in source and "line_buffer_stdout()" in source)
    check("支持 --plan 且不进入执行分支", '"--plan"' in source and "if args.plan:" in source)
    check("复用逻辑走 eval_common.plan_run", "plan_run(" in source)

    print(f"\n结果：{PASSED} 项通过，{len(FAILED)} 项失败")
    if FAILED:
        for name in FAILED:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
