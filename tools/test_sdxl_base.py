"""test_sdxl_base.py — 基座解析的回归测试（纯 CPU，不加载模型、不联网）。

为什么值得测：2026-09-13 SDXL 主线因为基座走 HF Hub id 触发联网取文件，在代理环境下以
SSL 错误失败，而探针把异常统一记成 "not feasible"，于是整条主线被误判成"6GB 跑不了 SDXL"。
`sdxl_base.resolve_base()` 现在负责把基座稳定地解析到本地快照，它是这条链路上最关键的一环。

用法: python tools/test_sdxl_base.py     （退出码 0 = 全部通过）
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdxl_base  # noqa: E402

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


def make_snapshot(root: Path, repo: str, name: str, complete: bool) -> Path:
    snapshot = root / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / name
    (snapshot / "unet").mkdir(parents=True, exist_ok=True)
    (snapshot / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionXLPipeline"}),
                                               encoding="utf-8")
    if not complete:
        # 半个快照：缺 unet 目录（比没有快照更危险，会让人以为模型没问题）
        for child in (snapshot / "unet").iterdir():
            child.unlink()
        (snapshot / "unet").rmdir()
    return snapshot


def main() -> int:
    original_hf_home = os.environ.get("HF_HOME")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        os.environ["HF_HOME"] = str(root)

        print("== 快照解析 ==")
        snapshot = make_snapshot(root, "org/repo", "aaa", complete=True)
        check("HF 缓存里的完整快照被解析出来",
              sdxl_base.resolve_base("org/repo") == str(snapshot),
              sdxl_base.resolve_base("org/repo"))
        check("快照被识别为本地", sdxl_base.is_local(str(snapshot)))
        check("本地目录要求离线加载",
              sdxl_base.offline_kwargs(str(snapshot)) == {"local_files_only": True})
        check("描述里标明是本地目录", "本地目录" in sdxl_base.describe(str(snapshot)))

        print("== 半个快照必须被跳过 ==")
        incomplete = make_snapshot(root, "org/half", "zzz", complete=False)
        # zzz 比 aaa 新，但缺 unet：不能选它
        check("缺 unet 的最新快照不会被选中",
              sdxl_base.resolve_base("org/half") == "org/half",
              sdxl_base.resolve_base("org/half"))
        make_snapshot(root, "org/half", "aaa", complete=True)
        chosen = sdxl_base.resolve_base("org/half")
        check("回退到完整的那份快照", chosen.endswith("aaa") and "snapshots" in chosen, chosen)
        check("不完整快照确实存在（说明测试有效）", incomplete.exists())

        print("== 本地目录与兜底行为 ==")
        local = root / "my_model"
        (local / "unet").mkdir(parents=True)
        (local / "model_index.json").write_text("{}", encoding="utf-8")
        check("项目内本地目录原样返回", sdxl_base.resolve_base(str(local)) == str(local))
        check("没有 model_index.json 的目录不当作模型",
              sdxl_base.resolve_base(str(root)) == str(root))
        check("缓存里没有的 repo 原样返回（便于报错时看清原值）",
              sdxl_base.resolve_base("no/such-repo") == "no/such-repo")
        check("空值安全", sdxl_base.resolve_base("") == "")
        check("远程路径不要求离线（保持旧行为）",
              sdxl_base.offline_kwargs("no/such-repo") == {})
        check("描述里提示远程可能联网", "非本地" in sdxl_base.describe("no/such-repo"))

    if original_hf_home is None:
        os.environ.pop("HF_HOME", None)
    else:
        os.environ["HF_HOME"] = original_hf_home

    print(f"\n结果：{PASSED} 项通过，{len(FAILED)} 项失败")
    for name in FAILED:
        print(f"  - {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
