"""test_registry.py — 版本台账的门禁（纯 CPU，用临时目录演练，不碰真权重）。

台账是"上线 = 翻指针"的基础，它一旦出错，后果是**静默用错权重**或**回滚失败**。
所以这里不测"函数能跑"，测的是设计稿 §2.1 的四条不变式与三个动作的边界：

  1. 重建（scan）：磁盘上多一个目录 → 收成 candidate；少一个目录 → 从台账移除，
     且若它占了通道指针要**立刻摘掉**（否则上线会指向不存在的权重）；
     哈希变了要报出来（已上线版本权重变化 = 事故信号）。
  2. 翻指针（channel）：default 必须指向在盘、哈希对、状态 promoted 的版本；
     旧 default 自动变成 previous 且状态 superseded；指向归档版本必须被拒。
  3. 归档（archived）：不得出现在任何通道里；previous 永不归档（回滚要用）；
     归档后从可选清单里消失但仍留在台账里（历史可追溯）。
  4. 校验（validate）：四条不变式逐条能被"故意弄坏"的数据触发。

演练用临时目录 + 伪造的小权重文件（几百字节），因此不需要 GPU、不需要真模型。

用法: python tools/test_registry.py     （退出码 0 = 全部通过）
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def make_fixture(root: Path, layout: dict[str, str]) -> None:
    """在临时根目录里造权重目录：key = 相对 models/ 的目录，value = 权重文件内容。

    内容决定哈希，所以"改内容"就等于"权重被换过"，用来演练哈希校验。
    """
    for relative, payload in layout.items():
        folder = root / "models" / relative
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "adapter_config.json").write_text('{"r": 4}', encoding="utf-8")
        (folder / "adapter_model.safetensors").write_text(payload, encoding="utf-8")


def run_cli(root: Path, *args: str) -> subprocess.CompletedProcess:
    """在临时根目录里跑真实的 CLI（LANDSCAPE_ROOT 指向 fixture，台账也落在里面）。"""
    env = {**__import__("os").environ, "LANDSCAPE_ROOT": str(root),
           "VERSION_REGISTRY": str(root / "registry" / "versions.json"),
           "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(ROOT / "tools" / "registry.py"), *args],
                          cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=120)


def registry(root: Path) -> dict:
    return json.loads((root / "registry" / "versions.json").read_text(encoding="utf-8"))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lsart-registry-"))
    try:
        # 按 KNOWN_VERSIONS 的真实布局造两个已知版本 + 一份微调过的文本编码器
        make_fixture(tmp, {"v4_640/adapter_best": "v4-payload-1",
                           "v5b_lora/adapter_best": "v5b-payload-1"})
        (tmp / "models" / "v5b_lora" / "adapter_best_text_encoder.pt").write_text(
            "v5b-te", encoding="utf-8")

        print("scan：从磁盘建立台账")
        done = run_cli(tmp, "scan")
        check("scan 成功", done.returncode == 0, (done.stdout + done.stderr)[-300:])
        data = registry(tmp)
        check("两个已知目录都被收进台账（按 KNOWN_VERSIONS 的 slug）",
              set(data["versions"]) == {"v4", "v5b"}, str(list(data["versions"])))
        check("权重哈希被记录且非空",
              all(v["adapter"]["sha256"] for v in data["versions"].values()))
        check("文本编码器被认出来（v5b 有、v4 无）",
              data["versions"]["v5b"]["adapter"]["text_encoder"] is not None
              and data["versions"]["v4"]["adapter"]["text_encoder"] is None)
        check("没有台账时按引导默认提拔 v5b，另一个是 candidate",
              data["versions"]["v5b"]["state"] == "promoted"
              and data["versions"]["v4"]["state"] == "candidate")
        check("台账里没有机器绝对路径",
              all(":" not in v["adapter"]["dir"] and not v["adapter"]["dir"].startswith("/")
                  for v in data["versions"].values()))

        print("\nchannel：翻指针就是上线/回滚")
        done = run_cli(tmp, "channel", "set", "default", "v4")      # 先上 v4
        check("把 default 指向 v4 成功", done.returncode == 0, (done.stdout + done.stderr)[-200:])
        data = registry(tmp)
        check("v4 变 promoted、v5b 变 superseded 且成为 previous（回滚目标自动就位）",
              data["versions"]["v4"]["state"] == "promoted"
              and data["versions"]["v5b"]["state"] == "superseded"
              and data["channels"]["previous"] == "v5b")
        check("promoted_at 被写入", bool(data["versions"]["v4"].get("promoted_at")))
        done = run_cli(tmp, "channel", "set", "default", "v5b")     # 再切回 v5b（演练回滚方向）
        data = registry(tmp)
        check("再切回 v5b：previous 变成 v4（一键回滚目标正确）",
              data["channels"]["default"] == "v5b" and data["channels"]["previous"] == "v4")
        before = dict(data["channels"])
        done = run_cli(tmp, "channel", "set", "default", "v9-not-exist")
        check("指向不存在的版本被拒", done.returncode != 0 and "没有版本" in (done.stdout + done.stderr))
        check("被拒后通道未被改动", registry(tmp)["channels"] == before)

        print("\nvalidate：四条不变式")
        done = run_cli(tmp, "validate")
        check("当前台账通过校验（promoted 缺评测只是提示，不算失败）",
              done.returncode == 0 and "[提示]" in done.stdout, (done.stdout + done.stderr)[-200:])

        # ① default 权重被换掉（哈希不符）→ 必须报出来
        (tmp / "models" / "v5b_lora" / "adapter_best" / "adapter_model.safetensors").write_text(
            "tampered", encoding="utf-8")
        done = run_cli(tmp, "validate")
        check("default 权重被篡改 → 校验失败（哈希不符）",
              done.returncode != 0 and "校验失败" in (done.stdout + done.stderr),
              (done.stdout + done.stderr)[-200:])
        done = run_cli(tmp, "scan")
        check("scan 会报告已上线版本的哈希变化",
              "哈希变化" in done.stdout and "已上线版本" in done.stdout, done.stdout[-200:])

        # ② archived 出现在通道里 → 必须报错
        data = registry(tmp)
        data["versions"]["v4"]["state"] = "archived"
        (tmp / "registry" / "versions.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        done = run_cli(tmp, "validate")
        check("previous 被归档 → 校验失败（回滚要用，不许归档）",
              done.returncode != 0 and "previous" in (done.stdout + done.stderr),
              (done.stdout + done.stderr)[-200:])
        done = run_cli(tmp, "channel", "set", "default", "v4")
        check("不能把 default 指向归档版本", done.returncode != 0 and "已归档" in (done.stdout + done.stderr))

        print("\nscan：磁盘变化后的收敛")
        data = registry(tmp)
        data["versions"]["v4"]["state"] = "candidate"          # 复原
        (tmp / "registry" / "versions.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        shutil.rmtree(tmp / "models" / "v4_640")
        done = run_cli(tmp, "scan")
        check("目录被删 → 从台账移除并明确报告",
              done.returncode == 0 and "磁盘上已不存在" in done.stdout, done.stdout[-200:])
        data = registry(tmp)
        check("被删版本不再占 previous 指针（否则回滚会指向空气）",
              "v4" not in data["versions"] and data["channels"].get("previous") is None,
              str(data["channels"]))
        check("default 仍然有效", data["channels"]["default"] == "v5b")

        print("\n未知版本：收进来但要求人工补说明（不猜标签）")
        (tmp / "models" / "v9_lora" / "adapter_best").mkdir(parents=True)
        (tmp / "models" / "v9_lora" / "adapter_best" / "adapter_config.json").write_text("{}", encoding="utf-8")
        (tmp / "models" / "v9_lora" / "adapter_best" / "adapter_model.safetensors").write_text(
            "v9", encoding="utf-8")
        done = run_cli(tmp, "scan")
        data = registry(tmp)
        auto = [s for s in data["versions"] if s.startswith("auto-")]
        check("新目录以 auto-<dir> 形式收进台账", len(auto) == 1, str(list(data["versions"])))
        if auto:
            entry = data["versions"][auto[0]]
            check("标记 needs_review 且说明是占位（不自动猜标签）",
                  entry["needs_review"] is True and "未登记" in entry["note"])

        print("\n台账缺失：按磁盘重建并告警（绝不静默退回硬编码）")
        (tmp / "registry" / "versions.json").unlink()
        done = run_cli(tmp, "validate")
        check("没有台账时 validate 会先重建再校验",
              "台账不存在" in (done.stdout + done.stderr) and "重建" in (done.stdout + done.stderr),
              (done.stdout + done.stderr)[-200:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n结果：{PASSED} 项通过，{len(FAILED)} 项失败")
    for name in FAILED:
        print(f"  - {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
