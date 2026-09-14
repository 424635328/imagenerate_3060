"""test_adapter_routing.py — 「模型版本能路由到前端」这件事的门禁（纯 CPU，不需要 GPU）。

背景：产品现在能按请求切版本（前端下拉 → POST 的 `adapter` 字段 → 服务端白名单解析 →
管线按版本重建）。这类改动最容易出的不是崩溃，而是**看起来对但实际没生效**：

1. **路由失效**：前端选了新版本，请求里却没带这个字段（或后端忽略了它）——
   用户以为在用新模型，其实还是旧的。本项目对"静默 no-op"有前科（两次 adapter 事故）。
2. **路径穿越**：`adapter` 是客户端字符串，若被直接拼成路径，就能加载任意目录的权重。
3. **缓存串味**：结果缓存键不含版本 ⇒ 换版本后拿到旧版本的图，且完全看不出来。
4. **归档后仍可选**：版本下架了，前端还能选中它，点生成就报错。

本测试把上面四条钉住（真机出图是否变化的验证在 `tools/test_adapter_switch_live.py`，
那条需要 GPU 与在跑的后端，不进 dev.ps1 的 CPU 门禁）。

用法: python tools/test_adapter_routing.py     （退出码 0 = 全部通过）
"""
from __future__ import annotations

import inspect
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import config  # noqa: E402

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


def main() -> int:
    print("白名单与解析（服务端唯一入口）")
    slugs = config.adapter_slugs()
    check("白名单非空且含四个训练版本",
          {"v4", "v5", "v5b", "v6q"} <= set(slugs), str(slugs))
    check("默认版本在白名单里", config.DEFAULT_ADAPTER in slugs, config.DEFAULT_ADAPTER)
    bad_inputs = ["../../etc/passwd", "..\\..\\windows", "v9", "", None, "C:/Windows",
                  "v4;rm -rf /", "v4/../../..", " merge  ", "v4\x00"]
    rejected = []
    accepted_bad = []
    for value in bad_inputs:
        try:
            config.adapter_dir(value)
            accepted_bad.append(value)
        except ValueError:
            rejected.append(value)
    check("所有非法/穿越输入都被拒", not accepted_bad, f"被接受: {accepted_bad}")
    check("合法 slug 都能解析到 models/ 下且存在 adapter_config.json",
          all((config.adapter_dir(s) / "adapter_config.json").exists() for s in slugs))
    check("解析结果永不越出 models/",
          all(str(config.adapter_dir(s)).startswith(str(config.MODELS_DIR.resolve())) for s in slugs))

    print("\n目录接口（/models 的数据源）")
    rows = config.adapter_catalogue()
    check("目录里的 id 全部来自白名单", all(r["id"] in slugs for r in rows))
    check("每条都带 label/slogan/note（诚实标签，前端提示直接用）",
          all(r["label"] and r["slogan"] and r["note"] for r in rows))
    check("恰好一条 default=True 且与服务端默认一致",
          sum(1 for r in rows if r["default"]) == 1
          and next(r["id"] for r in rows if r["default"]) == config.DEFAULT_ADAPTER)
    check("TE 状态与实际文件一致（v4/v5b 有微调过的文本编码器）",
          {r["id"]: r["text_encoder"] for r in rows}["v4"] is True
          and {r["id"]: r["text_encoder"] for r in rows}["v5b"] is True
          and {r["id"]: r["text_encoder"] for r in rows}["v5"] is False)
    check("不泄露绝对路径（只回相对 models/ 的 dir）",
          all(not str(r["dir"]).startswith(("/", "C:", "F:")) for r in rows))

    print("\n后端接线（字段真的被贯穿了）")
    server_src = (ROOT / "server.py").read_text(encoding="utf-8")
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    check("GenerateReq 有 adapter 字段", re.search(r"^\s+adapter: str = Field", server_src, re.M) is not None)
    check("有 adapter 的校验器（走白名单解析）",
          "def _check_adapter" in server_src and "config_resolve_adapter(value)" in server_src)
    check("校验失败的报错**不回显**传入内容，只列白名单",
          "只允许" in server_src and "_adapter_slugs()" in server_src)
    check("生成路径把 adapter 目录传到了 _gen_one / get_i2i",
          "adapter_dir=adapter_path" in server_src
          and server_src.count("get_i2i(fast, sampler, adapter_path)") >= 2)
    check("_gen_one 与 get_pipe/get_i2i 都接受 adapter_dir",
          "adapter_dir=None" in app_src and app_src.count("adapter_dir") >= 4)
    check("管线缓存键含版本（切版本必须重建）",
          "str(adapter_dir or ADAPTER)" in app_src)
    check("/health 报告当前驻留的版本",
          '"adapter": _pipe_mode[2]' in app_src and "current_mode().get(\"adapter\")" in server_src)
    check("任务元数据带 adapter（前端/历史可追溯）", '"adapter": job.get("adapter")' in server_src)

    print("\n结果缓存必须按版本隔离（否则换版本拿到旧图且看不出来）")
    from server import GenerateReq, _cache_key
    req_a = GenerateReq(prompt="x", seed=1, adapter="v4")
    req_b = GenerateReq(prompt="x", seed=1, adapter="v5b")
    key_a = _cache_key(req_a, 1, 20, 7.5, "dpmpp2m_karras", False, "none", req_a.adapter_slug())
    key_b = _cache_key(req_b, 1, 20, 7.5, "dpmpp2m_karras", False, "none", req_b.adapter_slug())
    check("同参数不同版本 → 不同缓存键", key_a != key_b, f"{key_a} == {key_b}")
    check("同参数同版本 → 相同缓存键（缓存仍然有效）",
          key_a == _cache_key(req_a, 1, 20, 7.5, "dpmpp2m_karras", False, "none", "v4"))
    check("空 adapter（跟随默认）与显式默认版本等价",
          GenerateReq(prompt="x").adapter_slug() == config.DEFAULT_ADAPTER)
    check("缓存键函数签名里确实要 adapter_slug（防止有人漏传）",
          "adapter_slug" in inspect.signature(_cache_key).parameters)

    print("\n前端接线")
    index = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    main_js = (ROOT / "site" / "js" / "main.js").read_text(encoding="utf-8")
    api_js = (ROOT / "site" / "js" / "api.js").read_text(encoding="utf-8")
    ui_js = (ROOT / "site" / "js" / "ui.js").read_text(encoding="utf-8")
    proxy_js = (ROOT / "netlify" / "functions" / "proxy.js").read_text(encoding="utf-8")
    check("页面有模型版本下拉", 'id="adapter"' in index)
    check("下拉在设置持久化列表里（刷新后保持）", "'adapter'" in main_js)
    check("请求体带 adapter", "adapter: ui.el('adapter')" in main_js)
    check("通过代理拉取 /models", "op=models" in api_js and 'op === "models"' in proxy_js)
    check("下拉由后端清单填充（含 TE 标记与诚实说明）",
          "renderModels" in ui_js and "text_encoder" in ui_js and "无法区分" in ui_js)
    check("提示里写明四个版本在有效判据上无法区分（不做宣传化改写）",
          "无法区分" in index and "口味" in index)

    print(f"\n结果：{PASSED} 项通过，{len(FAILED)} 项失败")
    for name in FAILED:
        print(f"  - {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
