"""test_judge_collector.py — 评判收集器的门禁（纯 CPU，起真实 HTTP 服务，不用 GPU）。

为什么要有它：这条链路是"人工判据能不能回到作者手里"的唯一通道，
而且它监听在本机回环上、还带 CORS。四类事故都要钉住：

1. **跨站写入**：任意网页都能 POST 一份"评判"进你的仓库 → 必须 403；
2. **畸形/超限输入**：超大 body、错 schema、非法 seed → 必须 4xx 且**不落盘**；
3. **数字不可复算**：页面传上来的 summary 可能与记录不符 → 服务端必须**独立重算**，
   并把差异写进文件（本项目一贯要求"数字要能复算"）；
4. **半截文件**：进程被杀留下坏 JSON → 必须原子落盘（.tmp + os.replace）。

另外用**前端真实产出的 payload**（`node tools/smoke_versions.mjs --emit-payload`）做一次
跨语言联调：JS 造的结构必须被 Python 侧接受，否则"点提交"会静默失败。

用法: python tools/test_judge_collector.py     （退出码 0 = 全部通过）
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_collector as collector  # noqa: E402

# 本机测试必须**绕开代理**：这台机器上有 HTTP_PROXY=http://127.0.0.1:7890，
# urllib 会连 127.0.0.1 的请求也丢给代理，于是拿到 502 Bad Gateway（不是服务的问题）。
# 实测踩过一次：静态站点那一段全部 502，看起来像"服务坏了"。
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
urllib.request.install_opener(OPENER)

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


def wilson_from_judge() -> "callable":
    """从 tools/vlm_judge.py 原文抽取 wilson()，验证收集器里的实现与它一致。"""
    source = (ROOT / "tools" / "vlm_judge.py").read_text(encoding="utf-8")
    match = re.search(r"^def wilson\(.*?\n(?=^def |\Z)", source, re.S | re.M)
    namespace = {"math": math}
    exec(match.group(0), namespace)          # noqa: S102 —— 只跑本项目自己的纯函数
    return namespace["wilson"]


def post(port: int, body: bytes, origin: str | None, content_type: str = "application/json",
         path: str = "/submit"):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=body, method="POST",
        headers={"Content-Type": content_type, **({"Origin": origin} if origin else {})})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8")), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8")), dict(error.headers)


def sample_records() -> list[dict]:
    """构造一份"人工盲测"记录：12 题，其中 V5b 被选 5 次、V4 4 次、V6q 2 次、V5 1 次。"""
    winners = ["V5b"] * 5 + ["V4"] * 4 + ["V6q"] * 2 + ["V5"]
    return [{"prompt_index": index, "seed": 101 if index % 2 == 0 else 108, "tie": False,
             "winner": winner, "mode": "blind4", "blind": True,
             "order": ["V5b", "V4", "V6q", "V5"], "at": "2026-09-15T00:00:00Z"}
            for index, winner in enumerate(winners)]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="lsart-judge-"))
    out = tmp / "human_judge"
    collector._state["out"] = out
    collector._state["received"] = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), collector.Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    base = {"schema": 1, "source": "site/versions.html（人工盲测）",
            "versions": ["V4", "V5", "V5b", "V6q"], "records": sample_records()}

    print(f"起服务：127.0.0.1:{port}（落盘到临时目录）")

    try:
        print("\n健康检查与路由")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        check("/health 返回服务信息", health.get("ok") and health.get("service") == "lsart-judge-collector")
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            check("GET / 也能探活", response.status == 200)
        code, body, _ = post(port, b"{}", None, path="/nope")
        check("POST 到未知路径 404", code == 404, f"{code} {body}")

        print("\n跨站防护（本机回环 + CORS，必须挡住别人的网页）")
        code, body, _ = post(port, json.dumps(base).encode(), "https://evil.example.com")
        check("陌生 Origin → 403", code == 403, f"{code} {body}")
        check("403 时错误信息里带上 Origin（可诊断）", "evil.example.com" in body.get("error", ""))
        check("被拒的提交不落盘", not out.exists() or not list(out.glob("*.json")))
        code, _, _ = post(port, json.dumps(base).encode(), "https://6aa7f8179b5591da0127157b--landscape-art-demo.netlify.app")
        check("deploy-preview 子域放行（会随每次部署变化）", code == 200, str(code))
        code, _, _ = post(port, json.dumps(base).encode(), "http://127.0.0.1:8799")
        check("本地静态站点放行", code == 200, str(code))

        print("\n预算与格式（畸形输入一律 4xx 且不落盘）")
        before = len(list(out.glob("*.json"))) if out.exists() else 0
        code, body, _ = post(port, b"x" * (collector.MAX_BODY + 10), "http://127.0.0.1:8799")
        check("超大 body → 413", code == 413, f"{code} {body}")
        code, body, _ = post(port, b"not json", "http://127.0.0.1:8799")
        check("非 JSON → 400", code == 400, f"{code} {body}")
        code, body, _ = post(port, json.dumps({"schema": 2, "records": sample_records()}).encode(),
                             "http://127.0.0.1:8799")
        check("schema 不为 1 → 400", code == 400 and "schema" in body.get("error", ""), str(body))
        code, body, _ = post(port, json.dumps({"schema": 1, "records": []}).encode(), "http://127.0.0.1:8799")
        check("空 records → 400", code == 400, str(body))
        code, body, _ = post(port, json.dumps({"schema": 1, "records": [{"prompt_index": 99, "seed": 101,
                                                                        "winner": "V4"}]}).encode(),
                             "http://127.0.0.1:8799")
        check("越界 prompt_index → 400", code == 400 and "prompt_index" in body.get("error", ""), str(body))
        code, body, _ = post(port, json.dumps({"schema": 1, "records": [{"prompt_index": 1, "seed": -5,
                                                                        "winner": "V4"}]}).encode(),
                             "http://127.0.0.1:8799")
        check("非法 seed → 400", code == 400 and "seed" in body.get("error", ""), str(body))
        code, body, _ = post(port, json.dumps({"schema": 1, "records": [{"prompt_index": 1, "seed": 101}]}).encode(),
                             "http://127.0.0.1:8799")
        check("既非平局又没有 winner → 400", code == 400, str(body))
        code, _, _ = post(port, json.dumps(base).encode(), "http://127.0.0.1:8799", content_type="text/plain")
        check("Content-Type 不是 JSON → 415", code == 415, str(code))
        too_many = {"schema": 1, "records": [{"prompt_index": 0, "seed": 1, "winner": "V4"}]
                    * (collector.MAX_RECORDS + 1)}
        code, _, _ = post(port, json.dumps(too_many).encode(), "http://127.0.0.1:8799")
        check("records 超上限 → 400", code == 400, str(code))
        after = len(list(out.glob("*.json"))) if out.exists() else 0
        check("所有畸形请求都没有写文件", after == before, f"{before} → {after}")

        print("\n正常提交与原子落盘")
        saved_before = len(list(out.glob("verdicts_*.json")))
        code, body, headers = post(port, json.dumps(base).encode(), "https://landscape-art-demo.netlify.app")
        check("正常提交 200 且 ok=true", code == 200 and body.get("ok"), f"{code} {body}")
        check("返回里带落盘文件名与条数", body.get("file", "").endswith(".json") and body.get("records") == 12,
              str(body.get("file")))
        check("CORS 头回给白名单来源",
              headers.get("Access-Control-Allow-Origin") == "https://landscape-art-demo.netlify.app",
              str(headers.get("Access-Control-Allow-Origin")))
        check("带 Access-Control-Allow-Private-Network（Chrome 回环访问要求）",
              headers.get("Access-Control-Allow-Private-Network") == "true")
        files = sorted(out.glob("verdicts_*.json"))
        check("每次提交恰好新增一份文件", len(files) == saved_before + 1,
              f"{saved_before} → {len(files)}")
        check("没有留下 .tmp 半截文件", not list(out.glob("*.tmp")))
        check("latest.json 同步更新", (out / "latest.json").exists())
        # 用回执给的文件名取回那一份（sorted()[0] 可能是更早那次白名单测试写的）
        saved = json.loads((out / Path(body["file"]).name).read_text(encoding="utf-8"))
        check("文件是合法 JSON 且记录了来源", saved["schema"] == 1 and saved["kind"] == "human_verdicts"
              and saved["origin"] == "https://landscape-art-demo.netlify.app")
        check("记录条数与提交一致", len(saved["records"]) == 12)
        check("时间戳是 UTC ISO", re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", saved["received_at"]) is not None,
              saved["received_at"])

        print("\n服务端独立重算（不信任页面传上来的 summary）")
        summary = saved["summary"]
        blind = summary["modes"]["blind4"]
        check("口径与已决题数正确", blind["decided"] == 12 and summary["headline"], str(summary["headline"]))
        wins = {row["version"]: row["wins"] for row in blind["versions"]}
        check("各版本胜场与记录一致（5/4/2/1）",
              wins == {"V5b": 5, "V4": 4, "V6q": 2, "V5": 1}, str(wins))
        wilson = wilson_from_judge()
        expected = wilson(5, 12)
        row = next(r for r in blind["versions"] if r["version"] == "V5b")
        check("Wilson 区间与 tools/vlm_judge.py 同式",
              abs(row["ci"][0] - expected[0]) < 1e-3 and abs(row["ci"][1] - expected[1]) < 1e-3,
              f"{row['ci']} vs [{expected[0]:.4f}, {expected[1]:.1f}]")
        check("12 题时最高 5/12 仍判「无法区分」（区间含 25%）",
              row["indistinguishable"] is True and row["chance"] == 0.25)
        check("结论措辞与判据一致", "无法区分" in summary["headline"] or "高于随机" in summary["headline"],
              summary["headline"])
        injected = dict(base, summary={"headline": "V4 全面胜出（伪造）",
                                       "modes": {"blind4": {"decided": 999, "versions": [
                                           {"version": "V4", "wins": 999}]}}})
        code, body, _ = post(port, json.dumps(injected).encode(), "http://127.0.0.1:8799")
        check("伪造的 page summary 不覆盖服务端结论", code == 200 and body["summary"]["headline"] == summary["headline"])
        latest = json.loads((out / "latest.json").read_text(encoding="utf-8"))
        check("伪造被识别并写入 warning（按数字比对）", "不一致" in latest.get("warning", ""),
              str(latest.get("warning")))

        print("\n前端真实 payload（跨语言联调）")
        payload_path = tmp / "payload.json"
        node = shutil.which("node")
        if not node:
            print("  [skip] 没有 node，跳过 JS↔Python 联调")
        else:
            smoke = ROOT / "tools" / "smoke_versions.mjs"
            done = subprocess.run([node, str(smoke), "--emit-payload", str(payload_path)],
                                  cwd=ROOT, capture_output=True, text=True,
                                  env={**os.environ,
                                       "NODE_PATH": str(Path(tempfile.gettempdir()) / "lsart-jsdom" / "node_modules")})
            if not payload_path.exists():
                check("能从真实页面导出 payload", False, (done.stdout or "")[-300:] + (done.stderr or "")[-300:])
            else:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                check("JS 侧 payload 含 schema/records/summary",
                      payload.get("schema") == 1 and payload.get("records") and payload.get("summary"))
                check("JS 侧 payload 不含个人绝对路径/凭据",
                      not re.search(r"[A-Za-z]:\\Users|/home/[a-z]|nfp_[A-Za-z0-9]{8,}", payload_path.read_text(encoding="utf-8")))
                code, body, _ = post(port, payload_path.read_bytes(), "https://landscape-art-demo.netlify.app")
                check("Python 侧接受 JS 造的结构（否则点提交会静默失败）", code == 200 and body.get("ok"),
                      f"{code} {body}")
                if code == 200:
                    saved2 = json.loads((out / Path(body["file"]).name).read_text(encoding="utf-8"))
                    check("页面自算与服务端重算的数字一致（无 warning）", "warning" not in saved2,
                          str(saved2.get("warning")))
                    check("落盘的每份都有服务端 summary",
                          all("summary" in json.loads(f.read_text(encoding="utf-8"))
                              for f in out.glob("verdicts_*.json")))

        print("\n同一进程同时当站点服务器（页面与提交同源 ⇒ 没有 CORS/预检/混合内容）")
        def get(path: str):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
                return response.status, response.headers.get("Content-Type", ""), response.read()

        code, ctype, body = get("/versions.html")
        check("GET /versions.html 200 且是评测页", code == 200 and "text/html" in ctype
              and "版本评判台" in body.decode("utf-8", "replace"))
        code2, _, body2 = get("/versions")           # Netlify 那套干净 URL
        check("GET /versions（干净 URL）也 200 且内容一致", code2 == 200 and body2 == body)
        code, ctype, body = get("/gallery.html")
        check("GET /gallery.html 200（零 JS 静态画廊）", code == 200 and "静态版" in body.decode("utf-8", "replace"))
        code, ctype, _ = get("/data/versions.json")
        check("GET /data/versions.json 200 + application/json", code == 200 and "application/json" in ctype)
        code, ctype, body = get("/img/versions/V4/p00_s101.webp")
        check("GET 图片 200 + image/webp", code == 200 and ctype == "image/webp" and body[:4] == b"RIFF")
        code, ctype, _ = get("/js/versions.js")
        check("GET /js/versions.js 200 + JS MIME", code == 200 and "javascript" in ctype)
        for evil in ("/../tools/dev.ps1", "/%2e%2e/tools/dev.ps1", "/..%2ftools%2fdev.ps1",
                     "/img/../../tools/dev.ps1"):
            try:
                code, _, _ = get(evil)
            except urllib.error.HTTPError as error:
                code = error.code
            check(f"路径穿越被挡：{evil}", code != 200, f"HTTP {code}")
        try:
            code, _, body = get("/nope.html")
        except urllib.error.HTTPError as error:
            code, body = error.code, error.read()
        check("未知路径 404 且提示可用页面", code == 404 and b"versions.html" in body, f"HTTP {code}")

        # --no-site：只当收集器（页面另起服务时用）
        collector._state["serve_site"] = False
        try:
            try:
                code, _, _ = get("/versions.html")
            except urllib.error.HTTPError as error:
                code = error.code
            check("--no-site 时不再发页面（但仍能收提交）", code == 404, f"HTTP {code}")
            code, body, _ = post(port, json.dumps(base).encode(), "http://127.0.0.1:8799")
            check("--no-site 时提交照常工作", code == 200 and body.get("ok"), f"{code}")
        finally:
            collector._state["serve_site"] = True

        print("\n真实进程 + 非 UTF-8 控制台（回归：日志不能把请求搞挂）")
        # 2026-09-15 真浏览器 E2E 实测：从管道启动时 Python 按 cp936 输出，日志里的
        # `⇒` 无法编码 → print 抛异常 → 文件已落盘但响应发不出去 → 浏览器报 "Failed to fetch"
        # 并重试（仓库里出现两份相同的评判）。这里把那个环境原样复现出来。
        # 注意：落盘目录必须在项目内（收集器会拒绝任意路径），所以自测目录建在 research/ 下。
        selftest_root = ROOT / "research" / "_collector_selftest"
        shutil.rmtree(selftest_root, ignore_errors=True)
        proc_dir = selftest_root / "out"
        out_of_repo = subprocess.run([sys.executable, str(ROOT / "tools" / "judge_collector.py"),
                                      "--out", str(tmp / "outside")],
                                     cwd=ROOT, capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=30)
        check("拒绝把结果写到项目外（防任意路径写入）",
              out_of_repo.returncode != 0 and "必须在项目内" in (out_of_repo.stdout or "")
              + (out_of_repo.stderr or ""), f"rc={out_of_repo.returncode}")

        env = {**os.environ, "PYTHONIOENCODING": "cp936", "JUDGE_PORT": "0"}
        proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "judge_collector.py"),
                                 "--port", "8791", "--out", str(proc_dir)],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", env=env)
        try:
            ready = False
            for _ in range(40):
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8791/health", timeout=2) as response:
                        ready = response.status == 200
                        break
                except Exception:                    # noqa: BLE001
                    time.sleep(0.15)
            check("子进程收集器在非 UTF-8 控制台下能起来", ready)
            if ready:
                code, body, _ = post(8791, json.dumps(base).encode(), "http://127.0.0.1:8799")
                check("非 UTF-8 控制台下提交仍返回 200（日志不再中断响应）", code == 200 and body.get("ok"),
                      f"{code} {body}")
                saved_proc = list(proc_dir.glob("verdicts_*.json"))
                check("且恰好落盘一份（浏览器不会因失败重试）", len(saved_proc) == 1,
                      f"{len(saved_proc)} 份")
        finally:
            proc.terminate()
            try:
                out_log = proc.communicate(timeout=10)[0] or ""
            except subprocess.TimeoutExpired:
                proc.kill()
                out_log = ""
            check("日志里能看到中文（已被强制成 UTF-8）", "评判已保存" in out_log, out_log[-200:])
            shutil.rmtree(selftest_root, ignore_errors=True)
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n结果：{PASSED} 项通过，{len(FAILED)} 项失败")
    for name in FAILED:
        print(f"  - {name}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())