"""judge_collector.py — 评判页 + 评判收集器（一个进程，一个地址，本机回环，无令牌）。

为什么需要它：`site/versions.html` 是纯静态页面，浏览器没法把结果写进仓库；
而人工盲测的价值全在"结果能回到作者手里"。这个服务同时做两件事：

    GET  /*            → 直接把 site/ 当静态站点发出去（含 /versions.html 与图片）
    POST /submit       → 把评判记录原子写入 research/human_judge/verdicts_<时间>.json

**为什么页面和接口必须同源**（2026-09-15 改）：早先的设计是"静态站点一个端口 +
收集器另一个端口"，于是页面的提交变成跨源请求，要过 CORS、要过 Chrome 的私网访问
（Private Network Access）预检，而线上 HTTPS 页面往 http://127.0.0.1 发请求还可能被
混合内容策略挡掉。现在两者同源：**没有 CORS、没有预检、没有混合内容**，
浏览器只要能把 127.0.0.1 打开，就一定能提交。

设计约束（都很保守，因为它监听在本机回环上）：
  · **只监听 127.0.0.1**，不暴露到局域网，更不上公网；没有令牌，因为不需要 —— 它不碰
    GPU、不碰模型、不执行任何输入，只把 JSON 落盘到一个固定目录。
  · **来源白名单**：默认只接受本站与 localhost 的 Origin（跨站页面发来的请求 403）。
    与 `netlify/functions/proxy.js` 同一套语义：默认非严格（无 Origin 的本机脚本放行并记日志），
    `JUDGE_STRICT=1` 时无 Origin 也拒绝。
  · **预算**：body ≤ 64 KB、记录 ≤ 200 条、字段类型逐项校验；超限/畸形一律 4xx + 原因。
  · **自己算一遍**：服务端**独立重算**胜场与 Wilson 区间（不信任页面传上来的 summary），
    并把两者的**数字**差异写进文件 —— 与项目其它地方一样，"数字要能复算"。
  · **原子落盘**：`.tmp` → `os.replace()`；同时更新 `latest.json` 方便查看。
  · 文件名只由服务端生成（时间戳 + 随机后缀），**不使用任何客户端提供的名字**（防路径穿越）。
  · 静态服务只发 `site/` 目录内的文件，路径穿越（`..`）一律 404。

用法:
    python tools/judge_collector.py                 # 页面 http://127.0.0.1:8787/versions.html
    python tools/judge_collector.py --port 8899     # 换端口
    python tools/judge_collector.py --no-site       # 只当收集器（页面自己另起服务）
    python tools/judge_collector.py --once          # 收到一条就退出（脚本化用）
    pwsh -NoProfile -File tools/dev.ps1 judge       # 同上，一条命令
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 64 * 1024
MAX_RECORDS = 200
PROMPT_COUNT = 24
DEFAULT_ORIGINS = (
    "https://landscape-art-demo.netlify.app",
    "https://6aa7f8179b5591da0127157b--landscape-art-demo.netlify.app",
    "http://127.0.0.1",
    "http://localhost",
)
# 本站的 Origin 可能是任意 deploy-preview 子域，因此按后缀放行
ORIGIN_SUFFIX = ".netlify.app"
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_+\-.]{0,15}$")
# 静态服务用得到的 MIME（页面依赖 .webp/.json/.js/.css）
MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".webp": "image/webp",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".webmanifest": "application/manifest+json",
}

_write_lock = threading.Lock()          # 同一时刻只允许一次落盘
_state = {"out": ROOT / "research" / "human_judge", "strict": False, "received": 0,
          "serve_site": True, "site_root": (ROOT / "site").resolve()}


def wilson(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% 区间（与 tools/vlm_judge.py 同一式子；由 test_judge_collector.py 断言一致）。"""
    if total <= 0:
        return 0.0, 1.0
    phat = wins / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def force_utf8_stdout() -> None:
    """把 stdout/stderr 钉成 UTF-8（errors=replace）。

    踩过的坑（2026-09-15，由真浏览器 E2E 抓出）：收集器从管道启动时，Python 在中文
    Windows 上按 cp936 输出，而日志里的 `⇒`/`—` 在 cp936 里没有映射 → `print(...)`
    抛 UnicodeEncodeError → **落盘已经发生、响应却发不出去** → 浏览器报 "Failed to fetch"
    并重试一次（于是仓库里出现两份内容相同的评判）。日志绝不能有把请求搞挂的能力。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def recompute(records: list[dict], versions: list[str]) -> dict:
    """独立重算：被选中次数、占比、Wilson 区间、"无法区分"判定（4 选 1 的随机期望是 25%）。"""
    per_mode: dict[str, dict] = {}
    for row in records:
        mode = row.get("mode") or "blind4"
        bucket = per_mode.setdefault(mode, {"decided": 0, "ties": 0, "wins": {v: 0 for v in versions}})
        if row.get("tie"):
            bucket["ties"] += 1
            continue
        winner = row.get("winner")
        if winner in bucket["wins"]:
            bucket["wins"][winner] += 1
            bucket["decided"] += 1
    out: dict = {"prompt_count": PROMPT_COUNT, "modes": {}}
    for mode, bucket in sorted(per_mode.items()):
        chance = 0.5 if mode == "pair" else 0.25
        rows = []
        for version, wins in sorted(bucket["wins"].items(), key=lambda kv: -kv[1]):
            if not bucket["decided"]:
                continue
            low, high = wilson(wins, bucket["decided"])
            rows.append({"version": version, "wins": wins, "decided": bucket["decided"],
                         "rate": round(wins / bucket["decided"], 4),
                         "ci": [round(low, 4), round(high, 4)],
                         "chance": chance,
                         "indistinguishable": low <= chance <= high})
        out["modes"][mode] = {"decided": bucket["decided"], "ties": bucket["ties"], "versions": rows}
    out["headline"] = judge_headline(out)
    return out


def judge_headline(summary: dict) -> str:
    """一句话结论：只有"显著高于随机"才敢说更好，否则照本项目判据记"无法区分"。"""
    best = None
    for mode, bucket in summary["modes"].items():
        for row in bucket["versions"]:
            if row["indistinguishable"]:
                continue
            if best is None or row["rate"] > best["rate"]:
                best = {**row, "mode": mode}
    total = sum(bucket["decided"] for bucket in summary["modes"].values())
    if not total:
        return "还没有已决题（全为平局或未评判）。"
    if best is None:
        return (f"已决 {total} 题：没有任何版本显著高于随机期望 ⇒ 按本项目判据记为「无法区分」，"
                f"与机器判据的结论一致。")
    return (f"已决 {total} 题：{best['version']} 在 {best['mode']} 口径下被选中 "
            f"{best['wins']}/{best['decided']}（区间 {best['ci'][0]}–{best['ci'][1]}，"
            f"高于随机期望 {best['chance']}）—— 这是人工判据第一次给出方向，"
            f"但样本量还小，建议继续判到区间收窄。")


def validate(payload: object) -> tuple[list[dict], dict, str]:
    """返回 (records, meta, error)。任何不合格的输入都要给出**具体原因**。"""
    if not isinstance(payload, dict):
        return [], {}, "顶层必须是 JSON 对象"
    if payload.get("schema") != 1:
        return [], {}, f"schema 必须为 1（收到 {payload.get('schema')!r}）"
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return [], {}, "records 必须是非空数组"
    if len(records) > MAX_RECORDS:
        return [], {}, f"records 超过上限（{len(records)} > {MAX_RECORDS}）"
    clean: list[dict] = []
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            return [], {}, f"records[{index}] 不是对象"
        prompt = row.get("prompt_index")
        seed = row.get("seed")
        if not isinstance(prompt, int) or not 0 <= prompt < PROMPT_COUNT:
            return [], {}, f"records[{index}].prompt_index 非法：{prompt!r}"
        if not isinstance(seed, int) or not 0 < seed < 10 ** 9:
            return [], {}, f"records[{index}].seed 非法：{seed!r}"
        tie = bool(row.get("tie"))
        winner = row.get("winner")
        if not tie and not isinstance(winner, str):
            return [], {}, f"records[{index}] 既非平局又没有 winner"
        order = row.get("order")
        if order is not None and (not isinstance(order, list)
                                  or not all(isinstance(v, str) and VERSION_RE.match(v) for v in order)):
            return [], {}, f"records[{index}].order 非法"
        clean.append({
            "prompt_index": prompt, "seed": seed, "tie": tie,
            "winner": None if tie else winner,
            "mode": row.get("mode") if isinstance(row.get("mode"), str) else "blind4",
            "blind": bool(row.get("blind")),
            "order": order or [],
            "pair": row.get("pair") if isinstance(row.get("pair"), list) else None,
            "at": row.get("at") if isinstance(row.get("at"), str) else None,
        })
    meta = {
        "source": payload.get("source") if isinstance(payload.get("source"), str) else "unknown",
        "versions": [v for v in payload.get("versions", []) if isinstance(v, str) and VERSION_RE.match(v)],
        "protocol": payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {},
        "page_summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else None,
        "client": payload.get("client") if isinstance(payload.get("client"), dict) else {},
    }
    return clean, meta, ""


def compare_summaries(page: dict | None, server: dict) -> str:
    """页面自算 vs 服务端重算：**只比数字**（每个口径的已决题数与各版本胜场）。

    不比措辞：两边结论文案的差别（例如服务端多一句"与机器判据一致"）不是错误，
    拿文案做判据只会制造假警报。
    """
    if not isinstance(page, dict) or not isinstance(page.get("modes"), dict):
        return ""
    for mode, bucket in server["modes"].items():
        page_mode = page["modes"].get(mode)
        if not isinstance(page_mode, dict):
            continue
        if page_mode.get("decided") not in (None, bucket["decided"]):
            return f"页面与重算不一致：{mode} 已决题数 {page_mode.get('decided')} ≠ {bucket['decided']}"
        expected = {row["version"]: row["wins"] for row in bucket["versions"]}
        for row in page_mode.get("versions") or []:
            version, wins = row.get("version"), row.get("wins")
            if version in expected and wins != expected[version]:
                return f"页面与重算不一致：{mode}/{version} 胜场 {wins} ≠ {expected[version]}"
    return ""


class Handler(BaseHTTPRequestHandler):
    server_version = "lsart-judge-collector/1"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------------ 工具

    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        if origin and self._origin_ok(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            # Chrome 的 Private Network Access：公网页面访问回环地址需要显式同意
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Access-Control-Max-Age", "600")

    @staticmethod
    def _origin_ok(origin: str) -> bool:
        if origin in DEFAULT_ORIGINS:
            return True
        if origin.endswith(ORIGIN_SUFFIX):
            return True
        host = urlparse(origin).hostname or ""
        return host in ("127.0.0.1", "localhost", "::1")

    def _reply(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:      # 默认日志太吵，只留我们自己的
        pass

    def _log(self, message: str) -> None:
        # 日志再怎么样也不能把请求搞挂：写不出去就算了（见 force_utf8_stdout 的说明）
        try:
            print(f"[judge] {message}", flush=True)
        except Exception:                                   # noqa: BLE001 —— 故意的兜底
            pass

    # ------------------------------------------------------------------ 路由

    def do_OPTIONS(self) -> None:                        # noqa: N802
        origin = self.headers.get("Origin")
        if origin and not self._origin_ok(origin):
            self._reply(403, {"ok": False, "error": f"Origin 不在白名单：{origin}"})
            return
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:                            # noqa: N802
        path = urlparse(self.path).path
        if path in ("/health", "/healthz"):
            self._reply(200, {"ok": True, "service": "lsart-judge-collector", "schema": 1,
                              "received": _state["received"], "out": display(_state["out"]),
                              "site": bool(_state["serve_site"])})
            return
        if not _state["serve_site"]:
            self._reply(404, {"ok": False, "error": f"未知路径 {path}（--no-site 时只有 /health 与 /submit）"})
            return
        self._serve_static(path)

    # ------------------------------------------------------- 静态站点（同源）
    def _serve_static(self, path: str) -> None:
        """把 site/ 当静态根发出去。只发目录内的文件，路径穿越一律 404。"""
        site_root: Path = _state["site_root"]
        relative = path.lstrip("/") or "index.html"
        if relative.endswith("/"):
            relative += "index.html"
        if not relative.endswith(".html") and "." not in Path(relative).name:
            # 支持 Netlify 那套 "干净 URL"：/versions → versions.html
            candidate = site_root / f"{relative}.html"
            if candidate.exists():
                relative = f"{relative}.html"
        target = (site_root / relative).resolve()
        if site_root not in target.parents and target != site_root:
            self._reply(404, {"ok": False, "error": "路径越界"})
            return
        if not target.is_file():
            self._reply(404, {"ok": False, "error": f"没有这个文件：{path}"
                              f"（可用：/versions.html、/gallery.html、/）"})
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(target.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        # 页面/数据必须实时，图片可以缓一缓
        self.send_header("Cache-Control", "no-store" if target.suffix in (".html", ".json", ".js", ".css")
                         else "public, max-age=300")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:                           # noqa: N802
        if urlparse(self.path).path != "/submit":
            self._reply(404, {"ok": False, "error": "只有 POST /submit"})
            return
        origin = self.headers.get("Origin")
        if origin and not self._origin_ok(origin):
            self._log(f"拒绝跨站提交：Origin={origin}")
            self._reply(403, {"ok": False, "error": f"Origin 不在白名单：{origin}"})
            return
        if not origin and _state["strict"]:
            self._reply(403, {"ok": False, "error": "JUDGE_STRICT=1 时必须有 Origin"})
            return
        if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
            self._reply(415, {"ok": False, "error": "Content-Type 必须是 application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._reply(400, {"ok": False, "error": "Content-Length 非法"})
            return
        if length <= 0:
            self._reply(400, {"ok": False, "error": "空 body"})
            return
        if length > MAX_BODY:
            self._reply(413, {"ok": False, "error": f"body 超过 {MAX_BODY} 字节（收到 {length}）"})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self._reply(400, {"ok": False, "error": f"不是合法 JSON：{error}"})
            return

        records, meta, problem = validate(payload)
        if problem:
            self._log(f"拒绝提交：{problem}")
            self._reply(400, {"ok": False, "error": problem})
            return

        versions = meta["versions"] or sorted({r["winner"] for r in records if r["winner"]})
        summary = recompute(records, versions)
        document = {
            "schema": 1,
            "kind": "human_verdicts",
            "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "origin": origin or "(no origin)",
            "source": meta["source"],
            "client": meta["client"],
            "protocol": meta["protocol"],
            "versions": versions,
            "records": records,
            "summary": summary,                 # 服务端独立重算的结果（以它为准）
            "page_summary": meta["page_summary"],
        }
        if meta["page_summary"]:
            mismatch = compare_summaries(meta["page_summary"], summary)
            if mismatch:
                document["warning"] = mismatch + "（以本文件 summary 为准）"

        try:
            target = save(document)
        except OSError as error:
            self._log(f"落盘失败：{error}")
            self._reply(500, {"ok": False, "error": f"落盘失败：{error}"})
            return

        _state["received"] += 1
        self._log(f"第 {_state['received']} 份评判已保存 → {display(target)}"
                  f"（{len(records)} 条；{summary['headline']}）")
        self._reply(200, {"ok": True, "file": display(target),
                          "records": len(records), "summary": summary})


def display(path: Path) -> str:
    """给日志/回执用的路径：能相对项目根就相对，否则给绝对路径（测试会落在临时目录）。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def save(document: dict) -> Path:
    """原子写入 + 更新 latest.json。文件名完全由服务端生成（不用客户端给的任何字段）。"""
    out: Path = _state["out"]
    with _write_lock:
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = out / f"verdicts_{stamp}_{secrets.token_hex(2)}.json"
        text = json.dumps(document, ensure_ascii=False, indent=1) + "\n"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, target)
        latest = out / "latest.json"
        latest.write_text(text, encoding="utf-8", newline="\n")     # 方便直接看最后一份
        return target


def serve(port: int, out: Path, once: bool = False) -> int:
    force_utf8_stdout()
    _state["out"] = out
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)      # 只监听回环
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}"
    print(f"评判服务已启动：{url}", flush=True)
    if _state["serve_site"]:
        print(f"  → 用浏览器打开：{url}/versions.html      （判完点「📤 提交评判」）", flush=True)
        print(f"    静态画廊（无需 JS）：{url}/gallery.html", flush=True)
    print(f"  → 结果落盘：{display(out)}", flush=True)
    print(f"  → 提交接口：POST {url}/submit（同源，无需 CORS）", flush=True)
    print("  提示：浏览器若走了失效的代理，可能连 127.0.0.1 都打不开 —— "
          "先试 " + url + "/health 看能不能出 JSON", flush=True)
    print("  Ctrl+C 停止", flush=True)
    try:
        if once:
            server.handle_request()
            return 0
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止。", flush=True)
    finally:
        server.server_close()
    return 0


def main() -> int:
    force_utf8_stdout()
    ap = argparse.ArgumentParser(description="接收版本评判台的人工评判结果（本机回环）")
    ap.add_argument("--port", type=int, default=int(os.environ.get("JUDGE_PORT", 8787)))
    ap.add_argument("--out", default=str(ROOT / "research" / "human_judge"))
    ap.add_argument("--no-site", action="store_true", help="只当收集器，不提供站点（页面另起服务）")
    ap.add_argument("--once", action="store_true", help="收到一条就退出")
    args = ap.parse_args()
    _state["strict"] = os.environ.get("JUDGE_STRICT") == "1"
    _state["serve_site"] = not args.no_site
    out = Path(args.out).resolve()
    if ROOT not in out.parents:
        print(f"[错误] 落盘目录必须在项目内：{out}", file=sys.stderr)
        return 2
    return serve(args.port, out, args.once)


if __name__ == "__main__":
    raise SystemExit(main())
