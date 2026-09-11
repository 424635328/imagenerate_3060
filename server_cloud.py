"""server_cloud.py — 云端 GPU 后端适配器（跑 SDXL/FLUX 等大模型），API 与 server.py 完全一致，
  这样 Netlify 的 Function 代理只需把 BACKEND_URL 指向本服务即可（前端零改动）。

用途：本机 6GB 跑不动 SDXL/FLUX。把本服务部署到任意有公网地址的容器平台（Render / Fly / Cloud Run 等），
     由它调用 Replicate 的云端 GPU 生成，再用 /result /chunk 回传图片。

环境变量：
  REPLICATE_API_TOKEN   必填（https://replicate.com/account/api-tokens）
  REPLICATE_MODEL       形如 "owner/name:version" 或 "owner/name"（默认 SDXL）
  API_TOKEN             与前端/代理一致的鉴权密钥（必填，无默认值）
  MAX_JOBS / MAX_QUEUE / RATE_PER_MIN / JOB_TIMEOUT / RESULTS_DIR / PORT
"""
import io, os, threading, time, uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
import requests
from fastapi import FastAPI, Header, HTTPException, Depends, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from config import clamp_cfg, clamp_steps, OUT_QUALITY, OUT_FORMAT
API_TOKEN = os.environ.get("API_TOKEN", "change-me")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")
REPLICATE_MODEL = os.environ.get("REPLICATE_MODEL", "stability-ai/sdxl")
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "20"))
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "8"))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "900"))
MAX_JOBS = int(os.environ.get("MAX_JOBS", "500"))
PORT = int(os.environ.get("PORT", "8000"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(Path(__file__).resolve().parent / "results")))

@asynccontextmanager
async def lifespan(_app):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, daemon=True).start()
    yield


app = FastAPI(title="LANDSCAPE·ART 云端后端（Replicate）", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_jobs = {}
_pending = deque()
_cv = threading.Condition()
_rate = defaultdict(deque)
_total = 0

class GenerateReq(BaseModel):
    prompt: str = ""
    style: str = "写实摄影"
    res: str = "512"
    aspect: str = "方"
    steps: int = 40
    cfg: float = 7.5
    seed: int = -1
    neg: str = "blurry, low quality, watermark, text, distorted, oversaturated"
    upscale: bool = False
    highres: int = 0
    enhance: int = 0
    sr_model: str = "ultrasharp"
    enhance_strength: float = 0.30
    enhance_steps: int = 0
    fast: bool = False
    sampler: str = "dpmpp2m_karras"
    # Accepted for API parity with server.py. The cloud path renders one image per
    # prediction, so a batch request is served as a single image (count stays 1).
    batch: int = 1
    # Parity with server.py; the cloud path forwards it as image prompt input.
    init_image: str = ""
    strength: float = 0.55

class WarmupReq(BaseModel):
    fast: bool = False
    sampler: str = "dpmpp2m_karras"
def _key(x_key: str = Header(default="", alias="X-API-Key")):
    if x_key != API_TOKEN: raise HTTPException(401, "invalid API key")

def _rate_ok(ip):
    now = time.time(); d = _rate[ip]
    while d and d[0] < now - 60: d.popleft()
    if len(d) >= RATE_PER_MIN: return False
    d.append(now); return True

def _size(res, aspect):
    res = int(res)
    if aspect == "横": return int(res*1.5), res
    if aspect == "竖": return res, int(res*1.5)
    return res, res

def _prune():
    if len(_jobs) <= MAX_JOBS: return
    for jid in sorted(_jobs, key=lambda x: _jobs[x]["created"])[:len(_jobs)-MAX_JOBS]:
        j = _jobs.pop(jid, None)
        if j and j.get("file"):
            try: Path(j["file"]).unlink(missing_ok=True)
            except Exception: pass

def _replicate_run(req: GenerateReq, job_id):
    """调用 Replicate 生成，下载结果图落盘。"""
    w, h = _size(req.res, req.aspect)
    target = int(req.enhance or req.highres or 0)
    fast = bool(req.fast or req.sampler.lower() in {"lcm", "tcd"})
    steps = clamp_steps(req.steps, fast=fast)
    cfg = clamp_cfg(req.cfg, fast=fast)
    inp = {"prompt": (req.prompt or "a scenic landscape"),
           "negative_prompt": req.neg, "num_inference_steps": steps,
           "guidance_scale": cfg, "width": w, "height": h}
    if req.seed >= 0: inp["seed"] = int(req.seed)
    if target: inp["width"], inp["height"] = target, target
    if getattr(req, "init_image", ""):
        payload = req.init_image
        if len(payload) > 8 * 1024 * 1024: raise RuntimeError("reference image too large")
        inp["image"] = payload if payload.startswith("data:") else f"data:image/webp;base64,{payload}"
        inp["prompt_strength"] = max(0.05, min(float(req.strength), 0.95))
    heads = {"Authorization": f"Token {REPLICATE_API_TOKEN}", "Content-Type": "application/json"}
    url = "https://api.replicate.com/v1/predictions"
    body = {"input": inp}
    if ":" in REPLICATE_MODEL:
        body["version"] = REPLICATE_MODEL.split(":", 1)[1]
    else:
        url = f"https://api.replicate.com/v1/models/{REPLICATE_MODEL}/predictions"
    r = requests.post(url, json=body, headers=heads, timeout=60); r.raise_for_status()
    pred = r.json()
    pid = pred["id"]; t0 = time.time()
    while True:
        if time.time() - t0 > JOB_TIMEOUT: raise RuntimeError("cloud timeout")
        time.sleep(3)
        pr = requests.get(f"https://api.replicate.com/v1/predictions/{pid}", headers=heads, timeout=60).json()
        st = pr.get("status")
        if st == "succeeded":
            out = pr.get("output")
            u = out[0] if isinstance(out, list) else out
            img = requests.get(u, timeout=180).content
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            fp = RESULTS_DIR / f"{job_id}.webp"
            # 统一转 WebP（原图可能是 PNG/JPEG），降低代理回传体积。
            try:
                from PIL import Image
                Image.open(io.BytesIO(img)).convert("RGB").save(fp, "WEBP", quality=OUT_QUALITY, method=6)
            except Exception:
                fp.write_bytes(img)
            return str(fp)
        if st in ("failed", "canceled"): raise RuntimeError(pr.get("error") or st)

def _worker():
    while True:
        with _cv:
            while not _pending: _cv.wait()
            jid = _pending.popleft()
        job = _jobs.get(jid)
        if not job: continue
        job["status"] = "running"
        try:
            job["file"] = _replicate_run(job["req"], jid); job["status"] = "done"
        except Exception as e:
            job["status"] = "failed"; job["error"] = str(e)

@app.get("/health")
def health():
    running = sum(1 for j in _jobs.values() if j["status"] == "running")
    done = sum(1 for j in _jobs.values() if j["status"] == "done")
    failed = sum(1 for j in _jobs.values() if j["status"] == "failed")
    q = len(_pending)
    return {"ok": True, "backend": "replicate:" + REPLICATE_MODEL, "format": OUT_FORMAT,
            "total_submitted": _total,
            "queued": q, "running": running, "completed": done, "failed": failed, "queue": q, "jobs": _total}

@app.post("/generate", dependencies=[Depends(_key)])
def gen(req: GenerateReq, request: Request):
    global _total
    if not REPLICATE_API_TOKEN: raise HTTPException(500, "REPLICATE_API_TOKEN not set")
    if not _rate_ok(request.client.host): raise HTTPException(429, "rate limited")
    if len(_pending) >= MAX_QUEUE: raise HTTPException(429, "queue full, try later")
    jid = uuid.uuid4().hex[:12]; _total += 1
    _jobs[jid] = {"id": jid, "status": "queued", "req": req, "created": time.time(), "file": None,
                  "error": None, "count": 1, "seeds": [req.seed], "cached": False}
    _prune()
    with _cv: _pending.append(jid); _cv.notify()
    return {"job_id": jid, "status": "queued", "queue_position": len(_pending),
            "count": 1, "seeds": [req.seed]}

@app.post("/warmup", dependencies=[Depends(_key)])
def do_warmup(req: WarmupReq):
    """Cloud GPUs are provisioned per request; nothing to preload locally."""
    return {"loaded": False, "seconds": 0.0, "fast": req.fast, "sampler": req.sampler,
            "note": "cloud backend has no local pipeline"}

@app.post("/gc", dependencies=[Depends(_key)])
def do_gc(unload: bool = False):
    _prune()
    return {"ok": True, "tracked_jobs": len(_jobs), "unload": unload}

@app.get("/preview/{jid}")
def preview(jid: str):
    raise HTTPException(404, "preview not supported on the cloud backend")

@app.get("/jobs/{jid}")
def job(jid: str):
    j = _jobs.get(jid)
    if not j: raise HTTPException(404, "not found")
    sz = Path(j["file"]).stat().st_size if (j["status"] == "done" and j.get("file") and Path(j["file"]).exists()) else 0
    steps = clamp_steps(j["req"].steps, fast=bool(j["req"].fast or j["req"].sampler.lower() in {"lcm", "tcd"}))
    done = steps if j["status"] == "done" else 0
    return {"id": jid, "status": j["status"], "error": j["error"], "bytes": sz,
            "prompt": j["req"].prompt, "style": j["req"].style, "res": j["req"].res,
            "aspect": j["req"].aspect, "seed": j["req"].seed,
            "upscale": j["req"].upscale, "highres": j["req"].highres, "enhance": j["req"].enhance,
            "sampler": j["req"].sampler, "fast": j["req"].fast, "steps": steps,
            "cfg": j["req"].cfg, "format": OUT_FORMAT, "stage": j["status"],
            "steps_done": done, "steps_total": steps,
            "progress": 1.0 if j["status"] == "done" else 0.0,
            "eta": 0, "elapsed": round(time.time() - j["created"], 1),
            "preview_ready": False, "preview_rev": 0, "preview_kind": None,
            "has_init": bool(getattr(j["req"], "init_image", "")),
            # batch/cache parity with server.py (cloud renders one image per job)
            "count": 1, "seeds": [j["req"].seed], "files": 1 if sz else 0,
            "bytes_each": [sz], "cached": bool(j.get("cached"))}

@app.get("/result/{jid}")
def result(jid: str, i: int = 0):
    """`i` exists for API parity; the cloud backend stores a single image."""
    j = _jobs.get(jid)
    if not j: raise HTTPException(404, "not found")
    if j["status"] != "done" or not j.get("file") or not Path(j["file"]).exists():
        return JSONResponse({"error": "result not ready"}, status_code=409)
    return FileResponse(j["file"], media_type="image/webp", headers={
        "Cache-Control": "public, max-age=31536000, immutable",
        "X-Job-Count": "1", "X-Job-Index": "0"})

@app.get("/chunk/{jid}")
def chunk(jid: str, off: int = 0, length: int = 2000000, i: int = 0):
    j = _jobs.get(jid)
    if not j or j["status"] != "done" or not j.get("file") or not Path(j["file"]).exists():
        raise HTTPException(404, "not ready")
    fp = Path(j["file"]); total = fp.stat().st_size
    off = max(0, min(int(off), total)); ln = max(1, min(int(length), 4000000, total - off))
    with open(fp, "rb") as f:
        f.seek(off); data = f.read(ln)
    return Response(content=data, media_type="application/octet-stream",
                    headers={"X-Total": str(total), "X-Job-Count": "1", "X-Job-Index": "0",
                             "Cache-Control": "public, max-age=31536000, immutable"})

if __name__ == "__main__":
    import uvicorn
    print(f"cloud backend on :{PORT}  model={REPLICATE_MODEL}  token_set={bool(REPLICATE_API_TOKEN)}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
