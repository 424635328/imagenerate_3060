"""normalize_base.py — 把 HF 缓存里的基础模型转换为本地 safetensors 干净目录。

解决三类告警 / 问题：
  1) "no file named diffusion_pytorch_model.safetensors ... Defaulting to unsafe serialization"
     —— 仓库只提供 .bin(pickle)，转为 safetensors 后既无告警也更安全。
  2) "CLIPFeatureExtractor appears to have been deprecated"
     —— 从 model_index.json 里移除 feature_extractor 条目（txt2img 用不到它）。
  3) 反复向 HF 发 HEAD 请求导致的 "unauthenticated requests" 告警与 SSL 重试
     —— 之后配合 HF_HUB_OFFLINE=1 完全离线加载。

用法: python normalize_base.py  [--src <repo_id|path>] [--out <dir>]
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="SG161222/Realistic_Vision_V6.0_B1_noVAE")
    ap.add_argument("--out", default=ROOT + "/models/base_rv6")
    ap.add_argument("--cache", default=ROOT + "/models/hf_cache")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    a = ap.parse_args()
    os.environ.setdefault("HF_HOME", a.cache)
    os.environ.setdefault("HF_HUB_CACHE", a.cache + "/hub")

    import torch
    from diffusers import StableDiffusionPipeline

    out = Path(a.out)
    if out.exists():
        print("out dir exists, removing:", out)
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    t0 = time.time()
    print(f"[1/4] loading pipeline from {a.src} (dtype={a.dtype}, cpu)...", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        a.src, dtype=dt, safety_checker=None, requires_safety_checker=False)
    print(f"      loaded in {time.time()-t0:.1f}s", flush=True)

    print("[2/4] saving as safetensors ->", out, flush=True)
    pipe.save_pretrained(str(out), safe_serialization=True, max_shard_size="4GB")

    print("[3/4] pruning unused entries (feature_extractor / safety_checker)...", flush=True)
    mi_path = out / "model_index.json"
    mi = json.loads(mi_path.read_text(encoding="utf-8"))
    for k in ("feature_extractor", "safety_checker"):
        if k in mi:
            mi.pop(k); print("      removed", k)
    # 清理已保存的目录
    for k in ("feature_extractor", "safety_checker"):
        d = out / k
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True); print("      deleted dir", d.name)
    # 记录来源，便于追溯
    mi["_normalized_from"] = a.src
    mi["_normalized_dtype"] = a.dtype
    mi_path.write_text(json.dumps(mi, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[4/4] result:", flush=True)
    total = 0
    for p in sorted(out.rglob("*")):
        if p.is_file():
            mb = p.stat().st_size / 1e6; total += mb
            print(f"      {p.relative_to(out)!s:56s} {mb:8.1f} MB")
    print(f"      TOTAL {total:.1f} MB   elapsed {time.time()-t0:.1f}s")
    print("OK ->", out)

if __name__ == "__main__":
    main()
