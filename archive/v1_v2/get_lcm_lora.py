"""get_lcm_lora.py — 下载 LCM-LoRA(SD1.5) 并检查结构（rank / target_modules），供叠加使用。"""
import os, json
from pathlib import Path
CACHE = ROOT + "/models/hf_cache"
os.environ.setdefault("HF_HOME", CACHE); os.environ.setdefault("HF_HUB_CACHE", CACHE + "/hub")
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

repo = "latent-consistency/lcm-lora-sdv1-5"
dst = Path(ROOT + "/models/lcm_lora"); dst.mkdir(parents=True, exist_ok=True)
p = hf_hub_download(repo, "pytorch_lora_weights.safetensors", local_dir=str(dst))
print("downloaded:", p, round(os.path.getsize(p)/1e6, 1), "MB")
sd = load_file(p)
keys = list(sd.keys())
print("tensor count:", len(keys))
print("sample keys:")
for k in keys[:6]:
    print("  ", k, tuple(sd[k].shape))
ups = [k for k in keys if "lora_up" in k or "lora_B" in k]
downs = [k for k in keys if "lora_down" in k or "lora_A" in k]
print("lora_up/B:", len(ups), " lora_down/A:", len(downs))
if downs:
    k = downs[0]; print("down shape:", k, tuple(sd[k].shape), "-> rank =", sd[k].shape[0])
mods = sorted({k.split(".lora_")[0].split("processor.")[-1] for k in keys})
print("modules:", len(mods)); [print("   ", m) for m in mods[:14]]
# 写出可用的 LoraConfig 摘要
info = {"repo": repo, "rank": int(sd[downs[0]].shape[0]) if downs else None,
        "alpha": 1.0, "n_tensors": len(keys), "u_keys": len(ups), "d_keys": len(downs)}
(dst / "lcm_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
print("INFO:", info)
