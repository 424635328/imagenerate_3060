"""precompute_v4.py — V4 数据预计算：从 1024 源图对每张图做 K 个随机裁剪(512) 的 latent，
并对标题编码 CLIP 文本嵌入；测试集用中心裁剪 latent 作为验证。

输出 cache: {latents:[...], text_ids:[...], texts:[...], caps:[...], test_latents:[...], test_texts:[...], empty_text:...}
每个 train latent 通过 text_ids 指向 texts（避免按裁剪重复存文本）。
"""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
import os
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))
Image_LANCZOS = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.Resampling.LANCZOS

def rand_crop(im, size, scale=(0.55, 1.0), ratio=(0.75, 1.34)):
    w, h = im.size
    for _ in range(12):
        area = w * h * random.uniform(*scale)
        ar = math.exp(random.uniform(math.log(ratio[0]), math.log(ratio[1])))
        cw, ch = int(round(math.sqrt(area*ar))), int(round(math.sqrt(area/ar)))
        if 8 <= cw <= w and 8 <= ch <= h:
            x, y = random.randint(0, w-cw), random.randint(0, h-ch)
            return im.crop((x, y, x+cw, y+ch)).resize((size, size), Image_LANCZOS)
    s = min(w, h); x, y = (w-s)//2, (h-s)//2
    return im.crop((x, y, x+s, y+s)).resize((size, size), Image_LANCZOS)

def center(im, size):
    w, h = im.size; s = min(w, h); x, y = (w-s)//2, (h-s)//2
    return im.crop((x, y, x+s, y+s)).resize((size, size), Image_LANCZOS)

@torch.no_grad()
def to_latent(im, vae, res, device):
    t = torch.from_numpy(np.asarray(im, dtype=np.float32)/255.0).permute(2, 0, 1)
    t = ((t - 0.5)/0.5).unsqueeze(0).to(device).float()
    return (vae.encode(t).latent_dist.sample() * 0.18215).squeeze(0).half().cpu()

@torch.no_grad()
def text_emb(cap, tokenizer, text_encoder, device):
    tok = tokenizer(cap, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").to(device)
    return text_encoder(tok.input_ids).last_hidden_state.squeeze(0).half().cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=ROOT + "/dataset1024")
    ap.add_argument("--out", default=ROOT + "/dataset1024/cache_v4.pt")
    ap.add_argument("--base", default="SG161222/Realistic_Vision_V6.0_B1_noVAE")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--crops", type=int, default=3)
    ap.add_argument("--cache_dir", default=ROOT + "/models/hf_cache")
    args = ap.parse_args()
    import os
    os.environ.setdefault("HF_HOME", args.cache_dir); os.environ.setdefault("HF_HUB_CACHE", args.cache_dir+"/hub")

    from diffusers import AutoencoderKL
    from transformers import CLIPTokenizer, CLIPTextModel
    dev = "cuda"
    tok = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder", torch_dtype=torch.float16).to(dev).eval()
    vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()

    man = json.loads((Path(args.data_dir)/"manifest.json").read_text(encoding="utf-8"))
    tr = [m for m in man if m["split"] == "train"]
    te_items = [m for m in man if m["split"] == "test"]
    print(f"train={len(tr)} test={len(te_items)} crops={args.crops}")

    texts, caps, latents, text_ids = [], [], [], []
    t0 = time.time()
    for i, m in enumerate(tr):
        cap = m.get("caption") or "a scenic landscape"
        im = Image.open(m["file"]).convert("RGB")
        ti = len(texts); texts.append(text_emb(cap, tok, te, dev)); caps.append(cap)
        for _ in range(args.crops):
            c = rand_crop(im, args.res)
            if random.random() < 0.5: c = c.transpose(Image.FLIP_LEFT_RIGHT)
            latents.append(to_latent(c, vae, args.res, dev)); text_ids.append(ti)
        if (i+1) % 200 == 0: print(f"  {i+1}/{len(tr)} latents={len(latents)} {time.time()-t0:.0f}s", flush=True)

    test_lat, test_txt = [], []
    for m in te_items:
        im = Image.open(m["file"]).convert("RGB")
        test_lat.append(to_latent(center(im, args.res), vae, args.res, dev))
        test_txt.append(text_emb(m.get("caption") or "a scenic landscape", tok, te, dev))
    empty = text_emb("", tok, te, dev)
    torch.save({"latents": latents, "text_ids": text_ids, "texts": texts, "caps": caps,
                "test_latents": test_lat, "test_texts": test_txt, "empty_text": empty,
                "res": args.res, "crops": args.crops}, args.out)
    print(f"saved {args.out}  train_latents={len(latents)} texts={len(texts)} test={len(test_lat)}")

if __name__ == "__main__":
    main()
