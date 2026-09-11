"""prepare_data.py — 从 Flickr zip 提取风景照片，预处理为训练/测试集。

职责（与训练解耦，仅依赖 Pillow + 标准库）：
  1. 枚举所有 zip 内的图片条目。
  2. 逐张解码 -> 转 RGB -> 缩放到 target 尺寸（短边对齐 target 后中心裁剪成方形）。
  3. 依据文件名生成描述性 caption，并附加风格标签。
  4. 计算感知哈希（aHash），用于跨数据集去重与"生成图≠训练/测试图"校验。
  5. 按 seed 确定性划分 train / test，写入 manifest.json 与 captions 文件。
  6. 损坏图片自动跳过并记录；支持多进程并行；断点续跑（跳过已生成的条目）。

用法（在被沙箱允许的脚本目录内运行）:
  python prepare_data.py --zip-dir "<数据集源目录>" \
      --out ROOT + "/dataset" --size 512 \
      --test-frac 0.08 --workers 8 --seed 42
"""
import argparse, hashlib, json, os, re, sys, time
from collections import Counter
from pathlib import Path
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))

# --- 风格 / 场景关键词 -> 附加风格标签（用于让模型更"懂"风格 prompt）---
STYLE_MAP = {
    "sunset": "golden hour light", "sunrise": "golden hour light",
    "beach": "tropical seaside", "sea": "coastal seascape", "ocean": "coastal seascape",
    "clouds": "dramatic sky", "sky": "expansive sky", "night": "night scene",
    "rain": "lush rainforest", "forest": "lush greenery", "jungle": "lush greenery",
    "desert": "arid golden dunes", "dune": "arid golden dunes",
    "mountain": "majestic mountains", "peak": "majestic mountains",
    "waterfall": "flowing waterfall", "falls": "flowing waterfall",
    "lake": "calm lake reflection", "river": "flowing river",
    "green": "vibrant green", "red": "warm red tones", "blue": "clear blue sky",
    "castle": "historic architecture", "cathedral": "historic architecture",
    "city": "cityscape", "skyline": "cityscape", "old town": "historic town",
    "island": "tropical island", "bay": "tropical bay", "valley": "scenic valley",
    "lavender": "lavender field", "tulip": "spring flowers",
    "autumn": "autumn colours", "winter": "snowy winter", "snow": "snowy winter",
    "road": "open road", "trail": "scenic trail", "hiking": "scenic trail",
    "panorama": "panoramic view", "aerial": "aerial view", "above": "aerial view",
    "morning": "soft morning light", "dusk": "dusk lighting",
}
# 通用前缀，让模型学到一个稳定的"风景照片"概念
GENRE_PREFIX = "professional landscape photograph"

def clean_tokens(fname):
    stem = Path(fname).stem  # 去掉扩展名
    # 去掉 Flickr 数字 id（尾部的 _数字 或 纯数字段）
    stem = re.sub(r"_\d+$", "", stem)
    stem = re.sub(r"(_\d{8,})", "", stem)
    # 下划线/连字符 -> 空格
    words = re.split(r"[\s_\-]+", stem.lower())
    words = [w for w in words if w and not w.isdigit() and len(w) > 1]
    # 去重保持顺序
    seen = set()
    out = []
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out

def build_caption(fname):
    toks = clean_tokens(fname)
    scene = " ".join(toks)
    style = []
    low = " " + scene + " " + fname.lower()
    for k, v in STYLE_MAP.items():
        if k in low and v not in style:
            style.append(v)
    parts = [GENRE_PREFIX + " of " + scene] if scene else [GENRE_PREFIX]
    parts += style
    return ", ".join(parts).strip(", ")

def ahash(img, hash_size=8):
    """Average hash：缩到 hash_size^2，按均值比较比特位。"""
    g = img.convert("L").resize((hash_size, hash_size), Image_LANCZOS)
    px = list(g.getdata())
    avg = sum(px) / len(px)
    bits = "".join("1" if p >= avg else "0" for p in px)
    return bits

# 预加载 PIL 常量
from PIL import Image
Image_LANCZOS = Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.Resampling.LANCZOS

def process_one(args):
    zip_path, entry, out_path, target = args
    if os.path.exists(out_path):
        return ("skip", entry)
    try:
        import zipfile, io
        with zipfile.ZipFile(zip_path) as zf, zf.open(entry) as fh:
            data = fh.read()
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return ("error", entry, str(e))
    try:
        w, h = im.size
        # 短边对齐 target 后中心裁剪为正方形
        scale = target / min(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        im = im.resize((nw, nh), Image_LANCZOS)
        left = (nw - target) // 2
        top = (nh - target) // 2
        im = im.crop((left, top, left + target, top + target))
        im.save(out_path, "JPEG", quality=92)
    except Exception as e:
        return ("error", entry, str(e))
    return ("ok", entry)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--test-frac", type=float, default=0.08)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="调试：仅处理前 N 张")
    args = ap.parse_args()

    zip_dir = Path(args.zip_dir)
    out_dir = Path(args.out)
    train_dir = out_dir / "train"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    zips = sorted([p for p in zip_dir.glob("*.zip") if p.is_file()])
    if not zips:
        print("NO ZIPS found in", zip_dir); sys.exit(1)
    print(f"zips: {len(zips)}")

    # 收集全部条目
    entries = []  # (zip_path, entry)
    import zipfile
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                for e in zf.namelist():
                    if e.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                        entries.append((str(z), e))
        except Exception as ex:
            print("warn: cannot read zip", z, ex)
    print(f"total entries: {len(entries)}")
    if args.limit:
        entries = entries[: args.limit]

    # 确定性划分 train/test（基于条目名的稳定哈希）
    import hashlib
    split = {}
    for idx, (z, e) in enumerate(entries):
        key = hashlib.md5(e.encode("utf-8")).hexdigest()
        r = int(key[:8], 16) / 0xFFFFFFFF
        split[e] = "test" if r < args.test_frac else "train"

    # 处理（并行）
    tasks = []
    for (z, e) in entries:
        s = split[e]
        dest = (test_dir if s == "test" else train_dir) / f"{hashlib.md5(e.encode()).hexdigest()[:12]}.jpg"
        tasks.append((z, e, str(dest), args.size))
    t0 = time.time()
    stats = Counter()
    if args.workers > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.Pool(processes=args.workers) as pool:
            for res in pool.imap_unordered(process_one, tasks, chunksize=4):
                stats[res[0]] += 1
                if res[0] == "error":
                    print("ERR", res[1], res[2])
    else:
        for t in tasks:
            res = process_one(t)
            stats[res[0]] += 1
            if res[0] == "error":
                print("ERR", res[1], res[2])
    print("processing done in %.1fs, stats=%s" % (time.time() - t0, dict(stats)))

    # 汇总 manifest：文件 + caption + 感知哈希 + split
    manifest = []
    _lenmap = {"train": train_dir, "test": test_dir}
    for (z, e) in entries:
        s = split[e]
        dest = (test_dir if s == "test" else train_dir) / f"{hashlib.md5(e.encode()).hexdigest()[:12]}.jpg"
        if not dest.exists():
            continue
        cap = build_caption(e)
        try:
            im = Image.open(dest)
            h = ahash(im)
        except Exception:
            h = ""
        manifest.append({
            "zip": os.path.basename(z),
            "entry": e,
            "file": str(dest),
            "split": s,
            "caption": cap,
            "ahash": h,
        })
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    # 便捷：训练 caption 列表
    with open(out_dir / "captions.txt", "w", encoding="utf-8") as f:
        for m in manifest:
            f.write(m["caption"] + "\t" + m["file"] + "\n")
    print("manifest entries:", len(manifest))
    print("train:", sum(1 for m in manifest if m["split"] == "train"),
          " test:", sum(1 for m in manifest if m["split"] == "test"))
    # 样例 caption
    print("\n=== sample captions ===")
    for m in manifest[:8]:
        print(m["caption"])

if __name__ == "__main__":
    main()
