import os
"""preview_icon.py — 生成鹈鹕图标多尺寸预览条（含最近邻放大，便于检查小尺寸辨识度）。"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
SITE = Path(ROOT + "/site")
OUT = Path(ROOT + "/outputs/icon_preview.png"); OUT.parent.mkdir(parents=True, exist_ok=True)
master = Image.open(SITE/"pelican-512.png").convert("RGBA")
sizes = [16, 24, 32, 48, 64, 96, 128, 180]
pad, top = 16, 60
H = top + 200 + 30
W = pad + len(sizes)*(200+pad)
canvas = Image.new("RGB", (W, H), (16, 20, 30)); d = ImageDraw.Draw(canvas)
def font(sz, b=False):
    try: return ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc" if b else r"C:\Windows\Fonts\msyh.ttc", sz)
    except Exception: return ImageFont.load_default()
d.text((pad, 12), "鹈鹕 web 图标 — 多尺寸预览（下排为最近邻放大，看像素级辨识度）", font=font(18, True), fill=(232, 237, 245))
for i, s in enumerate(sizes):
    x = pad + i*(200+pad)
    im = master.resize((s, s), Image.LANCZOS)
    d.text((x, top-24), f"{s}px", font=font(14, True), fill=(255, 220, 150))
    canvas.paste(im, (x, top), im)
    big = im.resize((160, 160), Image.NEAREST)
    canvas.paste(big, (x, top+200-160), big)
canvas.save(OUT)
print("saved", OUT, canvas.size)
