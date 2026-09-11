"""make_collage.py — 生成文档封面大图与“4 组并排”附录拼贴。

输出:
  outputs/collage_cover.png   宽幅封面（标题 + 8 张精选横条）
  outputs/collage_groups.png  4 组并排（V3画廊 / 艺术风格 / 大展示 / V2基线，每组 2x3）
"""
import os
from PIL import Image, ImageDraw, ImageFont
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))
BG = (13, 16, 26)
FG = (232, 237, 245)
ACC = (110, 168, 255)
ACC2 = (192, 132, 252)
MUTED = (147, 160, 180)

def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def CJK(bold=False, size=20):
    # 优先微软雅黑（含中文），回退 Segoe UI
    paths = [r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
             r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"]
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

def pick(name, size):
    return os.path.join(ROOT, name)

def load_square(path, cell):
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    left, top = (w - s)//2, (h - s)//2
    im = im.crop((left, top, left+s, top+s)).resize((cell, cell), Image.LANCZOS)
    return im

def hgradient(w, h, c1, c2):
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    for x in range(w):
        t = x / w
        col = tuple(int(c1[i]*(1-t) + c2[i]*t) for i in range(3))
        d.line([(x,0),(x,h)], fill=col)
    return im

def draw_panel(base, panel, title, images, cell, top):
    d = ImageDraw.Draw(base)
    d.rounded_rectangle([panel[0], panel[1], panel[2], panel[3]], radius=16, fill=(21,26,36), outline=(35,44,58))
    d.text((panel[0]+18, panel[1]+14), title, font=CJK(bold=True, size=20), fill=ACC)
    d.text((panel[0]+18, panel[1]+44), "1024 / 768", font=CJK(size=12), fill=MUTED)
    gx = panel[0] + 18
    gy = panel[1] + 74
    cols = 2
    per = (160, 160)
    for i, p in enumerate(images):
        r, c = divmod(i, cols)
        im = load_square(p, per[0])
        x = gx + c*(per[0]+16)
        y = gy + r*(per[1]+16)
        base.paste(im, (x, y))
    # draw panel bg behind label area done in rounded

def main():
    t0 = CJK(bold=True, size=64)
    tsub = CJK(size=24)

    # ---- 封面 Hero 条 ----
    W, H = 1920, 780
    base = hgradient(W, H, (12, 16, 30), (26, 18, 48))
    d = ImageDraw.Draw(base)
    # 装饰圆
    for r, c in [(340,(110,168,255)),(230,(192,132,252))]:
        x, y = W-140, 90
        d.ellipse([x-r,y-r,x+r,y+r], outline=(*c,)*1 if len(c)==3 else c, width=1)
    # 标题
    d.text((110, 96), "LANDSCAPE·ART", font=t0, fill=(255,255,255))
    d.text((114, 186), "风景生图模型 · 项目说明", font=tsub, fill=ACC2)
    d.text((114, 236), "Flickr 1664 张 · RTX 3060 6GB · S D 1.5 + LoRA · V1→V3→640 · 压测 PASS",
           font=CJK(size=18), fill=MUTED)
    # 底部精选横条
    highlights = [
        "outputs/gallery_v3/v2_20260909_152400_0_prompt.png",
        "outputs/gallery_v3/v2_20260909_152754_1_prompt.png",
        "outputs/styles/v2_20260909_174243_0_rand.png",
        "outputs/styles/v2_20260909_175649_2_rand.png",
        "outputs/showcase/v2_20260909_181255_0_rand.png",
        "outputs/showcase/v2_20260909_181535_5_rand.png",
        "outputs/gallery_v3/v2_20260909_153603_3_prompt.png",
        "outputs/gallery_v3/v2_20260909_154101_4_prompt.png",
    ]
    cell = 170
    xs = 110
    y = H - cell - 90
    for i, p in enumerate(highlights):
        im = load_square(pick(p, ""), cell)
        d.rounded_rectangle([xs-3,y-3,xs+cell+3,y+cell+3], radius=14, fill=(30,36,50))
        base.paste(im, (xs, y))
        xs += cell + 26
    os.makedirs(os.path.join(ROOT,"outputs"), exist_ok=True)
    base.save(os.path.join(ROOT,"outputs","collage_cover.png"))
    print("cover -> outputs/collage_cover.png")

    # ---- 4 组并排附录 ----
    groups = [
        ("V3 高清画廊", [
            "outputs/gallery_v3/v2_20260909_152400_0_prompt.png","outputs/gallery_v3/v2_20260909_152754_1_prompt.png",
            "outputs/gallery_v3/v2_20260909_153148_2_prompt.png","outputs/gallery_v3/v2_20260909_153603_3_prompt.png",
            "outputs/gallery_v3/v2_20260909_154101_4_prompt.png","outputs/gallery_v3/v2_20260909_154519_5_prompt.png"]),
        ("艺术风格", [
            "outputs/styles/v2_20260909_174243_0_rand.png","outputs/styles/v2_20260909_174903_1_rand.png",
            "outputs/styles/v2_20260909_175649_2_rand.png","outputs/styles/v2_20260909_180331_3_rand.png",
            "outputs/styles/v2_20260909_180751_4_rand.png","outputs/styles/v2_20260909_181206_5_rand.png"]),
        ("大规模展示", [
            "outputs/showcase/v2_20260909_181255_0_rand.png","outputs/showcase/v2_20260909_181321_1_rand.png",
            "outputs/showcase/v2_20260909_181437_3_rand.png","outputs/showcase/v2_20260909_181503_4_rand.png",
            "outputs/showcase/v2_20260909_181653_7_rand.png","outputs/showcase/v2_20260909_181836_11_rand.png"]),
        ("V2 基线", [
            "outputs/gallery_v2/v2_20260909_104059_0_prompt.png","outputs/gallery_v2/v2_20260909_104522_1_prompt.png",
            "outputs/gallery_v2/v2_20260909_105147_0_prompt.png","outputs/gallery_v2/v2_20260909_105636_1_prompt.png",
            "outputs/gallery_v2/v2_20260909_110101_2_prompt.png","outputs/gallery_v2/v2_20260909_110532_3_prompt.png"]),
    ]
    cell = 160
    ppad = 10
    pw = 2*cell + 3*18 + 2*ppad
    panel_h = 74 + 3*cell + 2*16 + 24 + ppad
    W2 = 4*pw + 5*24
    H2 = panel_h + 2*40
    canvas = Image.new("RGB", (W2, H2), BG)
    x0 = 24
    for title, imgs in groups:
        draw_panel(canvas, (x0, 40, x0+pw, 40+panel_h), title, imgs, cell, 40)
        x0 += pw + 24
    canvas.save(os.path.join(ROOT,"outputs","collage_groups.png"))
    print("groups -> outputs/collage_groups.png")

if __name__ == "__main__":
    main()
