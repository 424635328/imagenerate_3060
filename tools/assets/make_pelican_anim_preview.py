import os
"""make_pelican_anim_preview.py — 用 PIL 逐帧重绘动画版鹈鹕（验证 motion 设计），输出胶片条。
模拟 pelican-anim.svg 的：呼吸浮动 / 颈头轻摆 / 眨眼 / 喉囊微胀 / 徽章高光扫过。
"""
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

OUT = Path(ROOT + "/outputs"); OUT.mkdir(parents=True, exist_ok=True)
S = 320; SS = 4; W = S*SS; u = W/64.0
def P(x, y): return (x*u, y*u)

def bez(p0, p1, p2, p3, n=40):
    o = []
    for i in range(n+1):
        t = i/n; mt = 1-t
        o.append((mt**3*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t**3*p3[0],
                  mt**3*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t**3*p3[1]))
    return o

def stroke(d, pts, width, color):
    r = width/2.0
    for (x, y) in pts:
        d.ellipse([x-r, y-r, x+r, y+r], fill=color)

def rounded_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size-1, size-1], radius=radius, fill=255)
    return m

# 与 SVG keyframes 一致的取值
def ease(t): return (1 - math.cos(2*math.pi*t))/2.0     # 0→1→0
def bob(t): return -1.5*ease(t)
def nod(t): return 1.8*math.sin(2*math.pi*t)            # SVG 顺时针为正
def pouch_s(t): return 1 + 0.05*ease(t)
def blink_s(tt):                                        # tt 秒，周期 3.6s
    ph = (tt % 3.6)/3.6
    return 0.08 if 0.90 <= ph <= 0.96 else 1.0
def sheen_x(tt):                                        # tt 秒，周期 6s
    ph = (tt % 6.0)/6.0
    if ph <= 0.56: return -60.0
    if ph >= 0.86: return 120.0
    k = (ph-0.56)/0.30
    return -60.0 + k*180.0

WHITE = (255, 255, 255, 255); SHADE = (206, 222, 243, 255)
WING = (186, 206, 234, 255); WING2 = (208, 224, 245, 255)
ORANGE = (245, 158, 11, 255); BEAK = (251, 191, 36, 255); DARK = (11, 14, 20, 255)

yy, xx = np.mgrid[0:W, 0:W]
tg = (xx + yy)/(2*W - 2)
c1 = np.array([110, 168, 255]); c2 = np.array([192, 132, 252])
GRAD = np.zeros((W, W, 4), np.uint8)
for i in range(3): GRAD[..., i] = (c1[i]*(1-tg) + c2[i]*tg).astype(np.uint8)
GRAD[..., 3] = 255
BMASK = rounded_mask(W, int(14/64*W))

def frame(tt):
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    img.paste(Image.fromarray(GRAD, "RGBA"), (0, 0), BMASK)
    d = ImageDraw.Draw(img)
    dy = bob(tt % 3.0)
    # 身体组
    body = Image.new("RGBA", (W, W), (0, 0, 0, 0)); db = ImageDraw.Draw(body)
    db.polygon([P(16, 41), P(7, 36), P(11, 41.5), P(12.5, 47.5)], fill=SHADE)
    db.ellipse([P(28.5-14.2, 43.4-10.0), P(28.5+14.2, 43.4+10.0)], fill=SHADE)
    db.ellipse([P(28.5-13.9, 43.4-10.4), P(28.5+13.9, 43.4+8.6)], fill=WHITE)
    wu = bez((17.6, 40.6), (25.0, 35.4), (33.5, 37.6), (39.0, 42.4))
    wl = bez((39.0, 42.4), (31.5, 41.6), (23.5, 41.4), (17.6, 40.6))
    db.polygon([P(x, y) for x, y in wu+wl], fill=WING)
    db.line([P(20.5, 40.4), P(28.5, 38.4), P(36.5, 41.4)], fill=WING2, width=int(2.2*u), joint="curve")
    lw = int(3.1*u)
    for a, b in [((25.0, 51.5), (23.8, 58.4)), ((33.4, 51.5), (34.6, 58.4))]:
        pa, pb = P(*a), P(*b)
        db.line([pa, pb], fill=ORANGE, width=lw)
        db.ellipse([pb[0]-lw/2, pb[1]-lw/2, pb[0]+lw/2, pb[1]+lw/2], fill=ORANGE)
    img.alpha_composite(body, (0, int(dy*u)))

    # 颈+头组（画在独立层，再旋转）
    head = Image.new("RGBA", (W, W), (0, 0, 0, 0)); dh = ImageDraw.Draw(head)
    stroke(dh, [P(x, y) for x, y in bez((24.5, 40.0), (23.2, 27.5), (30.0, 17.6), (39.6, 16.4))], int(7.6*u), WHITE)
    dh.ellipse([P(40.3-5.7, 16.2-5.7), P(40.3+5.7, 16.2+5.7)], fill=WHITE)
    ps = pouch_s(tt % 3.0); pivot = (45.0, 17.5)
    def sc(x, y): return (pivot[0] + (x-pivot[0])*ps, pivot[1] + (y-pivot[1])*ps)
    poly = [sc(44.6, 13.3), sc(62.0, 18.7)] + [sc(x, y) for (x, y) in
            bez((62.0, 18.7), (55.6, 27.4), (47.2, 25.8), (44.2, 19.4))]
    dh.polygon([P(x, y) for x, y in poly], fill=BEAK)
    dh.line([P(44.6, 13.3), P(62.0, 18.7)], fill=ORANGE, width=int(1.7*u))
    bs = blink_s(tt); er = 1.55
    dh.ellipse([P(41.7-er, 14.8-er*bs), P(41.7+er, 14.8+er*bs)], fill=DARK)
    ang = nod(tt % 3.0)
    head = head.rotate(-ang, center=P(24.5, 40.0), resample=Image.BICUBIC)
    img.alpha_composite(head, (0, int(dy*u)))

    # 高光扫过
    sh = Image.new("RGBA", (W, W), (0, 0, 0, 0)); ds = ImageDraw.Draw(sh)
    ds.rectangle([P(sheen_x(tt), -40), P(sheen_x(tt)+9, 104)], fill=(255, 255, 255, 140))
    sh = sh.rotate(-20, center=P(32, 32), resample=Image.BICUBIC)
    m = Image.new("L", (W, W), 0); m.paste(BMASK, (0, 0))
    img.paste(sh, (0, 0), Image.composite(sh.split()[3], Image.new("L", (W, W), 0), m))
    ImageDraw.Draw(img).rounded_rectangle([P(0.75, 0.75), P(63.25, 63.25)], radius=int(13.4*u),
                                          outline=(11, 14, 20, 56), width=int(1.5*u))
    return img.resize((S, S), Image.LANCZOS)

times = [0.0, 0.6, 1.2, 1.8, 2.4, 3.42, 4.8]
frames = [frame(t) for t in times]
pad, top = 12, 64
Wc = pad + len(frames)*(S+pad); Hc = top + S + 30
canvas = Image.new("RGB", (Wc, Hc), (14, 18, 28)); d = ImageDraw.Draw(canvas)
def font(sz, b=False):
    try: return ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc" if b else r"C:\Windows\Fonts\msyh.ttc", sz)
    except Exception: return ImageFont.load_default()
d.text((pad, 12), "动画版鹈鹕 — 分帧预览（对应 pelican-anim.svg 的 3s 呼吸摆动 / 3.6s 眨眼 / 6s 高光扫过）",
       font=font(17, True), fill=(232, 237, 245))
for i, (t, im) in enumerate(zip(times, frames)):
    x = pad + i*(S+pad)
    canvas.paste(im, (x, top), im)
    d.text((x+4, top+S+4), f"t={t}s", font=font(13, True), fill=(255, 220, 150))
canvas.save(OUT/"pelican_anim_frames.png")
print("saved", OUT/"pelican_anim_frames.png", canvas.size)
