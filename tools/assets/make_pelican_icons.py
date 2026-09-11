import os
"""make_pelican_icons.py — 用 PIL 绘制鹈鹕图标（与 site/pelican.svg 同款），输出多尺寸 PNG。"""
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

OUT = Path(ROOT + "/site")
S = 512; SS = 4; W = S*SS; u = W/64.0
def P(x, y): return (x*u, y*u)

def bez(p0, p1, p2, p3, n=40):
    out = []
    for i in range(n+1):
        t = i/n; mt = 1-t
        out.append((mt**3*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t**3*p3[0],
                    mt**3*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t**3*p3[1]))
    return out

def stroke(d, pts, width, color):
    """沿采样点画圆实现平滑粗线（圆头圆角）。"""
    r = width/2.0
    for (x, y) in pts:
        d.ellipse([x-r, y-r, x+r, y+r], fill=color)

def rounded_mask(size, radius):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size-1, size-1], radius=radius, fill=255)
    return m

# 渐变底
yy, xx = np.mgrid[0:W, 0:W]
t = (xx + yy)/(2*W - 2)
c1 = np.array([110, 168, 255]); c2 = np.array([192, 132, 252])
grad = np.zeros((W, W, 4), np.uint8)
for i in range(3): grad[..., i] = (c1[i]*(1-t) + c2[i]*t).astype(np.uint8)
grad[..., 3] = 255
img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
img.paste(Image.fromarray(grad, "RGBA"), (0, 0), rounded_mask(W, int(14/64*W)))
d = ImageDraw.Draw(img)

WHITE = (255, 255, 255, 255); SHADE = (206, 222, 243, 255)
WING = (186, 206, 234, 255); WING2 = (208, 224, 245, 255)
ORANGE = (245, 158, 11, 255); BEAK = (251, 191, 36, 255); DARK = (11, 14, 20, 255)

# 脖子：平滑贝塞尔粗线
neck = bez((24.5, 40.0), (23.2, 27.5), (30.0, 17.6), (39.6, 16.4))
stroke(d, [P(x, y) for x, y in neck], int(7.6*u), WHITE)

# 尾羽
d.polygon([P(16.0, 41.0), P(7.0, 36.0), P(11.0, 41.5), P(12.5, 47.5)], fill=SHADE)

# 身体
d.ellipse([P(28.5-14.2, 43.4-10.0), P(28.5+14.2, 43.4+10.0)], fill=SHADE)
d.ellipse([P(28.5-13.9, 43.4-10.4), P(28.5+13.9, 43.4+8.6)], fill=WHITE)

# 翅羽（自然的水滴形）
wu = bez((17.6, 40.6), (25.0, 35.4), (33.5, 37.6), (39.0, 42.4))
wl = bez((39.0, 42.4), (31.5, 41.6), (23.5, 41.4), (17.6, 40.6))
d.polygon([P(x, y) for x, y in wu + wl], fill=WING)
d.line([P(20.5, 40.4), P(28.5, 38.4), P(36.5, 41.4)], fill=WING2, width=int(2.2*u), joint="curve")

# 腿
lw = int(3.1*u)
for a, b in [((25.0, 51.5), (23.8, 58.4)), ((33.4, 51.5), (34.6, 58.4))]:
    pa, pb = P(*a), P(*b)
    d.line([pa, pb], fill=ORANGE, width=lw)
    d.ellipse([pb[0]-lw/2, pb[1]-lw/2, pb[0]+lw/2, pb[1]+lw/2], fill=ORANGE)

# 头
d.ellipse([P(40.3-5.7, 16.2-5.7), P(40.3+5.7, 16.2+5.7)], fill=WHITE)

# 喙 + 喉囊
poly = [P(44.6, 13.3), P(62.0, 18.7)] + [P(x, y) for (x, y) in
        bez((62.0, 18.7), (55.6, 27.4), (47.2, 25.8), (44.2, 19.4))]
d.polygon(poly, fill=BEAK)
d.line([P(44.6, 13.3), P(62.0, 18.7)], fill=ORANGE, width=int(1.7*u))

# 眼
d.ellipse([P(41.7-1.55, 14.8-1.55), P(41.7+1.55, 14.8+1.55)], fill=DARK)

master = img.resize((S, S), Image.LANCZOS)
master.save(OUT/"pelican-512.png")
for sz in (180, 32, 16):
    master.resize((sz, sz), Image.LANCZOS).save(OUT/f"pelican-{sz}.png")
print("saved:", [p.name for p in sorted(OUT.glob("pelican-*.png"))])
