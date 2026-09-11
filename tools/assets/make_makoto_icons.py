import os
"""make_makoto_icons.py — 静态版结城理风格化图标（favicon / apple-touch）。
同人致敬：P3R 结城理 / Makoto Yuki，版权属 ATLUS。
"""
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

OUT = Path(ROOT + "/site")
S = 512; SS = 4; W = S*SS; u = W/64.0
def P(x, y): return (x*u, y*u)
def rmask(size, r):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size-1, size-1], radius=r, fill=255)
    return m
def stroke(d, pts, width, color):
    r = width/2.0
    for (x, y) in pts: d.ellipse([x-r, y-r, x+r, y+r], fill=color)
def bez(p0, p1, p2, p3, n=30):
    o = []
    for i in range(n+1):
        t = i/n; mt = 1-t
        o.append((mt**3*p0[0]+3*mt*mt*t*p1[0]+3*mt*t*t*p2[0]+t**3*p3[0],
                  mt**3*p0[1]+3*mt*mt*t*p1[1]+3*mt*t*t*p2[1]+t**3*p3[1]))
    return o

SKIN=(239,208,176,255); SKIN2=(226,190,157,255); HAIR=(27,34,68,255); HAIR2=(38,47,92,255)
BLAZER=(34,42,76,255); BLAZER2=(46,56,96,255); SHIRT=(241,245,252,255); COLLAR=(231,237,248,255)
TIE=(156,59,71,255); HP=(43,50,67,255); HP2=(130,144,172,255)
GUN=(60,68,96,255); GUN2=(120,134,166,255); DARKC=(29,37,68,255)

BACKHAIR = [(20.5,27),(21,20),(23,14.5),(26.5,11),(32,10),(37.5,11),(41,14.5),(43,20),(43.5,27),
            (42.5,34.5),(41,36.5),(38,34),(39,29),(39.5,22),(37.5,18.5),(26.5,18.5),(24.5,22),(25,29),(26,34),(23,36.5),(21.5,34.5)]
FRINGE = [(21.5,24),(22.5,16.5),(26,12.5),(32,11.5),(38,12.5),(41.5,16.5),(42.5,24),
          (40.5,21),(38.5,24.5),(36.5,20.5),(34.5,24),(32,20.5),(29.5,24),(27,20.5),(25,24),(23.5,21)]
BANG = [(21.8,22.5),(27.5,26),(26.5,33),(22.8,29.5)]

yy, xx = np.mgrid[0:W, 0:W]
t = (xx + yy)/(2*W - 2)
c1 = np.array([110, 168, 255]); c2 = np.array([90, 70, 190])
grad = np.zeros((W, W, 4), np.uint8)
for i in range(3): grad[..., i] = (c1[i]*(1-t) + c2[i]*t).astype(np.uint8)
grad[..., 3] = 255
img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
img.paste(Image.fromarray(grad, "RGBA"), (0, 0), rmask(W, int(14/64*W)))
d = ImageDraw.Draw(img)

# 身体
d.polygon([P(21,36.5), P(43,36.5), P(47,58), P(17,58)], fill=BLAZER)
d.polygon([P(21,36.5), P(43,36.5), P(44,41), P(20,41)], fill=BLAZER2)
d.polygon([P(27,36), P(37,36), P(32,45)], fill=SHIRT)
d.polygon([P(30.6,40.2), P(33.4,40.2), P(34.2,53), P(29.8,53)], fill=TIE)
d.polygon([P(25.6,35.6), P(31.4,36.2), P(28.4,41.6)], fill=COLLAR)
d.polygon([P(38.4,35.6), P(32.6,36.2), P(35.6,41.6)], fill=COLLAR)
d.rounded_rectangle([P(29,32.5), P(35,37.5)], radius=int(2*u), fill=SKIN2)
# 头
d.ellipse([P(21.5,14.5), P(42.5,35.5)], fill=SKIN)
d.polygon([P(x, y) for x, y in BACKHAIR], fill=HAIR)
d.polygon([P(x, y) for x, y in FRINGE], fill=HAIR2)
d.polygon([P(x, y) for x, y in BANG], fill=HAIR2)
# 眼（缩小并对齐）
d.ellipse([P(34.2,24.1), P(38.2,27.1)], fill=(242,246,255,255))
d.ellipse([P(35.55,24.65), P(37.45,26.55)], fill=DARKC)
d.ellipse([P(35.75,24.85), P(36.35,25.45)], fill=(255,255,255,255))
d.line([P(34.0,22.2), P(36.2,21.9), P(38.6,22.7)], fill=HAIR, width=max(1,int(1.05*u)), joint="curve")
d.line([P(31.8,30.2), P(32.9,30.7), P(34.0,30.2)], fill=(184,121,95,255), width=max(1,int(.9*u)), joint="curve")
# 耳机
stroke(d, [P(x, y) for x, y in bez((21.0,19.5),(22.5,12.0),(41.0,12.0),(42.5,19.5))], int(2.4*u), HP)
for x0 in (19.0, 40.8):
    d.rounded_rectangle([P(x0,22.4), P(x0+4.2,29.6)], radius=int(2.1*u), fill=HP)
    d.ellipse([P(x0+2.1-1.35,25.8-1.35), P(x0+2.1+1.35,25.8+1.35)], fill=HP2)
# 右臂 + 召唤器（抵住太阳穴，位于眼睛上方）
ARM = [P(41,38.5), P(47,31), P(47.5,25.5)]
d.line(ARM, fill=(24,30,56,255), width=int(6.2*u), joint="curve")
d.line(ARM, fill=(52,62,104,255), width=int(4.4*u), joint="curve")
for (px, py), rr in ((ARM[0], 3.1*u), (ARM[-1], 2.2*u)):
    d.ellipse([px-rr, py-rr, px+rr, py+rr], fill=(52,62,104,255))
d.ellipse([P(45.3,22.3), P(49.7,26.7)], fill=(24,30,56,255))
d.ellipse([P(45.6,22.6), P(49.4,26.4)], fill=SKIN)  # 手
d.polygon([P(42.5,18.6), P(49.5,20.6), P(48.6,23.6), P(41.8,21.6)], fill=GUN)
d.polygon([P(42.9,19.1), P(46.2,20.1), P(45.8,21.4), P(42.5,20.4)], fill=GUN2)
d.rounded_rectangle([P(48.4,21.4), P(51.0,25.4)], radius=int(1*u), fill=(47,53,73,255))
# 边框
d.rounded_rectangle([P(.75,.75), P(63.25,63.25)], radius=int(13.4*u),
                    outline=(11,14,20,64), width=int(1.5*u))

master = img.resize((S, S), Image.LANCZOS)
master.save(OUT/"makoto-512.png")
for sz in (180, 32, 16):
    master.resize((sz, sz), Image.LANCZOS).save(OUT/f"makoto-{sz}.png")
print("saved:", [p.name for p in sorted(OUT.glob("makoto-*.png"))])
