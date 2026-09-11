import os
"""make_makoto_preview.py — 逐帧重绘 makoto-thunder.svg（风格化同人致敬），验证造型与动效节奏。
同人致敬：P3R 结城理 / Makoto Yuki，版权属 ATLUS；此处为原创风格化演绎。
"""
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

OUT = Path(ROOT + "/outputs"); OUT.mkdir(parents=True, exist_ok=True)
S = 300; SS = 3; W = S*SS; u = W/64.0; LOOP = 7.0
def P(x, y): return (x*u, y*u)
def bez(p0, p1, p2, p3, n=30):
    o = []
    for i in range(n+1):
        t = i/n; mt = 1-t
        o.append((mt**3*p0[0]+3*mt*mt*t*p1[0]+3*mt*t*t*p2[0]+t**3*p3[0],
                  mt**3*p0[1]+3*mt*mt*t*p1[1]+3*mt*t*t*p2[1]+t**3*p3[1]))
    return o
def kf(pct, table):
    p = pct % 100; pts = sorted(table)
    if p <= pts[0][0]: return pts[0][1]
    if p >= pts[-1][0]: return pts[-1][1]
    for (a, va), (b, vb) in zip(pts, pts[1:]):
        if a <= p <= b:
            k = 0 if b == a else (p-a)/(b-a)
            if isinstance(va, tuple): return tuple(va[i]+(vb[i]-va[i])*k for i in range(len(va)))
            return va+(vb-va)*k
    return pts[-1][1]

DARK  = [(0,.85),(4,.2),(7,.78),(10,.25),(14,.6),(20,.08),(30,0),(70,0),(76,.3),(82,.08),(90,.7),(100,.85)]
FLASH = [(0,0),(2,0),(3.5,.95),(5,0),(7,.55),(9,0),(52,0),(55,.85),(58,0),(84,0),(86,.9),(88,0),(100,0)]
BOLT  = [(0,0),(2,0),(3,1),(4.5,.1),(6,.9),(9,0),(50,0),(53,1),(56,.2),(59,.9),(62,0),(85,0),(86.5,1),(88,0),(100,0)]
SPEED = [(0,0),(8,0),(10,.9),(16,0),(70,0),(74,.85),(82,0),(100,0)]
ARCS  = [(0,0),(44,0),(47,1),(49,.15),(51,.95),(53,.2),(55,1),(58,.35),(62,.9),(66,.2),(70,0),(100,0)]
AURA  = [(0,(.5,0)),(13,(.5,0)),(16,(1,.45)),(30,(1.1,.3)),(44,(1,.25)),(52,(1.4,1)),(60,(1.55,.9)),
         (68,(1.25,.6)),(76,(1.05,.4)),(82,(.6,0)),(100,(.6,0))]
RING  = [(0,(.15,0)),(11,(.15,0)),(13,(.3,.95)),(22,(2.1,0)),(72,(.2,0)),(74,(.4,.9)),(84,(2.4,0)),(100,(2.4,0))]
STREAK= [(0,(-46,0)),(84,(-46,0)),(86,(-40,1)),(92,(46,0)),(100,(46,0))]
IRIS  = [(0,(0,0)),(88,(0,0)),(93,(2.6,1)),(96,(2.6,1)),(100,(0,0))]
HERO_X= [(0,6),(7,6),(10,2),(13,0),(15,0),(18,0),(54,0),(58,0),(70,0),(74,0),(78,14),(84,34),(100,34)]
HERO_Y= [(0,-48),(7,-48),(10,-22),(13,2),(15,-1.5),(18,-1.5),(54,-1.5),(58,0),(70,0),(74,-2),(78,-34),(84,-70),(100,-70)]
HERO_S= [(0,2.6),(7,2.6),(10,1.75),(13,1.04),(15,.98),(18,1),(54,1),(58,1.05),(70,1.05),(74,1),(78,.44),(84,.18),(100,.18)]
HERO_R= [(0,-8),(7,-8),(10,-4),(13,0),(18,0),(70,0),(78,12),(84,18),(100,18)]
HERO_O= [(0,0),(7,0),(9,1),(18,1),(74,1),(80,1),(84,0),(100,0)]
ARM   = [(0,0),(30,0),(45,-10),(54,-26),(68,-26),(78,-14),(86,0),(100,0)]
SHAKE = [(0,(0,0)),(11,(0,0)),(12,(-.9,.5)),(13.5,(.8,-.6)),(15,(-.6,-.4)),(16.5,(.5,.5)),(18,(0,0)),
         (57,(0,0)),(58.5,(-.5,.4)),(60,(.6,-.4)),(61.5,(-.4,.3)),(63,(0,0)),(73,(0,0)),
         (74.5,(.7,-.5)),(76,(-.6,.4)),(77.5,(0,0)),(100,(0,0))]

SKIN=(239,208,176,255); SKIN2=(230,195,162,255); HAIR=(27,34,68,255); HAIR2=(35,44,85,255)
BLAZER=(34,42,76,255); BLAZER2=(43,52,89,255); SHIRT=(241,245,252,255); COLLAR=(231,237,248,255)
TIE=(156,59,71,255); HP=(43,50,67,255); HP2=(123,136,164,255)
GUN=(60,68,96,255); GUN2=(111,124,156,255); DARKC=(29,37,68,255)

yy, xx = np.mgrid[0:W, 0:W]
def lerp(c1, c2, t): return tuple(int(c1[i]+(c2[i]-c1[i])*t) for i in range(3))
SKY = np.zeros((W, W, 4), np.uint8)
for j in range(W):
    t = j/(W-1)
    c = lerp((13,18,48),(30,35,82), t/0.5) if t < 0.5 else lerp((30,35,82),(59,42,110),(t-0.5)/0.5)
    SKY[j, :, :3] = c
SKY[..., 3] = 255
def rmask(size, r):
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size-1, size-1], radius=r, fill=255)
    return m
CLIP = rmask(W, int(14/64*W))
def stroke(d, pts, width, color):
    r = width/2.0
    for (x, y) in pts: d.ellipse([x-r, y-r, x+r, y+r], fill=color)

BACKHAIR = [(20.5,27),(21,20),(23,14.5),(26.5,11),(32,10),(37.5,11),(41,14.5),(43,20),(43.5,27),
            (42.5,34.5),(41,36.5),(38,34),(39,29),(39.5,22),(37.5,18.5),(26.5,18.5),(24.5,22),(25,29),(26,34),(23,36.5),(21.5,34.5)]
FRINGE = [(22,22),(23.5,15),(27,12),(32,12),(37,12),(41,15),(42,22),(39,20),(37,23.5),(34.5,19.5),
          (31.5,23),(29,19.5),(26.5,22.5),(25,19.8)]
BANG = [(22.6,21.5),(28.2,25.5),(27,32.5),(23.4,29)]

def body_layer(t):
    L = Image.new("RGBA", (W, W), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    by = -1.1*(1-math.cos(2*math.pi*(t % 3.0)/3.0))/2
    d.polygon([P(22,37), P(42,37), P(46.5,58), P(17.5,58)], fill=BLAZER)
    d.polygon([P(22,37), P(42,37), P(43,41), P(21,41)], fill=BLAZER2)
    d.polygon([P(27.5,36.5), P(36.5,36.5), P(32,45)], fill=SHIRT)
    d.polygon([P(31,40.5), P(33,40.5), P(33.7,52.5), P(30.3,52.5)], fill=TIE)
    d.polygon([P(26.5,36.2), P(31.5,36.6), P(29,41.5)], fill=COLLAR)
    d.polygon([P(37.5,36.2), P(32.5,36.6), P(35,41.5)], fill=COLLAR)
    d.rounded_rectangle([P(29,32.5), P(35,37.5)], radius=int(2*u), fill=SKIN2)
    return L, by

def head_layer(t, blink):
    L = Image.new("RGBA", (W, W), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    d.ellipse([P(32-10.5,25-10.5), P(32+10.5,25+10.5)], fill=SKIN)
    d.polygon([P(x,y) for x,y in BACKHAIR], fill=HAIR)
    sway = 1.4*math.sin(2*math.pi*(t % 3.4)/3.4)
    hair = Image.new("RGBA", (W, W), (0,0,0,0)); dh = ImageDraw.Draw(hair)
    dh.polygon([P(x,y) for x,y in FRINGE], fill=HAIR2)
    dh.polygon([P(x,y) for x,y in BANG], fill=HAIR2)
    hair = hair.rotate(-sway, center=P(32,20), resample=Image.BICUBIC)
    L.alpha_composite(hair)
    # eye
    ey = 1.9*blink
    d.ellipse([P(36-2.5,25-1.9), P(36+2.5,25+1.9)], fill=(242,246,255,255))
    d.ellipse([P(36.2-1.15,25-1.15*blink), P(36.2+1.15,25+1.15*blink)], fill=DARKC)
    d.line([P(33.4,23.1), P(36,22.4), P(38.7,23.2)], fill=HAIR, width=max(1,int(1.0*u)), joint="curve")
    d.line([P(33.6,20.6), P(36.2,20.1), P(38.8,20.9)], fill=HAIR, width=max(1,int(1.1*u)), joint="curve")
    d.line([P(31.6,30.4), P(32.8,30.9), P(34,30.4)], fill=(184,121,95,255), width=max(1,int(.9*u)), joint="curve")
    # headphones
    stroke(d, [P(x,y) for x,y in bez((21.5,20.5),(23,12.5),(41,12.5),(42.5,20.5))], int(2.4*u), HP)
    for x0 in (18.6, 40.8):
        d.rounded_rectangle([P(x0,22.6), P(x0+4.6,30.0)], radius=int(2.2*u), fill=HP)
        d.ellipse([P(x0+2.3-1.5,26.3-1.5), P(x0+2.3+1.5,26.3+1.5)], fill=HP2)
    return L

def arm_layer(t):
    L = Image.new("RGBA", (W, W), (0,0,0,0)); d = ImageDraw.Draw(L)
    stroke(d, [P(40.5,39.5), P(44,35), P(46,33), P(43.5,26.5)], int(4.6*u), BLAZER)
    d.ellipse([P(43.2-2,26.2-2), P(43.2+2,26.2+2)], fill=SKIN)
    d.polygon([P(37.8,21.6), P(44.6,23.2), P(44,26.4), P(37.4,24.8)], fill=GUN)
    d.polygon([P(38.2,22.1), P(41.4,22.9), P(41.1,24.3), P(37.9,23.5)], fill=GUN2)
    d.rounded_rectangle([P(43.6,24.4), P(46.2,28.6)], radius=int(1*u), fill=(47,53,73,255))
    ang = kf((t % LOOP)/LOOP*100, ARM)
    return L.rotate(-ang, center=P(41,40), resample=Image.BICUBIC)

def frame(t):
    pct = (t % LOOP)/LOOP*100
    img = Image.new("RGBA", (W, W), (0,0,0,0))
    img.paste(Image.fromarray(SKY, "RGBA"), (0,0), CLIP)
    cl = Image.new("RGBA", (W, W), (0,0,0,0)); dc = ImageDraw.Draw(cl)
    for cx, cy, rx, ry in [(14,9,16,7),(40,6,18,6.5),(56,13,14,6),(26,14,13,5)]:
        dc.ellipse([P(cx-rx,cy-ry), P(cx+rx,cy+ry)], fill=(8,12,34,150))
    img.alpha_composite(cl)
    sx, sy = kf(pct, SHAKE)
    scene = Image.new("RGBA", (W, W), (0,0,0,0)); ds = ImageDraw.Draw(scene)
    bo = kf(pct, BOLT)
    if bo > .01:
        for pts, wd, col, al in [([P(33,1),P(25,21),P(31,21),P(22,42)], int(3.4*u), (223,243,255), .35),
                                 ([P(33,1),P(25,21),P(31,21),P(22,42)], int(1.5*u), (255,255,255), 1.),
                                 ([P(47,3),P(43,16),P(47,16),P(42,30)], int(2.2*u), (223,243,255), .3),
                                 ([P(47,3),P(43,16),P(47,16),P(42,30)], int(1.0*u), (255,255,255), 1.)]:
            ds.line(pts, fill=col+(int(255*al*bo),), width=wd, joint="curve")
    rs, ra = kf(pct, RING)
    if ra > .01:
        rr = 18*u*rs
        ds.ellipse([P(32,46)[0]-rr, P(32,46)[1]-rr, P(32,46)[0]+rr, P(32,46)[1]+rr],
                   outline=(191,230,255,int(255*ra)), width=max(1,int(1.6*u)))
    so = kf(pct, SPEED)
    if so > .01:
        for x0, y0, y1 in [(17,3,15),(25,-1,11),(39,1,13),(47,5,17)]:
            ds.line([P(x0,y0), P(x0,y1)], fill=(234,246,255,int(255*so)), width=max(1,int(1.1*u)))
    # 角色
    hero = Image.new("RGBA", (W, W), (0,0,0,0))
    as_, ao = kf(pct, AURA)
    if ao > .01:
        rr = 23*u*as_
        for i in range(12, 0, -1):
            k = i/12.0
            ds2 = ImageDraw.Draw(hero)
            ds2.ellipse([P(32,32)[0]-rr*k, P(32,32)[1]-rr*k, P(32,32)[0]+rr*k, P(32,32)[1]+rr*k],
                        fill=(95,141,255,int(30*ao*(1-k)+5)))
    body, by = body_layer(t)
    hero.alpha_composite(body, (0, int(by*u)))
    ph = (t % 3.6)/3.6; blink = .08 if .945 <= ph <= .96 else 1.0
    hd = head_layer(t, blink); hero.alpha_composite(hd, (0, int(by*u)))
    hero.alpha_composite(arm_layer(t), (0, int(by*u)))
    ao2 = kf(pct, ARCS)
    if ao2 > .01:
        dh2 = ImageDraw.Draw(hero)
        for pts in ([(12,24),(17,28),(13,31),(19,35)], [(52,36),(47,40),(53,43)], [(22,9),(26,13),(21,16)],
                    [(45,18),(49,21),(45,24)], [(16,44),(20,47),(15,50)]):
            dh2.line([P(x,y) for x,y in pts], fill=(168,236,255,int(255*ao2)), width=max(1,int(1.25*u)), joint="curve")
    s = kf(pct, HERO_S); rot = kf(pct, HERO_R); gx = kf(pct, HERO_X); gy = kf(pct, HERO_Y); go = kf(pct, HERO_O)
    if s != 1:
        ns = max(8, int(W*s)); g2 = Image.new("RGBA", (W, W), (0,0,0,0))
        g2.alpha_composite(hero.resize((ns, ns), Image.LANCZOS), ((W-ns)//2, (W-ns)//2)); hero = g2
    if abs(rot) > .1: hero = hero.rotate(-rot, center=(W//2, W//2), resample=Image.BICUBIC)
    if go < 1:
        a = hero.split()[3].point(lambda v: int(v*go)); hero.putalpha(a)
    scene.alpha_composite(hero, (int(gx*u), int(gy*u)))
    tx, to = kf(pct, STREAK)
    if to > .01:
        band = Image.new("RGBA", (W, W), (0,0,0,0))
        ImageDraw.Draw(band).rectangle([P(tx,-24), P(tx+26,88)], fill=(223,243,255,int(210*to)))
        scene.alpha_composite(band.rotate(-18, center=(W//2, W//2), resample=Image.BICUBIC))
    img.alpha_composite(scene, (int(sx*u), int(sy*u)))
    img.alpha_composite(Image.new("RGBA", (W, W), (4,6,15,int(255*kf(pct, DARK)))))
    fo = kf(pct, FLASH)
    if fo > .01: img.alpha_composite(Image.new("RGBA", (W, W), (223,241,255,int(255*fo))))
    isc, iop = kf(pct, IRIS)
    if iop > .01:
        ir = Image.new("RGBA", (W, W), (0,0,0,0)); rr = 30*u*isc
        ImageDraw.Draw(ir).ellipse([P(32,32)[0]-rr, P(32,32)[1]-rr, P(32,32)[0]+rr, P(32,32)[1]+rr],
                                   fill=(4,6,15,int(255*iop)))
        img.alpha_composite(ir)
    ImageDraw.Draw(img).rounded_rectangle([P(.75,.75), P(63.25,63.25)], radius=int(13.4*u),
                                          outline=(11,14,20,70), width=int(1.5*u))
    return img.resize((S, S), Image.LANCZOS)

beats = [(0.30,"雷暴+闪电"),(0.72,"登场砸地"),(0.95,"冲击波"),(2.60,"待机"),
         (3.60,"抬臂"),(4.10,"召唤器抵额"),(4.80,"电弧爆发"),(5.50,"蓝电消散"),(6.15,"光带转场"),(6.75,"黑场")]
frames = [frame(t) for t, _ in beats]
pad, top = 10, 70
cols = 5; rows = (len(frames)+cols-1)//cols
Wc = pad + cols*(S+pad); Hc = top + rows*(S+34) + pad
canvas = Image.new("RGB", (Wc, Hc), (10, 12, 22)); d = ImageDraw.Draw(canvas)
def font(sz, b=False):
    try: return ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc" if b else r"C:\Windows\Fonts\msyh.ttc", sz)
    except Exception: return ImageFont.load_default()
d.text((pad, 12), "雷霆·召唤（结城理 风格化致敬）— 分帧预览", font=font(20, True), fill=(232,237,245))
d.text((pad, 40), "雷暴 → 闪电 → 登场砸地 → 冲击波震屏 → 待机 → 抬臂 · 召唤器抵额 → 电弧与蓝焰爆发 → 蓝电消散 → 光带转场 → 黑场循环",
       font=font(13), fill=(150,170,195))
for i, (im, (t, name)) in enumerate(zip(frames, beats)):
    r, c = divmod(i, cols); x = pad + c*(S+pad); y = top + r*(S+34)
    canvas.paste(im, (x, y), im)
    d.text((x+4, y+S+4), f"t={t}s  {name}", font=font(14, True), fill=(255,220,150))
canvas.save(OUT/"makoto_thunder_frames.png")
print("saved", OUT/"makoto_thunder_frames.png", canvas.size)
