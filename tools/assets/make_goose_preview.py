import os
"""make_goose_preview.py — 逐帧重绘 goose-thunder.svg 的关键节拍，输出胶片条用于验证动效设计。
模拟：雷暴压暗 / 闪电闪光 / 大鹅砸地入场 / 冲击波 / 震屏 / 蓄力电弧 / 火箭出场 / 光带转场 / 黑场虹膜。
"""
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

OUT = Path(ROOT + "/outputs"); OUT.mkdir(parents=True, exist_ok=True)
S = 300; SS = 3; W = S*SS; u = W/64.0
LOOP = 7.0
def P(x, y): return (x*u, y*u)

def bez(p0, p1, p2, p3, n=36):
    o = []
    for i in range(n+1):
        t = i/n; mt = 1-t
        o.append((mt**3*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t**3*p3[0],
                  mt**3*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t**3*p3[1]))
    return o

def kf(pct, table):
    """按百分比线性插值 table=[(pct, val), ...]（val 可为 float 或 tuple）。"""
    p = pct % 100
    pts = sorted(table)
    if p <= pts[0][0]: return pts[0][1]
    if p >= pts[-1][0]: return pts[-1][1]
    for (a, va), (b, vb) in zip(pts, pts[1:]):
        if a <= p <= b:
            k = 0 if b == a else (p-a)/(b-a)
            if isinstance(va, tuple):
                return tuple(va[i] + (vb[i]-va[i])*k for i in range(len(va)))
            return va + (vb-va)*k
    return pts[-1][1]

DARK  = [(0,.85),(4,.2),(7,.78),(10,.25),(14,.62),(20,.1),(30,0),(70,0),(76,.35),(82,.1),(90,.7),(100,.85)]
FLASH = [(0,0),(2,0),(3.5,.95),(5,0),(7,.55),(9,0),(84,0),(86,.9),(88,0),(100,0)]
BOLT  = [(0,0),(2,0),(3,1),(4.5,.1),(6,.9),(9,0),(85,0),(86.5,1),(88,0),(100,0)]
SPEED = [(0,0),(8,0),(10,.9),(16,0),(70,0),(74,.85),(82,0),(100,0)]
ARCS  = [(0,0),(54,0),(56,1),(58,.15),(60,.95),(62,.2),(64,1),(66,.1),(70,0),(100,0)]
AURA  = [(0,(0.5,0)),(13,(0.5,0)),(16,(1,.5)),(30,(1.12,.32)),(54,(1,.28)),(58,(1.35,.95)),
         (66,(1.2,.7)),(72,(1.1,.5)),(80,(.6,0)),(100,(.6,0))]
RING  = [(0,(.15,0)),(11,(.15,0)),(13,(.3,.95)),(22,(2.1,0)),(72,(.2,0)),(74,(.4,.9)),(84,(2.4,0)),(100,(2.4,0))]
STREAK= [(0,(-46,0)),(84,(-46,0)),(86,(-40,1)),(92,(46,0)),(100,(46,0))]
IRIS  = [(0,(0,0)),(88,(0,0)),(93,(2.6,1)),(96,(2.6,1)),(100,(0,0))]
GX    = [(0,7),(7,7),(10,2),(13,0),(15,0),(18,0),(54,0),(58,0),(70,0),(74,0),(78,16),(84,36),(100,36)]
GY    = [(0,-48),(7,-48),(10,-22),(13,2),(15,-1.5),(18,-1.5),(54,-1.5),(58,0),(70,0),(74,-2),(78,-36),(84,-72),(100,-72)]
GSC   = [(0,2.7),(7,2.7),(10,1.8),(13,1.04),(15,.98),(18,1),(54,1),(58,1.04),(70,1.04),(74,1),(78,.42),(84,.16),(100,.16)]
GROT  = [(0,-9),(7,-9),(10,-4),(13,0),(18,0),(70,0),(78,14),(84,20),(100,20)]
GOP   = [(0,0),(7,0),(9,1),(18,1),(74,1),(80,1),(84,0),(100,0)]
SHAKE = [(0,(0,0)),(11,(0,0)),(12,(-.9,.5)),(13.5,(.8,-.6)),(15,(-.6,-.4)),(16.5,(.5,.5)),(18,(0,0)),
         (57,(0,0)),(58.5,(-.5,.4)),(60,(.6,-.4)),(61.5,(-.4,.3)),(63,(0,0)),(73,(0,0)),
         (74.5,(.7,-.5)),(76,(-.6,.4)),(77.5,(0,0)),(100,(0,0))]

WHITE=(255,255,255,255); SHADE=(206,222,243,255); WING=(186,206,234,255); WING2=(208,224,245,255)
ORANGE=(245,158,11,255); BEAK=(251,191,36,255); DARKC=(11,14,20,255)

yy, xx = np.mgrid[0:W, 0:W]
tv = yy/(W-1)
def lerp(c1, c2, t): return tuple(int(c1[i]+(c2[i]-c1[i])*t) for i in range(3))
SKY = np.zeros((W, W, 4), np.uint8)
for j in range(W):
    t = j/(W-1)
    c = lerp((27,35,80),(58,42,107), t/0.55) if t < 0.55 else lerp((58,42,107),(109,63,156),(t-0.55)/0.45)
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

def bird_layer(t):
    L = Image.new("RGBA", (W, W), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    tt = t % 3.0; dy = -1.5*(1-math.cos(2*math.pi*tt/3.0))/2
    ang = 1.8*math.sin(2*math.pi*tt/3.0)
    ps = 1 + 0.05*(1-math.cos(2*math.pi*tt/3.0))/2
    ph = (t % 3.6)/3.6; bs = 0.08 if 0.90 <= ph <= 0.96 else 1.0
    # body
    body = Image.new("RGBA", (W, W), (0, 0, 0, 0)); db = ImageDraw.Draw(body)
    db.polygon([P(16,41), P(7,36), P(11,41.5), P(12.5,47.5)], fill=SHADE)
    db.ellipse([P(28.5-14.2,43.4-10), P(28.5+14.2,43.4+10)], fill=SHADE)
    db.ellipse([P(28.5-13.9,43.4-10.4), P(28.5+13.9,43.4+8.6)], fill=WHITE)
    wu = bez((17.6,40.6),(25.0,35.4),(33.5,37.6),(39.0,42.4)); wl = bez((39.0,42.4),(31.5,41.6),(23.5,41.4),(17.6,40.6))
    db.polygon([P(x,y) for x,y in wu+wl], fill=WING)
    lw = int(3.1*u)
    for a,b in [((25,51.5),(23.8,58.4)),((33.4,51.5),(34.6,58.4))]:
        pa, pb = P(*a), P(*b); db.line([pa,pb], fill=ORANGE, width=lw)
        db.ellipse([pb[0]-lw/2,pb[1]-lw/2,pb[0]+lw/2,pb[1]+lw/2], fill=ORANGE)
    L.alpha_composite(body, (0, int(dy*u)))
    # head
    head = Image.new("RGBA", (W, W), (0,0,0,0)); dh = ImageDraw.Draw(head)
    stroke(dh, [P(x,y) for x,y in bez((24.5,40.0),(23.2,27.5),(30.0,17.6),(39.6,16.4))], int(7.6*u), WHITE)
    dh.ellipse([P(40.3-5.7,16.2-5.7), P(40.3+5.7,16.2+5.7)], fill=WHITE)
    piv = (45.0,17.5)
    def sc(x,y): return (piv[0]+(x-piv[0])*ps, piv[1]+(y-piv[1])*ps)
    poly = [sc(44.6,13.3), sc(62.0,18.7)] + [sc(x,y) for x,y in bez((62.0,18.7),(55.6,27.4),(47.2,25.8),(44.2,19.4))]
    dh.polygon([P(x,y) for x,y in poly], fill=BEAK)
    dh.line([P(44.6,13.3), P(62.0,18.7)], fill=ORANGE, width=int(1.7*u))
    dh.ellipse([P(41.7-1.55,14.8-1.55*bs), P(41.7+1.55,14.8+1.55*bs)], fill=DARKC)
    head = head.rotate(-ang, center=P(24.5,40.0), resample=Image.BICUBIC)
    L.alpha_composite(head, (0, int(dy*u)))
    return L

def frame(t):
    pct = (t % LOOP)/LOOP*100
    img = Image.new("RGBA", (W, W), (0,0,0,0))
    img.paste(Image.fromarray(SKY, "RGBA"), (0,0), CLIP)
    d = ImageDraw.Draw(img)
    # 雷云
    cl = Image.new("RGBA", (W, W), (0,0,0,0)); dc = ImageDraw.Draw(cl)
    for cx, cy, rx, ry in [(14,9,16,7),(40,6,18,6.5),(56,13,14,6),(26,14,13,5)]:
        dc.ellipse([P(cx-rx,cy-ry), P(cx+rx,cy+ry)], fill=(13,18,48,128))
    img.alpha_composite(cl)
    # shake
    sx, sy = kf(pct, SHAKE)
    scene = Image.new("RGBA", (W, W), (0,0,0,0)); ds = ImageDraw.Draw(scene)
    # 闪电
    bo = kf(pct, BOLT)
    if bo > 0.01:
        for wdt, col, al in [(int(3.4*u), (223,243,255), .35*bo), (int(1.5*u), (255,255,255), bo)]:
            pts1 = [P(33,1), P(25,21), P(31,21), P(22,42)]
            pts2 = [P(46,3), P(42,16), P(46,16), P(41,30)]
            for pts in (pts1, pts2):
                ds.line(pts, fill=col+(int(255*al),), width=wdt, joint="curve")
    # 冲击波
    rs, ra = kf(pct, RING)
    if ra > 0.01:
        rr = 18*u*rs
        ds.ellipse([P(32,44)[0]-rr, P(32,44)[1]-rr, P(32,44)[0]+rr, P(32,44)[1]+rr],
                   outline=(191,230,255,int(255*ra)), width=max(1,int(1.6*u)))
    # 高速线
    so = kf(pct, SPEED)
    if so > 0.01:
        for x0, y0, y1 in [(17,3,15),(25,-1,11),(39,1,13),(47,5,17)]:
            ds.line([P(x0,y0), P(x0,y1)], fill=(234,246,255,int(255*so)), width=max(1,int(1.1*u)))
    # 大鹅
    gs = bird_layer(t)
    s = kf(pct, GSC); rot = kf(pct, GROT); gx = kf(pct, GX); gy = kf(pct, GY); go = kf(pct, GOP)
    as_, ao = kf(pct, AURA)
    if ao > 0.01:
        rr = 22*u*as_
        aura = Image.new("RGBA", (W, W), (0,0,0,0)); da = ImageDraw.Draw(aura)
        for i in range(10, 0, -1):
            k = i/10.0
            da.ellipse([P(32,34)[0]-rr*k, P(32,34)[1]-rr*k, P(32,34)[0]+rr*k, P(32,34)[1]+rr*k],
                       fill=(110,168,255,int(28*ao*(1-k)+6)))
        scene.alpha_composite(aura)
    if s != 1:
        ns = max(8, int(W*s)); g2 = Image.new("RGBA", (W, W), (0,0,0,0))
        g2.alpha_composite(g2.copy().resize((1,1)) if False else gs.resize((ns,ns), Image.LANCZOS), ((W-ns)//2, (W-ns)//2))
        gs = g2
    if abs(rot) > .1: gs = gs.rotate(-rot, center=(W//2, W//2), resample=Image.BICUBIC)
    if go < 1:
        a = gs.split()[3].point(lambda v: int(v*go)); gs.putalpha(a)
    scene.alpha_composite(gs, (int(gx*u), int(gy*u)))
    # 电弧
    ao2 = kf(pct, ARCS)
    if ao2 > 0.01:
        for pts in ([(11,25),(16,29),(12,32),(18,36)], [(53,38),(48,42),(54,45)], [(21,10),(25,14),(20,17)]):
            ds.line([P(x,y) for x,y in pts], fill=(159,232,255,int(255*ao2)), width=max(1,int(1.2*u)), joint="curve")
    # 光带转场
    tx, to = kf(pct, STREAK)
    if to > 0.01:
        band = Image.new("RGBA", (W, W), (0,0,0,0)); dbd = ImageDraw.Draw(band)
        dbd.rectangle([P(tx,-24), P(tx+26,88)], fill=(223,243,255,int(210*to)))
        band = band.rotate(-18, center=(W//2, W//2), resample=Image.BICUBIC)
        scene.alpha_composite(band)
    img.alpha_composite(scene, (int(sx*u), int(sy*u)))
    # 压暗 / 闪光 / 虹膜
    dk = Image.new("RGBA", (W, W), (5,7,24,int(255*kf(pct, DARK))))
    img.alpha_composite(dk)
    fo = kf(pct, FLASH)
    if fo > 0.01:
        img.alpha_composite(Image.new("RGBA", (W, W), (255,255,255,int(255*fo))))
    isc, iop = kf(pct, IRIS)
    if iop > 0.01:
        ir = Image.new("RGBA", (W, W), (0,0,0,0)); di = ImageDraw.Draw(ir)
        rr = 30*u*isc
        di.ellipse([P(32,32)[0]-rr, P(32,32)[1]-rr, P(32,32)[0]+rr, P(32,32)[1]+rr],
                   fill=(5,7,24,int(255*iop)))
        img.alpha_composite(ir)
    ImageDraw.Draw(img).rounded_rectangle([P(.75,.75), P(63.25,63.25)], radius=int(13.4*u),
                                          outline=(11,14,20,70), width=int(1.5*u))
    return img.resize((S, S), Image.LANCZOS)

beats = [(0.30,"雷暴+闪电"),(0.72,"砸地"),(0.95,"冲击波"),(1.60,"震屏"),
         (2.60,"待机"),(4.30,"蓄力电弧"),(5.50,"火箭出场"),(6.15,"光带转场"),(6.75,"黑场")]
frames = [frame(t) for t, _ in beats]
pad, top = 10, 70
cols = 5; rows = (len(frames)+cols-1)//cols
Wc = pad + cols*(S+pad); Hc = top + rows*(S+34) + pad
canvas = Image.new("RGB", (Wc, Hc), (12, 15, 24)); d = ImageDraw.Draw(canvas)
def font(sz, b=False):
    try: return ImageFont.truetype(r"C:\Windows\Fonts\msyhbd.ttc" if b else r"C:\Windows\Fonts\msyh.ttc", sz)
    except Exception: return ImageFont.load_default()
d.text((pad, 12), "超级雷霆大鹅 — 分帧预览（7s 循环）", font=font(20, True), fill=(232,237,245))
d.text((pad, 40), "雷暴压暗 → 闪电闪光 → 大鹅砸地入场 → 冲击波+震屏 → 待机 → 蓄力电弧 → 火箭出场 → 光带转场 → 黑场虹膜",
       font=font(13), fill=(150,170,195))
for i, (im, (t, name)) in enumerate(zip(frames, beats)):
    r, c = divmod(i, cols); x = pad + c*(S+pad); y = top + r*(S+34)
    canvas.paste(im, (x, y), im)
    d.text((x+4, y+S+4), f"t={t}s  {name}", font=font(14, True), fill=(255,220,150))
canvas.save(OUT/"goose_thunder_frames.png")
print("saved", OUT/"goose_thunder_frames.png", canvas.size)
