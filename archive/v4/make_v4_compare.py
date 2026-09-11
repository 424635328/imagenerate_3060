"""make_v4_compare.py — V3 vs V4 同 prompt 对比拼图（上 V3 / 下 V4）。"""
import os
from PIL import Image, ImageDraw, ImageFont
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.abspath(__file__)))
def cjk(bold=False, size=20):
    for p in [r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
              r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except Exception: pass
    return ImageFont.load_default()
NAMES = ["mountain","rainforest","desert","coast","waterfall","lake"]
LABELS = ["高山日出","雨林瀑布","金色沙漠","热带海岸","苔藓瀑布","雪山湖泊"]
V3 = ["outputs/gallery_v3/v2_20260909_152400_0_prompt.png","outputs/gallery_v3/v2_20260909_152754_1_prompt.png",
      "outputs/gallery_v3/v2_20260909_153148_2_prompt.png","outputs/gallery_v3/v2_20260909_153603_3_prompt.png",
      "outputs/gallery_v3/v2_20260909_154101_4_prompt.png","outputs/gallery_v3/v2_20260909_154519_5_prompt.png"]
V4 = [f"outputs/gallery_v4/v4_{n}_1024.png" for n in NAMES]
cell=300; pad=12; top=64; left=150
W = left + len(NAMES)*(cell+pad) + pad
H = top + 2*(cell+34) + pad*2 + 40
canvas = Image.new("RGB",(W,H),(13,16,26)); d=ImageDraw.Draw(canvas)
d.text((20,18),"V3  vs  V4（同 prompt 对比）", font=cjk(True,26), fill=(232,237,245))
def cellimg(path):
    im=Image.open(os.path.join(ROOT,path)).convert("RGB"); w,h=im.size; s=min(w,h)
    im=im.crop(((w-s)//2,(h-s)//2,(w-s)//2+s,(h-s)//2+s)).resize((cell,cell),Image.LANCZOS); return im
for row,(title,paths,col) in enumerate([("V3 · 512+高清修复",V3,(147,160,180)),("V4 · 源增强+EMA+CLIP微调+两段式",V4,(110,168,255))]):
    y = top + row*(cell+34)
    d.text((20, y+cell//2-10), title, font=cjk(True,17), fill=col)
    for i,p in enumerate(paths):
        x = left + i*(cell+pad)
        canvas.paste(cellimg(p),(x,y))
        if row==0: d.text((x+6, y-26), LABELS[i], font=cjk(True,16), fill=(200,210,225))
canvas.save(os.path.join(ROOT,"outputs","v3_vs_v4.png"))
print("saved outputs/v3_vs_v4.png", canvas.size)
