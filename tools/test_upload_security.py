"""test_upload_security.py — 上传安全管线回归测试（不启动服务器，直接测函数）。

覆盖：
  1. 合法 PNG/JPEG/WebP 通过
  2. EXIF + GPS 被剥离（输出的 getexif() 为空）
  3. 伪装的非图片（重命名的文本/SVG/HTML/EXE 头）被拒绝
  4. SVG data URL 被拒绝
  5. 超像素炸弹被拒绝
  6. 超大 base64 被拒绝
  7. 解码后重新编码，不保留原图（含元数据）
"""
import base64, io, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import server  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


def data_url(raw: bytes, mime="image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def make_jpeg_with_gps() -> bytes:
    """JPEG carrying EXIF with GPS + camera serial, as a phone photo would."""
    img = Image.new("RGB", (900, 600), (40, 90, 140))
    exif = Image.Exif()
    exif[0x010F] = "TestCam"                     # Make
    exif[0x0110] = "LeakModel X"                 # Model
    exif[0x0131] = "SecretSoftware 1.0"          # Software
    exif[0x013B] = "Jane Photographer"           # Artist
    exif[0x8825] = {0x0001: "N", 0x0002: 48.8584, 0x0003: "E", 0x0004: 2.2945}  # GPS lat/lon
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def expect_reject(name, payload, code=None):
    try:
        server._decode_init_image(payload)
        check(name, False, "未被拒绝")
    except HTTPException as exc:
        check(name, code is None or exc.status_code == code, f"HTTP {exc.status_code} {exc.detail}")
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"异常类型不符: {type(exc).__name__}: {exc}")


print("== 1) 合法格式 ==")
for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), (10, 120, 90)).save(buf, fmt)
    raw = buf.getvalue()
    check(f"{fmt} 被接受（{len(raw)//1024} KB）", server._decode_init_image(data_url(raw, mime)) is not None)

print("\n== 2) EXIF/GPS 剥离 ==")
jpeg = make_jpeg_with_gps()
src = Image.open(io.BytesIO(jpeg))
src_exif = dict(src.getexif())
print(f"  输入 EXIF 标签: {[hex(k) for k in src_exif]}")
check("输入确实带 GPS(0x8825)", 0x8825 in src_exif)
clean = server._decode_init_image(data_url(jpeg))
out_exif = dict(clean.getexif()) if clean else {"?": "无输出"}
check("输出 EXIF 为空", not out_exif, f"残留={[hex(k) for k in out_exif]}")
check("输出为 RGB 且在像素上限内", clean.mode == "RGB" and max(clean.size) <= server.MAX_INIT_EDGE,
      f"{clean.mode} {clean.size}")

print("\n== 3) 伪装文件被拒绝 ==")
expect_reject("重命名的纯文本（.jpg 内容为 text）", data_url(b"this is not an image at all"))
expect_reject("SVG（XML 头）", data_url(b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"))
expect_reject("HTML 文件", data_url(b"<html><body>hi</body></html>"))
expect_reject("可执行头（MZ）", data_url(b"MZ\x90\x00\x03\x00\x00\x00"))
expect_reject("PDF 头", data_url(b"%PDF-1.7\n%...\n"))
expect_reject("ZIP/PK 头", data_url(b"PK\x03\x04\x14\x00\x00\x00"))
expect_reject("空负载", data_url(b""))
expect_reject("非法 base64", "data:image/jpeg;base64,!!!!not-base64!!!!")

print("\n== 4) data URL 白名单 ==")
png_buf = io.BytesIO()
Image.new("RGB", (300, 300), (200, 40, 40)).save(png_buf, "PNG")
expect_reject("data:image/svg+xml 被拒绝", "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode())
expect_reject("data:text/html 被拒绝", "data:text/html;base64," + base64.b64encode(b"<b>x</b>").decode())
check("合法 PNG data URL 仍可用", server._decode_init_image(data_url(png_buf.getvalue(), "image/png")) is not None)

print("\n== 5) 像素炸弹 / 体积上限 ==")
big = Image.new("RGB", (6000, 5000), (1, 2, 3))     # 30 MP > 24 MP 上限
buf = io.BytesIO()
big.save(buf, "PNG")
expect_reject(f"30MP 图像被拒绝（原始 {len(buf.getvalue())//1024//1024} MB）", data_url(buf.getvalue(), "image/png"), 413)
over = "A" * (server.MAX_UPLOAD_BYTES * 2 + 16)
expect_reject("超大 base64 载荷被拒绝", "data:image/png;base64," + over, 413)

print("\n== 6) 缩放：超大边长被压到上限 ==")
huge = Image.new("RGB", (4000, 3000), (90, 90, 90))  # 12 MP 合法但边长过大
buf = io.BytesIO()
huge.save(buf, "JPEG", quality=70)
out = server._decode_init_image(data_url(buf.getvalue()))
check("边长压到 MAX_INIT_EDGE", out is not None and max(out.size) == server.MAX_INIT_EDGE, f"{out.size}")

print(f"\n结果: {len(PASS)} 通过 / {len(FAIL)} 失败")
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
print("ALL PASS")
