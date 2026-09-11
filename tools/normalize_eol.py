"""normalize_eol.py — 按 .gitattributes 统一行尾：文本文件 LF，Windows 启动器 CRLF。

只处理会被提交的文本文件（跳过 .git、模型、数据集、输出、结果）。
"""
import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT  # noqa: E402  (path comes from LANDSCAPE_ROOT / repo location)

os.chdir(ROOT)

LF_EXT = {'.py', '.js', '.html', '.htm', '.css', '.md', '.json', '.toml', '.cfg',
          '.yml', '.yaml', '.txt', '.svg', '.gitignore', '.gitattributes', ''}
CRLF_EXT = {'.bat', '.cmd', '.ps1'}
SKIP_DIRS = {'.git', 'models', 'dataset', 'dataset1024', 'outputs', 'results',
             'node_modules', '__pycache__', '.netlify'}

def targets():
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in LF_EXT or ext in CRLF_EXT:
                yield os.path.relpath(os.path.join(root, name), '.').replace('\\', '/')

changed = []
for path in targets():
    try:
        data = open(path, 'rb').read()
    except Exception:
        continue
    ext = os.path.splitext(path)[1].lower()
    want = b'\r\n' if ext in CRLF_EXT else b'\n'
    # Normalise: collapse CRLF/CR to LF, then apply the wanted ending.
    flat = data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    out = flat.replace(b'\n', want) if want == b'\r\n' else flat
    if out != data:
        open(path, 'wb').write(out)
        changed.append((path, 'CRLF' if want == b'\r\n' else 'LF'))

print(f'normalised {len(changed)} files')
for path, kind in changed[:40]:
    print(f'   {kind:4s} {path}')
if len(changed) > 40:
    print(f'   … {len(changed) - 40} more')

# verify
bad = []
for path in targets():
    data = open(path, 'rb').read()
    ext = os.path.splitext(path)[1].lower()
    crlf = data.count(b'\r\n'); lf = data.count(b'\n') - crlf
    if ext in CRLF_EXT:
        if lf: bad.append((path, f'CRLF={crlf} bare-LF={lf}'))
    elif crlf:
        bad.append((path, f'CRLF={crlf} LF={lf}'))
print('remaining mixed endings:', 'NONE' if not bad else bad[:10])
