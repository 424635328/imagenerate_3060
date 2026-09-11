"""scan_secrets.py — 扫描「将要入库的文件」中的敏感信息（令牌/密钥/个人路径/邮箱/IP）。"""
import os, re, subprocess, sys
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True, encoding='utf-8', errors='replace').stdout

files = [f for f in (git('ls-files').splitlines() + git('ls-files', '--others', '--exclude-standard').splitlines()) if f.strip()]
files = sorted(set(files))

PATTERNS = [
    ('Netlify token',   r'nfp_[A-Za-z0-9]{20,}'),
    ('GitHub token',    r'gh[pousr]_[A-Za-z0-9]{20,}'),
    ('HF token',        r'hf_[A-Za-z0-9]{20,}'),
    ('Replicate token', r'r8_[A-Za-z0-9]{20,}'),
    ('OpenAI key',      r'sk-[A-Za-z0-9]{20,}'),
    ('Google key',      r'AIza[A-Za-z0-9_\-]{20,}'),
    ('AWS key',         r'AKIA[0-9A-Z]{16}'),
    ('Slack token',     r'xox[baprs]-[A-Za-z0-9\-]{10,}'),
    ('私钥块',          r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    ('Bearer 头',       r'Bearer\s+[A-Za-z0-9._\-]{20,}'),
    ('Win 个人路径',    r'[A-Za-z]:\\+Users\\+[A-Za-z0-9._\-]+'),
    ('Win 项目绝对路径', r'[A-Za-z]:[\\/]+Download[\\/]+images'),
    ('邮箱',            r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}'),
    ('ngrok 隧道域名',  r'[a-z0-9\-]+\.ngrok-free\.(app|dev)'),
    ('Netlify 站点ID',  r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b'),
    ('硬编码口令字段',  r'(?i)(password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*["\'][^"\']{6,}["\']'),
]

TEXT_EXT = {'.py', '.md', '.html', '.htm', '.js', '.json', '.cfg', '.toml', '.txt', '.css', '.svg', '.yml', '.yaml', '.example', '.gitignore', ''}
hits = {}
scanned = 0
for f in files:
    ext = os.path.splitext(f)[1].lower()
    if ext not in TEXT_EXT:
        continue
    try:
        data = open(f, encoding='utf-8', errors='ignore').read()
    except Exception:
        continue
    scanned += 1
    for name, pat in PATTERNS:
        for m in re.finditer(pat, data):
            line = data[:m.start()].count('\n') + 1
            val = m.group(0)
            masked = val[:6] + '…' + val[-4:] if len(val) > 12 else val
            hits.setdefault(name, []).append((f, line, masked))

print(f'扫描文件数: {scanned} / 待入库 {len(files)}')
print('=' * 70)
if not hits:
    print('未发现任何敏感信息 ✅')
for name, lst in hits.items():
    print(f'\n【{name}】共 {len(lst)} 处')
    seen = set()
    for f, line, v in lst:
        key = (f, v)
        if key in seen:
            continue
        seen.add(key)
        print(f'   {f}:{line}   {v}')
        if len(seen) >= 12:
            print(f'   … 其余 {len(lst)-12} 处省略')
            break
