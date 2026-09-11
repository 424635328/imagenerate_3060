import os
import re, unicodedata
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── README 目录锚点校验（GitHub 风格：小写、空格→-、去标点）──
p = ROOT + "/README.md"
md = open(p, encoding='utf-8').read()
heads = re.findall(r'^##\s+(.+?)\s*$', md, re.M)
def slug(s):
    s = s.strip().lower()
    s = re.sub(r'[^\w\u4e00-\u9fff\s-]', '', s)   # 去标点（保留中英文数字空格连字符）
    s = s.replace(' ', '-')                       # GitHub：每个空格各换一个连字符（不合并）
    return s
valid = {slug(h) for h in heads}
toc = re.findall(r'\]\(#([^)]+)\)', md)
bad = [t for t in toc if t not in valid]
print('H2 headings:', len(heads))
print('TOC links  :', len(toc))
print('broken TOC :', bad if bad else 'none  OK')
print('headings sample:', heads[:4])

# ── 报告 HTML 的 <script> 语法检查（交给 node 之外用括号配平近似判断）──
h = open(ROOT + "/PROJECT_REPORT.html", encoding='utf-8').read()
scripts = re.findall(r'<script>([\s\S]*?)</script>', h)
open(ROOT + "/research/_report.js", 'w', encoding='utf-8').write('\n'.join(scripts))
print('report scripts:', len(scripts), 'chars:', sum(len(s) for s in scripts))

# ── README_NETLIFY 目录锚点 ──
p2 = ROOT + "/README_NETLIFY.md"
md2 = open(p2, encoding='utf-8').read()
h2 = re.findall(r'^##\s+(.+?)\s*$', md2, re.M)
valid2 = {slug(x) for x in h2}
toc2 = re.findall(r'\]\(#([^)]+)\)', md2)
bad2 = [t for t in toc2 if t not in valid2]
print('NETLIFY H2:', len(h2), 'TOC:', len(toc2), 'broken:', bad2 if bad2 else 'none  OK')
