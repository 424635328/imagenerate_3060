import os, re
ROOT = os.environ.get("LANDSCAPE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
h = open(ROOT + "/PROJECT_REPORT.html", encoding='utf-8').read()
ids = set(re.findall(r'<section id="([^"]+)"', h))
navs = re.findall(r'<a href="#([^"]+)"', h)
print('sections:', sorted(ids))
print('nav hrefs:', navs)
missing = [n for n in navs if n not in ids]
print('missing anchors:', missing if missing else 'none  OK')
imgs = re.findall(r'<img[^>]+src="([^"]+)"', h)
print('img count:', len(imgs))
base = ROOT + "/"
bad = [s for s in imgs if not s.startswith('http') and not os.path.exists(base + s)]
print('missing image files:', bad if bad else 'none  OK')
for k in ['#v4', 'id="docs"', 'README_NETLIFY', 'bench_compare', 'landscape-art-demo']:
    print(f'has {k}:', k in h)
print('size KB:', len(h) // 1024)
