import requests, time
def get(u, **k):
    for i in range(5):
        try:
            return requests.get(u, timeout=60, **k)
        except Exception:
            if i == 4: raise
            time.sleep(3)

b = 'https://landscape-art-demo.netlify.app'
checks = {
    '/extra.css': 'stage-grid',
    '/js/extras.js': 'initExtras',
    '/js/store.js': 'toggleFavorite',
    '/js/api.js': 'resultUrl',
}
for path, needle in checks.items():
    r = get(b + path)
    print(f'{path:18s} HTTP {r.status_code} {len(r.content):>7} B   contains "{needle}": {needle in r.text}')

h = get(b + '/?v=5').text
for needle in ['运行统计', '词库 · 检索 · 收藏', 'statGrid', 'histSearch', 'extra.css', 'js/main.js']:
    print(f'index.html contains {needle!r}:', needle in h)

P = b + '/.netlify/functions/proxy'
try:
    r = get(P + '?op=health')
    print('proxy health:', r.status_code, r.text[:150])
except Exception as e:
    print('proxy health ERR:', str(e)[:90])
