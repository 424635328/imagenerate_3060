/* Landscape·Art — app-shell service worker.
 *
 * Scope: static assets ONLY (CSS/JS/images + the HTML shell).  Every API call
 * goes through /.netlify/functions/proxy and is deliberately never cached or
 * replayed here.  Bump CACHE_VERSION whenever the shell file list changes.
 *
 * Strategy:
 *   navigate        → network first, fall back to cached index (offline shell)
 *   static asset    → stale-while-revalidate (instant second load, fresh later)
 *   anything else   → passthrough
 */
const CACHE_VERSION = 'lsart-shell-v7';
const SHELL = [
  '/',
  '/index.html',
  '/versions.html',
  '/gallery.html',
  '/styles.css',
  '/extra.css',
  '/css/tokens.css',
  '/css/palette.css',
  '/css/studio.css',
  '/css/motion.css',
  '/css/mobile.css',
  '/css/scenes.css',
  '/css/composer.css',
  '/css/ab.css',
  '/css/shell.css',
  '/css/advisor.css',
  '/css/versions.css',
  '/data/versions.json',
  '/js/config.js',
  '/js/api.js',
  '/js/store.js',
  '/js/ui.js',
  '/js/extras.js',
  '/js/studio.js',
  '/js/palette.js',
  '/js/motion.js',
  '/js/upload.js',
  '/js/scenes.js',
  '/js/scene-art.js',
  '/js/composer.js',
  '/js/ab.js',
  '/js/shell.js',
  '/js/advisor.js',
  '/js/main.js',
  '/js/versions.js',
  '/makoto.svg',
  '/makoto-thunder.svg',
  '/makoto-180.png',
];

const STATIC_PREFIXES = ['/css/', '/js/', '/img/', '/data/', '/makoto', '/styles.css', '/extra.css'];

function isStatic(url) {
  if (url.origin !== location.origin) return false;
  return STATIC_PREFIXES.some((p) => url.pathname.startsWith(p));
}

async function putCache(request, response) {
  if (!response || response.status !== 200) return response;
  const cache = await caches.open(CACHE_VERSION);
  cache.put(request, response.clone());
  return response;
}

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE_VERSION);
    await Promise.allSettled(SHELL.map((url) =>
      fetch(url, { cache: 'no-cache' }).then((r) => (r.ok ? cache.put(url, r) : Promise.reject(r.status)))));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;               // never touch POST/DELETE

  const url = new URL(request.url);

  // navigation: network first, cached shell as offline fallback
  if (request.mode === 'navigate') {
    event.respondWith((async () => {
      try {
        return await putCache(request, await fetch(request));
      } catch (error) {
        // 离线回退必须**先找回这一条路径自己**：以前对所有导航都回退
        // /index.html，于是 /versions 一旦碰上网络抖动（或本机代理坏掉）就会
        // 变成"打开工作台"——用户看到的是「后端离线」，而不是他点的那一页。
        const path = new URL(request.url).pathname;
        for (const key of [request, `${path}.html`, '/index.html']) {
          const hit = await caches.match(key);
          if (hit) return hit;
        }
        return Response.error();
      }
    })());
    return;
  }

  if (!isStatic(url)) return;                         // API etc.: passthrough

  event.respondWith((async () => {
    const cached = await caches.match(request);
    const network = fetch(request)
      .then((r) => putCache(request, r))
      .catch(() => cached || Response.error());
    return cached || network;
  })());
});
