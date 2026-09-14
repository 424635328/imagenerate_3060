/**
 * smoke_reconcile.mjs — 前端与后端"对账"的回归（jsdom）。
 *
 * 为什么要有它：本地历史只存在 localStorage，前端原本**没有任何与后端对账的机制**。
 * 真实事故（2026-09-15）：用户提交了一张 4096²（极清档，8.7 分钟）的图，页面中途刷新过，
 * 本地记录永远停在 running，画布一直显示占位符 —— 而后端早就把图写好了、接口也能正常取到。
 * 于是加了两件事，各对应一条断言：
 *   1. heal：本地非终态 + 后端已完成 → 改成 done，并**把图显示到画布上**；
 *   2. adopt：后端有、本地没有（别的标签/浏览器提交的）→ 收编进历史，可看可下载。
 *
 *   npm install --prefix "$env:TEMP/lsart-jsdom" jsdom
 *   $env:NODE_PATH="$env:TEMP\lsart-jsdom\node_modules"; node tools/smoke_reconcile.mjs
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SITE = path.join(ROOT, 'site');

async function loadJsdom() {
  try { return await import('jsdom'); } catch { /* fall through */ }
  const candidates = [];
  if (process.env.JSDOM_PATH) candidates.push(path.join(process.env.JSDOM_PATH, 'jsdom'));
  try { candidates.push(createRequire(import.meta.url).resolve('jsdom')); } catch { /* ignore */ }
  for (const candidate of candidates) {
    try {
      const mod = await import(pathToFileURL(candidate).href);
      return mod.default?.JSDOM ? mod.default : mod;
    } catch { /* next */ }
  }
  return null;
}

const JSDOM_MODULE = await loadJsdom();
if (!JSDOM_MODULE) {
  console.log('SKIP: jsdom not installed (see header for the one-line install).');
  process.exit(0);
}
const { JSDOM, VirtualConsole } = JSDOM_MODULE;

const errors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (error) => errors.push(`jsdomError: ${error.message}`));
virtualConsole.on('error', (message) => errors.push(`console.error: ${message}`));

const dom = new JSDOM(readFileSync(path.join(SITE, 'index.html'), 'utf8'), {
  url: 'https://example.test/', pretendToBeVisual: true,
  runScripts: 'outside-only', virtualConsole,
});
const { window } = dom;
const document = window.document;
const g = globalThis;
const define = (name, value) => Object.defineProperty(g, name, { value, writable: true, configurable: true });
define('window', window);
define('document', document);
define('navigator', window.navigator);
define('location', window.location);
define('localStorage', window.localStorage);
define('HTMLElement', window.HTMLElement);
define('Node', window.Node);
define('Event', window.Event);
define('getComputedStyle', window.getComputedStyle.bind(window));
define('requestAnimationFrame', (fn) => setTimeout(() => fn(Date.now()), 0));
define('cancelAnimationFrame', (id) => clearTimeout(id));
define('matchMedia', () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
window.matchMedia = g.matchMedia;

/* --- 预置：一条本地"卡在 running"的记录（模拟长任务中途刷新过页面） --- */
const STUCK = {
  id: 'job-heal', status: 'running', seed: 4242, count: 1, seeds: [4242], steps: 32,
  sampler: 'dpmpp2m_karras', res: '768', prompt: 'a snow peak at dawn', created: Date.now() - 600000,
  images: [{ i: 0, bytes: 0 }], hasInit: false,
};
window.localStorage.setItem('lsart_v3', JSON.stringify({
  theme: 'dark', settings: {}, jobs: [STUCK], favorites: [],
}));

const HEALTH = {
  queued: 0, running: 0, upload: true, max_upload_mb: 4,
  memory: { allocated_mb: 0, reserved_mb: 0 }, storage: { results_mb: 0, results_files: 0 },
  pipeline: { model_loaded: true, sampler: 'dpmpp2m_karras', adapter: 'v5b' },
  registry: { default: 'v5b', previous: 'v4', versions: 10, default_verified: true },
};
/* 后端视角：job-heal 其实早就完成了；job-adopt 是别的标签页提交的、本地没有 */
const REMOTE_JOBS = {
  queue: 0,
  jobs: [
    { id: 'job-adopt', status: 'done', prompt: 'a foggy valley', seed: 77, count: 1, seeds: [77],
      steps: 28, sampler: 'dpmpp2m_karras', res: '512', created: Date.now() / 1000, bytes: 51200,
      files: 1, adapter: 'v5b' },
    { id: 'job-heal', status: 'done', prompt: 'a snow peak at dawn', seed: 4242, count: 1, seeds: [4242],
      steps: 32, sampler: 'dpmpp2m_karras', res: '768', created: Date.now() / 1000 - 600, bytes: 1856042,
      files: 1, adapter: 'v5b' },
  ],
};
const requests = [];
g.fetch = async (url) => {
  const target = String(url);
  requests.push(target);
  const json = (payload, status = 200) =>
    new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json' } });
  if (target.includes('op=health')) return json(HEALTH);
  if (target.includes('op=models')) {
    return json({ ok: true, default: 'v5b', previous: 'v4', count: 1,
      models: [{ id: 'v5b', label: 'V5b', slogan: 's', note: 'n', state: 'promoted',
                 verified: true, text_encoder: true, default: true, sha256: 'abc' }] });
  }
  if (target.includes('op=jobs')) return json(REMOTE_JOBS);
  return json({});
};
window.fetch = g.fetch;
process.on('unhandledRejection', (reason) => errors.push(`unhandledRejection: ${reason}`));

const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok: !!ok, detail });
const $ = (selector) => document.querySelector(selector);
const tick = (ms = 120) => new Promise((resolve) => setTimeout(resolve, ms));

let store;
try {
  await import(pathToFileURL(path.join(SITE, 'js/main.js')).href);
  store = await import(pathToFileURL(path.join(SITE, 'js/store.js')).href);
  await tick(400);                     // 等 boot() 里的对账跑完
  check('boot: main.js 加载并启动', true);
} catch (error) {
  check('boot: main.js 加载并启动', false, String(error?.stack || error));
}

const jobs = store ? store.getState().jobs : [];
const healed = jobs.find((job) => job.id === 'job-heal');
const adopted = jobs.find((job) => job.id === 'job-adopt');

check('对账: 启动时确实向后端要了任务列表', requests.some((u) => u.includes('op=jobs')),
  requests.slice(-4).join(' | '));
check('heal: 本地卡在 running、后端已完成 → 状态被改成 done',
  healed?.status === 'done', `status=${healed?.status}`);
check('heal: 字节数与版本从后端补齐（下载/展示要用）',
  healed?.bytes === 1856042 && healed?.adapter === 'v5b',
  `bytes=${healed?.bytes} adapter=${healed?.adapter}`);
check('heal: 记录里写明了是对账救回来的（可解释）',
  /对账/.test(healed?.info || ''), String(healed?.info));
check('heal: **画布上真的显示了这张图**（否则用户仍以为没出图）',
  !!$('#mainImage') && /op=image|result/.test($('#mainImage').getAttribute('src') || '')
  && /job-heal/.test($('#mainImage').getAttribute('src') || ''),
  ($('#mainImage')?.getAttribute('src') || '(无 mainImage)').slice(0, 90));
check('adopt: 后端有、本地没有的任务被收编进历史',
  adopted?.status === 'done' && adopted?.seed === 77, JSON.stringify(adopted || null).slice(0, 90));
check('adopt: 收编的记录标了来源（↻ 从后端同步）',
  /后端同步/.test(adopted?.info || ''), String(adopted?.info));
check('adopt: 画廊里能看到这两张', document.querySelectorAll('.gallery .gi, .gallery .tile').length >= 2,
  String(document.querySelectorAll('.gallery .gi, .gallery .tile').length));

const fatal = errors.filter((message) => !/Not implemented: navigation/.test(message));
check('运行期无错误', fatal.length === 0, fatal.slice(0, 3).join(' | '));

const failed = results.filter((row) => !row.ok);
results.forEach((row) => console.log(`${row.ok ? 'PASS' : 'FAIL'}  ${row.name}${row.ok || !row.detail ? '' : `  → ${row.detail}`}`));
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
