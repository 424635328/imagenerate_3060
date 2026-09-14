/**
 * Headless smoke test for site/ — boots the real index.html and every ES module
 * inside jsdom, then asserts the wiring that syntax/asset checks cannot see:
 * tab panels, scene grid + generated artwork, composer tokens, A/B submissions,
 * gallery/stat rendering, command palette.
 *
 * jsdom is deliberately NOT a project dependency.  Install it out of tree:
 *   npm install --prefix "$env:TEMP/lsart-jsdom" jsdom
 *   $env:NODE_PATH="$env:TEMP/lsart-jsdom/node_modules"; node tools/smoke_frontend.mjs
 *
 * Exit code 0 = all checks passed, 1 = at least one failed, 0 = skipped (no jsdom).
 */
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SITE = path.join(ROOT, 'site');

/* Resolve jsdom three ways: local install, JSDOM_PATH env, then the CJS
   resolver (which is what honours NODE_PATH for an out-of-tree install). */
async function loadJsdom() {
  try {
    return await import('jsdom');
  } catch { /* fall through */ }
  const candidates = [];
  if (process.env.JSDOM_PATH) candidates.push(path.join(process.env.JSDOM_PATH, 'jsdom'));
  try {
    candidates.push(createRequire(import.meta.url).resolve('jsdom'));
  } catch { /* not resolvable */ }
  for (const candidate of candidates) {
    try {
      const mod = await import(pathToFileURL(candidate).href);
      return mod.default?.JSDOM ? mod.default : mod;
    } catch { /* try the next candidate */ }
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
  url: 'https://example.test/',
  pretendToBeVisual: true,
  runScripts: 'outside-only',
  virtualConsole,
});
const { window } = dom;
const document = window.document;

/* Mirror the browser globals the modules touch (they run in Node, not jsdom).
   defineProperty because Node ≥21 exposes some of these as read-only getters. */
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

const requests = [];
const HEALTH = {
  queued: 0, running: 0, upload: true, max_upload_mb: 6,
  memory: { allocated_mb: 0, reserved_mb: 0 }, storage: { results_mb: 0, results_files: 0 },
  pipeline: { model_loaded: false, sampler: 'dpmpp2m_karras' },
};
g.fetch = async (url, options = {}) => {
  const target = String(url);
  requests.push({ url: target, body: options.body ? JSON.parse(options.body) : null });
  const json = (payload, status = 200) =>
    new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json' } });
  if (target.includes('op=health')) return json(HEALTH);
  if (target.includes('op=generate')) {
    return json({ job_id: `job-${requests.length}`, status: 'queued', count: 2, seeds: [7, 8], eta: 5 });
  }
  if (target.includes('op=job')) return json({ status: 'running', progress: 0.5, stage: 'denoise' });
  return json({});
};
window.fetch = g.fetch;

process.on('unhandledRejection', (reason) => errors.push(`unhandledRejection: ${reason}`));

/* ------------------------------------------------------------------ checks */
const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok: !!ok, detail });

let store;
try {
  await import(pathToFileURL(path.join(SITE, 'js/main.js')).href);
  store = await import(pathToFileURL(path.join(SITE, 'js/store.js')).href);
  await new Promise((resolve) => setTimeout(resolve, 60));   // let boot()'s async tail settle
  check('boot: main.js loads and boots', true);
} catch (error) {
  check('boot: main.js loads and boots', false, String(error?.stack || error));
}

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const panel = (side, name) => $(`.tab-panel[data-shell="${side}"][data-panel="${name}"]`);
const visible = (element) => !!element && element.hidden === false;

if ($('#scenes')) {
  check('shell: left create panel visible, params hidden', visible(panel('left', 'create')) && !visible(panel('left', 'params')));
  check('shell: right queue panel visible, gallery hidden', visible(panel('right', 'queue')) && !visible(panel('right', 'gallery')));

  $('.panel-tabs[data-shell="left"] [data-panel="params"]')?.click();
  check('shell: switching left tab keeps right column visible',
    visible(panel('left', 'params')) && !visible(panel('left', 'create')) && visible(panel('right', 'queue')));
  $('.panel-tabs[data-shell="left"] [data-panel="create"]')?.click();

  const cards = $$('#scenes .scene-card');
  check('scenes: 28 cards rendered', cards.length === 28, `got ${cards.length}`);
  check('scenes: every card has artwork', $$('#scenes .scene-img').length === 28,
    `got ${$$('#scenes .scene-img').length}`);

  const search = $('#sceneSearch');
  search.value = 'aurora';
  search.dispatchEvent(new window.Event('input'));
  const shown = cards.filter((card) => !card.hidden).length;
  check('scenes: search filters the grid', shown > 0 && shown < 28, `${shown} visible`);
  search.value = '';
  search.dispatchEvent(new window.Event('input'));

  const first = cards[0];
  const phrase = first.querySelector('.scene-en').textContent;
  first.click();
  check('scenes: picking a card writes its phrase into #prompt', $('#prompt').value === phrase);

  const star = cards[0].querySelector('[data-fav]');
  const favName = star.dataset.fav;
  star.click();
  check('scenes: star persists to settings.sceneFavs',
    (store.getState().settings.sceneFavs || []).includes(favName));
  $$('#scenes [data-fav]').find((node) => node.dataset.fav === favName)?.click();

  check('composer: 11 phrase groups rendered', $$('#composerGrid .composer-group').length === 11,
    `got ${$$('#composerGrid .composer-group').length}`);
  check('composer: tokens reflect the prompt', $$('#composerTokens .composer-token').length > 0);

  const chip = $$('#composerGrid .chip')[0];
  const chipPhrase = chip.dataset.phrase;
  chip.click();
  check('composer: clicking a phrase appends it', $('#prompt').value.includes(chipPhrase));
  const token = $$('#composerTokens .composer-token')
    .find((node) => node.textContent.includes(chipPhrase.slice(0, 12)));
  token?.querySelector('.tok-x').click();
  check('composer: × removes the token again', !$('#prompt').value.includes(chipPhrase));

  const ab = $('#abMount');
  check('ab: lab rendered with two prompt boxes', !!ab?.querySelector('[data-role="a"]') && !!ab?.querySelector('[data-role="b"]'));
  ab.querySelector('[data-role="a"]').value = 'a misty lake at dawn';
  ab.querySelector('[data-role="b"]').value = 'a misty lake at dusk';
  ab.querySelector('[data-role="run"]').click();
  await new Promise((resolve) => setTimeout(resolve, 80));
  const generates = requests.filter((r) => r.url.includes('op=generate'));
  check('ab: submits both variants', generates.length === 2, `got ${generates.length}`);
  check('ab: same seed baseline, batch 2, distinct prompts',
    generates.length === 2 && generates[0].body.seed === generates[1].body.seed
    && generates.every((r) => r.body.batch === 2)
    && generates[0].body.prompt !== generates[1].body.prompt);

  store.addJob({
    id: 'smoke-job', status: 'done', seed: 42, count: 1, seeds: [42], steps: 24,
    sampler: 'dpmpp2m_karras', res: '512', prompt: 'smoke test scene', created: Date.now(),
    seconds: 6, bytes_each: [12345], images: [{ i: 0, bytes: 12345 }],
  });
  $('#galleryFilter button[data-filter="all"]').click();
  await new Promise((resolve) => setTimeout(resolve, 30));
  check('gallery: renders history tiles', $$('#gallery .tile').length >= 1, `got ${$$('#gallery .tile').length}`);
  check('stats: stat grid populated', $$('#statGrid .stat-card').length === 6,
    `got ${$$('#statGrid .stat-card').length}`);

  const { openPalette } = await import(pathToFileURL(path.join(SITE, 'js/palette.js')).href);
  openPalette();
  const countText = $('.cmdk .cmdk-count')?.textContent || '';
  const total = Number(countText.split('/')[1]);
  check('palette: ≥10 commands registered', total >= 10, `count="${countText}"`);
  check('palette: renders rows', $$('.cmdk .cmdk-row').length > 0);

  /* ---- 参数顾问：推荐值必须是算出来的，而不是硬编码文案 ---- */
  const advisor = await import(pathToFileURL(path.join(SITE, 'js/advisor.js')).href);
  check('advisor: 参数字格都挂了说明', $$('.advice').length >= 8, `got ${$$('.advice').length}`);
  check('advisor: 汇总面板已注入', !!$('#advisorSummary .advisor-metrics'));
  check('advisor: 作用说明写到了 label 的 title 上',
    ($('#steps')?.closest('.field')?.querySelector('label')?.title || '').includes('去噪步数'));

  const satKarras = advisor.saturationSteps('dpmpp2m_karras');
  const satSde = advisor.saturationSteps('dpmpp2m_sde');
  const satLcm = advisor.saturationSteps('lcm');
  check('advisor: 饱和步数随采样器而变（同一公式推导）', satKarras < satSde && satLcm < satKarras,
    `lcm=${satLcm} karras=${satKarras} sde=${satSde}`);

  const cheap = advisor.estimateSeconds({ res: '512', steps: 12, clarity: 'std', aspect: '方' });
  const rich = advisor.estimateSeconds({ res: '640', steps: 24, clarity: 'std', aspect: '方' });
  const heavy = advisor.estimateSeconds({ res: '640', steps: 24, clarity: '4k', aspect: '方' });
  check('advisor: 成本随步数与清晰度单调增长', cheap < rich && rich < heavy,
    `${cheap.toFixed(1)} < ${rich.toFixed(1)} < ${heavy.toFixed(1)}`);

  const vramBase = advisor.estimateVram({ res: '512', clarity: 'std' });
  const vram4k = advisor.estimateVram({ res: '640', clarity: '4k' });
  check('advisor: 显存估计随清晰度上升（真在算）', vram4k > vramBase,
    `${vramBase.toFixed(1)} → ${vram4k.toFixed(1)} GB`);

  $('#steps').value = '58';
  advisor.refresh();
  $('#advisorSummary [data-advisor="apply"]').click();
  check('advisor: 一键采用推荐值生效', $('#steps').value !== '58', `steps → ${$('#steps').value}`);
  const summary = advisor.lastSummary();
  check('advisor: 汇总含成本/显存/饱和点',
    !!summary && summary.seconds > 0 && summary.vram > 0 && summary.saturated > 0,
    summary ? `${summary.seconds.toFixed(1)}s / ${summary.vram.toFixed(1)}GB / ${summary.saturated}步` : 'none');
}

check('runtime: no errors thrown', errors.length === 0, errors.slice(0, 3).join(' | '));

/* ------------------------------------------------------------------ report */
let failed = 0;
for (const { name, ok, detail } of results) {
  if (!ok) failed += 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok || !detail ? '' : `  [${detail}]`}`);
}
console.log(`\n${results.length - failed}/${results.length} checks passed`);
process.exit(failed ? 1 : 0);
