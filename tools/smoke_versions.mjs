/**
 * Headless smoke test for site/versions.html — 版本评判台。
 *
 * 和 smoke_frontend.mjs 同一套办法（jsdom + 真实文件），但这里多验一层：
 * **用磁盘上真实的 site/data/versions.json 启动页面**，因此同时校验了
 * 资源打包（tools/make_version_gallery.py）与前端渲染是否对得上。
 *
 *   npm install --prefix "$env:TEMP/lsart-jsdom" jsdom
 *   $env:NODE_PATH="$env:TEMP/lsart-jsdom/node_modules"; node tools/smoke_versions.mjs
 *
 * Exit code 0 = 全部通过（或未安装 jsdom 时 SKIP），1 = 有失败项。
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

const dom = new JSDOM(readFileSync(path.join(SITE, 'versions.html'), 'utf8'), {
  url: 'https://example.test/versions.html',
  pretendToBeVisual: true,
  runScripts: 'outside-only',
  virtualConsole,
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
define('URL', window.URL);

/* 真实数据：直接读磁盘上的 versions.json，顺便校验打包产物存在。 */
const DATA = JSON.parse(readFileSync(path.join(SITE, 'data', 'versions.json'), 'utf8'));
const requested = [];
g.fetch = async (url) => {
  requested.push(String(url));
  return new Response(JSON.stringify(DATA), { status: 200, headers: { 'Content-Type': 'application/json' } });
};
window.fetch = g.fetch;
process.on('unhandledRejection', (reason) => errors.push(`unhandledRejection: ${reason}`));

const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok: !!ok, detail });
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const tick = (ms = 40) => new Promise((resolve) => setTimeout(resolve, ms));

/* ------------------------------------------------------------------ boot */
let mod;
try {
  mod = await import(pathToFileURL(path.join(SITE, 'js/versions.js')).href);
  await tick(80);
  check('boot: versions.js 加载并完成首屏渲染', !!window.vbState, String($('#vbStage')?.innerHTML.length || 0));
} catch (error) {
  check('boot: versions.js 加载并完成首屏渲染', false, String(error?.stack || error));
}

const state = window.vbState;
const versions = DATA.versions;

/* ------------------------------------------------------- 数据与协议渲染 */
check('数据: 载入的是磁盘上的真实 versions.json', requested.some((url) => url.includes('/data/versions.json')));
check('数据: 四个候选（V4/V5/V5b/V6q）', versions.map((v) => v.id).join(',') === 'V4,V5,V5b,V6q',
  versions.map((v) => v.id).join(','));
check('数据: 每个候选都有 24×2 = 48 格且路径唯一',
  versions.every((v) => v.cells.length === 48 && new Set(v.cells.map((c) => c.file)).size === 48));
check('协议: 24 条留出提示词 × 2 seed', DATA.protocol.prompt_count === 24 && DATA.protocol.seeds.length === 2);
check('协议: 标注了「初始噪声逐位相同」这一前提', /逐位相同/.test(DATA.protocol.aligned));

check('说明卡: 渲染了语料成立的理由', /没有参与过任何训练/.test($('#vbWhy').textContent));
check('说明卡: 渲染了结论要点', $$('#vbConclusion li').length >= 4);
check('徽标: 显示语料规模', /24 条留出提示词/.test($('#vbCorpus').textContent));

/* ------------------------------------------------------------ 指标表 */
const metricRows = $$('#vbMetrics tbody tr');
check('指标表: 每个候选一行', metricRows.length === 4);
check('指标表: 含 KID / CLIP / 细节量 / 文本编码器列', $$('#vbMetrics thead th').length === 7,
  String($$('#vbMetrics thead th').length));
check('指标表: V6q 的 val 标为不可比（而不是编一个数）',
  /不可比/.test($('#vbMetrics').textContent));
check('指标表: 表头带口径与陷阱说明',
  $$('#vbMetrics thead th').slice(1).every((th) => th.title.includes('陷阱')));
check('val 完整榜: 含基座与免训练融合', $$('#vbValTable tbody tr').length >= 8);
check('裁判: 明确标注判据无效 + 位置偏置', /判据无效/.test($('#vbJudgeInvalid').textContent)
  && DATA.judge.position_bias.presented_a_wins === DATA.judge.position_bias.total);
check('裁判: 位置偏置 = 100% 选 A 位', DATA.judge.position_bias.fraction === 1);

/* ------------------------------------------------------------ 四联图 */
check('并排模式: 渲染 4 个格子', $$('.vb-cell').length === 4);
check('并排模式: 图片指向打包好的 WebP',
  $$('.vb-cell img').every((img) => /\/img\/versions\/\w+\/p\d{2}_s\d{3}\.webp$/.test(img.getAttribute('src'))));
check('并排模式: 每格有版本名与指标', $$('.vb-cell .vb-name').length === 4
  && $$('.vb-cell .vb-metric').length === 4);
check('未盲测时标签可见', $$('.vb-cell.masked').length === 0);

/* -------------------------------------------------------------- 盲测 */
$('#vbBlind').click();
await tick();
check('盲测: 标签被遮罩', $$('.vb-cell.masked').length === 4);
check('盲测: 槽位 A–D 就位', $$('.vb-cell .vb-slot').map((el) => el.textContent[0]).join('') === 'ABCD');
const blindSlots = $$('.vb-cell').map((el) => el.dataset.version);
check('盲测: 顺序与未盲测不同（确实打乱了）',
  blindSlots.join(',') !== versions.map((v) => v.id).join(','), blindSlots.join(','));
check('盲测: 同一题顺序确定（可复现）',
  JSON.stringify(mod.blindOrder(4, state.prompt, DATA.protocol.seeds[0]))
  === JSON.stringify(mod.blindOrder(4, state.prompt, DATA.protocol.seeds[0])));

$('.vb-cell [data-pick]').click();
await tick(320);
check('盲测: 选择被记录（含当时的洗牌顺序，可复核）',
  Object.keys(state.picks).length === 1
  && Array.isArray(Object.values(state.picks)[0].order)
  && Object.values(state.picks)[0].mode === 'blind4');
check('盲测: 选完自动跳到下一题', state.prompt === 1 || state.seedIndex === 1,
  `prompt=${state.prompt} seedIndex=${state.seedIndex}`);
check('进度: 已评判计数与进度条同步', /已评判 1 \/ 48/.test($('#vbProgressLabel').textContent));

$('[data-tie]').click();
await tick();
check('平局: 记录 tie=true 且不计入已决题', Object.values(state.picks).some((row) => row.tie === true));
$('#vbBlind').click();
await tick();
check('揭晓: 遮罩移除、按钮文案回到「开始盲测」', $$('.vb-cell.masked').length === 0
  && /开始盲测/.test($('#vbBlind').textContent));

/* ---------------------------------------------------------- 擦除对照 */
$('#vbMode').value = 'wipe';
$('#vbMode').dispatchEvent(new window.Event('change', { bubbles: true }));
await tick();
check('擦除对照: 双图 + 分割线 + 滑杆就位', !!$('#vbWipeFrame') && !!$('#vbWipeTop')
  && !!$('#vbWipeHandle') && !!$('#vbWipeRange'));
check('擦除对照: 两图来自不同版本的不同文件',
  $('.vb-wipe-b').getAttribute('src') !== $('#vbWipeTop').getAttribute('src'));
$('#vbWipeRange').value = '30';
$('#vbWipeRange').dispatchEvent(new window.Event('input', { bubbles: true }));
check('擦除对照: 滑杆驱动 clip-path 与分割线',
  $('#vbWipeTop').style.clipPath.includes('70%') && $('#vbWipeHandle').style.left === '30%',
  $('#vbWipeTop').style.clipPath);
$('[data-pair="A"]').click();
await tick();
check('擦除对照: 两两选择被记录（mode=pair，含 A/B 身份）',
  Object.values(state.picks).some((row) => row.mode === 'pair' && row.pair?.length === 2));
$('[data-swap]').click();
await tick();
check('擦除对照: A/B 可交换', $('#vbWipeA').value === state.wipeA);

/* ------------------------------------------------------------ 统计口径 */
$('#vbMode').value = 'grid';
$('#vbMode').dispatchEvent(new window.Event('change', { bubbles: true }));
await tick(320);
check('统计: 渲染了 4 选 1 与两两两种口径的表', $$('#vbTally tbody tr').length >= 2);
check('统计: 出现了 Wilson 区间与判定列',
  /无法区分|高于随机/.test($('#vbTally').textContent) && /–\d+(\.\d+)?%/.test($('#vbTally').textContent),
  $('#vbTally').textContent.replace(/\s+/g, ' ').slice(0, 120));
check('统计: 4 选 1 的随机期望标为 25%', /随机期望是 25%/.test($('.vb-tally .vb-foot').textContent));

/* Wilson 与 tools/vlm_judge.py 同式：用文档里的真实数字交叉验证 */
const ci = mod.wilson(30, 48);
check('Wilson: wilson(30,48) = 0.484–0.748（与 vlm_judge.py 输出一致）',
  Math.abs(ci[0] - 0.4836) < 0.002 && Math.abs(ci[1] - 0.7478) < 0.002, ci.map((v) => v.toFixed(4)).join('–'));
check('Wilson: 区间含 50% 判为无法区分', mod.verdictFor(30, 48, 0.5).indistinguishable === true);
check('Wilson: 4 选 1 用 25% 作期望（30/48 应判为高于随机）',
  mod.verdictFor(30, 48, 0.25).indistinguishable === false);

/* ---------------------------------------------------------------- 超分 */
check('超分: 六列（HR + 5 方法）', $$('.vb-sr-col').length === 6);
check('超分: 每列 8 个格子', $$('.vb-sr-col')[0].querySelectorAll('img').length === 8);
check('超分: 标注了细节量与 PSNR 代价',
  /6\.5/.test($('#vbSrNote').textContent) && /PSNR/.test($('#vbSrNote').textContent));
check('超分: 表格含四个方法与 bicubic 基准', $$('#vbSrTable tbody tr').length === 5);

/* -------------------------------------------------------------- 空数据 */
check('导出: 有记录时可导出（点击不抛错）', (() => {
  try { $('#vbExport').click(); return true; } catch { return false; }
})());

/* ---------------------------------------------------------------- 收尾 */
check('运行期无错误', errors.length === 0, errors.slice(0, 3).join(' | '));

const failed = results.filter((row) => !row.ok);
results.forEach((row) => console.log(`${row.ok ? 'PASS' : 'FAIL'}  ${row.name}${row.ok || !row.detail ? '' : `  → ${row.detail}`}`));
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
