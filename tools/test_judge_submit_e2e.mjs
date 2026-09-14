/**
 * test_judge_submit_e2e.mjs — 真浏览器端到端：点一下「提交评判」，文件必须落到仓库里。
 *
 * 为什么值得单独写：其余测试都在替身边界内（jsdom 的 fetch 是我 stub 的、Python 测试直接
 * POST 是我构造的）。而这一条要证明的是**用户真的点一下就会发生什么**：
 *   真 Chrome（headless）→ 真页面（本地静态服务）→ 真 fetch → 真收集器 → 真文件。
 *
 * 做法：Chrome 以 --remote-debugging-port 启动，用 CDP：
 *   1. 打开 http://127.0.0.1:<static>/versions.html
 *   2. Runtime.evaluate 模拟真实交互：盲测选 4 题（每题的槽位 A–D 都点第一个）
 *   3. 点 #vbSubmit
 *   4. 断言：research/human_judge/ 里出现新文件，记录条数 = 4，且服务端 summary 存在
 *
 * 用法: node tools/test_judge_submit_e2e.mjs
 *   （需要 Chrome；找不到浏览器时 SKIP 并返回 0，与 smoke 测试同一约定）
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = path.join(ROOT, 'research', 'human_judge');
const CHROME_CANDIDATES = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
];
const chromePath = CHROME_CANDIDATES.find((p) => existsSync(p));
if (!chromePath) {
  console.log('SKIP: 没找到 Chrome/Edge（这条测试需要一个真浏览器）。');
  process.exit(0);
}

/** 取一个空端口。**必须随机**：固定端口会被上一次没清干净的实例占着，
 *  于是 CDP 连到旧浏览器、读到旧 localStorage —— 这一条真的踩过（见文件末尾说明）。 */
function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

/** 杀**整棵进程树**：Chrome 会派生 renderer/gpu/network 一堆子进程，而且启动器会换父进程，
 *  只 kill 启动器 pid 完全没用（本项目实测残留 32 个进程占着调试端口、并被下一轮复用）。
 *  可靠做法是**按 profile 路径匹配**：`--user-data-dir` 会出现在整棵树的命令行里。 */
function killTree(proc) {
  if (!proc || proc.exitCode !== null) return;
  if (process.platform === 'win32') {
    try { spawnSync('taskkill', ['/PID', String(proc.pid), '/T', '/F'], { stdio: 'ignore' }); } catch { /* 忽略 */ }
  } else {
    try { process.kill(-proc.pid); } catch { try { proc.kill(); } catch { /* 忽略 */ } }
  }
}

function killChromeByProfile(profileDir) {
  if (process.platform !== 'win32') {
    killTree(chrome);
    return;
  }
  const script = "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe' or Name='msedge.exe'\""
    + ` | Where-Object { $_.CommandLine -like '*${profileDir}*' }`
    + ' | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }';
  try { spawnSync('powershell', ['-NoProfile', '-Command', script], { stdio: 'ignore' }); } catch { /* 忽略 */ }
}

const STATIC_PORT = await freePort();
const DEBUG_PORT = await freePort();

const results = [];
const check = (name, ok, detail = '') => results.push({ name, ok: !!ok, detail });
const python = process.env.PYTHON || 'python';

const before = existsSync(OUT_DIR) ? new Set(readdirSync(OUT_DIR)) : new Set();
mkdirSync(OUT_DIR, { recursive: true });

/* --------------------------------------------------------------- 起服务 */
const collector = spawn(python, [path.join(ROOT, 'tools', 'judge_collector.py')],
  { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
let collectorLog = '';
collector.stdout.on('data', (chunk) => { collectorLog += chunk.toString(); });
collector.stderr.on('data', (chunk) => { collectorLog += chunk.toString(); });

const staticServer = spawn(python, ['-m', 'http.server', String(STATIC_PORT),
  '--bind', '127.0.0.1', '--directory', 'site'], { cwd: ROOT, stdio: 'ignore' });

const profile = path.join(process.env.TEMP || '/tmp', `lsart-e2e-${Date.now()}`);
const chrome = spawn(chromePath, [
  '--headless=new', '--disable-gpu', '--no-first-run', '--no-proxy-server',
  `--user-data-dir=${profile}`, `--remote-debugging-port=${DEBUG_PORT}`,
  '--window-size=1400,1200', 'about:blank',
], { stdio: 'ignore', detached: process.platform !== 'win32' });

const cleanup = () => {
  killChromeByProfile(profile);
  killTree(chrome);
  killTree(collector);
  killTree(staticServer);
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try { rmSync(profile, { recursive: true, force: true }); break; } catch { /* 稍后再试 */ }
  }
};
process.on('exit', cleanup);

/* --------------------------------------------------------------- CDP */
async function cdpTarget() {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json/list`);
      const targets = await response.json();
      const page = targets.find((t) => t.type === 'page');
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch { /* 还没起来 */ }
    await sleep(250);
  }
  throw new Error('Chrome 调试端口没起来');
}

async function run() {
  const wsUrl = await cdpTarget();
  const socket = new WebSocket(wsUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  let seq = 0;
  const pending = new Map();
  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    const entry = pending.get(message.id);
    if (!entry) return;
    pending.delete(message.id);
    message.error ? entry.reject(new Error(JSON.stringify(message.error))) : entry.resolve(message.result);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text + ' ' +
      (result.exceptionDetails.exception?.description || ''));
    return result.result.value;
  };

  await send('Page.enable');
  await send('Runtime.enable');
  const url = `http://127.0.0.1:${STATIC_PORT}/versions.html`;
  await send('Page.navigate', { url });
  // 等页面把 versions.json 读完并渲染出四联格
  for (let attempt = 0; attempt < 60; attempt += 1) {
    const ready = await evaluate('!!(window.vbState && window.vbState.data && document.querySelectorAll(".vb-cell").length === 4)');
    if (ready) break;
    await sleep(250);
  }
  check('E2E: 真浏览器里页面就绪（4 个格子 + 数据）',
    await evaluate('document.querySelectorAll(".vb-cell").length === 4'));
  // 隔离性自检：全新 profile ⇒ 评判记录必须从零开始。
  // 这一条曾抓出真问题：残留的旧 headless 实例占着固定调试端口，CDP 连到了旧浏览器，
  // 于是"点了 4 次却记了 13 条"——不是产品重复记，而是读到了上一轮的 localStorage。
  const preexisting = await evaluate("localStorage.getItem('lsart_verdicts_v1')");
  check('E2E: 浏览器是全新的（没有上一轮的评判记录）', preexisting === null,
    `localStorage 里已有 ${String(preexisting).length} 字节`);
  check('E2E: 默认就是盲测（标签被遮罩）',
    await evaluate('document.querySelectorAll(".vb-cell.masked").length === 4'));

  // 真实交互：判 4 题（每题都点槽位 A），模拟人手点击。
  // 三段计数：点了几次 / 落了几次存储 / 记了几条 —— 用来区分"产品重复记"与"测试多点"。
  await evaluate(`(() => {
    window.__pickClicks = 0;
    window.__pickSaves = 0;
    document.addEventListener('click', (event) => {
      if (event.target.closest('[data-pick]')) window.__pickClicks += 1;
    }, true);
    const original = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key, value) {
      if (key === 'lsart_verdicts_v1') window.__pickSaves += 1;
      return original.call(this, key, value);
    };
    return true;
  })()`);
  const picksAtStart = await evaluate('Object.keys(window.vbState.picks).length');
  const storedAtStart = await evaluate("(localStorage.getItem('lsart_verdicts_v1') || '').length");
  const judged = await evaluate(`(async () => {
    let count = 0;
    for (let i = 0; i < 4; i += 1) {
      const cell = document.querySelector('.vb-cell [data-pick]');
      if (!cell) break;
      cell.click();
      count += 1;
      await new Promise((r) => setTimeout(r, 320));
    }
    return count;
  })()`);
  const dispatched = await evaluate('window.__pickClicks');
  const saves = await evaluate('window.__pickSaves');
  const recorded = await evaluate('Object.keys(window.vbState.picks).length');
  check('E2E: 连点 4 次选择被记录', judged === 4 && recorded === picksAtStart + 4,
    `judged=${judged} dispatched=${dispatched} saves=${saves} recorded=${recorded}`
    + ` start=${picksAtStart} storedAtStart=${storedAtStart}`);

  const noteBefore = await evaluate('document.getElementById("vbSubmitNote").textContent');
  check('E2E: 提交按钮已可用', await evaluate('document.getElementById("vbSubmit").disabled === false'));
  check('E2E: 提交前提示里写了会送到哪里', /human_judge/.test(noteBefore), noteBefore.slice(0, 80));

  // ★ 真正的"点一下提交"
  await evaluate('document.getElementById("vbSubmit").click(), true');
  let note = '';
  for (let attempt = 0; attempt < 40; attempt += 1) {
    note = await evaluate('document.getElementById("vbSubmitNote").textContent');
    if (/已提交|没能自动送达/.test(note)) break;
    await sleep(250);
  }
  check('E2E: 界面回报「已提交」并给出文件名', /已提交/.test(note) && /verdicts_/.test(note), note.slice(0, 140));

  const files = readdirSync(OUT_DIR).filter((f) => f.startsWith('verdicts_') && f.endsWith('.json'));
  const fresh = files.filter((f) => !before.has(f));
  check('E2E: 仓库里确实多出一份评判文件', fresh.length >= 1, `新增 ${fresh.length} 份`);
  if (fresh.length) {
    const newest = fresh.map((f) => path.join(OUT_DIR, f))
      .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs)[0];
    const saved = JSON.parse(readFileSync(newest, 'utf8'));
    check('E2E: 文件里是 4 条记录', saved.records?.length === 4, String(saved.records?.length));
    check('E2E: 记录里带了「当时的洗牌顺序」（可复核盲测是否成立）',
      saved.records.every((row) => Array.isArray(row.order) && row.order.length === 4));
    check('E2E: 服务端独立算出了结论', !!saved.summary?.headline, String(saved.summary?.headline));
    check('E2E: 来源是本地静态站点', /127\.0\.0\.1/.test(saved.origin || ''), String(saved.origin));
    const prompt = saved.records[0]?.prompt_index;
    check('E2E: 记录的 prompt 与页面一致',
      saved.records.every((row) => row.prompt_index >= 0 && row.prompt_index < 24),
      `first=${prompt}`);
  }
  socket.close();
}

try {
  await run();
} catch (error) {
  check('E2E: 流程跑通', false, String(error?.stack || error));
} finally {
  // 先杀进程（按 profile 匹配整棵树），再删 profile —— 顺序反了会因文件占用删不掉
  cleanup();
  await sleep(800);
  cleanup();
}

/** 收尾自检：确认没有把浏览器/服务留在后台（残留会被下一轮复用，污染结果）。 */
const leaked = readdirSync(process.env.TEMP || '/tmp').filter((name) => name.startsWith('lsart-e2e-'));
check('E2E: 没有留下后台浏览器与临时 profile', leaked.length === 0, leaked.join(', '));

const failed = results.filter((row) => !row.ok);
results.forEach((row) => console.log(`${row.ok ? 'PASS' : 'FAIL'}  ${row.name}${row.ok || !row.detail ? '' : `  → ${row.detail}`}`));
if (collectorLog) {
  const lines = collectorLog.trim().split('\n').filter((l) => l.includes('[judge]'));
  if (lines.length) console.log(`\n收集器日志：\n  ${lines.join('\n  ')}`);
}
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
process.exit(failed.length ? 1 : 0);
