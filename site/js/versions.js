/**
 * versions.js — 版本评判台（site/versions.html）
 *
 * 这一页存在的理由：机器判据全部走到"无法区分"，而其中一条（VLM 盲测裁判）已被证明
 * **无效**（Qwen2-VL-2B 144/144 次都选先出现的那张）。剩下唯一还没用尽的判据是人的眼睛，
 * 所以这页只做三件事：
 *   1. 把同一 (prompt, seed) 的四个版本**像素对齐**摆出来（初始噪声逐位相同）；
 *   2. 提供可复现的**盲测**（标签换成 A/B/C/D、顺序按题号确定性打乱）；
 *   3. 用与 tools/vlm_judge.py **同一个 Wilson 公式**统计你的选择，并给出「无法区分」判定。
 *
 * 不联网、不依赖后端：所有数据来自 /data/versions.json（tools/make_version_gallery.py 生成），
 * 你的选择只存在浏览器 localStorage 里，导出由你自己决定给谁。
 */
import * as motion from './motion.js';
import * as store from './store.js';

const $ = (selector) => document.querySelector(selector);
const PICK_KEY = 'lsart_verdicts_v1';
const PREF_KEY = 'lsart_judge_prefs_v1';
const SLOTS = ['A', 'B', 'C', 'D'];
/* 本机收集器（tools/judge_collector.py）。用 ?judge=http://127.0.0.1:8899 可换端口。 */
const COLLECTOR = new URLSearchParams(location.search).get('judge') || 'http://127.0.0.1:8787';

const state = {
  data: null,
  prompt: 0,
  seedIndex: 0,
  mode: 'grid',
  blind: true,             // 默认就盲测：不盲测的"评判"没有证据价值
  wipeA: 'V4',
  wipeB: 'V5b',
  picks: {},              // key → {prompt, seed, mode, order, winner, tie}
  submit: { state: 'idle', message: '' },
};

/* ------------------------------------------------------------------ 偏好 */

function loadPrefs() {
  try {
    const prefs = JSON.parse(localStorage.getItem(PREF_KEY) || '{}');
    if (typeof prefs.blind === 'boolean') state.blind = prefs.blind;
    if (prefs.mode === 'wipe' || prefs.mode === 'grid') state.mode = prefs.mode;
  } catch { /* 用默认值 */ }
}

function savePrefs() {
  localStorage.setItem(PREF_KEY, JSON.stringify({ blind: state.blind, mode: state.mode }));
}

/* ------------------------------------------------------------------ 统计 */

/** Wilson 95% 区间 —— 与 tools/vlm_judge.py 的实现逐式对应（同一口径）。 */
export function wilson(wins, total, z = 1.96) {
  if (!total) return [0, 1];
  const phat = wins / total;
  const denominator = 1 + (z * z) / total;
  const centre = (phat + (z * z) / (2 * total)) / denominator;
  const margin = (z * Math.sqrt((phat * (1 - phat)) / total + (z * z) / (4 * total * total))) / denominator;
  return [Math.max(0, centre - margin), Math.min(1, centre + margin)];
}

/** 确定性 PRNG（mulberry32）：同一题永远得到同一个打乱顺序 ⇒ 导出可复核。 */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function blindOrder(count, prompt, seed) {
  const rand = rng(prompt * 7919 + seed);
  const order = Array.from({ length: count }, (_, index) => index);
  for (let i = count - 1; i > 0; i -= 1) {
    const j = Math.floor(rand() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

const cellKey = (prompt, seed) => `p${String(prompt).padStart(2, '0')}_s${seed}`;

/* 数据里的文案用 `**强调**` 标注重点（与 docs/ 里的 Markdown 一致）。
   往 DOM 里插这些字符串时统一走这两个助手：
     emphasize() —— 先转义再渲染强调（可安全用于 innerHTML）
     plain()     —— 去掉标记（用于 title 等纯文本属性） */
const escapeHtml = (text) => String(text).replace(/[&<>"]/g,
  (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[char]));
const emphasize = (text) => escapeHtml(text).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
const plain = (text) => String(text).replace(/\*\*/g, '');

/** 判定：区间包含"随机期望"即无法区分（4 选 1 期望 25%，两两期望 50%）。 */
export function verdictFor(wins, total, chance) {
  const [low, high] = wilson(wins, total);
  return { low, high, indistinguishable: low <= chance && chance <= high };
}

/* ------------------------------------------------------------------ 数据 */

const V = () => state.data.versions;
const versionById = (id) => V().find((item) => item.id === id);
const seed = () => state.data.protocol.seeds[state.seedIndex];
const img = (versionId, prompt, seedValue) =>
  `/img/versions/${versionId}/p${String(prompt).padStart(2, '0')}_s${seedValue}.webp`;

function loadPicks() {
  try {
    const raw = JSON.parse(localStorage.getItem(PICK_KEY) || '{}');
    state.picks = raw && typeof raw === 'object' ? (raw.records ? Object.fromEntries(
      raw.records.map((row) => [cellKey(row.prompt, row.seed), row])) : {}) : {};
  } catch { state.picks = {}; }
}

function savePicks() {
  const records = Object.values(state.picks).sort((a, b) =>
    (a.prompt - b.prompt) || (a.seed - b.seed) || a.mode.localeCompare(b.mode));
  localStorage.setItem(PICK_KEY, JSON.stringify({ schema: 1, built_for: 'site/versions.html', records }));
}

/* ------------------------------------------------------------------ 渲染 */

function renderIntro() {
  const data = state.data;
  const protocol = data.protocol;
  const align = protocol.alignment;
  $('#vbCorpus').textContent =
    `${protocol.prompt_count} 条留出提示词 × ${protocol.seeds.length} seed × ${V().length} 个版本`;
  $('#vbWhy').innerHTML =
    `每条提示词都<b>没有参与过任何训练</b>，所以它测的是泛化而不是背诵。`
    + `生成时每个 (prompt, seed) 都用 <code>manual_seed(${protocol.seeds.join('/')})</code> 起同一份噪声，`
    + `四个版本共用同一基座、同一步数（${protocol.steps}）与 CFG（${protocol.cfg}）、同一分辨率（${protocol.res}px），`
    + `因此 <b>同一题的四张图初始噪声逐位相同</b>：差异只来自权重本身`
    + `（UNet LoRA；其中 V4 与 V5b 还额外微调过文本编码器）。这正是可以直接叠图对照的原因。`;
  if (align) {
    // 前提不能只靠断言：这里给的是实测相关（含"不同 prompt"的基线对照）
    $('#vbWhy').innerHTML +=
      `<br><br><b>这个前提是量过的</b>（${align.method}，${align.n_trials} 组）：`
      + `跨版本 <b>r = ${align.cross_version_r.toFixed(2)}</b>（最低 10% 也有 ${align.cross_version_p10.toFixed(2)}），`
      + `而不同 seed 时 r = ${align.different_seed_r.toFixed(2)}、`
      + `不同 prompt 的基线 r = ${align.different_prompt_r.toFixed(2)}。`
      + `也就是说构图确实对齐，肉眼感到的"像换了场景"来自配色与云层形状的改变，而不是构图跑掉了。`;
  }

  const conclusion = data.conclusion;
  $('#vbConclusion').innerHTML =
    `<ol>${conclusion.points.map((point) => `<li>${emphasize(point)}</li>`).join('')}</ol>`
    + `<p class="vb-ask">${emphasize(conclusion.ask)}</p>`;
}

function metricCells(version) {
  return state.data.columns.map((column) => {
    if (column.key === 'text_encoder') {
      const label = version.text_encoder === 'fine-tuned' ? '微调过' : '基座';
      return `<span>TE <b>${label}</b></span>`;
    }
    const value = version[column.key];
    if (value === null || value === undefined) return `<span>${column.label} <b>—</b></span>`;
    const digits = column.digits;
    return `<span>${column.label} <b>${Number(value).toFixed(digits)}</b></span>`;
  }).join('');
}

function renderMetrics() {
  const data = state.data;
  const head = `<thead><tr><th>版本</th>${
    data.columns.map((column) => `<th title="${plain(column.means)}｜陷阱：${plain(column.caveat)}">${column.label}</th>`).join('')
  }</tr></thead>`;
  // 每列的高低方向（1 = 越小越好，-1 = 越大越好，0 = 不可比）
  const direction = { val_mse: 1, kid: 1, clip_fid: 1, clip_score: -1, sharpness: -1, text_encoder: 0 };
  const body = V().map((version) => {
    const cells = data.columns.map((column) => {
      if (column.key === 'text_encoder') {
        const tuned = version.text_encoder === 'fine-tuned';
        return `<td class="${tuned ? 'best' : ''}">${tuned ? '微调过' : '基座'}</td>`;
      }
      const value = version[column.key];
      if (value === null || value === undefined) {
        return `<td class="vb-incomparable" title="${plain(version.val_note || '')}">不可比</td>`;
      }
      const others = V().map((item) => item[column.key]).filter((v) => typeof v === 'number');
      const best = direction[column.key] > 0 ? Math.min(...others) : Math.max(...others);
      const cls = value === best ? 'best' : '';
      return `<td class="${cls}">${Number(value).toFixed(column.digits)}</td>`;
    }).join('');
    const tag = version.id === 'V4' ? '<span class="vb-tag">基线</span>' : '';
    return `<tr><td><span class="vb-rowbase">${version.label}</span> ${version.slogan}${tag}</td>${cells}</tr>`;
  }).join('');
  $('#vbMetrics').innerHTML = head + `<tbody>${body}</tbody>`;

  const valHead = '<thead><tr><th>权重</th><th>固定协议 val ↓</th><th>样本 × 时间步</th></tr></thead>';
  const valBody = data.val_table.map((row) => {
    const mark = row.in_gallery ? '' : ' <span class="vb-tag">未展示</span>';
    return `<tr><td>${row.id}${mark}</td><td>${row.val_mse.toFixed(6)}</td><td>${row.samples} × 5</td></tr>`;
  }).join('');
  $('#vbValTable').innerHTML = valHead + `<tbody>${valBody}</tbody>`;

  const judge = data.judge;
  $('#vbJudgeInvalid').innerHTML =
    `<b>盲测裁判判据无效</b>（机器 ${judge.model}）：${emphasize(judge.invalid_reason)}`
    + `<br>保留原始数字只为留痕：${judge.pairs.map((pair) =>
      `${pair.pair} = ${pair.wins_left}/${pair.wins_right}（CI ${pair.ci[0]}–${pair.ci[1]}）`).join('、')}。`
    + `<br>所以下面这页不是"复核机器结论"，而是<b>补上一条机器已经用不动的判据</b>。`;
}

function renderSr() {
  const sr = state.data.sr;
  $('#vbSrBadge').textContent = '细节量 6.5–8.7× vs bicubic';
  $('#vbSrNote').innerHTML =
    `四个版本在"画得好不好"上没有可分辨差异，但把图放大 4 倍时差异是真实的：`
    + `自训 GAN 把细节量拉到插值的 <b>6.5×（EMA）/ 8.7×（末轮）</b>，与 RealESRGAN（7.09×）、`
    + `UltraSharp（8.03×）同档，代价是 PSNR 低 1.3–1.8 dB（感知-失真权衡）。`
    + `它已经是工作台三个入口的默认超分模型。`;
  const head = '<thead><tr><th>方法</th><th>细节量相对 bicubic ↑</th><th>PSNR 变化（dB）</th></tr></thead>';
  const rows = [`<tr><td>bicubic（插值下限）</td><td>1.00×</td><td>基准</td></tr>`]
    .concat(sr.metrics.rows.map((row) =>
      `<tr><td>${row.label}</td><td class="best">${row.detail_ratio.toFixed(2)}×</td>`
      + `<td>${row.psnr_delta_db.toFixed(2)}</td></tr>`)).join('');
  $('#vbSrTable').innerHTML = head + `<tbody>${rows}</tbody>`;

  $('#vbSrGrid').innerHTML = sr.columns.map((column) => {
    const tiles = Array.from({ length: sr.rows }, (_, row) =>
      `<img loading="lazy" decoding="async" alt="${column.label} 样本 ${row + 1}" `
      + `src="/img/versions/sr/${column.key}_r${String(row).padStart(2, '0')}.webp">`).join('');
    return `<div class="vb-sr-col ${column.key === 'hr' ? 'is-hr' : ''}"><h4>${column.label}</h4>${tiles}</div>`;
  }).join('');
}

function renderControls() {
  const data = state.data;
  $('#vbPromptNo').textContent = `第 ${state.prompt + 1} / ${data.prompts.length} 题`;
  $('#vbPromptText').textContent = data.prompts[state.prompt];
  $('#vbSeed').innerHTML = data.protocol.seeds.map((value, index) =>
    `<option value="${index}"${index === state.seedIndex ? ' selected' : ''}>${value}</option>`).join('');
  const options = V().map((version) => `<option value="${version.id}">${version.label}</option>`).join('');
  $('#vbWipeA').innerHTML = options;
  $('#vbWipeB').innerHTML = options;
  $('#vbWipeA').value = state.wipeA;
  $('#vbWipeB').value = state.wipeB;
  $('#vbMode').value = state.mode;
  $('#vbWipeFields').hidden = state.mode !== 'wipe';
  $('#vbBlind').textContent = state.blind ? '👁 揭晓标签' : '🎲 开始盲测';
  $('#vbBlind').classList.toggle('is-done', state.blind);

  const judged = Object.keys(state.picks).length;
  const total = data.prompts.length * data.protocol.seeds.length;
  $('#vbProgressLabel').textContent = `已评判 ${judged} / ${total}`;
  const bar = $('#vbProgressBar');
  bar.style.width = `${total ? Math.round((judged / total) * 100) : 0}%`;

  const flags = [];
  if (state.blind) flags.push('<span class="vb-flag on">盲测中：标签已隐藏、顺序已打乱</span>');
  else flags.push('<span class="vb-flag">未盲测：你知道哪张是哪个版本</span>');
  flags.push(`<span class="vb-flag">第 ${state.prompt + 1} 题 · seed ${seed()}</span>`);
  if (state.picks[cellKey(state.prompt, seed())]) flags.push('<span class="vb-flag on">这题已评判</span>');
  $('#vbFlagBar').innerHTML = flags.join('');
}

/** 四联格：盲测时按确定性洗牌顺序摆放，`order[slot]` 指向真实版本下标。 */
function renderGrid() {
  const prompt = state.prompt;
  const seedValue = seed();
  const order = blindOrder(V().length, prompt, seedValue);
  const slots = state.blind ? order : Array.from({ length: V().length }, (_, index) => index);
  const record = state.picks[cellKey(prompt, seedValue)];

  $('#vbStage').innerHTML = `<div class="vb-grid">${slots.map((versionIndex, slot) => {
    const version = V()[versionIndex];
    const picked = record && !record.tie && record.winner === version.id;
    return `<figure class="vb-cell ${state.blind ? 'masked' : ''} ${picked ? 'picked' : ''}"
        data-version="${version.id}" data-slot="${SLOTS[slot]}">
      <img src="${img(version.id, prompt, seedValue)}" width="512" height="512"
           loading="lazy" decoding="async" alt="${state.blind ? `候选 ${SLOTS[slot]}` : version.label}">
      <figcaption>
        <span class="vb-slot">${SLOTS[slot]}${state.blind ? ' · 揭晓后显示版本' : ''}</span>
        <span class="vb-name">${version.label}</span>
        <span class="vb-slogan">${version.slogan} · ${version.recipe}</span>
        <span class="vb-metric">${metricCells(version)}</span>
        <span class="vb-actions">
          <button type="button" class="btn ghost" data-pick="${version.id}">选它</button>
          <button type="button" class="btn ghost" data-set-a="${version.id}">设为 A</button>
          <button type="button" class="btn ghost" data-set-b="${version.id}">设为 B</button>
        </span>
      </figcaption>
    </figure>`;
  }).join('')}</div>
  <div class="vb-wipe-answer">
    <button type="button" class="btn ghost" data-tie="1">四张差不多（平局）</button>
    <button type="button" class="btn ghost" data-skip="1">跳过这题</button>
  </div>`;

  $('#vbStageFoot').textContent = state.blind
    ? '盲测中：选完会自动跳到下一题；随时点「揭晓标签」查看答案。键盘 1–4 选择，0 = 平局，B 切换盲测。'
    : '并排模式下四张图同构图同噪声，可直接比构图与纹理；想逐像素比对请切到「擦除对照」。';
}

function renderWipe() {
  const prompt = state.prompt;
  const seedValue = seed();
  const a = versionById(state.wipeA) || V()[0];
  const b = versionById(state.wipeB) || V()[1];
  $('#vbStage').innerHTML = `
    <div class="vb-wipe">
      <div class="vb-wipe-frame" id="vbWipeFrame">
        <img class="vb-wipe-b" src="${img(b.id, prompt, seedValue)}" alt="${b.label}">
        <img class="vb-wipe-top" id="vbWipeTop" src="${img(a.id, prompt, seedValue)}" alt="${a.label}">
        <span class="vb-wipe-tag a">A · ${state.blind ? '隐藏' : a.label}</span>
        <span class="vb-wipe-tag b">B · ${state.blind ? '隐藏' : b.label}</span>
        <div class="vb-compare-handle" id="vbWipeHandle" style="left:50%"></div>
      </div>
      <input type="range" min="0" max="100" value="50" id="vbWipeRange" aria-label="擦除位置">
      <div class="vb-wipe-answer">
        <button type="button" class="btn ghost" data-pair="A">A 更好</button>
        <button type="button" class="btn ghost" data-pair="TIE">差不多</button>
        <button type="button" class="btn ghost" data-pair="B">B 更好</button>
        <button type="button" class="btn ghost" data-swap="1">⇄ 交换 A/B</button>
      </div>
    </div>`;
  setWipe(50);
}

function setWipe(percent) {
  const top = $('#vbWipeTop');
  const handle = $('#vbWipeHandle');
  const range = $('#vbWipeRange');
  if (!top || !handle) return;
  const value = Math.max(0, Math.min(100, Number(percent)));
  top.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
  handle.style.left = `${value}%`;
  if (range && Number(range.value) !== value) range.value = String(value);
}

/* ------------------------------------------------------------------ 交互 */

function recordPick(winner, tie = false, mode = state.blind ? 'blind4' : 'open4') {
  const prompt = state.prompt;
  const seedValue = seed();
  const key = cellKey(prompt, seedValue);
  state.picks[key] = {
    prompt, seed: seedValue, mode, tie, winner: tie ? null : winner,
    order: blindOrder(V().length, prompt, seedValue).map((index) => V()[index].id),
    at: new Date().toISOString(),
  };
  savePicks();
  renderControls();
  renderTally();
  if (state.mode === 'grid') renderGrid();
  // 盲测流程：选完自动进入下一题，让节奏保持"看一眼就决定"
  if (state.blind) window.setTimeout(nextUnjudged, 260);
}

function recordPair(choice) {
  const prompt = state.prompt;
  const seedValue = seed();
  const key = cellKey(prompt, seedValue);
  state.picks[key] = {
    prompt, seed: seedValue, mode: 'pair', tie: choice === 'TIE',
    winner: choice === 'A' ? state.wipeA : choice === 'B' ? state.wipeB : null,
    pair: [state.wipeA, state.wipeB],
    order: [state.wipeA, state.wipeB],
    at: new Date().toISOString(),
  };
  savePicks();
  renderControls();
  renderTally();
  if (state.blind) window.setTimeout(nextUnjudged, 260);
}

function nextUnjudged() {
  const total = state.data.prompts.length;
  for (let step = 1; step <= total * state.data.protocol.seeds.length; step += 1) {
    const flat = (state.prompt * state.data.protocol.seeds.length + state.seedIndex + step)
      % (total * state.data.protocol.seeds.length);
    const prompt = Math.floor(flat / state.data.protocol.seeds.length);
    const seedIndex = flat % state.data.protocol.seeds.length;
    if (!state.picks[cellKey(prompt, state.data.protocol.seeds[seedIndex])]) {
      state.prompt = prompt;
      state.seedIndex = seedIndex;
      renderAll();
      return;
    }
  }
}

function step(delta) {
  const total = state.data.prompts.length;
  state.prompt = (state.prompt + delta + total) % total;
  renderAll();
}

function renderTally() {
  const records = Object.values(state.picks);
  $('#vbTallyCount').textContent = `${records.length} 题`;
  const note = $('#vbSubmitNote');
  if (note && state.submit.state === 'idle') {
    note.className = 'vb-submit-note';
    note.innerHTML = records.length
      ? `已记录 ${records.length} 题 · 点「提交评判」把结果送到 <code>research/human_judge/</code>（本机收集器）`
      : '先判几题，提交按钮才会亮。';
  }
  const submit = $('#vbSubmit');
  if (submit) submit.disabled = records.length === 0 || state.submit.state === 'sending';
  const groups = [
    { mode: 'blind4', label: '4 选 1 盲测', chance: 0.25, modes: ['blind4'] },
    { mode: 'open4', label: '4 选 1（未盲测）', chance: 0.25, modes: ['open4'] },
    { mode: 'pair', label: '两两擦除对照', chance: 0.5, modes: ['pair'] },
  ];
  const head = '<thead><tr><th>口径</th><th>版本</th><th>被选中</th><th>占比</th>'
    + '<th>Wilson 95% 区间</th><th>判定</th></tr></thead>';
  const body = [];
  groups.forEach((group) => {
    const rows = records.filter((row) => group.modes.includes(row.mode));
    const decided = rows.filter((row) => !row.tie).length;
    if (!decided) return;
    V().forEach((version) => {
      const wins = rows.filter((row) => !row.tie && row.winner === version.id).length;
      if (!wins && !rows.some((row) => row.order?.includes(version.id))) return;
      const { low, high, indistinguishable } = verdictFor(wins, decided, group.chance);
      body.push(`<tr><td>${group.label}</td><td>${version.label}</td><td>${wins} / ${decided}</td>`
        + `<td>${((wins / decided) * 100).toFixed(1)}%</td>`
        + `<td>${(low * 100).toFixed(1)}–${(high * 100).toFixed(1)}%</td>`
        + `<td>${indistinguishable ? '<span class="vb-badge warn">无法区分</span>'
          : `<span class="vb-badge ok">高于随机 ${(group.chance * 100).toFixed(0)}%</span>`}</td></tr>`);
    });
  });
  $('#vbTally').innerHTML = head + `<tbody>${body.join('') || '<tr><td colspan="6" class="muted">还没有评判记录</td></tr>'}</tbody>`;

  // 两两矩阵：只用"两两擦除对照"的记录，直接给对战胜率
  const pairs = records.filter((row) => row.mode === 'pair' && !row.tie && row.pair);
  const matrix = {};
  pairs.forEach((row) => {
    const loser = row.winner === row.pair[0] ? row.pair[1] : row.pair[0];
    matrix[row.winner] = matrix[row.winner] || {};
    matrix[row.winner][loser] = (matrix[row.winner][loser] || 0) + 1;
  });
  const rows = [];
  V().forEach((a) => V().forEach((b) => {
    if (a.id >= b.id) return;
    const aw = (matrix[a.id] || {})[b.id] || 0;
    const bw = (matrix[b.id] || {})[a.id] || 0;
    const decided = aw + bw;
    if (!decided) return;
    const { low, high, indistinguishable } = verdictFor(aw, decided, 0.5);
    rows.push(`<div class="vb-pair-row"><b>${a.label}</b> vs <b>${b.label}</b>`
      + `<span>${aw} : ${bw}</span>`
      + `<span class="vb-ci">CI ${(low * 100).toFixed(0)}–${(high * 100).toFixed(0)}%</span>`
      + `<span class="vb-verdict">${indistinguishable
        ? '<span class="vb-badge warn">无法区分</span>'
        : `<span class="vb-badge ok">${a.label} 占优</span>`}</span></div>`);
  }));
  $('#vbPairs').innerHTML = rows.length
    ? `<h3>两两对战（仅统计擦除对照的选择）</h3>${rows.join('')}`
    : '';
}

function renderAll() {
  renderControls();
  if (state.mode === 'wipe') renderWipe(); else renderGrid();
  renderTally();
}

/* ------------------------------------------------------------------ 导出 */

function exportRecords() {
  return Object.values(state.picks)
    .sort((a, b) => (a.prompt - b.prompt) || (a.seed - b.seed))
    .map((row) => ({
      prompt_index: row.prompt,
      prompt: state.data.prompts[row.prompt],
      seed: row.seed,
      mode: row.mode,
      blind: row.mode === 'blind4',
      order: row.order,
      winner: row.winner,
      tie: !!row.tie,
      at: row.at,
    }));
}

/** 本机统计（与服务端 tools/judge_collector.py 各自独立计算，提交后可互相核对）。 */
function localSummary() {
  const records = Object.values(state.picks);
  const modes = {};
  ['blind4', 'open4', 'pair'].forEach((mode) => {
    const rows = records.filter((row) => row.mode === mode);
    const decided = rows.filter((row) => !row.tie).length;
    if (!decided) return;
    const chance = mode === 'pair' ? 0.5 : 0.25;
    const versions = V().map((version) => {
      const wins = rows.filter((row) => !row.tie && row.winner === version.id).length;
      const { low, high, indistinguishable } = verdictFor(wins, decided, chance);
      return { version: version.id, wins, decided, rate: wins / decided,
               ci: [low, high], chance, indistinguishable };
    }).filter((row) => row.wins > 0);
    modes[mode] = { decided, ties: rows.length - decided, versions };
  });
  return {
    schema: 1,
    decided_total: Object.values(modes).reduce((sum, mode) => sum + mode.decided, 0),
    modes,
    headline: tallyHeadline(modes),
  };
}

/** 一句话结论：只有显著高于随机才敢说"更好" —— 与收集器里的措辞保持一致口径。 */
function tallyHeadline(modes) {
  const rows = Object.values(modes).flatMap((mode) => mode.versions);
  const decided = Object.values(modes).reduce((sum, mode) => sum + mode.decided, 0);
  if (!decided) return '还没有已决题（全为平局或未评判）。';
  const best = rows.filter((row) => !row.indistinguishable)
    .sort((a, b) => b.rate - a.rate)[0];
  if (!best) {
    return `已决 ${decided} 题：没有任何版本显著高于随机期望 ⇒ 按本项目判据记为「无法区分」。`;
  }
  return `已决 ${decided} 题：${best.version} 被选中 ${best.wins}/${best.decided}`
    + `（区间 ${(best.ci[0] * 100).toFixed(1)}–${(best.ci[1] * 100).toFixed(1)}%，`
    + `高于随机期望 ${(best.chance * 100).toFixed(0)}%）。`;
}

function buildPayload() {
  return {
    schema: 1,
    source: 'site/versions.html（人工盲测）',
    versions: V().map((version) => version.id),
    protocol: state.data.protocol,
    client: {
      url: location.origin + location.pathname,
      submitted_at: new Date().toISOString(),
      blind_default: true,
      // 刻意不发 User-Agent：对分析没用，而且属于可识别信息（AGENTS.md 脱敏要求）
    },
    summary: localSummary(),
    records: exportRecords(),
  };
}

/* ------------------------------------------------------------------ 提交 */

function setSubmit(status, message = '') {
  state.submit = { state: status, message };
  const button = $('#vbSubmit');
  const note = $('#vbSubmitNote');
  if (button) {
    button.disabled = status === 'sending' || Object.keys(state.picks).length === 0;
    button.textContent = status === 'sending' ? '⏳ 正在提交…' : '📤 提交评判';
  }
  if (note) {
    note.className = `vb-submit-note ${status}`;
    note.innerHTML = message;
  }
}

/** 一次性授权：把结果直接写进你选的仓库目录（之后每次提交都不再弹窗）。 */
let dirHandle = null;
async function saveToChosenFolder(payload) {
  if (typeof window.showSaveFilePicker !== 'function') throw new Error('浏览器不支持文件系统访问');
  const name = `verdicts_${new Date().toISOString().replace(/[-:T]/g, '').slice(0, 15)}.json`;
  const handle = await window.showSaveFilePicker({
    suggestedName: name,
    types: [{ description: '评判结果 JSON', accept: { 'application/json': ['.json'] } }],
  });
  const writable = await handle.createWritable();
  await writable.write(JSON.stringify(payload, null, 1));
  await writable.close();
  return handle.name;
}

async function submit() {
  const payload = buildPayload();
  if (!payload.records.length) return;
  setSubmit('sending', '正在提交到本机收集器…');
  // 1) 首选：本机收集器（tools/judge_collector.py）—— 落盘到 research/human_judge/
  try {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 6000);
    const response = await fetch(`${COLLECTOR}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    window.clearTimeout(timer);
    const body = await response.json().catch(() => ({}));
    if (!response.ok || !body.ok) throw new Error(body.error || `HTTP ${response.status}`);
    setSubmit('done', `✅ 已提交：<code>${body.file}</code>（${body.records} 条）<br>`
      + `<span class="muted">${escapeHtml(body.summary?.headline || '')}</span>`);
    return;
  } catch (error) {
    // 2) 次选：直接写进你选的目录（File System Access）
    try {
      const file = await saveToChosenFolder(payload);
      setSubmit('done', `✅ 已写入 <code>${escapeHtml(file)}</code>（${payload.records.length} 条）<br>`
        + '<span class="muted">把它放进仓库的 research/human_judge/ 即可</span>');
      return;
    } catch (fallbackError) {
      // 3) 兜底：下载 + 说清楚怎么办
      download('my-verdicts.json', JSON.stringify(payload, null, 1));
      const why = error && error.name === 'AbortError' ? '收集器无响应' : (error?.message || error);
      setSubmit('manual', `⚠ 没能自动送达（${escapeHtml(String(why))}）—— 已下载 `
        + `<code>my-verdicts.json</code>。<br>`
        + `要做成"一键送达"：先在本机跑 <code>python tools/judge_collector.py</code>`
        + `（或 <code>pwsh -NoProfile -File tools/dev.ps1 judge</code>）再用本地页面提交；`
        + `也可以把这个文件放进 <code>research/human_judge/</code> 告诉我一声。`);
      void fallbackError;
    }
  }
}

function download(name, text, type = 'application/json') {
  const link = document.createElement('a');
  link.download = name;
  if (typeof URL.createObjectURL === 'function') {
    const url = URL.createObjectURL(new Blob([text], { type }));
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
    return;
  }
  // 退化路径（无 Blob URL 的环境，例如 jsdom）：用 data: URI，同样是纯本地导出
  link.href = `data:${type};charset=utf-8,${encodeURIComponent(text)}`;
  link.click();
}

function onExport() {
  const records = exportRecords();
  if (!records.length) return void 0;
  const summary = {
    schema: 1,
    source: 'site/versions.html（人工盲测）',
    protocol: state.data.protocol,
    versions: V().map((version) => version.id),
    records,
  };
  const lines = ['prompt_index,seed,mode,tie,winner,prompt'];
  records.forEach((row) => lines.push(
    [row.prompt_index, row.seed, row.mode, row.tie, row.winner || '', `"${row.prompt.replace(/"/g, '""')}"`].join(',')));
  download('my-verdicts.json', JSON.stringify(summary, null, 1));
  window.setTimeout(() => download('my-verdicts.csv', lines.join('\n') + '\n', 'text/csv'), 300);
}

/* ------------------------------------------------------------------ 启动 */

/** 载入失败时必须"说清楚"，不能留一个转圈的「载入中…」让人以为后端挂了。
 *  这一页是纯静态的：不需要后端、不需要隧道；失败只会来自网络/缓存/代理。 */
function renderLoadFailure(error) {
  const badge = $('#vbCorpus');
  if (badge) { badge.textContent = '载入失败'; badge.classList.add('bad'); }
  const stage = $('#vbStage');
  if (stage) {
    stage.innerHTML = `
      <div class="vb-fail">
        <h3>没能读到 <code>data/versions.json</code></h3>
        <p class="vb-fail-err">${escapeHtml(String(error && error.message ? error.message : error))}</p>
        <p><b>这一页是纯静态的，不需要后端、也不需要隧道</b>；打不开只可能是网络、代理或浏览器缓存。
           按顺序试：</p>
        <ol>
          <li><button type="button" class="btn ghost tiny" id="vbRetry">重试</button>
              点一下重新拉取（不做整页刷新）</li>
          <li>硬刷新：<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>R</kbd>（Mac：<kbd>⌘</kbd>+<kbd>⇧</kbd>+<kbd>R</kbd>），
              绕过 Service Worker 与缓存</li>
          <li>看<b>零 JS 静态画廊</b>：<a href="gallery.html">/gallery</a>
              （无脚本、自包含，任何浏览器都能开）</li>
          <li>本机仓库里直接双击 <code>site/gallery.html</code>：连网络都不需要</li>
        </ol>
        <p class="muted">如果你的浏览器走了 127.0.0.1:7890 代理而该代理已失效，
          本站会整体打不开（与本项目后端无关）。</p>
      </div>`;
    const retry = $('#vbRetry');
    if (retry) retry.onclick = () => window.location.reload();
  }
  const progress = $('#vbProgressLabel');
  if (progress) progress.textContent = '未载入';
}

async function boot() {
  const theme = store.getState().theme;
  document.documentElement.dataset.theme = theme;
  $('#vbTheme').textContent = theme === 'light' ? '🌙 暗色' : '☀️ 亮色';

  let data = null;
  try {
    const response = await fetch('/data/versions.json', { cache: 'no-cache' });
    data = await response.json();
  } catch (error) {
    renderLoadFailure(error);
    return;
  }
  state.data = data;

  $('#vbBlind').onclick = () => {
    state.blind = !state.blind;
    savePrefs();
    renderAll();
  };
  $('#vbPrev').onclick = () => step(-1);
  $('#vbNext').onclick = () => step(1);
  $('#vbSeed').onchange = (event) => { state.seedIndex = Number(event.target.value); renderAll(); };
  $('#vbMode').onchange = (event) => { state.mode = event.target.value; savePrefs(); renderAll(); };
  $('#vbWipeA').onchange = (event) => { state.wipeA = event.target.value; renderWipe(); };
  $('#vbWipeB').onchange = (event) => { state.wipeB = event.target.value; renderWipe(); };
  $('#vbUnjudged').onclick = nextUnjudged;
  $('#vbResetPicks').onclick = () => {
    if (!window.confirm('清空本机记录的评判？此操作不可撤销。')) return;
    state.picks = {};
    savePicks();
    renderAll();
  };
  $('#vbExport').onclick = onExport;
  $('#vbSubmit').onclick = submit;
  const submitCard = $('#vbSubmitCard');
  if (submitCard) submitCard.onclick = submit;
  $('#vbTheme').onclick = () => {
    const next = store.setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light');
    motion.withThemeWipe(() => { document.documentElement.dataset.theme = next; }, null);
    $('#vbTheme').textContent = next === 'light' ? '🌙 暗色' : '☀️ 亮色';
  };

  // 事件委托：格子按钮 / 平局 / 跳过 / 擦除对照 / 放大
  document.addEventListener('click', (event) => {
    const target = event.target.closest('[data-pick],[data-tie],[data-skip],[data-pair],[data-swap],[data-set-a],[data-set-b]');
    if (target) {
      if (target.dataset.pick) recordPick(target.dataset.pick);
      else if (target.dataset.tie) recordPick(null, true);
      else if (target.dataset.skip) step(1);
      else if (target.dataset.pair) recordPair(target.dataset.pair);
      else if (target.dataset.swap) {
        [state.wipeA, state.wipeB] = [state.wipeB, state.wipeA];
        renderControls();     // 两个下拉框必须跟着换，否则显示的 A/B 与实际叠图不符
        renderWipe();
      } else if (target.dataset.setA) { state.wipeA = target.dataset.setA; state.mode = 'wipe'; renderAll(); }
      else if (target.dataset.setB) { state.wipeB = target.dataset.setB; state.mode = 'wipe'; renderAll(); }
      return;
    }
    const image = event.target.closest('.vb-cell img');
    if (image) openZoom(image);
  });

  document.addEventListener('keydown', (event) => {
    if (event.target.tagName === 'SELECT' || event.target.tagName === 'INPUT') return;
    const key = event.key;
    if (key === 'ArrowLeft') { step(-1); event.preventDefault(); }
    else if (key === 'ArrowRight') { step(1); event.preventDefault(); }
    else if (key === 'b' || key === 'B') { state.blind = !state.blind; renderAll(); }
    else if (key === '0' || key === 't' || key === 'T') recordPick(null, true);
    else if (/^[1-4]$/.test(key) && state.mode === 'grid') {
      const cell = document.querySelectorAll('.vb-cell')[Number(key) - 1];
      if (cell) recordPick(cell.dataset.version);
    } else if (key === 'Escape') closeZoom();
  });

  // 擦除对照的拖动
  const frame = () => $('#vbWipeFrame');
  document.addEventListener('pointerdown', (event) => {
    const element = frame();
    if (!element || !element.contains(event.target)) return;
    element.setPointerCapture?.(event.pointerId);
    const move = (pointerEvent) => {
      const rect = element.getBoundingClientRect();
      setWipe(((pointerEvent.clientX - rect.left) / rect.width) * 100);
    };
    move(event);
    element.onpointermove = move;
    element.onpointerup = () => { element.onpointermove = null; element.onpointerup = null; };
  });
  document.addEventListener('input', (event) => {
    if (event.target.id === 'vbWipeRange') setWipe(event.target.value);
  });

  $('#vbZoom').onclick = closeZoom;
  loadPrefs();
  loadPicks();
  renderIntro();
  renderMetrics();
  renderSr();
  renderAll();
  window.vbState = state;                       // 供无头冒烟测试断言（jsdom 专用）
  window.vbBuildPayload = buildPayload;         // 冒烟测试用：拿到与「提交评判」逐字段相同的 payload
}

function openZoom(image) {
  const zoom = $('#vbZoom');
  $('#vbZoomImg').src = image.src;
  const label = image.closest('.vb-cell')?.dataset.version || '';
  $('#vbZoomLabel').textContent = state.blind ? '盲测中：版本已隐藏' : label;
  zoom.hidden = false;
}

function closeZoom() { $('#vbZoom').hidden = true; }

boot();
motion.initMotion();
