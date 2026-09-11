/**
 * Optional workbench extras: run statistics, a reusable prompt library, and a
 * searchable/sortable history with favourites.
 *
 * Kept in its own module so main.js stays about the generation flow.  All state
 * lives in store.js (localStorage), so nothing here is lost on reload.
 */
import * as store from './store.js';
import { el, toast } from './ui.js';
import * as motion from './motion.js';

let handlers = { getPrompt: () => '', onUsePrompt: () => {}, onShowJob: () => {}, onReuseJob: () => {} };
let filterText = '';
let sortMode = 'new';

function fmtSeconds(value) {
  const s = Number(value) || 0;
  return s >= 10 ? `${s.toFixed(1)}s` : `${s.toFixed(2)}s`;
}

/* ------------------------------------------------------------------- stats */

export function renderStats() {
  const host = el('statGrid');
  if (!host) return;
  const s = store.stats();
  const top = Object.entries(s.bySampler).sort((a, b) => b[1] - a[1])[0];
  const cards = [
    ['生成图片', s.images, `${s.jobs} 个任务`],
    ['成功率', Math.round(s.success * 100), '% · 失败 ' + s.failed],
    ['平均耗时', s.avg ? fmtSeconds(s.avg) : '—', '含高清/批量'],
    ['命中缓存', s.cached, '⚡ 秒回次数'],
    ['常采样器', top ? top[0].replace('dpmpp2m', 'DPM++ 2M').replace('_', ' ') : '—', top ? `${top[1]} 次` : '还没有记录'],
    ['收藏 / 词库', `${s.favorites} / ${s.prompts}`, '本地保存'],
  ];
  host.innerHTML = cards.map(([label, value, hint]) => {
    const numeric = typeof value === 'number' ? value : parseFloat(String(value).replace(/[^\d.]/g, '')) || 0;
    const countable = typeof value === 'number' || /^\d/.test(String(value));
    const suffix = typeof value === 'number' ? '' : String(value).replace(/^[\d.]+/, '');
    return `
    <div class="stat-card">
      <b${countable ? ` data-count="${numeric}"` : ''}>${countable ? numeric : value}</b>${countable ? `<b>${suffix}</b>` : ''}
      <small>${label}</small>
      <div class="stat-bar"><i style="width:${Math.min(100, s.success * 100)}%"></i></div>
      <small>${hint}</small>
    </div>`;
  }).join('');
  motion.animateNumbers(host);
}

/* ---------------------------------------------------------- prompt library */

export function renderLibrary() {
  const host = el('libList');
  if (!host) return;
  const items = store.getState().prompts;
  if (!items.length) {
    host.innerHTML = '<p class="lib-empty">还没有收藏 prompt。写好一句后点「存入词库」，下次一键取回。</p>';
    return;
  }
  host.innerHTML = '';
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'lib-item';
    const text = document.createElement('span');
    text.textContent = item.text;
    text.title = item.text;
    const use = document.createElement('button');
    use.type = 'button';
    use.className = 'btn ghost tiny';
    use.textContent = '套用';
    use.onclick = () => { handlers.onUsePrompt(item.text); toast('已套用词库 prompt'); };
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'btn ghost tiny';
    del.textContent = '删除';
    del.onclick = () => { store.removePrompt(item.ts); renderLibrary(); renderStats(); };
    row.append(text, use, del);
    host.appendChild(row);
  });
}

/* ---------------------------------------------------------------- history */

function historyRows() {
  let jobs = store.getState().jobs.slice();
  if (sortMode === 'old') jobs.reverse();
  if (sortMode === 'seed') jobs.sort((a, b) => (a.seed || 0) - (b.seed || 0));
  if (sortMode === 'fav') jobs = jobs.filter((j) => store.isFavorite(j.id));
  const needle = filterText.trim().toLowerCase();
  if (needle) {
    jobs = jobs.filter((j) => `${j.prompt || ''} ${j.seed ?? ''} ${j.sampler || ''}`.toLowerCase().includes(needle));
  }
  return jobs.slice(0, 30);
}

export function renderHistoryList() {
  const host = el('histList');
  if (!host) return;
  const rows = historyRows();
  if (!rows.length) {
    host.innerHTML = '<p class="lib-empty">没有匹配的记录。换个关键词，或把筛选切回「全部」。</p>';
    return;
  }
  host.innerHTML = '';
  rows.forEach((job) => {
    const row = document.createElement('div');
    row.className = 'lib-item';
    const star = document.createElement('button');
    star.type = 'button';
    star.className = 'star' + (store.isFavorite(job.id) ? ' on' : '');
    star.textContent = store.isFavorite(job.id) ? '★' : '☆';
    star.title = '收藏 / 取消收藏';
    star.onclick = () => { store.toggleFavorite(job.id); renderHistoryList(); renderStats(); };
    const text = document.createElement('span');
    text.textContent = `${job.status === 'done' ? '✅' : job.status === 'failed' ? '❌' : '⏳'} `
      + `${(job.prompt || '(随机)').slice(0, 46)} · seed ${job.seed ?? '?'}`
      + `${(job.count || 1) > 1 ? ` ×${job.count}` : ''}`;
    text.title = `${job.prompt || ''} · ${job.info || ''}`;
    const show = document.createElement('button');
    show.type = 'button';
    show.className = 'btn ghost tiny';
    show.textContent = '查看';
    show.disabled = job.status !== 'done';
    show.onclick = () => handlers.onShowJob(job.id);
    const reuse = document.createElement('button');
    reuse.type = 'button';
    reuse.className = 'btn ghost tiny';
    reuse.textContent = '复用';
    reuse.onclick = () => handlers.onReuseJob(job);
    row.append(star, text, show, reuse);
    host.appendChild(row);
  });
}

/* ------------------------------------------------------------------ wiring */

export function refreshExtras() {
  renderStats();
  renderLibrary();
  renderHistoryList();
}

export function initExtras(next) {
  handlers = { ...handlers, ...(next || {}) };

  el('libSave')?.addEventListener('click', () => {
    const text = handlers.getPrompt();
    if (!text || !text.trim()) { toast('先写一句 prompt 再存入词库'); return; }
    store.addPrompt(text);
    renderLibrary();
    renderStats();
    toast('已存入词库');
  });

  el('libRandom')?.addEventListener('click', () => {
    const items = store.getState().prompts;
    if (!items.length) { toast('词库还是空的'); return; }
    handlers.onUsePrompt(items[Math.floor(Math.random() * items.length)].text);
    toast('随机取了一条词库 prompt');
  });

  el('histSearch')?.addEventListener('input', (event) => {
    filterText = event.target.value || '';
    renderHistoryList();
  });
  el('histSort')?.addEventListener('change', (event) => {
    sortMode = event.target.value || 'new';
    renderHistoryList();
  });

  refreshExtras();
}
