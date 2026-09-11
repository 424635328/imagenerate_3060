/**
 * All DOM rendering lives here.  Modules above this one stay framework-free and
 * testable: `main.js` owns behaviour, `ui.js` owns pixels.
 */
import { CLARITY, MODIFIER_GROUPS, SHORTCUTS, STAGE_LABELS } from './config.js';
import { formatBytes } from './upload.js';

export const el = (id) => document.getElementById(id);
const logLines = [];

/* ---------------------------------------------------------------- feedback */

export function toast(message, kind = '') {
  const node = el('toast');
  node.textContent = message;
  node.className = `toast show ${kind}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.className = 'toast'; }, 2400);
}

export function setStatus(text, kind = '') {
  const node = el('status');
  node.textContent = text;
  node.dataset.kind = kind;
}

export function log(label, payload) {
  const body = typeof payload === 'string' ? payload : JSON.stringify(payload);
  logLines.push(`[${new Date().toLocaleTimeString()}] ${label} :: ${body}`);
  if (logLines.length > 200) logLines.splice(0, logLines.length - 200);
  el('debug').textContent = logLines.slice(-60).join('\n');
}

export function logText() {
  return logLines.join('\n');
}

/* ------------------------------------------------------------------ theme */

export function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  el('themeBtn').textContent = theme === 'light' ? '🌙 暗色' : '☀️ 亮色';
  el('themeBtn').setAttribute('aria-label', theme === 'light' ? '切换到暗色主题' : '切换到亮色主题');
}

/* ------------------------------------------------------- prompt & presets */

/* Scene Explorer moved to js/scenes.js (V6 big cards) — same onPick contract. */

export function renderModifiers(selected, onToggle) {
  const root = el('modifiers');
  root.innerHTML = '';
  MODIFIER_GROUPS.forEach(([group, items]) => {
    const wrap = document.createElement('div');
    wrap.className = 'mod-group';
    const title = document.createElement('span');
    title.className = 'mod-title';
    title.textContent = group;
    wrap.appendChild(title);
    const row = document.createElement('div');
    row.className = 'chips';
    items.forEach(([label, fragment]) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = `chip small${selected.has(fragment) ? ' active' : ''}`;
      chip.textContent = label;
      chip.title = fragment;
      chip.setAttribute('aria-pressed', selected.has(fragment) ? 'true' : 'false');
      chip.onclick = () => onToggle(fragment, chip);
      row.appendChild(chip);
    });
    wrap.appendChild(row);
    root.appendChild(wrap);
  });
}

export function renderComposed(text) {
  el('composed').textContent = text || '（留空将随机抽取场景）';
}

export function renderRecipes(builtin, custom, { onUse, onDelete }) {
  const root = el('recipes');
  root.innerHTML = '';
  builtin.filter((r) => r.id !== 'custom').forEach((recipe) => {
    const row = document.createElement('div');
    row.className = 'recipe';
    row.innerHTML = `<div><b>${recipe.label}</b><span>${recipe.detail}</span></div>`;
    const use = document.createElement('button');
    use.type = 'button';
    use.className = 'btn ghost tiny';
    use.textContent = '使用';
    use.onclick = () => onUse(recipe);
    row.appendChild(use);
    root.appendChild(row);
  });
  custom.forEach((recipe) => {
    const row = document.createElement('div');
    row.className = 'recipe custom';
    row.innerHTML = `<div><b>💾 ${recipe.name}</b><span>${recipe.summary}</span></div>`;
    const use = document.createElement('button');
    use.type = 'button';
    use.className = 'btn ghost tiny';
    use.textContent = '载入';
    use.onclick = () => onUse(recipe);
    const drop = document.createElement('button');
    drop.type = 'button';
    drop.className = 'btn ghost tiny';
    drop.textContent = '删除';
    drop.onclick = () => onDelete(recipe);
    row.append(use, drop);
    root.appendChild(row);
  });
}

/* ------------------------------------------------------------------ stage */

let stageHandlers = { prev: null, next: null, zoom: null };

export function bindStage(handlers) {
  stageHandlers = { ...stageHandlers, ...handlers };
  const stage = el('stage');
  let startX = 0;
  stage.addEventListener('touchstart', (e) => { startX = e.touches[0].clientX; }, { passive: true });
  stage.addEventListener('touchend', (e) => {
    const dx = e.changedTouches[0].clientX - startX;
    if (Math.abs(dx) > 45) (dx > 0 ? stageHandlers.prev : stageHandlers.next)?.();
  }, { passive: true });
  el('stagePrev').onclick = () => stageHandlers.prev?.();
  el('stageNext').onclick = () => stageHandlers.next?.();
}

export function showEmptyStage(message = '生成后这里会先出现渐进预览，再替换为 WebP 原图') {
  const canvas = el('stageCanvas');
  canvas.innerHTML = `<div class="empty"><div class="glyph" aria-hidden="true">🏔️</div><p>${message}</p>
    <p class="muted">← / → 切换历史图 · C 对比 · F 全屏</p></div>`;
  el('previewBadge').hidden = true;
}

export function showImage(url, { preview = false, alt = '生成的风景图' } = {}) {
  const canvas = el('stageCanvas');
  canvas.innerHTML = '';
  const img = document.createElement('img');
  img.id = 'mainImage';
  img.src = url;
  img.alt = alt;
  img.decoding = 'async';
  img.onclick = () => stageHandlers.zoom?.();
  canvas.appendChild(img);
  const badge = el('previewBadge');
  badge.hidden = !preview;
  if (preview) badge.textContent = '渐进预览 · 仍在渲染';
}

/**
 * Batch result: one job can produce several images.  They are shown as a grid so
 * the whole set is visible at once; ←/→ still walks them one by one.
 */
export function showGrid(urls, caption = '') {
  const canvas = el('stageCanvas');
  canvas.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = `stage-grid n${Math.min(Math.max(urls.length, 1), 4)}`;
  urls.forEach((url, index) => {
    const img = document.createElement('img');
    img.src = url;
    img.alt = `批量结果 ${index + 1}`;
    img.loading = 'lazy';
    img.decoding = 'async';
    img.onclick = () => stageHandlers.zoom?.(index);
    grid.appendChild(img);
  });
  canvas.appendChild(grid);
  const badge = el('previewBadge');
  badge.hidden = false;
  badge.textContent = caption || `批量 ${urls.length} 张`;
}

/** Split-view comparison between the current image and any other history item.
 *  `tags` optionally renames the corner labels (A/B lab passes 左 A / 右 B). */
export function showCompare(currentUrl, otherUrl, caption, tags = {}) {
  const leftTag = tags.left || '对比';
  const rightTag = tags.right || '当前';
  const canvas = el('stageCanvas');
  canvas.innerHTML = `
    <div class="compare" id="compareBox">
      <img class="compare-base" src="${otherUrl}" alt="对比参考图">
      <div class="compare-top" id="compareTop"><img src="${currentUrl}" alt="当前图"></div>
      <div class="compare-handle" id="compareHandle" role="slider" tabindex="0"
           aria-label="对比分割位置" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50"></div>
      <span class="compare-tag left">${leftTag}</span><span class="compare-tag right">${rightTag}</span>
    </div>`;
  el('previewBadge').hidden = true;
  el('resultMeta').textContent = caption;
  const box = el('compareBox');
  const top = el('compareTop');
  const handle = el('compareHandle');
  const setPercent = (pct) => {
    const value = Math.max(0, Math.min(100, pct));
    top.style.width = `${value}%`;
    handle.style.left = `${value}%`;
    handle.setAttribute('aria-valuenow', String(Math.round(value)));
  };
  const fromEvent = (event) => {
    const rect = box.getBoundingClientRect();
    const x = (event.touches ? event.touches[0].clientX : event.clientX) - rect.left;
    setPercent((x / rect.width) * 100);
  };
  let dragging = false;
  const start = (e) => { dragging = true; fromEvent(e); };
  const move = (e) => { if (dragging) fromEvent(e); };
  const end = () => { dragging = false; };
  box.addEventListener('pointerdown', start);
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', end);
  box.addEventListener('touchstart', start, { passive: true });
  box.addEventListener('touchmove', move, { passive: true });
  handle.onkeydown = (e) => {
    const now = Number(handle.getAttribute('aria-valuenow')) || 50;
    if (e.key === 'ArrowLeft') { e.preventDefault(); setPercent(now - 4); }
    if (e.key === 'ArrowRight') { e.preventDefault(); setPercent(now + 4); }
  };
  setPercent(50);
}

export function setStageNav(enabled, position = '') {
  el('stagePrev').disabled = !enabled;
  el('stageNext').disabled = !enabled;
  el('stagePos').textContent = position;
}

export function setResultMeta(text, transfer = '') {
  el('resultMeta').textContent = text;
  if (transfer) el('transferMeta').textContent = transfer;
}

export function setProgress({ percent = 0, stage = '', eta = 0, visible = true }) {
  const bar = el('progressBar');
  const wrap = el('progress');
  wrap.hidden = !visible;
  bar.style.width = `${Math.max(2, Math.min(100, percent * 100))}%`;
  bar.classList.toggle('indeterminate', percent <= 0);
  const label = STAGE_LABELS[stage] || stage || '';
  const etaText = eta > 0 ? ` · 预计还需 ${eta < 1 ? '<1' : Math.round(eta)} s` : '';
  el('progressLabel').textContent = visible ? `${label}${percent > 0 ? ` ${Math.round(percent * 100)}%` : ''}${etaText}` : '';
}

/* ------------------------------------------------------------------ queue */

export function renderQueue(jobs) {
  const root = el('queue');
  const active = jobs.filter((job) => ['queued', 'running'].includes(job.status));
  el('queueCount').textContent = `${active.length} 个进行中`;
  if (!active.length) {
    root.innerHTML = '<div class="placeholder">队列空闲。提交任务后这里会显示逐步进度与 ETA。</div>';
    return;
  }
  root.innerHTML = '';
  active.forEach((job) => {
    const row = document.createElement('div');
    row.className = 'queue-row';
    const percent = Math.round((job.progress || 0) * 100);
    row.innerHTML = `
      <div class="queue-head"><b>${STAGE_LABELS[job.stage] || job.status}</b>
        <span class="muted">seed ${job.seed} · ${job.steps || '?'} 步 · ${job.sampler || ''}</span></div>
      <div class="bar"><i style="width:${Math.max(3, percent)}%"></i></div>
      <div class="queue-foot muted">${percent}% ${job.eta ? `· ETA ${Math.round(job.eta)} s` : ''} ${job.elapsed ? `· 已用 ${Math.round(job.elapsed)} s` : ''}</div>`;
    root.appendChild(row);
  });
}

/* ---------------------------------------------------------------- gallery */

const BADGES = {
  queued: '🕒 排队中', running: '⏳ 生成中', failed: '❌ 失败',
  expired: '⌛ 后端已清理', done: '',
};

export function renderGallery(jobs, filter, handlers) {
  const root = el('gallery');
  root.innerHTML = '';
  el('historyCount').textContent = `${jobs.length} 张`;
  const list = jobs.filter((job) => (filter === 'all' ? true : filter === 'done' ? job.status === 'done' : job.status !== 'done'));
  if (!list.length) {
    root.innerHTML = '<div class="placeholder">暂无记录。生成后任务状态会保存在浏览器里，刷新不丢。</div>';
    return;
  }
  list.forEach((job) => {
    const card = document.createElement('figure');
    card.className = 'tile';
    if (job.status === 'done') {
      const img = document.createElement('img');
      img.loading = 'lazy';
      img.alt = job.info || '生成的风景图';
      img.src = job.url || handlers.imageUrl(job.id);
      img.onerror = () => handlers.onExpired(job);
      img.onclick = () => handlers.onOpen(job);
      card.appendChild(img);
    } else {
      const box = document.createElement('div');
      box.className = 'placeholder';
      box.textContent = job.status === 'failed' ? `❌ ${job.error || '失败'}` : BADGES[job.status] || job.status;
      card.appendChild(box);
    }
    const tools = document.createElement('div');
    tools.className = 'tile-tools';
    const mkButton = (label, title, fn) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      b.title = title;
      b.setAttribute('aria-label', title);
      b.onclick = (event) => { event.stopPropagation(); fn(job); };
      return b;
    };
    if (job.status === 'done') tools.appendChild(mkButton('⤓', '下载这张', handlers.onDownload));
    tools.appendChild(mkButton('♻', '复用这组参数', handlers.onReuse));
    tools.appendChild(mkButton('🗑', '从历史移除', handlers.onDelete));
    card.appendChild(tools);
    const caption = document.createElement('figcaption');
    caption.textContent = job.info || BADGES[job.status] || job.status;
    card.appendChild(caption);
    root.appendChild(card);
  });
}

export function setGalleryFilter(filter) {
  el('galleryFilter').querySelectorAll('button').forEach((btn) => {
    const active = btn.dataset.filter === filter;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });
}

/* --------------------------------------------------------------- lightbox */

let lightboxNav = { prev: null, next: null };

/* Zoom state: 100% means "fits the viewport"; 1:1 means real pixels. */
const zoom = { scale: 1, tx: 0, ty: 0, bound: false };
const ZOOM_MIN = 1;
const ZOOM_MAX = 10;

function zoomApply(animate = true) {
  const img = el('lbImage');
  const badge = el('lbZoomLevel');
  img.classList.toggle('zooming', !animate);
  img.style.transform = `translate(${zoom.tx}px, ${zoom.ty}px) scale(${zoom.scale})`;
  if (badge) {
    badge.textContent = `${Math.round(zoom.scale * 100)}%`;
    badge.classList.toggle('active', zoom.scale > 1.01);
  }
}

/** Centre the image when it fits; clamp panning so it can never leave the stage. */
function zoomClamp() {
  const stage = el('lbStage');
  const img = el('lbImage');
  if (!stage || !img) return;
  const overflowX = Math.max(0, (img.clientWidth * zoom.scale - stage.clientWidth) / 2);
  const overflowY = Math.max(0, (img.clientHeight * zoom.scale - stage.clientHeight) / 2);
  zoom.tx = Math.max(-overflowX, Math.min(overflowX, zoom.tx));
  zoom.ty = Math.max(-overflowY, Math.min(overflowY, zoom.ty));
}

/** Zoom to `scale`, optionally keeping the point under (cx,cy) fixed. */
function zoomTo(scale, cx = null, cy = null, animate = true) {
  const stage = el('lbStage');
  if (!stage) return;
  const rect = stage.getBoundingClientRect();
  const px = (cx === null ? rect.left + rect.width / 2 : cx) - (rect.left + rect.width / 2);
  const py = (cy === null ? rect.top + rect.height / 2 : cy) - (rect.top + rect.height / 2);
  const next = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, scale));
  const k = next / zoom.scale;
  zoom.tx = px - (px - zoom.tx) * k;
  zoom.ty = py - (py - zoom.ty) * k;
  zoom.scale = next;
  if (zoom.scale <= 1.001) { zoom.scale = 1; zoom.tx = 0; zoom.ty = 0; }
  zoomClamp();
  zoomApply(animate);
}

function zoomReset() { zoom.scale = 1; zoom.tx = 0; zoom.ty = 0; zoomApply(true); }

/** 1:1 — show the file at its native pixel size (relative to the fitted size). */
function zoomPixel() {
  const img = el('lbImage');
  if (!img || !img.naturalWidth || !img.clientWidth) return;
  zoomTo(img.naturalWidth / img.clientWidth, null, null);
}

function bindZoomOnce() {
  if (zoom.bound) return;
  zoom.bound = true;
  const stage = el('lbStage');
  const img = el('lbImage');

  // wheel: zoom around the cursor (never scroll the page behind the overlay)
  stage.addEventListener('wheel', (event) => {
    if (!lightboxOpen()) return;
    event.preventDefault();
    zoomTo(zoom.scale * Math.exp(-event.deltaY * 0.0016), event.clientX, event.clientY, false);
  }, { passive: false });

  // double click: zoom in around the pointer, or return to fit
  stage.addEventListener('dblclick', (event) => {
    event.preventDefault();
    if (zoom.scale > 1.05) zoomReset();
    else zoomTo(2.5, event.clientX, event.clientY);
  });

  // drag to pan, two pointers to pinch
  const pointers = new Map();
  let pinchStart = 0;
  let pinchScale = 1;
  const spread = () => {
    const [a, b] = [...pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };
  stage.addEventListener('pointerdown', (event) => {
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    stage.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size === 2) { pinchStart = spread(); pinchScale = zoom.scale; }
    stage.classList.add('dragging');
  });
  stage.addEventListener('pointermove', (event) => {
    const prev = pointers.get(event.pointerId);
    if (!prev) return;
    const dx = event.clientX - prev.x;
    const dy = event.clientY - prev.y;
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (pointers.size === 2 && pinchStart > 0) {
      zoomTo(pinchScale * (spread() / pinchStart), null, null, false);
      return;
    }
    if (zoom.scale <= 1.001) return;      // nothing to pan while fitted
    zoom.tx += dx;
    zoom.ty += dy;
    zoomClamp();
    zoomApply(false);
  });
  const release = (event) => {
    pointers.delete(event.pointerId);
    if (pointers.size < 2) pinchStart = 0;
    if (!pointers.size) stage.classList.remove('dragging');
  };
  stage.addEventListener('pointerup', release);
  stage.addEventListener('pointercancel', release);

  el('lbZoomIn').onclick = () => zoomTo(zoom.scale * 1.3);
  el('lbZoomOut').onclick = () => zoomTo(zoom.scale / 1.3);
  el('lbZoomFit').onclick = zoomReset;
  el('lbZoomPixel').onclick = zoomPixel;
  img.addEventListener('load', zoomReset);
  img.addEventListener('dragstart', (event) => event.preventDefault());

  // + / - / 0 / 1 while the lightbox is open (main.js keeps ←/→ for navigation)
  document.addEventListener('keydown', (event) => {
    if (!lightboxOpen()) return;
    const key = event.key;
    if (key === '+' || key === '=') { event.preventDefault(); zoomTo(zoom.scale * 1.3); }
    else if (key === '-' || key === '_') { event.preventDefault(); zoomTo(zoom.scale / 1.3); }
    else if (key === '0') { event.preventDefault(); zoomReset(); }
    else if (key === '1') { event.preventDefault(); zoomPixel(); }
  });
}

export function bindLightbox(nav) {
  lightboxNav = nav;
  el('lbPrev').onclick = () => lightboxNav.prev?.();
  el('lbNext').onclick = () => lightboxNav.next?.();
  el('lbClose').onclick = closeLightbox;
  const remix = el('lbRemix');
  if (remix) remix.onclick = () => { closeLightbox(); lightboxNav.remix?.(); };
  el('lightbox').onclick = (event) => { if (event.target.id === 'lightbox') closeLightbox(); };
  bindZoomOnce();
  let startX = 0;
  el('lightbox').addEventListener('touchstart', (e) => { startX = e.touches[0].clientX; }, { passive: true });
  el('lightbox').addEventListener('touchend', (e) => {
    if (zoom.scale > 1.01) return;      // zoomed in: swipes are panning, not navigation
    const dx = e.changedTouches[0].clientX - startX;
    if (Math.abs(dx) > 45) (dx > 0 ? lightboxNav.prev : lightboxNav.next)?.();
  }, { passive: true });
}

export function openLightbox(url, caption, downloadUrl, filename) {
  const box = el('lightbox');
  const img = el('lbImage');
  bindZoomOnce();
  img.classList.add('zooming');
  img.src = url;
  zoom.scale = 1;
  zoom.tx = 0;
  zoom.ty = 0;
  img.style.transform = '';
  if (el('lbZoomLevel')) el('lbZoomLevel').textContent = '100%';
  el('lbCaption').textContent = caption;
  const link = el('lbDownload');
  link.href = downloadUrl || url;
  link.download = filename || 'landscape.webp';
  box.classList.add('show');
  box.setAttribute('aria-hidden', 'false');
  el('lbClose').focus();
}

export function closeLightbox() {
  const box = el('lightbox');
  box.classList.remove('show');
  box.setAttribute('aria-hidden', 'true');
  const img = el('lbImage');
  img.src = '';
  img.style.transform = '';
  zoom.scale = 1;
  zoom.tx = 0;
  zoom.ty = 0;
}

export function lightboxOpen() {
  return el('lightbox').classList.contains('show');
}

/* ------------------------------------------------------------------ modal */

export function renderShortcuts() {
  el('shortcutList').innerHTML = SHORTCUTS
    .map(([keys, what]) => `<div><kbd>${keys}</kbd><span>${what}</span></div>`).join('');
}

export function openModal(id) {
  const box = el(id);
  box.classList.add('show');
  box.setAttribute('aria-hidden', 'false');
  box.querySelector('button, [href], input, select, textarea')?.focus();
}

export function closeModal(id) {
  const box = el(id);
  box.classList.remove('show');
  box.setAttribute('aria-hidden', 'true');
}

export function anyModalOpen() {
  return [...document.querySelectorAll('.modal.show')].length > 0;
}

/* ----------------------------------------------------------------- health */

export function renderHealth(data, error) {
  const badge = el('srv');
  if (error || !data?.ok) {
    badge.textContent = '离线';
    badge.className = 'bad';
    el('healthGrid').innerHTML = `<div class="kv"><small>错误</small><b>${error || '后端未就绪'}</b></div>`;
    return;
  }
  badge.textContent = `在线 · ${data.format || 'webp'}`;
  badge.className = 'ok';
  el('queueMetric').textContent = `${data.queued ?? 0} / ${data.running ?? 0}`;
  el('doneMetric').textContent = data.completed ?? 0;
  const memory = data.memory || {};
  const storage = data.storage || {};
  const pipeline = data.pipeline || {};
  const perStep = Object.values(data.sec_per_step || {});
  const average = perStep.length ? (perStep.reduce((a, b) => a + b, 0) / perStep.length) : 0;
  el('vramMetric').textContent = memory.cuda
    ? `${Math.round(memory.allocated_mb)} / ${Math.round(memory.total_mb)} MB` : 'CPU';
  const rows = [
    ['显存已分配', memory.cuda ? `${memory.allocated_mb} MB` : '—'],
    ['显存保留', memory.cuda ? `${memory.reserved_mb} MB` : '—'],
    ['显存峰值', memory.cuda ? `${memory.peak_mb} MB` : '—'],
    ['结果占用', `${storage.results_mb ?? 0} MB / ${storage.quota_mb || '∞'} MB (${storage.usage_pct ?? 0}%)`],
    ['已回收', `${storage.reclaimed_files ?? 0} 个 · ${storage.reclaimed_mb ?? 0} MB`],
    ['管线', pipeline.model_loaded ? `已加载 ${pipeline.sampler || ''}${pipeline.fast ? ' (少步)' : ''}` : '未加载（懒加载）'],
    ['空闲释放', `${pipeline.idle_sec ?? 0} s / ${pipeline.idle_unload_sec ?? 0} s · 已释放 ${pipeline.idle_unloads ?? 0} 次`],
    ['平均每步', average ? `${average.toFixed(3)} s` : '暂无样本'],
    ['任务', `${data.tracked_jobs ?? 0} 条记录 · 队列上限 ${data.max_queue ?? '—'}`],
    ['上传', data.upload ? `允许 ≤ ${data.max_upload_mb} MB` : '已关闭'],
  ];
  el('healthGrid').innerHTML = rows
    .map(([k, v]) => `<div class="kv"><small>${k}</small><b>${v}</b></div>`).join('');
}

export function describeClarity(key) {
  return CLARITY[key]?.label || key;
}

export function renderUploadInfo(info) {
  const box = el('uploadInfo');
  if (!info) {
    box.innerHTML = '';
    el('uploadPreview').innerHTML = '';
    el('uploadClear').hidden = true;
    return;
  }
  el('uploadClear').hidden = false;
  el('uploadPreview').innerHTML = `<img src="${info.thumbUrl}" alt="参考图预览">`;
  box.innerHTML = `<b>${info.width}×${info.height}</b> · ${formatBytes(info.originalBytes)} →
    <b>${formatBytes(info.bytes)}</b>（压缩 ${info.ratio.toFixed(1)}× · ${info.ms} ms · ${info.mime.replace('image/', '')}）`;
}
