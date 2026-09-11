/**
 * Behaviour layer: reads the form, submits jobs, tracks progress, and drives the
 * canvas / gallery / lightbox.  Keep DOM writes in ui.js and network calls in
 * api.js so this file stays about *what happens*, not *how it looks*.
 */
import {
  CLARITY, DIRECT_LIMIT, MAX_BLOB_CACHE, MAX_HISTORY, NEG_PRESETS, RANDOM_PROMPTS, RECIPES,
} from './config.js';
import * as api from './api.js';
import * as store from './store.js';
import * as ui from './ui.js';
import * as extras from './extras.js';
import * as palette from './palette.js';
import * as studio from './studio.js';
import * as motion from './motion.js';
import * as scenes from './scenes.js';
import * as composer from './composer.js';
import * as ab from './ab.js';
import * as shell from './shell.js';
import { compressForUpload, extractImage, formatBytes } from './upload.js';

const SETTING_IDS = ['prompt', 'style', 'random', 'recipe', 'sampler', 'steps', 'cfg',
  'seed', 'res', 'aspect', 'batch', 'clarity', 'sr', 'neg', 'strength'];

const modifiers = new Set();
const watchers = new Map();
let galleryFilter = 'all';
let viewIndex = 0;
let compareOn = false;
let upload = null;
let healthTimer = null;
let healthBackoff = 30000;

const pick = (list) => list[Math.floor(Math.random() * list.length)];

/* --------------------------------------------------------------- settings */

function readSettings() {
  const settings = store.getState().settings;
  SETTING_IDS.forEach((id) => {
    const node = ui.el(id);
    if (node) settings[id] = node.type === 'checkbox' ? node.checked : node.value;
  });
  settings.modifiers = [...modifiers];
  store.save();
}

function applySettings() {
  const settings = store.getState().settings;
  SETTING_IDS.forEach((id) => {
    const node = ui.el(id);
    const value = settings[id];
    if (!node || value === undefined) return;
    if (node.type === 'checkbox') node.checked = !!value;
    else node.value = value;
  });
  modifiers.clear();
  (settings.modifiers || []).forEach((m) => modifiers.add(m));
  ui.renderModifiers(modifiers, toggleModifier);
  syncRecipe(false);
  refreshComposed();
}

function composedPrompt() {
  const base = ui.el('random').checked
    ? pick(RANDOM_PROMPTS)
    : ui.el('prompt').value.trim();
  // Prompt Studio (camera / lighting / composition) contributes plain English
  // fragments; the user's own words are never rewritten by that module.
  const studioBits = studio.fragmentsText();
  const parts = [base, studioBits, ...modifiers];
  return parts.filter(Boolean).join(', ').slice(0, 900);
}

function refreshComposed() {
  const value = ui.el('prompt').value;
  ui.el('promptCount').textContent = `${value.length}/500`;
  ui.renderComposed(composedPrompt());
  composer.refresh();
}

function toggleModifier(fragment, chip) {
  if (modifiers.has(fragment)) modifiers.delete(fragment);
  else modifiers.add(fragment);
  chip.classList.toggle('active', modifiers.has(fragment));
  chip.setAttribute('aria-pressed', modifiers.has(fragment) ? 'true' : 'false');
  refreshComposed();
  readSettings();
}

/** Recipe select drives sampler/steps/cfg unless the user picked 自定义. */
function syncRecipe(notify = true) {
  const id = ui.el('recipe').value;
  const recipe = RECIPES.find((r) => r.id === id);
  if (recipe && recipe.steps) {
    ui.el('steps').value = recipe.steps;
    ui.el('cfg').value = recipe.cfg;
    ui.el('sampler').value = recipe.sampler;
  }
  const fast = ['lcm', 'tcd'].includes(ui.el('sampler').value);
  ui.el('steps').max = fast ? 12 : 60;
  ui.el('cfg').max = fast ? 2.5 : 15;
  if (fast) {
    ui.el('steps').value = Math.min(12, +ui.el('steps').value || 6);
    ui.el('cfg').value = Math.min(2.5, +ui.el('cfg').value || 1.5);
  }
  ui.el('stepsHint').textContent = ui.el('steps').value;
  ui.el('cfgHint').textContent = ui.el('cfg').value;
  ui.el('negHint').textContent = fast ? '少步模式下负向引导很弱' : '';
  if (notify) readSettings();
}

function requestBody(seed) {
  const clarity = CLARITY[ui.el('clarity').value] || CLARITY.std;
  const sampler = ui.el('sampler').value;
  const fast = ['lcm', 'tcd'].includes(sampler);
  const body = {
    prompt: composedPrompt() || pick(RANDOM_PROMPTS),
    style: ui.el('style').value,
    res: ui.el('res').value,
    aspect: ui.el('aspect').value,
    steps: Math.max(2, Math.min(60, +ui.el('steps').value || 24)),
    cfg: +ui.el('cfg').value || 7.5,
    seed,
    neg: ui.el('neg').value.trim(),
    upscale: false,
    highres: clarity.highres,
    enhance: clarity.enhance,
    sr_model: ui.el('sr').value,
    enhance_strength: ui.el('clarity').value === '4k' ? 0 : 0.3,
    fast,
    sampler,
  };
  if (upload) {
    body.init_image = upload.dataUrl;
    body.strength = +ui.el('strength').value || 0.55;
  }
  return body;
}

/* ------------------------------------------------------------- image flow */

/**
 * One job can carry several images (server-side batch).  Everything that shows
 * or navigates images works on a flattened list of {job, id, i, bytes} items, so
 * ←/→ walks every generated image across every job.
 */
function itemsOf(job) {
  const count = Math.max(1, Math.min(4, job.count || 1));
  const sizes = Array.isArray(job.bytes_each) ? job.bytes_each : [];
  if (count === 1) return [{ job, id: job.id, i: 0, bytes: sizes[0] || job.bytes || 0 }];
  return Array.from({ length: count }, (_, i) => ({ job, id: job.id, i, bytes: sizes[i] || 0 }));
}

function viewList() {
  return store.doneJobs().flatMap(itemsOf);
}

const itemBlobs = [];

function rememberItemBlob(item, url) {
  item.url = url;
  if (!url.startsWith('blob:')) return url;
  itemBlobs.push(url);
  while (itemBlobs.length > MAX_BLOB_CACHE) URL.revokeObjectURL(itemBlobs.shift());
  return url;
}

async function ensureUrl(item) {
  if (item.url) return item.url;
  if (!item.bytes || item.bytes <= DIRECT_LIMIT) return api.resultUrl(item.id, item.i);
  ui.setResultMeta(ui.el('resultMeta').textContent, `分块下载 ${formatBytes(item.bytes)}…`);
  const result = await api.fetchImage(item.id, item.bytes, (got, total) => {
    ui.setResultMeta(ui.el('resultMeta').textContent, `分块下载 ${Math.round((got / total) * 100)}%`);
  }, item.i);
  ui.setResultMeta(ui.el('resultMeta').textContent,
    `WebP · ${result.chunks} 分块并行 · ${formatBytes(result.bytes)}`);
  return rememberItemBlob(item, result.url);
}

function describe(job) {
  const size = job.width && job.height ? `${job.width}×${job.height}` : `${job.res || ''}`;
  const bits = [
    `seed ${job.seed}`, `${job.steps || '?'} 步`, job.sampler || '',
    size, job.bytes ? formatBytes(job.bytes) : '', job.seconds ? `${job.seconds}s` : '',
    job.cached ? '⚡缓存' : '', (job.count || 1) > 1 ? `×${job.count}` : '',
  ];
  return bits.filter(Boolean).join(' · ');
}

async function showByIndex(index) {
  const list = viewList();
  if (!list.length) {
    ui.showEmptyStage();
    ui.setStageNav(false, '');
    return;
  }
  viewIndex = (index + list.length) % list.length;
  const item = list[viewIndex];
  store.getState().last = item.id;
  store.save();
  const total = item.job.count || 1;
  if (total > 1 && !compareOn) {
    const urls = await Promise.all(itemsOf(item.job).map(ensureUrl));
    ui.showGrid(urls, `${describe(item.job)} · ${urls.length} 张一组`);
    ui.setResultMeta(`${describe(item.job)} · 批量 ${urls.length} 张`);
  } else if (compareOn && list.length > 1) {
    const other = list[(viewIndex + 1) % list.length];
    ui.showCompare(await ensureUrl(item), await ensureUrl(other),
      `${describe(item.job)}  ⟷  ${describe(other.job)}`);
  } else {
    ui.showImage(await ensureUrl(item), { alt: item.job.prompt || '生成的风景图' });
    ui.setResultMeta(total > 1 ? `${describe(item.job)} · 第 ${item.i + 1}/${total} 张` : describe(item.job));
  }
  ui.setStageNav(list.length > 1, `${viewIndex + 1} / ${list.length}`);
}

async function openLightboxFor(job) {
  const list = viewList();
  const index = list.findIndex((entry) => entry.id === job.id);
  if (index >= 0) viewIndex = index;
  const item = list[index >= 0 ? index : 0];
  if (!item) return;
  const url = await ensureUrl(item);
  ui.openLightbox(url, `${viewIndex + 1} / ${list.length} · ${describe(item.job)}`,
    url, `landscape_${item.id}${item.i ? `_${item.i}` : ''}.webp`);
}

async function stepView(delta) {
  const list = viewList();
  if (!list.length) return;
  await showByIndex(viewIndex + delta);
  if (ui.lightboxOpen()) await openLightboxFor(viewList()[viewIndex].job);
}

/* --------------------------------------------------------------- generate */

/* Heavy-usage guard: several jobs polling at once each used to rebuild the
 * 60-tile gallery and the queue on every tick.  Coalesce into one render per
 * animation frame, and skip it entirely while the tab is hidden. */
let galleryScheduled = false;
let queueScheduled = false;

function scheduleGallery() {
  if (galleryScheduled || document.hidden) return;
  galleryScheduled = true;
  requestAnimationFrame(() => { galleryScheduled = false; renderGallery(); });
}

function scheduleQueue() {
  if (queueScheduled || document.hidden) return;
  queueScheduled = true;
  requestAnimationFrame(() => { queueScheduled = false; ui.renderQueue(store.getState().jobs); });
}

function trackJob(record, onSettled) {
  if (watchers.has(record.id)) return;
  const stop = api.watchJob(record.id, {
    onUpdate: (data) => {
      if (data.status === 'retry') { ui.log('poll retry', data.error); return; }
      store.updateJob(record.id, {
        status: data.status, stage: data.stage, progress: data.progress,
        eta: data.eta, elapsed: data.elapsed, seed: data.seed, steps: data.steps,
        sampler: data.sampler,
      });
      if (store.getState().last === record.id || !store.doneJobs().length) {
        ui.setProgress({ percent: data.progress || 0, stage: data.stage, eta: data.eta });
        if (data.preview_ready && data.preview_rev !== record.previewRev) {
          record.previewRev = data.preview_rev;
          ui.showImage(api.previewUrl(record.id, data.preview_rev), { preview: true });
          ui.setResultMeta(`预览 · ${data.preview_kind === 'latent' ? '潜空间' : '基础图'} · ${describe(data)}`);
        }
      }
      ui.setStatus(data.status === 'running' ? 'GPU 渲染中…' : '排队中…', 'busy');
      motion.setButtonState(ui.el('generate'), 'busy', motion.busyLabel(data.progress));
      scheduleQueue();
      scheduleGallery();
    },
    onDone: async (data) => {
      watchers.delete(record.id);
      const sizes = Array.isArray(data.bytes_each) && data.bytes_each.length
        ? data.bytes_each : [data.bytes || 0];
      const job = store.updateJob(record.id, {
        ...data, status: 'done', info: describe(data),
        count: data.count || sizes.length || 1,
        images: sizes.map((bytes, i) => ({ i, bytes })),
      });
      store.getState().last = record.id;
      store.save();
      ui.setProgress({ visible: false });
      const n = job?.count || 1;
      ui.setStatus(`✅ 完成${n > 1 ? ` ${n} 张` : ''} · ${data.seconds ?? '?'} s`, 'ok');
      const genBtn = ui.el('generate');
      genBtn.disabled = false;
      const dock = ui.el('dockGenerate');
      if (dock) dock.disabled = false;
      motion.setButtonState(genBtn, 'done', '✓ 已完成');
      if (dock) motion.setButtonState(dock, 'done', '✓ 已完成');
      window.setTimeout(() => { motion.setButtonState(genBtn, 'idle'); if (dock) motion.setButtonState(dock, 'idle'); }, 900);
      ui.renderQueue(store.getState().jobs);
      const list = viewList();
      await showByIndex(list.findIndex((x) => x.id === job.id));
      renderGallery();
      ui.log('job done', data);
      if (typeof onSettled === 'function') onSettled(job);
    },
    onFail: (data) => {
      watchers.delete(record.id);
      store.updateJob(record.id, { status: data.status, error: data.error, info: data.error || '失败' });
      ui.setProgress({ visible: false });
      ui.setStatus(`失败：${data.error || '未知错误'}`, 'bad');
      const genBtn = ui.el('generate');
      genBtn.disabled = false;
      const dock = ui.el('dockGenerate');
      if (dock) dock.disabled = false;
      motion.setButtonState(genBtn, 'idle');
      motion.setButtonState(dock, 'idle');
      ui.renderQueue(store.getState().jobs);
      renderGallery();
      ui.log('job failed', data);
      if (typeof onSettled === 'function') onSettled({ id: record.id, status: 'failed' });
    },
    onGone: () => {
      watchers.delete(record.id);
      store.updateJob(record.id, { status: 'expired', info: '后端已清理' });
      const genBtn = ui.el('generate');
      genBtn.disabled = false;
      const dock = ui.el('dockGenerate');
      if (dock) dock.disabled = false;
      motion.setButtonState(genBtn, 'idle');
      motion.setButtonState(dock, 'idle');
      renderGallery();
    },
  });
  watchers.set(record.id, stop);
}

/**
 * Submit a job.  Optional overrides power the A/B lab and Seed Lab:
 *   opts.prompt    replace the composed prompt (A/B variants)
 *   opts.seed      force a seed baseline (A/B comparability)
 *   opts.batch     image count per job (A/B uses 2)
 *   opts.onSettled(job)  called with the final record (done/failed)
 */
async function generate(forceNewSeed = false, opts = {}) {
  readSettings();
  const button = ui.el('generate');
  const dock = ui.el('dockGenerate');
  button.disabled = true;
  if (dock) dock.disabled = true;
  motion.setButtonState(button, 'busy', '◌ 准备中…');
  motion.setButtonState(dock, 'busy', '◌ 准备中…');
  ui.setStatus('提交任务…', 'busy');
  ui.setProgress({ percent: 0, stage: 'queued', visible: true });
  let seed = forceNewSeed ? -1 : +ui.el('seed').value;
  if (opts.seed !== undefined) seed = opts.seed;
  const count = Math.max(1, Math.min(4, opts.batch || +ui.el('batch').value || 1));
  if (seed < 0) seed = Math.floor(Math.random() * 1e9);
  const body = requestBody(seed);
  if (opts.prompt) body.prompt = opts.prompt;
  body.batch = count;            // one queue slot, one poll, N images on the GPU
  try {
    ui.log('generate', { ...body, init_image: body.init_image ? `<${formatBytes(upload.bytes)} webp>` : '' });
    const response = await api.submit(body);

    // ---- cache hit: identical request already produced this exact image ----
    if (response.cached && response.status === 'done') {
      const job = store.addJob({
        id: response.job_id, status: 'done', seed, count: 1, seeds: [seed],
        steps: body.steps, sampler: body.sampler, res: body.res, prompt: body.prompt,
        created: Date.now(), cached: true, seconds: 0, info: '⚡ 缓存命中 · 0 s',
        images: [{ i: 0, bytes: 0 }], hasInit: !!body.init_image,
      });
      store.getState().last = job.id;
      store.save();
      ui.setStatus('⚡ 缓存命中 · 秒回', 'ok');
      ui.setProgress({ visible: false });
      ui.toast('⚡ 同参数同 seed 已有结果，直接复用');
      motion.setButtonState(button, 'done', '⚡ Instant');
      if (dock) motion.setButtonState(dock, 'done', '⚡ Instant');
      button.disabled = false;
      if (dock) dock.disabled = false;
      window.setTimeout(() => { motion.setButtonState(button, 'idle'); if (dock) motion.setButtonState(dock, 'idle'); }, 900);
      renderGallery();
      ui.renderQueue(store.getState().jobs);
      await showByIndex(viewList().findIndex((v) => v.id === job.id));
      opts.onSettled?.(job);
      return;
    }

    const job = store.addJob({
      id: response.job_id, status: 'queued', seed, count: response.count || count,
      seeds: response.seeds || [], steps: body.steps, sampler: body.sampler, res: body.res,
      prompt: body.prompt, eta: response.eta, created: Date.now(),
      info: `seed ${seed}${count > 1 ? ` ×${count}` : ''} · 排队中`, hasInit: !!body.init_image,
    });
    renderGallery();
    ui.renderQueue(store.getState().jobs);
    trackJob(job, opts.onSettled);
    ui.setStatus(count > 1 ? `已提交 ${count} 张（单任务批量）` : '已提交 1 个任务', 'busy');
    ui.toast(count > 1 ? `${count} 张已入队 · 单任务批量` : '任务已入队');
  } catch (error) {
    ui.log('generate error', String(error.message || error));
    ui.setStatus(`提交失败：${error.message || error}`, 'bad');
    ui.setProgress({ visible: false });
    button.disabled = false;
    const dock = ui.el('dockGenerate');
    if (dock) dock.disabled = false;
    motion.setButtonState(button, 'idle');
    motion.setButtonState(dock, 'idle');
    opts.onSettled?.({ status: 'failed' });
  }
}

/* ---------------------------------------------------------------- gallery */

/** Push a history entry's parameters back into the form (gallery + library). */
function reuseJob(job) {
  if (!job) return;
  if (job.prompt) ui.el('prompt').value = job.prompt;
  if (job.seed !== undefined) ui.el('seed').value = job.seed;
  if (job.steps) ui.el('steps').value = job.steps;
  if (job.sampler) {
    ui.el('sampler').value = job.sampler;
    ui.el('recipe').value = 'custom';
  }
  if (job.res) ui.el('res').value = job.res;
  syncRecipe();
  refreshComposed();
  ui.toast('已复用该图参数');
}

function renderGallery() {
  ui.renderGallery(store.getState().jobs, galleryFilter, {
    imageUrl: api.resultUrl,
    onOpen: openLightboxFor,
    onExpired: (job) => { store.updateJob(job.id, { status: 'expired', info: '后端已清理' }); renderGallery(); },
    onDownload: async (job) => {
      const url = await ensureUrl(job);
      const link = document.createElement('a');
      link.href = url;
      link.download = `landscape_${job.id}.webp`;
      link.click();
    },
    onReuse: reuseJob,
    onDelete: (job) => {
      watchers.get(job.id)?.();
      watchers.delete(job.id);
      store.removeJob(job.id);
      renderGallery();
      showByIndex(viewIndex);
    },
  });
  extras.refreshExtras();
}

/* ----------------------------------------------------------------- health */

async function refreshHealth() {
  try {
    const data = await api.health();
    ui.renderHealth(data);
    ui.el('uploadHint').textContent = data.upload
      ? `参考图会在浏览器压缩到 ≤ ${data.max_upload_mb} MB 后上传`
      : '后端已关闭上传功能';
    healthBackoff = 30000;
    ui.log('health', {
      queued: data.queued, running: data.running, vram: data.memory?.allocated_mb,
      results_mb: data.storage?.results_mb, loaded: data.pipeline?.model_loaded,
    });
  } catch (error) {
    ui.renderHealth(null, String(error.message || error));
    healthBackoff = Math.min(healthBackoff * 2, 240000);
  }
  clearTimeout(healthTimer);
  healthTimer = setTimeout(refreshHealth, document.hidden ? healthBackoff * 2 : healthBackoff);
}

/* ------------------------------------------------------------ share links */

function encodeShare() {
  const payload = { s: store.getState().settings, m: [...modifiers] };
  return btoa(unescape(encodeURIComponent(JSON.stringify(payload))))
    .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function applyShare(hash) {
  try {
    const base = hash.replace(/^#p=/, '').replace(/-/g, '+').replace(/_/g, '/');
    const data = JSON.parse(decodeURIComponent(escape(atob(base))));
    if (data.s) store.getState().settings = { ...store.getState().settings, ...data.s };
    if (Array.isArray(data.m)) store.getState().settings.modifiers = data.m;
    applySettings();
    ui.toast('已载入分享参数');
    return true;
  } catch (error) {
    return false;
  }
}

/* -------------------------------------------------------------- wiring up */

/** Compress + register a reference image (dropzone, paste, and ♻ Remix share this). */
async function applyUploadFile(file) {
  if (!file) return;
  try {
    ui.el('uploadInfo').textContent = '压缩中…';
    upload = await compressForUpload(file);
    ui.renderUploadInfo(upload);
    ui.el('strengthRow').hidden = false;
    ui.log('upload compressed', { from: upload.originalBytes, to: upload.bytes, ms: upload.ms });
    ui.toast(`参考图已压缩 ${upload.ratio.toFixed(1)}×`);
  } catch (error) {
    upload = null;
    ui.renderUploadInfo(null);
    ui.toast(`上传失败：${error.message || error}`, 'bad');
    throw error;
  }
}

/** Load the image currently on stage as an img2img reference (keeps the composition). */
async function remixCurrent() {
  const item = viewList()[viewIndex];
  if (!item) { ui.toast('画布上还没有可 Remix 的图'); return; }
  ui.toast('♻ 正在载入 Remix 参考图…');
  try {
    // ensureUrl() takes a flattened view item — passing item.job would always
    // resolve image 0 of a batch instead of the one on screen.
    const url = await ensureUrl(item);
    const blob = await (await fetch(url)).blob();
    if (!blob || !blob.size) throw new Error('图片不可用');
    const file = new File([blob], `remix_${item.id}.webp`, { type: blob.type || 'image/webp' });
    await applyUploadFile(file);
    ui.el('strength').value = 0.45;
    ui.el('strengthHint').textContent = '0.45';
    readSettings();
    ui.toast('♻ 已设为参考图（重绘强度 0.45）· 点「生成」即可保留构图再创作');
  } catch (error) {
    ui.log('remix error', String(error?.message || error));
    ui.toast(`Remix 失败：${error?.message || error}`, 'bad');
  }
}

function bindUpload() {
  const zone = ui.el('dropzone');
  const input = ui.el('file');
  const accept = (file) => { if (file) applyUploadFile(file).catch(() => {}); };
  zone.onclick = () => input.click();
  zone.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } };
  input.onchange = () => accept(input.files?.[0]);
  ['dragenter', 'dragover'].forEach((type) => zone.addEventListener(type, (e) => {
    e.preventDefault();
    zone.classList.add('hover');
  }));
  ['dragleave', 'drop'].forEach((type) => zone.addEventListener(type, () => zone.classList.remove('hover')));
  zone.addEventListener('drop', (e) => { e.preventDefault(); accept(extractImage(e)); });
  window.addEventListener('paste', (e) => {
    if (document.activeElement?.tagName === 'TEXTAREA' || document.activeElement?.tagName === 'INPUT') return;
    const file = extractImage(e);
    if (file) accept(file);
  });
  ui.el('uploadClear').onclick = () => {
    if (upload?.thumbUrl) URL.revokeObjectURL(upload.thumbUrl);
    upload = null;
    ui.renderUploadInfo(null);
    ui.el('strengthRow').hidden = true;
    input.value = '';
  };
}

function bindShortcuts() {
  document.addEventListener('keydown', (event) => {
    const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') { event.preventDefault(); generate(); return; }
    if (event.key === 'Escape') { ui.closeLightbox(); ['helpModal', 'dataModal'].forEach(ui.closeModal); return; }
    if (event.key === 'ArrowLeft') { if (!typing) stepView(-1); return; }
    if (event.key === 'ArrowRight') { if (!typing) stepView(1); return; }
    if (typing) return;
    if (event.key === '?') ui.openModal('helpModal');
    if (event.key.toLowerCase() === 'f') { const job = store.doneJobs()[viewIndex]; if (job) openLightboxFor(job); }
    if (event.key.toLowerCase() === 'c') ui.el('compareBtn').click();
    if (event.key.toLowerCase() === 'r') generate(true);
    if (event.key.toLowerCase() === 's') ui.el('download').click();
  });
}

function bindControls() {
  ui.el('generate').onclick = () => generate();
  ui.el('reroll').onclick = () => generate(true);
  ui.el('dice').onclick = () => { ui.el('seed').value = Math.floor(Math.random() * 1e9); readSettings(); };
  ui.el('recipe').onchange = () => syncRecipe();
  ui.el('sampler').onchange = () => { ui.el('recipe').value = 'custom'; syncRecipe(); };
  ['steps', 'cfg', 'strength'].forEach((id) => {
    ui.el(id).oninput = () => {
      const hint = ui.el(`${id}Hint`);
      if (hint) hint.textContent = ui.el(id).value;
      readSettings();
    };
  });
  SETTING_IDS.forEach((id) => ui.el(id)?.addEventListener('change', readSettings));
  ui.el('prompt').oninput = () => { refreshComposed(); readSettings(); };
  ui.el('random').onchange = () => { refreshComposed(); readSettings(); };
  ui.el('sceneSearch').oninput = () => scenes.filterScenes(ui.el('sceneSearch').value);
  ui.el('surprise').onclick = () => {
    ui.el('prompt').value = pick(RANDOM_PROMPTS);
    refreshComposed();
    readSettings();
  };
  ui.el('clearPrompt').onclick = () => {
    ui.el('prompt').value = '';
    modifiers.clear();
    ui.renderModifiers(modifiers, toggleModifier);
    refreshComposed();
    readSettings();
  };
  ui.el('copyPrompt').onclick = () => {
    navigator.clipboard?.writeText(composedPrompt()).then(() => ui.toast('完整 prompt 已复制'));
  };
  ui.el('negPreset').onchange = () => {
    const preset = NEG_PRESETS.find(([name]) => name === ui.el('negPreset').value);
    if (preset) { ui.el('neg').value = preset[1]; readSettings(); }
  };
  ui.el('download').onclick = async () => {
    const job = store.doneJobs()[viewIndex];
    if (!job) return ui.toast('还没有完成的图片');
    const link = document.createElement('a');
    link.href = await ensureUrl(job);
    link.download = `landscape_${job.id}.webp`;
    link.click();
  };
  ui.el('copySeed').onclick = () => {
    const job = store.doneJobs()[viewIndex];
    if (job) navigator.clipboard?.writeText(String(job.seed)).then(() => ui.toast('seed 已复制'));
  };
  ui.el('compareBtn').onclick = () => {
    if (store.doneJobs().length < 2) return ui.toast('至少需要两张完成的图');
    compareOn = !compareOn;
    ui.el('compareBtn').setAttribute('aria-pressed', compareOn ? 'true' : 'false');
    showByIndex(viewIndex);
  };
  ui.el('fullscreen').onclick = () => {
    const job = store.doneJobs()[viewIndex];
    if (job) openLightboxFor(job);
  };
  ui.el('warmBtn').onclick = async () => {
    ui.el('warmBtn').disabled = true;
    ui.setStatus('预热管线中（首次约 30 s）…', 'busy');
    try {
      const info = await api.warmup({ fast: ['lcm', 'tcd'].includes(ui.el('sampler').value), sampler: ui.el('sampler').value });
      ui.setStatus(`✅ 管线已就绪（${info.seconds}s），下一张不再冷启动`, 'ok');
      ui.log('warmup', info);
    } catch (error) {
      ui.setStatus(`预热失败：${error.message || error}`, 'bad');
    } finally {
      ui.el('warmBtn').disabled = false;
      refreshHealth();
    }
  };
  ui.el('gcBtn').onclick = async () => {
    try {
      const info = await api.collectGarbage(ui.el('gcUnload').checked);
      ui.log('gc', info);
      ui.toast('后端已回收');
      refreshHealth();
    } catch (error) {
      ui.toast(`回收失败：${error.message || error}`, 'bad');
    }
  };
  ui.el('healthBtn').onclick = refreshHealth;
  ui.el('helpBtn').onclick = () => ui.openModal('helpModal');
  ui.el('dataBtn').onclick = () => {
    ui.el('dataText').value = store.exportPayload();
    ui.openModal('dataModal');
  };
  document.querySelectorAll('[data-close]').forEach((btn) => {
    btn.onclick = () => ui.closeModal(btn.dataset.close);
  });
  ui.el('themeBtn').onclick = () => {
    ui.applyTheme(store.setTheme(store.getState().theme === 'light' ? 'dark' : 'light'));
  };
  ui.el('shareBtn').onclick = () => {
    readSettings();
    const url = `${location.origin}${location.pathname}#p=${encodeShare()}`;
    navigator.clipboard?.writeText(url).then(() => ui.toast('分享链接已复制')).catch(() => ui.toast(url));
  };
  ui.el('saveRecipe').onclick = () => {
    readSettings();
    const name = prompt('给这个配方起个名字', `我的配方 ${store.getState().userRecipes.length + 1}`);
    if (!name) return;
    store.saveUserRecipe({
      name,
      summary: `${ui.el('sampler').value} · ${ui.el('steps').value} 步 · CFG ${ui.el('cfg').value} · ${ui.el('res').value}`,
      settings: { ...store.getState().settings },
    });
    drawRecipes();
    ui.toast('配方已保存');
  };
  ui.el('galleryFilter').querySelectorAll('button').forEach((btn) => {
    btn.onclick = () => {
      galleryFilter = btn.dataset.filter;
      ui.setGalleryFilter(galleryFilter);
      renderGallery();
    };
  });
  ui.el('clearHistory').onclick = () => {
    if (!confirm('清空浏览器里的历史记录？（不会删除后端文件）')) return;
    watchers.forEach((stop) => stop());
    watchers.clear();
    store.clearJobs();
    renderGallery();
    ui.renderQueue([]);
    ui.showEmptyStage('历史已清空');
    ui.setStageNav(false, '');
  };
  ui.el('copyLog').onclick = () => {
    const text = ui.logText();
    if (!text) return ui.toast('日志为空');
    navigator.clipboard?.writeText(text).then(() => ui.toast('日志已复制')).catch(() => ui.toast('剪贴板不可用', 'bad'));
  };
  ui.el('importBtn').onclick = () => {
    try {
      const info = store.importPayload(ui.el('dataText').value);
      renderGallery();
      drawRecipes();
      ui.toast(`已导入：${info.jobs} 条记录`);
      ui.closeModal('dataModal');
    } catch (error) {
      ui.toast(`导入失败：${error.message || error}`, 'bad');
    }
  };
  ui.el('downloadJson').onclick = () => {
    const blob = new Blob([store.exportPayload()], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `landscape-art-${Date.now()}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 4000);
  };
}

function drawRecipes() {
  ui.renderRecipes(RECIPES, store.getState().userRecipes, {
    onUse: (recipe) => {
      if (recipe.settings) {
        store.getState().settings = { ...store.getState().settings, ...recipe.settings };
        applySettings();
      } else {
        ui.el('recipe').value = recipe.id;
        syncRecipe();
      }
      ui.toast(`已套用 ${recipe.label || recipe.name}`);
    },
    onDelete: (recipe) => { store.deleteUserRecipe(recipe.name); drawRecipes(); },
  });
}

function fillNegPresets() {
  const select = ui.el('negPreset');
  NEG_PRESETS.forEach(([name]) => {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  });
}

async function restore() {
  store.load();
  ui.applyTheme(store.getState().theme);
  fillNegPresets();
  ui.renderModifiers(modifiers, toggleModifier);
  ui.renderShortcuts();
  drawRecipes();
  applySettings();
  scenes.renderScenes();           // re-render with loaded favorites (sceneFavs)
  if (location.hash.startsWith('#p=')) applyShare(location.hash);
  ui.setGalleryFilter(galleryFilter);
  renderGallery();
  ui.renderQueue(store.getState().jobs);
  const done = store.doneJobs();
  const lastId = store.getState().last;
  const index = Math.max(0, done.findIndex((job) => job.id === lastId));
  if (done.length) await showByIndex(index);
  else ui.showEmptyStage();
  const pending = store.getState().jobs.filter((job) => ['queued', 'running'].includes(job.status));
  pending.forEach(trackJob);
  if (pending.length) ui.setStatus(`已恢复 ${pending.length} 个进行中任务`, 'busy');
  else ui.setProgress({ visible: false });
  ui.el('historyLimit').textContent = String(MAX_HISTORY);
}

/* --------------------------------------------------------- command palette */

/**
 * Ctrl+K palette.  Commands are defined here (main.js owns app wiring) while
 * js/palette.js owns the overlay, filtering and keyboard handling.
 * See docs/V6_PLAN.md §2.
 */
function bindPalette() {
  const currentJob = () => viewList()[viewIndex]?.job || store.doneJobs()[0] || null;
  const lastName = () => store.getState().jobs.find((j) => j.status === 'done') || null;
  const scrollTo = (id) => { shell.reveal(id); ui.el(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); };
  const click = (id) => ui.el(id)?.click();

  const list = [
    { id: 'gen', label: '✨ 生成图像', hint: '用当前参数提交任务', keys: 'Ctrl+↵', group: '创作', run: () => generate(false) },
    { id: 'reroll', label: '🎲 换 seed 再生成', hint: '保留参数，随机 seed', keys: 'R', group: '创作', run: () => generate(true) },
    { id: 'batch4', label: '🧩 一次生成 4 张变体', hint: '单任务批量，一次轮询', group: '创作',
      run: () => { ui.el('batch').value = '4'; readSettings(); generate(true); } },
    { id: 'batch2', label: '🧩 一次生成 2 张变体', hint: '单任务批量', group: '创作',
      run: () => { ui.el('batch').value = '2'; readSettings(); generate(true); } },

    { id: 'surprise', label: '✨ 一键灵感（随机组合）', hint: '场景×天气×时间×色调×画质', group: '灵感',
      run: () => studio.surpriseMe(false) },
    { id: 'surprise-keep', label: '♻ 换灵感但保持构图', hint: '只换天气/时间/色调', group: '灵感',
      run: () => studio.surpriseMe(true) },
    { id: 'draft', label: '⚡ 极速草稿（6 步 LCM）', hint: '先确认构图', group: '创作',
      run: () => studio.runAction('draft') },
    { id: 'final', label: '✨ 出片（沿用 seed 与构图）', hint: '标准 24 步', group: '创作',
      run: () => studio.runAction('final') },
    { id: 'master', label: '💎 大师档（精细 32 步 + 2K）', hint: '最高细节', group: '创作',
      run: () => studio.runAction('master') },
    { id: 'random', label: '🌄 随机一句灵感 prompt', hint: '从内置灵感池抽取', group: '灵感',
      run: () => { ui.el('prompt').value = pick(RANDOM_PROMPTS); ui.el('random').checked = false; refreshComposed(); readSettings(); } },
    { id: 'scenes', label: '🗺 浏览场景库', hint: '28 个预制场景', group: '灵感', run: () => scrollTo('sceneSearch') },
    { id: 'mods', label: '🎛 浏览修饰词', hint: '光线 / 天气 / 镜头 / 质感', group: '灵感', run: () => scrollTo('modifiers') },

    { id: 'fav', label: '★ 收藏当前图', hint: '加入收藏列表', group: '资产',
      run: () => { const job = currentJob(); if (!job) { ui.toast('还没有可收藏的图'); return; }
        const on = store.toggleFavorite(job.id); extras.refreshExtras(); ui.toast(on ? '★ 已收藏' : '已取消收藏'); } },
    { id: 'seed', label: '📋 复制当前 seed', hint: '方便复现', keys: 'S', group: '资产', run: () => click('copySeed') },
    { id: 'promptcopy', label: '📋 复制完整 prompt', hint: '含风格与修饰词', group: '资产',
      run: () => navigator.clipboard?.writeText(ui.el('prompt').value).then(() => ui.toast('已复制 prompt'), () => ui.toast('复制失败')) },
    { id: 'dl', label: '⤓ 下载当前图', hint: 'WebP 原图', group: '资产', run: () => click('download') },
    { id: 'full', label: '⤢ 全屏查看', hint: '可滚轮缩放', keys: 'F', group: '资产', run: () => click('fullscreen') },
    { id: 'compare', label: '⇄ 开关对比滑块', hint: '当前图 ⟷ 下一张', keys: 'C', group: '资产', run: () => click('compareBtn') },
    { id: 'lightbox', label: '🔍 打开当前图灯箱', hint: '滚轮缩放 / 拖拽平移', group: '资产',
      run: () => { const item = viewList()[viewIndex]; if (item) openLightboxFor(item.job); } },

    { id: 'gallery', label: '🖼 跳到历史画廊', hint: '筛选 / 检索 / 收藏', group: '面板', run: () => scrollTo('gallery') },
    { id: 'stats', label: '📊 跳到运行统计', hint: '成功率 / 平均耗时 / 缓存命中', group: '面板', run: () => scrollTo('statGrid') },
    { id: 'library', label: '📚 跳到词库与检索', hint: 'Prompt 词库 / 历史搜索', group: '面板', run: () => scrollTo('libList') },
    { id: 'health', label: '❤ 跳到运行状态', hint: '显存 / 队列 / 日志', group: '面板', run: () => scrollTo('healthGrid') },

    { id: 'warm', label: '🔥 预热管线', hint: '消除首图冷启动', group: '维护', run: () => click('warmBtn') },
    { id: 'gc', label: '♻ 触发回收', hint: '清理显存与过期结果', group: '维护', run: () => click('gcBtn') },
    { id: 'theme', label: '🌗 切换明暗主题', group: '维护', run: () => click('themeBtn') },
    { id: 'clear', label: '🧹 清空本地历史', hint: '不影响后端结果', group: '维护', run: () => click('clearHistory') },
    { id: 'help', label: '❔ 快捷键帮助', keys: '?', group: '维护', run: () => click('helpBtn') },
    { id: 'export', label: '🗂 导出 / 导入数据', hint: '历史与配方 JSON', group: '维护',
      run: () => { const job = lastName(); void job; click('dataBtn'); } },
    { id: 'cancel', label: '✖ 取消排队中的任务', hint: '正在 GPU 上跑的任务不会被打断', group: '创作',
      run: async () => {
        const queued = store.getState().jobs.filter((j) => j.status === 'queued');
        if (!queued.length) { ui.toast('当前没有排队中的任务'); return; }
        const target = queued[queued.length - 1];   // oldest queued first
        try {
          await api.cancelJob(target.id);
          watchers.get(target.id)?.();
          watchers.delete(target.id);
          store.updateJob(target.id, { status: 'cancelled', info: '已取消（未占用 GPU）' });
          ui.toast('已取消排队任务');
          renderGallery();
          ui.renderQueue(store.getState().jobs);
        } catch (error) {
          ui.toast(`取消失败：${error.message || error}`);
        }
      } },
    { id: 'reuse', label: '♻ 复用最近一张的参数', hint: 'prompt + seed + 采样器', group: '创作',
      run: () => { const job = lastName(); if (job) reuseJob(job); else ui.toast('还没有历史记录'); } },
  ];

  palette.initPalette(list);
  ui.el('cmdkBtn').onclick = () => palette.openPalette();
  ui.log('palette', { commands: list.length });
}

function boot() {
  ui.bindStage({ prev: () => stepView(-1), next: () => stepView(1), zoom: () => {
    const item = viewList()[viewIndex];
    if (item) openLightboxFor(item.job);
  } });
  ui.bindLightbox({ prev: () => stepView(-1), next: () => stepView(1), remix: remixCurrent });
  bindUpload();
  bindControls();
  bindShortcuts();
  bindPalette();
  const dock = ui.el('dockGenerate');
  if (dock) dock.onclick = () => generate(false);
  document.querySelectorAll('.mobile-tabs a').forEach((a) => {
    a.addEventListener('click', () => {
      document.querySelectorAll('.mobile-tabs a').forEach((x) => x.classList.remove('on'));
      a.classList.add('on');
    });
  });
  extras.initExtras({
    getPrompt: () => ui.el('prompt').value,
    onUsePrompt: (text) => {
      ui.el('prompt').value = text;
      ui.el('random').checked = false;
      refreshComposed();
      readSettings();
    },
    onShowJob: (id) => {
      const index = viewList().findIndex((entry) => entry.id === id);
      if (index >= 0) showByIndex(index);
    },
    onReuseJob: (job) => reuseJob(job),
  });
  studio.initStudio({
    mount: ui.el('studioMount'),
    toast: (m) => ui.toast(m),
    getPrompt: () => ui.el('prompt').value,
    setPrompt: (text) => {
      ui.el('prompt').value = text;
      ui.el('random').checked = false;
      refreshComposed();
      readSettings();
    },
    getSeed: () => +ui.el('seed').value,
    setSeed: (value) => { ui.el('seed').value = value; readSettings(); },
    variation: (delta) => {
      const base = +ui.el('seed').value;
      const seed = (base < 0 ? Math.floor(Math.random() * 1e9) : base) + Math.max(0, delta | 0);
      ui.el('seed').value = seed;
      readSettings();
      ui.toast(`🧬 相似构图 seed=${seed}`);
      generate(false);
    },
    draft: () => {
      ui.el('recipe').value = 'fast';
      ui.el('steps').value = 6;
      ui.el('sampler').value = 'lcm';
      syncRecipe();
      readSettings();
      ui.toast('⚡ 极速草稿（6 步 LCM）');
      generate(false);
    },
    final: () => {
      ui.el('recipe').value = 'std';
      ui.el('steps').value = 24;
      ui.el('sampler').value = 'dpmpp2m_karras';
      syncRecipe();
      readSettings();
      ui.toast('✨ 出片（沿用 seed 与构图）');
      generate(false);
    },
    master: () => {
      ui.el('recipe').value = 'fine';
      ui.el('steps').value = 32;
      ui.el('sampler').value = 'dpmpp2m_sde';
      ui.el('clarity').value = 'ultra';
      syncRecipe();
      readSettings();
      ui.toast('💎 大师档：精细 32 步 + 2K 超清');
      generate(false);
    },
    onChange: () => refreshComposed(),
  });
  ui.el('statsRefresh')?.addEventListener('click', () => extras.refreshExtras());

  // ---- V6 wave 2: scene explorer / prompt composer / A-B lab ----
  scenes.initScenes({
    onPick: (prompt) => {
      ui.el('prompt').value = prompt;
      ui.el('random').checked = false;
      refreshComposed();
      readSettings();
      ui.toast('场景已填入 prompt');
    },
  });
  composer.initComposer({
    textarea: ui.el('prompt'),
    onChange: () => { refreshComposed(); readSettings(); },
  });
  ab.initAB({
    mount: ui.el('abMount'),
    toast: (m, kind) => ui.toast(m, kind),
    getPrompt: () => composedPrompt(),
    getSeed: () => +ui.el('seed').value,
    submit: (text, seed, onSettled) => generate(false, { prompt: text, batch: 2, seed, onSettled }),
    resolve: (job) => ensureUrl(itemsOf(job)[0]),
    compare: (aUrl, bUrl, caption) => ui.showCompare(aUrl, bUrl, caption, { left: 'A', right: 'B' }),
  });

  // ---- v7 shell: desktop panel tabs (persisted), mobile keeps full scroll ----
  shell.initShell();

  // ---- app-shell service worker (static assets only; API never cached) ----
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/sw.js').catch(() => {});
    });
  }

  restore();
  refreshHealth();
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      store.flush();                     // settings/jobs durable before tab sleeps
    } else {
      renderGallery();                   // catch up on renders skipped while hidden
      ui.renderQueue(store.getState().jobs);
      refreshHealth();
    }
  });
  window.addEventListener('beforeunload', () => { store.flush(); store.releaseAllBlobs(); });
  window.addEventListener('pagehide', () => store.flush());
}

boot();
