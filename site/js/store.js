/**
 * Client state: settings, job history, and blob-URL lifetime management.
 *
 * Browser-side GC matters here: every chunk-assembled image creates an object
 * URL that survives until it is revoked.  `rememberBlob` keeps a bounded LRU and
 * revokes evicted URLs so long sessions do not grow without bound.
 */
import { STORE_KEY, MAX_HISTORY, MAX_BLOB_CACHE } from './config.js';

const state = {
  settings: {},
  jobs: [],
  last: null,
  theme: 'dark',
  userRecipes: [],
  favorites: [],
  prompts: [],
};

const blobOrder = [];

/* Write amplification guard: typing and poll updates fire save() dozens of
 * times per second; JSON.stringify of 60 jobs each time would jank the page.
 * save() coalesces into one flush, and flush() forces an immediate write on
 * unload / hide so nothing is ever lost. */
let saveTimer = null;
let dirty = false;

export function getState() {
  return state;
}

export function save() {
  dirty = true;
  if (saveTimer) return;
  saveTimer = setTimeout(() => { saveTimer = null; flush(); }, 300);
}

export function flush() {
  if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
  if (!dirty) return;                 // nothing pending: skip the write entirely
  dirty = false;
  writeNow();
}

function writeNow() {
  try {
    const slim = {
      settings: state.settings,
      theme: state.theme,
      userRecipes: state.userRecipes.slice(0, 24),
      favorites: state.favorites.slice(0, 200),
      prompts: state.prompts.slice(0, 60),
      last: state.last,
      jobs: state.jobs.slice(0, MAX_HISTORY).map(stripJob),
    };
    localStorage.setItem(STORE_KEY, JSON.stringify(slim));
  } catch (error) {
    // Quota exceeded: shed the oldest half of the history and retry once.
    state.jobs = state.jobs.slice(0, Math.floor(state.jobs.length / 2));
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ settings: state.settings, jobs: state.jobs.map(stripJob) }));
    } catch (ignored) {
      /* storage unavailable (private mode); run in memory only */
    }
  }
}

/** Runtime-only fields (blob URLs, timers) must never reach localStorage. */
function stripJob(job) {
  const { url, previewUrl, timer, ...rest } = job;
  // Per-image blob URLs live on the flattened view list, not on the record.
  if (Array.isArray(rest.images)) rest.images = rest.images.map(({ i, bytes }) => ({ i, bytes }));
  return rest;
}

export function load() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
    state.settings = raw.settings || {};
    state.jobs = Array.isArray(raw.jobs) ? raw.jobs : [];
    state.last = raw.last || null;
    state.theme = raw.theme === 'light' ? 'light' : 'dark';
    state.userRecipes = Array.isArray(raw.userRecipes) ? raw.userRecipes : [];
    state.favorites = Array.isArray(raw.favorites) ? raw.favorites.filter((x) => typeof x === 'string') : [];
    state.prompts = Array.isArray(raw.prompts) ? raw.prompts.filter((p) => p && p.text) : [];
    state.jobs.forEach((job) => { delete job.url; delete job.previewUrl; });
  } catch (error) {
    state.jobs = [];
  }
  return state;
}

export function addJob(job) {
  state.jobs = [job, ...state.jobs.filter((x) => x.id !== job.id)];
  if (state.jobs.length > MAX_HISTORY) {
    state.jobs.slice(MAX_HISTORY).forEach(releaseBlob);
    state.jobs = state.jobs.slice(0, MAX_HISTORY);
  }
  save();
  return job;
}

export function updateJob(id, patch) {
  const job = state.jobs.find((x) => x.id === id);
  if (!job) return null;
  Object.assign(job, patch);
  save();
  return job;
}

export function removeJob(id) {
  const job = state.jobs.find((x) => x.id === id);
  if (job) releaseBlob(job);
  state.jobs = state.jobs.filter((x) => x.id !== id);
  if (state.last === id) state.last = state.jobs.find((x) => x.status === 'done')?.id || null;
  save();
}

export function clearJobs() {
  state.jobs.forEach(releaseBlob);
  state.jobs = [];
  state.last = null;
  save();
}

export function doneJobs() {
  return state.jobs.filter((job) => job.status === 'done');
}

export function findJob(id) {
  return state.jobs.find((job) => job.id === id) || null;
}

/** Track a blob URL for a job and revoke the least recently used ones. */
export function rememberBlob(job, url) {
  job.url = url;
  blobOrder.push(job.id);
  while (blobOrder.length > MAX_BLOB_CACHE) {
    const evicted = blobOrder.shift();
    if (evicted === job.id) continue;
    const target = findJob(evicted);
    if (target) releaseBlob(target);
  }
  return url;
}

export function releaseBlob(job) {
  if (job?.url?.startsWith('blob:')) {
    URL.revokeObjectURL(job.url);
    job.url = null;
  }
}

export function releaseAllBlobs() {
  state.jobs.forEach(releaseBlob);
  blobOrder.length = 0;
}

export function setTheme(theme) {
  state.theme = theme === 'light' ? 'light' : 'dark';
  save();
  return state.theme;
}

export function saveUserRecipe(recipe) {
  state.userRecipes = [recipe, ...state.userRecipes.filter((r) => r.name !== recipe.name)].slice(0, 24);
  save();
}

export function deleteUserRecipe(name) {
  state.userRecipes = state.userRecipes.filter((r) => r.name !== name);
  save();
}

/* ------------------------------------------------- favourites & prompt library */

export function isFavorite(id) {
  return state.favorites.includes(id);
}

export function toggleFavorite(id) {
  state.favorites = isFavorite(id)
    ? state.favorites.filter((x) => x !== id)
    : [id, ...state.favorites].slice(0, 200);
  save();
  return isFavorite(id);
}

export function favoriteJobs() {
  const wanted = new Set(state.favorites);
  return state.jobs.filter((job) => wanted.has(job.id));
}

export function addPrompt(text) {
  const value = String(text || '').trim();
  if (!value) return state.prompts;
  state.prompts = [{ text: value, ts: Date.now() },
    ...state.prompts.filter((p) => p.text !== value)].slice(0, 60);
  save();
  return state.prompts;
}

export function removePrompt(ts) {
  state.prompts = state.prompts.filter((p) => p.ts !== ts);
  save();
  return state.prompts;
}

export function stats() {
  const jobs = state.jobs;
  const done = jobs.filter((j) => j.status === 'done');
  const failed = jobs.filter((j) => j.status === 'failed');
  const images = done.reduce((sum, j) => sum + (j.count || 1), 0);
  const cached = done.filter((j) => j.cached).length;
  const times = done.map((j) => Number(j.seconds) || 0).filter((s) => s > 0);
  const avg = times.length ? times.reduce((a, b) => a + b, 0) / times.length : 0;
  const bySampler = {};
  done.forEach((j) => {
    const key = j.sampler || 'unknown';
    bySampler[key] = (bySampler[key] || 0) + 1;
  });
  return {
    jobs: jobs.length, done: done.length, failed: failed.length, images, cached, avg, bySampler,
    success: jobs.length ? done.length / jobs.length : 0,
    favorites: state.favorites.length, prompts: state.prompts.length,
  };
}

export function exportPayload() {
  return JSON.stringify({
    version: 5,
    exported: new Date().toISOString(),
    settings: state.settings,
    userRecipes: state.userRecipes,
    favorites: state.favorites,
    prompts: state.prompts,
    jobs: state.jobs.map(stripJob),
  }, null, 2);
}

export function importPayload(text) {
  const data = JSON.parse(text);
  if (data.settings) state.settings = { ...state.settings, ...data.settings };
  if (Array.isArray(data.userRecipes)) state.userRecipes = data.userRecipes.slice(0, 24);
  if (Array.isArray(data.favorites)) {
    state.favorites = [...new Set([...state.favorites, ...data.favorites.filter((x) => typeof x === 'string')])].slice(0, 200);
  }
  if (Array.isArray(data.prompts)) {
    const seen = new Set(state.prompts.map((p) => p.text));
    state.prompts = [...state.prompts,
      ...data.prompts.filter((p) => p && p.text && !seen.has(p.text))].slice(0, 60);
  }
  if (Array.isArray(data.jobs)) {
    const seen = new Set(state.jobs.map((j) => j.id));
    const merged = data.jobs.filter((j) => j && j.id && !seen.has(j.id));
    state.jobs = [...state.jobs, ...merged].slice(0, MAX_HISTORY);
  }
  save();
  return { jobs: state.jobs.length, recipes: state.userRecipes.length,
    favorites: state.favorites.length, prompts: state.prompts.length };
}
