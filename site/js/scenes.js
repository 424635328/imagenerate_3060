/**
 * Scene Explorer — the 28-scene library as big art cards (V6).
 *
 * Replaces the old text-chip list in ui.js.  Same data (config.SCENE_GROUPS),
 * same contract: picking a card hands the scene's English phrase to main.js
 * via `onPick`, which is exactly what the old chips did.
 *
 *   initScenes({ onPick })   render once, bind once
 *   filterScenes(query)      live search (bound to #sceneSearch in main.js)
 *   renderScenes(onPick)     full re-render (used by init + refresh)
 */
import { SCENE_GROUPS } from './config.js';
import { SCENE_ART } from './scene-art.js';
import * as store from './store.js';

const ART = {
  '自然奇观': { icon: '🏔️', a: '#7c5cff', b: '#22d3ee' },
  '海岸与水': { icon: '🌊', a: '#0ea5e9', b: '#34d399' },
  '四季': { icon: '🌸', a: '#f472b6', b: '#fbbf24' },
  '光效与天象': { icon: '🌌', a: '#8b5cf6', b: '#f59e0b' },
};

let currentGroup = '全部';
let currentQuery = '';
let onPick = null;

function favorites() {
  const favs = store.getState().settings.sceneFavs;
  return Array.isArray(favs) ? favs : [];
}

function saveFavs(favs) {
  store.getState().settings.sceneFavs = favs;
  store.save();
}

function toggleFav(name) {
  const favs = favorites();
  saveFavs(favs.includes(name) ? favs.filter((n) => n !== name) : [...favs, name]);
}

/* ------------------------------------------------------------------ render */

export function renderScenes(handler) {
  if (handler) onPick = handler;
  const host = document.getElementById('scenes');
  const count = document.getElementById('sceneCount');
  if (!host) return;
  const total = SCENE_GROUPS.reduce((n, [, items]) => n + Object.keys(items).length, 0);
  if (count) count.textContent = `${total} 个场景`;

  const favs = favorites();
  const cards = [];
  SCENE_GROUPS.forEach(([group, items]) => {
    const art = ART[group] || { icon: '🖼️', a: '#7c5cff', b: '#22d3ee' };
    Object.entries(items).forEach(([name, prompt]) => {
      cards.push({ group, name, prompt, art });
    });
  });

  host.innerHTML = `
    <div class="scene-filters">
      <button type="button" class="chip${currentGroup === '全部' ? ' on' : ''}" data-group="全部">全部</button>
      ${SCENE_GROUPS.map(([g]) => `<button type="button" class="chip${currentGroup === g ? ' on' : ''}" data-group="${g}">${g}</button>`).join('')}
      <button type="button" class="chip${currentGroup === '⭐' ? ' on' : ''}" data-group="⭐">⭐ 收藏 ${favs.length ? `· ${favs.length}` : ''}</button>
    </div>
    <div class="scene-grid"></div>
    <button type="button" class="btn ghost scene-random" id="sceneRandom">🎲 随机场景</button>`;

  const grid = host.querySelector('.scene-grid');
  grid.innerHTML = cards
    .filter((c) => (currentGroup === '⭐' ? favs.includes(c.name) : currentGroup === '全部' || c.group === currentGroup))
    .map((c) => {
      const art = SCENE_ART[c.name];
      return `
      <button type="button" class="scene-card" data-name="${c.name}"
              style="--a:${c.art.a};--b:${c.art.b}" title="点击填入 prompt：${c.name}">
        <span class="scene-art">${art ? `<img class="scene-img" loading="lazy" decoding="async" src="${art}" alt="" onerror="this.remove()">` : ''}<span class="scene-glyph">${c.art.icon}</span></span>
        <span class="scene-name">${c.name}<i class="star${favs.includes(c.name) ? ' on' : ''}" data-fav="${c.name}" title="收藏场景">★</i></span>
        <span class="scene-en">${c.prompt}</span>
      </button>`;
    }).join('')
    || `<p class="scene-empty">没有匹配的场景。</p>`;

  grid.querySelectorAll('.scene-card').forEach((card) => {
    card.onclick = () => {
      const name = card.dataset.name;
      const scene = cards.find((c) => c.name === name);
      if (scene && onPick) onPick(scene.prompt);
    };
    const star = card.querySelector('[data-fav]');
    if (star) star.onclick = (event) => {
      event.stopPropagation();
      toggleFav(star.dataset.fav);
      renderScenes(onPick);
      filterScenes(currentQuery);
    };
  });

  host.querySelectorAll('[data-group]').forEach((chip) => {
    chip.onclick = () => { currentGroup = chip.dataset.group; renderScenes(onPick); filterScenes(currentQuery); };
  });
  const random = host.querySelector('#sceneRandom');
  if (random) random.onclick = () => {
    const pool = cards.filter((c) => currentGroup === '⭐' ? favs.includes(c.name) : currentGroup === '全部' || c.group === currentGroup);
    if (pool.length && onPick) onPick(pool[Math.floor(Math.random() * pool.length)].prompt);
  };
  filterScenes(currentQuery);
}

export function filterScenes(query) {
  currentQuery = (query || '').trim().toLowerCase();
  const grid = document.querySelector('#scenes .scene-grid');
  if (!grid) return;
  grid.querySelectorAll('.scene-card').forEach((card) => {
    const hay = `${card.dataset.name} ${card.querySelector('.scene-en')?.textContent || ''}`.toLowerCase();
    card.hidden = currentQuery !== '' && !hay.includes(currentQuery);
  });
}

export function initScenes(opts) {
  renderScenes(opts?.onPick);
  return { renderScenes, filterScenes };
}
