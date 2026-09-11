/**
 * Command Palette (Ctrl / Cmd + K).
 *
 * Self-contained component: it builds and owns its overlay DOM, so index.html
 * needs no markup and no ids.  Commands are injected by main.js, which keeps the
 * palette free of app knowledge (see docs/V6_PLAN.md §2 interaction contracts).
 *
 *   initPalette([{ id, label, hint, group, keys, keywords, run }])
 */

let overlay = null;
let input = null;
let listEl = null;
let commands = [];
let filtered = [];
let cursor = 0;
let lastFocus = null;

const MAX_ROWS = 9;

function ensureDom() {
  if (overlay) return;
  overlay = document.createElement('div');
  overlay.className = 'cmdk';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-label', '命令面板');
  overlay.innerHTML = `
    <div class="cmdk-box" role="document">
      <div class="cmdk-head">
        <span class="cmdk-glyph" aria-hidden="true">⌘</span>
        <input class="cmdk-input" type="text" autocomplete="off" spellcheck="false"
               placeholder="想做什么？输入命令或关键词…" aria-label="命令输入">
        <kbd class="cmdk-esc">Esc</kbd>
      </div>
      <div class="cmdk-list" role="listbox" aria-label="命令列表"></div>
      <div class="cmdk-foot">
        <span><kbd>↑</kbd><kbd>↓</kbd> 选择</span>
        <span><kbd>Enter</kbd> 执行</span>
        <span><kbd>Ctrl</kbd>+<kbd>K</kbd> 开关</span>
        <span class="cmdk-count" aria-live="polite"></span>
      </div>
    </div>`;
  document.body.appendChild(overlay);

  input = overlay.querySelector('.cmdk-input');
  listEl = overlay.querySelector('.cmdk-list');

  input.addEventListener('input', () => { cursor = 0; render(); });
  input.addEventListener('keydown', onKey);
  overlay.addEventListener('mousedown', (event) => {
    if (event.target === overlay) closePalette();
  });

  document.addEventListener('keydown', (event) => {
    const k = event.key.toLowerCase();
    if ((event.ctrlKey || event.metaKey) && k === 'k') {
      event.preventDefault();
      paletteOpen() ? closePalette() : openPalette();
    } else if (event.key === 'Escape' && paletteOpen()) {
      event.preventDefault();
      closePalette();
    }
  }, true);
}

function score(command, needle) {
  if (!needle) return 1;
  const hay = `${command.label} ${command.hint || ''} ${command.group || ''} ${(command.keywords || []).join(' ')}`.toLowerCase();
  const at = hay.indexOf(needle);
  if (at >= 0) return 100 - at;
  // subsequence match: "gen" → "Generate image"
  let i = 0;
  for (const ch of hay) if (ch === needle[i]) i += 1;
  return i === needle.length ? 10 : 0;
}

function render() {
  const needle = input.value.trim().toLowerCase();
  filtered = commands
    .map((command) => ({ command, s: score(command, needle) }))
    .filter((row) => row.s > 0)
    .sort((a, b) => b.s - a.s)
    .slice(0, MAX_ROWS)
    .map((row) => row.command);

  if (!filtered.length) {
    listEl.innerHTML = '<p class="cmdk-empty">没有匹配的命令</p>';
  } else {
    listEl.innerHTML = filtered.map((command, index) => `
      <button type="button" class="cmdk-row${index === cursor ? ' active' : ''}"
              role="option" aria-selected="${index === cursor}" data-index="${index}">
        <span class="cmdk-label">${command.label}</span>
        ${command.hint ? `<span class="cmdk-hint">${command.hint}</span>` : ''}
        ${command.keys ? `<span class="cmdk-keys">${command.keys}</span>` : ''}
      </button>`).join('');
    listEl.querySelectorAll('.cmdk-row').forEach((row) => {
      row.addEventListener('mouseenter', () => { cursor = +row.dataset.index; paint(); });
      row.addEventListener('click', () => run(+row.dataset.index));
    });
  }
  const count = overlay.querySelector('.cmdk-count');
  if (count) count.textContent = `${filtered.length} / ${commands.length}`;
}

function paint() {
  listEl.querySelectorAll('.cmdk-row').forEach((row) => {
    const on = +row.dataset.index === cursor;
    row.classList.toggle('active', on);
    row.setAttribute('aria-selected', String(on));
  });
}

function onKey(event) {
  if (event.key === 'ArrowDown') {
    event.preventDefault();
    cursor = (cursor + 1) % Math.max(1, filtered.length);
    paint();
  } else if (event.key === 'ArrowUp') {
    event.preventDefault();
    cursor = (cursor - 1 + Math.max(1, filtered.length)) % Math.max(1, filtered.length);
    paint();
  } else if (event.key === 'Enter') {
    event.preventDefault();
    run(cursor);
  }
}

function run(index) {
  const command = filtered[index];
  if (!command) return;
  closePalette();
  try {
    command.run();
  } catch (error) {
    // never let a command break the palette
    console.warn('palette command failed', command.id, error);
  }
}

export function initPalette(list) {
  commands = (list || []).filter((c) => c && c.label && typeof c.run === 'function');
  ensureDom();
  return commands.length;
}

export function openPalette(preset = '') {
  ensureDom();
  lastFocus = document.activeElement;
  overlay.classList.add('show');
  overlay.setAttribute('aria-hidden', 'false');
  input.value = preset;
  cursor = 0;
  render();
  input.focus();
  input.select();
}

export function closePalette() {
  if (!overlay) return;
  overlay.classList.remove('show');
  overlay.setAttribute('aria-hidden', 'true');
  if (lastFocus && lastFocus.focus) lastFocus.focus();
}

export function paletteOpen() {
  return !!overlay && overlay.classList.contains('show');
}
