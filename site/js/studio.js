/**
 * Prompt Studio — 摄影师模式 / 光线实验室 / 一键灵感 / Seed Lab / Draft→Final.
 *
 * Owns its own DOM inside a mount point and keeps structured "fragments" that
 * main.js appends to the prompt it sends.  The textarea is never rewritten by
 * this module, so there is no base-prompt feedback loop: the user's own words
 * stay theirs, and the studio only contributes the extra English fragments it
 * shows in its preview line.
 *
 * See docs/V6_PLAN.md §2 (component contracts).
 */
import * as store from './store.js';
import { SCENE_GROUPS, RANDOM_PROMPTS } from './config.js';

const FOCAL = [
  ['16mm', 'ultra wide angle, 16mm'], ['24mm', 'wide angle, 24mm'],
  ['35mm', 'natural perspective, 35mm'], ['50mm', 'standard lens, 50mm'],
  ['85mm', 'short telephoto, 85mm'], ['200mm', 'telephoto compression, 200mm'],
];
const ANGLE = [
  ['低机位', 'low angle view'], ['平视', 'eye level view'],
  ['高机位', 'high angle view'], ['航拍', 'aerial drone view'],
];
const COMPOSITION = [
  ['广角全景', 'wide establishing shot'], ['居中对称', 'centered symmetrical composition'],
  ['引导线', 'leading lines'], ['三分法', 'rule of thirds'],
  ['框架构图', 'natural framing'], ['前景纵深', 'strong foreground depth'],
];
const LIGHTING = [
  ['黄金时刻', 'golden hour light', '🌅'], ['日出', 'sunrise light, warm glow', '☀️'],
  ['日落', 'sunset light, vivid colors', '🌇'], ['蓝调时刻', 'blue hour, cool twilight', '🌌'],
  ['月光', 'moonlit night, soft silver light', '🌙'], ['阴天', 'overcast soft light', '☁️'],
  ['雨后', 'after rain, wet surfaces, fresh', '🌧️'], ['雷暴', 'thunderstorm, dramatic lightning', '⚡'],
  ['丁达尔光', 'volumetric god rays', '✨'],
];
const WEATHER = ['light mist', 'clear sky', 'heavy dramatic clouds', 'low fog', 'falling snow', 'storm clouds'];
const TIME = ['at dawn', 'at sunrise', 'at golden hour', 'at sunset', 'at blue hour', 'under the milky way'];
const COLOR = ['warm amber tones', 'cool blue tones', 'muted natural palette', 'vivid saturated colors',
  'pastel soft palette', 'monochrome high contrast'];
const QUALITY = ['professional landscape photography', 'ultra detailed, sharp focus', 'cinematic color grading'];

let mount = null;
let handlers = {};
let state = { focal: null, angle: null, comp: [], light: null, dof: 60, lock: false, variation: 0 };

function save() {
  store.getState().settings.studio = { ...state };
  store.save();
}

function load() {
  const s = store.getState().settings.studio;
  if (s && typeof s === 'object') state = { ...state, ...s, comp: Array.isArray(s.comp) ? s.comp : [] };
}

/* ------------------------------------------------------------ fragments */

/** English fragments contributed by the studio, in a stable order. */
export function fragmentsText() {
  const bits = [];
  const focal = FOCAL.find(([k]) => k === state.focal);
  if (focal) bits.push(focal[1]);
  if (state.angle) { const a = ANGLE.find(([k]) => k === state.angle); if (a) bits.push(a[1]); }
  // depth of field: 0 = deep (everything sharp) … 100 = shallow (bokeh)
  if (state.dof <= 25) bits.push('deep depth of field, everything in focus');
  else if (state.dof >= 75) bits.push('shallow depth of field, soft bokeh background');
  state.comp.forEach((k) => { const c = COMPOSITION.find(([n]) => n === k); if (c) bits.push(c[1]); });
  if (state.light) { const l = LIGHTING.find(([n]) => n === state.light); if (l) bits.push(l[1]); }
  return bits.join(', ');
}

export function fragmentChips() {
  const out = [];
  if (state.focal) out.push(state.focal);
  if (state.angle) out.push(state.angle);
  state.comp.forEach((c) => out.push(c));
  if (state.light) out.push(state.light);
  if (state.dof <= 25) out.push('深景深');
  else if (state.dof >= 75) out.push('浅景深');
  return out;
}

/* ---------------------------------------------------------------- render */

function chips(names, activeList, onPick) {
  return names
    .map((n) => `<button type="button" class="chip${activeList.includes(n) ? ' on' : ''}" data-v="${n}">${n}</button>`)
    .join('');
}

function render() {
  if (!mount) return;
  const chipsNow = fragmentChips();
  mount.innerHTML = `
    <section class="card">
      <p class="eyebrow">Camera</p>
      <h2>📷 摄影师模式</h2>
      <p class="muted">点选后自动追加英文提示词片段，不改动你自己写的 prompt。</p>

      <h3>焦段</h3>
      <div class="chip-row" data-group="focal">${chips(FOCAL.map(([k]) => k), [state.focal].filter(Boolean))}</div>

      <h3>视角</h3>
      <div class="chip-row" data-group="angle">${chips(ANGLE.map(([k]) => k), [state.angle].filter(Boolean))}</div>

      <h3>景深 <span class="muted" data-dof-label>${state.dof <= 25 ? '深（全清晰）' : state.dof >= 75 ? '浅（虚化）' : '中'}</span></h3>
      <input type="range" min="0" max="100" step="5" value="${state.dof}" data-role="dof">

      <h3>构图 <span class="muted">可多选</span></h3>
      <div class="chip-row" data-group="comp">${chips(COMPOSITION.map(([k]) => k), state.comp)}</div>
    </section>

    <section class="card">
      <p class="eyebrow">Lighting Lab</p>
      <h2>💡 光线实验室</h2>
      <p class="muted">风景图最吃光。单选一种主光，立刻看到 prompt 变化。</p>
      <div class="chip-row" data-group="light">
        ${LIGHTING.map(([n, , icon]) => `<button type="button" class="chip${state.light === n ? ' on' : ''}" data-v="${n}">${icon} ${n}</button>`).join('')}
        <button type="button" class="chip${state.light ? '' : ' on'}" data-v="">默认</button>
      </div>
    </section>

    <section class="card">
      <p class="eyebrow">Inspiration</p>
      <h2>✨ 一键灵感</h2>
      <div class="btn-row tight">
        <button type="button" class="btn big" data-role="surprise">🎲 Surprise Me</button>
        <button type="button" class="btn ghost" data-role="surprise-keep">♻ 换一个，但保持构图</button>
      </div>
      <p class="muted">随机组合 场景 × 天气 × 时间 × 镜头 × 色调 × 画质，直接写进 Prompt 框。</p>
    </section>

    <section class="card">
      <p class="eyebrow">Seed Lab</p>
      <h2>🎲 Seed Lab</h2>
      <div class="row">
        <div class="field"><label>当前 seed</label>
          <input type="number" data-role="seed" value="${handlers.getSeed ? handlers.getSeed() : -1}"></div>
        <div class="field align-end">
          <label class="toggle small"><input type="checkbox" data-role="lock" ${state.lock ? 'checked' : ''}> 锁定 seed</label>
        </div>
      </div>
      <div class="row">
        <div class="field"><label>变异幅度 <b data-var-label>+${state.variation}</b></label>
          <input type="range" min="0" max="5" step="1" value="${state.variation}" data-role="variation"></div>
      </div>
      <div class="btn-row tight">
        <button type="button" class="btn ghost tiny" data-role="seed-rand">🎲 随机 seed</button>
        <button type="button" class="btn ghost tiny" data-role="seed-apply">↻ 写回表单</button>
        <button type="button" class="btn ghost tiny" data-role="seed-var">🧬 生成相似构图（seed+${state.variation}）</button>
      </div>
    </section>

    <section class="card">
      <p class="eyebrow">Draft → Final</p>
      <h2>⚡ 两段式出片</h2>
      <p class="muted">先用 6 步草稿确认构图，再用标准/精细档沿用同一 seed 出片 —— 比盲调 500 步快约 33×。</p>
      <div class="btn-row tight">
        <button type="button" class="btn ghost" data-role="draft">⚡ 极速草稿（6 步）</button>
        <button type="button" class="btn" data-role="final">✨ 出片（沿用 seed）</button>
        <button type="button" class="btn ghost" data-role="master">💎 大师（精细 + 2K）</button>
      </div>
    </section>

    <div class="studio-preview" ${chipsNow.length ? '' : 'hidden'}>
      <b>将追加到 prompt：</b><span>${chipsNow.join(' · ')}</span>
      <button type="button" class="btn ghost tiny" data-role="clear">清空修饰</button>
    </div>`;

  bind();
}

function bind() {
  mount.querySelectorAll('[data-group]').forEach((group) => {
    const key = group.dataset.group;
    group.querySelectorAll('.chip').forEach((btn) => {
      btn.onclick = () => {
        const v = btn.dataset.v;
        if (key === 'comp') {
          state.comp = state.comp.includes(v) ? state.comp.filter((x) => x !== v) : [...state.comp, v];
        } else if (key === 'light') {
          state.light = v || null;
        } else {
          state[key] = state[key] === v ? null : v;   // clicking again clears
        }
        save(); render();
        handlers.onChange?.();
      };
    });
  });

  const dof = mount.querySelector('[data-role="dof"]');
  if (dof) dof.oninput = () => {
    state.dof = +dof.value;
    const label = mount.querySelector('[data-dof-label]');
    if (label) label.textContent = state.dof <= 25 ? '深（全清晰）' : state.dof >= 75 ? '浅（虚化）' : '中';
    save();
  };
  if (dof) dof.onchange = () => { save(); render(); handlers.onChange?.(); };

  const seedInput = mount.querySelector('[data-role="seed"]');
  if (seedInput) seedInput.onchange = () => handlers.setSeed?.(+seedInput.value);

  const lock = mount.querySelector('[data-role="lock"]');
  if (lock) lock.onchange = () => { state.lock = lock.checked; save(); handlers.toast?.(state.lock ? '🔒 已锁定 seed' : '已解锁 seed'); };

  const varSlider = mount.querySelector('[data-role="variation"]');
  if (varSlider) varSlider.oninput = () => {
    state.variation = +varSlider.value;
    const label = mount.querySelector('[data-var-label]');
    if (label) label.textContent = `+${state.variation}`;
    const btn = mount.querySelector('[data-role="seed-var"]');
    if (btn) btn.textContent = `🧬 生成相似构图（seed+${state.variation}）`;
    save();
  };

  const on = (role, fn) => { const el = mount.querySelector(`[data-role="${role}"]`); if (el) el.onclick = fn; };
  on('surprise', () => surprise(false));
  on('surprise-keep', () => surprise(true));
  on('seed-rand', () => { const s = Math.floor(Math.random() * 1e9); if (seedInput) seedInput.value = s; handlers.setSeed?.(s); handlers.toast?.(`🎲 seed ${s}`); });
  on('seed-apply', () => handlers.setSeed?.(+(seedInput?.value ?? -1)));
  on('seed-var', () => handlers.variation?.(state.variation));
  on('draft', () => handlers.draft?.());
  on('final', () => handlers.final?.());
  on('master', () => handlers.master?.());
  on('clear', () => {
    state.focal = null; state.angle = null; state.comp = []; state.light = null;
    save(); render(); handlers.onChange?.();
  });
}

/* ------------------------------------------------------------- surprise */

function pickOne(list) { return list[Math.floor(Math.random() * list.length)]; }

function surprise(keepComposition) {
  const scenes = SCENE_GROUPS.flatMap(([, items]) => Object.values(items));
  const scene = keepComposition && handlers.getPrompt
    ? (handlers.getPrompt() || pickOne(scenes))
    : (Math.random() < 0.35 ? pickOne(RANDOM_PROMPTS) : pickOne(scenes));
  const text = [
    scene, pickOne(WEATHER), pickOne(TIME), pickOne(COLOR), pickOne(QUALITY),
  ].join(', ');
  handlers.setPrompt?.(text);
  if (!keepComposition) { state.light = null; state.focal = null; save(); }
  render();
  handlers.onChange?.();
  handlers.toast?.(keepComposition ? '♻ 已换灵感，构图相关修饰保留' : '✨ 已生成一条新灵感');
}

/* ------------------------------------------------------------------ api */

export function initStudio(opts) {
  mount = opts.mount;
  handlers = opts;
  load();
  render();
  return { fragmentsText, fragmentChips };
}

export function refreshStudio() { render(); }

/** Run a studio action from outside (command palette, shortcuts, …). */
export function runAction(name, arg) {
  const fn = handlers[name];
  if (typeof fn === 'function') fn(arg);
}

/** Random inspiration combo; `keep` preserves the current scene/composition. */
export function surpriseMe(keep = false) { surprise(keep); }
