/**
 * Prompt Composer — token strip + 11 structured phrase groups (V6).
 *
 * The textarea is never replaced: it stays the source of truth so the backend
 * protocol is untouched.  This module only
 *   1. renders the current prompt as removable tokens (split on "," / "，"),
 *   2. toggles structured phrases (subject/env/weather/…/style) into the text.
 *
 *   initComposer({ textarea, onChange })   bind once
 *   refresh()                              re-render tokens (called on input)
 */
const GROUPS = [
  ['主体', ['a lone cabin', 'a winding river', 'snow-capped peaks', 'a lighthouse', 'an old stone bridge', 'a sailboat on still water']],
  ['环境', ['in a pine forest', 'above a sea of clouds', 'by a rocky coastline', 'in a mountain valley', 'in a desert', 'in a bamboo grove']],
  ['天气', ['light mist', 'clear blue sky', 'heavy dramatic clouds', 'falling snow', 'after rain', 'thunderstorm']],
  ['光线', ['golden hour light', 'blue hour, cool twilight', 'soft dawn light', 'moonlit night', 'god rays, volumetric light', 'backlit, rim light']],
  ['时间', ['at dawn', 'at sunrise', 'at sunset', 'at night', 'under the milky way', 'at midday']],
  ['季节', ['in spring', 'in summer', 'in autumn', 'in winter', 'cherry blossom season', 'autumn foliage']],
  ['镜头', ['wide angle, 16mm', 'telephoto compression, 200mm', 'aerial drone view', 'low angle view', 'eye level view', 'long exposure']],
  ['景深', ['deep depth of field, everything in focus', 'shallow depth of field, soft bokeh']],
  ['构图', ['rule of thirds', 'leading lines', 'centered symmetry', 'natural framing', 'strong foreground depth']],
  ['色调', ['warm amber tones', 'cool blue tones', 'muted natural palette', 'vivid saturated colors', 'monochrome high contrast']],
  ['风格', ['photorealistic', 'cinematic color grading', 'oil painting', 'watercolor', 'ink wash', 'cyberpunk']],
];

let textarea = null;
let onChange = null;

function tokensOf(value) {
  const seen = new Set();
  const out = [];
  value.split(/[,，]/).map((s) => s.trim()).filter(Boolean).forEach((t) => {
    const key = t.toLowerCase();
    if (!seen.has(key)) { seen.add(key); out.push(t); }
  });
  return out;
}

function join(tokens) { return tokens.join(', '); }

function write(value) {
  textarea.value = value;
  onChange?.();
  refresh();
}

function togglePhrase(phrase) {
  const tokens = tokensOf(textarea.value);
  const index = tokens.findIndex((t) => t.toLowerCase() === phrase.toLowerCase());
  if (index >= 0) tokens.splice(index, 1);
  else tokens.push(phrase);
  write(join(tokens));
  textarea.focus();
}

function removeToken(phrase) {
  const tokens = tokensOf(textarea.value).filter((t) => t !== phrase);
  write(join(tokens));
  textarea.focus();
}

/* ------------------------------------------------------------------ render */

export function refresh() {
  const host = document.getElementById('composerTokens');
  const grid = document.getElementById('composerGrid');
  if (!host) return;
  if (!textarea) { textarea = document.getElementById('prompt'); if (!textarea) return; }

  const tokens = tokensOf(textarea.value);
  const MAX = 10;
  host.innerHTML = tokens.slice(0, MAX).map((t) => `
    <span class="composer-token"><span class="tok-text">${t.replace(/[<>&"]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[c]))}</span><i class="tok-x" title="移除该词条">×</i></span>
  `).join('') + (tokens.length > MAX ? `<span class="composer-token more">+${tokens.length - MAX} 项</span>` : '');

  host.querySelectorAll('.composer-token .tok-x').forEach((x, i) => {
    x.onclick = () => removeToken(tokens[i]);
  });

  if (grid) {
    grid.querySelectorAll('.chip').forEach((chip) => {
      chip.classList.toggle('on', tokens.some((t) => t.toLowerCase() === chip.dataset.phrase.toLowerCase()));
    });
  }
}

function renderGroups() {
  const grid = document.getElementById('composerGrid');
  if (!grid) return;
  grid.innerHTML = GROUPS.map(([name, phrases]) => `
    <div class="composer-group">
      <b>${name}</b>
      <div class="chip-row">${phrases.map((p) => `<button type="button" class="chip" data-phrase="${p.replace(/"/g, '&quot;')}">${p}</button>`).join('')}</div>
    </div>`).join('');
  grid.querySelectorAll('.chip').forEach((chip) => {
    chip.onclick = () => togglePhrase(chip.dataset.phrase);
  });
}

export function initComposer(opts) {
  textarea = opts.textarea;
  onChange = opts.onChange;
  renderGroups();
  refresh();
  return { refresh };
}
