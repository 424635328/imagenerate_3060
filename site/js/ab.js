/**
 * A/B Prompt — two variants × two images, auto compare (V6).
 *
 * Submits A and B through the main.js `submit` handler (same seed baseline so
 * the split is prompt-only), then asks main.js to open the split-view compare
 * between the first image of each job.  A/B texts persist in localStorage.
 *
 *   initAB({ mount, submit, getPrompt, compare, toast })
 *     submit(text, seed, onSettled)  → void   (main.js generate with override)
 *     getPrompt()                    → string (current composed prompt)
 *     compare(aUrl, bUrl, caption)   → void   (main.js ui.showCompare)
 */
import * as store from './store.js';
import { resultUrl } from './api.js';

let mount = null;
let handlers = {};

function texts() {
  const s = store.getState().settings.ab;
  return {
    a: s?.a || '',
    b: s?.b || '',
  };
}

function save() {
  store.getState().settings.ab = {
    a: mount?.querySelector('[data-role="a"]')?.value || '',
    b: mount?.querySelector('[data-role="b"]')?.value || '',
  };
  store.save();
}

function status(html) {
  const el = mount?.querySelector('[data-role="status"]');
  if (el) el.innerHTML = html;
}

function escapeHtml(text) {
  return text.replace(/[<>&"]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;' }[c]));
}

/* -------------------------------------------------------------------- run */

async function run() {
  const ta = mount.querySelector('[data-role="a"]');
  const tb = mount.querySelector('[data-role="b"]');
  const button = mount.querySelector('[data-role="run"]');
  if (!ta.value.trim() || !tb.value.trim()) {
    handlers.toast?.('A/B 两组 prompt 都要填写', 'bad');
    return;
  }
  save();
  button.disabled = true;
  // one shared seed baseline → the only difference is the prompt
  const seed = handlers.getSeed && handlers.getSeed() >= 0
    ? handlers.getSeed()
    : Math.floor(Math.random() * 1e9);

  let jobA = null;
  let jobB = null;
  const finish = async () => {
    if (!(jobA && jobB)) return;
    status('✓ 两组都已完成，正在打开对比…');
    // resolve() goes through main.js's chunk-aware ensureUrl: a 2K/4K result
    // exceeds the Netlify function response limit if fetched as one request.
    const resolve = handlers.resolve || (async (job) => resultUrl(job.id, 0));
    try {
      const [urlA, urlB] = await Promise.all([resolve(jobA), resolve(jobB)]);
      handlers.compare?.(urlA, urlB, `A/B · seed ${seed} · 左 A 右 B`);
      handlers.toast?.('⚖ A/B 完成，已打开并排对比');
    } catch (error) {
      status(`<span class="bad">取图失败：${escapeHtml(String(error?.message || error))}</span>`);
    } finally {
      button.disabled = false;
    }
  };

  status(`<span class="busy">◌ A 组排队中…</span>`);
  try {
    handlers.submit?.(ta.value.trim(), seed, (job) => {
      jobA = job.status === 'done' ? job : null;
      if (!jobA) button.disabled = false;
      status(`<span class="${jobA ? 'ok' : 'bad'}">${jobA ? '✓ A 组完成' : '✗ A 组失败'}</span> · <span class="busy">◌ B 组排队中…</span>`);
      if (jobA) finish();
    });
    handlers.submit?.(tb.value.trim(), seed, (job) => {
      jobB = job.status === 'done' ? job : null;
      if (!jobB) button.disabled = false;
      if (!jobB) status(`<span class="bad">✗ B 组失败</span>`);
      else if (!jobA) status(`<span class="busy">◌ A 组排队中…</span> · <span class="ok">✓ B 组完成</span>`);
      else status('✓ 两组都已完成，正在打开对比…');
      if (jobB) finish();
    });
  } catch (error) {
    button.disabled = false;
    status(`<span class="bad">提交失败：${escapeHtml(String(error?.message || error))}</span>`);
  }
}

/* ------------------------------------------------------------------ render */

function render() {
  if (!mount) return;
  const t = texts();
  mount.innerHTML = `
    <section class="card">
      <p class="eyebrow">A/B Test</p>
      <h2>⚖ A/B 出片对比</h2>
      <p class="muted">两版 prompt 各生成 2 张、共用同一个 seed，完成后自动打开并排对比 —— 只比较 prompt 的差别。</p>
      <div class="ab-grid">
        <div class="ab-cell">
          <label><span><i class="tag-a">A</i> 组 prompt</span>
            <button type="button" class="btn ghost tiny" data-role="a-copy">从当前复制</button></label>
          <textarea class="ta-a" data-role="a" maxlength="500" placeholder="Prompt A…" spellcheck="false">${escapeHtml(t.a)}</textarea>
        </div>
        <div class="ab-cell">
          <label><span><i class="tag-b">B</i> 组 prompt</span>
            <button type="button" class="btn ghost tiny" data-role="b-copy">从当前复制</button></label>
          <textarea class="ta-b" data-role="b" maxlength="500" placeholder="Prompt B…" spellcheck="false">${escapeHtml(t.b)}</textarea>
        </div>
      </div>
      <div class="btn-row tight">
        <button type="button" class="btn big" data-role="run">⚖ 生成 A/B（各 2 张）</button>
        <button type="button" class="btn ghost" data-role="swap">⇄ 交换 A/B</button>
      </div>
      <p class="ab-status" data-role="status" aria-live="polite"></p>
    </section>`;

  mount.querySelectorAll('textarea').forEach((el) => { el.oninput = save; });
  mount.querySelector('[data-role="run"]').onclick = run;
  mount.querySelector('[data-role="swap"]').onclick = () => {
    const a = mount.querySelector('[data-role="a"]');
    const b = mount.querySelector('[data-role="b"]');
    [a.value, b.value] = [b.value, a.value];
    save();
    handlers.toast?.('⇄ 已交换 A/B');
  };
  mount.querySelector('[data-role="a-copy"]').onclick = () => {
    mount.querySelector('[data-role="a"]').value = handlers.getPrompt?.() || '';
    save();
  };
  mount.querySelector('[data-role="b-copy"]').onclick = () => {
    mount.querySelector('[data-role="b"]').value = handlers.getPrompt?.() || '';
    save();
  };
}

/* -------------------------------------------------------------------- api */

export function initAB(opts) {
  mount = opts.mount;
  handlers = opts;
  render();
  return { render };
}
