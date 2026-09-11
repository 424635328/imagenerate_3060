/**
 * Motion service — one place for every JS-driven interaction state.
 *
 * CSS handles *what it looks like* (css/motion.css); this module only flips
 * state classes and tweens numbers, so components stay free of animation math.
 *
 *   prefersReducedMotion()   → gate for JS-driven motion
 *   setButtonState(btn, s)   → idle | busy | done  (label crossfade)
 *   animateNumber(el, to)    → 150–300 ms count-up (stats panel)
 *   animateNumbers(host)     → every [data-count] inside `host`
 */

const reduced = () => window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches === true;
export function prefersReducedMotion() { return reduced(); }

/* ----------------------------------------------------------------- buttons */

const BUSY_LABELS = ['◌ 准备中…', '◌ 排队中…', '◌ 渲染中…'];

/**
 * Button lifecycle.  The label swaps behind a quick opacity dip so state
 * changes read as one continuous motion instead of a text pop.
 *   idle → normal label · busy → progress label + energy flow · done → ✓ flash
 */
export function setButtonState(btn, state, label = '') {
  if (!btn) return;
  const span = btn.querySelector('.btn-label');
  if (span) {
    btn.classList.add('is-swap');
    if (reduced()) {
      span.textContent = label || span.dataset.idle || span.textContent;
      btn.classList.remove('is-swap');
    } else {
      window.setTimeout(() => {
        span.textContent = label || span.dataset.idle || span.textContent;
        btn.classList.remove('is-swap');
      }, 120);
    }
  } else if (label) {
    btn.textContent = label;
  }
  btn.classList.toggle('is-busy', state === 'busy');
  btn.classList.toggle('is-done', state === 'done');
}

/** Pick a friendly busy label; callers may override with a real percentage. */
export function busyLabel(progress) {
  if (typeof progress === 'number' && progress > 0) return `◌ ${Math.round(progress * 100)}%`;
  return BUSY_LABELS[Math.floor(Math.random() * BUSY_LABELS.length)];
}

/* ------------------------------------------------------------------ reveal */

/* Blur-up reveal is owned by CSS (css/tokens.css `ls-fade-in` on stage and
   gallery imagery) so the lightbox zoom engine keeps exclusive control of
   `.lb-stage img` transforms.  No JS twin is needed. */

/* ----------------------------------------------------------------- numbers */

function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

export function animateNumber(el, to, durationMs = 260) {
  if (!el) return;
  const from = parseFloat(el.textContent.replace(/[^\d.]/g, '')) || 0;
  if (reduced()) { el.textContent = String(to); return; }
  const start = performance.now();
  const tick = (now) => {
    const t = Math.min(1, (now - start) / durationMs);
    el.textContent = String(Math.round(from + (to - from) * easeOutCubic(t)));
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/** Tween every `[data-count]` inside `host` (stats panel refresh). */
export function animateNumbers(host) {
  if (!host) return;
  host.querySelectorAll('[data-count]').forEach((el) => {
    animateNumber(el, parseInt(el.dataset.count, 10) || 0);
  });
}
