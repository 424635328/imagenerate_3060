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
 *   initMotion()             → scroll reveals, large-title topbar, theme wipe
 *   enterPanel(el)           → spring entrance for a freshly shown tab panel
 *   withThemeWipe(fn, ev)    → circular view-transition around a theme swap
 *
 * Every feature below degrades to "no motion at all" rather than to a broken
 * state: unsupported APIs are detected, and prefers-reduced-motion skips them.
 */

const reduced = () => window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches === true;
export function prefersReducedMotion() { return reduced(); }

/** Safe `CSS.supports` (jsdom has no CSS global). */
function supports(prop, value) {
  try {
    return typeof CSS !== 'undefined' && !!CSS.supports && CSS.supports(prop, value);
  } catch { return false; }
}

const finePointer = () => window.matchMedia?.('(hover: hover) and (pointer: fine)')?.matches === true;


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

/**
 * Scroll reveal for below-the-fold blocks.
 *
 * One mechanism everywhere (IntersectionObserver + a CSS animation) instead of
 * a scroll-timeline/IO split: elements are only tagged when they start below
 * the fold, and the classes are removed on `animationend` so the reveal can
 * never fight a hover transform.  Skipped entirely under reduced motion.
 */
const REVEAL_SELECTOR = '.card, .scene-card, .stat-card, .lib-item, .gallery .gi, .imgs .shot';

function revealOne(el, io) {
  el.classList.add('ls-reveal', 'is-in');
  el.addEventListener('animationend', () => {
    el.classList.remove('ls-reveal', 'is-in');
    io.unobserve(el);
  }, { once: true });
}

export function initReveal(root = document) {
  if (reduced() || typeof IntersectionObserver !== 'function') return null;
  const fold = window.innerHeight || 800;
  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry) => { if (entry.isIntersecting) revealOne(entry.target, io); });
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.02 });
  const scan = () => {
    root.querySelectorAll(REVEAL_SELECTOR).forEach((el) => {
      if (el.dataset.lsReveal === '1') return;
      el.dataset.lsReveal = '1';                   // tag once, never re-tag
      // Anything already visible at boot must NOT be hidden and re-revealed.
      if (el.getBoundingClientRect().top > fold) io.observe(el);
    });
  };
  scan();
  return { scan, observer: io };
}

/* ------------------------------------------------- large-title topbar state */

/**
 * iOS-style condensed header: past 24 px the title block scales, the subtitle
 * dissolves and the scroll-edge hairline fades in (css/motion.css owns looks).
 * Desktop only — the bar is `position: static` on phones, so nothing to pin.
 */
function initTopbar() {
  const bar = document.querySelector('.shell > .topbar, header.topbar');
  if (!bar) return;
  const wide = window.matchMedia?.('(min-width: 721px)');
  let ticking = false;
  const sync = () => {
    ticking = false;
    const on = (wide?.matches !== false) && (window.scrollY || window.pageYOffset || 0) > 24;
    bar.classList.toggle('is-condensed', on);
  };
  const onScroll = () => {
    if (ticking) return;                            // rAF-coalesced, never per-event
    ticking = true;
    window.requestAnimationFrame(sync);
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  wide?.addEventListener?.('change', onScroll);
  sync();
}

/* ------------------------------------------------------------ panel switch */

/**
 * Spring entrance for a tab panel that just became visible.  The class is
 * removed on animationend so the panel returns to its natural styling.
 */
export function enterPanel(el) {
  if (!el || reduced()) return;
  el.classList.remove('is-enter');
  void el.offsetWidth;                              // restart the animation
  el.classList.add('is-enter');
  el.addEventListener('animationend', () => el.classList.remove('is-enter'), { once: true });
}

/* --------------------------------------------------------- theme switching */

/**
 * Run `apply` inside a circular view-transition wipe centred on the click
 * (Apple-style reveal).  Falls back to calling `apply` directly whenever the
 * API, the pointer event or motion preferences make the effect unavailable —
 * the theme change itself must never depend on the animation.
 */
export function withThemeWipe(apply, event) {
  const canWipe = typeof document.startViewTransition === 'function'
    && supports('view-transition-name', 'none')
    && !reduced();
  if (!canWipe) { apply(); return; }
  const root = document.documentElement;
  if (event && Number.isFinite(event.clientX)) {
    root.style.setProperty('--vt-x', `${event.clientX}px`);
    root.style.setProperty('--vt-y', `${event.clientY}px`);
  }
  root.classList.add('vt-theme');
  try {
    const t = document.startViewTransition(() => apply());
    const done = () => root.classList.remove('vt-theme');
    t.finished.then(done, done);
  } catch {
    root.classList.remove('vt-theme');
    apply();                                        // never lose the theme swap
  }
}

/* ----------------------------------------------------------------- numbers */

/** Spring-ish settle used by the counters: quick out, long soft landing. */
function easeOutQuint(t) { return 1 - Math.pow(1 - t, 5); }

export function animateNumber(el, to, durationMs = 340) {
  if (!el) return;
  const from = parseFloat(el.textContent.replace(/[^\d.]/g, '')) || 0;
  if (reduced()) { el.textContent = String(to); return; }
  const start = performance.now();
  const tick = (now) => {
    const t = Math.min(1, (now - start) / durationMs);
    el.textContent = String(Math.round(from + (to - from) * easeOutQuint(t)));
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

/* ------------------------------------------------------------------- boot */

/** Wire the ambient motion layer.  Safe to call once from main.js boot(). */
export function initMotion() {
  document.documentElement.classList.toggle('has-fine-pointer', finePointer());
  try { initTopbar(); } catch { /* header stays expanded — cosmetic only */ }
  try { return initReveal(); } catch { return null; }
}

