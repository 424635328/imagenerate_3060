/**
 * v7 shell — desktop panel tabs for both columns.
 *
 * One panel per column is visible at a time (canvas stays pinned above the
 * right tabs); the selection persists in settings.shellTabs.  `reveal(id)`
 * makes sure an element referenced by a command/shortcut/mobile anchor is
 * actually visible on desktop before anything scrolls to it.
 */
import * as store from './store.js';
import * as motion from './motion.js';

const DEFAULT_TABS = { left: 'create', right: 'queue' };

function activeTabs() {
  const saved = store.getState().settings.shellTabs;
  return {
    left: saved?.left || DEFAULT_TABS.left,
    right: saved?.right || DEFAULT_TABS.right,
  };
}

function apply(side, name, animate = false) {
  const bar = document.querySelector(`.panel-tabs[data-shell="${side}"]`);
  if (!bar) return name;
  const buttons = [...bar.querySelectorAll('[data-panel]')];
  // A persisted tab that no longer exists (renamed/removed panel) must fall
  // back instead of hiding every panel and blanking the whole column.
  const match = buttons.find((b) => b.dataset.panel === name)
    || buttons.find((b) => b.dataset.panel === DEFAULT_TABS[side])
    || buttons[0];
  const active = match?.dataset.panel;
  buttons.forEach((button) => {
    const on = button.dataset.panel === active;
    button.classList.toggle('on', on);
    button.setAttribute('aria-selected', String(on));
  });
  // Only this side's panels are touched — hiding every .tab-panel would blank
  // the other column the moment its own tabs were applied.
  document.querySelectorAll(`.tab-panel[data-shell="${side}"]`).forEach((panel) => {
    const show = panel.dataset.panel === active;
    const changed = panel.hidden === show;        // about to become visible
    panel.hidden = !show;
    if (show && changed && animate) motion.enterPanel(panel);
  });
  return active;
}

function setTab(side, name) {
  const tabs = activeTabs();
  const applied = apply(side, name, true) || name;
  tabs[side] = applied;
  store.getState().settings.shellTabs = tabs;
  store.save();
}

/** Make the panel containing element `id` active (desktop); mobile no-op. */
export function reveal(id) {
  const target = document.getElementById(id);
  const panel = target?.closest('.tab-panel');
  if (!panel || !panel.dataset.shell) return;
  if (panel.hidden) setTab(panel.dataset.shell, panel.dataset.panel);
}

export function initShell() {
  const tabs = activeTabs();
  document.querySelectorAll('.panel-tabs').forEach((bar) => {
    const side = bar.dataset.shell;
    const applied = apply(side, tabs[side]);
    if (applied && applied !== tabs[side]) setTab(side, applied);   // heal a stale tab name
    bar.querySelectorAll('[data-panel]').forEach((button) => {
      button.onclick = () => setTab(side, button.dataset.panel);
    });
  });
  return { reveal };
}
