/**
 * Proxy client.  Everything goes through the same-origin Netlify function, so
 * the tunnel URL and API key stay server-side.
 */
import { API, CHUNK_BYTES, CHUNK_WORKERS, DIRECT_LIMIT, POLL_STEPS } from './config.js';

async function readJson(response) {
  try {
    return await response.json();
  } catch (error) {
    return {};
  }
}

export async function health() {
  const r = await fetch(`${API}?op=health`, { cache: 'no-store' });
  const data = await readJson(r);
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

export async function warmup(body) {
  const r = await fetch(`${API}?op=warmup`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await readJson(r);
  if (!r.ok) throw new Error(data.detail || data.error || `HTTP ${r.status}`);
  return data;
}

export async function collectGarbage(unload = false) {
  const r = await fetch(`${API}?op=gc&unload=${unload ? '1' : '0'}`, { method: 'POST' });
  const data = await readJson(r);
  if (!r.ok) throw new Error(data.detail || data.error || `HTTP ${r.status}`);
  return data;
}

export async function submit(body) {
  const r = await fetch(`${API}?op=generate`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await readJson(r);
  if (!r.ok) throw new Error(data.detail || data.error || `HTTP ${r.status}`);
  return data;
}

export async function job(id) {
  const r = await fetch(`${API}?op=job&id=${encodeURIComponent(id)}`, { cache: 'no-store' });
  if (r.status === 404) return null;
  const data = await readJson(r);
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}

/** Cancel a job that is still queued (running GPU work answers 409). */
export async function cancelJob(id) {
  const r = await fetch(`${API}?op=cancel&id=${encodeURIComponent(id)}`, { method: 'POST' });
  const data = await readJson(r);
  if (!r.ok) throw new Error(data.detail || data.error || `HTTP ${r.status}`);
  return data;
}

export function resultUrl(id, index = 0) {
  const i = Math.max(0, Math.min(3, index | 0));
  return i ? `${API}?op=image&id=${encodeURIComponent(id)}&i=${i}`
           : `${API}?op=image&id=${encodeURIComponent(id)}`;
}

/** Preview URL is revision-busted because the file changes while denoising. */
export function previewUrl(id, rev = 0) {
  return `${API}?op=preview&id=${encodeURIComponent(id)}&rev=${rev}`;
}

/**
 * Download a finished image.  Small files come straight from the proxy (browser
 * cache friendly); large ones are fetched as ordered parallel byte ranges and
 * assembled into a Blob, which is how 2K/4K results get past the 6 MB function
 * response limit.
 */
export async function fetchImage(id, bytes, onProgress, index = 0) {
  const i = Math.max(0, Math.min(3, index | 0));
  if (!bytes || bytes <= DIRECT_LIMIT) return { url: resultUrl(id, i), chunks: 0, bytes: bytes || 0 };
  const offsets = [];
  for (let off = 0; off < bytes; off += CHUNK_BYTES) offsets.push(off);
  const parts = new Array(offsets.length);
  let next = 0;
  let received = 0;

  async function pump() {
    while (next < offsets.length) {
      const slot = next++;
      const off = offsets[slot];
      const r = await fetch(`${API}?op=chunk&id=${encodeURIComponent(id)}&off=${off}&len=${CHUNK_BYTES}&i=${i}`);
      if (!r.ok) throw new Error(`chunk HTTP ${r.status}`);
      parts[slot] = await r.arrayBuffer();
      received += parts[slot].byteLength;
      if (onProgress) onProgress(received, bytes);
    }
  }

  const workers = Array.from({ length: Math.min(CHUNK_WORKERS, offsets.length) }, pump);
  await Promise.all(workers);
  const total = parts.reduce((sum, part) => sum + (part?.byteLength || 0), 0);
  if (total !== bytes) throw new Error(`assembled ${total} of ${bytes} bytes`);
  return { url: URL.createObjectURL(new Blob(parts, { type: 'image/webp' })), chunks: parts.length, bytes: total };
}

/** Interval that starts tight and relaxes, so early previews appear fast. */
export function pollDelay(elapsedMs) {
  return (POLL_STEPS.find((step) => elapsedMs < step.until) || POLL_STEPS.at(-1)).every;
}

/**
 * Poll a job until it settles.  Uses self-scheduling timeouts (not setInterval)
 * so a slow response can never stack requests, and drops to a slow heartbeat
 * (5 s) while the tab is hidden so a background page stops burning proxy
 * invocations; main.js catches the UI up on visibilitychange.
 */
export function watchJob(id, { onUpdate, onDone, onFail, onGone }) {
  const started = Date.now();
  let stopped = false;
  let timer = null;

  const stop = () => { stopped = true; if (timer) clearTimeout(timer); };

  const tick = async () => {
    if (stopped) return;
    if (document.hidden) {
      timer = setTimeout(tick, 5000);
      return;
    }
    try {
      const data = await job(id);
      if (stopped) return;
      if (!data) { stop(); onGone?.(); return; }
      if (data.status === 'done') { stop(); await onDone?.(data); return; }
      if (data.status === 'failed' || data.status === 'expired' || data.status === 'cancelled') {
        stop(); onFail?.(data); return;
      }
      onUpdate?.(data);
    } catch (error) {
      onUpdate?.({ status: 'retry', error: String(error.message || error) });
    }
    if (!stopped) timer = setTimeout(tick, pollDelay(Date.now() - started));
  };

  tick();
  return stop;
}
