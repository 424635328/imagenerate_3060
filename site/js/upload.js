/**
 * Reference-image uploads (img2img).
 *
 * A phone photo is 4–12 MB of JPEG; the proxy caps a request at 6 MB and the
 * model only consumes <=1024 px anyway.  So we decode, downscale and re-encode
 * to WebP in the browser: uploads shrink by roughly 10–40x and the request
 * finishes long before the GPU is free, which is the real transfer win.
 */
import { UPLOAD_MAX_EDGE, UPLOAD_QUALITY, UPLOAD_TARGET_BYTES } from './config.js';

const SUPPORTED = /^image\/(png|jpeg|jpg|webp|avif|bmp)$/i;

function loadBitmap(file) {
  if (window.createImageBitmap) return createImageBitmap(file);
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('图片无法解码')); };
    img.src = url;
  });
}

function toBlob(canvas, type, quality) {
  return new Promise((resolve) => canvas.toBlob(resolve, type, quality));
}

async function blobToBase64(blob) {
  const buffer = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  for (let i = 0; i < buffer.length; i += 0x8000) {
    binary += String.fromCharCode.apply(null, buffer.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

/**
 * @returns {Promise<{dataUrl:string, base64:string, mime:string, width:number,
 *   height:number, originalBytes:number, bytes:number, ratio:number, ms:number,
 *   thumbUrl:string}>}
 */
export async function compressForUpload(file, { maxEdge = UPLOAD_MAX_EDGE, quality = UPLOAD_QUALITY } = {}) {
  if (!file) throw new Error('没有选择文件');
  if (file.type && !SUPPORTED.test(file.type)) throw new Error('仅支持 PNG / JPEG / WebP / AVIF / BMP');
  const started = performance.now();
  const bitmap = await loadBitmap(file);
  const scale = Math.min(1, maxEdge / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(64, Math.round(bitmap.width * scale));
  const height = Math.max(64, Math.round(bitmap.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { alpha: false });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close?.();

  let mime = 'image/webp';
  let blob = await toBlob(canvas, mime, quality);
  if (!blob) { mime = 'image/jpeg'; blob = await toBlob(canvas, mime, quality); }
  // Two extra passes at lower quality are cheaper than a rejected 6 MB request.
  let step = quality;
  while (blob && blob.size > UPLOAD_TARGET_BYTES && step > 0.5) {
    step -= 0.12;
    blob = await toBlob(canvas, mime, step) || blob;
  }
  if (!blob) throw new Error('浏览器无法编码该图片');

  const base64 = await blobToBase64(blob);
  return {
    dataUrl: `data:${mime};base64,${base64}`,
    base64,
    mime,
    width,
    height,
    originalBytes: file.size,
    bytes: blob.size,
    ratio: file.size ? Math.max(1, file.size / blob.size) : 1,
    ms: Math.round(performance.now() - started),
    thumbUrl: URL.createObjectURL(blob),
  };
}

export function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

/** Accept a drop / paste event and return the first usable image file. */
export function extractImage(event) {
  const items = event.dataTransfer?.files || event.clipboardData?.files;
  if (items?.length) return items[0];
  const clip = event.clipboardData?.items;
  if (clip) {
    for (const item of clip) {
      if (item.type?.startsWith('image/')) return item.getAsFile();
    }
  }
  return null;
}
