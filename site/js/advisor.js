/**
 * advisor.js — 参数顾问：为每个参数格子给出「作用 + 推荐值 + 理由」。
 *
 * 设计要点（为什么不直接写死提示文案）：
 *   1. **成本模型自标定**：seconds ≈ k · steps · MP^b。k/b 用**本机历史任务**
 *      （含分辨率、步数、耗时）做最小二乘拟合；样本不足时退回 config 里的实测锚点。
 *   2. **步数不是查表，而是求饱和点**：把质量拟合为 Q(n) = 1 − e^(−n/τ)，
 *      τ 来自采样器画像；推荐 Q 达到 95% 的最小整数步数（≈ 3τ），
 *      因此「LCM 6 步」与「SDE 30 步」是同一个公式算出来的，不是硬编码两句话。
 *   3. **显存是像素的线性模型**：vram ≈ base + slope·MP + 清晰度附加项，
 *      超出 6GB 卡的可用预算（默认 5.6GB）时给出降档建议。
 *   4. **规则只做约束校验**：少步采样器必须低 CFG、高清单张、超分模型与画风匹配等，
 *      都由数据（SAMPLER_PROFILE / CLARITY）推导，不在文案里重复。
 */
import { ADVISOR_MODEL, CLARITY, PARAM_SPECS, SAMPLER_PROFILE } from './config.js';

const el = (id) => document.getElementById(id);
const num = (id, fallback = 0) => {
  const node = el(id);
  const value = parseFloat(node?.value);
  return Number.isFinite(value) ? value : fallback;
};
const text = (id, fallback = '') => (el(id)?.value ?? fallback).toString();

/* ----------------------------------------------------------- 成本 / 显存 */

export function megapixels(res, aspect) {
  const side = parseFloat(res) || 512;
  // 方/横/竖只改变边长分配，总像素不变（后端按最长边生成），所以直接用边长的平方
  return (side * side) / 1e6;
}

/** 用历史任务拟合 k、b（只看基础路径：clarity=std，单张，有耗时）。 */
export function calibrate(base = ADVISOR_MODEL, jobs = []) {
  const rows = jobs
    .filter((job) => job && job.status === 'done' && Number(job.seconds) > 0)
    .filter((job) => Number(job.steps) > 0 && !Number(job.enhance) && !Number(job.highres))
    .map((job) => ({
      x: Math.log((Number(job.steps) || 1) * megapixels(job.res || 512, job.aspect || '方')),
      y: Math.log(Math.max(0.2, Number(job.seconds))),
    }));
  if (rows.length < 6) {
    return { model: base, samples: rows.length, fitted: false };
  }
  const n = rows.length;
  const mx = rows.reduce((sum, r) => sum + r.x, 0) / n;
  const my = rows.reduce((sum, r) => sum + r.y, 0) / n;
  const cov = rows.reduce((sum, r) => sum + (r.x - mx) * (r.y - my), 0);
  const varX = rows.reduce((sum, r) => sum + (r.x - mx) ** 2, 0);
  if (varX < 1e-6) return { model: base, samples: n, fitted: false };
  const b = cov / varX;
  const k = Math.exp(my - b * mx - Math.log(base.cost.warmup || 1));   // 预留 warmup 项
  const model = {
    ...base,
    cost: { ...base.cost, k: Math.max(0.2, Math.min(20, k)), b: Math.max(0.5, Math.min(3, b)) },
  };
  const rms = Math.sqrt(rows.reduce((sum, r) => {
    const predicted = Math.log((model.cost.warmup || 0.8) + model.cost.k * Math.exp(r.x));
    return sum + (predicted - r.y) ** 2;
  }, 0) / n);
  return { model, samples: n, fitted: true, rms: Math.exp(rms) };
}

/** 单张预计耗时（秒）：基础去噪 + 超分 + 分块重绘。
 *  注意：设置对象里只有 clarity 这个键，具体的 enhance/highres 目标边长要从 CLARITY 映射取。 */
export function estimateSeconds(s, model = ADVISOR_MODEL) {
  const mp = megapixels(s.res, s.aspect);
  const steps = Math.max(1, Number(s.steps) || 1);
  let seconds = (model.cost.warmup || 0.8) + model.cost.k * steps * mp ** model.cost.b;
  const mode = CLARITY[String(s.clarity || 'std')] || CLARITY.std;
  const enhancerTarget = Number(s.enhance) || Number(mode.enhance) || 0;
  const highresTarget = Number(s.highres) || Number(mode.highres) || 0;
  const target = enhancerTarget || highresTarget;
  if (target) {
    const targetMp = (target * target) / 1e6;
    seconds += model.sr.srK * targetMp + model.sr.srC;
    // 4K 档后端把 strength 置 0（只超分不重绘），所以重绘项要按实际强度算
    const strength = s.hasInit ? Math.max(0, Number(s.strength) || 0)
      : (String(s.clarity) === '4k' ? 0 : 0.3);
    if (enhancerTarget && strength > 0.02) seconds += model.sr.redrawK * strength * targetMp;
  }
  return seconds;
}

/** 峰值显存估计（GB）。 */
export function estimateVram(s, model = ADVISOR_MODEL) {
  const mp = megapixels(s.res, s.aspect);
  const clarity = String(s.clarity || 'std');
  const extra = model.vram.clarityExtra[clarity] ?? 0;
  return model.vram.base + model.vram.slope * mp + extra;
}

/* --------------------------------------------------------------- 推荐值 */

/** 质量饱和曲线：Q(n) = 1 − e^(−n/τ)，返回达到 target 比例所需的最小步数。 */
export function saturationSteps(sampler, target = 0.95) {
  const tau = (SAMPLER_PROFILE[sampler] || SAMPLER_PROFILE.dpmpp2m_karras).tau;
  return Math.max(2, Math.ceil(tau * Math.log(1 / (1 - target))));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

const ASPECT_HINTS = {
  横: ['valley', 'coast', 'ocean', 'canyon', 'field', 'desert', 'panorama', 'dunes', 'ridge', 'bay'],
  竖: ['waterfall', 'forest', 'cliff', 'tree', 'gorge', 'cascade', 'bamboo'],
};

const PAINTERLY = ['水彩', '水墨', '油画', '丙烯印象', '动漫'];

/**
 * 生成全部参数建议。
 * @param {object} s   当前设置（从 DOM 读出）
 * @param {object} ctx { jobs: 历史任务, favorites: 收藏 id, fitted: 成本模型是否已标定 }
 * @returns {{ items: object[], summary: object }}
 */
export function recommend(s, ctx = {}, model = ADVISOR_MODEL) {
  const jobs = Array.isArray(ctx.jobs) ? ctx.jobs : [];
  const profile = SAMPLER_PROFILE[s.sampler] || SAMPLER_PROFILE.dpmpp2m_karras;
  const items = [];
  const push = (id, level, recommended, reason) => items.push({ id, level, recommended, reason });

  // --- 步数：由饱和曲线求，而不是查表 ---
  const sat = saturationSteps(s.sampler);
  const fast = profile.fewStep;
  const stepRec = clamp(fast ? Math.min(sat, 8) : sat, 2, fast ? 12 : 60);
  const steps = Number(s.steps) || 0;
  const q = 1 - Math.exp(-steps / profile.tau);
  const qRec = 1 - Math.exp(-stepRec / profile.tau);
  const baseSec = estimateSeconds({ ...s, steps: 1 }, model) - estimateSeconds({ ...s, steps: 0 }, model);
  const wasted = Math.max(0, steps - stepRec) * baseSec;
  push('steps', steps > stepRec * 1.35 ? 'warn' : 'ok', `${stepRec} 步`,
    `质量已到饱和的 ${(q * 100).toFixed(0)}%（${profile.label} 的 τ≈${profile.tau}）；`
    + (steps > stepRec ? `再往上每步只增加约 ${baseSec.toFixed(2)} s，当前多花 ${wasted.toFixed(1)} s` : `已接近饱和点 ${stepRec} 步（≈95%）`));

  // --- CFG：区间由采样器画像给出 ---
  const [cfgLo, cfgHi] = profile.cfgSweet;
  const cfg = Number(s.cfg) || 0;
  push('cfg', cfg < cfgLo - 0.5 || cfg > cfgHi + 1 ? 'warn' : 'ok',
    `${((cfgLo + cfgHi) / 2).toFixed(1)}`,
    `${profile.label} 的经验区间是 ${cfgLo}–${cfgHi}；${profile.fewStep ? '少步采样器 CFG 高会直接糊掉' : '过高会过饱和、过低则不跟随提示词'}`);

  // --- 基础分辨率：清晰度决定该用 512 还是 640 ---
  const clarity = String(s.clarity || 'std');
  const resRec = clarity === 'std' ? '640' : '512';
  push('res', String(s.res) === resRec ? 'ok' : 'tip', `${resRec} px`,
    clarity === 'std'
      ? '标准路径直接出图：640 在原生化域和细节之间最划算'
      : `高清/超清路径先用 512 出底图再由超分补细节，底图分辨率不必拉高（这是 6GB 卡的省显存做法）`);

  // --- 批量：由清晰度（单张成本）反推 ---
  const perImage = estimateSeconds(s, model);
  const batchRec = clarity === 'std' ? 4 : (clarity === 'hd' ? 2 : 1);
  push('batch', Number(s.batch) > batchRec ? 'tip' : 'ok', `${batchRec} 张`,
    `当前清晰度下单张约 ${perImage.toFixed(1)} s，${batchRec} 张 ≈ ${(perImage * batchRec).toFixed(0)} s`
    + (Number(s.batch) > batchRec ? `；再翻倍会让这一轮等待接近 ${(perImage * Number(s.batch)).toFixed(0)} s` : ''));

  // --- 画幅：从提示词关键词推断 ---
  const prompt = `${s.prompt || ''}`.toLowerCase();
  const score = { 横: 0, 竖: 0 };
  Object.entries(ASPECT_HINTS).forEach(([key, words]) => {
    score[key] = words.reduce((sum, w) => sum + (prompt.includes(w) ? 1 : 0), 0);
  });
  const aspectRec = score.横 > score.竖 ? '横' : (score.竖 > score.横 ? '竖' : '方');
  push('aspect', 'ok', aspectRec, aspectRec === '方'
    ? '提示词里没有明显方向线索：方构图最稳'
    : `提示词里出现${aspectRec === '横' ? '开阔地貌' : '纵向元素'}关键词，${aspectRec}构图更贴合`);

  // --- 清晰度方案：迭代用标准，出片用高清 ---
  const doneCount = jobs.filter((j) => j.status === 'done').length;
  const favorites = Number(ctx.favorites) || 0;
  const clarityRec = favorites > 0 ? 'hd' : (doneCount < 3 ? 'std' : 'ultra');
  push('clarity', clarity === clarityRec ? 'ok' : 'tip', clarityRec,
    favorites > 0
      ? '已有收藏：可以用「高清 1024 两段式」把选中的构图出成片'
      : (doneCount < 3 ? '还在探索阶段：标准路径最快，先定构图' : '已经跑了若干轮：可以上超清补细节'));

  // --- 超分模型：按画风选 ---
  // 自训权重（ours）是在**本地风景分布**上做 Real-ESRGAN 高阶退化微调得到的：
  // 细节量 6.53×（EMA）/8.69×（末轮）vs bicubic，与 UltraSharp 8.03× 同档（research/sr_probe_final）。
  // 写实风景优先用它；绘画风格仍建议柔和的 Real-ESRGAN。
  const painterly = PAINTERLY.includes(String(s.style));
  const srRec = painterly ? 'realesrgan' : 'ours';
  const srLabel = painterly ? 'Real-ESRGAN' : '自训·风景';
  push('sr', String(s.sr) === srRec ? 'ok' : 'tip', srLabel,
    painterly ? `${s.style} 属绘画风格，柔和插值更自然`
              : '写实风景优先用本机自训权重：它在同一分布上微调，纹理更贴合本地数据');

  // --- 负向：非少步模式下别留空 ---
  const neg = String(s.neg || '').trim();
  push('neg', !profile.fewStep && neg.length < 8 ? 'warn' : 'ok', '标准预设',
    profile.fewStep
      ? '少步采样器的负向引导很弱，留空也可接受'
      : (neg ? '已填写；保持 5–10 个高价值词即可，堆太多会削弱构图' : '质量型采样器下负向有效，建议至少排除 blurry/watermark/text'));

  // --- img2img 强度：只在有参考图时有意义 ---
  if (s.hasInit) {
    const strength = Number(s.strength) || 0;
    const rec = clarity === 'std' ? 0.45 : 0.35;
    push('strength', strength > 0.75 || strength < 0.15 ? 'warn' : 'ok', rec.toFixed(2),
      strength > 0.75 ? '超过 0.75 基本会丢掉参考图构图'
        : (strength < 0.15 ? '低于 0.15 几乎不改变参考图' : '0.35–0.5 能保留构图又补足细节'));
  }

  // --- seed：有收藏就建议锁定复现 ---
  const favJob = jobs.find((j) => j.status === 'done' && Number(j.seed) >= 0 && (ctx.favoriteIds || []).includes(j.id));
  const seed = Number(s.seed);
  if (favJob) {
    push('seed', seed === Number(favJob.seed) ? 'ok' : 'tip', String(favJob.seed),
      '这是你收藏过的 seed：锁定它可直接复现那张图，再微调其它参数');
  } else {
    push('seed', seed >= 0 ? 'ok' : 'ok', '-1',
      seed >= 0 ? '已锁定 seed：便于对比参数变化；想探索就设回 -1' : '随机 seed：探索阶段保持 -1');
  }

  const vram = estimateVram(s, model);
  const seconds = estimateSeconds(s, model);
  const summary = {
    seconds, total: seconds * (Number(s.batch) || 1), vram,
    budget: model.vram.budget, saturated: stepRec, tau: profile.tau,
    perStep: baseSec, calibrated: !!ctx.fitted, samples: ctx.samples || 0,
    overBudget: vram > model.vram.budget,
    items,
  };
  return summary;
}

/* ------------------------------------------------------------------ 渲染 */

let options = {};
let last = null;

function fieldHelp(id, spec) {
  const input = el(id);
  const field = input?.closest('.field');
  if (!field) return null;
  let node = field.querySelector(`.advice[data-advice="${id}"]`);
  if (!node) {
    node = document.createElement('p');
    node.className = 'advice';
    node.dataset.advice = id;
    node.setAttribute('role', 'note');
    field.appendChild(node);
  }
  if (spec) {
    // 「作用」说明挂在 label 与输入框上：悬停即可看，不额外占版面
    const label = field.querySelector('label');
    if (label && label.title !== spec.what) label.title = spec.what;
    if (input.title !== spec.what) input.title = spec.what;
  }
  return node;
}

function paint(summary) {
  // 所有已知参数的「作用」工具提示都要挂上（即使本轮没有推荐项）
  Object.keys(PARAM_SPECS).forEach((id) => { if (el(id)) fieldHelp(id, PARAM_SPECS[id]); });
  summary.items.forEach((item) => {
    const node = fieldHelp(item.id, PARAM_SPECS[item.id]);
    if (!node) return;
    const badge = item.level === 'warn' ? '⚠' : (item.level === 'tip' ? '💡' : '✓');
    node.className = `advice advice-${item.level}`;
    node.innerHTML = `<b>${badge} 推荐 ${item.recommended}</b><span>${item.reason}</span>`;
  });
  const host = el('advisorSummary');
  if (host) {
    const vramFlag = summary.overBudget ? 'advice-warn' : 'advice-ok';
    host.innerHTML = `
      <div class="advisor-head">
        <b>参数体检</b>
        <span class="muted">成本模型${summary.calibrated ? `已用本机 ${summary.samples} 条历史标定` : '为内置锚点（历史样本不足）'}</span>
      </div>
      <div class="advisor-metrics">
        <span>单张 ≈ <b>${summary.seconds.toFixed(1)} s</b></span>
        <span>本批 ≈ <b>${summary.total.toFixed(0)} s</b></span>
        <span class="${vramFlag}">峰值显存 ≈ <b>${summary.vram.toFixed(1)} GB</b> / 预算 ${summary.budget} GB</span>
        <span>饱和点 <b>${summary.saturated} 步</b>（τ=${summary.tau}）</span>
      </div>
      <div class="advisor-actions">
        <button type="button" class="btn ghost tiny" data-advisor="apply">采用推荐值</button>
        <button type="button" class="btn ghost tiny" data-advisor="explain">为什么这样推荐</button>
      </div>
      <p class="advisor-explain" hidden></p>`;
    host.querySelector('[data-advisor="apply"]').onclick = () => apply(summary);
    host.querySelector('[data-advisor="explain"]').onclick = (event) => {
      const node = host.querySelector('.advisor-explain');
      node.hidden = !node.hidden;
      if (!node.hidden) {
        node.innerHTML = summary.items.map((item) => {
          const label = el(item.id)?.closest('.field')?.querySelector('label')?.textContent?.trim() || item.id;
          return `<b>${label.replace(/\s*\d.*$/, '')}</b>：${item.reason}`;
        }).join('<br>');
      }
    };
  }
  last = summary;
}

function apply(summary) {
  summary.items.forEach((item) => {
    const input = el(item.id);
    if (!input) return;
    const value = item.recommended.toString().replace(/[^\d.\-]/g, '');
    if (!value) return;
    if (input.tagName === 'SELECT') {
      const match = [...input.options].find((option) => option.value === value || option.textContent.startsWith(value));
      if (match) input.value = match.value;
    } else {
      input.value = value;
    }
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  });
  options.onApply?.();
  refresh();
}

export function refresh() {
  if (!options.getSettings) return null;
  const settings = options.getSettings();
  const jobs = options.getJobs?.() || [];
  const favorites = jobs.filter((job) => options.isFavorite?.(job.id));
  const { model, fitted, samples } = calibrate(ADVISOR_MODEL, jobs);
  const summary = recommend(settings, { jobs, favorites: favorites.length, favoriteIds: favorites.map((j) => j.id), fitted, samples }, model);
  paint(summary);
  return summary;
}

export function init(opts) {
  options = opts || {};
  // 汇总面板插在「质量、速度与尺寸」卡片顶部（即包含 #steps 的那张卡）
  const card = el('steps')?.closest('.card');
  if (card && !el('advisorSummary')) {
    const host = document.createElement('div');
    host.id = 'advisorSummary';
    host.className = 'advisor';
    const hint = card.querySelector('.hint');
    if (hint && hint.nextSibling) card.insertBefore(host, hint.nextSibling);
    else card.insertBefore(host, card.firstChild);
  }
  refresh();
  return { refresh, recommend, estimateSeconds, estimateVram, saturationSteps };
}

export function lastSummary() { return last; }
