/**
 * Static data and tuning constants for the Landscape·Art workbench.
 * Pure data only: no DOM access, no network calls, no secrets.
 */

export const API = '/.netlify/functions/proxy';
export const STORE_KEY = 'lsart_v3';
export const CHUNK_BYTES = 2000000;      // must stay <= proxy MAX_CHUNK
export const DIRECT_LIMIT = 4000000;     // above this we switch to parallel chunks
export const CHUNK_WORKERS = 4;
export const MAX_HISTORY = 60;
export const MAX_BLOB_CACHE = 12;        // blob URLs kept alive in memory
export const UPLOAD_MAX_EDGE = 1024;
export const UPLOAD_QUALITY = 0.82;
export const UPLOAD_TARGET_BYTES = 1200000;

/** Poll cadence: fast right after submit, then relaxed to save requests. */
export const POLL_STEPS = [
  { until: 8000, every: 700 },
  { until: 30000, every: 1200 },
  { until: Infinity, every: 2500 },
];

export const SCENE_GROUPS = [
  ['自然奇观', {
    '山川日出': 'a dramatic mountain valley at sunrise, mist, alpine lake, golden light',
    '雪山湖泊': 'a tranquil alpine lake reflecting snow-capped peaks, morning fog',
    '大峡谷': 'a vast red rock canyon with layered cliffs, a river far below, dramatic light',
    '冰川峡湾': 'a glacier-carved fjord with steep cliffs, calm dark water, low clouds',
    '火山地貌': 'a volcanic landscape with black lava fields, steaming craters, moody sky',
    '巨石海岸': 'a rugged coastline with tall sea stacks and crashing waves at sunset',
    '雨林瀑布': 'a lush green rainforest waterfall, mossy rocks, sun rays through the canopy',
    '梯田云海': 'terraced rice fields in mountain mist at dawn, layered green hills',
  }],
  ['海岸与水', {
    '热带海岸': 'a serene turquoise tropical coastline, palm trees, calm sea, golden hour',
    '珊瑚礁海': 'a crystal clear tropical lagoon with coral reef, aerial view, turquoise water',
    '礁石浪花': 'ocean waves crashing over dark rocks, sea spray, long exposure, moody light',
    '白沙海岛': 'a white sand island surrounded by turquoise water, aerial view, sunlight',
    '湖心倒影': 'a calm lake at dusk with perfect mirror reflections of mountains and clouds',
    '瀑布深潭': 'a powerful waterfall plunging into a teal pool in a lush mossy gorge',
  }],
  ['四季', {
    '秋色枫林': 'an autumn forest with golden and red maple leaves, misty path, soft light',
    '春日花海': 'a blooming wildflower meadow below snowy peaks, spring, clear blue sky',
    '冬雪森林': 'a snow-covered pine forest under soft winter light, quiet and cold',
    '樱花湖畔': 'cherry blossoms framing a calm lake with mountains behind, spring',
    '夏日草原': 'rolling green grassland with scattered trees and big white clouds',
    '薰衣草田': 'rows of lavender fields leading to a distant village at sunset',
  }],
  ['光效与天象', {
    '极光雪原': 'aurora borealis over a snowy landscape, winter night, stars, green glow',
    '银河星空': 'the milky way over a desert rock formation, starry night, long exposure',
    '云海日出': 'a sea of clouds below mountain peaks at sunrise, warm golden light',
    '金色日落': 'a coastline at golden sunset, warm light, glowing reflections on water',
    '晨雾森林': 'a misty forest at dawn with light rays through the trees, dew',
    '暴风乌云': 'dramatic storm clouds over open plains, moody light, distant rain',
    '彩虹山谷': 'a rainbow over a green valley after rain, wet grass, soft light',
    '沙漠日落': 'golden sand dunes with long shadows at sunset, rippled texture',
  }],
];

/** Prompt Studio modifiers: clicking a chip appends the English fragment. */
export const MODIFIER_GROUPS = [
  ['光线', [
    ['黄金时刻', 'golden hour light'], ['蓝调时刻', 'blue hour, cool twilight'],
    ['逆光', 'backlit, rim light'], ['丁达尔光', 'god rays, volumetric light'],
    ['柔和漫射', 'soft diffused light'], ['高对比', 'high contrast lighting'],
  ]],
  ['天气', [
    ['薄雾', 'light mist'], ['雨后', 'after rain, wet surfaces'],
    ['雪', 'falling snow'], ['雷暴', 'thunderstorm, lightning'],
    ['晴空', 'clear blue sky'], ['厚云', 'heavy dramatic clouds'],
  ]],
  ['镜头', [
    ['广角', 'wide angle, 16mm'], ['长焦压缩', 'telephoto compression, 200mm'],
    ['航拍', 'aerial drone view'], ['低机位', 'low angle view'],
    ['长曝光', 'long exposure, smooth water'], ['浅景深', 'shallow depth of field'],
  ]],
  ['质感', [
    ['电影感', 'cinematic color grading'], ['胶片', 'analog film grain, kodak portra'],
    ['高细节', 'ultra detailed, sharp focus'], ['通透', 'crisp clean atmosphere'],
    ['冷调', 'cool blue tones'], ['暖调', 'warm amber tones'],
  ]],
];

export const RANDOM_PROMPTS = [
  'a scenic landscape, golden hour light, dramatic clouds',
  'a rugged mountain valley at sunrise, mist, alpine lake',
  'a tropical island coastline, turquoise sea, palm trees',
  'a lush green rainforest river, mossy rocks, sunlight rays',
  'a vast golden desert with dunes under a clear blue sky',
  'a majestic waterfall in a mossy gorge, flowing water',
  'a tranquil alpine lake reflecting snow-capped peaks',
  'an aerial view of a winding river through green valleys',
  'a storm breaking over a highland loch, moody light',
  'a frozen lake with cracked ice under pale winter sun',
];

export const NEG_PRESETS = [
  ['标准', 'blurry, low quality, watermark, text, distorted, oversaturated, bad anatomy'],
  ['干净风景', 'people, buildings, text, watermark, logo, blurry, low quality, jpeg artifacts'],
  ['自然色彩', 'oversaturated, hdr halo, neon colors, overprocessed, watermark, text'],
  ['去人造物', 'power lines, fence, road signs, cars, text, watermark, frame, border'],
  ['空', ''],
];

/**
 * Generation recipes. `steps`/`cfg`/`sampler` are applied on selection; the
 * backend clamps them again, so a broken value can never reach the GPU.
 */
export const RECIPES = [
  { id: 'fast', label: '⚡ 极速草稿', detail: 'LCM · 6 步 · CFG 1.5 · ≈4 s', steps: 6, cfg: 1.5, sampler: 'lcm', fast: true },
  { id: 'std', label: '⚖️ 标准成片', detail: 'DPM++ 2M Karras · 24 步 · ≈6 s', steps: 24, cfg: 7.5, sampler: 'dpmpp2m_karras', fast: false },
  { id: 'tcd', label: '🚀 TCD 少步', detail: 'TCD · 6 步（需 TCD 权重）', steps: 6, cfg: 1.5, sampler: 'tcd', fast: true },
  { id: 'fine', label: '✨ 精细纹理', detail: 'DPM++ 2M SDE Karras · 32 步', steps: 32, cfg: 7.5, sampler: 'dpmpp2m_sde', fast: false },
  { id: 'custom', label: '🎛 自定义', detail: '手动控制采样器与步数', steps: null, cfg: null, sampler: null, fast: false },
];

export const CLARITY = {
  std: { highres: 0, enhance: 0, label: '标准 512/640' },
  hd: { highres: 1024, enhance: 0, label: '高清 1024 两段式' },
  ultra: { highres: 0, enhance: 2048, label: '超清 2048' },
  max: { highres: 0, enhance: 3072, label: '极清 3072' },
  '4k': { highres: 0, enhance: 4096, label: '4K 4096（分块回传）' },
};

export const STAGE_LABELS = {
  queued: '排队中', denoise: '去噪中', highres: '两段式高清',
  upscale: '神经超分', enhance: '分块重绘', encode: '编码输出', done: '完成',
};

export const SHORTCUTS = [
  ['Ctrl / ⌘ + Enter', '立即生成'],
  ['←  /  →', '切换上一张 / 下一张'],
  ['F', '全屏灯箱'],
  ['C', '开关对比滑块'],
  ['R', '换随机 seed 并生成'],
  ['S', '保存当前图'],
  ['Esc', '关闭弹层'],
  ['?', '打开本帮助'],
];

/* ------------------------------------------------------------------ advisor
 * 参数顾问的「知识库」：这里只放**静态语义**（每个参数是干什么的、单位、类型）。
 * 「推荐值」一律由 js/advisor.js 现算——依据采样器画像、质量饱和曲线、
 * 显存模型，以及**本机历史实测**（store.stats）标定后的成本模型。
 */

export const PARAM_SPECS = {
  prompt: { what: '主体内容。留空则随机抽一个场景；修饰词与实验室片段会自动追加在后面。', kind: 'text' },
  style: { what: '全局画风先验。写实摄影最贴合本项目的数据集与 LoRA。', kind: 'select' },
  recipe: { what: '一键配方：同时设定采样器、步数与 CFG。选「自定义」后三者可独立调整。', kind: 'select' },
  sampler: { what: '采样器决定去噪轨迹。少步采样器（LCM/TCD）用 4–8 步出草图，质量型采样器需要 20+ 步。', kind: 'select' },
  steps: { what: '去噪步数：每步都消耗时间，但质量在饱和点之后几乎不再提升。', kind: 'range', unit: '步' },
  cfg: { what: '提示词引导强度（CFG）：越高越贴合文字、也越容易过饱和与僵硬。', kind: 'range' },
  seed: { what: '随机种子：同 seed + 同参数 = 完全相同的图；-1 表示每次随机。', kind: 'number' },
  res: { what: '基础生成边长。SD1.5 原生域是 512，640 是画质与速度的折中点。', kind: 'select', unit: 'px' },
  aspect: { what: '画幅比例：横构图适合山脉与海岸，竖构图适合瀑布与森林。', kind: 'select' },
  batch: { what: '一次任务产出几张变体（同一队列位、一次轮询）。耗时随张数线性增长。', kind: 'select', unit: '张' },
  clarity: { what: '清晰度方案：先基础生成，再超分/分块重绘补细节，避免 6GB 直接出 2K+。', kind: 'select' },
  sr: { what: '超分模型：自训·风景在本机数据上微调（细节 6.5–8.7× bicubic），UltraSharp 最锐，Real-ESRGAN 柔和适合云雾。', kind: 'select' },
  negPreset: { what: '负向预设：换一组常用排除词。少步模式下负向引导很弱。', kind: 'select' },
  neg: { what: '负向提示词：告诉模型不要出现什么。', kind: 'text' },
  strength: { what: '重绘强度（img2img）：越小越贴近参考图，越大越自由。', kind: 'range' },
};

/** 采样器画像：tau 是质量饱和常数（步数到 ~3τ 时边际收益已很低），
 *  cfgSweet 是经验最优引导区间，fewStep 表示少步采样器。 */
export const SAMPLER_PROFILE = {
  lcm: { label: 'LCM', tau: 2.0, cfgSweet: [1.0, 2.0], fewStep: true, note: '少步：6 步左右即饱和，CFG 必须压低' },
  tcd: { label: 'TCD', tau: 2.0, cfgSweet: [1.0, 2.0], fewStep: true, note: '少步：与 LCM 同族，需对应权重' },
  dpmpp2m_karras: { label: 'DPM++ 2M Karras', tau: 8.0, cfgSweet: [6.5, 8.0], fewStep: false, note: '标准成片：Karras 调度让 24 步左右到饱和' },
  dpmpp2m_sde: { label: 'DPM++ 2M SDE', tau: 10.0, cfgSweet: [6.0, 7.5], fewStep: false, note: '纹理更细腻：需要 ~30 步，且随机性更强' },
  dpmpp2m: { label: 'DPM++ 2M', tau: 9.0, cfgSweet: [6.5, 8.0], fewStep: false, note: '不带 Karras：同步数下略逊于 Karras 版' },
  euler_a: { label: 'Euler Ancestral', tau: 9.0, cfgSweet: [6.0, 7.5], fewStep: false, note: '每步注入噪声：构图变化大，适合探索' },
  euler: { label: 'Euler', tau: 8.0, cfgSweet: [6.5, 8.0], fewStep: false, note: '确定性欧拉：稳定但细节一般' },
  ddim: { label: 'DDIM', tau: 12.0, cfgSweet: [6.5, 8.0], fewStep: false, note: '老式调度：需要更多步，性价比低' },
};

/** 成本/显存模型系数（可由 advisor.calibrate() 用本机实测覆盖）。
 *  cost:  seconds ≈ warmup + k * steps * MP^b      （MP = 百万像素）
 *  sr:    seconds ≈ srK * targetMP + srC            （超分阶段）
 *  redraw:seconds ≈ redrawK * strength * targetMP   （分块重绘，4K 档后端置 strength=0）
 *  vram:  GB ≈ base + slope * MP + clarityExtra[mode]，budget 为 6GB 卡可用上限
 *  初值锚点全部来自项目实测：
 *    512/24 步 ≈ 6 s；768/24 步 ≈ 22.6 s（README §12.1）
 *    2048 超清 ≈ 63 s、4096（strength=0）≈ 30 s（README §6.3）
 *    峰值显存：512 → 2.8 GB、768 → 3.7 GB
 */
export const ADVISOR_MODEL = {
  cost: { k: 2.14, b: 1.6, warmup: 0.8 },
  sr: { srK: 1.0, srC: 8, redrawK: 35 },
  vram: { base: 2.08, slope: 2.74, budget: 5.6, clarityExtra: { std: 0, hd: 0.6, ultra: 0.4, max: 0.5, '4k': 0.6 } },
};
