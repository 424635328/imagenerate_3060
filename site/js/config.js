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
