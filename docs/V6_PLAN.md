# V6 — Landscape·Art Studio 前端升级方案（审计 + 文件级计划）

> 目标定位：从「一个能调用 SD1.5 的网页」升级为 **AI Landscape Photography Creation Workbench**。
> 本文是**可执行的工程计划**，不是愿望清单：每个条目都对应具体文件、组件名、交互契约与验收方式。

### 已交付（进度台账）

| 波次 | 条目 | 状态 | 落点 |
|---|---|---|---|
| 1 | tokens / studio 皮肤 / palette(Ctrl+K) / studio 组件（相机/光线/灵感/Seed Lab/两段式） | ✅ | `css/tokens.css` `css/studio.css` `css/palette.css` `js/studio.js` `js/palette.js` |
| 1 | 动效层 + 移动端企业级适配（S 级） | ✅ | `css/motion.css` `css/mobile.css` `js/motion.js` + `.mobile-dock` |
| 2 | Scene Explorer 大图卡（分类/搜索/收藏/随机） | ✅ | `js/scenes.js` + `css/scenes.css`（替代 ui.js 旧 chips） |
| 2 | Prompt Composer 词条化（token 条 + 11 组词条） | ✅ | `js/composer.js` + `css/composer.css` |
| 2 | A/B Prompt（各 2 张 · 同 seed · 自动并排对比） | ✅ | `js/ab.js` + `css/ab.css` |
| 2 | Remix（灯箱一键以当前图为参考图再创作） | ✅ | `#lbRemix` + `remixCurrent()` |
| 2 | Service Worker（静态壳缓存，API 永不缓存） | ✅ | `sw.js` + main.js 注册 |
| 3 | 路由 + 六页面 / Workflow 步骤条 / 作品详情页 / 资产筛选 | ⬜ | 待做 |
| 3 | Preview/Thumbnail 分级 · 后端 services/ 拆分 · 结构化错误 | ⬜ | 待做 |
| 4 | 智能化（自动标签/推荐/相似图） | ⬜ | 可选 |

---

## 0. 现状审计（事实数据）

| 层 | 文件 | 规模 | 评价 |
|---|---|---|---|
| 结构 | `site/index.html` | 365 行 / 1 个页面 | 单页承载全部功能，已接近上限 |
| 视觉 | `site/styles.css` + `site/extra.css` | 325 + ~180 行 | 功能齐但视觉同质（卡片 + 分栏），缺"产品感" |
| 逻辑 | `site/js/main.js` | 731 行 | **已是最大的技术债**：编排 + 交互 + 快捷键混在一起 |
| 逻辑 | `site/js/ui.js` | 602 行 | DOM 层，含画布/画廊/灯箱/缩放（新增） |
| 逻辑 | `site/js/api.js` | 135 行 | 干净：轮询自适应、分块并行、`i` 索引 |
| 逻辑 | `site/js/store.js` | ~200 行 | 干净：设置/历史/收藏/词库/统计 + blob LRU |
| 逻辑 | `site/js/config.js` | 142 行 | 纯数据：28 场景 / 修饰词 / 配方 / 快捷键 |
| 逻辑 | `site/js/extras.js` | 165 行 | 统计/词库/检索面板 |
| 逻辑 | `site/js/upload.js` | ~100 行 | 参考图压缩上传 |
| 后端 | `server.py` | 735 行 | 队列/限流/缓存/批量/预览/分块/配额 —— **够用，不需要大改** |

**结论**：后端停止投入（已具备队列、批量、缓存、4K 分块、状态恢复）；火力全部转到前端。

### 当前缺失（对照参考图）
1. **无导航/信息架构**：没有左侧导航栏、没有页面概念、没有顶部工作流步骤条。
2. **Prompt 只是 textarea**：没有 token 化的 composer，没有结构化词条。
3. **场景是纯文字按钮**：没有大图卡片 / 分类 / 搜索 / 悬停预览。
4. **参数平铺**：没有「快速 / 画面 / 专业」三级分层与折叠。
5. **结果区单薄**：无 hover 操作层、无选片模式、无 A/B、无作品详情面板。
6. **缺少"专业工具"感**：无 Command Palette、无相机/光线/构图预设、无 Seed Lab、无 Workflow。
7. **移动端只是响应式缩放**，没有专门的移动 IA。
8. **无微交互/动效层**：无 skeleton、无 reveal、无 crossfade、无 toast 动效。

---

## 1. 目标文件结构（V6）

```
site/
├── index.html                 # 应用外壳：导航 + 路由容器 + 全局弹层
├── rome.html 等               # 不再需要：改为 SPA 路由（见下）
├── css/
│   ├── tokens.css             # 设计令牌：色板/间距/圆角/阴影/字号/动效时长
│   ├── base.css               # reset + 排版 + 焦点可见性
│   ├── layout.css             # 应用外壳：左导航 / 顶栏 / 三栏栅格 / 移动端
│   ├── components.css         # 卡片/按钮/输入/chip/slider/segmented/tabs/badge
│   ├── panels.css             # 参数面板、Prompt Composer、场景浏览器、画廊
│   ├── animations.css         # skeleton / reveal / crossfade / pulse
│   └── themes.css             # dark（默认）+ light 覆盖
└── js/
    ├── core/
    │   ├── api.js             # ← 迁移自 js/api.js（不动协议）
    │   ├── store.js           # ← 迁移自 js/store.js（schema 升 v6）
    │   ├── router.js          # hash 路由：#/create #/gallery #/explore #/prompt #/studio #/stats
    │   ├── keyboard.js        # 统一快捷键注册表（含 Ctrl+K）
    │   ├── palette.js         # Command Palette
    │   └── bus.js             # 极简事件总线（页面 ↔ 组件解耦）
    ├── components/
    │   ├── prompt-composer.js # token 化 prompt 编辑 + 分组词条
    │   ├── scene-picker.js    # 场景大图卡 + 分类/搜索/悬停/随机/收藏
    │   ├── parameter-panel.js # 快速/画面/专业 三层参数
    │   ├── camera-panel.js    # 摄影师模式：焦段/景深/视角/构图
    │   ├── lighting-panel.js  # 光线实验室
    │   ├── image-card.js      # 统一图片卡（hover 操作层/元数据/收藏）
    │   ├── result-grid.js     # 批量选片 + 多选 + 批量下载/收藏
    │   ├── compare.js         # 对比（滑块 + A/B 并排 + 参数表）
    │   ├── lightbox.js        # ← 迁自 ui.js（含 v5 缩放能力，补平移记忆）
    │   ├── queue.js           # 队列/进度/阶段/ETA
    │   ├── detail.js          # 作品详情面板
    │   ├── seedlab.js         # Seed Lab：锁定/变异/相似构图
    │   └── toast.js           # 通知
    ├── pages/
    │   ├── create.js  ├── gallery.js  ├── explore.js
    │   ├── prompt.js  ├── studio.js   └── stats.js
    └── main.js                # 只做：初始化 + 注册路由 + 装配组件
```

**迁移原则**：`api.js` / `store.js` **原样搬迁不改协议**；`ui.js` 按组件拆解，拆完即删；`main.js` 只保留装配逻辑（目标 < 150 行）。

---

## 2. 交互契约（组件接口）

```js
// 事件总线（core/bus.js）—— 页面与组件只通过事件通信
bus.emit('job:submitted', job)
bus.emit('job:progress',  meta)
bus.emit('job:settled',   job)     // done | failed | expired
bus.emit('image:select',  { jobId, index })
bus.emit('prompt:compose', text)   // composer / 场景 / 相机 / 光线 都发这个
bus.emit('settings:change', patch)
bus.emit('view:change',   route)
```

| 组件 | init 签名 | 关键契约 |
|---|---|---|
| `prompt-composer` | `init({ mount, value, onChange })` | 输出**单一** `onChange(text)`；词条以 token 形式渲染，可单删 |
| `scene-picker` | `init({ mount, onPick, onFavorite })` | 只发 `prompt:compose`；支持 `limit`（首页 6 个）/ 全部模式 |
| `parameter-panel` | `init({ mount, settings, onChange })` | 三层折叠；所有值经 `config.clamp*` 后回写 |
| `camera-panel` / `lighting-panel` | `init({ mount, onChange })` | 输出**英文提示词片段数组**，由 composer 拼接 |
| `result-grid` | `init({ mount, items, onSelect, onBulk })` | `items = [{job,id,i,bytes}]`；支持多选与 Compare |
| `detail` | `init({ mount, job })` | 只读元数据 + 三个动作（再生成/复制/下载） |

---

## 3. 分波实施顺序（每波独立可交付、可回滚）

### 🚀 第一波：视觉骨架 + 核心 UX（本轮已开工）
1. `css/tokens.css` + `css/base.css` + `css/layout.css`：近黑底、紫/青点缀、左导航栏、顶部步骤条、三栏栅格、移动端底部 Tab。
2. `js/core/palette.js` + `keyboard.js`：**Ctrl+K 命令面板**（低成本、高感知）。
3. `js/components/prompt-composer.js`：token 化 + 「主体/环境/天气/光线/时间/季节/镜头/景深/构图/色调/风格」11 组词条。
4. 结果区升级：hover 操作层（放大/下载/收藏）、`1 / N` 指示、crossfade。
5. **验收**：桌面三栏 + 移动底部 Tab 均可用；`Ctrl+K` 可执行 ≥ 10 条命令；composer 产出与原 prompt 兼容（后端零改动）。

### 🔥 第二波：创作能力
6. `camera-panel`（16/24/35/50/85/200mm、景深、视角、构图）、`lighting-panel`（9 种光）、`seedlab`（锁定/变异）。
7. Surprise Me（场景 × 天气 × 时间 × 镜头 × 风格 交叉组合）+「换一个但保持构图」。
8. A/B Prompt：两条 prompt × 各 2 张 = 4 图，自动进 Compare。
9. **验收**：任一组预设点击后 prompt 变化可解释；A/B 结果可并排看参数表。

### 💎 第三波：专业工作流
10. `router.js` + 六个页面（`/create /explore /gallery /prompt /studio /stats`）。
11. Workflow 步骤条：灵感 → 6 步草图 → 选构图 → 24 步精修 → 1024 → 4K（每步可回退，参数沿链传递）。
12. `detail.js` 作品详情页 + 资产管理（筛选：Prompt/Seed/Sampler/分辨率/日期/收藏；排序：时间/耗时/分辨率）。
13. Remix：基于某张图 + 保留项（构图/色彩/天气/时间/风格）→ 走已有 `init_image` 通道。
14. **验收**：每个页面可直达（hash 可分享）；Workflow 全链路生成 1 次通过；Remix 复用已有 img2img 协议。

### 🧠 第四波：智能化（可选）
15. Prompt 自动增强/推荐、相似场景、相似图片、自动标签、自动标题、生成关系图。
16. **注意**：全部在浏览器端可离线实现（纯规则/嵌入），不引入新的后端依赖。

---

## 4. 风险与约束

| 风险 | 处理 |
|---|---|
| 一次性重写导致线上不可用 | **渐进增强**：新外壳与旧 `index.html` 并存，路由默认落到 `#/create`；每波上线后可回滚到上一部署 |
| `main.js` 拆分期间功能断裂 | 先搬 `api/store`（零改动），再按组件抽离并在每步跑 `tools/check_frontend.py` |
| 移动端手势与灯箱缩放冲突 | 灯箱已 `touch-action:none`；底部 Tab 与画布滑动手势分区（画布区不接管横向滑动） |
| 后端字段新增不同步 | 遵守 `AGENTS.md`：`server.py` / `server_cloud.py` / `proxy.js` / 前端 四处同改 |
| 行尾与脱敏回归 | 每波结束跑 `tools/normalize_eol.py` + `tools/check_paths.py` + `research/final_check.py` |

---

## 5. 每波交付前必须跑的检查

```powershell
python tools/check_frontend.py     # id/模块引用一致性
python tools/check_paths.py        # 无个人绝对路径
python tools/normalize_eol.py      # 行尾（应 0 变化）
python research/final_check.py     # 无敏感项
node --check site/js/<每个模块>.js
```

---

## 6. 本方案对参考图视觉语言的落点

| 参考图特征 | V6 落点 |
|---|---|
| 左侧图标导航栏 | `layout.css` 的 `.app-rail`（桌面固定 72px，移动端底部 Tab） |
| 顶部工作流步骤条 | `.step-strip`（Draft → Refine → Edit → Export 对应本项目：草图 → 精修 → 高清 → 导出） |
| 中央画布 + 底部历史缩略条 | `.canvas` + `.film-strip`（复用现有 history + `viewList()`） |
| 右侧紧凑参数面板 | `parameter-panel`（分段控件 + 滑杆 + 色板行 + 小数字输入） |
| Prompt token 化 | `prompt-composer`（chip 可删、分组插入） |
| 近黑底 + 霓虹点缀 | `tokens.css`（`--bg:#07070c`、`--accent:#7c5cff`、`--accent2:#22d3ee`） |
| 资产网格 + hover 操作层 | `image-card` + `result-grid` |
