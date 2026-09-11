# Landscape·Art — 风景图片生成模型

在 **RTX 3060 6GB** 上用 **Stable Diffusion 1.5 + LoRA** 从 1664 张专业风景摄影微调出的
**文生图（text-to-image）** 模型；输入一句风格描述 prompt（留空即随机），输出一张**新的、区别于训练/测试集**的风景图。

支持**本地 Gradio 应用**与**线上 Web（Netlify 前端 + 你本机 GPU 后端）**两种用法。

> **当前最新：V4 / V4-640**（部署基线）+ **V5 训练中**（min-SNR-γ / Prodigy / DoRA，见 `docs/TRAINING.md`）。
> 推荐权重：`models/v4_640/adapter_best`（640 原生）+ `models/v4_640/adapter_best_text_encoder.pt`（微调 CLIP）。
> 二者均由 **EMA 最优**导出。

> 🚀 **新协作者请先看 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)**：从 clone 到跑通训练/服务/部署的完整步骤，
> 以及「哪些内容不入库、如何一条命令重建」（权重、数据集、缓存）。一分钟自检：
> `python tools/fetch_models.py all` → `pwsh -NoProfile -File tools/dev.ps1 check`。

---

## 目录

1. [项目速览与关键指标](#1-项目速览与关键指标)
2. [快速开始](#2-快速开始)
3. [系统架构](#3-系统架构)
4. [模型演进 V1 → V4](#4-模型演进-v1--v4)
5. [V4 详解（当前最新）](#5-v4-详解当前最新)
6. [推理：质量 / 速度 / 清晰度](#6-推理质量--速度--清晰度)
7. [前端 UX 功能](#7-前端-ux-功能)
8. [线上部署](#8-线上部署)
9. [工程规范化与环境变量](#9-工程规范化与环境变量)
10. [命令行参考](#10-命令行参考)
11. [目录结构与文件清单](#11-目录结构与文件清单)
12. [质量验证与压力测试](#12-质量验证与压力测试)
13. [常见问题与排错](#13-常见问题与排错)
14. [已知限制与后续路线](#14-已知限制与后续路线)

---

## 1. 项目速览与关键指标

| 项目 | 值 |
|---|---|
| 基座模型 | `SG161222/Realistic_Vision_V6.0_B1_noVAE`（摄影向 SD1.5 微调）→ 已转为本地 `models/base_rv6/`（safetensors，2.1 GB） |
| 微调方式 | LoRA（V4：rank 64 / alpha 64 / dropout 0.1）+ **CLIP 文本编码低 LR 微调** |
| 数据集 | 1664 张专业风景摄影（原图约 52 GB），**train 1520 / test 144** |
| V4 最优验证损失 | **0.1188**（step 4800，EMA 权重） |
| V4-640 最优验证损失 | **0.1262**（step 400，EMA） |
| 512×512 出图速度 | 标准 **24 步 ≈ 6 s**；⚡极速 6 步（LCM）**≈ 4 s**；500 步 ≈ 200 s（不推荐） |
| 高清输出 | 两段式 1024；ultimate upscale **2048 ≈ 63 s**、**4096 ≈ 30 s**（快速路径） |
| 显存 | 512：2.8 GB / 768：3.7 GB / 1024：峰值 6.7 GB（临界）；服务端用 `cpu_offload` 稳跑 6 GB 共享卡 |
| 稳定性 | 七维压测全部 PASS（见 §12） |

---

## 2. 快速开始

```powershell
$PY = "<PYTHON>"
$ROOT = "<你的项目根目录>"

# ── 方式 A：本地 Gradio 应用（最简单）──
$env:LORA_ADAPTER = "$ROOT/models/v4_640/adapter_best"
& $PY "$ROOT/app.py"          # 浏览器打开 http://127.0.0.1:7860

# ── 方式 B：命令行单张生成 ──
& $PY "$ROOT/inference.py" --base "$ROOT/models/base_rv6" `
    --lora "$ROOT/models/v4_640/adapter_best" `
    --prompt "a misty alpine lake at dawn, dramatic clouds" `
    --width 768 --height 768 --steps 24 --scheduler dpmpp2m_karras

# ── 方式 C：线上 Web（需先起后端 + 隧道，详见 README_NETLIFY.md）──
$env:API_TOKEN   = "<你的强随机密钥>"
$env:LORA_ADAPTER = "$ROOT/models/v4_640/adapter_best"
& $PY "$ROOT/server.py"       # http://127.0.0.1:8001
```

---

## 3. 系统架构

```
┌─────────────────────────── 方式 A：本地 ───────────────────────────┐
│  浏览器 ──► app.py (Gradio, 127.0.0.1:7860) ──► 本机 GPU 模型      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────── 方式 B：线上 ───────────────────────────┐
│  浏览器                                                             │
│     │  ① 访问静态页（含 28 场景 / 档位 / 画廊 / 调试面板）           │
│     ▼                                                               │
│  Netlify 静态站点  https://<your-site>.netlify.app           │
│     │  ② 同源调用 /.netlify/functions/proxy?op=…                    │
│     ▼                                                               │
│  Netlify Function 代理（服务器端注入 X-API-Key，非浏览器 UA）        │
│     │  ③ 转发到 BACKEND_URL（隧道公网地址）                          │
│     ▼                                                               │
│  ngrok / Cloudflare 隧道  https://xxx → http://localhost:8001       │
│     │                                                               │
│     ▼                                                               │
│  server.py（FastAPI：鉴权 / 队列 / 限流 / 轮询 / 结果落盘）           │
│     │                                                               │
│     ▼                                                               │
│  app.py 推理层（SD1.5 + V4 LoRA + 微调 CLIP + 两段式 / ultimate）    │
└─────────────────────────────────────────────────────────────────────┘
```

**为什么需要代理层**：Netlify 没有 GPU，模型必须跑在你本机；而隧道地址与密钥不应暴露给浏览器，
因此用 Function 做一层**同源代理**（顺带绕开 ngrok 免费版的浏览器告知页）。

| 组件 | 文件 | 职责 |
|---|---|---|
| 推理层 | `app.py` | 模型加载（两种模式）、prompt 组装、生成、超分、img2img |
| 后端服务 | `server.py` | HTTP API、鉴权、单 worker 串行、队列/限流/超时、结果落盘与分块回传 |
| 云端适配器 | `server_cloud.py` | 同一套 API，改由 Replicate 云端 GPU 执行（可选） |
| 前端 | `site/index.html` | 交互界面、持久化、画廊、调试面板 |
| 代理 | `netlify/functions/proxy.js` | 同源转发 health/generate/job/image/chunk |
| 高清流水线 | `enhance.py` | 超分 → 分块 img2img 重绘 → 锐化 |

---

## 4. 模型演进 V1 → V4

| 维度 | V1 | V2 | V3 | **V4（当前）** |
|---|---|---|---|---|
| 基座 | `stable-diffusion-v1-5` | `Realistic_Vision_V6.0_B1` | 同 V2 | 同 V2（本地 safetensors） |
| Caption | 文件名模板 | **BLIP 自然语言** | 同 V2 | 同 V2（`dataset1024` 复用） |
| LoRA rank | 16 | 32 | 32 + dropout 0.05 | **64 + dropout 0.1 + weight decay** |
| 数据 | 512 中心裁剪 | 同 V1 | 同 V1 | **1024 源增强 + 每图 3 随机裁剪 → 4560 latent** |
| 增强 | 无 | latent 水平翻转 | 翻转 / 缩放 / 平移 | 随机缩放取景 + 翻转（**源级**） |
| 正则 | 无 | 无 | 无 | **caption 随机丢弃 0.05** |
| 权重平均 | 无 | 无 | 无 | **EMA（0.9995）**，验证与导出均用 EMA |
| 文本编码 | 冻结 | 冻结 | 冻结 | **CLIP 低 LR(1e-5) 微调** |
| 调度 | 常数 | 余弦 | 余弦 | 长余弦 + 验证最优 + 断点续训 |
| 最优 val | — | 0.1157 @1500 | 0.1176 @1000 | **0.1188 @4800**（V4-640：0.1262 @400） |

**核心结论（贯穿全项目）**：rank 与数据量一旦受限，**约 1000~1500 步后就过拟合**（val 先降后升）。
因此项目一律采用「**长训练 + 验证集选优 + 导出 EMA 最优**」，而不是训到底。
V4 通过「源增强把有效数据放大 3×」+「EMA」+「CLIP 微调」把这条曲线显著推后，并让最终画质上一个台阶。

---

## 5. V4 详解（部署基线；训练脚本已归档）

> **归档说明**：V1–V4 的训练/预计算脚本已移入 `archive/v1_v2/`、`archive/v3/`、`archive/v4/`
> （见 `archive/README.md`）。本章保留历史记录，正文中的 `train_v4.py` / `precompute_v4.py`
> 等均指归档后的同名文件。**当前训练路线见 `docs/TRAINING.md`（V5：min-SNR-γ + Prodigy + DoRA）。**
> V4 的产物 `models/v4_640/adapter_best` 仍是线上默认权重。

### 5.1 数据集（源增强）
- `prepare_v4_data.py`：从 31 个 zip 按 **1024** 重新提取（短边缩放 + 中心裁剪），输出 `dataset1024/{train,test}`。
- **复用已有 BLIP 标题**（按 entry 匹配 `dataset/manifest.json`），并保持**与旧版完全相同的 train/test 划分**（md5 决定性划分）。
- 规模：train **1520** / test **144**；耗时 ≈ 336 s（8 进程）。

### 5.2 数据倍增（关键抗过拟合手段）
- `precompute_v4.py`：对每张 1024 源图生成 **3 个随机裁剪**（scale 0.55–1.0、随机比例、随机翻转），
  各编码为 512 latent → 训练 latent 从 1520 → **4560 条**，且每条都有**真实的取景/缩放变化**。
- 输出 `dataset1024/cache_v4.pt`（含训练 latent、逐条 caption、测试集中心裁剪 latent）。
- 640 版本用同一脚本 `--res 640` 生成 `cache_v4_640.pt`。

### 5.3 训练配方（`train_v4.py` / `config_v4.cfg`）
| 项 | 值 |
|---|---|
| LoRA | rank **64** / alpha 64 / dropout **0.1**，目标 `to_q,to_k,to_v,to_out.0` |
| 优化器 | AdamW，lr 1e-4，weight_decay **0.01**，CLIP 文本编码 lr **1e-5** |
| batch | batch 1 × accum 4（等效 4；6 GB 卡上的省显存选择） |
| EMA | decay **0.9995**（每步更新；**验证与导出都用 EMA 权重**） |
| caption 丢弃 | **0.05**（提升 prompt 跟随与泛化） |
| 调度 | 长余弦，warmup 1000 步，总计 15000 步预算（按验证最优提前收敛使用） |
| 验证 | 每 200 步在 144 张测试集评估；**在线编码 caption**（因 CLIP 可训练，不能用预计算文本） |
| 产物 | `adapter_best`（EMA 最优）、`adapter`（最终）、`*_text_encoder.pt`、`checkpoints/step_*.pt`、`metrics.csv`、`params.json` |
| 续训 | `--resume latest`（第 5.5 节记录了踩坑与修复） |

### 5.4 640 高清微调
- `config_v4_640.cfg`：从 **V4 best** 初始化（`--init_lora models/v4_lora/adapter_best`），
  640 分辨率、lr 3e-5、800 步、warmup 50。
- 结果：best val **0.1262 @400**（640 尺度与 512 不可直接比较），产出 `models/v4_640/`。
- 线上默认使用该权重。

### 5.5 训练期踩坑与修复（真实记录）
1. **累积梯度死循环**：`(gstep+1) % accum == 0` 而 `gstep` 只在该分支自增 → `accum>1` 时永不成立，
   表现为「GPU 满载但永不进入下一步」。修复：引入独立 `micro` 计数器。
2. **续训 EMA 设备不一致**：checkpoint 里的 EMA 张量被 `map_location="cpu"` 读回后仍是 CPU，
   与 GPU 参数做 `mul_/add_` 直接崩。修复：读取后 `copy_` 回参数设备。
3. **src 增强后不能用预计算文本**：一旦微调 CLIP，缓存的文本嵌入即失效 → 训练/验证均改为**在线编码 caption**。

---

## 6. 推理：质量 / 速度 / 清晰度

### 6.1 两种模式（`app.py`）
| 模式 | 组成 | 适用 |
|---|---|---|
| **质量模式** | V4 LoRA(peft) + `DPM++ 2M Karras` | 默认，24 步最佳性价比 |
| **极速模式** | 先把 V4 LoRA **合并进权重** → 叠加 **LCM-LoRA**（UNet-only，diffusers 自动 kohya 转换）+ `LCMScheduler` | 快速试稿，4–6 步 |

两种模式**同一时间只驻留一个管道**（切换时释放旧的并 `empty_cache()`），以免 6 GB 卡内存翻倍。
服务端对极速模式**强制夹取参数**：步数 2–12、CFG 1.0–2.5，避免参数不当把画质搞崩。

> ⚠️ 极速模式下 CFG≈1.5，**负向 prompt 基本失效**，且画风更浓烈（历史对比图见 §12.3 的生成命令）。

### 6.2 步数基准（512×512，同 prompt 同 seed，`bench_steps.py`）
| 档位 | 耗时 | 说明 |
|---|---|---|
| 500 步 | **~200 s** | ⚠️ 远超收敛点，**又慢又易过饱和**，不要用 |
| 精细 40 步 | ~15 s | 基线 |
| **标准 24 步** | **~6 s** | **与 40 步质量基本一致（默认）** |
| ⚡极速 8 / 6 / 4 步 | 11.5 s（含切换重建）/ **4.1 s** / **3.0 s** | LCM |

**结论：24 步 vs 500 步 ≈ 33× 提速且质量不降。**

### 6.3 高清晰度（`enhance.py`，ultimate upscale）
SD1.5 原生 512，直接生成更大分辨率会结构崩坏，正确做法是「放大 + 高分辨率重绘」：

```
512/640 生成 ──► ① 神经超分(1 轮) ──► ② 分块(512) img2img 低强度重绘(余弦羽化融合) ──► ③ 轻锐化 ──► 2K/3K/4K
```

- 超分模型可选：**`4x-UltraSharp`**（默认，`models/sr/4x-UltraSharp.pth`）或 `Real-ESRGAN`（`sr_model` 参数）。
- **神经超分只跑一轮**，其余用 LANCZOS 补齐（多轮超分极慢，实测是主要瓶颈）。
- 分块让每块都落在 512 原生域内 → **6 GB 显存可安全产出 2K/3K/4K**。
- 参数：`enhance`(目标边长) / `sr_model` / `enhance_strength`(重绘强度，0=仅超分) / `enhance_steps`。
- 实测：2048 ≈ 63 s；**4096 快速路径（strength=0）≈ 30 s**。

### 6.4 大图回传（突破代理 6 MB 上限）
- 后端 `GET /chunk/{id}?off=&length=`（字节区间），`/jobs/{id}` 返回 `bytes` 总大小。
- 代理 `?op=chunk&id=&off=&len=` 分块转发。
- 前端 `loadImageUrl()`：`bytes > 4 MB` 时自动**分块下载并拼装 Blob**（显示与下载均可用）。
- 已验证：分块拼装字节数与总大小**完全一致**。

---

## 7. 前端 UX 功能

| 功能 | 说明 |
|---|---|
| **刷新不丢** | 任务（job_id/seed/状态/大小）、界面设置、历史画廊全部存 `localStorage`（键 `lsart_v1`）；刷新后自动重连进行中的任务并继续轮询 |
| **档位选择** | 质量/速度：⚡极速 6 步 / ⚖️标准 24 步 / ✨精细 40 步 / 自定义；结果信息显示「seed · 步数 · 尺寸」 |
| **清晰度** | 标准 512 / 高清 1024（两段式）/ 超清 2048 / 极清 3072 / 4K 4096（分块下载） |
| **超分模型** | 4x-UltraSharp（默认）/ Real-ESRGAN |
| **预制场景 28 个** | 分 4 类：自然奇观 8 / 海岸与水 6 / 四季 6 / 光效与天象 8；悬停显示英文 prompt；随机池 20 条 |
| **历史画廊** | 状态徽标（🕒 排队中 / ⏳ 生成中 / ❌ 失败 / ⌛ 已过期）、点击放大、悬停下载、清空历史 |
| **过期自愈** | 后端只保留最近 500 个任务，被清理的结果标记「⌛ 已过期」而非坏图 |
| **调试面板** | 健康指标（排队/进行中/已完成/累计）、每次请求与响应的完整 JSON、耗时 |
| **其它** | 复制 seed / 复制 prompt、`Ctrl+Enter` 生成、多张出图（1/4 网格）、负向 prompt 编辑 |

---

## 8. 线上部署

完整步骤、环境变量、隧道与排错手册见 **[`README_NETLIFY.md`](README_NETLIFY.md)**。核心三步：

```powershell
# ① 本机后端
$env:API_TOKEN='<你的强随机密钥>'; $env:LORA_ADAPTER='$ROOT/models/v4_640/adapter_best'
& "<PYTHON>" $ROOT\server.py

# ② 隧道（必须在当前终端清代理，否则免费版报 ERR_NGROK_9009）
Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY,Env:http_proxy,Env:https_proxy,Env:all_proxy -ErrorAction SilentlyContinue
ngrok http 8001

# ③ Netlify 站点环境变量（只需一次；隧道换域名时才需要更新）
BACKEND_URL = https://<你的隧道域名>
API_TOKEN   = <你的强随机密钥>
```

---

## 9. 工程规范化与环境变量

### 9.1 告警清零
后端启动/首次加载原本会刷 6 类告警，现已全部处理（实测启动日志已无告警）：

| 原告警 | 处理方式 |
|---|---|
| `on_event is deprecated` | FastAPI 迁移到 **lifespan** 上下文管理器 |
| `torch_dtype is deprecated` | 全部改用 `dtype=` |
| `unauthenticated requests to the HF Hub` | 设 `HF_HUB_OFFLINE=1`（权重已本地化） |
| SSL 重试 `UNEXPECTED_EOF_WHILE_READING` | 同上，**不再发网络 HEAD 请求**，加载更快 |
| `Defaulting to unsafe serialization`（`.bin`/pickle） | `normalize_base.py` 转换出本地 **safetensors** 目录 `models/base_rv6/`（2.1 GB，9.3 s） |
| `CLIPFeatureExtractor deprecated` | 转换时从 `model_index.json` **移除 `feature_extractor` / `safety_checker`**（管道组件 6 → 5） |

### 9.2 环境变量总表
**推理/服务（`app.py` / `server.py`）**

| 变量 | 默认 | 说明 |
|---|---|---|
| `API_TOKEN` | `<你的强随机密钥>` | 请求头 `X-API-Key` 校验；**线上务必改强密钥** |
| `LORA_ADAPTER` | `models/v3_640/adapter_best` | 指向要用的 adapter 目录；**线上建议 `models/v4_640/adapter_best`** |
| `BASE_MODEL` | 自动：本地 `models/base_rv6` → 回退 HF 仓库 id | 基础模型路径 |
| `LCM_LORA` | `models/lcm_lora` | 极速模式的 LCM-LoRA 目录 |
| `HOST` / `PORT` | `127.0.0.1` / `8001` | 监听地址与端口 |
| `LOG_LEVEL` | `warning` | 设 `info` 可在后端窗口看到每条请求日志 |
| `PROGRESS_BAR` | `1` | 设 `0/false` 完全静音 tqdm 进度条 |
| `MAX_QUEUE` | `20` | 排队上限，超出返回 429 |
| `RATE_PER_MIN` | `8` | 每 IP 每分钟次数，超出 429 |
| `JOB_TIMEOUT` | `900` | 单任务超时（秒）；4K 分块重绘较慢，故放宽 |
| `MAX_JOBS` | `500` | 内存中保留的最近任务数，超出则删除最旧的**及其结果文件** |
| `RESULTS_DIR` | `$ROOT/results` | 结果落盘目录 |

**云端适配器（`server_cloud.py`）**：`REPLICATE_API_TOKEN`、`REPLICATE_MODEL`（如 `stability-ai/sdxl`）、`API_TOKEN`、`PORT`。

**Netlify 站点**：`BACKEND_URL`、`API_TOKEN`（由 Function 代理在服务器端使用，浏览器不可见）。

### 9.3 显存策略
- 服务端固定 `enable_model_cpu_offload()`：权重常驻内存、按需上卡，与其它程序共享 6 GB 也不易 OOM。
- 训练用 `fp16 + UNet 梯度检查点 + 预计算 latent`，512 训练峰值约 2.5–3.3 GB。
- 1024 原生直出峰值 **6.7 GB（临界）**，因此大图一律走 **两段式 / ultimate upscale**。

---

## 10. 命令行参考

```powershell
$PY="<PYTHON>"; $R="<你的项目根目录>"

# ── 数据（当前工具链在 tools/dataset/）──
& $PY $R\tools\dataset\prepare_v4_data.py --size 1024                  # 由源图构建 dataset1024（1024 方形）
& $PY $R\tools\dataset\caption_images.py --data-dir $R\dataset1024     # BLIP 标题（旧方案，保留回退）
& $PY $R\recaption.py --manifest $R\dataset1024\manifest.json          # 现代 VLM 重写 caption（推荐）

# ── 预计算与训练（V5，根目录）──
& $PY $R\precompute_v5.py --res 640 --out $R\dataset1024\cache_v6_640.pt        # SD1.5 latent+文本缓存
& $PY $R\train_v5.py --config $R\config_v5.cfg                                  # V5 主训练（min-SNR/Prodigy/EMA）
& $PY $R\train_v5.py --config $R\config_v5.cfg --resume latest --max_train_steps 6000
& $PY $R\precompute_v5.py --base SG161222/RealVisXL_V5.0 --arch sdxl --res 1024 `
      --out $R\dataset1024\cache_v5_sdxl1024.pt                                 # SDXL 缓存（双文本编码器）
& $PY $R\train_v5.py --config $R\config_v5_sdxl.cfg                             # SDXL QLoRA（int8 基座）
& $PY $R\tools\probe_sdxl_train.py                                             # 6GB 可行性探针
& $PY $R\tools\compare_adapters.py --adapters "V4:$R\models\v4_640\adapter_best" "V5:$R\models\v5_lora\adapter_best"
& $PY $R\tools\test_adapter_load.py $R\models\v5_lora\adapter_best              # 部署门禁：能否被推理栈加载

# ── 推理与产物（历史脚本见 archive/）──
& $PY $R\archive\v1_v2\inference.py --base $R\models\base_rv6 --lora $R\models\v4_640\adapter_best --prompt "..." --width 768 --height 768 --steps 24
& $PY $R\archive\v1_v2\generate_v2_gallery.py --adapter $R\models\v4_640\adapter_best --prompts-file $R\showcase_prompts.txt
& $PY $R\archive\v1_v2\bench_steps.py ; & $PY $R\archive\v1_v2\stress_test.py --runs 30

# ── 素材与工程工具 ──
& $PY $R\tools\assets\make_makoto_icons.py ; & $PY $R\tools\assets\make_makoto_preview.py

# ── 服务 ──
& $PY $R\app.py                                                       # Gradio :7860
& $PY $R\server.py                                                    # FastAPI :8001
& $PY $R\server_cloud.py                                              # 云端适配器（需 Replicate 令牌）
```

---

## 11. 目录结构与文件清单

```
landscape_gen/
├─ app.py                 推理层（质量/极速双模式、prompt 组装、高清、img2img）
├─ server.py              FastAPI 后端（鉴权/队列/限流/轮询/落盘/分块）
├─ server_cloud.py        云端 GPU 适配器（Replicate，同一 API）
├─ enhance.py             ultimate upscale（超分→分块重绘→锐化，多超分模型）
├─ config.py              共享路径/环境变量/安全边界
├─ runtime.py             GPU 管线生命周期与显存治理
│
├─ train_v5.py            V5 训练器（min-SNR-γ 加权 / Prodigy / DoRA / QLoRA / EMA / 验证选优 / 续训）
├─ precompute_v5.py       latent + 文本嵌入缓存（SD1.5 与 SDXL 双架构）
├─ recaption.py           现代 VLM（Qwen2-VL）重写 caption
├─ config_v5.cfg          V5 SD1.5 配方
├─ config_v5_sdxl.cfg     V5 SDXL QLoRA 配方
│
├─ tools/                 工程工具（check_* / dev.ps1 / 训练评测 / 素材）
│   ├─ dataset/           数据集构建与打标（prepare_v4_data.py、caption_images.py）
│   ├─ assets/            站点素材生成（makoto / pelican / goose 图标与预览）
│   ├─ dev.ps1            统一开发入口（check / eol / security / bench / verify / start / tunnel / deploy）
│   ├─ check_frontend.py  前端 id/模块/CSS 回归门禁
│   ├─ smoke_frontend.mjs jsdom 真实 DOM 冒烟（21 项）
│   ├─ test_proxy_guard.mjs 代理来源守卫（跨站 POST 必须 403）
│   ├─ compare_adapters.py  多 adapter 客观对比 + 对比拼版
│   ├─ probe_sdxl_train.py  6GB 能否训 SDXL 的显存探针
│   ├─ test_adapter_load.py 部署门禁：adapter 能否被推理栈加载
│   └─ fetch_models.py      下载基座（sdxl 等）到项目 HF 缓存
│
├─ archive/               旧版本代码归档（按版本号）
│   ├─ v1_v2/             diffusers 官方训练脚本、单图超分、诊断压测、v2 配方
│   ├─ v3/                V3 训练器与配方
│   └─ v4/                V4 训练器/预计算/样张脚本与配方（产物仍是线上权重）
│
├─ site/                  Netlify 静态前端（css/ js/ img/ sw.js）
├─ netlify/functions/     同源代理
├─ docs/                  TRAINING.md（训练方法与实测）/ SCALE.md（大用量韧性）/ V6_PLAN.md（前端计划）
├─ dataset1024/           1024 数据集 + manifest + cache_v4.pt / cache_v4_640.pt（旧 512 版 dataset/ 已在清理中删除）
├─ models/
│   ├─ base_rv6/          本地 safetensors 基础模型（SD1.5，线上使用）
│   ├─ v4_640/            线上默认 adapter（adapter_best）
│   ├─ v5_lora/           V5 训练产物
│   ├─ lcm_lora/          LCM-LoRA（极速模式）
│   ├─ sr/                RealESRGAN_x4plus.pth / 4x-UltraSharp.pth
│   ├─ hf_cache/          HF 缓存（含 SDXL RealVisXL V5.0）
│   └─ （历史版本权重已清理：释放约 50 GB，见下方 §12.3 说明）
├─ results/               线上后端结果落盘（自动保留最近 500 个）
├─ netlify.toml           publish=site, functions=netlify/functions
├─ research/              可复现检查脚本（final_check.py / scan_secrets.py / boot_verify.py 等）
├─ README.md              本文
└─ README_NETLIFY.md      部署手册
```

---

## 12. 质量验证与压力测试

### 12.1 七维压测（`stress_test.py`，PASS）
| 维度 | 结果 |
|---|---|
| 确定性（同 seed） | MSE 0.0003 → PASS |
| 多样性（同 prompt 多 seed） | min MSE 0.103 → PASS |
| 边界 prompt（空/中文/超长/仅负向） | 全部通过（**空 prompt 必须用 `""`，不能用 `None`**） |
| 分辨率 | 512：2.8 GB / 7.7 s；768：3.7 GB / 22.6 s；1024：峰值 **6.7 GB（临界）** |
| 长程稳定（30 次连续） | 7.71 s ± 0.02 s，显存零增长，无退化 → PASS |
| 内存泄漏 | 0.0 GB 漂移 → PASS |
| 崩溃/异常 | 无 → PASS |

### 12.2 「生成图 ≠ 训练/测试图」保证
- 生成图是模型新合成产物，天然不属于数据集。
- `inference.py` 加载全部 1664 张图的**感知哈希**，与生成图比对海明距离；
  过近（默认阈值 6）会自动重新采样，日志输出 `min_hamming`。

### 12.3 主要验证产物索引

> ⚠️ **2026-09 仓库清理**：`outputs/`（187 MB 历史样张）、`dataset/`（627 MB，与 `dataset1024/` 重复的 512 副本）、
> 以及 `models/` 下的历史版本权重与训练检查点（49.5 GB）已删除，共释放约 **50 GB**。
> 下表索引的是这些产物的**生成方式**——它们是可再生的，需要时按脚本重新生成即可。

| 产物 | 生成命令 |
|---|---|
| V4 / V4-640 样张 | `archive/v1_v2/gen_v4.py`、`archive/v4/make_v4_compare.py`（需先有 `outputs/` 目录） |
| 步数/模式基准 | `archive/v1_v2/bench_steps.py` |
| 高清流水线对比（512 / ESRGAN / ultimate） | `archive/v1_v2/test_enhance.py` |
| 风格集 / 展示集 | `archive/v1_v2/generate_v2_gallery.py --prompts-file showcase_prompts.txt` |
| 压测报告 | `archive/v1_v2/stress_test.py` |
| 图标与动效预览 | `tools/assets/make_makoto_icons.py`、`tools/assets/make_makoto_preview.py` |
| **V4↔V5 出图对比（当前）** | `tools/compare_adapters.py --adapters "V4:models/v4_640/adapter_best" "V5:models/v5_lora/adapter_best"` |
| 项目报告（v4 时期快照） | `archive/v4/report/PROJECT_REPORT.html`（其中的样张图已随 `outputs/` 删除） |

---

## 13. 常见问题与排错

| 症状 | 原因 | 解决 |
|---|---|---|
| 页面「后端离线」 | 隧道没起 / `BACKEND_URL` 与实际域名不符 | 起隧道；用 `/api/tunnels` 取实际地址并更新 Netlify 环境变量 |
| `ERR_NGROK_9009` | ngrok 免费版不能走 HTTP 代理 | 在启动 ngrok 的**终端内**先清 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` |
| 页面取图 502 且 JSON 提示 `ResponseSizeTooLarge` | Netlify Function 单次响应约 6 MB | 已实现分块回传；确认前端走 `op=chunk` |
| 生成很慢（数分钟一张） | 步数被设得过大（如 500） | 档位改「标准 24 步」；结果信息会显示步数 |
| `CUDA out of memory` | 与其它程序共享 6 GB | 已用 `cpu_offload`；关掉其它占卡程序；避免同时跑两个模型 |
| 换模型不生效 | 只改了 `LORA_ADAPTER` 但没重启 | **必须重启** `server.py` / `app.py`（模型懒加载） |
| 历史画廊出现「⌛ 已过期」 | 后端重启或结果已被 500 条上限清理 | 正常现象；重新生成即可 |
| 首次生成特别慢 | 模型懒加载 + 模式切换重建管道 | 首次约 30 s，之后 512 标准档 ≈ 6 s |
| 极速模式负向 prompt 无效果 | LCM 低 CFG 下引导失效 | 属预期；需要负向控制就用质量模式 |

---

## 14. 已知限制与后续路线

**限制**
- 6 GB 显存 + SD1.5 架构是天花板：细节/语义理解无法与 SDXL/FLUX 相比。
- 数据集仅 1520 张训练图，长训必过拟合 → 只能靠「源增强 + EMA + 验证选优」缓解。
- 线上可用性依赖本机在线 + 隧道（免费 ngrok 域名会变）。
- `server_cloud.py` 需要你自己的 Replicate 令牌，本项目未实测其真实出图。

**后续可做**
1. **固定域名**：Cloudflare 命名隧道 + 自有域名，免去每次改 `BACKEND_URL`。
2. **访问口令**：给页面加登录，防止他人消耗本机算力。
3. **官方风格 LoRA**：为油画/水墨/赛博朋克各训一个小 LoRA（需收集风格参考图）。
4. **上云 SDXL/FLUX**：真正跨代提升，需按量付费的云端 GPU。
5. **ControlNet**：构图/边缘控制，需额外模型与更多显存。
# imagenerate_3060



## 15. 本次工程升级（2026）

- **速度**：标准档固定在 24 步 DPM++ 2M Karras；精细档使用 DPM++ 2M SDE Karras；极速档使用 LCM 6 步，另支持 TCD（安装对应权重后启用）。后端最终将普通请求夹在 4–60 步、极速请求夹在 2–12 步，500 步会被安全截断。
- **显存**：质量/极速管线不会同时驻留；切换时释放 hooks、Python GC、CUDA cache 和 IPC cache；VAE slicing/tiling 与 PyTorch SDPA 路径默认开启。
- **传输**：结果默认 WebP（`OUT_FORMAT=webp`、`OUT_QUALITY=88`），基础图完成即生成 WebP 预览；大图通过代理分块并行下载。旧客户端仍可使用 `/result/{id}` 和 `/chunk/{id}`。
- **结果回收**：`MAX_JOBS` 控制任务数量，`RESULTS_QUOTA_MB` 控制结果目录磁盘配额；超额时按最旧完成文件 LRU 回收，并向前端报告「已过期」。
- **环境变量新增**：`TCD_LORA`、`OUT_FORMAT`、`OUT_QUALITY`、`PREVIEW_MAX`、`RESULTS_QUOTA_MB`、`MAX_STEPS`、`MAX_FAST_STEPS`、`USE_XFORMERS`。
- **推荐配方**：先用 LCM/TCD 6 步确认构图，再用标准 24 步出片；需要更细微纹理时用 SDE 32 步。不要用 500 步作为“高质量”档位。

## 16. v5：批内多图 + 结果缓存 + 前端工作台扩展

### 16.1 服务端批量（减少往返）
`GenerateReq.batch`（1–4）：**一个队列位、一次轮询产出 N 张**（种子 `seed..seed+N-1`）。
此前「一次出 4 张」= 4 次提交 + 4 个队列位 + 4 条轮询流；现在为 1 次。
- 取图带索引：`GET /result/{id}?i=k`、`GET /chunk/{id}?i=k&off=&len=`（不传 `i` = 第 0 张，**旧客户端不变**）。
- `/jobs/{id}` 新增字段：`count`、`seeds`、`files`、`bytes_each`、`batch_index`、`cached`。
- 代理透传 `i`，并回传 `X-Job-Count` / `X-Job-Index`；`server_cloud.py` 同步接受同名字段（云端按单图处理）。

### 16.2 结果缓存（同参数秒回）
当 `seed>=0 && batch==1 && 无参考图` 时按**参数指纹**缓存（prompt/style/res/aspect/steps/cfg/seed/neg/sampler/fast/highres/enhance/sr_model/enhance_strength/enhance_steps/upscale/strength）。
命中时 `/generate` 立即返回 `{status:"done",cached:true}` —— **不排队、不动 GPU、无需轮询**。
- 缓存位于 `results/cache/`（独立目录，**不受任务裁剪影响**），命中时用**硬链接**给任务自己的路径，零额外磁盘占用。
- `CACHE_ENABLED`（默认 1）、`CACHE_MAX_FILES`（默认 200，LRU 淘汰）；`/health.cache` 报告 `entries/max/hits`。
- 适用场景：同 seed 反复对比、刷新页面回看、重复演示。

### 16.3 传输与导航
- `/result`、`/chunk` 返回 `Cache-Control: public, max-age=31536000, immutable`（job id 唯一）→ 浏览器与 CDN 复用，不再重复下载。
- 继续使用 **WebP 88 + 并行分块**（>4 MB 自动切换），预览走 `no-store`（文件会被就地改写）。
- **←/→ 现在跨任务、跨批量成员**切换：前端把任务摊平成 `viewList()` 的 `{job,id,i,bytes}` 列表。

### 16.4 前端模块化（按职责拆分）
| 模块 | 职责 |
|---|---|
| `site/js/config.js` | 纯数据：28 场景库 / 修饰词 / 配方 / 负向预设 / 快捷键 |
| `site/js/api.js` | 网络层：健康、提交、自适应轮询（后台自动暂停）、分块并行取图 |
| `site/js/store.js` | 状态与持久化：设置、历史、**收藏**、**词库**、统计、blob LRU |
| `site/js/ui.js` | DOM 层：画布 / **批量网格** / 对比滑块 / 画廊 / 灯箱 / 队列 / 健康面板 |
| `site/js/extras.js` | **新增**：运行统计、Prompt 词库、历史检索与 ★ 收藏 |
| `site/js/main.js` | 编排层：表单 → 提交 → 跟踪 → 展示 |
| `site/extra.css` | **新增**：批量网格、统计卡、词库/检索样式 |

新增面板：**运行统计**（生成数 / 成功率 / 平均耗时 / 缓存命中 / 常用采样器）、
**Prompt 词库**（存入 / 套用 / 随机取用）、**历史检索**（按 prompt·seed·采样器搜索，四种排序，★ 收藏，一键复用参数）。

### 16.5 行尾与提交卫生
- 文本文件统一 **LF**，仅 `*.bat/*.cmd/*.ps1` 保持 CRLF（见 `.gitattributes`）。
- 一键归一化：`python tools/normalize_eol.py` → 应输出 `normalised 0 files` 且 `remaining mixed endings: NONE`。
- 首次提交前执行 `git add --renormalize .`，让索引行尾与 `.gitattributes` 对齐（修复 GitHub 上的整文件 diff）。

### 16.6 改动文件
`server.py`（批内多图、结果缓存、索引取图、`/health.cache`）、`server_cloud.py`（字段对齐 + lifespan）、
`netlify/functions/proxy.js`（`i` 透传与 `X-Job-*`）、`site/js/{api,store,ui,main}.js`、`site/extra.css`、
`tools/normalize_eol.py`、`AGENTS.md`、本 README。

## 17. V6 起点：上传安全 + 任务取消 + 工程门禁（已实施）

> 完整 V6 方案（前端工作台重构、六页面、Workflow、Compare、Seed Lab…）见 **`docs/V6_PLAN.md`**。
> 本节只记录**已经落地并测过**的部分。

### 17.1 上传安全管线（强制）
`server.py::_sanitize_upload_bytes()` 是唯一入口：**magic bytes → 解码 → 像素/体积预算 → 元数据剥离 → 重新编码**。
- 只接受 **PNG / JPEG / WebP**；`data:` URL 必须匹配 `data:image/(png|jpe?g|webp);base64,`；
  **SVG / HTML / JS / PDF / ZIP / EXE 一律在解码前拒绝**（不再相信文件名或 `Content-Type`）。
- **EXIF / XMP / IPTC / ICC / GPS 全部剥离**：通过 `_strip_metadata()` **重建像素**（不是删字段），
  并先按 EXIF 方向摆正，再交给正常保存路径重新编码。
- **原始上传文件不落盘**：字节只在内存里处理，没有临时原图残留。
- 环境变量：`MAX_UPLOAD_BYTES`(4MB) · `MAX_INIT_PIXELS`(24MP) · `MAX_INIT_EDGE`(1024) · `ALLOW_UPLOAD`。
- 浏览器侧本来就安全：`upload.js` 走 `createImageBitmap → canvas → toBlob(WebP)`，元数据在**上传前**就已清除。

**回归测试**：`python tools/test_upload_security.py` → **20/20 通过**（含 GPS 标签 `0x8825` 剥离、
伪装文件拒绝、30MP 像素炸弹 413、超大 base64 413、边长压到上限）。

### 17.2 任务取消
`DELETE /jobs/{id}`：**只取消排队中的任务**（立刻生效、不占 GPU、结果文件为 0）；
**运行中的任务返回 409** —— 故意不允许中断 CUDA 步进（共享管线，强杀会污染显存）。
前端经代理 `op=cancel` 调用，命令面板里有「✖ 取消排队中的任务」。
**实测**：排队→`200 cancelled / files=0`；运行中→`409`；运行中任务随后正常 `done`。

### 17.3 工程门禁与开发入口
- `tools/check_eol.py`：行尾门禁，发现混合行尾即 `exit 1`（可挂 CI / pre-commit）。
- `tools/dev.ps1`：统一入口 —— `check / eol / security / bench / verify / start / tunnel / deploy`。
  提交前跑 `pwsh -NoProfile -File tools/dev.ps1 check`，一次串起六道门禁：
  **Python 编译 · JS 语法 · 前端 id 一致性 · 个人路径 · 脱敏 · 行尾**。
- 实测输出：`ALL CHECKS PASSED`（82 个文本文件路径扫描 / 敏感项 CLEAN / 行尾无混合）。

### 17.4 V6 视觉与命令面板（前端第一步）
- `site/css/tokens.css`：V6 设计令牌 + 质感层（近黑底、紫 `#7c5cff` / 青 `#22d3ee`、
  玻璃顶栏、渐变标题、分段控件药丸化、图片 fade-in reveal、细化阴影与焦点环），**零结构改动**即可全局换肤。
- `site/js/palette.js` + `site/css/palette.css`：**Ctrl/⌘+K 命令面板**（自带 DOM 的独立组件，
  模糊+子序列匹配、`↑↓` 选择、`Enter` 执行、结果计数），内含 **25 条真实命令**
  （生成 / 换 seed / 一次 2·4 张 / 随机灵感 / 收藏 / 复制 seed·prompt / 下载 / 全屏 / 对比 /
  跳到画廊·统计·词库·状态 / 预热 / 回收 / 主题 / 清空历史 / 取消排队任务 / 复用参数 / 导入导出）。
