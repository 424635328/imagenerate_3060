# AGENTS.md — Landscape·Art 项目规范

## 项目定位

Landscape·Art 是面向 RTX 3060 6GB 的 Stable Diffusion 1.5 + LoRA 风景生成项目：

- `app.py`：本地 Gradio 推理界面。
- `server.py`：单 GPU worker 的 FastAPI 后端、队列、鉴权、结果文件管理。
- `site/index.html`：Netlify 静态 Web 前端。
- `netlify/functions/proxy.js`：同源代理，不在浏览器暴露后端地址和 API token。
- `enhance.py`：超分与分块 img2img 高清流水线。
- `config.py`：共享路径、环境变量和安全边界；新增配置优先放这里。

## 路径规范

1. 禁止提交个人机器绝对路径（例如 `C:\Users\...`、`F:\...`、`/home/<user>/...`）。
2. 所有运行时路径必须从 `config.ROOT` / `LANDSCAPE_ROOT` 派生；文档使用 `<PROJECT_ROOT>` 或 `<你的项目根目录>`。
3. 模型、数据集、缓存、结果和日志默认被 `.gitignore` 排除；不要为了“方便运行”取消排除。
4. token、站点 ID、隧道真实域名只能来自环境变量或本地未跟踪文件；不要写进源码、HTML、研究脚本或批处理。
5. `research/` 只保存可复现检查脚本和脱敏报告；在线实测脚本必须从环境变量读取 URL。

## 生成性能基线

- 标准质量：DPM++ 2M Karras，20–28 步，CFG 6.5–8.0。
- 极速试稿：LCM/TCD，4–8 步，CFG 1.0–2.0；负向 prompt 控制能力较弱。
- 精细模式：DPM++ 2M SDE Karras，28–40 步。不要使用 500 步：本项目 SD1.5 在更早阶段收敛，500 步只会增加耗时和过饱和风险。
- 所有 API 与 CLI 输入都必须夹在安全范围内；后端是最终边界，不能只依赖前端校验。
- 高分辨率使用“基础生成 → 超分 → 可选分块重绘”，不要直接让 SD1.5 原生生成 2K/4K。

## 代码习惯

- Python 3.10+；函数和变量使用 `snake_case`，前端使用清晰的 camelCase。
- 新增依赖必须写入 `requirements.txt`，使用固定版本，并说明 CUDA/平台差异。
- 共享配置不复制到多个脚本；从 `config.py` 导入。
- 外部输入先做长度、范围和格式校验；错误信息不得包含 token 或本机路径。
- GPU 管线切换后释放 hooks、引用、Python GC 和 CUDA cache；不要同时驻留质量与极速管线。
- 结果文件必须有数量上限和磁盘配额；删除结果时同步删除预览文件。
- Web 图片默认 WebP；修改 MIME、扩展名或分块协议时必须同步修改后端、代理和前端。

## 训练脚本架构规范（并行 / 续训 / 显存自适应 · 强制）

> **适用范围**：任何训练、微调、蒸馏、超分脚本（`train_*.py`、`tools/probe_*train*.py`、缓存构建脚本）。
> 四条底线：**可续训 · 可观察 · 可降级 · 可验证**。缺一条就不算写完。
>
> **反面教材（都是本项目真实事故，不是假想）**：
> 1. 只在全部跑完后才写结果 → 进程被终止时证据归零（评测脚本，2026-09-12）；
> 2. 显存超配时进程**不报错、不退出**：180 秒 0 步推进、GPU 显示 100%、CPU 空转（V5b@640，2026-09-13）；
> 3. 训练 `print` 被重定向后块缓冲 → 日志空白十几分钟，无法判断是"慢"还是"死"；
> 4. adapter 加载静默失败（`lora_B` 全零）→ 键与形状全对，指标与基座逐位相同，差点被当成"方法无效"。

### 0. CLI 契约（缺一个就不合格）

| 开关 | 作用 | 怎么验收 |
|---|---|---|
| `--resume latest\|<path>` / `--fresh` | 默认自动发现最新**有效**检查点；`--fresh` 显式重开 | 跑到一半杀进程，重跑应从断点继续 |
| `--smoke N` | 真实数据上跑 N 步 + 一次验证 + 一次导出 | 5 分钟内跑通才算"代码没问题" |
| `--plan` | 只打印工作量/显存预估/断点情况，**不加载模型** | 零显存、零 GPU 即可回答"要跑多久" |
| `--min-free GB` | 启动前等显存回到阈值（共享卡必备） | 桌面占用高峰时不硬上 |
| `--stall-seconds N` | 心跳超时判定卡死并重启（driver 侧） | 见 §3 |
| `--save-every` / `--keep-checkpoints` | 落盘频率与保留数 | 磁盘不爆，且**永不删最新** |

### 1. 并发与 I/O 重叠：先判断瓶颈，别无脑开 workers

- **由数据源决定 worker 数**：
  - 数据是**预计算 latent/embedding 缓存**（本项目 `dataset1024/cache_*.pt`，`mmap=True`）：`num_workers=0`。
    这里没有磁盘解码可以重叠，多进程只会**各自复制一份缓存**、把 RAM 顶上去（本机已因此被用户报过内存问题）。
  - 数据需要真实 CPU 工作（JPEG 解码、增广、tokenize）：`num_workers=2~4` + `pin_memory=True` +
    `persistent_workers=True` + `prefetch_factor=2`，且必须写在 `if __name__ == "__main__":` 保护内（Windows spawn）。
- **能预计算的一律离线**：文本嵌入、latent、尺寸分桶、caption 索引。热路径里只留 `.to(device)`。
- **异步落盘要"有界 + 原子 + 可等待"**：checkpoint/图表交给后台线程，但同时最多 1 个待写任务，
  写 `.tmp` 后 `os.replace()`，进程退出前 `join()`。**不要**把"把张量拷到 CPU"也异步化——那会让显存被长时间引用。
- **日志必须行缓冲**：`print(..., flush=True)` 或启动时 `sys.stdout.reconfigure(line_buffering=True)`。
  被重定向到文件时块缓冲会让日志出现"十几分钟空白"。
- 多线程不等于更快：先量 `s/step` 与 CPU/GPU 利用率，再决定加并发。**任何并发改动都要用 step 时间证明有效**。

### 2. 全状态断点续训（Crash-safe）

检查点必须装下"继续跑所需的一切"，缺一项就会静默改变训练轨迹：

```python
payload = {
    "global_step": step, "epoch": epoch,
    "unet_lora": {k: v for k, v in unet.state_dict().items() if "lora_" in k or "magnitude" in k},
    "text_state": text_encoder.state_dict() if text_encoder is not None else None,
    "optimizer": optimizer.state_dict(), "scheduler": schedule.state_dict(),
    "scaler": scaler.state_dict() if scaler.is_enabled() else None,
    "ema": ema,                                    # EMA 影子权重，别只存 adapter
    "rng": rng_state(),                            # python / numpy / torch / cuda
    "cfg": cfg, "arch": arch,                      # 复现时的配方快照
    "protocol": {"eval_timesteps": EVAL_TIMESTEPS, "eval_subset_seed": EVAL_SUBSET_SEED},
    "profile": {"resolution": res, "micro_batch": mb, "accum": accum, "base_8bit": bool(...)},
}
```

- **RNG 是必需项**：不存 RNG，恢复后数据顺序与噪声都变了，"续训一致性"无从谈起。
  ⚠️ torch 2.6 起 `torch.load` 默认 `weights_only=True`，**不接受 ndarray** —— numpy state 要拆成
  `{"name", "keys": [int...], "pos", "has_gauss", "cached_gaussian"}` 再存；读取时
  `try weights_only=True except → False`（自己的 ckpt 才允许回退）。
- **原子落盘**：同目录 `step_N.pt.tmp` → `torch.save` → `os.replace()` 覆盖目标。**禁止**直接写目标文件
  （进程被杀会留下半截权重，比没有更糟）。
- **保留策略**：`--keep-checkpoints N` 只删**旧**的，**永不删最新**；删除放在新文件落盘成功之后。
- **加载即校验**：步数、`optimizer.state` 长度、adapter 张量形状与 `adapter_config.json` 是否一致；
  不一致就**明确报错**，绝不能"当作新训练从 0 开始"（那是静默丢进度）。
- **不要用"跳过前 N 个 batch"续训**：那会改变数据顺序。应恢复 sampler/epoch 偏移或使用确定性排列，
  保证"第 k 步看到的数据"与未中断时一致（本项目评测协议就是这么钉死的：`SUBSET_SEED`/`NOISE_SEED`）。
- **验收**：跑到 step k 杀掉 → 重启 → 断言从 k 继续，且后续 loss 轨迹在容差内一致。

### 3. 显存监视 + 卡死看门狗（WDDM 下 OOM 常常不抛异常）

- 每 N 步（建议 20）采样并**同时记录四元组**，只看一个数一定会漏：

```python
free_bytes, total_bytes = torch.cuda.mem_get_info()   # 真实可用/总量（含其它进程占用）
allocated = torch.cuda.memory_allocated()             # 我们持有的张量
reserved  = torch.cuda.memory_reserved()              # 含碎片；与 allocated 的差距=碎片
peak      = torch.cuda.max_memory_allocated()
```

  - `free` 决定"还能不能跑"；`reserved - allocated` 变大 = 碎片/泄漏信号。
- **日志每行都带** `step / loss / lr / elapsed / s_per_step / vram`，并写入 `<out_dir>/metrics.csv`
  （列升级时把旧文件改名为 `metrics_prev.csv`，否则会出现参差不齐的行无法解析）。
- **心跳 = metrics.csv 的 mtime**：外部守护（`tools/train_driver.py --stall-seconds`）据此判定卡死并重启。
  **判据只能是 mtime** —— 卡死时"进程存在""GPU 利用率 100%"两个指标都看起来完全正常。
- **警戒线**（6GB 卡）：`reserved > 5.3GB` 或 `free < 0.8GB` → `gc.collect()` + `torch.cuda.empty_cache()`
  并打印一次 WARN；**连续 WARN** 才升级为降级（见 §4），单次 WARN 不要乱动配置。
- **共享卡前提**：桌面/浏览器/LLM server 随时会抢占显存。启动前 `--min-free` 等余量，
  跑起来后靠心跳看门狗兜底；**不要在别人的占用高峰硬上**。

### 4. 显存不足自适应（等效批次 + 分级降级）

- **等效批次恒等**：`Total_Batch = micro_batch × accum`。降 micro-batch 必须等比升 accum。
- **loss 归一化按"本轮实际累积次数"**（`loss / actual_accum`，不是配置里的计划值），
  否则降级后梯度尺度会跳变，学习率曲线失去意义。
- **只在 optimizer step 边界改配置**，改完写一条结构化日志：`step, old→new, 原因, free VRAM`。
  禁止在 step 中途重建 DataLoader。
- **分级降级顺序**（从最不影响收敛到最影响；6GB 实测有效）：

  | 级别 | 动作 | 本项目实测 |
  |---|---|---|
  | 1 | `gc.collect()` + `empty_cache()`；缩小验证子集 | V5b `eval_subset 48→8` |
  | 2 | 开/保持 `gradient_checkpointing`；确认 `attn_implementation="sdpa"` | Windows 上 xformers/flash-attn 常不可用，别依赖 |
  | 3 | micro-batch → 1 并等比放大 accum（保持等效批次） | — |
  | 4 | **降分辨率/缓存分桶**（6GB 上收益最大） | 640→512：显存 5.00→4.46GB，步时 3.4→1.85s |
  | 5 | int8 基座（QLoRA）/ 8-bit Adam | SDXL 实测 int8 比 fp16 **慢一倍以上**，只在内存是硬约束时用 |

- **热降级的正确落法**：把生效后的 `profile` 写进检查点，让 driver/配置层带着新 profile 重启
  （比在进程内重建模型/优化器安全得多）。重启后必须打印"用了哪个 profile"。
- **崩溃捕获**：`torch.cuda.OutOfMemoryError`（并覆盖"静默卡死"）→ `zero_grad(set_to_none=True)` +
  `gc.collect()` + `empty_cache()` → 重试当前步；连续失败升级降级；**全部手段用尽才抛出**，
  抛出前先 `save_checkpoint()` 保存现场（步数 + 四元组 + profile）。
- 已知约束（别浪费时间去试）：Prodigy 只支持**单一全局 LR**（给 UNet 与 TE 设不同步长会直接报错，
  要分步长就用 `adamw8bit`）；`expandable_segments` 在 Windows 上不支持（会告警）；
  `bitsandbytes` 的 int8 线性层在小 batch 下反量化开销占主导。

### 5. 验收（写完必须自证，不能只"跑起来了"）

1. `--smoke 4` 在**真实缓存**上跑通（本项目 SDXL 的冒烟门正是这样抓出"缓存没建好"的）；
2. **续训一致性测试**：kill → resume → 步数连续 + loss 轨迹一致（容差内）；
3. **卡死测试**：把 `--stall-seconds` 调小，确认 driver 能终止并重启；
4. **纯函数单测**（不需要 GPU）：批次/累积换算、显存阈值判定、检查点发现与选择、协议签名；
5. 运行方式遵循「长任务与工具调用超时」：**后台作业 + 独立日志 + 心跳文件**，前台只看汇总。

### 6. 本项目现状对照（差距清单，改脚本时顺手补齐）

| 规则 | `train_v5.py` | `train_sr_gan.py` | `train_driver.py` |
|---|---|---|---|
| 全状态续训 | ✅ adapter/optimizer/scheduler/EMA/TE + **RNG/scaler**（2026-09-13 补） | ✅ 含 RNG/EMA | ✅ `--resume latest` |
| 原子落盘 | ✅ `.tmp` + `os.replace` | ✅ 同 | — |
| 保留策略 | ✅ 只删旧、保留最新 N | ✅ | — |
| 心跳 + 卡死看门狗 | ✅ `metrics.csv`（10s） | ✅ 240s 无迭代告警 | ✅ `--stall-seconds`（mtime 判据） |
| 显存四元组 | ✅ `vram/reserved/free` 三列 + 峰值 | ⚠️ 仅峰值 | — |
| 热降级 | ⚠️ OOM 重试 5 次后保存退出；**降分辨率仍需人工换配置** | ⚠️ 同上 | ✅ 等显存 + OOM 自动重启 |
| `--plan` | ✅ 零显存打印缓存/检查点/工作量/上次实测速率（2026-09-13 补） | ⚠️ 无 | — |
| 验收测试 | ⚠️ 续训一致性测试脚本已就绪（`tools/test_resume_consistency.py`），**尚未在真机跑过** | ⚠️ 同 | ✅ 有单测（`test_pipeline_logic.py`） |

> 待补（按价值排序）：① 跑通一次续训一致性测试并把结果写进 §6（脚本已就绪）；
> ② 进程内"降分辨率"热降级（现在靠配置层重启，够用但慢）；③ `train_sr_gan.py` 的 `--plan`。
> 改这三项时**必须**同时补上对应的门禁脚本。

---

## 常用命令（PowerShell）

```powershell
$ROOT = (Get-Location).Path
$env:LANDSCAPE_ROOT = $ROOT
$env:API_TOKEN = '<strong-random-token>'
python -m py_compile app.py server.py enhance.py config.py
python tools/check_paths.py
python research/final_check.py
python server.py
```

本地 Gradio：`python app.py`；静态站点本地预览可使用任意静态 HTTP 服务器；线上代理配置只放 Netlify 环境变量。

## 修改与验证要求

1. 先读调用链：前端 → Function → `/generate` → worker → 结果/分块接口。
2. API 字段新增时，至少同步 `server.py`、`server_cloud.py`、代理和前端。
3. 每次修改后运行 Python 语法编译、Node 语法检查和路径/脱敏扫描。
4. 涉及图像协议时，验证 MIME、扩展名、预览、分块总字节数和下载行为。
5. 不提交模型、数据集、结果、token、个人路径或临时脚本输出。
6. 提交前确认 `git diff --check` 无空白错误，`git check-attr eol -- <file>` 符合 `.gitattributes`。
7. 不要为了通过检查而删除真实功能；发现环境缺失时记录原因和替代验证。

## 升级优先级

性能安全边界 > 生成稳定性 > 传输体验 > 前端装饰。每次升级保持旧 API 字段兼容，并在 README 记录用户可感知的变化。

## v5 新增约定（批内多图 / 结果缓存 / 前端扩展）

- **服务端批量**：`GenerateReq.batch`（1–4）。一个队列位、一次轮询产出 N 张，种子为 `seed..seed+N-1`。
  取图带索引：`GET /result/{id}?i=k`、`GET /chunk/{id}?i=k&off=&len=`；代理透传 `i` 并回传
  `X-Job-Count` / `X-Job-Index`。`/jobs/{id}` 新增 `count`、`seeds`、`files`、`bytes_each`、`batch_index`。
  旧客户端不传 `i` 即取第 0 张，保持兼容。
- **结果缓存**：仅当 `seed>=0 && batch==1 && 无 init_image` 时按参数指纹缓存
  （prompt/style/res/aspect/steps/cfg/seed/neg/sampler/fast/highres/enhance/sr_model/enhance_strength/enhance_steps/upscale/strength）。
  缓存文件在 `results/cache/`（**独立目录，切勿纳入任务裁剪范围**），命中时用**硬链接**给任务自己的路径；
  `CACHE_ENABLED` / `CACHE_MAX_FILES`（默认 200，LRU）可调；命中时 `/generate` 直接返回 `{status:"done",cached:true}`。
- **前端模块边界**：`config.js` 纯数据；`api.js` 只做网络；`store.js` 只做状态与持久化；`ui.js` 只做 DOM；
  `extras.js` 承担统计/词库/检索等附加面板；`main.js` 只编排流程。新 UI 优先放 `extras.js` + `extra.css`，
  不要继续膨胀 `main.js` 与 `styles.css`。
- **视图列表**：展示与导航都基于 `viewList()` 展开的 `{job, id, i, bytes}` 扁平列表，←/→ 因此可跨任务、跨批量成员切换。
- **行尾**：文本文件一律 LF，仅 Windows 启动器（`*.bat/*.cmd/*.ps1`）用 CRLF。
  改完跑 `python tools/normalize_eol.py`，提交前用 `git check-attr eol -- <file>` 复核。

## v6 新增约定（上传安全 / 任务取消 / 工具链）

- **上传安全管线（强制）**：`server.py::_sanitize_upload_bytes()` 是唯一入口，顺序为
  **magic bytes → 解码 → 像素/体积预算 → 元数据剥离 → 重新编码**。
  - 只接受 **PNG / JPEG / WebP**；`data:` URL 必须匹配 `data:image/(png|jpe?g|webp);base64,`，
    SVG/HTML/JS 一律拒绝（`ALLOWED_UPLOAD_FORMATS` / `MAX_INIT_PIXELS` / `MAX_INIT_BYTES` 可调）。
  - **不得**相信文件名或 `Content-Type`；畸形/伪装文件必须在解码前被拒。
  - 元数据（EXIF/XMP/IPTC/ICC/**GPS**）通过 `_strip_metadata()` 重建像素来剥离，禁止"只删字段"。
  - **不落盘原始上传文件**：上传字节只在内存中处理，临时文件不得保留。
  - 回归测试：`python tools/test_upload_security.py`（20 项，含 GPS 剥离与伪装文件拒绝）。
- **任务取消**：`DELETE /jobs/{id}` 只取消 **queued** 任务（立刻生效、不占 GPU、`files=0`）；
  运行中的任务返回 **409** —— 不允许中断 CUDA 步进（会污染共享管线）。前端 `op=cancel` 经代理转发。
- **统一开发入口**：`tools/dev.ps1`（check / eol / security / bench / verify / start / tunnel / deploy）。
  提交前一律 `pwsh -NoProfile -File tools/dev.ps1 check`，它串起编译、JS 语法、前端一致性、CSS 解析、
  动效弹簧曲线、路径、脱敏与行尾八道门禁。
- **行尾门禁**：`tools/check_eol.py` 发现任何混合行尾即 `exit 1`（可挂 CI/pre-commit）；
  修复用 `tools/normalize_eol.py`。**`.ps1` 属 Windows 启动器，必须 CRLF。**

## 长任务与工具调用超时（重要）

- **单次前台命令上限 10 分钟**（600000 ms）。该上限由 harness 执行器
  （`@deepseek-ai/dsh-pwsh-local` / `dsh-bash-local`）的 `maxTimeoutMs` 默认值强制，
  工具调用里的 `timeoutMs` 只能**调小**，调大无效；超时会输出 `[timed out after 600000ms]`。
- **超过 ~5 分钟的命令一律用后台作业**：调用时加 `run_in_background: true`，它不受该上限约束。
  启动后立即拿到 job id，用 `job_output` 取输出、`job_kill` 停止；作业结束会自动通知，**不要 sleep 轮询**。
- 训练 / 评测 / 模型下载 / 缓存构建等长任务必须「后台作业 + 日志文件」双写
  （日志放 `%TEMP%\lsart_*.log` 或 `research/*.log`），前台只做短查询与结果汇总。
  历史上 V5 训练曾在 step 395 被一次过长的前台等待整体杀掉。
- **不要为了放宽这个上限去改 harness 组合**（把 `maxTimeoutMs` 调大会牵动 profile 组合并要求重启 `dsh`，
  且升级时会被覆盖）。正确做法是后台作业；需要跨会话续跑时用 `tools/train_driver.py` 的自动续训。
- 暂停 / 恢复训练与环境回收的配方见 `docs/TRAINING.md` §6。

## 修改后必跑的检查

```powershell
python -m py_compile app.py server.py server_cloud.py config.py runtime.py enhance.py
python tools/check_frontend.py     # 前端 id / 模块引用一致性
node tools/check_css_syntax.mjs    # 全部 CSS 走 css-tree 严格解析（缺依赖自动 SKIP）
python tools/check_paths.py        # 个人绝对路径与敏感串
python tools/normalize_eol.py      # 行尾统一（应报 0 或仅启动器）
python research/final_check.py     # 脱敏总检（应无敏感项）
node --check site/js/main.js       # 逐个 JS 模块

# 前端逻辑回归（jsdom 可选依赖，未安装会自动 SKIP）
node tools/test_proxy_guard.mjs    # 代理来源守卫（跨站 POST 必须 403）
$env:NODE_PATH="$env:TEMP\lsart-jsdom\node_modules"; node tools/smoke_frontend.mjs

# 评测与融合的纯 CPU 门禁（不需要 GPU）
python tools/test_eval_plan.py     # 评测协议 / 断点续跑 / 配对统计 / 报告并表（91 项）
python tools/test_merge_math.py    # 融合算术 + no-op 事故回归（28 项）
python tools/test_sdxl_base.py     # 基座解析到本地快照（13 项，防"联网失败被误判成显存不足"）
python tools/test_pipeline_logic.py  # 探针解析 / 分辨率决策 / 故障归类（16 项）
python tools/test_sr_and_judge.py  # 自训超分接入 + sr_model 输入校验 + 盲测裁判统计（32 项）
python tools/test_spring_easing.py # 前端动效 spring 曲线 vs 解析解（31 项，改 site/css/motion.css 必跑）
python tools/deploy_check.py --adapter models/v5b_lora/adapter_best --expect-text-encoder
       # 部署形态静态检查：线上到底会加载哪份 UNet LoRA 与文本编码器
python tools/verify_merge.py --sources models/v4_640/adapter_best models/v5_lora/adapter_best `
       --merged models/merged --weights 0.5 0.5     # 融合产物必须全 PASS 才能进评测
```

```powershell
# 需要 GPU 的验收（改过训练/续训逻辑后必跑）
python tools/test_resume_consistency.py   # kill→resume 与一口气跑完的权重必须一致（约 2-3 分钟）
python tools/test_adapter_load.py models/v5b_lora/adapter_best   # 适配器"确实生效 + TE 真的变了"
```

> **适配器与融合产物的门禁必须验"数值"，不能只验"能加载"**：本项目两次被静默 no-op 咬过
> （`load_lora_weights` 不加载 peft 权重；PEFT `add_weighted_adapter` 产出全零 `lora_B`），
> 两次的键集合与形状都完全正常。判据只能是等效增量 `(alpha/r)·(B@A)` 非零且符合预期。