# 从零复现（REPRODUCE）

> 目标：另一位协作者 `git clone` 之后，按本文即可**跑起服务、复现训练、并得到同等质量的结果**。
> 仓库里只有「源码 + 配置 + 文档 + 站点资源 + 归档代码」；权重、数据集、缓存、结果都不入库（见 §8 清单）。

---

## 1. 前置条件

| 项 | 要求 | 说明 |
|---|---|---|
| Python | **3.11**（3.10+ 可用，本项目在 3.11.16 验证） | 训练/推理同一环境即可 |
| GPU | NVIDIA，**≥6 GB 显存**（compute ≥ 7.5） | 本项目在 RTX 3060 Laptop 6GB（8.6）验证；SDXL 训练走 int8 基座 + QLoRA |
| 驱动 / CUDA | 驱动 ≥ 550，torch 用 **cu124** wheel | 见 §2 安装命令 |
| 磁盘 | ≥ 25 GB 可用 | 模型 ~10 GB + 数据集 ~1.3 GB + 缓存/结果 |
| Node.js | ≥ 18（仅前端工具链与 Netlify CLI 需要） | 后端与训练不需要 |

## 2. 安装与自检（约 10 分钟）

```powershell
git clone https://github.com/424635328/imagenerate_3060.git
cd imagenerate_3060

python -m venv .venv ; .\.venv\Scripts\Activate.ps1
# ① 先装与驱动匹配的 torch（cu124 为本项目验证版本）
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
# ② 再装其余依赖（版本已 pin 到「本机验证过的组合」）
pip install -r requirements.txt

$env:LANDSCAPE_ROOT = (Get-Location).Path
pwsh -NoProfile -File tools/dev.ps1 check          # 六道门禁：应输出 ALL CHECKS PASSED
```

> `transformers==5.16.1` 是大版本；本项目训练脚本在其上验证通过。若必须回退 4.x，
> 请同步确认 diffusers/peft 兼容性，并重跑门禁。

## 3. 环境变量清单

后端会**直接读取**这些变量（`config.py` 是唯一来源，不要在别处复制）：

| 变量 | 默认 | 用途 |
|---|---|---|
| `LANDSCAPE_ROOT` | 仓库根 | 所有路径的基准 |
| `HF_HOME` / `HF_CACHE` | `models/hf_cache` | HF 下载缓存 |
| `BASE_MODEL` | `models/base_rv6` | SD1.5 基座；**也可直接填 HF 仓库 id** |
| `LORA_ADAPTER` | `models/v4_640/adapter_best` | 线上使用的 LoRA |
| `LCM_LORA` / `SR_DIR` | `models/lcm_lora` / `models/sr` | 极速模式与超分权重 |
| `API_TOKEN` | `change-me` | **必须改**：后端鉴权（`X-API-Key`） |
| `HOST` / `PORT` | `127.0.0.1` / `8001` | 监听地址 |
| `MAX_QUEUE` / `RATE_PER_MIN` / `MAX_JOBS` | 20 / 8 / 500 | 队列、限流、任务表上限 |
| `RESULTS_QUOTA_MB` / `IDLE_UNLOAD_SEC` | 2048 / 900 | 结果磁盘配额、空闲卸载显存 |

前端/代理侧（**只放 Netlify 站点环境变量，绝不进仓库**）：

| 变量 | 用途 |
|---|---|
| `BACKEND_URL` | 隧道或公网后端地址（如 `https://<你的隧道>`） |
| `API_TOKEN` | 与后端一致 |
| `ALLOWED_ORIGINS` | 自定义域名的白名单（逗号分隔） |
| `PROXY_STRICT=1` | 连无 Origin 的脚本调用也拒绝（最严） |

本地真实取值（路径、令牌、站点 ID、隧道域名）写在**未入库**的 `LOCAL_NOTES.md`：
`cp LOCAL_NOTES.example.md LOCAL_NOTES.md` 后自行填写（该文件已被 `.gitignore` 忽略）。

## 4. 拉齐模型资产（一条命令）

```powershell
python tools/fetch_models.py all
#  = SD1.5 基座(RV6) + SDXL 基座(RealVisXL V5.0) + LCM-LoRA + 4x-UltraSharp + RealESRGAN_x4plus
```

落到约定路径：`models/hf_cache/`（HF 快照）、`models/lcm_lora/`、`models/sr/`。
**不入库的中间产物**里，`models/base_rv6` 是本地 diffusers 目录，协作者不必转换——
直接 `$env:BASE_MODEL="SG161222/Realistic_Vision_V6.0_B1_noVAE"` 即可（训练与推理都接受 HF id）。

## 5. 数据集（**仓库不含图片，需自备**）

仓库**不包含训练图片**（版权与体积原因）。本项目的 `dataset1024/` 由作者本地的
31 个风景图 zip 构建，协作者需自备同构数据。期望结构：

```
dataset1024/
├─ train/   *.jpg         # 1511 张，1024×1024（短边缩放 + 中心裁剪）
├─ test/    *.jpg         # 144 张，验证集
└─ manifest.json          # [{"entry": <原始文件名>, "file": <绝对路径>, "split": "train|test", "caption": <英文描述>}]
```

用自己的图片构建：

```powershell
# a) 有原始 zip（每张图一个 entry）时：
python tools\dataset\prepare_v4_data.py --zip-dir "<你的 zip 目录>" --size 1024
# b) 已有 1024 图目录时，直接写 manifest（字段见上），然后打标：
python tools\dataset\caption_images.py --manifest dataset1024\manifest.json   # BLIP（旧方案）
python recaption.py --manifest dataset1024\manifest.json --limit 5 --dry-run  # Qwen2-VL（推荐）
```

> 训练脚本只依赖 `manifest.json` + 图片路径；`file` 可为绝对路径，也可写相对 `--data_dir` 的相对路径
> （`precompute_v5.py` 会自动解析）。

## 6. 复现训练

```powershell
# ── V5（SD1.5，6 GB 可跑，实测 1.39 s/step @640）──
python precompute_v5.py --res 640 --out dataset1024\cache_v6_640.pt        # latent + 文本缓存
python train_v5.py --config config_v5.cfg                                  # min-SNR-γ + Prodigy + EMA
python train_v5.py --config config_v5.cfg --resume latest                  # 断点续训
python tools\compare_adapters.py --adapters "V4:models\v4_640\adapter_best" "V5:models\v5_lora\adapter_best"
python tools\test_adapter_load.py models\v5_lora\adapter_best              # 部署门禁：能被推理栈加载

# ── SDXL QLoRA（**本机实测不可行**，见 docs/TRAINING.md §10.3；命令保留供云端/大显存复现）──
python tools\test_qlora_path.py                                            # 先跑单测（1 分钟）
python tools\probe_sdxl_train.py                                           # 逐档实测：本机 50-65 s/micro-batch
python precompute_v5.py --base SG161222/RealVisXL_V5.0 --arch sdxl --res 1024 `
       --out dataset1024\cache_v5_sdxl1024.pt
python train_v5.py --config config_v5_sdxl.cfg
# 基座一律经 tools\sdxl_base.py 解析到**本地快照**并离线加载：走 HF Hub id 会联网取文件，
# 在代理环境下以 SSL 错误失败，且会被探针误记成"显存不足"（2026-09-13 事故）

# ── 无人值守（共享显卡：自动等显存 + OOM 续训 + 全链路接管）──
python tools\train_driver.py  --config config_v5.cfg --save_every 250 --resume latest
python tools\train_pipeline.py --v5-out models/v5_lora --sdxl-config config_v5_sdxl.cfg
# ↑ 流水线内含「4 步冒烟门」：用真实缓存跑通量化+缓存文本条件+验证+导出，才启动长跑
#   探针全 FAIL 且无 OOM 时会被判为 infrastructure（环境故障），而不是"显存不够"

# ── V5b：补训 CLIP 文本编码器（对齐 V4 部署形态，实测 3.4 s/step，约 70 分钟）──
python tools\train_driver.py --script train_v5.py --config config_v5b.cfg `
       --save_every 200 --min-free 2.5
# 产出 models\v5b_lora\adapter_best\{adapter_model.safetensors,adapter_config.json}
#      + models\v5b_lora\adapter_best_text_encoder.pt（正是 app.py 期望的文件名）

# ── 免训练模型融合（CPU 自实现 LoRA 算术 + SVD 重压缩，可与 GPU 训练并行）──
python tools\merge_lora.py --adapters models\v4_640\adapter_best models\v5_lora\adapter_best `
       --methods linear slerp ties dare_ties --weights 0.5 0.5 --out models\merged
# 门禁：融合产物必须证明"等效增量非零且符合算子定义"，否则不许进评测
python tools\verify_merge.py --sources models\v4_640\adapter_best models\v5_lora\adapter_best `
       --merged models\merged --weights 0.5 0.5
python tools\test_merge_math.py                     # 融合算术单测 + no-op 事故回归（28 项，纯 CPU）
# 先看代价（不加载模型、不占显存），确认复用/重算与 ETA 后再真跑
python tools\eval_val_mse.py --plan --adapters "V4:models\v4_640\adapter_best" "ties:models\merged\ties_0.50_0.50" `
       --cache dataset1024\cache_v4_640.pt --csv research\eval_merged.csv
python tools\eval_val_mse.py --adapters "BASE:" "V4:models\v4_640\adapter_best" "V5:models\v5_lora\adapter_best" `
       "linear:models\merged\linear_0.50_0.50" "slerp:models\merged\slerp_0.50_0.50" `
       "ties:models\merged\ties_0.50_0.50" "dare_ties:models\merged\dare_ties_0.50_0.50" `
       --cache dataset1024\cache_v4_640.pt --csv research\eval_merged.csv    # 每完成一行即落盘
# ↑ 锚点行可从 research\eval_val.csv 预置（同协议会自动复用）；被中断后重跑即从断点继续

# ── 评测（三层证据）──
python tools\eval_val_mse.py --adapters "BASE:" "V4:models\v4_640\adapter_best" "V5:models\v5_lora\adapter_best" `
       --cache dataset1024\cache_v4_640.pt --csv research\eval_val.csv     # 固定协议 val（可比）
python tools\make_eval_report.py --run-val --with-images   # 汇总 val + 出图 + CLIP → docs/EVAL_REPORT.md
#   ↑ 含 SDXL 线：自动带 SDXL 基座锚点跑 val（research\eval_val_sdxl.csv），
#     并调 tools\compare_sdxl.py 出 SDXL 的前后对比页（research\compare_sdxl\）
python tools\test_pipeline_logic.py                # 分辨率决策 / 探针解析 / 故障归类（16 项）
python tools\test_eval_plan.py                     # 评测协议/续跑/配对统计/并表（91 项）
python tools\test_sr_and_judge.py                  # 自训超分接入 + 输入校验 + 裁判统计（32 项）
python tools\test_merge_math.py                    # 融合算术 + no-op 事故回归（28 项）
python tools\test_sdxl_base.py                     # 基座解析到本地快照（13 项）

# ── 自研判据：留出 prompt 的分布距离 + 盲测成对偏好裁判 ──
python tools\eval_fid.py --adapters "V4:models\v4_640\adapter_best" "V5b:models\v5b_lora\adapter_best" `
       --out research\fid --steps 24 --res 512 --seeds 2
#   ↑ 24 条自拟留出提示词；KID/CLIP-FID（与真实照片的分布距离）+ 留出 prompt 的 CLIP 一致性
python tools\vlm_judge.py --a "V4:models\v4_640\adapter_best" --b "V6:models\v6_qwen\adapter_best" `
       --out research\judge_final --steps 24 --res 512 --seeds 2
#   ↑ 本地 Qwen2-VL-2B 当裁判：随机左右交换（盲测）+ 强制选择 + Wilson 95% 区间

# ── caption 质量实验（VLM 重写 → 重建缓存 → 同配方训练）──
python tools\recaption_driver.py --manifest dataset1024\manifest.json `
       --out dataset1024\captions_qwen.json --checkpoint-every 25 --max-edge 512
#   ↑ 带心跳看门狗（卡死自动续跑）；每 25 张增量落盘，manifest 有 .bak 备份
python precompute_v5.py --base SG161222/Realistic_Vision_V6.0_B1_noVAE --arch sd15 `
       --res 640 --crops 3 --out dataset1024\cache_v6_qwen640.pt
python tools\train_driver.py --script train_v5.py --config config_v6q.cfg --save_every 500 --min-free 1.8

# ── 推理侧：6 GB 能否跑 SDXL、有多快 ──
python tools\fetch_models.py lightning             # SDXL-Lightning 4 步 LoRA（约 376 MB）
python tools\probe_sdxl_infer.py                   # 逐档实测峰值显存与 s/image
```

> 定性判读（agent 用视觉逐格看对比页的结论）记录在 `docs/VISUAL_REVIEW.md`；
> 它与定量判据互为补充：定性判读适合发现指标盲区（伪影、臆造文字），不适合判定 0.3% 量级的差异。

### 6.1 超分 GAN（细节化放大，约 35 分钟）

```powershell
# 从官方 RealESRGAN_x4plus 初始化做风景领域微调：L1 + VGG19 感知 + 对抗
python train_sr_gan.py --iters 4000 --hr-size 256 --batch 3
# 或经驱动（断点续训 + 显存等待）
python tools\train_driver.py --script train_sr_gan.py --iters 4000 --hr-size 256 --batch 3

# 证据：bicubic vs RealESRGAN vs UltraSharp vs 本项目 GAN（PSNR/SSIM + 细节量 + 对比拼版）
python tools\probe_sr_model.py --images 8 --out research\sr_probe

# 接入生产线（已完成，2026-09-14）：注册名 `ours` / `ours_final`
#   Gradio / FastAPI / 网页前端三处默认都指向它；可用 SR_MODEL 环境变量回退商用权重
python tools\deploy_check.py                       # 静态检查线上到底加载哪份权重
python tools\test_sr_and_judge.py                  # 注册表 + 输入校验 + 三入口一致性（32 项）
```

方法与论文依据见 [TRAINING.md](TRAINING.md)（min-SNR-γ / Prodigy / DoRA / QLoRA 均有引用与实测数据）。

## 7. 跑起服务与前端

```powershell
$env:API_TOKEN = "<强随机串>" ; $env:LORA_ADAPTER = "$PWD\models\v4_640\adapter_best"
python server.py                     # FastAPI :8001
python app.py                        # 可选：Gradio 本地界面 :7860
```

前端（Netlify）：`npx netlify-cli deploy --dir site --prod --site <SITE_ID> --auth <TOKEN>`，
站点环境变量按 §3 配置。代理已内置跨站防护（`netlify/functions/proxy.js`），
只有 `*.netlify.app` / `localhost` / `ALLOWED_ORIGINS` 里的来源能触发 GPU 任务（详见 [SCALE.md](SCALE.md)）。

## 8. 不入库内容与重建方式（协作者对照表）

| 不入库内容 | 体积 | 如何得到 |
|---|---|---|
| `models/**` 权重（基座/LoRA/LCM/超分/HF 缓存） | ~17 GB | `python tools/fetch_models.py all` + 训练产出 |
| `dataset1024/`（图片 + manifest + latent 缓存） | ~1.3 GB | §5 自备数据 → `precompute_v5.py` |
| `results/`（后端任务结果） | 数十 MB | 运行时自动产生，配额自动回收 |
| `outputs/`（历史样张/对比图） | — | 已删除；用 `tools/compare_adapters.py` 等重新生成 |
| `*.log`、`research/` 一次性脚本 | — | 运行时产生 / 已归档到 `archive/` |
| `LOCAL_NOTES.md`、`.netlify/`、`*.token`、`ngrok.yml` | — | 本地私有：按 §3 自行填写 |

## 9. 校验清单（提交/发布前）

```powershell
pwsh -NoProfile -File tools/dev.ps1 check      # 编译 + JS 语法 + 前端一致性 + 路径 + 脱敏 + 行尾
node tools/test_proxy_guard.mjs                # 代理跨站防护（10 项）
python tools/test_qlora_path.py                # int8 QLoRA 路径单测（7 项）
python tools/test_upload_security.py           # 上传安全（20 项）
npm install --prefix "$env:TEMP\lsart-jsdom" jsdom
$env:NODE_PATH="$env:TEMP\lsart-jsdom\node_modules"; node tools/smoke_frontend.mjs   # 真实 DOM 冒烟（21 项）
python research/final_check.py                 # 脱敏总检（打印 git 将提交的文件清单）
```

## 10. 脱敏约定（协作者必读）

1. **绝不提交**：令牌（`nfp_…`/`hf_…`/`sk-…`）、后端 `API_TOKEN`、Netlify 站点 ID、隧道真实域名、个人绝对路径（`C:\Users\...`）。
2. 真实取值只放两类地方：**环境变量**，或本地未跟踪的 `LOCAL_NOTES.md`（模板见 `LOCAL_NOTES.example.md`）。
3. 文档与脚本里一律用占位符：`<你的项目根目录>`、`<强随机串>`、`https://<你的隧道>`、`<SITE_ID>`。
4. 提交前跑 `python research/final_check.py`，它按 git 的忽略规则扫描**将被提交的文件**；
   发现命中即修正，不要用 `--no-verify` 绕过。
5. 若曾把令牌写进过提交历史，**该令牌视为已泄漏**：先轮换，再按 Git 历史清理流程处理。
