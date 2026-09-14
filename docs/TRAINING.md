# 训练方法与硬件约束（TRAINING）

> 硬件事实：**RTX 3060 Laptop 6GB**（compute 8.6），torch 2.6+cu124，diffusers 0.40，peft 0.20，
> bitsandbytes 0.50.2，prodigyopt。所有结论都基于本机实测，不是估计。

## 1. 现有基线（V4）

| 项 | 值 |
|---|---|
| 基座 | `SG161222/Realistic_Vision_V6.0_B1_noVAE`（SD1.5 写实向） |
| 方法 | LoRA r64 + EMAlen + caption dropout + 多随机裁剪 latent 缓存 |
| 数据 | `dataset1024/`：1511 训练 + 144 验证，全部 1024×1024，含英文 caption；缓存 `cache_v4.pt`(512)/`cache_v4_640.pt`(640) |
| 速度 | 640 分辨率 **1.12 s/step**（batch1 × grad accum 2） |
| 验证 | cache_v4_640 测试集 MSE **0.1262**（best） |

## 2. V5 采用的方法（2023–2025）

| 方法 | 论文 | 在本项目的落点 |
|---|---|---|
| **min-SNR-γ 损失加权** | Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy*, [arXiv:2303.09556](https://arxiv.org/abs/2303.09556) | `train_v5.py::min_snr_weights`，γ=5，按 SNR 重加权各 timestep 的 MSE |
| **Prodigy 优化器** | Mishchenko & Defazio, *Prodigy: An Expeditious and Stable Adaptive Optimizer*, [arXiv:2306.06101](https://arxiv.org/abs/2306.06101) | `optimizer=prodigy`，自适应估计更新尺度，LoRA 上免手调 LR |
| **DoRA** | Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)（WACV 2025） | `use_dora=true`；**已实测可被 diffusers 0.40 加载并出图**（`tools/test_adapter_load.py`） |
| **QLoRA（int8 基座）** | Dettmers et al., [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) | SDXL 路线：fp16 UNet 4.9GB 放不下 → `base_8bit=true` 把基座压到 ~2.5GB |
| LoRA | Hu et al., [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | 推理侧权重格式不变，后端 `load_lora_weights` 零改动 |
| SDXL | Podell et al., [arXiv:2307.01952](https://arxiv.org/abs/2307.01952) | 质量跃升的路线，RealVisXL V5.0（同作者 SG161222） |
| Sana（备选未采用） | Xie et al., ICLR 2025（0.6B 线性 DiT，[项目页](https://github.com/NVlabs/Sana)） | 显存友好，但生态/写实质量不及 SDXL，且需另建推理栈 |

## 3. 实测数据与路线决策

下表是"**在 RTX 3060 6GB 上到底哪条路能走**"的全部依据，每一行都是本项目在**这台机器上**实测出来的
（不是论文里的数字，也不是估算）。判定列给出采纳/否决及理由。

| 路线 / 配方 | 峰值显存 | 吞吐 | 判定 |
|---|---|---|---|
| SD1.5 640，LoRA r64 + Prodigy + min-SNR | **2.36 GB** | 1.25–1.53 s/step | ✅ **采纳**（V5 主配方；余量充足） |
| SD1.5 640，DoRA r64 | 2.41 GB | ~2.0 s/step | ✅ 可用（且已验证能被 diffusers 0.40 加载出图） |
| SD1.5 512 + CLIP 文本编码器 fp32 微调（V5b） | **4.46 GB** | 1.85–1.95 s/step | ✅ **采纳**（唯一真微调 CLIP 的可行形态，39.5 分钟跑完 1200 步） |
| SD1.5 640 + CLIP 文本编码器 fp32 微调 | 5.00 GB | 3.4 s/step → **卡死** | ❌ **否决**：WDDM 超配后 180 秒 0 步推进（见 §7.5） |
| SDXL int8（QLoRA）768 | 5.45 GB | 65.1 s/micro-batch | ❌ **否决**：2500 步 × 累积 4 ⇒ **≈180 小时** |
| SDXL int8（QLoRA）896 | 6.27 GB | 51.5 s/micro-batch | ❌ 同上（≈143 小时） |
| SDXL int8（QLoRA）1024 | **7.21 GB（越过物理显存）** | 62.1 s/micro-batch | ❌ 溢出到共享内存，最慢且最不稳 |
| SDXL fp16 基座 640 / 768 | 5.41 / 5.52 GB（权重就 5.07 GB） | 24.1 / 28.2 s/micro-batch | ❌ **否决**：卸掉优化器状态也放不下，无法训练 |
| 超分 GAN（RRDBNet x4，HR 256） | 2.46 GB | 0.53 s/it | ✅ **采纳**（4000 迭代 37 分钟；细节量 6.5–8.7× bicubic） |
| 免训练融合（linear / TIES / DARE / SLERP） | 0（**纯 CPU**） | 约 2 分钟 | ✅ **采纳**（4 个产物全部优于 V4，见 §9） |

**决策结论**：在 6GB 硬约束下，**可训练**的只有 SD1.5 系（LoRA / DoRA / TE 微调）与超分 GAN；
SDXL 只能用于**推理**（`tools/probe_sdxl_infer.py` / `tools/compare_sdxl.py`），训练需要云端 GPU。
因此"质量最优且可行"的路线 = **SD1.5 LoRA(V5) + 真微调 CLIP(V5b) + 免训练融合 + 超分 GAN**，
其中融合与超分都能在 CPU 上完成，不占用显卡时间。

> 反例提醒：如果当初没有实测，只按"论文里 SDXL 更好"去跑 2500 步，结果是**一周也跑不完**
> —— 这正是 §10.3 记录的那次误判（探针把网络故障当成显存不足）值得单独成节的原因。

## 4. 复现命令

```powershell
$env:LANDSCAPE_ROOT = (Get-Location).Path
$env:HF_HOME = "$env:LANDSCAPE_ROOT\models\hf_cache"
$py = "<你的 python>"        # 本项目使用 ldm 环境

# 0) 下载 SDXL 基座（fp16 unet/两个文本编码器/vae，共 6.46GB）
& $py tools/fetch_models.py sdxl

# 1) V5（SD1.5）训练 —— 数据用已有 640 缓存，无需重建
& $py train_v5.py --config config_v5.cfg

# 2) SDXL 数据缓存（1024，SDXL VAE 缩放因子 0.13025，文本嵌入预计算）
& $py precompute_v5.py --base SG161222/RealVisXL_V5.0 --arch sdxl --res 1024 `
      --out dataset1024/cache_v5_sdxl1024.pt

# 3) SDXL 显存可行性探针（逐档试 8bit-512/768/1024、fp16-640/768/1024）
& $py tools/probe_sdxl_train.py

# 4) SDXL 训练（QLoRA 路线）
& $py train_v5.py --config config_v5_sdxl.cfg

# 5) 客观对比 + 人工可看的对比拼版
& $py tools/compare_adapters.py --adapters "V4:models/v4_640/adapter_best" "V5:models/v5_lora/adapter_best" `
      --out research/compare_v5 --steps 24 --res 512
& $py tools/test_adapter_load.py models/v5_lora/adapter_best      # 确认能被推理栈加载
```

## 5. 在共享的 6GB 卡上无人值守训练

这张卡同时在跑系统桌面、浏览器、本地 LLM 服务与游戏。实测：当其它进程把显存吃到只剩 0.3GB 时，
训练会从 **1.35 s/step 退化到 58 s/step（慢 43 倍）**，而且**不会报错**——只是"看起来在跑"。为此加了三层防护：

| 机制 | 位置 | 作用 |
|---|---|---|
| 显存预检 | `train_v5.py` 启动时打印 `VRAM free x / y GB`，低于 `min_free_gb`(默认 2.0) 时告警 | 一开始就知道卡被占了 |
| 停滞看门狗 | 训练循环内，>240s 没有完成一次优化步就打印 `[stall] ...`（附当前空闲显存） | 把静默劣化变成显式告警 |
| OOM 退避 | OOM 不再跳过样本空转，而是退避 20s 重试；连续 5 次则存档退出 | 抢占发生时不浪费算力 |
| 断点续训驱动 | `tools/train_driver.py` | 训练退出后自动 `--resume latest` 重试，最多 20 次；每轮前等待显存恢复到阈值 |
| 全链路流水线 | `tools/train_pipeline.py` | V5 结束 → SDXL 显存探针 → 按可行分辨率建缓存 → **4 步冒烟门** → 启动 SDXL 训练，全程写 `models/pipeline.log` |
| 分辨率决策单测 | `tools/test_pipeline_logic.py` | 11 项：确保探针输出被正确解析（选错分辨率 = 白跑一整夜） |
| 长跑冒烟门 | `train_pipeline.py` 步骤 3.5 | 用**真实缓存**跑 4 步训练 + 1 次验证 + 1 次导出，全通才启动 2500 步长跑；失败只花 3 分钟 |

```powershell
# 无人值守跑完整条链（V5 → SDXL）
& $py tools/train_pipeline.py --v5-out models/v5_lora --sdxl-config config_v5_sdxl.cfg `
      --res-preference 1024,896,768 --crops 3 --wait-v5-hours 8
```

> 经验：训练期间关闭本地 LLM 服务与游戏，速度差异是数十倍；如果只能共用，就用上面的驱动，
> 它会自己等显存、自己续训，但总时长按竞争程度放大。

## 6. 暂停点与恢复（2026-09-14 14:50：**实验全部完成，结论为否证**）

**当前状态**：所有 GPU 作业已停止（显存 164 MiB = 桌面基线），无遗留进程。
caption 对照实验（V6q）已完成并评测；读者应从 **`docs/FINAL_VERDICT.md`**（四条独立判据并列）
读结论，从 §11.1 读 caption 实验的负结果，从 §10.4 读"SDXL 连推理也不可行"的实测。

**结论摘要**：SD1.5 微调路线（换优化器 / DoRA / 融合 / 补训 TE / 重写 caption）在本机 6GB + 现有数据下
**全部只到"与 V4 打平"**；SDXL 既不可训也不可推理。唯一被客观证据支持的质量提升是**超分 GAN**
（细节量 6.5×/8.7×，已作为三个入口的默认模型交付）。

| 项 | 状态 |
|---|---|
| **V5 (SD1.5)** | ✅ 已完成 4000/4000 步；`models/v5_lora/{adapter,adapter_best}` |
| **V5b（UNet LoRA + 真微调 CLIP）** | ✅ 已完成 1200 步（39.5 分钟，1.95 s/step，4.46 GB）；`models/v5b_lora/adapter_best{,_text_encoder.pt}`；部署门禁 PASS（TE `max|Δ|=0.00195`，V4 为 0.00000） |
| **超分 GAN x4** | ✅ 已完成 4000 迭代；最终探针：细节量 6.53×（EMA）/8.69×（final）vs bicubic，与 RealESRGAN/UltraSharp 同档 |
| **免训练融合（6 个产物）** | ✅ `models/merged/*_0.50_0.50`（V4⊕V5 四种）+ `models/merged_v4v5b/*`（V4⊕V5b 两种），全部通过 `verify_merge.py` |
| **固定协议 val（10 候选 + 配对检验）** | ✅ `research/eval_val.csv` + `eval_val_paired.csv`（基准 V4）：V5b 0.18583 < V5 0.18589 < slerp 0.18590 < … < V4 0.18644；**但效应量仅 0.3%，画质指标区分不开，不能据此声称"质量更好"** |
| **新判据（留出 prompt 的 KID/CLIP-FID）** | ✅ 已完成（`research/fid_v6/fid_metrics.csv`）：V4 KID 最低（0.000359），其余 0.000377–0.000404，**差异小于 KID 标准差 3.5e-05** |
| **caption 对照实验（V6q）** | ✅ **已完成**：重标注 1649/1664（90 分钟，增量落盘 + 看门狗）、缓存 640/3 crops、训练 4000 步（108.9 分钟）；**结论：画质没有变好**（见 §11.1） |
| **盲测裁判（自研）** | ✅ 三轮全部「无法区分」：V4 vs V5b 62.5%、V4 vs V6q 47.9%、V5b vs V6q 45.8%（Wilson 95% 区间均跨 50%） |
| SDXL 主线 | ❌ 实测判定不可行：训练 2500 步需 125–180 小时；**推理同样不可行**（峰值 5.5–5.9 GB 撞墙，见 §10.3/§10.4） |

### 暂停期间修好的三个真问题

1. **续训数据顺序不是纯函数**（`DataLoader(shuffle=True)` 每 epoch 重抽排列）→ 已改为
   `(seed, step, micro)` 的确定性采样；续训一致性差异从 **1.5e-3 降到 5.4e-4**。
   残留量级疑似**进程间浮点不确定性**（同配置重跑也会有），诊断脚本已加对照组 D。
2. **推理脚本显存不省** → `compare_adapters` / `eval_fid` 现在 VAE 分块+切片常开、换 adapter 时
   释放 UNet/hook、余量不足自动切 CPU offload（否则 6GB 卡上会 WDDM 抖动：16 分钟 0 张且不报错）。
3. **`--plan` 已实现**：零显存回答缓存/检查点/工作量/上次实测速率。

### 恢复命令（按需选择）

```powershell
$py = "<你的 python>"; $env:LANDSCAPE_ROOT = (Get-Location).Path
$env:HF_HOME = "$env:LANDSCAPE_ROOT\models\hf_cache"

# A) 只要 SR GAN 的最终评测证据（3-5 分钟 GPU，最轻）
& $py tools/probe_sr_model.py --images 8 --out research/sr_probe_final `
      --models "bicubic:" "RealESRGAN:models/sr/RealESRGAN_x4plus.pth" `
      "UltraSharp:models/sr/4x-UltraSharp.pth" "Ours-EMA:models/sr/landscape_gan_x4_best.pth" `
      "Ours-final:models/sr/landscape_gan_x4.pth"
#    产出：PSNR/SSIM/细节量对比表 + sr_compare_sheet.png（含 HR 参考列）

# B) 合并 adapter 评测（先 --plan 看复用情况，再真跑；每行落盘、可续跑）
& $py tools/eval_val_mse.py --plan --adapters "V4:models/v4_640/adapter_best" `
      "linear:models/merged/linear_0.50_0.50" "slerp:models/merged/slerp_0.50_0.50" `
      "ties:models/merged/ties_0.50_0.50" "dare_ties:models/merged/dare_ties_0.50_0.50" `
      --cache dataset1024/cache_v4_640.pt --csv research/eval_merged.csv

# C) 继续 SDXL 主线（无人值守，6.5-10 小时；含探针→缓存→冒烟→训练）
& $py tools/train_pipeline.py --skip-v5-wait --sdxl-config config_v5_sdxl.cfg `
      --res-preference 1024,896,768 --crops 3

# D) 可选：V5b 文本编码器阶段（需先把 config_v5b.cfg 的 cache 换成 512 版 cache_v4.pt）
& $py tools/train_driver.py --script train_v5.py --config config_v5b.cfg --save_every 200 --min-free 3.0
```

> 说明：`train_pipeline.py --skip-v5-wait` 会跳过"等 V5"直接接管 SDXL；它内部已含**冒烟门**
> （真实缓存 4 步 + 验证 + 导出，全通才启动长跑）。训练中可随时 `Ctrl+C`/停进程，
> 检查点每 250 步落盘，`--resume latest` 可续。
> A/B/C/D 都超过前台单次命令的 10 分钟上限 —— 在 dsh 里必须用后台作业启动（见 `AGENTS.md`）。

## 7. 评测协议

### 7.1 三个层次，别混用

| 层次 | 工具 | 适用 |
|---|---|---|
| **在线 val（训练中）** | `train_v5.py` 内置 | 判断"这一轮有没有在学"。自 2026-09-12 起改为**确定性协议**：固定子集（seed 1234）、固定噪声（seed 4321+idx）、固定时间步 `[50,250,450,650,850]` |
| **跨模型 val（可比）** | `tools/eval_val_mse.py` | 判断"谁更低"。**全部 144 个测试样本 × 5 个固定时间步**，文本条件统一用 base CLIP 的缓存嵌入 |
| **画质与 prompt 跟随** | `tools/compare_adapters.py` | 判断"谁更好看/更听话"：固定 prompt×seed 出图 → Laplacian 方差（细节）、饱和度、对比度、**CLIP 图文一致性**，并输出 `compare_sheet.png` 供人工判断 |

```powershell
python tools/eval_val_mse.py --adapters "BASE:" "V4:models/v4_640/adapter_best" "V5:models/v5_lora/adapter_best" `
       --cache dataset1024/cache_v4_640.pt
python tools/compare_adapters.py --adapters "V4:models/v4_640/adapter_best" "V5:models/v5_lora/adapter_best" `
       --out research/compare_v5 --steps 24 --res 512
```

### 7.2 为什么不用 V4 训练日志里的 `0.1262` 直接比

1. **在线 val 噪声大**：V4/V5 的旧验证每次**随机抽 48/144 样本 + 随机时间步**，噪声可达 ±0.02。
   V5 实测 `0.1783 → 0.1648 → 0.1755` 的起伏就含这份噪声，不足以判定优劣 —— 故新增固定协议的
   `eval_val_mse.py`，并已把 `train_v5.py` 的在线验证也改为确定性。
2. **文本条件不同**：V4 的部署配置是 **UNet LoRA + 微调过的 CLIP 文本编码器**
   （`app.py` 会额外加载 `<adapter>_text_encoder.pt`，240 MB），而 V5 目前只训了 UNet、
   文本编码器冻结。`eval_val_mse.py` 刻意统一使用 base CLIP 嵌入，只比较 UNet 侧适配质量；
   要比较**部署配置**，用 `compare_adapters.py` 出图对比。

> 结论：报告数字时必须写明用的是哪一层协议。**"V5 的 val 是否低于 V4"** 看 §7.1 第二层，
> 而**"V5 是否比线上 V4 更好用"** 看第三层。

### 7.3 部署门禁

任何新 adapter 上线前必须先过 `tools/test_adapter_load.py`（**必须验证权重真的生效**，
而不只是"加载没报错"——peft 格式的 adapter 用 `load_lora_weights` 会被静默忽略，
该脚本通过"固定 seed 出图两次 + 平均像素差阈值"来兜住这种情况）。

### 7.4 评测驱动的工程约束（2026-09-12 事故后加固）

事故：`eval_val_mse.py` 的旧版本**只在全部 adapter 跑完后才写 CSV**。一次 6 个 adapter 的后台评测
跑了 10 分钟被用户要求终止（要用显卡），结果一个字节的证据都没留下，只能全部重来。
现在驱动脚本保证：

| 保证 | 机制 | 怎么验证 |
|---|---|---|
| **被 kill 不丢已完成行** | 每算完一个 adapter 立刻整表落盘（`eval_common.save_rows`） | `python tools/test_eval_plan.py`（91 项，含临时 ROOT 的报告集成测试） |
| **续跑不重算** | `--resume`（默认）按**协议签名**跳过已完成行；签名 = `cache\|arch\|device\|样本数\|时间步` | `--plan` 会逐行标注 reuse / reuse-legacy / compute |
| **不混协议** | 签名不一致一律重算；旧格式行（无 `protocol` 列）需所有可核对项一致才认，标为 `reuse-legacy` | 单测覆盖 6 种不一致情形 |
| **不覆盖正式证据** | 用非正式协议写 `research/eval_val.csv` 会被拒绝并 `exit 2`（`--allow-overwrite-canonical` 可显式放行） | 便宜协议请写 `research/eval_quick.csv` |
| **日志可判断进展** | 行缓冲输出 + 每行打印 `用时 / ETA`；建议 `python -X utf8` 避免 GBK 乱码 | 第一个 adapter 完成后即给出整轮 ETA |
| **小差异必须配对检验** | `--paired --paired-with V4`：同一 (样本, 时间步) 上逐对相减，输出 Δ均值 ± 标准误与 t 值，并写 `research/eval_val_paired.csv` | `tools/eval_common.py::paired_stats`（单测覆盖） |

**为什么必须配对**：本协议**样本间**标准差约 0.19，只看均值的标准误 ≈ 0.19/√720 ≈ **0.007**；
而候选模型之间的真实差异只有 **5e-4** 量级 —— 用非配对口径根本判不出差别，只能靠"谁的数字小一点"下结论。
但两个模型是在同样的样本、同样的噪声、同样的时间步上测的，逐对相减后样本间方差大部分抵消，
差值标准误小两个数量级，t 值才有意义；`docs/EVAL_REPORT.md` §3 会把这配对表直接列出来。

**先看代价再决定跑不跑**（`--plan` 不加载模型、不占显存）：

```powershell
# 正式协议：144 样本 × 5 时间步 = 720 次前向/adapter
python tools/eval_val_mse.py --plan --adapters "BASE:" "V4:models/v4_640/adapter_best" `
       "V5:models/v5_lora/adapter_best" "ties:models/merged/ties_0.50_0.50" `
       --cache dataset1024/cache_v4_640.pt --csv research/eval_merged.csv

# 便宜判定协议：48 样本 × 2 时间步 = 96 次前向/adapter（约 4-6 分钟；必须写进单独的 CSV）
python tools/eval_val_mse.py --adapters "BASE:" "V4:models/v4_640/adapter_best" "ties:models/merged/ties_0.50_0.50" `
       --cache dataset1024/cache_v4_640.pt --limit 48 --timesteps 50,450 --csv research/eval_quick.csv
```

样本子集是"固定随机排列的前 N 个"，所以 `--limit 48` 是 `--limit 144` 的子集，同一 (样本, 时间步)
的噪声也不变——便宜协议与正式协议**方向可比**，但数字不可混进同一张表（工具会用签名/写保护兜住）。
报告工具（`tools/make_eval_report.py`）只把**可比较键一致**的行并进 `docs/EVAL_REPORT.md` 的主表，
协议不同的行单独列出并标注"未并入"。

前台单次命令上限 10 分钟，长评测必须走后台作业：见 `AGENTS.md`「长任务与工具调用超时」。

### 7.6 自研判据：盲测成对偏好裁判（`tools/vlm_judge.py`，2026-09-14）

**为什么自研**：固定协议 val 与训练目标同源（都是去噪 MSE），训练越久越低 —— 它衡量"拟合得狠不狠"，
不是"画得好看不好看"（V5b val 最低 0.18583，但锐度/CLIP 与 V4 区分不开，效应量仅 0.3%）。
KID/CLIP-FID 衡量"像不像真实照片"，仍不看**提示词是否被满足**、也不看构图是否合理。
人类评审最贴近"质量"但不能自动化 → 用本地 VLM 当裁判（VLM-as-a-judge，思路同 ImageReward / PickScore）。

**算法（三个防作弊要点，缺一个结论就不可信）**：

1. **盲测**：每题把两张图**随机左右交换**（固定种子决定），裁判不知道哪边是谁 → 抵消 VLM 的位置偏好；
2. **强制选择 + 允许平局**：只接受 `WINNER: A / B / TIE`，解析失败的样本**单独统计**（不静默丢弃）；
3. **统计口径**：胜率给 **Wilson 95% 区间**，**区间跨过 50% 就是"无法区分"** —— 不许拿 52% 说"更好"。

裁判只看到「提示词 + 两张图」，被要求在三个维度比较（场景是否符合提示词 / 细节与清晰度 / 整体观感）。

```powershell
python tools/vlm_judge.py --a "V4:models/v4_640/adapter_best" `
       --b "V5b:models/v5b_lora/adapter_best" --c "V6q:models/v6_qwen/adapter_best" `
       --out research/judge_final --steps 24 --res 512 --seeds 2
python tools/vlm_judge.py --dry-run        # 只看题目与用法，不加载任何模型
```

产物：`judge_verdicts.csv`（每题一行 + 每对一行汇总）、`judge_meta.json`（提示词/种子/交换种子）、
`images/<候选>/` 原图。纯逻辑（Wilson 区间、判决解析）由 `tools/test_sr_and_judge.py` 覆盖。

### 7.5 共享显卡上的"卡死"：看门狗与降配（2026-09-13 事故）

在 6GB 笔记本卡上，显存超配**不会**报错，而是让进程**静默卡死**：

| 现象 | 2026-09-13 V5b@640 实测 |
|---|---|
| 步进 | 180 秒 **0 步**推进（metrics.csv 时间戳不动） |
| GPU | 利用率 100%，显存常驻 5.0 / 6.0 GB |
| CPU | 进程仍占用 100% 单核（不是死锁，是驱动层反复失败重试） |
| 日志 | 没有任何异常、没有 OOM —— 只能靠"心跳不动"发现 |

三道防线（都已落地）：

1. **`tools/train_driver.py --stall-seconds 300`**：监控 `<out_dir>/metrics.csv`
   （从 `--config` 的 `out_dir` 自动推导）的 mtime，超过阈值即判定卡死、终止子进程并按
   `--resume latest` 重启。此前 driver 只能等进程退出，卡死时就会一直等到天亮。
2. **降配重跑**：把 640 缓存换成 512（`cache=dataset1024/cache_v4.pt`）使激活内存减少 36%，
   并把 `eval_subset` 从 48 降到 8（验证会额外申请一批激活，正是压垮显存的最后一根稻草）。
   实测：**5.0 GB → 4.46 GB，步进 3.4 s → 1.85 s**，从"必然卡死"变成稳定推进。
3. **`--min-free` 门槛**：driver 每次尝试前等显存回到 2.5 GB 以上，避免在桌面占用高峰硬上。

> 经验：在共享显存环境里，"慢"和"卡死"必须区分开。判据是**心跳文件的 mtime**，
> 不是进程是否存在、也不是 GPU 利用率（卡死时这两个指标看起来都"正常"）。

## 8. 超分 GAN（细节化放大）
`train_sr_gan.py` —— 把「放大」从插值变成**重建**。插值只是把像素摊平（细节量为 1×），
GAN 超分则重建高频纹理。方法遵循 Real-ESRGAN（Wang et al., [arXiv:2107.10833](https://arxiv.org/abs/2107.10833)）：

| 组件 | 实现 | 说明 |
|---|---|---|
| 生成器 | RRDBNet（x4，16.7M 参数） | 结构**与官方 x4plus 逐层一致**，实测 `missing=0 unexpected=0` 可直接载入官方权重做初始化 |
| 判别器 | U-Net + 谱归一化 | 3 次下采样 + 3 次上采样 + 通道对齐跳连（逐像素真伪） |
| 损失 | L1×1.0 + VGG19 感知×1.0 + 对抗×0.1 | 与 Real-ESRGAN 配方一致 |
| 退化 | blur → resize → noise（`real` 再叠一层二阶） | **关键**：只训 bicubic 逆运算学不到"还原细节" |
| 导出 | `params_ema` | spandrel / `enhance.py` 直接可读 |

```powershell
python train_sr_gan.py --iters 4000 --hr-size 256 --batch 3      # 约 35 分钟（0.53 s/it, 2.5GB 显存）
python tools/train_driver.py --script train_sr_gan.py --iters 4000 --hr-size 256 --batch 3   # 带断点续训
python tools/probe_sr_model.py --images 8                        # 对比 bicubic / RealESRGAN / UltraSharp / Ours
```

### 8.1 已并入生产线（2026-09-14）

自训权重不再只是"评测产物"，而是三个入口的**默认超分模型**：

| 入口 | 改动 | 位置 |
|---|---|---|
| 高清流水线 | 注册 `ours`（EMA，验证集最优）与 `ours_final`（末轮，细节更强） | `enhance.py::SR_MODELS` |
| Gradio 界面 | 不再硬编码 RealESRGAN，改走同一注册表，可用 `SR_MODEL` 环境变量切换 | `app.py::get_sr()` |
| FastAPI 后端 | 默认值 `ultrasharp → ours`；新增 `field_validator` 夹住 `sr_model` | `server.py::GenerateReq` |
| 网页前端 | 下拉加入"自训·风景（推荐）/ 自训·末轮"，并把它作为写实风景的推荐项 | `site/index.html`、`site/js/advisor.js`、`site/js/config.js` |

**顺带修掉一个真实的安全缺口**：`sr_model` 直接来自 HTTP 请求体、又会被用来**打开文件**，
原实现 `SR_MODELS.get(name, name)` 等于允许客户端传任意路径。现在
`enhance.resolve_sr_path()` 只接受已注册名字或 `models/sr/` 内的文件名，越界一律 422，
错误信息不回显路径（`tools/test_sr_and_judge.py` 覆盖 9 种输入，含目录穿越与空值）。

> 部署形态一致性由 `python tools/deploy_check.py` 与 `tools/test_sr_and_judge.py` 两道 CPU 门禁保证。

**实测（iter≈500 的中途检查点，2 张测试图，HR 256 → LR 64 ×4）**

| 模型 | PSNR | SSIM | 细节量（Laplacian，0-255 量纲） | 相对 bicubic |
|---|---|---|---|---|
| bicubic | 23.19 dB | 0.7284 | 461 | 1.00× |
| RealESRGAN_x4plus | 20.71 dB | 0.7167 | 2337 | 5.06× |
| 4x-UltraSharp | 21.74 dB | 0.7400 | 2021 | 4.38× |
| 本项目 GAN（中途） | 21.12 dB | 0.7228 | 2239 | 4.85× |

> 判读：GAN 把细节量拉到插值的 4-5 倍，代价是 PSNR 低约 2 dB —— 这是感知-失真权衡，
> 说明模型在"重建纹理"而不是"拟合像素"。**是否更好看必须结合 `sr_compare_sheet.png` 人工确认**。

**踩坑记录（都由冒烟测试抓出）**：`spectral_norm` 只能包卷积（套 LeakyReLU 会 `KeyError: 'weight'`）；
判别器必须是 3 下采样（不是 8），且跳连两端通道要对齐；自写 SSIM 的通道轴要用 `shape[-3]`。

## 9. 免训练模型融合（Model Soup / TIES / DARE / SLERP）

V4 与 V5 是**同一基座、同一份数据、不同配方**得到的两个解，各自在验证集上打平。这类
"同一盆地里的多个解"正是权重融合的经典适用场景（零算力、零推理开销）：

| 方法 | 论文 | 思路 |
|---|---|---|
| Model Soups | Wortsman et al., ICML 2022, [arXiv:2203.05482](https://arxiv.org/abs/2203.05482) | 直接加权平均多个微调权重，常超过任一单模型 |
| **TIES-Merging** | Yadav et al., NeurIPS 2023, [arXiv:2306.01708](https://arxiv.org/abs/2306.01708) | 裁剪小幅度参数 → 符号多数投票消解冲突 → 仅合并同号项 |
| **DARE** | Yu et al., 2023, [arXiv:2311.03099](https://arxiv.org/abs/2311.03099) | 随机丢弃增量并重缩放，再走 TIES（`dare_ties`） |
| SLERP | 球面线性插值（社区标准做法） | 沿球面插值，保持模长；PEFT 不支持故本项目自实现 |

```powershell
python tools/merge_lora.py --adapters models/v4_640/adapter_best models/v5_lora/adapter_best `
       --methods linear slerp ties dare_ties --weights 0.5 0.5 --out models/merged
# 产物为**标准 PEFT 目录**，因此 app.py 与所有评测工具零改动可用：
#   models/merged/{linear,slerp,ties,dare_ties}_0.50_0.50/
python tools/eval_val_mse.py --adapters "V4:models/v4_640/adapter_best" "ties:models/merged/ties_0.50_0.50" `
       --cache dataset1024/cache_v4_640.pt
```

**实测（固定协议，144 样本 × 5 时间步）**

| adapter | mean MSE ↓ | Δ vs V4 |
|---|---|---|
| V4 | 0.18644 | — |
| V5 | 0.18589 | −0.00055 |
| linear 0.5/0.5 | _见 `research/eval_merged.csv`_ | |
| slerp 0.5/0.5 | 同上 | |
| ties 0.5/0.5 | 同上 | |
| dare_ties 0.5/0.5 | 0.18604 | −0.00040 |

**实测（2026-09-13，固定协议 144 样本 × 5 时间步，`research/eval_merged.csv`）**

| 模型 | mean MSE ↓ | Δ vs V4 | 备注 |
|---|---|---|---|
| V5 | 0.18589 | −0.00055 | SD1.5 最优 |
| slerp 0.5/0.5 | 0.18590 | −0.00054 | 自实现球面插值 |
| linear 0.5/0.5 | 0.18597 | −0.00047 | SVD 保留 99.88% 能量 |
| ties 0.5/0.5 | 0.18603 | −0.00041 | SVD 保留 96.73% |
| dare_ties 0.5/0.5 | 0.18604 | −0.00040 | SVD 保留 73.15%，实际丢弃率 0.50 |
| **V4（部署基线）** | 0.18644 | — | |
| BASE（未挂 adapter） | 0.19153 | −0.00509 | |

> 所有融合产物都**低于 V4**，但优势只有 4e-4 量级 —— 与 V5 单模型（0.18589）基本重叠。
> 这就是"SD1.5 在这份数据与算力下已饱和"的又一证据：融合不是质量跃升的来源，SDXL 才是。

### 9.1 ⚠️ 事故记录：PEFT 融合静默产出 no-op（2026-09-13）

第一次评测 `models/merged/*` 时，`linear` 与 `ties` 的 val = **0.19153**，与未挂 adapter 的
BASE **逐位相同**（Δ vs V4 = −0.00000）。根因：

1. `LoraModel.add_weighted_adapter` 在本环境静默失败，`lora_B` 全部停留在**零初始化**
   （LoRA 的 B 初值就是 0）→ 增量 `B @ A = 0` → 产物恒等于基座；
2. 旧门禁只检查"键集合 / 形状一致"，对这种失败**完全免疫**（现在仍全部一致）；
3. 唯一真正生效的是自实现的 `slerp`（0.18590），它掩盖了问题，看起来像"只有球面插值有用"。

**修复**：不再依赖 PEFT 的融合实现，改为自己算 LoRA 算术 —— 每个模块取等效增量
`delta = (alpha/r) * (B @ A)`，在**增量空间**做加权 / TIES / DARE，再用 SVD 重压缩回目标秩
（`B = U√S`，`A = √S·Vᵀ`），导出仍是标准 PEFT 目录（`app.py` 与所有评测工具零改动）。
SVD 是最优秩逼近，因此"压秩损失"有明确上界：`相对误差 ≤ sqrt(1 − 保留能量)`。

**新增两道门禁**（都在 CPU 上跑，不需要 GPU）：

```powershell
python tools/verify_merge.py --sources models/v4_640/adapter_best models/v5_lora/adapter_best `
       --merged models/merged --weights 0.5 0.5      # 结构 + 等效增量双重校验，必须全 PASS
python tools/test_merge_math.py                       # 28 项：融合算术与本次事故的回归测试
```

`verify_merge.py` 的判据落在**数值**上（键集合与形状一致并不足以证明适配器有效）：
不存在"期望非零却整体为 0"的 `lora_A`/`lora_B`、秩与 `adapter_config.json` 的 `r` 相符、
`lora_alpha != 0`（否则缩放系数为 0，同样恒等于 no-op），并且 `linear` 的**等效增量**
`B@A` 必须等于 `Σ wᵢ·deltaᵢ`（当前实测平均相对误差 2.78%，即秩 64 压缩秩 128 目标的代价）。

> 融合全程 **CPU 完成**（约 2 分钟，含 SVD），可与 GPU 训练并行；`--out` 相对项目根解析。
> `--rank` 可指定产物秩（默认沿用源秩 64；调到 128 可无损表达两个秩 64 适配器之和，代价是推理变慢）。

## 10. 已知局限与下一步

1. **caption 质量是当前最大瓶颈**：数据集用的是 BLIP-base 自动 caption（偏短、句式模板化）。
   结构化 caption（主体/环境/光线/时间/镜头）通常比换优化器带来的提升更大 ——
   现成工具：`tools/dataset/caption_images.py`（旧 BLIP 方案）与根目录 `recaption.py`（Qwen2-VL 重写，推荐）。
2. **分辨率**：SD1.5 在 640 上工作良好，但架构上限明确；真正的质量跃升在 SDXL 1024（进行中）。
3. **多卡/更大模型**（Flux/SD3.5/Sana）在 6GB 上不具备可训练性，除非用云端 GPU（`server_cloud.py` 已预留）。

### 10.1 证据进度（2026-09-13 全面续训）

| # | 项目 | 状态 | 结果 / 命令 |
|---|---|---|---|
| 1 | **SR GAN 最终权重探针** | ✅ 已完成（39 秒） | `research/sr_probe_final/`：Ours-EMA 细节量 **6.53×**（PSNR −1.45 dB）、Ours-final **8.69×**（−1.56 dB）、RealESRGAN 7.09×（−1.81 dB）、UltraSharp 8.03×（−1.29 dB）→ 自训模型与两个商用超分模型同档 |
| 2 | **合并 adapter 评测** | ✅ 已完成（8 分钟） | 4 个融合产物全部低于 V4（见 §9）；修复了 PEFT no-op 事故并加了 `verify_merge.py` / `test_merge_math.py` 两道门禁 |
| 3 | **SDXL 主线** | ❌ 实测判定不可行 → 已转为 V5b | 探针实测 50-65 s/micro-batch（int8），2500 步配方需 **125-180 小时**，见 §10.3；改为执行第 4 项 |
| 4 | **V5b 文本编码器阶段** | ✅ **已完成**（39.5 分钟，1200/1200 步） | 512 缓存 + `base_8bit` + `adamw8bit`：实测 **1.95 s/step、4.46 GB**；产出 `models/v5b_lora/adapter_best/` + `adapter_best_text_encoder.pt`（469 MB） |

**V5b 为什么值得单独做**（本项目第一次真正微调文本编码器）：

| 项 | V4 | V5b |
|---|---|---|
| `adapter_best_text_encoder.pt` 与 base CLIP 的 `max|Δ|` | **0.00000**（逐位相同，等于从未微调） | **0.00195**（确实变了） |
| 部署形态 | UNet LoRA（+名义上的 TE） | UNet LoRA **+ 真微调 CLIP** |
| 部署门禁 `tools/test_adapter_load.py` | 通过（UNet 侧） | 通过：256 张量、`mean|Δ|=60.78`、TE `max|delta|=0.00195 APPLIED` |

训练过程的两处工程结论：
1. **640 缓存会卡死**：稳定占用 5.0 GB 触发 WDDM 超配，180 秒 0 步推进（GPU 100%、CPU 空转、无异常）；
   换 512 缓存 + `eval_subset 48→8` 后降到 4.46 GB、步时 3.4→1.95 s，稳定跑完（详见 §7.5）。
2. **续训缺口已补**：检查点现在含 RNG/scaler/协议签名，`--plan` 可在零显存下回答工作量与续训点。

**当前结论（必须如实说明）**：SD1.5 主线上 V4 / V5 / 融合产物全部落在 0.1859–0.1864 的窄带里
（V5 与各融合产物都略低于 V4，但优势仅 4e-4 量级），出图指标（锐度 / CLIP 一致性）也彼此不可区分。
真正可能拉开差距的是 **V5b**：它是本项目第一个"UNet LoRA + 真微调 CLIP"的部署形态
（V4 的 TE 实测与 base 逐位相同，等于从未微调过），是否更好要看部署形态的 CLIP 一致性对比。

### 10.2 ⚠️ 事故记录：SDXL 基座走 HF Hub id，被误判成"6GB 跑不了 SDXL"（2026-09-13）

第一次重启 SDXL 主线时，`tools/probe_sdxl_train.py` 的五个显存模式**全部 FAIL**：

```
8bit-1024    FAIL ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol
...
0/5 configurations fit in 6 GB
```

`train_pipeline.py` 据此打印 "no feasible SDXL mode" 并中止整条主线。真实原因与显存无关：

1. 该脚本用 HF Hub id `SG161222/RealVisXL_V5.0` 直接 `from_pretrained`，diffusers 认为需要取文件，
   于是联网执行 "Fetching 2 files" —— 在代理环境下卡了 **5 分 22 秒**后抛 SSL 错误；
2. 本地 HF 缓存里的那份快照其实是**完整**的（unet / text_encoder / text_encoder_2 / vae /
   两个 tokenizer / scheduler / model_index.json 全在，含 fp16 变体）；
3. 探针把任何异常都归为 `not feasible / peak=0.00GB`，而流水线只区分"有 OK 档位 / 没有"，
   于是**基础设施故障被读成了硬件容量结论**——这是最危险的一类误判：它会让人放弃可行的路线。

**修复**（两层）：

* `tools/sdxl_base.py`：把基座解析到**本地目录**（项目内目录 > HF 缓存里含 `model_index.json`
  与 `unet` 的完整快照 > 原样返回），并统一要求 `local_files_only=True`；`train_v5.py`、
  `precompute_v5.py`、`probe_sdxl_train.py`、`compare_sdxl.py` 全部走它。
* `train_pipeline.py::probe_failure_kind()`：区分 `capacity`（有 OOM）与 `infrastructure`
  （有 FAIL 却一个 OOM 都没有）。后者会明确提示"先修环境，不是显存不足"。
  注意探针把 OOM 也常打印成 `FAIL RuntimeError: CUDA out of memory`，因此识别 OOM 时
  `OOM` 与 `out of memory` 都要认，否则正常的能力结论会被误报成环境故障（单测覆盖）。

> 经验：**任何"资源不够"的结论，都要先排除"取不到资源"**。看到 FAIL 而没有 OOM 时，
> 先怀疑网络 / 路径 / 缓存，再怀疑显存。

### 10.3 SDXL 训练的实测可行性：在 6GB 上**不可行**（2026-09-13 实测）

修好基座路径之后，`tools/probe_sdxl_train.py` 给出的是**真实吞吐**（单次 fwd+bwd，batch=1，
开启梯度检查点；桌面此时已占用约 0.3-2.7 GB 显存）：

| 模式 | 峰值分配 | 单 micro-batch | 权重常驻 |
|---|---|---|---|
| 8bit-1024 | 7.21 GB（**超过物理 6GB → 溢出到共享内存**） | 62.1 s | 2.97 GB |
| 8bit-896 | 6.27 GB | 51.5 s | 2.99 GB |
| 8bit-768 | 5.45 GB | 65.1 s | 2.99 GB |
| fp16-768 | 5.52 GB | 28.2 s | 5.07 GB |
| fp16-640 | 5.41 GB | 24.1 s | 5.07 GB |

判读：

1. **int8 QLoRA 是唯一装得下的形态**（权重 2.97-2.99 GB），但吞吐只有 50-65 s / micro-batch；
   `config_v5_sdxl.cfg` 的 2500 步 × 梯度累积 4 = 10,000 个 micro-batch ⇒ **125-180 小时**。
   即使退到 768 且累积 1（等效 batch 1，与配方意图不符）仍需约 18 小时。
2. **fp16 基座更快（24-28 s）但不可训**：UNet 单权重就 5.07 GB，加上 LoRA 优化器状态必然溢出。
3. 作为对照，同一张卡上的 SD1.5 训练是 **~1.35 s/step（640, batch 1）**；SDXL 的 int8 路径慢约 40 倍
   （bitsandbytes 的 Linear8bitLt 在小批量下反量化开销占主导，且 1024 的激活直接顶爆 6GB）。
4. 结论：**"在 6GB 上把 SDXL LoRA 训到可用"这条路线不成立**（实测，而非估计）。
   可行的替代是云端 GPU（`server_cloud.py` 已预留）或把 SDXL 只用于**推理**
   （`tools/compare_sdxl.py` / `tools/probe_sdxl_infer.py` 已能跑）。

因此本项目把交付重心放在**同一 SD1.5 基座上真正可训、可部署**的改进上：
`V5b = V5 的 UNet LoRA + 真正微调过的 CLIP 文本编码器`。依据是第 5 节那条实测 ——
V4 的 `adapter_best_text_encoder.pt` 与 base CLIP **逐位相同**，说明 V4 从未真正微调过文本编码器；
所以"UNet LoRA + 真微调的 CLIP"是 V4 不具备的能力，而成本只有约 1 小时（实测 3.4 s/step）。

### 10.4 补充实测：**SDXL 连推理也不可行**（2026-09-14）

上一节只说了"训练不可行"。做完 caption 实验后，本机唯一还剩的质量空间就是"换基座"（SDXL 只做推理），
于是按三种配置各试了一次，**全部在同一处撞墙**：

| 配置 | 结果 | 实测显存 |
|---|---|---|
| SDXL 1024，24 步，`--mode offload` | 17 分钟 0 张产出，进程烧 CPU 而 GPU 100% 空转 | 5816 MiB 占用 / **181 MiB 余量** |
| SDXL 768，24 步，`--mode offload` | 7 分钟 0 张产出，同样症状 | 5840 MiB / **157 MiB** |
| SDXL 1024，**Lightning 4 步**（guidance=0，服务形态） | 6 分钟 0 张产出，同样症状 | 5935 MiB / **62 MiB** |

判读：SDXL 管线的峰值（UNet 前向 + VAE 解码上采样）约 **5.5–5.9 GB**，而桌面基线就要占 ~0.5 GB，
6.0 GB 的卡上直接顶到墙；**WDDM 不会报 OOM，而是让进程卡死**（与 §7.5 的 V5b 事故同一机制）。
Lightning 4 步只缩短了时间，没有降低峰值，所以照样卡死。

> 结论修正：**SDXL 在本机既不可训（125–180 小时）也不可推理（峰值撞墙）**。
> 要用 SDXL 必须换硬件或云端 GPU（`server_cloud.py` 已预留）；不要再把"SDXL 只做推理"写成本机可行选项。

## 11. caption 对照实验（V6q，2026-09-14）

> **命名说明**：本节（以及 `config_v6q.cfg`、`models/v6_qwen/`）里的 **V6q** 指**caption 对照实验权重**；
> README 第 17 节的 **V6** 指**产品阶段**（上传安全 / 任务取消 / 工程门禁）。两者无关。


**假设**：所有既有路线（换优化器 Prodigy、换权重分解 DoRA、换 rank、融合、补训 TE）在固定协议 val 上
都只动了 0.3%，出图指标完全区分不开 —— 说明瓶颈不在训练方法，而在**数据标注**。

**数据侧的实测证据**（`tools/compare_captions.py`）：

| 指标 | 旧（BLIP-base） | 新（Qwen2-VL-2B 结构化） |
|---|---|---|
| caption 形态 | 一句话 + 固定模板尾巴（`..., professional landscape photography, <地点>`） | 12 个逗号短语 |
| 词数（中位） | 14 | 23 |
| **具体内容短语数（中位）** | 6 个实词（去掉模板后） | **9 个**（去掉出现在 >15% 图里的 10 个套话短语后） |
| 信息量偏低的图 | 多数 | 20 / 1649 |

**方法（严格控制变量）**：`config_v6q.cfg` 与 `config_v5.cfg` **逐项相同**（r64 / Prodigy lr1.0 /
min-SNR γ=5 / EMA 0.9995 / 640 / accum 2 / 4000 步），唯一差异是 caption 与缓存文件。
重标注由 `tools/recaption_driver.py` 执行（心跳看门狗 + 每 25 张增量落盘）：
**1649/1664 完成，90 分钟**；15 张未完成的是夜景/城市建筑类图（指令禁止臆造建筑，
VLM 反复给出不合格输出而被拒）。

**为什么不能用 val 判胜负**：V6q 的在线 val（0.1891）是在**新 caption** 的条件上测的，
与 V4/V5 的 0.18589/0.18644（旧 caption）**不是同一个协议**，数值不可比。判定必须用与训练目标无关的判据：

1. **留出 prompt 的 CLIP 一致性**（`tools/eval_fid.py`）—— 模型没见过的 24 条提示词上的 prompt 跟随度；
2. **KID / CLIP-FID**（同上）—— 生成图与真实风景照的分布距离；
3. **盲测成对偏好裁判**（`tools/vlm_judge.py`）—— 本地 VLM 盲测，Wilson 95% 区间跨 50% 即"无法区分"。

> 已知副作用（必须写进结论）：新 caption 的情绪词高度重复（`tranquil` 553 次、`serene` 542 次、
> `peaceful` 386 次），即"地点模板"部分被换成了"情绪模板"。但它只占 12 个短语中的 1-3 个，
> 具体内容短语数仍从 6 提升到 9，因此判定该实验**仍然有效**，同时保留这条局限。

### 11.1 结果：caption 变好了，画质没有变好（2026-09-14）

V6q（与 V5 逐项同配方，唯一变量 = caption）训练完成 4000 步（108.9 分钟，峰值 2.36 GB），
用**与训练目标无关**的判据在留出 prompt（24 条自拟提示词 × 2 seed = 48 张）上评测：

| 候选 | KID ↓（±σ） | CLIP-FID ↓ | 留出 prompt CLIP ↑ | 锐度 ↑ | 盲测胜率 vs V4 |
|---|---|---|---|---|---|
| **V4（部署基线）** | **0.000359**（3.5e-05） | **0.4646** | 0.2716 | 698.0 | — |
| V5b | 0.000377（3.5e-05） | 0.4649 | 0.2720 | 708.1 | 37.5%（CI 25.2%–51.6%）→ 无法区分 |
| V6q（结构化 caption） | 0.000396（3.6e-05） | 0.4722 | 0.2719 | **742.1** | 52.1%（CI 38.3%–65.5%）→ 无法区分 |
| V5 | 0.000404（3.6e-05） | 0.4848 | **0.2726** | 632.5 | — |

**结论（这是本项目最重要的负结果，必须原样保留）**：

1. **caption 的改造确实生效**（信息量 6 → 9 个具体短语），但**没有转化为可检测的画质提升**：
   KID 差 1.8e-5 远小于其标准差 3.5e-05；留出 prompt 的 CLIP 一致性差 0.0003（噪声级）；
2. **盲测裁判三轮全部"无法区分"**：V4 vs V5b（V4 62.5%，CI 48.4%–74.8%）、
   V4 vs V6q（V4 47.9%，CI 34.5%–61.7%）、V5b vs V6q（V5b 45.8%，CI 32.6%–59.7%）；
3. **没有任何候选在盲测中胜过 V4**；V6q 甚至是**最锐的**（742 vs 698）却没被裁判偏好 ——
   说明"更锐"不等于"更好看"；
4. 结合固定协议 val（V5b 0.18583 < … < V4 0.18644，效应量 0.3%）与视觉逐格判读
   （`docs/VISUAL_REVIEW.md`：V4/V5 无法区分），可以给出**明确判断**：
   **在本机 6GB + 这份数据/算力下，SD1.5 的微调（换优化器 / 权重分解 / 融合 / 补训 TE / 重写 caption）
   都已到天花板，产出的是"与 V4 打平"而非"更好"。**

因此本项目对目标里"产出比 V4 质量更高的权重"这一条给出的是**否证**，而不是勉强的正例。
真正被客观证据支持的质量提升只有一处：**超分 GAN（细节量 6.5×/8.7×，已并入生产线）**。
