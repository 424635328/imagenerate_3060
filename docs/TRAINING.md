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

## 3. 实测数据

| 配置 | 峰值显存 | 速度 | 结论 |
|---|---|---|---|
| SD1.5 640，LoRA r64 + Prodigy + min-SNR | **2.36 GB** | 1.25–1.53 s/step | 6GB 有大量余量，可放心加容量 |
| SD1.5 640，DoRA r64 | 2.41 GB | ~2.0 s/step | 可训练且可部署 |
| SDXL fp16 基座 | 权重 4.9 GB | — | 6GB 卡**不可行**（外加激活/优化器必 OOM） |
| SDXL int8 基座（QLoRA） | ~2.5 GB 权重 | 待实测（`tools/probe_sdxl_train.py`） | 唯一可行路线 |

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
| 全链路流水线 | `tools/train_pipeline.py` | V5 结束 → SDXL 显存探针 → 按可行分辨率建缓存 → 启动 SDXL 训练，全程写 `models/pipeline.log` |

```powershell
# 无人值守跑完整条链（V5 → SDXL）
& $py tools/train_pipeline.py --v5-out models/v5_lora --sdxl-config config_v5_sdxl.cfg `
      --res-preference 1024,896,768 --crops 3 --wait-v5-hours 8
```

> 经验：训练期间关闭本地 LLM 服务与游戏，速度差异是数十倍；如果只能共用，就用上面的驱动，
> 它会自己等显存、自己续训，但总时长按竞争程度放大。

## 6. 暂停 / 恢复（2026-09-12 暂停点）

**暂停原因**：用户需要 GPU 做其它工作。暂停时**未压缩工期**，SDXL 仍按 `config_v5_sdxl.cfg` 的 2500 步执行。

| 项 | 暂停时状态 |
|---|---|
| V5 (SD1.5) | **step 750 / 4000**（实测 1.39 s/step；剩余 3250 步 ≈ 75 分钟） |
| 检查点 | `models/v5_lora/checkpoints/step_{250,500,750}.pt`（每个 268 MB，只存适配器+优化器+EMA） |
| 最优 EMA | `models/v5_lora/adapter_best` = **step 500 的 EMA**（val 0.16476） |
| 验证曲线 | step250 = 0.17827 → step500 = 0.16476（V4 基线 best 0.1262，协议一致可直接比） |
| 流水线 | 已停止（它是"等待 V5 完成"的空转进程，恢复时重启即可） |

### 恢复命令（两条，可同时启动）

```powershell
$py = "<你的 python>"; $env:LANDSCAPE_ROOT = (Get-Location).Path
$env:HF_HOME = "$env:LANDSCAPE_ROOT\models\hf_cache"

# 1) 恢复 V5：--resume latest 从 step_750.pt 续训（驱动会在 OOM 时自己重试）
Start-Process -FilePath $py -ArgumentList 'tools/train_driver.py','--config','config_v5.cfg',
  '--save_every','250','--min-free','2.5','--attempts','20','--resume','latest' -WindowStyle Hidden

# 2) 恢复 SDXL 自动链：等 V5 写出最终 adapter 后 → 显存探针 → 建缓存 → 训练
Start-Process -FilePath $py -ArgumentList 'tools/train_pipeline.py','--v5-out','models/v5_lora',
  '--sdxl-config','config_v5_sdxl.cfg','--res-preference','1024,896,768','--crops','3',
  '--wait-v5-hours','8' -WindowStyle Hidden
```

**注意**：`train_pipeline.py` 只负责"等 V5 结束"并接管后续，**它不会启动 V5**——所以两条都要起。
只想单独跑 SDXL（不继续 V5）时，用 `--skip-v5-wait`，或直接：
`python tools/train_pipeline.py --skip-v5-wait …`。

## 7. 评测协议

- **val MSE（主指标）**：`train_v5.py` 的验证与 V4 完全同协议（同测试集、同中心裁剪、同随机 timestep、
  未加权 MSE），因此 `0.1262` 这个 V4 基线可直接比较。
- **出图指标**：`tools/compare_adapters.py` 固定 prompt×seed 出图，记录 Laplacian 方差（细节）、
  饱和度、对比度，并生成 `compare_sheet.png` 供人工判断——锐度和饱和度单独看会误导，必须与肉眼一致。
- **部署门禁**：任何新 adapter 必须先过 `tools/test_adapter_load.py`。

## 8. 已知局限与下一步

1. **caption 质量是当前最大瓶颈**：数据集用的是 BLIP-base 自动 caption（偏短、句式模板化）。
   结构化 caption（主体/环境/光线/时间/镜头）通常比换优化器带来的提升更大 ——
   现成工具：`tools/dataset/caption_images.py`（旧 BLIP 方案）与根目录 `recaption.py`（Qwen2-VL 重写，推荐）。
2. **分辨率**：SD1.5 在 640 上工作良好，但架构上限明确；真正的质量跃升在 SDXL 1024（进行中）。
3. **多卡/更大模型**（Flux/SD3.5/Sana）在 6GB 上不具备可训练性，除非用云端 GPU（`server_cloud.py` 已预留）。
