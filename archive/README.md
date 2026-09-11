# archive/ — 旧版本代码归档

这里保存的是**已被取代、但仍有参考价值**的历史版本代码，按版本号分目录。
当前工程只引用仓库根目录的 v5/v6 代码（见 `docs/TRAINING.md` 与 `README.md`）。

| 目录 | 内容 | 为什么归档 | 被什么取代 |
|---|---|---|---|
| `v1_v2/` | `train_text_to_image.py`（diffusers 官方训练脚本）、`prepare_data.py`、`inference.py`、`generate_v2_gallery.py`、`normalize_base.py`、`make_collage.py`、`plot_loss.py`、`upscale.py`、`upscale_batch.py`、`get_lcm_lora.py`、`bench_steps.py`、`diag_prompt.py`、`diag_seed.py`、`smoke_test.py`、`stress_test.py`、`test_enhance.py`、`styles_prompts.txt`、`config.cfg`、`config_v2.cfg`、`logs/` | v1/v2 阶段的一次性脚本：官方脚本训练、单图超分、诊断/压测、v2 配方 | `train_v5.py`、`precompute_v5.py`、`enhance.py`、`tools/` 下的检查脚本 |
| `v3/` | `train_v3.py`、`config_v3.cfg`、`config_v3_640.cfg`、`logs/` | v3 训练器（在 v5 中以 min-SNR 加权、Prodigy、DoRA、缓存文本嵌入全面取代） | `train_v5.py` + `config_v5.cfg` |
| `v4/` | `train_v4.py`、`precompute_v4.py`、`gen_v4.py`、`make_v4_compare.py`、`config_v4.cfg`、`config_v4_640.cfg`、`logs/` | v4 训练器与预计算（其产物 `models/v4_640/adapter_best` **仍是当前线上模型**，脚本本身已被取代） | `train_v5.py`、`precompute_v5.py`、`tools/compare_adapters.py` |
| `v4/report/` | `PROJECT_REPORT.html`（v4 时期项目报告）、`check_docs.py`、`check_report.py`（该报告的校验脚本） | 报告引用的大量样张图位于 `outputs/`，已在仓库清理中删除（见根 `README.md` §12.3） | — |

## 2026-09 清理记录（已删除，不可恢复）

| 删除内容 | 体积 | 说明 |
|---|---|---|
| `models/_archive/`（lora、v2_lora、v3_lora、v3_640、v4_lora、v5_smoke、v5_dora_probe、v4/v4_640 检查点） | **49.5 GB** | 历史版本权重与训练检查点；可由 `archive/*/` 的脚本重新训练得到 |
| `dataset/` | 627 MB | 512 版数据集，与 `dataset1024/` 图像同名同源，属重复副本 |
| `outputs/` | 187 MB | 历史样张/对比图/压测报告，全部可由脚本再生（见 README §12.3） |
| `research/` 下的运行日志与一次性排查脚本 | ~1.7 MB | 79 个文件；`research/` 现只保留可复现检查脚本（`final_check.py`、`scan_secrets.py`、`boot_verify.py` 等） |

合计释放约 **50 GB**。归档目录内的代码**不参与** CI 门禁，但会被 `tools/check_eol.py` 与 `research/final_check.py` 扫描，因此保持与主工程相同的行尾（LF）与脱敏标准。

## 迁移到 tools/ 的脚本（仍在用，只是不再堆在根目录）

| 新位置 | 说明 |
|---|---|
| `tools/dataset/prepare_v4_data.py` | 由源图构建 `dataset1024/`（含 manifest 与 train/test 划分） |
| `tools/dataset/caption_images.py` | BLIP-base 打标（旧方案，保留作回退） |
| `tools/assets/*.py` | 站点素材生成（makoto 图标与预览、pelican、goose 等） |

## 使用归档代码的注意事项

1. 归档脚本里的路径与配置**指向旧的数据/权重布局**（如 `dataset/`、`models/v3_lora/`），直接运行前先核对。
2. 归档代码不参与门禁：`tools/dev.ps1 check` 只编译当前工程（`app/server/server_cloud/config/runtime/enhance`）。
3. 归档文本文件与全仓库一致使用 **LF** 行尾；`tools/check_eol.py` 同样会扫描这里。
4. 不要恢复旧训练脚本去继续训练历史权重：数据缓存格式、文本嵌入与验证协议都已升级（见 `docs/TRAINING.md`）。
