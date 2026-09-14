# 版本路由与归档架构（设计稿）

> 目标：**下一版新模型能快速、可回滚地上线；旧版本能安全归档，且不破坏历史与证据。**
> 现状基线：已实现"请求级选版本"（`config.ADAPTER_CHOICES` + `GET /models` + 前端下拉 +
> 管线/结果缓存按版本隔离 + 真机像素验证）。本文件设计的是**从"改代码上线"到"翻指针上线"**那一步。

---

## 1. 现在上线一个新版本，要动哪些地方（痛点）

| 步骤 | 今天怎么做 | 问题 |
|---|---|---|
| 登记版本 | 手改 `config.py` 的 `ADAPTER_CHOICES` 字典 | 台账在代码里；没有 hash/base/配方/评测/状态；评审要看 diff 才知道上线了什么 |
| 设默认 | 手改 `DEFAULT_ADAPTER` 或设环境变量 `DEFAULT_ADAPTER` | 与登记分开做，容易忘其一 |
| 生效 | 重启后端（`server.py`），再 `/warmup` 约 30 s | 有停机窗口；重启期间前端生成全部失败 |
| 回滚 | 再改一次代码 + 再重启 | 与上线等价的成本 ⇒ 出事时手忙脚乱 |
| 归档旧版 | 手工挪目录 | 可能留下悬空引用：前端仍可选、历史/缓存失去出处；没有"已归档"状态可表达 |
| 带证据上线 | 靠人自觉把评测写进 docs | 本项目的方法论恰恰是"感觉不可信"，而流程却在鼓励凭感觉上线 |

**结论**：缺的不是"选版本"能力（已有），而是**一份可机读、可校验、可热更新的版本台账**，
以及围绕它的**上线/回滚/归档三个动作**。

---

## 2. 目标架构：版本是数据，上线是翻指针

三句话概括：

1. **单一真相**：`models/registry.json` 记录每个版本的身份、权重校验和、配方、评测、
   状态与通道指针；代码只读它，不再各自维护字典。
2. **不可变 + 可校验**：一旦 `promoted`，权重的 sha256 冻结；服务端加载前校验，
   不符**拒绝加载**（宁可拒绝，也不静默用错权重 —— 本项目被静默 no-op 咬过两次）。
3. **归档是墓碑，不是删除**：权重移入 `_archive/` 或冷存，台账保留条目与 hash，
   历史记录照常可追溯"这张图是哪个版本出的"。

### 2.1 台账结构（`models/registry.json`）

```jsonc
{
  "schema": 1,
  "channels": { "default": "v5b", "previous": "v4", "staging": null },
  "versions": {
    "v5b": {
      "label": "V5b", "slogan": "V5 + 微调 CLIP",
      "state": "promoted",                          // candidate | promoted | superseded | archived
      "arch": "sd15",
      "base":    { "id": "base_rv6", "sha256": "…" },
      "adapter": { "dir": "v5b_lora/adapter_best", "sha256": "…", "bytes": 152_000_000 },
      "text_encoder": { "file": "v5b_lora/adapter_best_text_encoder.pt", "sha256": "…" },
      "recipe":  { "config": "config_v5b.cfg", "steps": 1200, "res": 512, "optimizer": "adamw8bit" },
      "eval": {                                     // 全部复用现有评测脚本的产物
        "val_mse": 0.185834, "kid": 0.000377, "clip": 0.2720,
        "human": { "decided": 48, "rate": 0.312, "ci": [0.199, 0.453], "verdict": "无法区分" },
        "recorded_at": "2026-09-15T…"
      },
      "created_at": "2026-09-13T…", "promoted_at": "2026-09-15T…",
      "notes": "唯一变量 = 文本编码器是否真的被微调"
    },
    "v5": { "state": "archived", "archived_at": "…", "cold_store": "D:\\lsart_archive\\v5.tar.zst" }
  }
}
```

**状态机**（只允许这些转移，非法转移由门禁拒绝）：

```
candidate ──promote──► promoted ──supersede──► superseded ──archive──► archived
    ▲                      │                                              │
    └──────────────────────┴──────────── restore ─────────────────────────┘
```

不变式（写成断言，进 `tools/test_registry.py`）：

- `channels.default` 必须指向一个 `promoted` 且**权重在盘上、hash 对得上**的版本；
- `channels.previous` 同样必须在盘上（**永不归档**，回滚要用）；
- `archived` 版本不得出现在 `channels.*` 里；
- 每个版本的 `adapter.sha256` 与磁盘实际文件一致（promote 之后不允许再改文件）；
- 台账里不出现机器绝对路径（`dir` 一律相对 `models/`）。

### 2.2 热路由：翻指针 + 预热（不做双缓冲）

- 服务端启动读台账，并**watch mtime**（或提供 `POST /reload`）：`channels` 一变，
  `/models` 立刻反映新清单与状态，前端下拉自动更新，**不需要重启进程、不需要改代码**。
- **驻留权重仍在下一个请求时切换**：6 GB 卡放不下两条 SD1.5 管线（两条 ≈ 4–5 GB，
  实测撞过 WDDM 静默卡死），所以不做"新管线预热好再切"的双缓冲。
- 代价用**预热前置**消除：`promote` 的过程本身就 `/warmup?adapter=新版本`，
  把 ~30 s 加载成本在上线动作里付掉；用户第一次点生成时它已经是热的。
- `previous` 版本必须留在盘上 ⇒ 回滚同样可以"翻指针 + 预热"，秒级完成、无需重新训练或重新下载。

### 2.3 归档策略（笔记本磁盘 + 6 GB 卡的现实约束）

| 规则 | 内容 |
|---|---|
| 永不归档 | `channels.default`、`channels.previous` |
| 可归档 | `superseded` 且 90 天未被使用；或被新版本取代且评测/人工判据都不支持它 |
| 归档动作 | 移到 `models/_archive/<id>/`（或 `--cold-store <盘外目录>` 打包 `.tar.zst` + 清单），台账置 `state: archived` + `archived_at` |
| 归档前检查 | 不是 default/previous；`/jobs` 里没有 running 任务引用它（避免跑到一半权重被挪走） |
| 归档后行为 | `/models` 仍返回墓碑条目（`state: archived`）⇒ 前端灰显、历史与画廊仍能显示"这张是 V5 出的"；结果缓存不删，但每条缓存记录 `adapter_sha`，**恢复时若 hash 变了则该版本缓存全部失效** |
| 恢复 | `restore` 把权重移回 + `state: candidate`，再走一次 promote（需要重新 warmup） |

### 2.4 上线/回滚/归档的四个动作（每个都可脚本化、可进 CI）

```powershell
python tools/registry.py scan                 # 扫描 models/ 建/更新台账：算 hash、认 TE、标 candidate
python tools/registry.py eval <id>            # 跑固定协议 val + 留出集 KID/CLIP，写回 eval 字段（复用现有脚本）
python tools/promote.py <id>                  # 预检 → 冒烟出图 → 翻 default 指针 → 预热新版本
python tools/promote.py --rollback            # 回到 channels.previous（同样翻指针 + 预热）
python tools/archive_version.py <id>          # 归档：墓碑 + 冷存 + 前端自动灰显
```

`promote.py` 的**预检门**（缺一不放过）：

1. 状态是 `candidate`，且权重 hash 与台账一致；
2. **评测记录存在**（val/KID/CLIP；人工判据可选）；
3. 若没有任何判据显示显著优势，工具打印"无法区分"并要求显式 `--ack-indistinguishable` —— 
   **把"新版本 ≠ 更好"写进工具，而不是写进自律**（本项目四条判据的结论正是"无法区分"）；
4. 冒烟出图：与当前 `default` 同 seed 出图，**必须像素不同**（证明不是 no-op）；
5. 翻转 `default`/`previous`，随后 warmup 新版本；把 `promoted_at` 与操作者写进台账。

---

## 3. 与现有组件的衔接（不重复造轮子）

| 现有 | 关系 |
|---|---|
| `tools/eval_val_mse.py` / `eval_fid.py` / `vlm_judge.py` / `human_verdict.py` | 产物 CSV 直接喂 `registry.py eval`，写进 `eval` 字段；不新建评测体系 |
| `site/data/versions.json`（评判台数据） | 生成器改为**从台账取版本列表**，产品侧与评测页共用一份台账 |
| `research/human_judge/*.json` | 人工判据作为 `eval.human` 的来源之一 |
| `_cache_key(...)`（已含版本 slug） | 台账再加 `adapter_sha`：归档后恢复、或有人换了文件，缓存必须失效 |
| `/models`（已实现） | 增加 `state` 与 `sha256` 两个字段即可，前端据此分组与灰显 |
| `jobs` 的 `adapter` 字段（已实现） | 历史追溯的基础，归档后仍可显示 |

---

## 4. 门禁

| 门禁 | 验什么 |
|---|---|
| `tools/test_registry.py`（纯 CPU） | schema、sha256、状态机转移合法性、四条不变式；**用临时目录 + 伪造小权重演练 promote/rollback/archive**，断言非法操作被拒 |
| `tools/test_adapter_routing.py`（已有） | 白名单/穿越/字段贯穿/缓存按版本隔离/前端接线 |
| `tools/test_adapter_switch_live.py`（已有，需 GPU） | 切版本真的改变像素（不是 no-op） |
| `tools/test_promote_live.py`（新增，需 GPU） | promote → 生成 → 像素变化 → rollback → 生成 → 回到原版本；`/models` 的 default 跟着变 |

---

## 5. 分阶段落地（每阶段独立可交付）

| 阶段 | 内容 | 交付判据 |
|---|---|---|
| **P1 已完成** | 请求级选版本：白名单 + `/models` + 前端下拉 + 管线/结果缓存按版本隔离 + 真机像素验证 | 14 道门禁全过；v4 vs v5b 平均像素差 28.83/255 |
| **P2** | 台账 `models/registry.json` + `tools/registry.py`（scan/eval/validate）+ 服务端改读台账 + mtime 热重载 + `/models` 暴露 `state`/`sha256` | 上线从"改代码"变成"改数据"；进程不重启即反映新清单 |
| **P3** | `tools/promote.py`（预检 + 冒烟 + 翻指针 + 预热）与 `tools/archive_version.py`（墓碑 + 冷存） | 上线/回滚各一条命令；归档后历史仍可追溯 |
| **P4** | 前端分组（默认/候选/已归档灰显）+ 历史版本徽标 + 评测脚本输出并入台账 + 文档 | 界面上能直接看出"这张是哪个版本、是否已归档" |

---

## 6. 取舍与风险（写清楚，避免以后被当成"设计缺陷"）

1. **单管线驻留 ⇒ 切版本有 10–30 s 代价**：不做双缓冲是显存决定的（6 GB 撞墙会静默卡死）。
   缓解：promote 时预热；UI 显示"当前驻留版本"（已实现 `/health.pipeline.adapter`）。
2. **台账是新的失败点**：采取"缺失/不符即拒绝加载"，并提供 `registry.py scan` 一键重建；
   绝不允许"台账读不到就退回硬编码默认"这种看起来稳健、实际会静默用错权重的兜底。
3. **归档 vs 回滚冲突**：`previous` 永不归档；归档清单每次 promote 时重新计算。
4. **磁盘预算**：adapter ≈ 150 MB/个、TE ≈ 500 MB（仅 V4/V5b 有）。
   保留 default + previous + 最近 2 个候选 ≈ 1.5 GB 常驻，其余冷存。
5. **别把"新"当"好"**：promote 需要 `--ack-indistinguishable`。这是产品原则，
   不是流程装饰 —— 本项目的四条机器判据 + 48 题人工盲测都指向"无法区分"。
