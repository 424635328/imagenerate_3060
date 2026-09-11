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
  提交前一律 `pwsh -NoProfile -File tools/dev.ps1 check`，它串起编译、JS 语法、前端一致性、路径、脱敏与行尾六道门禁。
- **行尾门禁**：`tools/check_eol.py` 发现任何混合行尾即 `exit 1`（可挂 CI/pre-commit）；
  修复用 `tools/normalize_eol.py`。**`.ps1` 属 Windows 启动器，必须 CRLF。**


## 修改后必跑的检查

```powershell
python -m py_compile app.py server.py server_cloud.py config.py runtime.py enhance.py
python tools/check_frontend.py     # 前端 id / 模块引用一致性
python tools/check_paths.py        # 个人绝对路径与敏感串
python tools/normalize_eol.py      # 行尾统一（应报 0 或仅启动器）
python research/final_check.py     # 脱敏总检（应无敏感项）
node --check site/js/main.js       # 逐个 JS 模块

# 前端逻辑回归（jsdom 可选依赖，未安装会自动 SKIP）
node tools/test_proxy_guard.mjs    # 代理来源守卫（跨站 POST 必须 403）
$env:NODE_PATH="$env:TEMP\lsart-jsdom\node_modules"; node tools/smoke_frontend.mjs
```
