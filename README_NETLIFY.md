# 部署手册：Netlify 前端 + 本机 GPU 后端

> 线上站点：**https://<your-site>.netlify.app**
> 架构要点：Netlify 只托管**静态页面**，模型跑在**你本机的 GPU**；
> 页面通过 **Netlify Function 同源代理** 访问隧道地址 → 你本机的 `server.py`。

---

## 目录

1. [架构与数据流](#1-架构与数据流)
2. [前置条件](#2-前置条件)
3. [第一步：启动本机后端](#3-第一步启动本机后端)
4. [第二步：把后端暴露为公网（隧道）](#4-第二步把后端暴露为公网隧道)
5. [第三步：Netlify 站点与环境变量](#5-第三步netlify-站点与环境变量)
6. [第四步：部署 / 更新前端](#6-第四步部署--更新前端)
7. [第五步：验收清单](#7-第五步验收清单)
8. [日常运维：隧道换域名怎么办](#8-日常运维隧道换域名怎么办)
9. [并发、队列与配额](#9-并发队列与配额)
10. [安全建议](#10-安全建议)
11. [排错手册](#11-排错手册)
12. [文件清单](#12-文件清单)

---

## 1. 架构与数据流

```
浏览器
  │ ① 打开站点，页面自身不含任何后端地址/密钥
  ▼
Netlify 静态站点（site/）
  │ ② 同源请求 /.netlify/functions/proxy?op=health|generate|job|image|chunk
  ▼
Netlify Function（netlify/functions/proxy.js）
  │ ③ 服务端注入  X-API-Key: $API_TOKEN
  │    使用非浏览器 User-Agent（landscape-art-proxy/1.0）
  │    转发到 $BACKEND_URL
  ▼
隧道（ngrok / Cloudflare）  https://xxx  →  http://localhost:8001
  │
  ▼
server.py（FastAPI）
  │ ④ 鉴权 → 限流 → 入队 → 单 worker 串行生成 → WebP 预览/结果落盘 results/<job_id>.webp
  ▼
GET /preview/{id}（基础图完成即返回 WebP 预览）；GET /result/{id}（最终 WebP）；GET /chunk/{id}?off=&length=（并行分块，用于大图）
```

**为什么必须走 Function 代理，而不是让页面直接请求隧道？**
1. **隐藏敏感信息**：隧道地址与 `API_TOKEN` 只存在于 Netlify 服务器端环境变量，浏览器拿不到。
2. **绕过 ngrok 免费版告知页**：免费 ngrok 对浏览器请求会插入「Visit Site」拦截页，
   代理用**非浏览器 UA** 请求即可绕过。
3. **同源**：页面与代理同域，避免 CORS，也便于分块拼接大图。

---

## 2. 前置条件

| 项 | 要求 |
|---|---|
| 本机 | Windows + NVIDIA GPU（本项目按 **RTX 3060 6GB** 调优），Python 环境 `ldm` |
| Python | `<PYTHON>` |
| 项目根 | `$ROOT` |
| 模型 | `models/base_rv6/`（本地 safetensors）+ `models/v4_640/adapter_best/`（+ `_text_encoder.pt`） |
| 隧道 | 已安装并登录 **ngrok**（`ngrok version` 可执行），或 `cloudflared` |
| Netlify | 已有站点 + 个人访问令牌（`NETLIFY_AUTH_TOKEN`，**不要写进任何文件/仓库**） |

---

## 3. 第一步：启动本机后端

`server.py` 是 FastAPI 服务，默认监听 `127.0.0.1:8001`。

```powershell
cd $ROOT
$env:API_TOKEN    = '<你的强随机密钥>'                                        # 与 Netlify 站点环境变量保持一致
$env:LORA_ADAPTER = '$ROOT/models/v4_640/adapter_best' # 线上要用的权重
<PYTHON> server.py
```

启动成功的标志（stdout）：
```
LANDSCAPE·ART backend on http://127.0.0.1:8001  (token: <你的强随机密钥>)  keep=500 jobs
expose via tunnel: ngrok http 8001   OR   cloudflared tunnel --url http://localhost:8001
```
> 模型是**懒加载**：第一次生成时才加载（首次约 30 s），之后稳定。
> 若想后台隐藏运行并落日志：
> ```powershell
> $env:API_TOKEN='<你的强随机密钥>'; $env:LORA_ADAPTER="$ROOT/models/v4_640/adapter_best"
> Start-Process -FilePath '<PYTHON>' `
>   -ArgumentList '$ROOT\server.py' `
>   -RedirectStandardOutput '$ROOT\research\srv.out.log' `
>   -RedirectStandardError  '$ROOT\research\srv.err.log' -WindowStyle Hidden
> ```

**可选环境变量**（见主 README §9.2）：

| 变量 | 默认 | 用途 |
|---|---|---|
| `LOG_LEVEL` | `warning` | 设 `info` 可在后端窗口看到每条请求（排查"请求到底有没有到后端"最有用） |
| `PROGRESS_BAR` | `1` | 设 `0` 完全静音 tqdm 进度条 |
| `PORT` / `HOST` | `8001` / `127.0.0.1` | 监听地址 |
| `MAX_QUEUE` / `RATE_PER_MIN` / `JOB_TIMEOUT` / `MAX_JOBS` | `20` / `8` / `900` / `500` | 队列、限流、超时、保留任务数 |

**自检**（等 8~10 秒）：
```powershell
(Invoke-WebRequest -Uri 'http://127.0.0.1:8001/health' -UseBasicParsing).Content
# 期望：{"ok":true,"total_submitted":0,"queued":0,"running":0,"completed":0,"failed":0,"queue":0,"jobs":0}
```
> 首次可能返回 **502**（uvicorn 还没 bind 完），等几秒重试即可。

---

## 4. 第二步：把后端暴露为公网（隧道）

### 方式 A：ngrok（默认，免费版域名每次重启会变）

```powershell
# ⚠️ 必须先清代理：免费版 ngrok 不能走 HTTP 代理，否则报 ERR_NGROK_9009
Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY,Env:http_proxy,Env:https_proxy,Env:all_proxy -ErrorAction SilentlyContinue
ngrok http 8001
```

> 该命令**只影响当前终端进程**，不会修改你的全局/用户环境变量。
> 如果你本机开着代理软件（如 Clash 系列），系统级代理变量会让 ngrok 走代理而失败，所以必须在启动它的终端里清掉。

ngrok 窗口会显示：
```
Forwarding    https://your-tunnel.ngrok-free.dev -> http://localhost:8001
Web Interface http://127.0.0.1:4040
```
**这条 `https://…ngrok-free.dev` 就是要填进 Netlify 的 `BACKEND_URL`。**

也可以随时用 API 取当前地址：
```powershell
(Invoke-WebRequest -Uri 'http://127.0.0.1:4040/api/tunnels' -UseBasicParsing).Content
```

### 方式 B：Cloudflare Tunnel（推荐长期使用，可固定域名）
```powershell
cloudflared tunnel --url http://localhost:8001
# 或使用具名隧道 + 自有域名，可得到永久固定地址（免去每次改 BACKEND_URL）
```

---

## 5. 第三步：Netlify 站点与环境变量

站点信息：`landscape-art-demo`（站点 ID `$SITE`）。

在 **Site configuration → Environment variables** 设置（浏览器端看不到这两个值）：

| 变量 | 值 | 说明 |
|---|---|---|
| `BACKEND_URL` | `https://<你的隧道域名>` | 例如 `https://your-tunnel.ngrok-free.dev`；**不要带结尾斜杠** |
| `API_TOKEN` | `<你的强随机密钥>` | 必须与后端 `API_TOKEN` 完全一致 |

> 改完环境变量后**需要重新部署**（Functions 才会读取新值）：
> 见下一步的部署命令。

---

## 6. 第四步：部署 / 更新前端

```powershell
cd $ROOT
$env:NETLIFY_AUTH_TOKEN = '<你的 Netlify 令牌>'      # 只放在当前终端，不要写入文件
npx --yes netlify-cli@latest deploy --dir site --prod `
  --site $SITE --auth $env:NETLIFY_AUTH_TOKEN
```

要点与踩坑：
- **必须在 `$ROOT` 下执行**，否则 `--dir site` 会解析错目录（曾误把整个上级下载目录当成发布目录）。
- `--auth` 要**作为独立参数**传递；把令牌拼进一个字符串变量会导致 netlify-cli 打出帮助信息而不部署。
- `netlify.toml` 已配置 `publish = "site"`、`functions = "netlify/functions"` 与安全响应头。

---

## 7. 第五步：验收清单

| # | 检查 | 命令 / 期望 |
|---|---|---|
| 1 | 本机后端在线 | `http://127.0.0.1:8001/health` → `{"ok":true,...}` |
| 2 | 隧道在线 | ngrok 窗口有 `Forwarding`；`http://127.0.0.1:4040/api/tunnels` 可访问 |
| 3 | 代理能连到后端 | 浏览器打开 `https://<站点>/.netlify/functions/proxy?op=health` → 返回同一段 JSON |
| 4 | 页面可用 | 打开站点 → 状态卡显示「后端在线 排队 0 · 进行中 0 · 已完成 N · 累计 M」 |
| 5 | 能出图 | 点「生成」（默认标准 24 步）→ 约 6 s 出图；调试面板可见请求/响应 JSON |
| 6 | 大图可取 | 选「4K 4096」→ 完成后自动分块下载并正常显示/下载 |

命令行版端到端自检（推荐用 python，避开 pwsh 走系统代理时的 TLS 抖动）：
```powershell
& <PYTHON> $ROOT\research\boot_verify.py
```

---

## 8. 日常运维：隧道换域名怎么办

免费 ngrok 每次重启域名都可能变化。**域名一变，页面就会显示「后端离线」**。处理：

**方法 1（推荐）**：把新域名通过 CLI 更新到站点环境变量，然后重新部署。
```powershell
# 更新单个环境变量（示例：BACKEND_URL）
npx --yes netlify-cli@latest env:set BACKEND_URL "https://新的域名.ngrok-free.dev" `
  --site $SITE --auth $env:NETLIFY_AUTH_TOKEN
npx --yes netlify-cli@latest env:list --site $SITE --auth $env:NETLIFY_AUTH_TOKEN

# 重新部署使新变量生效
npx --yes netlify-cli@latest deploy --dir site --prod `
  --site $SITE --auth $env:NETLIFY_AUTH_TOKEN
```

**方法 2（一劳永逸）**：改用 **Cloudflare 命名隧道 + 自有域名**，或使用付费 ngrok 的保留域名，
这样 `BACKEND_URL` 一次配置永久有效。

---

## 9. 并发、队列与配额

单张 GPU 决定了**同一时刻只能跑一个任务**，后端据此做了保护：

| 机制 | 值 | 行为 |
|---|---|---|
| 处理方式 | 单 worker 串行 | 提交后立刻返回 `job_id`，前端轮询 `/jobs/{id}`，长任务不阻塞 HTTP |
| 队列上限 | `MAX_QUEUE=20` | 满则返回 **429 queue full** |
| 每 IP 限流 | `RATE_PER_MIN=8` | 超限返回 **429 rate limited** |
| 单任务超时 | `JOB_TIMEOUT=900` s | 超时标记 `failed: timeout`（4K 分块重绘较慢，故放宽） |
| 结果保留 | `MAX_JOBS=500` | 超出则删除最旧任务的记录**及其图片文件**；前端会显示「⌛ 已过期」 |
| 鉴权 | `X-API-Key` | 不匹配返回 **401** |

**估算**（512 标准 24 步 ≈ 6 s/张）：排队 3 张 ≈ 20 s；2048 ≈ 63 s；4K ≈ 30 s（快速路径）。
建议页面「一次出图」选 1 张，尤其在使用高清晰度档位时。

---

## 10. 安全建议

1. **务必设置强密钥**：`API_TOKEN` **没有可用默认值**——不设置时后端会用占位值 `change-me`
   并打印醒目警告。请用强随机串，例如：
   ```powershell
   $env:API_TOKEN = -join ((48..57)+(65..90)+(97..122) | Get-Random -Count 40 | % {[char]$_})
   ```
   并**同时更新 Netlify 站点环境变量**（两侧必须完全一致，否则 401）。
2. **不要把令牌写进仓库/文档**（`NETLIFY_AUTH_TOKEN`、`API_TOKEN` 都只放在终端环境变量里）。
3. **加访问口令**：可给页面加一层登录（本项目未内置，可后续实现）。
4. **注意 Netlify 令牌泄露**：曾在对话中粘贴过的个人访问令牌建议**立即在 Netlify 后台轮换（revoke）**；
   已部署的站点不受影响。
5. 后端只应监听 `127.0.0.1`（默认），由隧道对外；**不要**直接把 `HOST` 改成 `0.0.0.0` 暴露到局域网/公网。

### 10.1 代理来源守卫（已内置）

`netlify/functions/proxy.js` 对**会消耗 GPU 的操作**（`generate` / `warmup` / `gc` / `cancel`）做来源校验：
只有本站、`*.netlify.app`、`localhost` 或 `ALLOWED_ORIGINS` 中列出的来源可以触发；
跨站页面发起的 POST 一律 **403 `origin not allowed`**。只读操作（`health` / `job` / `image` / `chunk`）
不做来源限制，因为 `<img>` 请求不带 `Origin`。

| 环境变量 | 取值 | 效果 |
|---|---|---|
| `ALLOWED_ORIGINS` | 逗号分隔，如 `https://art.example.com` | 追加白名单（自定义域名必填） |
| `PROXY_STRICT` | `1` | 连**不带 Origin 的脚本调用**也拒绝（最严；本机脚本请直连 `127.0.0.1:8001`） |

### 10.2 轮换 `API_TOKEN`（含两个坑）

```powershell
# ① 生成强随机串（48 hex ≈ 192 bit）
python -c "import secrets; print(secrets.token_hex(24))"

# ② 写入 Netlify 站点环境变量
#    坑 1：netlify env:set 对已存在的 API_TOKEN 可能静默不生效（返回 0 但值不变）
#          → 用 API 写入并回读确认：
$aid = (Invoke-RestMethod https://api.netlify.com/api/v1/accounts -Headers @{Authorization="Bearer $env:NETLIFY_AUTH_TOKEN"})[0].id
$body = @(@{key='API_TOKEN'; values=@(@{context='all'; value='<新令牌>'})}) | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Put -Uri "https://api.netlify.com/api/v1/accounts/$aid/env/API_TOKEN?site_id=<SITE_ID>" `
  -Headers @{Authorization="Bearer $env:NETLIFY_AUTH_TOKEN"; 'Content-Type'='application/json'} -Body $body

# ③ 坑 2：环境变量改动必须**重新部署**才对线上 Function 生效
npx --yes netlify-cli@latest deploy --dir site --prod --site <SITE_ID> --auth $env:NETLIFY_AUTH_TOKEN
```

最后用后端启动命令带上同一串（`$env:API_TOKEN='<新令牌>'`），两侧一致才算完成。
验证：跨站 POST 应 403，本站来源 POST 应能到达后端（后端未启动时会看到 ngrok 502/404）。

---

## 11. 排错手册

| 症状 | 可能原因 | 排查与解决 |
|---|---|---|
| 页面显示「后端离线」 | 隧道没起 / `BACKEND_URL` 是旧域名 | 查 `http://127.0.0.1:4040/api/tunnels` 取实际地址，按 §8 更新 |
| 代理 health 返回 **502** + `hint` | Function 拿不到后端（域名错/隧道断/令牌错） | 先用浏览器直开 `.../proxy?op=health`；再看 Netlify 的 Function 日志 |
| `ERR_NGROK_9009` | ngrok 走了 HTTP 代理 | 按 §4 在**启动 ngrok 的终端**里清代理变量 |
| 取图 **502**，JSON 里 `ResponseSizeTooLarge` | 单次响应超过 Netlify 约 6 MB 上限 | 已实现分块；确认前端走 `op=chunk`（`loadImageUrl()` 会在 `bytes>4MB` 时自动分块） |
| 出图很慢（数分钟） | 步数过大（例如 500 步 ≈ 200 s） | 档位改「⚖️标准 24 步」；画廊条目会显示步数 |
| 页面打开是**源码文本** | 发布目录/类型不对 | 用 `netlify deploy --dir site --prod`（不要用 zip 直传） |
| `netlify-cli` 只打印帮助 | `--auth` 传参方式不对 | 让 `--auth $env:NETLIFY_AUTH_TOKEN` 作为独立参数 |
| 部署到了错误目录 | 没在项目根执行 | `cd $ROOT` 再部署 |
| `CUDA out of memory` | 与其它程序（浏览器/其它推理）争抢 6 GB | 后端已 `cpu_offload`；关掉占卡程序；不要同时跑两个模型进程 |
| 改了 `LORA_ADAPTER` 不生效 | 没重启后端 | **重启** `server.py`（模型懒加载，只在启动后首次生成时读取环境变量） |
| 历史画廊「⌛ 已过期」 | 后端重启，或结果超过 500 条被清理 | 正常；重新生成即可 |
| 本地 `Invoke-WebRequest` 报 SSL EOF | pwsh 走系统代理时的偶发 TLS 抖动 | 改用 `python research/*.py` 里的 requests 脚本自检 |

---

## 12. 文件清单

| 文件 | 作用 |
|---|---|
| `server.py` | 本机 GPU 后端：鉴权 / 队列 / 限流 / 轮询 / 落盘 / 分块回传 |
| `app.py` | 推理层：质量 / 极速双模式、prompt 组装、两段式与 ultimate 高清、img2img |
| `enhance.py` | ultimate upscale（神经超分 → 分块 img2img 重绘 → 锐化） |
| `site/index.html` | Netlify 前端（28 场景 / 档位 / 清晰度 / 画廊 / 调试面板 / 本地持久化） |
| `netlify/functions/proxy.js` | 同源代理：`op=health\|generate\|job\|image\|chunk` |
| `netlify.toml` | `publish=site`、`functions=netlify/functions`、安全响应头 |
| `server_cloud.py` | 可选：把同一套 API 换成云端 GPU（Replicate）执行 |
| `research/boot_verify.py` | 端到端自检脚本（health → 出图 → 取图） |
| `research/srv.*.log` | 后端运行日志（若用 Start-Process 启动） |

---

## 附：最短操作序列（重启一次全部）

```powershell
# 终端 A：后端
cd $ROOT
$env:API_TOKEN='<你的强随机密钥>'
$env:LORA_ADAPTER='$ROOT/models/v4_640/adapter_best'
<PYTHON> server.py

# 终端 B：隧道
Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY,Env:http_proxy,Env:https_proxy,Env:all_proxy -ErrorAction SilentlyContinue
ngrok http 8001

# 若隧道域名有变 → 更新 Netlify 环境变量并重新部署（见 §8）
# 若域名未变 → 直接打开 https://<your-site>.netlify.app 生成即可
```
