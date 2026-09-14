# 本地私有笔记 · 模板（**可入库**；真实文件 `LOCAL_NOTES.md` 已被 .gitignore 忽略）

> 复制成 `LOCAL_NOTES.md` 后填写你机器上的真实取值。该真实文件**绝不提交**。
> 规则见 `docs/REPRODUCE.md` §10：令牌 / 站点 ID / 隧道域名 / 个人绝对路径只允许出现在
> 环境变量或这个未跟踪文件里。

## 本机真实路径

| 项目 | 值 |
|---|---|
| 项目根目录 | `<你的项目根目录>` |
| Python 解释器 | `<Python 路径>` |
| 结果目录 | `<项目根目录>\results` |
| 数据集源 | `<你的原始图片/zip 目录>` |

## 线上服务

| 项目 | 值 |
|---|---|
| Netlify 站点 | `https://<你的站点>.netlify.app` |
| Netlify 站点 ID | `<SITE_ID>` |
| 后端令牌 | `<强随机串>`（务必轮换，勿用示例值） |
| 隧道域名 | `https://<你的隧道>`（每次重启会变） |
| Netlify 环境变量 | `BACKEND_URL=<隧道域名>`、`API_TOKEN=<后端令牌>` |

## 常用命令（真实路径版）

```powershell
$PY   = "<Python 路径>"
$ROOT = "<你的项目根目录>"

# 后端
$env:API_TOKEN    = '<强随机串>'
$env:LORA_ADAPTER = "$ROOT\models\v4_640\adapter_best"
& $PY "$ROOT\server.py"

# 隧道（先清代理变量，否则部分隧道客户端会 ERR_NGROK_9009）
Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY,Env:http_proxy,Env:https_proxy,Env:all_proxy -ErrorAction SilentlyContinue
ngrok http 8001

# 部署前端
$env:NETLIFY_AUTH_TOKEN = '<你的 Netlify 令牌>'
$env:NETLIFY_SITE_ID    = '<SITE_ID>'
# 本机代理（FlClash 等）对 api.netlify.com 会 TLS 断连，而直连可达 → 部署前先清代理变量
$env:HTTP_PROXY = ''; $env:HTTPS_PROXY = ''; $env:ALL_PROXY = ''; $env:NO_PROXY = '*'
pwsh -NoProfile -File tools/dev.ps1 deploy
```

## 网络与远端（2026-09-14 实测）

| 场景 | 现象 | 处理 |
|---|---|---|
| `git push` 走 HTTPS | 代理下 `schannel: failed to receive handshake`；直连也会被 reset | **改用 SSH**：`git remote set-url origin git@github.com:<用户>/<仓库>.git`（本机 `~/.ssh/id_rsa_github` 已授权） |
| `netlify deploy` | Node 读取 `HTTPS_PROXY` → `Client network socket disconnected before secure TLS connection` | 部署前清空 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 并设 `NO_PROXY=*`（直连 `api.netlify.com` 返回 200） |
| HuggingFace 下载 | 代理下 `SSL: UNEXPECTED_EOF_WHILE_READING`；新版 Xet 协议更容易卡 | 设 `HF_HUB_DISABLE_XET=1` + 逐个文件 `hf_hub_download`；权重已下好后统一 `HF_HUB_OFFLINE=1` |
| 代理本身 | 某段时间 `127.0.0.1:7890` 对 GitHub/HF 均返回 000 | 先 `curl -x http://127.0.0.1:7890 https://github.com/` 探测，再决定走代理还是直连 |

> 注意：`git config --global https.proxy` 曾被写成 `https://127.0.0.1:7890`（协议应为 `http://`），
> 这会让 git 对代理本身发起 TLS；已修正为 `http://127.0.0.1:7890`。

## 待办（安全）

- [ ] 轮换 `API_TOKEN`：同步后端启动命令、Netlify 站点环境变量、隧道配置。
- [ ] 若令牌曾出现在对话/截图/提交历史里，到对应后台 **revoke 并重建**。
- [ ] 长期使用建议改用固定域名的隧道（Cloudflare Named Tunnel），免去每次改 `BACKEND_URL`。
