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
npx --yes netlify-cli@latest deploy --dir site --prod --site <SITE_ID> --auth $env:NETLIFY_AUTH_TOKEN
```

## 待办（安全）

- [ ] 轮换 `API_TOKEN`：同步后端启动命令、Netlify 站点环境变量、隧道配置。
- [ ] 若令牌曾出现在对话/截图/提交历史里，到对应后台 **revoke 并重建**。
- [ ] 长期使用建议改用固定域名的隧道（Cloudflare Named Tunnel），免去每次改 `BACKEND_URL`。
