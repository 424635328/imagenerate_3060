# 大使用量下的韧性设计（SCALE）

> 目标：单个 RTX 3060 6GB 后端 + Netlify 静态前端的条件下，把「使用量增大」时的
> 瓶颈从"页面卡顿 / 写放大 / 代理配额"上移开。本文记录**已落地**的防御与**下一步**路线。

## 已落地的负载防御

| 层 | 机制 | 位置 |
|---|---|---|
| 轮询 | 自适应退避（先密后疏）；**后台标签页停轮询**（只保持 1.5 s 心跳位） | `api.js watchJob / pollDelay` |
| DOM | 画廊与队列改为 **rAF 合并渲染**：多个任务同帧轮询只触发一次重绘；标签页隐藏时直接跳过 | `main.js scheduleGallery / scheduleQueue` |
| 存储 | `store.save()` **300 ms 合并写**（打字、轮询进度不再逐次全量序列化 60 条历史）；`pagehide / beforeunload / visibilitychange(hidden)` 时 `flush()` 兜底，零丢失 | `store.js save/flush` + `main.js` |
| 长页渲染 | `.gallery .tile / .queue-row / .lib-item / .stat-card` 加 `content-visibility: auto`，屏外项跳过布局与绘制 | `css/shell.css` |
| 结构 | 两栏均改为**分区标签页**：任何时刻只渲染一组卡片，页面高度不再随功能数量增长；手机端 Dock 保持全量可见 | `js/shell.js` + `css/shell.css` |
| 静态资源 | Service Worker 壳缓存（stale-while-revalidate）；API 永不进缓存 | `sw.js` |
| 内存 | 结果 blob URL 有界 LRU（12 个），淘汰即 revoke | `store.js rememberBlob` |
| 后端已有 | 单 worker 队列 + 限流 + 鉴权；结果 LRU 缓存（硬链接）；任务结果配额回收；分块回传；预览渐进 | `server.py` |

## 代理防滥用（Netlify Function）

`netlify/functions/proxy.js` 是公网可达的网关：拿到 URL 的任何页面都能调 `?op=generate` 消耗本机 GPU。
现已按「浏览器跨站必须被拒」的原则加固：

| 场景 | 结果 |
|---|---|
| 本站 / 站点预览域名（`*.netlify.app`）/ `localhost` 发起的 POST | ✅ 放行 |
| 其它站点页面发起的 POST（带 Origin / Referer） | ⛔ 403 `origin not allowed` |
| 无 Origin 的脚本调用（curl / 监控） | ✅ 默认放行；设 `PROXY_STRICT=1` 后拒绝 |
| `image` / `chunk` / `preview` / `job` / `health` 只读操作 | 不做来源限制（`<img>` 请求不带 Origin） |

环境变量（Netlify 站点设置）：
- `ALLOWED_ORIGINS`：逗号分隔的额外白名单（自定义域名加这里）。
- `PROXY_STRICT=1`：连无 Origin 的客户端一并拒绝（最严）。

## 测试与扫描工具

```powershell
node tools/test_proxy_guard.mjs          # 代理来源守卫（无依赖，10 项）
python -m pyflakes server.py app.py config.py runtime.py enhance.py server_cloud.py tools
$env:NODE_PATH="$env:TEMP\lsart-jsdom\node_modules"; node tools/smoke_frontend.mjs
                                         # 真实 DOM 冒烟（21 项）：标签页/场景卡/词条/A-B/画廊/命令面板
```
`smoke_frontend.mjs` 需要 jsdom，刻意不进项目依赖：
`npm install --prefix "$env:TEMP\lsart-jsdom" jsdom`（未安装时自动 SKIP 并以 0 退出）。

## 下一步路线（按收益排序）

1. **画廊虚拟化**：历史上限 60 时全量 `innerHTML` 仍是 O(60)；用 IntersectionObserver 只挂载可视 tile（含缩略图分级：画廊用后端小图，点开再拉全图）。
2. **轮询合并器**：多个排队任务各自 `watchJob` → 改为单一 ticker 每拍批量 `/job` 查询，代理调用数从 N 降到 1（对 Netlify function 配额最关键）。
3. **推流替代轮询**：后端加 SSE（`/events`）或 WebSocket，前端只订阅增量；后端单进程实现成本低，收益大。
4. **缩略图分级**：后端生成 ~256 px thumb（`/thumb/{id}`），画廊/历史只加载 thumb，全图按需；可省 90% 画廊带宽。
5. **后端背压**：队列长度上限 + 明确 429 反馈（前端显示排队预计时长）；按客户端限流令牌桶改为按任务权重（4K 任务权重更高）。
6. **静态资源指纹化**：构建期给 css/js 加内容哈希 + `immutable` 头，配合 SW 版本号自动更新。
7. **多卡 / 多进程**（远期）：`server.py` 已单 worker；扩到 N worker 需把共享管线改为按 worker 隔离或加 GPU 互斥（本机仅 1 张 6 GB 卡，优先级低）。

## 验收指标（建议）

- 排队 8 个任务时页面主线程长任务 < 50 ms（DevTools Performance 面板）。
- 输入 prompt 连续打字 10 s：localStorage 写入次数 ≤ 10（可用 Performance monitor 观察）。
- 后台标签页挂 30 分钟：代理调用增量 ≈ 0；回前台立即补齐 UI。
- 画廊 60 条历史：滚动帧率稳定，屏外 tile 不触发 layout（Rendering → Layout Shift Regions 无抖动）。
