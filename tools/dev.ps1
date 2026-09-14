<#
.SYNOPSIS
  Landscape·Art 开发总入口 —— 一条命令替代记不住的 N 条命令。

.EXAMPLE
  .\tools\dev.ps1 check        # 全量检查：python 编译 + JS 语法 + 前端一致性 + 路径/脱敏 + 行尾
  .\tools\dev.ps1 eol          # 行尾归一化后门禁
  .\tools\dev.ps1 bench        # 批量/缓存端到端基准（需本机后端在跑）
  .\tools\dev.ps1 security     # 上传安全回归测试
  .\tools\dev.ps1 start        # 启动后端（+ 提示隧道命令）
  .\tools\dev.ps1 tunnel       # 启动 ngrok 隧道（进程内清代理）
  .\tools\dev.ps1 deploy       # 部署前端到 Netlify（需 NETLIFY_AUTH_TOKEN）
#>
[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet('check', 'eol', 'bench', 'security', 'verify', 'start', 'tunnel', 'deploy', 'judge', 'help')]
  [string]$Task = 'help',

  [string]$Python = $env:PYTHON,
  [string]$SiteId = $env:NETLIFY_SITE_ID
)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot          # tools/ 的上一级 = 项目根
Set-Location $Root

if (-not $Python) {
  $candidates = @(
    "$env:USERPROFILE\anaconda3\envs\ldm\python.exe",
    "$env:USERPROFILE\miniconda3\envs\ldm\python.exe"
  )
  $Python = ($candidates | Where-Object { Test-Path $_ } | Select-Object -First 1)
}
$env:LANDSCAPE_ROOT = $Root
# 让中文日志在 PowerShell / harness 里都能正确显示（否则按控制台代码页输出会乱码）
$env:PYTHONIOENCODING = 'utf-8'

function Say($m) { Write-Host "» $m" -ForegroundColor Cyan }
function Ok($m) { Write-Host "  [OK] $m" -ForegroundColor Green }
function Bad($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red }
function Need-Python { if (-not $Python -or -not (Test-Path $Python)) { Bad "找不到 python（设置 `$env:PYTHON 或安装 conda 环境 ldm）"; exit 1 } }

switch ($Task) {

  'help' {
    Write-Host 'Landscape·Art dev entry' -ForegroundColor White
    Write-Host '  .\tools\dev.ps1 check     全量检查（编译/JS/前端/路径/脱敏/行尾）'
    Write-Host '  .\tools\dev.ps1 eol       行尾归一 + 门禁'
    Write-Host '  .\tools\dev.ps1 security  上传安全回归'
    Write-Host '  .\tools\dev.ps1 judge     人工评判：起收集器 + 本地站点，结果落到 research/human_judge/'
    Write-Host '  .\tools\dev.ps1 bench     批量/缓存基准（需后端在跑）'
    Write-Host '  .\tools\dev.ps1 verify    线上端到端（需隧道在线）'
    Write-Host '  .\tools\dev.ps1 start     启动后端'
    Write-Host '  .\tools\dev.ps1 tunnel    启动隧道（清代理）'
    Write-Host '  .\tools\dev.ps1 deploy    部署前端到 Netlify'
  }

  'check' {
    Need-Python
    $fail = 0

    Say 'python 语法编译'
    & $Python -m py_compile app.py server.py server_cloud.py config.py runtime.py enhance.py
    if ($LASTEXITCODE -eq 0) { Ok 'app/server/server_cloud/config/runtime/enhance' } else { Bad 'py_compile'; $fail++ }

    Say 'JS 语法（逐个模块）'
    $bad = 0
    Get-ChildItem -Path (Join-Path $Root 'site\js') -Filter *.js -ErrorAction SilentlyContinue |
      Select-Object -Unique FullName | ForEach-Object {
        node --check $_.FullName 2>$null
        if ($LASTEXITCODE -ne 0) { Bad $_.Name; $bad++ }
      }
    node --check netlify\functions\proxy.js 2>$null
    if ($LASTEXITCODE -ne 0) { Bad 'proxy.js'; $bad++ }
    if ($bad -eq 0) { Ok 'all JS modules' } else { $fail++ }

    Say '前端 id / 模块一致性'
    & $Python tools\check_frontend.py
    if ($LASTEXITCODE -eq 0) { Ok 'frontend' } else { $fail++ }

    Say 'CSS 语法（css-tree 严格解析）'
    node tools\check_css_syntax.mjs
    if ($LASTEXITCODE -eq 0) { Ok 'css syntax' } else { $fail++ }

    Say '动效弹簧曲线（与解析解比对）'
    & $Python tools\test_spring_easing.py
    if ($LASTEXITCODE -eq 0) { Ok 'spring easing' } else { $fail++ }

    Say '版本评级台语料（格子完整 / 数字可复算 / 判据无效标记）'
    & $Python tools\test_version_gallery.py
    if ($LASTEXITCODE -eq 0) { Ok 'version gallery' } else { $fail++ }

    Say '版本评判台前端（jsdom，缺依赖自动 SKIP）'
    node tools\smoke_versions.mjs
    if ($LASTEXITCODE -eq 0) { Ok 'versions page' } else { $fail++ }

    Say '人工评判收集器（回环 / 跨站防护 / 独立重算 / 非 UTF-8 控制台回归）'
    & $Python tools\test_judge_collector.py
    if ($LASTEXITCODE -eq 0) { Ok 'judge collector' } else { $fail++ }

    Say '提交链路真浏览器 E2E（点一下 → 文件落到 research/human_judge/）'
    node tools\test_judge_submit_e2e.mjs
    if ($LASTEXITCODE -eq 0) { Ok 'submit e2e' } else { $fail++ }

    Say '人工评判结果（可复算 / 不自夸 / 可重生成）'
    & $Python tools\test_human_verdict.py
    if ($LASTEXITCODE -eq 0) { Ok 'human verdict' } else { $fail++ }

    Say '模型版本路由（白名单 / 贯穿 / 缓存隔离 / 前端接线）'
    & $Python tools\test_adapter_routing.py
    if ($LASTEXITCODE -eq 0) { Ok 'adapter routing' } else { $fail++ }

    Say '路径与敏感串'
    & $Python tools\check_paths.py
    if ($LASTEXITCODE -eq 0) { Ok 'paths' } else { $fail++ }

    Say '脱敏总检'
    & $Python research\final_check.py
    if ($LASTEXITCODE -eq 0) { Ok 'sanitization' } else { $fail++ }

    Say '行尾门禁'
    & $Python tools\check_eol.py
    if ($LASTEXITCODE -eq 0) { Ok 'eol' } else { $fail++ }

    if ($fail -eq 0) { Write-Host "`nALL CHECKS PASSED" -ForegroundColor Green; exit 0 }
    Write-Host "`n$fail 项检查失败" -ForegroundColor Red; exit 1
  }

  'eol' {
    Need-Python
    Say 'normalize_eol'
    & $Python tools\normalize_eol.py
    Say 'check_eol'
    & $Python tools\check_eol.py
    exit $LASTEXITCODE
  }

  'security' {
    Need-Python
    Say '上传安全回归（magic bytes / EXIF+GPS / 像素炸弹）'
    & $Python tools\test_upload_security.py
    exit $LASTEXITCODE
  }

  'judge' {
    Need-Python
    $port = if ($env:JUDGE_PORT) { $env:JUDGE_PORT } else { 8787 }
    Say "启动评判服务（页面 + 提交接口同源）：http://127.0.0.1:$port/versions.html"
    Write-Host ''
    Write-Host "  1) 浏览器打开  http://127.0.0.1:$port/versions.html" -ForegroundColor White
    Write-Host '  2) 默认就是盲测：标签是 A/B/C/D，顺序每题为随机但可复现' -ForegroundColor White
    Write-Host '  3) 判完点「📤 提交评判」→ 结果直接落到 research/human_judge/' -ForegroundColor White
    Write-Host ''
    Write-Host "  打不开先试 http://127.0.0.1:$port/health —— 应返回一段 JSON。" -ForegroundColor DarkGray
    Write-Host '  如果连它都打不开，说明浏览器把 127.0.0.1 也走了代理（见 LOCAL_NOTES 的代理说明），' -ForegroundColor DarkGray
    Write-Host '  把 127.0.0.1/localhost 加进代理绕过列表，或换一个没配代理的浏览器。' -ForegroundColor DarkGray
    Write-Host ''
    Write-Host '  Ctrl+C 停止' -ForegroundColor DarkGray
    & $Python tools\judge_collector.py --port $port
    exit $LASTEXITCODE
  }

  'bench' {
    Need-Python
    Say '批量 + 缓存基准（需要本机后端 127.0.0.1:8001 在跑）'
    & $Python tools\verify_v5.py
    exit $LASTEXITCODE
  }

  'verify' {
    Need-Python
    Say '线上端到端（经 Netlify 代理）'
    & $Python tools\verify_live_v5.py
    exit $LASTEXITCODE
  }

  'start' {
    Need-Python
    Say "启动后端（root=$Root）"
    if (-not $env:API_TOKEN) { Write-Host '  提示：未设置 API_TOKEN，后端会用占位值 change-me' -ForegroundColor Yellow }
    if (-not $env:LORA_ADAPTER) { Write-Host '  提示：未设置 LORA_ADAPTER，将使用 config.py 默认权重' -ForegroundColor Yellow }
    & $Python server.py
  }

  'tunnel' {
    Say '启动隧道（仅在本进程清代理）'
    & "$Root\ngrok.bat"
  }

  'deploy' {
    if (-not $env:NETLIFY_AUTH_TOKEN) { Bad '缺少环境变量 NETLIFY_AUTH_TOKEN'; exit 1 }
    if (-not $SiteId) { Bad '缺少 NETLIFY_SITE_ID（或设置 $env:NETLIFY_SITE_ID）'; exit 1 }
    Say '部署 site/ 到 Netlify（生产）'
    npx --yes netlify-cli@latest deploy --dir site --prod --site $SiteId --auth $env:NETLIFY_AUTH_TOKEN
    exit $LASTEXITCODE
  }
}
