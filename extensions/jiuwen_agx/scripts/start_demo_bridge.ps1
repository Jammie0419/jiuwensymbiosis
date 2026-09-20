# AGX 桥接 —— 演示/联调模式启动器（⚠ 不是真 AGX）
#
# 本脚本起的是 agx_bridge_server.py --demo：一台纯内存的假挖掘机。
# 它只把命令记在字典里，进程内没有 AGX、没有物理、没有画面。
# 用途：在没有 AGX 的机器上验证 jiuwensymbiosis 侧的协议/规划链路。
#
# 想让真 AGX 动起来，用 agx_viewer_bridge.bat（在 AGX Command Line 里跑）。
#
# 只用标准库，任意 Python 3.11+ 都能跑；优先用仓库的 .venv。

param(
    [int]$Port = 9700,
    [string]$HostAddr = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

# 定位仓库 .venv（本文件在 extensions/jiuwen_agx/scripts/ 下）
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$venvPy = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPy) { $venvPy } else { "python" }

Write-Host "========================================" -ForegroundColor Yellow
Write-Host "  WARNING: DEMO 模式（内存 mock，无 AGX）" -ForegroundColor Yellow
Write-Host "  命令会显示成功，但不会有任何画面/物理动作。" -ForegroundColor Yellow
Write-Host "  要驱动真 AGX，请用 agx_viewer_bridge.bat" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Yellow
Write-Host "[jiuwen] demo bridge on ${HostAddr}:${Port}"

& $python (Join-Path $PSScriptRoot "agx_bridge_server.py") --demo --mode headless --host $HostAddr --port $Port

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
