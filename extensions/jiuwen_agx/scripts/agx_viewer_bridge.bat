@echo off
rem AGX viewer 桥接一键启动（Windows 本机 = 身体 + 屏幕，服务器 = 大脑）
rem
rem 用法（双击或在开好环境的命令提示符里运行）：
rem   1. 打开 "AGX Command Line"（开始菜单，AGX 自带环境）
rem   2. cd /d C:\path\to\jiuwensymbiosis\extensions\jiuwen_agx\scripts
rem   3. agx_viewer_bridge.bat
rem
rem 效果：agxViewer 窗口弹出（挖掘机 + 沙地，或加载你给的场景），
rem       同时桥接服务监听 0.0.0.0:9700 —— 服务器上的 jiuwensymbiosis
rem       通过局域网 IP 连过来即可 LLM 控制；窗口就在你面前，实时观看。
rem
rem 可选环境变量（改下面几行即可）：
rem   JIUWEN_BRIDGE_PORT   桥接端口（默认 9700）
rem   JIUWEN_KEYBOARD=1    开启键盘手动模式（窗口聚焦后 a/z 铲斗等）
rem   JIUWEN_BRIDGE_SCENE  加载自建场景：set JIUWEN_BRIDGE_SCENE=C:\...\my.agx
rem                        （需配合 --attachScript 方式，见 README）

setlocal
set "JIUWEN_SCRIPTS_DIR=%~dp0"
if not defined JIUWEN_BRIDGE_HOST set "JIUWEN_BRIDGE_HOST=0.0.0.0"
if not defined JIUWEN_BRIDGE_PORT set "JIUWEN_BRIDGE_PORT=9700"

rem 首次运行放行防火墙（管理员权限的 cmd 里执行一次）：
rem   netsh advfirewall firewall add rule name="AGX Bridge" dir=in action=allow protocol=TCP localport=9700

echo [jiuwen] 启动 agxViewer + 桥接 :%JIUWEN_BRIDGE_PORT% ...
agxViewer "%~dp0agx_viewer_bridge.agxPy"