@echo off
rem ===========================================================================
rem 215F (Sunward SWE215F) AGX window + bridge, one click.
rem
rem Same workflow as start_agx_bridge.bat (365): AGX window + bridge on 9700.
rem Difference: the scene is built by agx_215f_scene:build (Excavator215FEnv
rem from the delivery package) instead of AGX's bundled 365 model.
rem
rem Usage: double-click, or
rem   cd /d <repo>\extensions\jiuwen_agx\scripts
rem   start_agx_bridge_215f.bat
rem
rem Env (optional): AGX_DIR / JIUWEN_BRIDGE_PORT / JIUWEN_BRIDGE_HOST /
rem   JIUWEN_215F_* (pile, force limits, copy paths ... see agx_215f_scene.py)
rem
rem Headless alternative (no window):
rem   %AGX_DIR%\python-x64\python.exe agx_bridge_server.py --machine 215f --port 9700
rem ===========================================================================
setlocal
if not defined AGX_DIR set "AGX_DIR=E:\AGX-2.42.2.1"
if not defined JIUWEN_BRIDGE_PORT set "JIUWEN_BRIDGE_PORT=9700"
if not defined JIUWEN_BRIDGE_HOST set "JIUWEN_BRIDGE_HOST=127.0.0.1"

set "AGX_PY=%AGX_DIR%\python-x64\python.exe"
if not exist "%AGX_PY%" (
    echo [jiuwen] AGX python not found: %AGX_PY%
    echo [jiuwen] set AGX_DIR to your AGX install root and retry.
    pause
    exit /b 1
)

set "PATH=%AGX_DIR%\python-x64;%AGX_DIR%\python-x64\Scripts;%AGX_DIR%\bin\x64;%PATH%"
set "PYTHONHOME=%AGX_DIR%\python-x64"
set "PYTHONPATH=%AGX_DIR%\bin\x64\agxpy;%AGX_DIR%\data\python\modules"
set "JIUWEN_SCRIPTS_DIR=%~dp0"

echo [jiuwen] 215F excavator + bridge on %JIUWEN_BRIDGE_HOST%:%JIUWEN_BRIDGE_PORT%
echo [jiuwen] an AGX window will open; Ctrl-C here (or close the window) to stop.
echo [jiuwen] no AGX on this machine? use start_demo_bridge.ps1 for a protocol-only mock.

"%AGX_PY%" "%~dp0agx_viewer_bridge_215f.agxPy"

echo [jiuwen] bridge exited with %ERRORLEVEL%
pause
