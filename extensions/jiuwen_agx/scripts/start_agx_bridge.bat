@echo off
rem ===========================================================================
rem AGX excavator + bridge, one click. Opens its own AGX window.
rem
rem Why %AGX_DIR%\python-x64\python.exe and NOT agxViewer:
rem   agxViewer re-creates its process with the *profile* environment, so it
rem   loses the PATH/PYTHONPATH we set and ends up loading a foreign Python
rem   (on this machine: Anaconda 3.12.7). AGX's python modules are built for
rem   3.12.10 and refuse to load -> "No module named 'agxPythonModules'".
rem   Running the plugin with AGX's own python makes init_app() build its own
rem   ExampleApplication: same window, and our environment survives.
rem
rem Usage: double-click, or from any cmd:
rem   cd /d <repo>\extensions\jiuwen_agx\scripts
rem   start_agx_bridge.bat
rem
rem Env (optional, set before running):
rem   AGX_DIR               AGX install root (default on the line below)
rem   JIUWEN_BRIDGE_PORT    default 9700
rem   JIUWEN_BRIDGE_HOST    default 127.0.0.1 (0.0.0.0 to accept a remote brain)
rem   JIUWEN_KEYBOARD=1     enable keyboard manual control in the window
rem ============================================================================
setlocal
if not defined AGX_DIR set "AGX_DIR=E:\AGX-2.42.2.1"
if not defined JIUWEN_BRIDGE_PORT set "JIUWEN_BRIDGE_PORT=9700"
if not defined JIUWEN_BRIDGE_HOST set "JIUWEN_BRIDGE_HOST=127.0.0.1"
if not defined JIUWEN_BRIDGE_EXCAVATOR set "JIUWEN_BRIDGE_EXCAVATOR=1"

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

echo [jiuwen] AGX excavator + bridge on %JIUWEN_BRIDGE_HOST%:%JIUWEN_BRIDGE_PORT%
echo [jiuwen] an AGX window will open; Ctrl-C here (or close the window) to stop.
echo [jiuwen] no AGX on this machine? use start_demo_bridge.ps1 for a protocol-only mock.

"%AGX_PY%" "%~dp0agx_viewer_bridge.agxPy"

echo [jiuwen] bridge exited with %ERRORLEVEL%
pause
