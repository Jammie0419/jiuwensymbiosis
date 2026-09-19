#!/usr/bin/env bash
# AGX 挖掘机实时演示 —— 在图形会话（RDP / 本机桌面）里运行本脚本：
#   bash scripts/agx_viewer_live.sh
# 弹出 agxViewer 窗口：挖掘机 + 沙地，同时桥接服务监听 :9700，
# jiuwensymbiosis 的控制命令会让画面里的挖掘机实时动起来。
#
# 前置：AGX 已安装（/opt/Algoryx），license 文件已放 ~/.config/agx/。
set -e
AGX_DIR=${AGX_DIR:-/opt/Algoryx/AGX-2.42.2.1}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$AGX_DIR/setup_env.bash"
export JIUWEN_SCRIPTS_DIR="$REPO_DIR/scripts"
export JIUWEN_BRIDGE_HOST="${JIUWEN_BRIDGE_HOST:-0.0.0.0}"
export JIUWEN_BRIDGE_PORT="${JIUWEN_BRIDGE_PORT:-9700}"

echo "[agx_viewer_live] agxViewer + bridge on :$JIUWEN_BRIDGE_PORT"
echo "[agx_viewer_live] 控制端验证: python scripts/agx_scene_probe.py --bridge --port $JIUWEN_BRIDGE_PORT"
exec agxViewer "$JIUWEN_SCRIPTS_DIR/agx_viewer_bridge.agxPy" 2>>/tmp/agx_viewer_noise.log
