#!/usr/bin/env bash
# AGX 挖掘机浏览器直播 —— 在 Windows 浏览器里看服务器的 agxViewer 画面。
#
# 用法（服务器终端，无需图形会话）：
#   bash scripts/agx_viewer_stream.sh
# 然后在 Windows 浏览器打开脚本打印的地址（形如 http://<服务器IP>:6080/vnc.html），
# 点"连接"即可看到挖掘机画面，鼠标键盘还能直接操作 agxViewer。
#
# 组件：Xvfb 虚拟显示(:99) → agxViewer(+桥接:9700) → x11vnc → websockify/noVNC(:6080)
# 仅限局域网使用（VNC 无密码）。
set -e
AGX_DIR=${AGX_DIR:-/opt/Algoryx/AGX-2.42.2.1}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISPLAY_NUM=${DISPLAY_NUM:-99}
VNC_PORT=5900
WEB_PORT=${WEB_PORT:-6080}
BRIDGE_PORT=${JIUWEN_BRIDGE_PORT:-9700}

cleanup() {
  echo "[stream] shutting down..."
  kill $AGX_PID $VNC_PID $WEB_PID $XVFB_PID 2>/dev/null || true
}
trap cleanup EXIT

# 1. 虚拟显示
Xvfb :$DISPLAY_NUM -screen 0 1280x800x24 &
XVFB_PID=$!
sleep 1
export DISPLAY=:$DISPLAY_NUM

# 2. VNC 服务（把 :99 的画面暴露成 VNC）
x11vnc -display :$DISPLAY_NUM -forever -shared -nopw -quiet -rfbport $VNC_PORT &
VNC_PID=$!
sleep 1

# 3. noVNC 网页代理（浏览器 ←websocket→ VNC）
websockify --web /usr/share/novnc $WEB_PORT localhost:$VNC_PORT &
WEB_PID=$!
sleep 1

# 4. AGX viewer + 桥接
# shellcheck disable=SC1091
source "$AGX_DIR/setup_env.bash"
export JIUWEN_SCRIPTS_DIR="$REPO_DIR/scripts"
export JIUWEN_BRIDGE_HOST="0.0.0.0"
export JIUWEN_BRIDGE_PORT="$BRIDGE_PORT"
agxViewer "$JIUWEN_SCRIPTS_DIR/agx_viewer_bridge.agxPy" 2>>/tmp/agx_viewer_noise.log &
AGX_PID=$!

SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "[stream] ===================== 就绪 ====================="
echo "[stream] Windows 浏览器打开:  http://$SERVER_IP:$WEB_PORT/vnc.html"
echo "[stream] 连接后即可看到挖掘机画面（可鼠标操作视角）"
echo "[stream] 控制端验证:  python scripts/agx_scene_probe.py --bridge --port $BRIDGE_PORT"
echo "[stream] ==============================================="
echo ""

wait $AGX_PID
