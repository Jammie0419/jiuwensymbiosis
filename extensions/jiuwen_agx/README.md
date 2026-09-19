# jiuwen_agx — jiuwensymbiosis 扩展包：AGX 仿真机器

通过 jiuwensymbiosis 框架控制 AGX Dynamics 仿真场景里的机器。当前包含：
**履带挖掘机**（回转 + 动臂 + 斗杆 + 铲斗 + 履带行走）+ 共享仿真底座
（以后装载机/卡车等仿真机器复用这一层）。

## 安装

```bash
# 前置：jiuwensymbiosis 本体已 editable 安装（见核心仓库 README）
uv pip install -e ./extensions/jiuwen_agx --no-deps   # --no-deps：core 不在 PyPI
```

安装后本体自动被发现：config 顶层 `adapter: agx_excavator` 即可，
run_task / introspect / GUI 均通过 entry-points 找到会话构建器。

## 使用（两步）

**① 启动仿真 + 桥接**（三选一）：

```bash
# 浏览器直播（Windows 浏览器打开打印的 URL 观看，零安装）
bash scripts/agx_viewer_stream.sh

# RDP/桌面会话里弹出 agxViewer 窗口
bash scripts/agx_viewer_live.sh

# 无头（自动化/CI）
source /opt/Algoryx/AGX-2.42.2.1/setup_env.bash
JIUWEN_SCRIPTS_DIR=$PWD/scripts xvfb-run -a agxViewer scripts/agx_viewer_bridge.agxPy
```

**② 发任务**（另开终端）：

```bash
python examples/run_task.py \
  --config extensions/jiuwen_agx/configs/agx_excavator/agx_excavator.local.yaml \
  --query "把土堆的土挖一斗倒到挖点右边3米处"
```

手动键盘模式（noVNC/RDP 画面聚焦后直接按键）：`JIUWEN_KEYBOARD=1` 加在启动命令前。
按键：`a/z` 铲斗、`s/x` 斗杆、`↑↓` 动臂、`←→` 回转、`PgUp/PgDn` 履带。

## 动作契约

| 动作 | 能力门 | 说明 |
| --- | --- | --- |
| `dig(dig_x_m, dig_y_m, dump_x_m, dump_y_m)` | motion.excavator | 一次完整挖-倒循环（基座系米） |
| `get_terrain()` | sensing.terrain | 料堆真值 {name, x_m, y_m, volume_m3} |
| `move_joint` / `navigate_relative` / `rotate_base` / `drive_arc` / `get_joint_positions` | motion.joint / motion.base | 通用动作（共享底座） |
| `home` | — | 回安全收拢姿态 |

单位注意：混合单位——swing 是弧度、三个液压缸是米（`joint_units: null` +
`swing_unit: "rad"`，探针输出里有标注）。

## 后端

| backend | 场景 |
| --- | --- |
| `mock` | 内存模拟，零依赖，离线开发/测试 |
| `remote`（推荐） | TCP 桥接 `scripts/agx_bridge_server.py`（可跑在 AGX 自带 Python，Windows/Linux 均可；`--demo` 无需 AGX） |
| `inprocess` | 本进程 import agx（需 AGX Python 3.10 环境；骨架已就位） |

## 添加下一台仿真机器

见 `docs/add-sim-machine.md` 四步配方（共享底座全部复用，每台约 150~250 行）。

## license 说明

AGX 是商业软件：无 license 时场景/链路/命令全部可用，但**执行器（电机/液压）
被全局禁用**——关节不动。license 文件放 `~/.config/agx/` 后重启 viewer 即解锁。
