# jiuwen_agx scripts

脚本按“谁运行、依赖什么”分组。**根目录的 AGX 运行时入口不要随意移动**：

- `agx_viewer_bridge*.agxPy` 用 `from agx_bridge_server import ...`；
- `agx_bridge_server.py --machine 215f` 用 `import agx_215f_scene`；
- `start_agx_bridge*.bat` 用 `%~dp0` 找同目录的脚本。

## 根目录：AGX 运行时入口

| 文件 | 运行环境 | 作用 |
| --- | --- | --- |
| `agx_bridge_server.py` | AGX 自带 Python | 桥接服务；`--demo` / `--scene` / `--machine 215f` |
| `agx_viewer_bridge.agxPy` | AGX 自带 Python | 365 窗口 + 桥接 |
| `agx_viewer_bridge_215f.agxPy` | AGX 自带 Python | 215F 窗口 + 桥接 |
| `agx_215f_scene.py` | AGX 自带 Python | 215F 场景工厂（被桥接/viewer import） |
| `start_agx_bridge.bat` | Windows | 365 一键启动 |
| `start_agx_bridge_215f.bat` | Windows | 215F 一键启动 |
| `start_demo_bridge.ps1` | Windows | 无 AGX 的协议 mock |

## control/：控制端与排障（仓库 venv）

| 文件 | 作用 |
| --- | --- |
| `agx_scene_probe.py` | `--direct` 场景盘点 / `--bridge` 远程清单 / `--check-config` 配置闸门 |
| `agx_215f_dig_smoke.py` | 215F 一斗验收（同一后端、同一个 `dig`） |
| `monitor_ui.py` | NiceGUI 关节面板：读数、手动设目标、标定姿态 |

## act/：数据采集、转换、训练、验收

| 文件 | 作用 |
| --- | --- |
| `record_demos.py` | AGX 侧采集演示 → npz |
| `npz_to_lerobot.py` | npz → LeRobotDataset（训练机跑） |
| `train_act.py` | lerobot-train 的 Windows 包装（跳过 checkpoint 符号链接） |
| `make_smoke_checkpoint.py` | 随机权重 checkpoint，验证加载/推理链路 |
| `act_acceptance.py` | 留出点回放 + `dig` 同点对照 |
| `lerobot_to_goalact_h5.py` | LeRobot → goal-act H5 |

## calibration/：每机器标定

| 文件 | 作用 |
| --- | --- |
| `calibrate_agx365.py` | 365 OpenPLX 纯几何扫描 + 数据回放 → `machine_agx365.json` |
| `agx_machine_fit.py` | 桥接扫缸采样 → 拟合每机器 json（license 恢复后用） |

## 扩展约定

- **新机型**：场景工厂放根目录 `agx_<machine>_scene.py`，在 `agx_bridge_server.py` 的
  `builders` 里注册一个 `--machine <name>`；控制端配置放
  `configs/agx_excavator/agx_excavator.<name>.yaml`；本目录只放该机型的专用工厂/启动器。
- **新控制/排障工具**：放 `control/`；默认不要依赖同目录的运行时脚本，确实需要时
  （如 `agx_scene_probe.py --direct` 复用桥接的 SWIG 下转型）把 `scripts/` 加入 `sys.path`。
- **数据/训练脚本**：放 `act/`；采集侧不得依赖 torch/lerobot，转换/训练侧不得依赖 AGX。
- 所有命令在**仓库根目录**执行，路径写 `extensions/jiuwen_agx/scripts/...`。
