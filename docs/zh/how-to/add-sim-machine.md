# 接入一台新的仿真机器（AGX 家族配方）

> 类别：How-to。适用于把 AGX（或任何提供"关节 + 地形真值"的）仿真模型接进
> jiuwensymbiosis。共享仿真底座已就位（`adapters/_common/sim/`），**新机器是
> 一个薄包，不是从零开始的适配器**。

## 前置：底座已经替你做了什么

| 层 | 位置 | 状态 |
| --- | --- | --- |
| 仿真器接缝 | `adapters/_common/sim/backend.py`（`SimBackend` 协议 + mock/inprocess/remote 三后端 + 注册表） | 现成 |
| 驱动 | `adapters/_common/sim/driver.py`（NamedJointDriver + BaseDriver 切片、关节名校验） | 现成 |
| Env / Api | `adapters/_common/sim/env.py`、`api.py`（能力按配置收窄、通用动作绑定） | 现成 |
| 通用动作 | `move_joint` / `get_joint_positions` / `navigate_relative` / `rotate_base` / `drive_arc` / `get_terrain` / `home` | 现成 |

## 配方（以第二台机器"装载机"为例）

### 1. 核心契约（约 30 行）

- `env/base.py`：`KNOWN_CAPABILITIES` 加 `"motion.loader"`（机器族工作能力，一族一个）。
- `contracts.py`：加结果 TypedDict（如 `ScoopResult | ScoopFailure`）。
- `api/actions.py`：加 `ActionSpec`（capability 指向上面的能力，`requires/provides`
  写清载荷状态）并注册进 `ACTIONS`。

### 2. 薄包 `jiuwensymbiosis/adapters/agx_loader/`（约 150 行）

- `config.py`：`AgxLoaderConfig(SimMachineConfig)`——只加机器参数（铲斗容积、
  工作包络、关键帧微调），并给 `joint_names` / `has_base` 等设出厂默认。
- `work.py`：工作循环纯函数（照抄 `agx_excavator/work.py` 的骨架：校验 →
  关键帧序列 → `mark_scoop` → 返回结果字典）。
- `api.py`：`AgxLoaderApi(SimMachineApi)`——只 `@implements` 本族的循环动作。
- `env.py`：`AgxLoaderEnv(SimMachineEnv)`——三行：类能力超集加
  `"motion.loader"`，`_capabilities_for_config` 并上它。
- `session.py` / `__init__.py`：`build_agx_loader_session = make_builder(...)`。

### 3. 注册与配置（3 处数据）

- `examples/run_task.py:_robot_session_builders()` 加一条。
- `jiuwensymbiosis/gui/data/bodies.yaml` 加一条。
- `configs/agx_loader/agx_loader.yaml`（从 `agx_excavator.yaml` 复制改）。

### 4. 测试与验证

- `tests/unit_tests/adapters/agx_loader/`：配置覆盖、工作循环几何（spy 驱动）、
  工具集合（照抄 `test_agx_excavator.py` 改）。
- `python scripts/validate_adapter.py --module jiuwensymbiosis.adapters.agx_loader`
- `python scripts/smoke_test_adapter.py --module jiuwensymbiosis.adapters.agx_loader`

## 后端怎么选

- **离线开发**：`backend: mock`（默认），零依赖，全链路可测。
- **接 AGX**：二选一——
  - `remote` + `scripts/agx_bridge_server.py`（跑在 AGX 自带 Python 里）：桥接
    机器无关，**一台服务所有仿真机器**；先 `--demo` 联通链路，再填
    `AgxSceneAdapter` 的 TODO(AGX) 方法。
  - `inprocess`：jiuwensymbiosis 进程内 `import agx`（需 AGX Python 兼容时）。
- 新后端形态（ros2/grpc）：`@register_backend("xxx")` 一个类即可，不改调用方。
