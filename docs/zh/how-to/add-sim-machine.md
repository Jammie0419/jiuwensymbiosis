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

## AGX 真机验收（四步，每步有通过标准）

适用于"组里的 AGX 模型（Windows）能不能被控制"这个问题——按步走，每步都
有明确判据，卡在哪一步就修哪一步。

### 第 ① 步：模型可识别（Windows 跑探针）

在 AGX 那台 Windows 机器上，用 **AGX 自带的 Python** 运行：

```bat
python agx_scene_probe.py --direct 场景文件.agx
```

**通过标准**：报告里列出挖掘机的 4 个铰链（对准 swing/boom/arm/bucket 的
约束），每个都有实际角度范围和单位；末尾输出建议配置片段。
卡住的话看提示：`import agx` 失败 = 没用 AGX 的 Python；列表为空 = 模型
约束没有挂电机/命名不规范，把报告发给写模型的同事。

> `--direct` 的 AGX 遍历代码待场景接口确认后启用（脚本内有 TODO(AGX) 注释，
> 说明需要遍历哪些对象）。在那之前可用第 ②③ 步的 `inventory` 路径代替。

### 第 ② 步：链路通（Windows 起桥接服务）

```bat
python agx_bridge_server.py --scene 场景文件.agx --port 9700
:: 首次运行放行防火墙：
netsh advfirewall firewall add rule name="AGX Bridge" dir=in action=allow protocol=TCP localport=9700
```

**通过标准**：打印 `listening on 0.0.0.0:9700 (protocol v1, joints=[...])`，
无报错。演示链路可用 `--demo` 参数先验证（不需要加载场景）。

### 第 ③ 步：控制端可达（Linux 跑探针）

```bash
python scripts/agx_scene_probe.py --bridge --host <Windows机器IP> --port 9700
```

**通过标准**：`[OK] 链路通（协议 v1）` + `[OK] 场景清单已取回`，清单与第 ①
步一致，并打印建议配置片段（同时落盘 `agx_scene_report.json`）。

### 第 ④ 步：模型真的动（最终证明）

把第 ③ 步生成的配置片段填进 `configs/agx_excavator/agx_excavator.local.yaml`，
先核对配置自洽，再发一条真实任务：

```bash
# 核对：关节/限位/关键帧自洽性
python scripts/agx_scene_probe.py --check-config configs/agx_excavator/agx_excavator.local.yaml

# 真实任务（LLM key 已配置时）：
python examples/run_task.py \
  --config configs/agx_excavator/agx_excavator.local.yaml \
  --query "把左边土堆挖一斗倒到右边"
```

**通过标准**：Windows 窗口里挖掘机完成一次完整的挖-倒动作（对准→下铲→收斗
→摆转→卸料），Linux 侧命令以退出码 0 结束。走到这一步即"确定能控制"。

### 卡车（自卸车）怎么接

跟随同一份配方：`has_base: true` + `joint_names: ["dump_bed", ...]` + 新能力
`motion.dump` + `DUMP` 动作契约（货箱举升→倾倒循环，结构同 dig）。等挖掘机
走完上面四步、桥接服务稳定后，按 [配方](#配方以第二台机器装载机为例) 实现
薄包即可——共享底座（关节/底盘/协议/桥接）全部复用，预计 ~150 行。

### 多模型同场景

后续挖掘机 + 卡车放进同一个 `.agx` 场景时：桥接服务仍是**一个**（管整个场景），
每台机器在 config 里用 `joint_constraint_map` 声明自己占用的约束名（探针报告
里列出全部约束），互不冲突；两份 config 各自指向同一台桥接服务的地址。

