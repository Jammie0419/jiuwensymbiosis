# 编写扩展功能包（Extension Package）

> 类别：How-to。把不属于核心框架的本体/能力（实验室专属机型、仿真机器、
> 特殊工作循环）做成独立的扩展包，核心仓库保持与上游同步。

## 核心提供的四条扩展缝

| 缝 | 位置 | 用途 |
| --- | --- | --- |
| `register_capability(name)` | `jiuwensymbiosis/env/base.py` | 运行时把新能力加入 `KNOWN_CAPABILITIES`（原地 set，幂等） |
| `register_actions(*specs)` | `jiuwensymbiosis/api/actions.py` | 把本地声明的 ActionSpec 推入共享词表 `ACTIONS`（复用词表三重校验） |
| `register_capability_spec(name, actions=, driver_members=)` | `adapters/_common/capability_spec.py` | 注册能力→动作/驱动成员映射（validator A-10/D-14 检查用） |
| entry-points 组 `jiuwensymbiosis.adapters` | 扩展包自己的 pyproject | 让 run_task / introspect / GUI 发现你的 `build_<name>_session` |

技能不需要缝：`agent/fast/registry.register_skill_dir(path)` 是公开 API，
扩展包 import 时自行调用。

三个注册函数都是**原地可变**设计（set.add / dict.update / dict 赋值），所有
`from-import` 绑定自动看到新增——这是整个机制的基石。

## 最小骨架

```
my-ext/
├── pyproject.toml
├── src/my_ext/
│   ├── __init__.py          # 注册链（顺序见下）
│   ├── actions.py           # 本地声明 ActionSpec
│   ├── contracts.py         # 结果 TypedDict（零依赖）
│   └── my_robot/            # 薄适配器包
│       ├── __init__.py      # 导出 build_my_robot_session
│       ├── config.py work.py api.py env.py session.py
│       └── ...
├── skills/<skill>/SKILL.md
├── configs/my_robot/default.yaml
└── tests/
```

`pyproject.toml` 关键段：

```toml
[project]
name = "my-ext"
dependencies = ["jiuwensymbiosis"]

[project.entry-points."jiuwensymbiosis.adapters"]
my_robot = "my_ext.my_robot:build_my_robot_session"

[tool.setuptools.packages.find]
where = ["src"]
include = ["my_ext*"]
```

`__init__.py` 注册链（**顺序敏感**——能力必须先于 ActionSpec 构造）：

```python
from jiuwensymbiosis.env.base import register_capability
from jiuwensymbiosis.adapters._common.capability_spec import register_capability_spec

register_capability("motion.my_family")
register_capability_spec("motion.my_family", actions=["my_cycle"])

from jiuwensymbiosis.api.actions import register_actions
from my_ext.actions import MY_CYCLE
register_actions(MY_CYCLE)

from my_ext import my_robot          # @implements(MY_CYCLE) 在此绑定
from my_ext.skills import register   # 或 register_skill_dir(...)
```

安装（core 已 editable 安装的前提下）：

```bash
uv pip install -e ./my-ext --no-deps   # --no-deps：core 不在 PyPI，避免重新解析
```

之后 `run_task --config`、`jiuwensymbiosis-actions --config`、GUI 会自动发现
你的本体（config 顶层 `adapter: my_robot` 即可）。

## 约定与注意

- **可上游的部分**：上面四条缝本身是通用机制，欢迎提 PR 给上游；你的机型/工作
  循环留在扩展包里。
- **词表可见性**：本地声明的动作在 `jiuwensymbiosis-actions --config`（读 api
  实例）完全可见；全局 `--vocabulary` 视图只在扩展包被 import 后才包含它们。
- **导入顺序**：扩展包 `__init__` 里 register_capability 必须先于任何
  `ActionSpec(capability=...)` 构造（构造期校验成员资格）。
- **实例**：仓库内 `extensions/jiuwen_agx/` 是完整可参照的实现（AGX 仿真挖掘机）。
