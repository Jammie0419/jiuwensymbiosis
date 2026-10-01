# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""jiuwen_agx — jiuwensymbiosis 扩展包：AGX 仿真机器（挖掘机等）。

Import-time registration chain (ORDER MATTERS):

1. register_capability × 3   —— 能力先入表（ActionSpec 构造期校验成员资格）
2. contracts                 —— 零依赖结果类型
3. actions + register_actions —— DIG/GET_TERRAIN/ACT_EXECUTE 入共享词表
4. policy + policy_act       —— Policy 接缝注册表（FakePolicy 即装即用；
                                ActPolicy 只注册名字，torch/lerobot 惰性导入）
5. sim                       —— 底座 Env/Api 类定义（能力校验再触发）+ 动作绑定
6. agx_excavator             —— 挖掘机薄包（绑定 DIG/ACT_EXECUTE）
7. register_skill_dir        —— excavate / excavate_act 技能入目录

安装（core 仓库的 venv 里）：
    uv pip install -e extensions/jiuwen_agx --no-deps   # --no-deps：core 已 editable 安装
    uv pip install -e "extensions/jiuwen_agx[policy]"   # 可选：学习策略栈（lerobot/torch）
"""

from __future__ import annotations

from pathlib import Path

from jiuwensymbiosis.adapters._common.capability_spec import register_capability_spec

# -- 1. 能力注册（必须在任何 ActionSpec / Env 子类构造之前）
# 注册链顺序敏感：能力 → spec → 底座 → 薄包 → 技能（详见本文件 docstring）
from jiuwensymbiosis.env.base import register_capability

register_capability("motion.excavator")
register_capability("sensing.terrain")
register_capability("policy.act")
register_capability_spec("motion.excavator", actions=["dig"])
register_capability_spec(
    "sensing.terrain", actions=["get_terrain"], driver_members=["read_terrain"]
)
register_capability_spec("policy.act", actions=["act_exec"])

# -- 2-3. 契约 + 动作声明 + 词表注册
from jiuwen_agx.contracts import (  # noqa: F401, I001
    ActFailure,
    ActResult,
    DigFailure,
    DigResult,
    PileInfo,
    TerrainScan,
)
from jiuwen_agx.actions import ACT_EXECUTE, DIG, GET_TERRAIN
from jiuwensymbiosis.api.actions import register_actions

register_actions(DIG, GET_TERRAIN, ACT_EXECUTE)

# -- 4. Policy 接缝（注册表即装即用；torch/lerobot 只在构造 ActPolicy 时导入）
# -- 5. 仿真底座（Env/Api 类定义在此触发；动作绑定 GET_TERRAIN 来自上面已注册的词表）
from jiuwen_agx import (
    policy,  # noqa: F401
    policy_act,  # noqa: F401
    sim,  # noqa: F401
)

# -- 6. 挖掘机薄包
from jiuwen_agx.agx_excavator import (
    AgxExcavatorApi,
    AgxExcavatorConfig,
    AgxExcavatorEnv,
    build_agx_excavator_session,
)

# -- 7. 技能目录（excavate / excavate_act）
try:
    from jiuwensymbiosis.agent.fast.registry import register_skill_dir

    register_skill_dir(Path(__file__).parent / "skills")
except Exception:  # pragma: no cover - 目录已被注册（模块重复导入）等非致命场景
    pass

__all__ = [
    "ACT_EXECUTE",
    "DIG",
    "GET_TERRAIN",
    "AgxExcavatorApi",
    "AgxExcavatorConfig",
    "AgxExcavatorEnv",
    "build_agx_excavator_session",
]
