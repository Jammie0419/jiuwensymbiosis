# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""GoalActV2Policy — 小脑组 goal-act-v2 模型接到本框架 Policy 接缝的适配器。

对接情形 B（自研格式/模型，见 docs/cerebellum-interface-draft.md）：模型侧
的一切"方言"（35 维状态 + 10 帧历史、13 维目标特征、8 相位、16 步弧度增量、
度/弧度混用）都在这个文件里被消化；``act_exec`` 契约、执行循环、护栏、
判据一律不动。

三层分离（通用性的落点）：

- 通用链路层：本文件除注册名外不写死任何机器参数；
- 每机器层：``policy_cfg["machine"]`` 指向一份标定 json（schema 见
  ``configs/machine_agx365.template.json``，加载与换算在 ``jiuwen_agx.machine_map``）
  ——单位映射（Prismatic 米↔等效角）、sign/offset 校零、goal 的 z 高度都在
  那里，换机型只换文件；
- 每模型层：``policy_cfg["ckpt"]``（goal-act-v2 checkpoint，config 与归一化
  内嵌）+ ``policy_cfg["src"]``（含 ``act_trajectory`` 包的目录，通常为
  ``<project>/src``；加入 sys.path 后 ``from act_trajectory.inference import ACTPolicy``）。

执行规则等价性：他们的闭环是"每控制周期 观测 → 预测 16 步 → 只执行
chunk[0]（q_target = 实测 + 增量），逐步重规划"。本适配器**每次 predict 只
返回一个目标**，于是 ``act_exec`` 的拍循环（队列空 → 重新观测 → 再预测）
自动等价于他们的闭环——模型不需要改动，链路也不需要为它加逻辑。

相位供给：goal-act-v2 的相位既是目标切换开关（≤3 取挖点、≥4 取卸点）也是
嵌入输入。本适配器用本环境独有的载料真值（``obs["scoop"]``，语义见
``driver.scoop_state``）估计：未载料 → ``phase_dig``（默认 2，对准挖点），
载料 → ``phase_dump``（默认 4，摆向卸点）；``reset()`` 清零。这两个值可经
policy 配置覆盖。若小脑组交付 phase-free 变体，这里退化为常量即可。
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Any

from jiuwen_agx.machine_map import MachineMap
from jiuwen_agx.policy import register_policy

__all__ = ["GoalActV2Policy"]

logger = logging.getLogger(__name__)


@register_policy("goal_act_v2")
class GoalActV2Policy:
    """小脑组 goal-act-v2 checkpoint behind the Policy seam (torch loaded lazily)."""

    def __init__(
        self,
        policy_cfg: dict[str, Any],
        *,
        joint_names: tuple[str, ...] = (),
        joint_limits: dict[str, tuple[float, float]] | None = None,
        home_joints: dict[str, float] | None = None,
    ) -> None:
        del (
            joint_limits,
            home_joints,
        )  # the checkpoint self-describes; body truth unused
        ckpt = policy_cfg.get("ckpt")
        if not ckpt:
            raise ValueError(
                "policy 'goal_act_v2' needs 'ckpt' in the YAML policy section "
                "(a goal-act-v2 .pt from the cerebellum team)"
            )
        src = policy_cfg.get("src")
        if not src:
            raise ValueError(
                "policy 'goal_act_v2' needs 'src' in the YAML policy section "
                "(the directory containing the 'act_trajectory' package, e.g. <project>/src)"
            )
        machine_path = policy_cfg.get("machine")
        if not machine_path:
            raise ValueError(
                "policy 'goal_act_v2' needs 'machine' in the YAML policy section "
                "(a per-machine calibration JSON, see configs/machine_agx365.template.json)"
            )
        machine = MachineMap.load(machine_path)
        if not machine.calibrated:
            logger.warning(
                "[goal_act_v2] machine file %s is NOT calibrated — predictions are only "
                "meaningful for chain/dry-run checks, not for evaluation",
                machine_path,
            )
        if tuple(joint_names) != machine.joint_order:
            raise ValueError(
                f"policy 'goal_act_v2': machine file joint order {machine.joint_order} does "
                f"not match this body's joint_names {tuple(joint_names)} — the checkpoint "
                "is bound to a machine layout"
            )
        self._names = machine.joint_order
        self._maps = machine.joints
        self._pick_z = machine.pick_z_m
        self._dump_z = machine.dump_z_m
        self._phase_dig = int(policy_cfg.get("phase_dig", 2))
        self._phase_dump = int(policy_cfg.get("phase_dump", 4))

        src_path = str(src)
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        try:
            from act_trajectory.inference import ACTPolicy as _AcTPolicy
        except ImportError as exc:  # readable: torch/numpy or the package missing
            raise ValueError(
                f"policy 'goal_act_v2': cannot import act_trajectory from {src_path!r} "
                f"({exc}) — check the 'src' path and that torch/numpy/h5py are installed"
            ) from exc
        self._policy = _AcTPolicy(str(ckpt))
        config = self._policy.config
        self._state_dim = int(config["state_dim"])
        self._joint_slots = [int(i) for i in config["joint_state_indices"]]
        if len(self._joint_slots) != len(self._names):
            raise ValueError(
                f"policy 'goal_act_v2': checkpoint joint slots {self._joint_slots} do not "
                f"cover this body's {len(self._names)} joints"
            )
        self._loaded_seen = False

    def reset(self) -> None:
        """Clear both the model-side episode state and our phase estimator."""
        self._loaded_seen = False
        self._policy.reset()

    # ---- one beat of prediction ----------------------------------------------
    def predict(self, obs: dict[str, Any]) -> list[dict[str, float]]:
        """ONE absolute joint target: measured pose + the chunk's first increment.

        Returning a single target (not the 16-step block) is what makes our
        beat loop an exact replica of the model's own per-cycle replan loop.
        """
        joints = obs.get("joints") or {}
        goal = obs.get("goal") or {}
        if obs.get("scoop") is True:
            self._loaded_seen = True
        phase = self._phase_dump if self._loaded_seen else self._phase_dig

        q_policy = [self._maps[name].to_policy(joints[name]) for name in self._names]
        state35 = [0.0] * self._state_dim
        for slot, angle_rad in zip(self._joint_slots, q_policy, strict=True):
            state35[slot] = math.degrees(angle_rad)  # their state channels are degrees
        pick_xyz = (float(goal["dig_x_m"]), float(goal["dig_y_m"]), self._pick_z)
        dump_xyz = (float(goal["dump_x_m"]), float(goal["dump_y_m"]), self._dump_z)

        chunk = self._policy.predict(state35, pick_xyz, dump_xyz, phase)
        increment = chunk[
            0
        ]  # (4,) radians, already clipped to max_joint_step by their policy
        target: dict[str, float] = {}
        for index, name in enumerate(self._names):
            value = self._maps[name].to_native(
                float(q_policy[index]) + float(increment[index])
            )
            if not math.isfinite(value):  # paranoia: never emit NaN/inf
                value = float(joints[name])
            target[name] = value
        return [target]
