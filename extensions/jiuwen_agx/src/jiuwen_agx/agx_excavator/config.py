# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AgxExcavatorConfig — the excavator's slice of the shared sim config.

Only what makes an excavator AN EXCAVATOR lives here: the work-tool reach
envelope and the dig-cycle keyframe tuning. Everything else (backend, joints,
limits, undercarriage, terrain, camera) comes from
:class:`~jiuwen_agx.sim.config.SimMachineConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jiuwen_agx.sim.config import SimMachineConfig

__all__ = ["AgxExcavatorConfig"]


@dataclass
class AgxExcavatorConfig(SimMachineConfig):
    """履带挖掘机（回转 + 动臂 + 斗杆 + 铲斗）配置。"""

    # --- 覆盖通用默认：挖掘机出厂即 4 工作关节 + 履带底盘
    name: str = "agx_excavator"
    joint_names: tuple[str, ...] = ("swing", "boom", "arm", "bucket")
    has_base: bool = True
    base_step_limits: tuple[float, float] | None = (1.0, 0.7)  # 每命令 ≤1 m / ≤0.7 rad
    # 占位限位（deg）：接入 AGX 模型后按探针实测改；须覆盖 dig_cycle_tuning
    # 的全部关键帧（driver 会在执行前校验）。swing 视回转支承行程。
    joint_limits: dict[str, tuple[float, float]] | None = field(
        default_factory=lambda: {
            "swing": (-180.0, 180.0),
            "boom": (-45.0, 60.0),
            "arm": (-135.0, 60.0),
            "bucket": (-160.0, 40.0),
        }
    )
    # 安全收拢姿态（走行/转场）。
    home_joints: dict[str, float] | None = field(
        default_factory=lambda: {
            "swing": 0.0,
            "boom": -20.0,
            "arm": 25.0,
            "bucket": 10.0,
        }
    )

    # ==================== 挖掘工作参数 ====================
    # 铲齿可达半径包络（基座系，米）：dig 前置校验，超出即拒绝并提示先走底盘。
    reach_min_m: float = 1.0
    reach_max_m: float = 6.0
    # swing 关节的角度单位（"deg" 或 "rad"）。AGX 模型的回转是弧度、液压缸是
    # 米（混合单位，joint_units 留空），此时 swing_unit 必须填 "rad"。
    swing_unit: str = "deg"
    # 关键帧微调，键集见 work.DEFAULT_DIG_TUNING；YAML 里只写要改的键。
    # 注意：boom/arm/bucket 关键帧的值 = 该关节的原生单位（deg 机体是角度，
    # AGX 液压缸机体是米——local.yaml 里按探针实测填米值）。
    dig_cycle_tuning: dict[str, float] | None = None

    # ==================== 学习策略（policy.act 能力） ====================
    # 配置了才声明 policy.act 并暴露 act_exec；未配置 = 词表里没有这个动作。
    # name 对应 jiuwen_agx.policy.POLICIES 的注册名（"act"=lerobot ACT checkpoint，
    # "fake"=无依赖假策略，测试/演示用）；其余键由各策略类自取（act 要 ckpt/device）。
    # 注意：from_dict 会静默丢弃未知键——这个字段必须存在，YAML 的 policy 节才生效。
    policy: dict[str, Any] | None = None
    # act_exec 执行参数：单拍下发超时（数据采集与执行用同一常数，见接入计划 §4.5）。
    policy_beat_timeout_s: float = 1.0
    # act_exec 兜底拍数上限：超过即判失败（取"实测教师拍数 × 2"，先给占位值）。
    policy_max_beats: int = 300

    @classmethod
    def from_dict(cls, data):  # type: ignore[override]
        cfg = super().from_dict(data)
        if cfg.swing_unit not in ("deg", "rad"):
            raise ValueError(
                f"{cls.__name__}: swing_unit must be 'deg' or 'rad', got {cfg.swing_unit!r}"
            )
        if cfg.dig_cycle_tuning:
            cfg.dig_cycle_tuning = {
                str(k): float(v) for k, v in cfg.dig_cycle_tuning.items()
            }
        if cfg.policy is not None:
            if (
                not isinstance(cfg.policy, dict)
                or not str(cfg.policy.get("name", "")).strip()
            ):
                raise ValueError(
                    f"{cls.__name__}: policy must be a mapping with a non-empty "
                    f"'name' (a jiuwen_agx.policy.POLICIES registration), got {cfg.policy!r}"
                )
            cfg.policy = {str(k): v for k, v in cfg.policy.items()}
        cfg.policy_beat_timeout_s = float(cfg.policy_beat_timeout_s)
        cfg.policy_max_beats = int(cfg.policy_max_beats)
        return cfg
