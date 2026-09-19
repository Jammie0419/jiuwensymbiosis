# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AgxExcavatorConfig — the excavator's slice of the shared sim config.

Only what makes an excavator AN EXCAVATOR lives here: the work-tool reach
envelope and the dig-cycle keyframe tuning. Everything else (backend, joints,
limits, undercarriage, terrain, camera) comes from
:class:`~jiuwensymbiosis.adapters._common.sim.config.SimMachineConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jiuwensymbiosis.adapters._common.sim.config import SimMachineConfig

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
        default_factory=lambda: {"swing": 0.0, "boom": -20.0, "arm": 25.0, "bucket": 10.0}
    )

    # ==================== 挖掘工作参数 ====================
    # 铲齿可达半径包络（基座系，米）：dig 前置校验，超出即拒绝并提示先走底盘。
    reach_min_m: float = 1.0
    reach_max_m: float = 6.0
    # 关键帧微调（deg / m³），键集见 work.DEFAULT_DIG_TUNING；YAML 里只写要改的键。
    dig_cycle_tuning: dict[str, float] | None = None

    @classmethod
    def from_dict(cls, data):  # type: ignore[override]
        cfg = super().from_dict(data)
        if cfg.dig_cycle_tuning:
            cfg.dig_cycle_tuning = {str(k): float(v) for k, v in cfg.dig_cycle_tuning.items()}
        return cfg
