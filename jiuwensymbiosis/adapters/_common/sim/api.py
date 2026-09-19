# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SimMachineApi — the shared action bindings for every simulated machine.

Binds the GENERIC shared-vocabulary actions onto the sim driver surface once;
each action gates on its own capability, so a machine only advertises what its
config switches on (api ∩ env). Machine-family work cycles (excavator → dig)
are NOT here — a per-machine Api subclass binds those.
"""

from __future__ import annotations

from typing import Any

from jiuwensymbiosis.api import defaults
from jiuwensymbiosis.api.actions import (
    DRIVE_ARC,
    GET_JOINT_POSITIONS,
    GET_TERRAIN,
    MOVE_JOINT,
    NAVIGATE_RELATIVE,
    ROTATE_BASE,
    implements,
)
from jiuwensymbiosis.api.base import BaseRobotApi

__all__ = ["SimMachineApi"]


class SimMachineApi(BaseRobotApi):
    """Generic sim-machine surface: joints + undercarriage + terrain truth."""

    # ============================================================ Joint (motion.joint)
    @implements(MOVE_JOINT)
    def move_joint(self, targets: dict[str, float]) -> Any:
        return defaults.move_joint(self, targets)

    @implements(GET_JOINT_POSITIONS)
    def get_joint_positions(self) -> dict:
        """Read the latest joint positions keyed by joint name."""
        return self.env.get_joint_positions()

    # ============================================================ Undercarriage (motion.base)
    @implements(NAVIGATE_RELATIVE)
    def navigate_relative(self, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0) -> dict:
        return defaults.navigate_relative(self, dx_m, dy_m, dyaw_rad)

    @implements(ROTATE_BASE)
    def rotate_base(self, dyaw_rad: float) -> dict:
        return defaults.rotate_base(self, dyaw_rad)

    @implements(DRIVE_ARC)
    def drive_arc(self, radius_m: float, dyaw_rad: float) -> dict:
        return defaults.drive_arc(self, radius_m, dyaw_rad)

    # ============================================================ Terrain truth (sensing.terrain)
    @implements(GET_TERRAIN)
    def get_terrain(self) -> dict:
        """Material piles as {name, x_m, y_m, volume_m3} — the dig planner's map."""
        return {"ok": True, "piles": [dict(pile) for pile in self.env.read_terrain()]}

    # home() is inherited from BaseRobotApi (@implements(HOME) → defaults.home).
    # Stage B adds GET_IMAGE / PIXEL_TO_BASE_XYZ here once a sim camera reports
    # frames; until then no vision action is advertised.
