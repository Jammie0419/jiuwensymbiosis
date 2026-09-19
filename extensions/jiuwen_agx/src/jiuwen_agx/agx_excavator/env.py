# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AgxExcavatorEnv — the shared sim env + the excavator's work capability.

Three lines of substance: add ``motion.excavator`` to the class-level superset
(so the static validator sees DIG's capability tag) and to every instance
(so the api∩env gate actually emits the dig tool). Everything else is inherited.
"""

from __future__ import annotations

from jiuwen_agx.sim.env import SimMachineEnv

__all__ = ["AgxExcavatorEnv"]


class AgxExcavatorEnv(SimMachineEnv):
    """Simulated tracked excavator (swing / boom / arm / bucket + undercarriage)."""

    capabilities = SimMachineEnv.capabilities | {"motion.excavator"}
    name = "agx_excavator"

    def _capabilities_for_config(self) -> frozenset[str]:
        return super()._capabilities_for_config() | {"motion.excavator"}
