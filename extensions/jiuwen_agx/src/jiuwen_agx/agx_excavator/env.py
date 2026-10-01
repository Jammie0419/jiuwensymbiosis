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

    capabilities = SimMachineEnv.capabilities | {"motion.excavator", "policy.act"}
    name = "agx_excavator"

    def _capabilities_for_config(self) -> frozenset[str]:
        caps = set(super()._capabilities_for_config()) | {"motion.excavator"}
        # policy.act only when a policy is configured — an unconfigured body
        # never sees act_exec in its vocabulary (the tool gate is api ∩ env).
        if self.cfg.policy:
            caps.add("policy.act")
        return frozenset(caps)
