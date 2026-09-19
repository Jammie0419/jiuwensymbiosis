# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AgxExcavatorApi — the excavator's only addition to the shared sim surface.

Everything generic (move_joint / navigate / get_terrain / home) is inherited
from ``SimMachineApi``; this class binds the machine-family work cycle ``dig``
and nothing else. Domain-level refusals (reach envelope, loaded bucket) come
back as a ``DigFailure`` dict so the LLM can read the reason and re-plan;
hardware-level problems still raise.
"""

from __future__ import annotations

from jiuwensymbiosis.adapters._common.sim.api import SimMachineApi
from jiuwensymbiosis.adapters._common.sim.env import SimMachineEnv
from jiuwensymbiosis.adapters.agx_excavator.config import AgxExcavatorConfig
from jiuwensymbiosis.adapters.agx_excavator.work import execute_dig_cycle
from jiuwensymbiosis.api.actions import DIG, implements
from jiuwensymbiosis.contracts import DigFailure, DigResult

__all__ = ["AgxExcavatorApi"]


class AgxExcavatorApi(SimMachineApi):
    """SimMachine surface + the dig work cycle."""

    @implements(DIG)
    def dig(self, dig_x_m: float, dig_y_m: float, dump_x_m: float, dump_y_m: float) -> DigResult | DigFailure:
        """One dig-and-dump cycle (see the DIG contract). Config carries the
        reach envelope and keyframe tuning; geometry lives in work.execute_dig_cycle."""
        env = self.env
        if not isinstance(env, SimMachineEnv) or not isinstance(getattr(env, "cfg", None), AgxExcavatorConfig):
            raise RuntimeError("AgxExcavatorApi requires an env built from AgxExcavatorConfig")
        cfg: AgxExcavatorConfig = env.cfg  # type: ignore[assignment]
        try:
            result = execute_dig_cycle(
                env.driver,
                dig_x_m=float(dig_x_m),
                dig_y_m=float(dig_y_m),
                dump_x_m=float(dump_x_m),
                dump_y_m=float(dump_y_m),
                tuning=cfg.dig_cycle_tuning,
                reach_min_m=cfg.reach_min_m,
                reach_max_m=cfg.reach_max_m,
                swing_unit=cfg.swing_unit,
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "volume_m3": result["volume_m3"], "cycle_s": result["cycle_s"]}
