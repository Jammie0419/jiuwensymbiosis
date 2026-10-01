# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AgxExcavatorApi — the excavator's additions to the shared sim surface.

Everything generic (move_joint / navigate / get_terrain / home) is inherited
from ``SimMachineApi``; this class binds the machine-family work cycles — the
scripted ``dig`` and the learned-policy ``act_exec``. Domain-level refusals
(reach envelope, loaded bucket, illegal policy output) come back as a failure
dict so the LLM can read the reason and re-plan; hardware-level problems still
raise.
"""

from __future__ import annotations

import time
from typing import Any

from jiuwensymbiosis.api.actions import implements

from jiuwen_agx.actions import ACT_EXECUTE, DIG
from jiuwen_agx.agx_excavator.config import AgxExcavatorConfig
from jiuwen_agx.agx_excavator.work import check_cycle_preconditions, execute_dig_cycle
from jiuwen_agx.contracts import ActFailure, ActResult, DigFailure, DigResult
from jiuwen_agx.sim.api import SimMachineApi
from jiuwen_agx.sim.env import SimMachineEnv

__all__ = ["AgxExcavatorApi"]


class AgxExcavatorApi(SimMachineApi):
    """SimMachine surface + the dig work cycle + the policy seam consumer."""

    def __init__(self, env: SimMachineEnv) -> None:
        super().__init__(env)
        # The learned policy is constructed LAZILY on the first act_exec call:
        # session build stays cheap and torch-free until a policy is actually
        # used (and an unconfigured policy never constructs one at all).
        self._policy_obj: Any = None

    # ------------------------------------------------------------------ scripted cycle
    @implements(DIG)
    def dig(
        self, dig_x_m: float, dig_y_m: float, dump_x_m: float, dump_y_m: float
    ) -> DigResult | DigFailure:
        """One dig-and-dump cycle (see the DIG contract). Config carries the
        reach envelope and keyframe tuning; geometry lives in work.execute_dig_cycle."""
        env = self.env
        if not isinstance(env, SimMachineEnv) or not isinstance(
            getattr(env, "cfg", None), AgxExcavatorConfig
        ):
            raise RuntimeError(
                "AgxExcavatorApi requires an env built from AgxExcavatorConfig"
            )
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
        return {
            "ok": True,
            "volume_m3": result["volume_m3"],
            "cycle_s": result["cycle_s"],
        }

    # ------------------------------------------------------------------ learned cycle
    @implements(ACT_EXECUTE)
    def act_exec(
        self, dig_x_m: float, dig_y_m: float, dump_x_m: float, dump_y_m: float
    ) -> ActResult | ActFailure:
        """One policy-driven dig-and-dump cycle (see the ACT_EXECUTE contract).

        The policy emits chunks of absolute joint targets; each beat goes
        through ``move_joints_blocking`` so driver validation is the last gate
        on network output. Success = the scoop truth goes loaded→unloaded (the
        same measured mass threshold dig relies on); ``policy_max_beats`` is
        the failure backstop. Requires a configured policy (env gating already
        hides this action when none is set).
        """
        env = self.env
        if not isinstance(env, SimMachineEnv) or not isinstance(
            getattr(env, "cfg", None), AgxExcavatorConfig
        ):
            raise RuntimeError(
                "AgxExcavatorApi requires an env built from AgxExcavatorConfig"
            )
        cfg: AgxExcavatorConfig = env.cfg  # type: ignore[assignment]
        driver = env.driver
        try:
            check_cycle_preconditions(
                driver,
                dig_x_m=float(dig_x_m),
                dig_y_m=float(dig_y_m),
                dump_x_m=float(dump_x_m),
                dump_y_m=float(dump_y_m),
                reach_min_m=cfg.reach_min_m,
                reach_max_m=cfg.reach_max_m,
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        goal = {
            "dig_x_m": float(dig_x_m),
            "dig_y_m": float(dig_y_m),
            "dump_x_m": float(dump_x_m),
            "dump_y_m": float(dump_y_m),
        }

        def observe() -> dict[str, Any]:
            return {"joints": driver.get_joint_positions(), "goal": dict(goal)}

        try:
            policy = self._get_policy(cfg)
        except Exception as exc:  # noqa: BLE001 - surfaced as a readable failure
            return {"ok": False, "error": f"policy unavailable: {exc}"}

        started = time.perf_counter()
        policy.reset()
        try:
            queue = list(policy.predict(observe()))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"policy predict failed: {exc}"}

        beats = 0
        was_loaded = False
        while beats < cfg.policy_max_beats:
            if not queue:  # chunk exhausted → re-observe and re-query
                try:
                    queue = list(policy.predict(observe()))
                except Exception as exc:  # noqa: BLE001
                    return {"ok": False, "error": f"policy predict failed: {exc}"}
            target = queue.pop(0)
            try:
                driver.move_joints_blocking(target, timeout_s=cfg.policy_beat_timeout_s)
            except ValueError as exc:
                # The driver's own validation is the per-beat guard on network
                # output; keep the reason readable instead of raising past the
                # runner's structured-failure path.
                return {
                    "ok": False,
                    "error": f"policy output rejected at beat {beats}: {exc}",
                }
            beats += 1
            if driver.scoop_state():
                was_loaded = True
            elif was_loaded:
                # loaded → unloaded: scooped, then dumped — one full cycle.
                return {
                    "ok": True,
                    "beats": beats,
                    "cycle_s": time.perf_counter() - started,
                }
        return {
            "ok": False,
            "error": (
                f"cycle incomplete after {beats} beats "
                f"(bucket {'was loaded but never dumped' if was_loaded else 'never loaded'})"
            ),
        }

    def _get_policy(self, cfg: AgxExcavatorConfig) -> Any:
        """Lazily construct the configured policy (cached on the api instance)."""
        if self._policy_obj is not None:
            return self._policy_obj
        from jiuwen_agx.policy import POLICIES

        policy_cfg = dict(cfg.policy or {})
        name = str(policy_cfg.get("name", ""))
        cls = POLICIES.get(name)
        if cls is None:
            raise ValueError(
                f"unknown policy {name!r} — registered policies: {sorted(POLICIES)}; "
                "for 'act' install the learning stack first "
                "(pip install -e extensions/jiuwen_agx[policy])"
            )
        self._policy_obj = cls(
            policy_cfg,
            joint_names=tuple(cfg.joint_names),
            joint_limits=cfg.joint_limits,
            home_joints=cfg.home_joints,
        )
        return self._policy_obj
