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

import logging
import math
import time
from typing import Any

from jiuwensymbiosis.api.actions import implements

from jiuwen_agx.actions import ACT_EXECUTE, DIG
from jiuwen_agx.agx_excavator.config import AgxExcavatorConfig
from jiuwen_agx.agx_excavator.work import check_cycle_preconditions, execute_dig_cycle
from jiuwen_agx.agx_excavator.work_ik import SnapshotUnavailable, execute_pose_dig_cycle
from jiuwen_agx.contracts import ActFailure, ActResult, DigFailure, DigResult
from jiuwen_agx.sim.api import SimMachineApi
from jiuwen_agx.sim.env import SimMachineEnv

__all__ = ["AgxExcavatorApi"]

logger = logging.getLogger(__name__)


class AgxExcavatorApi(SimMachineApi):
    """SimMachine surface + the dig work cycle + the policy seam consumer."""

    def __init__(self, env: SimMachineEnv) -> None:
        super().__init__(env)
        # The learned policy is constructed LAZILY on the first act_exec call:
        # session build stays cheap and torch-free until a policy is actually
        # used (and an unconfigured policy never constructs one at all).
        self._policy_obj: Any = None
        self._ik: Any = None
        self._ik_machine: str | None = None

    @property
    def capabilities(self) -> frozenset[str]:
        """Advertise ``policy.act`` only when a policy is configured.

        ``act_exec`` is always implemented, so the MRO derivation would claim
        the capability unconditionally and trip the session capability-mismatch
        alarm on every unconfigured build — even though the tool gate (api ∩
        env) already hides the action. Withholding the claim on the api side
        too keeps the vocabulary AND the alarm consistent with the config.
        """
        caps = super().capabilities
        env_cfg = getattr(self.env, "cfg", None)
        if not getattr(env_cfg, "policy", None):
            caps = caps - {"policy.act"}
        return caps

    # ------------------------------------------------------------------ scripted cycle
    @implements(DIG)
    def dig(
        self,
        dig_x_m: float,
        dig_y_m: float,
        dump_x_m: float,
        dump_y_m: float,
        dig_z_m: float | None = None,
        dump_z_m: float | None = None,
        dig_pitch_deg: float | None = None,
        dump_pitch_deg: float | None = None,
    ) -> DigResult | DigFailure:
        """One dig-and-dump cycle (see the DIG contract). Config carries the
        reach envelope and keyframe tuning; geometry lives in work.execute_dig_cycle.

        z/pitch are optional 3D-pose fields. The keyframe path ignores them (365
        compatibility); the IK path uses them to place the bucket tip.
        """
        env = self.env
        if not isinstance(env, SimMachineEnv) or not isinstance(
            getattr(env, "cfg", None), AgxExcavatorConfig
        ):
            raise RuntimeError(
                "AgxExcavatorApi requires an env built from AgxExcavatorConfig"
            )
        cfg: AgxExcavatorConfig = env.cfg  # type: ignore[assignment]
        if cfg.dig_mode == "ik":
            try:
                dig_z = float(dig_z_m) if dig_z_m is not None else None
                dump_z = float(dump_z_m) if dump_z_m is not None else None
                if dig_z is None or dump_z is None:
                    snap = env.driver.snapshot(
                        points=[
                            [float(dig_x_m), float(dig_y_m)],
                            [float(dump_x_m), float(dump_y_m)],
                        ]
                    )
                    surfaces = (snap or {}).get("surface_m") or []
                    if dig_z is None and len(surfaces) > 0 and surfaces[0] is not None:
                        dig_z = float(surfaces[0])
                    if dump_z is None and len(surfaces) > 1 and surfaces[1] is not None:
                        # A dump point at ground level cannot be reached with the
                        # bucket open; keep the brain's XY but release clear of
                        # the ground. An explicit dump_z_m is used exactly.
                        dump_z = float(surfaces[1]) + float(cfg.dump_clearance_m)
                if dig_z is None:
                    raise SnapshotUnavailable(
                        "dig_z_m missing and no surface height at the dig point"
                    )
                if dump_z is None:
                    raise SnapshotUnavailable(
                        "dump_z_m missing and no surface height at the dump point"
                    )
                result = execute_pose_dig_cycle(
                    env.driver,
                    ik=self._get_ik(cfg.dig_ik_machine or ""),
                    dig_pose=(float(dig_x_m), float(dig_y_m), dig_z),
                    dump_pose=(float(dump_x_m), float(dump_y_m), dump_z),
                    dig_pitch=(
                        math.radians(float(dig_pitch_deg))
                        if dig_pitch_deg is not None
                        else None
                    ),
                    dump_pitch=(
                        math.radians(float(dump_pitch_deg))
                        if dump_pitch_deg is not None
                        else None
                    ),
                    tuning=cfg.dig_cycle_tuning,
                    reach_min_m=cfg.reach_min_m,
                    reach_max_m=cfg.reach_max_m,
                    reach_z_min_m=cfg.reach_z_min_m,
                    reach_z_max_m=cfg.reach_z_max_m,
                    approach_gap_m=cfg.approach_gap_m,
                    bite_m=cfg.bite_m,
                    cut_length_m=cfg.cut_length_m,
                    carry_lift_m=cfg.carry_lift_m,
                    carry=cfg.dig_carry,
                    dig_bucket_open_native=cfg.dig_bucket_open_rad,
                    dig_bucket_curl_native=cfg.dig_bucket_curl_rad,
                    dump_bucket_open_native=cfg.dump_bucket_open_rad,
                    ik_tolerance_m=cfg.ik_tolerance_m,
                    dump_tolerance_m=cfg.dump_tolerance_m,
                    contact_tolerance_m=cfg.contact_tolerance_m,
                    ik_max_iters=cfg.ik_max_iters,
                    ik_stall_iters=cfg.ik_stall_iters,
                    dump_dwell_s=cfg.dig_dump_dwell_s,
                )
            except SnapshotUnavailable as exc:
                logger.warning("[dig] %s — 回落到 keyframe 循环", exc)
            else:
                return {"ok": True, **result}
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
                travel=cfg.dig_travel,
                cut_path=cfg.dig_cut_path,
                carry=cfg.dig_carry,
                dump_dwell_s=cfg.dig_dump_dwell_s,
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "volume_m3": result["volume_m3"],
            "cycle_s": result["cycle_s"],
        }

    def _get_ik(self, machine_path: str) -> Any:
        """Load/caches the per-machine IK model; native limits from this body."""
        if self._ik is None or self._ik_machine != machine_path:
            from jiuwen_agx.excavator_ik import ExcavatorIk, machine_file_path

            try:
                ik = ExcavatorIk.from_machine_file(machine_file_path(machine_path))
                cfg = self.env.cfg
                limits = getattr(cfg, "joint_limits", None)
                if limits:
                    ik.set_execution_limits_native(limits)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                raise ValueError(
                    f"cannot load IK machine file {machine_path!r}: {exc}"
                ) from exc
            self._ik = ik
            self._ik_machine = machine_path
        return self._ik

    # ------------------------------------------------------------------ learned cycle
    @implements(ACT_EXECUTE)
    def act_exec(
        self,
        dig_x_m: float,
        dig_y_m: float,
        dump_x_m: float,
        dump_y_m: float,
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
            # ``scoop`` is the body's measured carry truth: the goal-act-v2
            # adapter consumes it for phase estimation; other policies ignore it.
            return {
                "joints": driver.get_joint_positions(),
                "goal": dict(goal),
                "scoop": bool(driver.scoop_state()),
            }

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
