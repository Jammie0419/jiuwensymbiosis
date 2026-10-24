# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Excavator dig-cycle geometry — pure functions over the shared sim driver.

A dig is ONE compound work cycle expressed as parameterised joint keyframes:

  对准挖点 → 举臂就位 → 下铲 → 收斗(装满) → 摆转到倒点 → 卸料

All numbers are computed here (swing = atan2 of the ground point; every other
keyframe comes from the tuning table) — the LLM only chooses WHERE to dig and
where to dump. The functions are deliberately driver-only (no env, no api), so
they unit-test against a spy and any future digging-family machine can reuse
them by naming the same four joints.
"""

from __future__ import annotations

import math
import time
from typing import Any

__all__ = [
    "DEFAULT_DIG_TUNING",
    "REQUIRED_JOINTS",
    "check_cycle_preconditions",
    "check_dig_preconditions",
    "check_dump_preconditions",
    "execute_dig_cycle",
    "execute_dig_phase",
    "execute_dump_phase",
]

# The four joints a dig cycle needs, by conventional name (config joint_names).
REQUIRED_JOINTS: tuple[str, ...] = ("swing", "boom", "arm", "bucket")

# Keyframe tuning (degrees; metres for volumes). Angles are rig-convention
# placeholders — tune against the real model via YAML dig_cycle_tuning, which
# overlays (not replaces) these defaults.
DEFAULT_DIG_TUNING: dict[str, float] = {
    # 回转零位校准：真实模型 swing=0 的朝向不一定是基座 +X；探针实测后填。
    # 摆转目标 = atan2(ground point) + swing_offset_deg，归一化到 [-180, 180)。
    "swing_offset_deg": 0.0,
    # 举臂就位（行走/转场姿态上方）
    "ready_boom_deg": 10.0,
    "ready_arm_deg": -25.0,
    "ready_bucket_deg": 20.0,
    # 下铲（铲齿入地）
    "dig_boom_deg": -35.0,
    "dig_arm_deg": 55.0,
    "dig_bucket_deg": -70.0,
    # 收斗提升（满斗）
    "curl_boom_deg": 25.0,
    "curl_arm_deg": -55.0,
    "curl_bucket_deg": 40.0,
    # 卸料
    "dump_boom_deg": 15.0,
    "dump_arm_deg": -35.0,
    "dump_bucket_deg": -120.0,
    # 名义斗容：mock/无测量后端报告的方量；AGX 后端以仿真实测为准
    "bucket_volume_m3": 0.6,
}


def _normalise_swing(angle_deg: float) -> float:
    """Wrap a swing target into [-180, 180) — 190° is the same physical
    position as -170°, and joint limits are stated in this range."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _normalise_swing_rad(angle_rad: float) -> float:
    """Wrap a swing target into [-pi, pi) — the radian counterpart."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _check_point_finite_annulus(
    label: str, x_m: float, y_m: float, reach_min_m: float, reach_max_m: float
) -> None:
    if not (math.isfinite(x_m) and math.isfinite(y_m)):
        raise ValueError(f"{label} point is not finite: ({x_m}, {y_m})")
    radius = math.hypot(x_m, y_m)
    if not (reach_min_m <= radius <= reach_max_m):
        raise ValueError(
            f"{label} point at {radius:.2f} m is outside the reachable annulus "
            f"[{reach_min_m:.2f}, {reach_max_m:.2f}] m — navigate_relative the "
            "undercarriage closer, then dig."
        )


def _check_work_joints(driver: Any) -> None:
    available = set(driver.joint_names)
    missing = [name for name in REQUIRED_JOINTS if name not in available]
    if missing:
        raise ValueError(
            f"dig cycle needs joints {REQUIRED_JOINTS}; this body lacks {missing}"
        )


def check_dig_preconditions(
    driver: Any,
    *,
    dig_x_m: float,
    dig_y_m: float,
    reach_min_m: float,
    reach_max_m: float,
) -> None:
    """Dig-side preconditions: dig point finite/in annulus, joints present, bucket EMPTY.

    Shared by ``dig_soil`` and the compound ``dig``. The empty-bucket rule is the
    dig contract: a loaded bucket must be dumped (or homed) first.
    """
    _check_point_finite_annulus(
        "dig", float(dig_x_m), float(dig_y_m), reach_min_m, reach_max_m
    )
    _check_work_joints(driver)
    if driver.scoop_state():
        raise ValueError(
            "bucket is still loaded — dump it (or home) before digging again"
        )


def check_dump_preconditions(
    driver: Any,
    *,
    dump_x_m: float,
    dump_y_m: float,
    reach_min_m: float,
    reach_max_m: float,
) -> None:
    """Dump-side preconditions: dump point finite/in annulus, joints present, bucket LOADED.

    The loaded rule is the dump contract: an empty bucket has nothing to
    release — ``dump_soil`` follows ``dig_soil``.
    """
    _check_point_finite_annulus(
        "dump", float(dump_x_m), float(dump_y_m), reach_min_m, reach_max_m
    )
    _check_work_joints(driver)
    if not driver.scoop_state():
        raise ValueError("bucket is empty — nothing to dump (dig first)")


def check_cycle_preconditions(
    driver: Any,
    *,
    dig_x_m: float,
    dig_y_m: float,
    dump_x_m: float,
    dump_y_m: float,
    reach_min_m: float = 1.0,
    reach_max_m: float = 6.0,
) -> None:
    """Compound-cycle preconditions: BOTH points valid and the bucket EMPTY.

    Composed of the two split checks minus the loaded rule (the compound starts
    empty and ends empty); ``dig_soil``/``dump_soil`` use their own halves.
    """
    _check_point_finite_annulus(
        "dig", float(dig_x_m), float(dig_y_m), reach_min_m, reach_max_m
    )
    _check_point_finite_annulus(
        "dump", float(dump_x_m), float(dump_y_m), reach_min_m, reach_max_m
    )
    _check_work_joints(driver)
    if driver.scoop_state():
        raise ValueError(
            "bucket is still loaded — dump it (or home) before digging again"
        )


def _work_tuning(tuning: dict[str, float] | None) -> dict[str, float]:
    return {**DEFAULT_DIG_TUNING, **(tuning or {})}


def _frame(t: dict[str, float], prefix: str) -> dict[str, float]:
    """One boom/arm/bucket keyframe (named `*_deg` for history, native units)."""
    return {
        "boom": t[f"{prefix}_boom_deg"],
        "arm": t[f"{prefix}_arm_deg"],
        "bucket": t[f"{prefix}_bucket_deg"],
    }


def _swing_for(t: dict[str, float], x_m: float, y_m: float, swing_unit: str) -> float:
    """The swing-joint target that aims the upper body at a ground point."""
    bearing = math.atan2(y_m, x_m)
    if "swing_offset" in t:
        offset = t["swing_offset"]
    else:
        offset = (
            t["swing_offset_deg"]
            if swing_unit == "deg"
            else math.radians(t["swing_offset_deg"])
        )
    if swing_unit == "deg":
        return _normalise_swing(math.degrees(bearing) + offset)
    return _normalise_swing_rad(bearing + offset)


def execute_dig_phase(
    driver: Any,
    *,
    dig_x_m: float,
    dig_y_m: float,
    tuning: dict[str, float] | None = None,
    reach_min_m: float = 1.0,
    reach_max_m: float = 6.0,
    swing_unit: str = "deg",
    depth_scale: float = 1.0,
    travel: dict[str, float] | None = None,
    cut_path: list[dict[str, float]] | None = None,
    carry: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Dig half of the keyframe cycle: travel → swing → ready → cut → scoop(True) → carry.

    Ends with the bucket LOADED at the carry posture (or at the curl frame when
    no carry keyframe is configured). Raises the dig-side preconditions.
    """
    if not (0.0 <= depth_scale <= 1.0):
        raise ValueError(f"depth_scale must be within [0, 1], got {depth_scale!r}")
    check_dig_preconditions(
        driver,
        dig_x_m=dig_x_m,
        dig_y_m=dig_y_m,
        reach_min_m=reach_min_m,
        reach_max_m=reach_max_m,
    )
    t = _work_tuning(tuning)
    swing_dig = _swing_for(t, dig_x_m, dig_y_m, swing_unit)
    ready = _frame(t, "ready")
    dig = {
        joint: ready[joint] + depth_scale * (value - ready[joint])
        for joint, value in _frame(t, "dig").items()
    }
    curl = {
        joint: ready[joint] + depth_scale * (value - ready[joint])
        for joint, value in _frame(t, "curl").items()
    }
    started = time.perf_counter()
    if travel is not None:
        driver.move_joints_blocking(dict(travel))
    driver.move_joints_blocking({"swing": swing_dig})
    driver.move_joints_blocking(dict(ready))
    if cut_path:
        for frame in cut_path:
            scaled = {
                joint: (
                    ready[joint] + depth_scale * (value - ready[joint])
                    if joint in ready
                    else value
                )
                for joint, value in frame.items()
            }
            driver.move_joints_blocking(scaled)
    else:
        driver.move_joints_blocking(dig)
        driver.move_joints_blocking(curl)
    driver.mark_scoop(True)
    if carry is not None:
        driver.move_joints_blocking(dict(carry))
    return {"loaded": True, "cycle_s": time.perf_counter() - started}


def execute_dump_phase(
    driver: Any,
    *,
    dump_x_m: float,
    dump_y_m: float,
    tuning: dict[str, float] | None = None,
    reach_min_m: float = 1.0,
    reach_max_m: float = 6.0,
    swing_unit: str = "deg",
    dump_dwell_s: float = 0.0,
) -> dict[str, Any]:
    """Dump half of the keyframe cycle: swing to the dump bearing → dump → scoop(False).

    Requires a LOADED bucket (the dump contract); ends unloaded.
    """
    if dump_dwell_s < 0:
        raise ValueError(f"dump_dwell_s must be >= 0, got {dump_dwell_s!r}")
    check_dump_preconditions(
        driver,
        dump_x_m=dump_x_m,
        dump_y_m=dump_y_m,
        reach_min_m=reach_min_m,
        reach_max_m=reach_max_m,
    )
    t = _work_tuning(tuning)
    swing_dump = _swing_for(t, dump_x_m, dump_y_m, swing_unit)
    started = time.perf_counter()
    driver.move_joints_blocking({"swing": swing_dump})
    driver.move_joints_blocking(_frame(t, "dump"))
    if dump_dwell_s > 0:
        time.sleep(float(dump_dwell_s))
    driver.mark_scoop(False)
    return {
        "volume_m3": float(t["bucket_volume_m3"]),
        "cycle_s": time.perf_counter() - started,
    }


def execute_dig_cycle(
    driver: Any,
    *,
    dig_x_m: float,
    dig_y_m: float,
    dump_x_m: float,
    dump_y_m: float,
    tuning: dict[str, float] | None = None,
    reach_min_m: float = 1.0,
    reach_max_m: float = 6.0,
    swing_unit: str = "deg",
    depth_scale: float = 1.0,
    travel: dict[str, float] | None = None,
    cut_path: list[dict[str, float]] | None = None,
    carry: dict[str, float] | None = None,
    dump_dwell_s: float = 0.0,
) -> dict[str, float]:
    """Run one dig-and-dump cycle on ``driver``; return {volume_m3, cycle_s}.

    The compound cycle — exactly :func:`execute_dig_phase` followed by
    :func:`execute_dump_phase` — kept as a single call for the common case
    where one machine position serves both the dig point and the dump point.

    ``swing_unit`` — the swing joint's angle unit ("deg" or "rad"); keyframes
    for boom/arm/bucket are always in that joint's NATIVE unit (a hydraulic
    cylinder body speaks metres, not degrees — the tuning keys keep their
    historical *_deg names but carry native values via config).

    ``depth_scale`` — how deep the dig/curl keyframes bite, in [0, 1]: 1.0 is
    the full tuned depth (default, identical to historical behaviour), 0.0
    would stay at the ready pose. The dig and curl frames interpolate toward
    the ready frame; swing targets and the dump frame are untouched. The demo
    recorder uses this to vary bite depth with goal radius (远挖浅咬) so the
    learned policy sees goal-dependent behaviour, not one fixed curve.

    Optional natural-cycle stages (all default off, so 365 keeps the original
    six-move behaviour):
      ``travel``      raise to a clear pose before slewing to the dig bearing;
      ``cut_path``    follow a list of frames instead of the dig/curl pair,
                      making the cut a dragged slice rather than one jump;
      ``carry``       lift/retract the loaded bucket before the dump swing;
      ``dump_dwell_s`` hold the dump pose so the soil can empty.

    Raises ValueError (surfaced by the api as a DigFailure dict) through
    :func:`check_cycle_preconditions`, or when ``depth_scale`` leaves [0, 1].
    """
    if not (0.0 <= depth_scale <= 1.0):
        raise ValueError(f"depth_scale must be within [0, 1], got {depth_scale!r}")
    if dump_dwell_s < 0:
        raise ValueError(f"dump_dwell_s must be >= 0, got {dump_dwell_s!r}")
    check_cycle_preconditions(
        driver,
        dig_x_m=dig_x_m,
        dig_y_m=dig_y_m,
        dump_x_m=dump_x_m,
        dump_y_m=dump_y_m,
        reach_min_m=reach_min_m,
        reach_max_m=reach_max_m,
    )
    started = time.perf_counter()
    execute_dig_phase(
        driver,
        dig_x_m=dig_x_m,
        dig_y_m=dig_y_m,
        tuning=tuning,
        reach_min_m=reach_min_m,
        reach_max_m=reach_max_m,
        swing_unit=swing_unit,
        depth_scale=depth_scale,
        travel=travel,
        cut_path=cut_path,
        carry=carry,
    )
    execute_dump_phase(
        driver,
        dump_x_m=dump_x_m,
        dump_y_m=dump_y_m,
        tuning=tuning,
        reach_min_m=reach_min_m,
        reach_max_m=reach_max_m,
        swing_unit=swing_unit,
        dump_dwell_s=dump_dwell_s,
    )
    t = _work_tuning(tuning)
    return {
        "volume_m3": float(t["bucket_volume_m3"]) * depth_scale,
        "cycle_s": time.perf_counter() - started,
    }
