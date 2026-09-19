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

__all__ = ["DEFAULT_DIG_TUNING", "REQUIRED_JOINTS", "execute_dig_cycle"]

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
) -> dict[str, float]:
    """Run one dig-and-dump cycle on ``driver``; return {volume_m3, cycle_s}.

    ``swing_unit`` — the swing joint's angle unit ("deg" or "rad"); keyframes
    for boom/arm/bucket are always in that joint's NATIVE unit (a hydraulic
    cylinder body speaks metres, not degrees — the tuning keys keep their
    historical *_deg names but carry native values via config).

    Raises ValueError (surfaced by the api as a DigFailure dict) when a ground
    point lies outside the reachable annulus, a required joint is missing, or
    the bucket is still loaded from a previous cycle.
    """
    points = (
        ("dig", float(dig_x_m), float(dig_y_m)),
        ("dump", float(dump_x_m), float(dump_y_m)),
    )
    for label, x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{label} point is not finite: ({x}, {y})")
        radius = math.hypot(x, y)
        if not (reach_min_m <= radius <= reach_max_m):
            raise ValueError(
                f"{label} point at {radius:.2f} m is outside the reachable annulus "
                f"[{reach_min_m:.2f}, {reach_max_m:.2f}] m — navigate_relative the "
                "undercarriage closer, then dig."
            )

    available = set(driver.joint_names)
    missing = [name for name in REQUIRED_JOINTS if name not in available]
    if missing:
        raise ValueError(
            f"dig cycle needs joints {REQUIRED_JOINTS}; this body lacks {missing}"
        )

    if driver.scoop_state():
        raise ValueError(
            "bucket is still loaded — dump it (or home) before digging again"
        )

    t = {**DEFAULT_DIG_TUNING, **(tuning or {})}
    # swing 目标单位跟随 swing_unit；偏移键兼容旧名 swing_offset_deg（deg）
    if "swing_offset" in t:
        offset = t["swing_offset"]
    else:
        offset = (
            t["swing_offset_deg"]
            if swing_unit == "deg"
            else math.radians(t["swing_offset_deg"])
        )
    bearing_dig = math.atan2(dig_y_m, dig_x_m)
    bearing_dump = math.atan2(dump_y_m, dump_x_m)
    if swing_unit == "deg":
        swing_dig = _normalise_swing(math.degrees(bearing_dig) + offset)
        swing_dump = _normalise_swing(math.degrees(bearing_dump) + offset)
    else:
        swing_dig = _normalise_swing_rad(bearing_dig + offset)
        swing_dump = _normalise_swing_rad(bearing_dump + offset)

    started = time.perf_counter()
    # 对准 → 就位 → 下铲 → 收斗(装满) → 摆转 → 卸料
    driver.move_joints_blocking({"swing": swing_dig})
    driver.move_joints_blocking(
        {
            "boom": t["ready_boom_deg"],
            "arm": t["ready_arm_deg"],
            "bucket": t["ready_bucket_deg"],
        }
    )
    driver.move_joints_blocking(
        {
            "boom": t["dig_boom_deg"],
            "arm": t["dig_arm_deg"],
            "bucket": t["dig_bucket_deg"],
        }
    )
    driver.move_joints_blocking(
        {
            "boom": t["curl_boom_deg"],
            "arm": t["curl_arm_deg"],
            "bucket": t["curl_bucket_deg"],
        }
    )
    driver.mark_scoop(True)
    driver.move_joints_blocking({"swing": swing_dump})
    driver.move_joints_blocking(
        {
            "boom": t["dump_boom_deg"],
            "arm": t["dump_arm_deg"],
            "bucket": t["dump_bucket_deg"],
        }
    )
    driver.mark_scoop(False)
    cycle_s = time.perf_counter() - started

    return {"volume_m3": float(t["bucket_volume_m3"]), "cycle_s": float(cycle_s)}
