# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""3D bucket-tip pose work cycle for the 215F excavator (IK-driven).

The brain supplies two 3D bucket-tip poses in the base frame:

    dig_pose  = (x, y, z)   # where the bucket is placed at the start of the cut
    dump_pose = (x, y, z)   # where the bucket is placed to release

This module owns the motion between them: approach above the dig point, entry,
press down, drag/cut with the bucket OPEN, curl INWARD to hold the load, carry,
swing, reach the dump point with the bucket CLOSED, then open it OUTWARD to
release. Free-space segments (approach / carry / dump approach) play one smooth
joint-space trajectory; the contact segment (entry/press/drag) stays closed-loop
on the measured bucket-tip pose from ``driver.snapshot()``.

It does NOT choose or clamp the points. An unreachable pose fails with a
readable error so the brain can move the base or pick another point.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from jiuwen_agx.agx_excavator.work import (
    DEFAULT_DIG_TUNING,
    check_cycle_preconditions,
)
from jiuwen_agx.trajectory import build_trajectory

__all__ = ["SnapshotUnavailable", "execute_pose_dig_cycle"]

logger = logging.getLogger(__name__)


class SnapshotUnavailable(RuntimeError):
    """The backend cannot report the bucket tip / soil surface (old bridge, mock)."""


def _policy_point(
    x_m: float, y_m: float, z_m: float, origin: tuple[float, float, float]
) -> np.ndarray:
    return np.array(
        [float(x_m) - origin[0], float(y_m) - origin[1], float(z_m) - origin[2]],
        dtype=np.float64,
    )


def execute_pose_dig_cycle(
    driver: Any,
    *,
    ik: Any,
    dig_pose: tuple[float, float, float],
    dump_pose: tuple[float, float, float],
    dig_pitch: float | None = None,
    dump_pitch: float | None = None,
    tuning: dict[str, float] | None = None,
    reach_min_m: float = 1.0,
    reach_max_m: float = 6.0,
    reach_z_min_m: float | None = None,
    reach_z_max_m: float | None = None,
    approach_gap_m: float = 0.35,
    bite_m: float = 0.25,
    cut_length_m: float = 1.0,
    min_drag_radius_m: float = 0.0,
    carry_lift_m: float = 1.0,
    carry: dict[str, float] | None = None,
    dig_bucket_open_native: float = 1.5,
    dig_bucket_curl_native: float = 0.1,
    dump_bucket_open_native: float | None = None,
    ik_tolerance_m: float = 0.05,
    dump_tolerance_m: float = 0.25,
    contact_tolerance_m: float = 0.25,
    ik_max_iters: int = 80,
    ik_stall_iters: int = 8,
    ik_step_timeout_s: float = 5.0,
    dump_dwell_s: float = 0.0,
) -> dict[str, float]:
    """Run one 3D-pose dig/dump cycle; return {volume_m3, cycle_s, mode, tip_error_m}."""
    dig_x, dig_y, dig_z = (float(v) for v in dig_pose)
    dump_x, dump_y, dump_z = (float(v) for v in dump_pose)
    for label, value in (
        ("dig", (dig_x, dig_y, dig_z)),
        ("dump", (dump_x, dump_y, dump_z)),
    ):
        if not all(math.isfinite(v) for v in value):
            raise ValueError(f"{label} pose is not finite: {value}")
    check_cycle_preconditions(
        driver,
        dig_x_m=dig_x,
        dig_y_m=dig_y,
        dump_x_m=dump_x,
        dump_y_m=dump_y,
        reach_min_m=reach_min_m,
        reach_max_m=reach_max_m,
    )
    for label, z in (("dig", dig_z), ("dump", dump_z)):
        if reach_z_min_m is not None and z < reach_z_min_m:
            raise ValueError(f"{label} z={z:.3f} below reach_z_min_m={reach_z_min_m}")
        if reach_z_max_m is not None and z > reach_z_max_m:
            raise ValueError(f"{label} z={z:.3f} above reach_z_max_m={reach_z_max_m}")

    snap = driver.snapshot(points=[[dig_x, dig_y]])
    if not snap or not snap.get("tip"):
        raise SnapshotUnavailable(
            "后端没有提供铲尖位姿（snapshot）；无法执行 3D IK 位姿挖掘"
        )
    tip = snap["tip"]
    if not tip.get("policy_m"):
        raise SnapshotUnavailable("snapshot.tip.policy_m 缺失；无法执行 3D IK")
    surface = None
    surfaces = snap.get("surface_m")
    if isinstance(surfaces, (list, tuple)) and surfaces:
        surface = surfaces[0]
    if surface is not None and not math.isfinite(float(surface)):
        surface = None
    if isinstance(surfaces, (list, tuple)) and surfaces and surfaces[0] is None:
        raise ValueError(
            f"dig point ({dig_x:.2f}, {dig_y:.2f}) is outside the terrain grid"
        )

    # The cut always drags INWARD by cut_length_m. A close dig point therefore
    # puts the drag endpoint inside the arm's collapsed posture (215F measured:
    # arm hits its 0 limit and the tip stalls ~0.45 m short). Refuse before the
    # ready move; the planner can move the undercarriage back and re-read terrain.
    radius = math.hypot(dig_x, dig_y) or 1e-6
    drag_endpoint_m = max(radius - float(cut_length_m), 0.6)
    if min_drag_radius_m > 0 and drag_endpoint_m < float(min_drag_radius_m):
        raise ValueError(
            f"dig point at {radius:.2f} m leaves only {drag_endpoint_m:.2f} m after the "
            f"{float(cut_length_m):.2f} m inward cut; keep the drag endpoint at >= "
            f"{float(min_drag_radius_m):.2f} m (dig radius >= "
            f"{float(min_drag_radius_m) + float(cut_length_m):.2f} m) or move the "
            "undercarriage back and re-read the terrain"
        )

    origin = tuple(getattr(ik, "origin_in_base", (0.0, 0.0, 0.0)))
    t = {**DEFAULT_DIG_TUNING, **(tuning or {})}
    # Start from the calibrated ready posture, not from wherever the previous
    # command left the arm: a half-dumped pose makes the local IK solver chase
    # an arbitrary pitch and can stall for minutes before reaching the dig point.
    ready_native = {
        joint: float(t[f"ready_{joint}_deg"]) for joint in ("boom", "arm", "bucket")
    }
    ready_native["swing"] = 0.0
    q_ready = ik.native_to_policy(ready_native)
    q_ready[0] = math.atan2(dig_y, dig_x)
    driver.move_joints_blocking(ik.policy_to_native(q_ready))
    q = ik.native_to_policy(driver.get_joint_positions())
    started = time.perf_counter()
    worst_error = 0.0

    def goto(
        target: np.ndarray,
        *,
        pitch: float | None,
        label: str,
        tolerance: float | None = None,
    ) -> float:
        nonlocal q, worst_error
        tol = float(ik_tolerance_m if tolerance is None else tolerance)
        error = float("inf")
        last_error = float("inf")
        stall = 0
        for _ in range(int(ik_max_iters)):
            state = driver.snapshot() or {}
            tip_now = state.get("tip") or {}
            measured = tip_now.get("policy_m")
            if measured is None:
                raise SnapshotUnavailable(
                    f"{label}: snapshot 没有铲尖位姿，无法闭环"
                )
            measured = np.asarray(measured, dtype=np.float64)
            error = float(np.linalg.norm(target - measured))
            if error < tol:
                break
            # Progress is judged on the measured TIP error, not on joint motion:
            # under soil load the joints can move very little while the tip is
            # still converging.
            if error >= last_error - 5e-3:
                stall += 1
                if stall >= int(ik_stall_iters):
                    raise ValueError(
                        f"{label}: 铲尖误差连续 {stall} 次没有改善"
                        f"（停在 {error:.3f} m；可能埋在土里/被限位卡住）"
                    )
            else:
                stall = 0
            last_error = error
            q_next = ik.step_ik(q, target, 0.35, target_pitch=pitch)
            driver.move_joints_blocking(
                ik.policy_to_native(q_next), timeout_s=float(ik_step_timeout_s)
            )
            q = ik.native_to_policy(driver.get_joint_positions())
        if error > tol:
            raise ValueError(
                f"{label}: IK 误差 {error:.3f} m 超过容差 {tol:.3f} m"
            )
        worst_error = max(worst_error, error)
        return error

    def goto_fixed(
        target: np.ndarray,
        *,
        joint_index: int,
        value: float,
        label: str,
        tolerance: float | None = None,
    ) -> float:
        """IK with one joint pinned (bucket open at the cut / closed at the dump)."""
        nonlocal q, worst_error
        tol = float(ik_tolerance_m if tolerance is None else tolerance)
        error = float("inf")
        last_error = float("inf")
        stall = 0
        for _ in range(int(ik_max_iters)):
            state = driver.snapshot() or {}
            tip_now = state.get("tip") or {}
            measured = tip_now.get("policy_m")
            if measured is None:
                raise SnapshotUnavailable(
                    f"{label}: snapshot 没有铲尖位姿，无法闭环"
                )
            measured = np.asarray(measured, dtype=np.float64)
            error = float(np.linalg.norm(target - measured))
            if error < tol:
                break
            if error >= last_error - 5e-3:
                stall += 1
                if stall >= int(ik_stall_iters):
                    raise ValueError(
                        f"{label}: 铲尖误差连续 {stall} 次没有改善"
                        f"（停在 {error:.3f} m；可能埋在土里/被限位卡住）"
                    )
            else:
                stall = 0
            last_error = error
            q_next = ik.step_ik_fixed(q, target, joint_index, value, 0.35)
            driver.move_joints_blocking(
                ik.policy_to_native(q_next), timeout_s=float(ik_step_timeout_s)
            )
            q = ik.native_to_policy(driver.get_joint_positions())
        if error > tol:
            raise ValueError(
                f"{label}: IK 误差 {error:.3f} m 超过容差 {tol:.3f} m"
            )
        worst_error = max(worst_error, error)
        return error

    def native_limits() -> np.ndarray:
        rows = []
        for index in range(len(driver.joint_names)):
            low = (ik.limits[index, 0] - ik.offsets[index]) / ik.signs[index]
            high = (ik.limits[index, 1] - ik.offsets[index]) / ik.signs[index]
            rows.append((min(low, high), max(low, high)))
        return np.asarray(rows, dtype=np.float64)

    def play_native(target: dict[str, float], label: str) -> bool:
        """Play a smooth joint-space move to ``target`` (native units).

        Returns False when the backend has no trajectory player (mock/old
        bridge); the caller then uses the closed-loop path. A trajectory is a
        free-space move: it does not try to touch soil.
        """
        names = tuple(driver.joint_names)
        current = driver.get_joint_positions()
        wanted = {name: float(target.get(name, current[name])) for name in names}
        waypoints = np.asarray(
            [
                [float(current[name]) for name in names],
                [float(wanted[name]) for name in names],
            ],
            dtype=np.float64,
        )
        delta = np.abs(waypoints[1] - waypoints[0])
        vmax = np.full(len(names), 0.5, dtype=np.float64)
        vmax[0] = 0.8
        duration = max(0.3, float(np.max(delta / np.maximum(vmax, 1e-6))))
        try:
            trajectory = build_trajectory(
                waypoints,
                [duration],
                rate_hz=60.0,
                rest_ends=True,
                monotone=True,
                vmax=vmax,
                unwrap_joints=[0],
                limits=native_limits(),
            )
            driver.play_trajectory(
                trajectory.frames.tolist(), rate_hz=trajectory.rate_hz
            )
        except Exception as exc:  # noqa: BLE001 - fall back to the closed loop
            logger.warning(
                "[ik] %s: smooth playback unavailable (%s); using closed-loop",
                label,
                exc,
            )
            return False
        return True

    low_native = float((ik.limits[3, 0] - ik.offsets[3]) / ik.signs[3])
    high_native = float((ik.limits[3, 1] - ik.offsets[3]) / ik.signs[3])
    open_native = float(dig_bucket_open_native)
    curl_native = float(dig_bucket_curl_native)
    for label, value in (
        ("dig_bucket_open_native", open_native),
        ("dig_bucket_curl_native", curl_native),
    ):
        if not (low_native <= value <= high_native):
            raise ValueError(
                f"{label}={value:.3f} rad is outside the native bucket range "
                f"[{low_native:.3f}, {high_native:.3f}]"
            )
    if curl_native > open_native:
        raise ValueError(
            "dig_bucket_curl_native must be <= dig_bucket_open_native: the cut "
            "runs with the bucket open (larger angle), then curls inward"
        )
    open_value = float(ik.signs[3]) * open_native + float(ik.offsets[3])
    curl_value = float(ik.signs[3]) * curl_native + float(ik.offsets[3])
    dump_open_native = (
        float(dump_bucket_open_native)
        if dump_bucket_open_native is not None
        else float(getattr(ik, "bucket_open_native", open_native))
    )
    if not (low_native <= dump_open_native <= high_native):
        raise ValueError(
            f"dump bucket opening {dump_open_native:.3f} rad is outside the "
            f"native range [{low_native:.3f}, {high_native:.3f}]"
        )

    approach = _policy_point(dig_x, dig_y, dig_z + approach_gap_m, origin)
    entry = _policy_point(dig_x, dig_y, dig_z, origin)
    press = _policy_point(dig_x, dig_y, dig_z - bite_m, origin)
    radius = math.hypot(dig_x, dig_y) or 1e-6
    scaled = max(radius - cut_length_m, 0.6) / radius
    drag = _policy_point(
        dig_x * scaled, dig_y * scaled, dig_z - bite_m + 0.05, origin
    )
    lift = _policy_point(
        dig_x * scaled, dig_y * scaled, dig_z + carry_lift_m, origin
    )
    dump = _policy_point(dump_x, dump_y, dump_z, origin)
    dump_approach = _policy_point(
        dump_x, dump_y, dump_z + approach_gap_m, origin
    )

    # Early 3D feasibility gate: reject an unreachable dump BEFORE moving soil.
    # The later check uses the actual carry pose; this one runs while the bucket
    # is still empty, so a bad brain point does not leave a loaded bucket that
    # the next plan then refuses to dig with.
    if carry is not None and "boom" in carry and "arm" in carry:
        seed_native = {
            "swing": 0.0,
            "boom": float(carry["boom"]),
            "arm": float(carry["arm"]),
            "bucket": curl_native,
        }
    else:
        seed_native = {
            "swing": 0.0,
            "boom": float(t["ready_boom_deg"]),
            "arm": float(t["ready_arm_deg"]),
            "bucket": curl_native,
        }
    q_seed = ik.native_to_policy(seed_native)
    q_seed[0] = math.atan2(dump_y, dump_x)
    for _ in range(int(ik_max_iters)):
        q_seed = ik.step_ik_fixed(q_seed, dump, 3, curl_value, 0.35)
    seed_error = float(np.linalg.norm(ik.forward(q_seed) - dump))
    if seed_error > float(dump_tolerance_m):
        raise ValueError(
            f"dump pose ({dump_x:.2f}, {dump_y:.2f}, {dump_z:.2f}) cannot be "
            f"reached while carrying the bucket closed ({seed_error:.2f} m "
            "short). Keep the dump point at radius ~4.5-5.5 m and z ~3-4 m "
            "(for example (5,-1,3.5)); do not push it toward the machine."
        )

    def cut(target: np.ndarray, *, label: str, tolerance: float) -> float:
        if dig_pitch is not None:
            return goto(target, pitch=dig_pitch, label=label, tolerance=tolerance)
        return goto_fixed(
            target,
            joint_index=3,
            value=open_value,
            label=label,
            tolerance=tolerance,
        )

    def cleanup_after_contact_failure() -> None:
        """Open the bucket (lifting if possible) after a failed cut.

        The generic recovery cannot release an excavator: it looks for a gripper
        or suction cup and ``set_end_effector`` refuses a body without those
        capabilities. Without this cleanup the next re-plan sees
        ``payload.held`` while ``dig`` requires ``payload.clear`` — a deadlock.
        Best-effort: never masks the original failure.
        """
        logger.warning(
            "[ik] cycle failed during the cut; opening the bucket before returning"
        )
        targets: dict[str, float] = {"bucket": dump_open_native}
        if carry is not None and "boom" in carry and "arm" in carry:
            targets = {
                "boom": float(carry["boom"]),
                "arm": float(carry["arm"]),
                "bucket": dump_open_native,
            }
        try:
            driver.move_joints_blocking(targets, timeout_s=10.0)
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            logger.warning("[ik] cleanup: lift+open failed (%s); opening in place", exc)
            try:
                driver.move_joints_blocking(
                    {"bucket": dump_open_native}, timeout_s=10.0
                )
            except Exception as open_exc:  # noqa: BLE001
                logger.warning("[ik] cleanup: opening the bucket failed: %s", open_exc)
        if dump_dwell_s > 0:
            time.sleep(float(dump_dwell_s))
        try:
            driver.mark_scoop(False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ik] cleanup: mark_scoop(False) failed: %s", exc)

    # Free-space approach: solve the joint target, then play one smooth move.
    q_approach = np.asarray(q, dtype=np.float64).copy()
    for _ in range(int(ik_max_iters)):
        q_approach = ik.step_ik_fixed(q_approach, approach, 3, open_value, 0.35)
    approach_error = float(np.linalg.norm(ik.forward(q_approach) - approach))
    if approach_error <= float(ik_tolerance_m) and play_native(
        ik.policy_to_native(q_approach), "approach"
    ):
        q = ik.native_to_policy(driver.get_joint_positions())
    else:
        cut(approach, label="approach", tolerance=ik_tolerance_m)

    # Cut with the bucket OPEN: the teeth enter and drag through the soil.
    # Any failure here happens BEFORE mark_scoop(True), so the generic retreat
    # would home a bucket that is physically holding soil; clean up in place.
    try:
        cut(entry, label="entry", tolerance=ik_tolerance_m)
        cut(press, label="press", tolerance=contact_tolerance_m)
        for step in range(1, 4):
            alpha = step / 3.0
            cut(
                press + (drag - press) * alpha,
                label=f"drag{step}",
                tolerance=contact_tolerance_m,
            )
        cut(drag, label="drag", tolerance=contact_tolerance_m)

        # Curl INWARD (toward the machine) to close the bucket and hold the load.
        # Increasing the angle here would open it further and spill the soil.
        driver.move_joints_blocking({"bucket": curl_native})
        driver.mark_scoop(True)
    except Exception:
        cleanup_after_contact_failure()
        raise

    # Carry lift: the calibrated carry boom/arm when available, otherwise the
    # IK solution for the lift point with the bucket open; the bucket itself
    # stays closed so the load does not fall out during the swing.
    if carry is not None and "boom" in carry and "arm" in carry:
        carry_target = {
            "boom": float(carry["boom"]),
            "arm": float(carry["arm"]),
            "bucket": curl_native,
        }
        if not play_native(carry_target, "carry"):
            driver.move_joints_blocking(carry_target)
    else:
        q_lift = np.asarray(q, dtype=np.float64).copy()
        for _ in range(int(ik_max_iters)):
            q_lift = ik.step_ik_fixed(q_lift, lift, 3, open_value, 0.35)
        native_lift = ik.policy_to_native(q_lift)
        native_lift["bucket"] = curl_native
        if not play_native(native_lift, "carry"):
            driver.move_joints_blocking(native_lift)
    q = ik.native_to_policy(driver.get_joint_positions())
    if not driver.scoop_state():
        raise ValueError(
            "dig did not load the bucket — the cut was too shallow; increase "
            "bite_m / cut_length_m or choose a spot with more material"
        )

    # Dump: arrive with the bucket CLOSED (carrying), then open it OUTWARD at
    # the dump point. The brain's dump point is where release starts; opening
    # moves the tip outward/down and the soil falls.
    q_try = np.asarray(q, dtype=np.float64).copy()
    for _ in range(int(ik_max_iters)):
        q_try = ik.step_ik_fixed(q_try, dump, 3, curl_value, 0.35)
    closed_error = float(np.linalg.norm(ik.forward(q_try) - dump))
    if closed_error > float(dump_tolerance_m):
        raise ValueError(
            f"dump pose ({dump_x:.2f}, {dump_y:.2f}, {dump_z:.2f}) cannot be "
            f"reached while carrying the bucket closed ({closed_error:.2f} m "
            "short). Raise/move the dump point (about 4.5-5.5 m out and 3-4 m "
            "high), or move the undercarriage first."
        )
    if dump_pitch is not None:
        goto(
            dump_approach,
            pitch=dump_pitch,
            label="dump_approach",
            tolerance=dump_tolerance_m,
        )
        dump_error = goto(
            dump, pitch=dump_pitch, label="dump", tolerance=dump_tolerance_m
        )
    else:
        q_dump_approach = np.asarray(q, dtype=np.float64).copy()
        for _ in range(int(ik_max_iters)):
            q_dump_approach = ik.step_ik_fixed(
                q_dump_approach, dump_approach, 3, curl_value, 0.35
            )
        dump_approach_error = float(
            np.linalg.norm(ik.forward(q_dump_approach) - dump_approach)
        )
        # The approach is only a staging point. At a far-side dump it can be
        # unreachable even when the dump point itself is reachable (the point
        # is higher); trying it anyway drives the arm into its lower limit and
        # aborts before the bucket ever opens. Skip it in that case.
        if dump_approach_error <= float(dump_tolerance_m):
            if play_native(ik.policy_to_native(q_dump_approach), "dump_approach"):
                q = ik.native_to_policy(driver.get_joint_positions())
            else:
                try:
                    goto_fixed(
                        dump_approach,
                        joint_index=3,
                        value=curl_value,
                        label="dump_approach",
                        tolerance=dump_tolerance_m,
                    )
                except ValueError as exc:
                    logger.warning(
                        "[ik] dump approach failed (%s); going directly to the dump point",
                        exc,
                    )
        else:
            logger.warning(
                "[ik] dump approach unreachable by %.2f m; going directly to the dump point",
                dump_approach_error,
            )
        dump_error = goto_fixed(
            dump,
            joint_index=3,
            value=curl_value,
            label="dump",
            tolerance=dump_tolerance_m,
        )
    if dump_error > float(ik_tolerance_m):
        logger.warning(
            "[ik] dump reached within %.3f m (dump tolerance %.3f m); "
            "opening the bucket at the achieved pose",
            dump_error,
            dump_tolerance_m,
        )

    driver.move_joints_blocking({"bucket": dump_open_native})
    if dump_dwell_s > 0:
        time.sleep(float(dump_dwell_s))
    driver.mark_scoop(False)
    if driver.scoop_state():
        for _ in range(2):
            driver.move_joints_blocking({"bucket": curl_native})
            driver.move_joints_blocking({"bucket": dump_open_native})
            if dump_dwell_s > 0:
                time.sleep(float(dump_dwell_s))
        if driver.scoop_state():
            raise ValueError(
                f"dump pose ({dump_x:.2f}, {dump_y:.2f}, {dump_z:.2f}) was "
                "reached, but the bucket is still loaded after opening it "
                "outward. Choose a different dump point/posture."
            )

    t = {**DEFAULT_DIG_TUNING, **(tuning or {})}
    return {
        "volume_m3": float(t["bucket_volume_m3"]),
        "cycle_s": time.perf_counter() - started,
        "mode": "ik",
        "tip_error_m": worst_error,
    }
