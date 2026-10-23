# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""3D IK work-cycle tests: the dump point must be reachable with the bucket OPEN."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jiuwen_agx.agx_excavator.work_ik import execute_pose_dig_cycle
from jiuwen_agx.excavator_ik import ExcavatorIk, machine_file_path

_JOINTS = ("swing", "boom", "arm", "bucket")
_LIMITS = {
    "swing": (-3.14159, 3.14159),
    "boom": (-1.570796, 1.570796),
    "arm": (0.0, 2.2183),
    "bucket": (0.0, 3.248105),
}
# The 215F keyframes, in native radians (the config carries these under the
# historical *_deg names). The IK path needs a real ready/dig posture; the
# generic DEFAULT_DIG_TUNING is in degrees for the 365 keyframe path.
_TUNING = {
    "ready_boom_deg": -0.392224,
    "ready_arm_deg": 1.329440,
    "ready_bucket_deg": 0.001275,
    "dig_boom_deg": -0.550090,
    "dig_arm_deg": 1.084667,
    "dig_bucket_deg": 0.386657,
    "curl_boom_deg": -0.641547,
    "curl_arm_deg": 1.109371,
    "curl_bucket_deg": 0.449709,
}


def _ik() -> ExcavatorIk:
    ik = ExcavatorIk.from_machine_file(
        machine_file_path("configs/machine_swe215f.json")
    )
    ik.set_execution_limits_native(_LIMITS)
    return ik


class _FakeDriver:
    """Exact joint/tip feedback; optionally reports a load that never empties."""

    joint_names = _JOINTS

    def __init__(
        self,
        ik: ExcavatorIk,
        *,
        stuck_load: bool = False,
        fail_when_tip_near: tuple[float, float, float] | None = None,
    ) -> None:
        self._ik = ik
        self._stuck_load = stuck_load
        self._loaded = False
        self.fail_when_tip_near = fail_when_tip_near
        self.calls: list[dict[str, float]] = []
        self.q = ik.native_to_policy(
            dict(zip(_JOINTS, (0.0, -0.670305, 0.678116, 1.992748), strict=True))
        )

    def get_joint_positions(self) -> dict[str, float]:
        return self._ik.policy_to_native(self.q)

    def move_joints_blocking(self, targets, **_kwargs):
        self.calls.append({str(k): float(v) for k, v in targets.items()})
        native = self._ik.policy_to_native(self.q)
        native.update({str(k): float(v) for k, v in targets.items()})
        next_q = self._ik.native_to_policy(native)
        if self.fail_when_tip_near is not None:
            tip = np.asarray(self._ik.forward(next_q), dtype=float)
            target = np.asarray(self.fail_when_tip_near, dtype=float)
            if float(np.linalg.norm(tip - target)) < 0.05:
                raise ValueError("drag3: simulated stall")
        self.q = next_q
        return native

    def snapshot(self, points=None):
        return {
            "tip": {"policy_m": list(self._ik.forward(self.q))},
            "surface_m": [1.339, 0.0],
        }

    def scoop_state(self) -> bool:
        return True if self._stuck_load else self._loaded

    def mark_scoop(self, loaded: bool) -> None:
        self._loaded = bool(loaded)


def _run(
    ik: ExcavatorIk,
    driver: _FakeDriver,
    dump,
    *,
    dump_tolerance_m: float = 0.25,
    dig=(5.0, 2.0, 1.339),
    min_drag_radius_m: float = 0.0,
):
    return execute_pose_dig_cycle(
        driver,
        ik=ik,
        dig_pose=dig,
        dump_pose=dump,
        tuning=_TUNING,
        approach_gap_m=0.35,
        bite_m=0.25,
        cut_length_m=1.0,
        min_drag_radius_m=min_drag_radius_m,
        carry_lift_m=1.0,
        dig_bucket_open_native=1.5,
        dig_bucket_curl_native=0.1,
        dump_bucket_open_native=2.0,
        ik_tolerance_m=0.05,
        dump_tolerance_m=dump_tolerance_m,
        ik_max_iters=80,
        ik_stall_iters=8,
        dump_dwell_s=0.0,
    )


def test_dump_point_is_reached_with_bucket_open():
    ik = _ik()
    result = _run(ik, _FakeDriver(ik), (5.0, -1.0, 3.5))
    assert result["mode"] == "ik"
    # The dump uses its own (looser) arrival tolerance; the achieved residual
    # is allowed to exceed the dig tolerance before the bucket is opened.
    assert result["tip_error_m"] <= 0.25
    assert result["volume_m3"] > 0.0


def test_low_dump_point_is_refused_not_half_opened():
    ik = _ik()
    with pytest.raises(
        ValueError, match="cannot be reached while carrying the bucket closed"
    ):
        _run(ik, _FakeDriver(ik), (3.0, 0.0, 0.0))


def test_bucket_still_loaded_after_open_is_refused():
    ik = _ik()
    with pytest.raises(ValueError, match="still loaded"):
        _run(ik, _FakeDriver(ik, stuck_load=True), (5.0, -1.0, 3.5))


def test_near_dig_point_is_refused_before_any_motion():
    ik = _ik()
    driver = _FakeDriver(ik)

    with pytest.raises(ValueError, match="inward cut"):
        _run(
            ik,
            driver,
            (5.0, -1.0, 3.5),
            dig=(3.0, 0.72, 1.339),
            min_drag_radius_m=3.0,
        )

    assert driver.calls == [], "the drag-radius check must run before the ready move"


def test_contact_failure_opens_the_bucket_before_raising():
    ik = _ik()
    origin = getattr(ik, "origin_in_base", (0.0, 0.0, 0.0))
    radius = math.hypot(5.0, 2.0)
    scaled = max(radius - 1.0, 0.6) / radius
    drag_tip = (
        5.0 * scaled - origin[0],
        2.0 * scaled - origin[1],
        1.339 - 0.25 + 0.05 - origin[2],
    )
    driver = _FakeDriver(ik, fail_when_tip_near=drag_tip)

    with pytest.raises(ValueError, match="simulated stall"):
        _run(ik, driver, (5.0, -1.0, 3.5))

    # The failure itself, then the cleanup opens the bucket in place.
    assert driver.calls[-1] == {"bucket": 2.0}
    assert driver._loaded is False
