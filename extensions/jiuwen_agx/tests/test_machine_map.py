# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""每机器标定文件（machine_map）的换算测试 — 无 torch/小脑组依赖，恒可运行。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jiuwen_agx.machine_map import MachineMap, load_machine_file


def _machine(tmp_path: Path, joints: list[dict], **extra: object) -> Path:
    payload = {
        "machine": "unit-test",
        "calibrated": True,
        "policy_joint_order": [j["name"] for j in joints],
        "joints": joints,
        **extra,
    }
    path = tmp_path / "machine.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _cyl(
    name: str, lo_d: float, hi_d: float, lo_a: float, hi_a: float, coeffs=None
) -> dict:
    return {
        "name": name,
        "native_unit": "m",
        "sign": 1,
        "offset_rad": 0.0,
        "hydraulic_conversion": {
            "min_displacement_m": lo_d,
            "max_displacement_m": hi_d,
            "min_angle_deg": lo_a,
            "max_angle_deg": hi_a,
            "poly_coeffs": coeffs,
        },
    }


def test_linear_cylinder_round_trip(tmp_path: Path) -> None:
    path = _machine(
        tmp_path,
        [
            {"name": "swing", "native_unit": "rad", "sign": 1, "offset_rad": 0.0},
            _cyl("boom", 0.5, 1.5, -35.0, 75.0),
        ],
    )
    machine = MachineMap.load(path)
    jmap = machine.joints["boom"]
    for displacement in (0.5, 0.8, 1.0, 1.4, 1.5):
        angle = jmap.to_policy(displacement)
        assert jmap.to_native(angle) == pytest.approx(displacement, abs=1e-9)
    # endpoints map to the declared angles
    assert pytest.approx(math_deg(jmap.to_policy(0.5)), abs=1e-6) == -35.0
    assert pytest.approx(math_deg(jmap.to_policy(1.5)), abs=1e-6) == 75.0


def math_deg(rad: float) -> float:
    import math

    return math.degrees(rad)


def test_swing_sign_and_offset(tmp_path: Path) -> None:
    path = _machine(
        tmp_path,
        [{"name": "swing", "native_unit": "rad", "sign": -1, "offset_rad": 0.25}],
    )
    jmap = MachineMap.load(path).joints["swing"]
    native = 0.3
    angle = jmap.to_policy(native)
    assert angle == pytest.approx(-0.3 + 0.25)
    assert jmap.to_native(angle) == pytest.approx(native)


def test_polynomial_inverse_bisection(tmp_path: Path) -> None:
    # Monotone on dn in [0, 1]: angle_deg = 0 + 90*dn + 10*dn^2
    path = _machine(
        tmp_path, [_cyl("bucket", 0.4, 1.2, 0.0, 100.0, coeffs=[0.0, 90.0, 10.0])]
    )
    jmap = MachineMap.load(path).joints["bucket"]
    for displacement in (0.4, 0.6, 0.8, 1.0, 1.2):
        angle = jmap.to_policy(displacement)
        assert math_deg(angle) == pytest.approx(
            90.0 * ((displacement - 0.4) / 0.8)
            + 10.0 * ((displacement - 0.4) / 0.8) ** 2,
            abs=1e-4,
        )
        assert jmap.to_native(angle) == pytest.approx(displacement, abs=1e-6)


def test_load_errors_are_readable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="joint"):
        load_machine_file(_machine(tmp_path, []))  # empty joint list -> empty order
    bad = _machine(tmp_path, [{"name": "boom", "native_unit": "m", "sign": 1}])
    with pytest.raises(ValueError, match="hydraulic_conversion"):
        MachineMap.load(bad)
    degenerate = _machine(tmp_path, [_cyl("boom", 1.0, 1.0, 0.0, 10.0)])
    with pytest.raises(ValueError, match="increasing"):
        MachineMap.load(degenerate)
