# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""365 纯几何标定回归：数学单元 + 真 OpenPLX 集成（缺文件自动 skip）+ 真尺往返。

集成测试需要 ``E:/AGX-2.42.2.1/data/models/365_model.openplx``（导出文本，
不在仓库里）；机器文件 ``configs/machine_agx365.json`` 已提交，往返测试恒可跑。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from jiuwen_agx.cylinder_calibration import (
    OPENPLX_DEFAULT,
    _circle_intersect,
    _interp,
    fit_poly_table,
    geometry_report,
    load_openplx,
    scan_machine,
    stroke_window,
)
from jiuwen_agx.machine_map import MachineMap

_EXT = Path(__file__).resolve().parents[1]
_MACHINE = _EXT / "configs" / "machine_agx365.json"


# ---------------------------------------------------------------------------- math
def test_circle_intersect_two_solutions_and_failures():
    a = [0.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    p1, p2 = _circle_intersect(a, b, 0.6, 0.6, [0, 0, 1], None)
    assert math.isclose(p1[0], 0.5, abs_tol=1e-9) and math.isclose(
        p2[0], 0.5, abs_tol=1e-9
    )
    assert abs(p1[1]) == pytest.approx(math.sqrt(0.11), abs=1e-9)
    assert p1[1] * p2[1] == pytest.approx(-0.11, abs=1e-9)  # opposite sides
    with pytest.raises(ValueError):
        _circle_intersect(a, b, 0.4, 0.4, [0, 0, 1], None)  # too far apart
    with pytest.raises(ValueError):
        _circle_intersect(a, b, 5.0, 0.1, [0, 0, 1], None)  # one inside the other
    # branch continuity: prev picks the nearer solution
    first, second = (p1, p2) if p1[1] > 0 else (p2, p1)
    near = _circle_intersect(a, b, 0.6, 0.6, [0, 0, 1], [0.5, 0.4, 0.0])
    assert near[0] == first


def test_fit_poly_table_direction_and_error():
    d = np.linspace(-0.7, 0.4, 60)
    inc = fit_poly_table(d, 10.0 + 30.0 * (d + 0.7) / 1.1)
    assert inc["monotone"] and inc["increasing"] and inc["max_err_deg"] < 1e-6
    dec = fit_poly_table(d, 50.0 - 30.0 * (d + 0.7) / 1.1)
    assert dec["monotone"] and not dec["increasing"] and dec["max_err_deg"] < 1e-6


def test_stroke_window_and_interp():
    thetas = list(np.arange(-100.0, 101.0, 1.0))
    disp = [t / 100.0 for t in thetas]
    scan = {
        "scan": {"theta_deg": thetas, "disp_m": disp, "angle_deg": [-t for t in thetas]}
    }
    window = stroke_window(scan, (-0.5, 0.5))
    assert window == pytest.approx((-50.0, 50.0))
    assert _interp(thetas, disp, 0.25, True) == pytest.approx(25.0)
    assert _interp(thetas, disp, 2.0, True) is None


# ---------------------------------------------------------------------------- machine file
def test_machine_file_round_trip_and_strokes():
    machine = MachineMap.load(_MACHINE)
    assert machine.calibrated
    assert machine.data["links_m"] == pytest.approx([7.8, 3.6, 2.2733], abs=1e-3)
    for name, stroke in (
        ("boom", (-0.7, 0.4)),
        ("arm", (-0.88, 0.8)),
        ("bucket", (-1.0, 0.312)),
    ):
        jmap = machine.joints[name]
        conv = jmap.conv
        assert (
            conv["min_displacement_m"],
            conv["max_displacement_m"],
        ) == pytest.approx(stroke)
        assert conv["poly_coeffs"], f"{name}: polynomial mapping expected"
        xs = np.linspace(stroke[0], stroke[1], 101)
        worst = max(abs(jmap.to_native(jmap.to_policy(x)) - x) for x in xs)
        assert worst < 1e-9, f"{name}: round trip err {worst}"
    # recorded teacher strokes sit inside the calibrated intervals
    assert machine.joints["boom"].native_unit == "m"
    assert machine.pick_z_m < -2.0 and machine.dump_z_m < -1.0  # pivot-frame heights


def test_decreasing_poly_inverse_is_direction_aware(tmp_path: Path):
    path = tmp_path / "machine.json"
    path.write_text(
        json.dumps(
            {
                "machine": "x",
                "calibrated": True,
                "policy_joint_order": ["bucket"],
                "joints": [
                    {
                        "name": "bucket",
                        "native_unit": "m",
                        "sign": 1,
                        "offset_rad": 0.0,
                        "hydraulic_conversion": {
                            "min_displacement_m": 0.0,
                            "max_displacement_m": 1.0,
                            "min_angle_deg": 100.0,
                            "max_angle_deg": 10.0,
                            # ascending powers: 100 - 90*dn
                            "poly_coeffs": [100.0, -90.0],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    jmap = MachineMap.load(path).joints["bucket"]
    assert math.degrees(jmap.to_policy(0.0)) == pytest.approx(100.0)
    assert math.degrees(jmap.to_policy(1.0)) == pytest.approx(10.0)
    assert jmap.to_native(math.radians(55.0)) == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------- live scan
_has_openplx = Path(OPENPLX_DEFAULT).exists()
skip_without_openplx = pytest.mark.skipif(
    not _has_openplx,
    reason="365_model.openplx (AGX export) not present on this machine",
)


@skip_without_openplx
def test_scan_matches_independently_measured_geometry():
    geo = load_openplx(OPENPLX_DEFAULT)
    report = geometry_report(geo)
    assert report["four_bar_m"]["crank_len"] == pytest.approx(1.0310, abs=2e-3)
    assert report["four_bar_m"]["coupler_len"] == pytest.approx(1.0297, abs=2e-3)
    assert report["four_bar_m"]["rocker_len"] == pytest.approx(0.6350, abs=2e-3)
    assert report["bucket_cyl_initial_m"] == pytest.approx(3.2015, abs=2e-3)

    scan = scan_machine(geo, n=201)
    assert scan["links_m"] == pytest.approx([7.800, 3.600, 2.273], abs=5e-3)
    assert scan["boom_pivot_world_m"][2] == pytest.approx(
        2.5076, abs=2e-3
    )  # above grade
    assert geometry_report(geo)["swing_axis"] == pytest.approx(
        [0.0, 0.0, 1.0], abs=1e-6
    )


@skip_without_openplx
def test_scan_windows_are_monotone_and_match_the_machine_file():
    geo = load_openplx(OPENPLX_DEFAULT)
    scan = scan_machine(geo, n=201)
    machine = MachineMap.load(_MACHINE)
    for name in ("boom", "arm", "bucket"):
        joint = scan["joints"][name]
        assert joint["mechanism"]["psi_residual_max"] < 5e-3
        lo, hi = joint["stroke_range_m"]
        window = stroke_window(joint, (lo, hi))
        assert window is not None, name
        assert -100.0 <= window[0] < window[1] <= 100.0
        grid = np.linspace(window[0], window[1], 51)
        disp = np.interp(grid, joint["scan"]["theta_deg"], joint["scan"]["disp_m"])
        assert np.all(np.diff(disp) > 0) or np.all(np.diff(disp) < 0), name
        # the committed machine file's endpoint angles must be this scan's numbers
        conv = machine.joints[name].conv
        for end_d, end_a in (
            (lo, conv["min_angle_deg"]),
            (hi, conv["max_angle_deg"]),
        ):
            theta = _interp(
                joint["scan"]["theta_deg"],
                joint["scan"]["disp_m"],
                end_d,
                joint["scan"]["disp_m"][-1] >= joint["scan"]["disp_m"][0],
            )
            assert theta is not None
            angle = float(
                np.interp(theta, joint["scan"]["theta_deg"], joint["scan"]["angle_deg"])
            )
            assert angle == pytest.approx(end_a, abs=0.05), f"{name} @ {end_d}"
