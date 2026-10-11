# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""标定拟合器的合成往返测试 — 无 AGX 依赖，恒可运行。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from jiuwen_agx.machine_fit import (
    fit_machine,
    machine_file_from_fit,
    make_synthetic_samples,
)

_LINKS = (6.4, 2.9, 1.2)
_AFFINE = ((0.4, 0.9), (-1.1, 1.4), (0.2, -1.0))


@pytest.fixture(scope="module")
def fitted() -> dict:
    payload = make_synthetic_samples(_LINKS, _AFFINE, noise_m=5e-4, seed=11)
    return fit_machine(payload, init_links=(5.0, 5.0, 5.0), tolerance_m=0.05)


def test_links_recovered(fitted: dict) -> None:
    for recovered, truth in zip(fitted["links_m"], _LINKS, strict=True):
        assert recovered == pytest.approx(truth, abs=0.02)


def test_cylinder_affine_recovered(fitted: dict) -> None:
    for name, (base, slope) in zip(("boom", "arm", "bucket"), _AFFINE, strict=True):
        got = fitted["cylinder_affine_rad_per_m"][name]
        assert got["base_rad"] == pytest.approx(base, abs=0.05)
        assert got["slope_rad_per_m"] == pytest.approx(slope, abs=0.05)


def test_swing_offset_and_tolerance(fitted: dict) -> None:
    assert fitted["swing_offset_rad"] == pytest.approx(0.37, abs=0.01)
    assert fitted["calibrated"] is True
    assert fitted["median_residual_m"] <= 0.05


def test_machine_file_fill(tmp_path: Path) -> None:
    template_path = (
        Path(__file__).resolve().parents[1] / "configs" / "machine_agx365.template.json"
    )
    template = json.loads(template_path.read_text(encoding="utf-8"))
    payload = make_synthetic_samples(_LINKS, _AFFINE, noise_m=5e-4, seed=11)
    fit = fit_machine(payload, tolerance_m=0.05)
    machine = machine_file_from_fit(
        fit,
        template,
        stroke_bounds={
            "boom": (-0.7, 0.4),
            "arm": (-0.88, 0.8),
            "bucket": (-1.0, 0.312),
        },
    )
    assert machine["calibrated"] is True
    assert machine["links_m"] == fit["links_m"]
    boom = next(j for j in machine["joints"] if j["name"] == "boom")
    conv = boom["hydraulic_conversion"]
    # endpoints follow the fitted affine
    affine = fit["cylinder_affine_rad_per_m"]["boom"]
    expected_low = math.degrees(affine["base_rad"] + affine["slope_rad_per_m"] * -0.7)
    expected_high = math.degrees(affine["base_rad"] + affine["slope_rad_per_m"] * 0.4)
    assert conv["min_angle_deg"] == pytest.approx(
        min(expected_low, expected_high), abs=0.01
    )
    assert conv["max_angle_deg"] == pytest.approx(
        max(expected_low, expected_high), abs=0.01
    )


def test_uncalibrated_fit_is_refused(tmp_path: Path) -> None:
    template = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "machine_agx365.template.json"
        ).read_text(encoding="utf-8")
    )
    payload = make_synthetic_samples(
        _LINKS, _AFFINE, noise_m=0.4, seed=3
    )  # heavy noise
    fit = fit_machine(payload, tolerance_m=0.01)
    assert fit["calibrated"] is False
    with pytest.raises(ValueError, match="exceeds tolerance"):
        machine_file_from_fit(
            fit,
            template,
            stroke_bounds={
                "boom": (-0.7, 0.4),
                "arm": (-0.88, 0.8),
                "bucket": (-1.0, 0.312),
            },
        )
