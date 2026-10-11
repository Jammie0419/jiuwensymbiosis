# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AGX 挖掘机标定拟合：把"扫缸采样"拟合成每机器标定文件（machine_map schema）。

背景：goal-act-v2 的模型世界是"四个关节角（度）"的平面三连杆模型，AGX 365
的真实关节是 1 个 Hinge（弧度）+ 3 个 Prismatic（米）。两者的桥 = 每个液压缸
的"位移 ↔ 等效角"映射，外加连杆长度定义"等效角"的几何含义。这些量在 AGX
里**测不出来直接的角**——只能扫缸、记录齿尖坐标，再用本模块反解：

    采样（每缸若干位移点 × 齿尖坐标）→ 最小二乘拟合
    → {links_m, 每缸 affine 映射, swing 零位偏移} → machine_agx365.json

模型（与他们 planar 约定一致：q=0 时三杆水平前伸，arm/bucket 为相对角）：

    a1 = boom, a2 = a1 + arm, a3 = a2 + bucket        （均自 affine 映射求得）
    r  = Σ L_i cos(a_i)    z = Σ L_i sin(a_i)         （策略系：x 前 / y 左 / z 上）

未知量：L1..L3 + 每缸 (b, k)（angle_rad = b + k·displacement_m）。用高斯-牛顿
（数值雅可比）最小化 Σ‖FK(θ) − (r,z)_observed‖²。

采样格式（JSON；记录方法与验收标准见 docs/cerebellum-interface-draft.md §8.4）：

    {"machine": "...", "samples": [
        {"q": {"swing": 0.0, "boom": -0.2, "arm": 0.1, "bucket": -0.3},
         "tip_m": [3.1, 0.02, 0.4]}, ...]}
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "fit_machine",
    "make_synthetic_samples",
    "planar_fk",
]

_CYLINDERS = ("boom", "arm", "bucket")


def planar_fk(links: np.ndarray, angles_rad: np.ndarray) -> tuple[float, float]:
    """Planar 3-link FK in their convention: tip (radius, z) from relative angles."""
    a1 = float(angles_rad[0])
    a2 = a1 + float(angles_rad[1])
    a3 = a2 + float(angles_rad[2])
    radius = links[0] * math.cos(a1) + links[1] * math.cos(a2) + links[2] * math.cos(a3)
    z = links[0] * math.sin(a1) + links[1] * math.sin(a2) + links[2] * math.sin(a3)
    return radius, z


def _predict(params: np.ndarray, sample: dict[str, Any]) -> tuple[float, float]:
    links = params[0:3]
    affine = params[3:9].reshape(3, 2)  # (b, k) per cylinder
    q = sample["q"]
    angles = np.array(
        [
            affine[i, 0] + affine[i, 1] * float(q[name])
            for i, name in enumerate(_CYLINDERS)
        ]
    )
    return planar_fk(links, angles)


def _residuals(params: np.ndarray, samples: list[dict[str, Any]]) -> np.ndarray:
    out = np.empty(2 * len(samples))
    for index, sample in enumerate(samples):
        radius, z = _predict(params, sample)
        tip = sample["tip_m"]
        out[2 * index] = radius - math.hypot(float(tip[0]), float(tip[1]))
        out[2 * index + 1] = z - float(tip[2])
    return out


_LINK_BOUNDS = (0.5, 15.0)
_BASE_BOUNDS = (-math.pi, math.pi)
_SLOPE_BOUNDS = (-5.0, 5.0)


def _clamp_physical(params: np.ndarray) -> np.ndarray:
    """Project onto the physically meaningful box.

    Without this the fit happily lands on gauge-equivalent branches — e.g. a
    sign-flipped link pair with the boom/arm angle shifted by pi reproduces the
    same tip positions exactly (residual ~0, parameters nonsense). Positive
    links + a bounded angle base pin the branch that matches their planar
    convention, which is the one the training config speaks.
    """
    out = params.copy()
    out[0:3] = np.clip(out[0:3], *_LINK_BOUNDS)
    for index in range(3):
        out[3 + 2 * index] = ((out[3 + 2 * index] + math.pi) % (2 * math.pi)) - math.pi
        out[4 + 2 * index] = np.clip(out[4 + 2 * index], *_SLOPE_BOUNDS)
    return out


def _gauss_newton(
    samples: list[dict[str, Any]], init: np.ndarray, *, iterations: int = 400
) -> np.ndarray:
    params = _clamp_physical(init)
    scale = np.array([1.0, 1.0, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0])  # conditioning
    damping = 1e-6
    for _ in range(iterations):
        residual = _residuals(params, samples)
        jacobian = np.empty((len(residual), len(params)))
        for column in range(len(params)):
            step = 1e-6 * max(1.0, abs(params[column]))
            bumped = params.copy()
            bumped[column] += step
            jacobian[:, column] = (_residuals(bumped, samples) - residual) / step
        jacobian *= scale
        lhs = jacobian.T @ jacobian + damping * np.eye(len(params))
        rhs = -jacobian.T @ residual
        delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0] * scale
        if not np.all(np.isfinite(delta)):
            break
        params = _clamp_physical(params + delta)
        if float(np.linalg.norm(delta)) < 1e-10:
            break
    return params


def make_synthetic_samples(
    links: tuple[float, float, float],
    affine: tuple[tuple[float, float], ...],
    *,
    per_joint: int = 6,
    noise_m: float = 1e-3,
    swing_offset_rad: float = 0.37,
    seed: int = 7,
) -> dict[str, Any]:
    """Synthetic sweep samples from known ground truth (self-test / demo)."""
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    strokes = {"boom": (-0.7, 0.4), "arm": (-0.88, 0.8), "bucket": (-1.0, 0.312)}
    for name in _CYLINDERS:
        low, high = strokes[name]
        base = {other: float(rng.uniform(*strokes[other])) for other in _CYLINDERS}
        for value in np.linspace(low, high, per_joint):
            q = dict(base)
            q[name] = float(value)
            angles = np.array(
                [
                    affine[i][0] + affine[i][1] * q[cylinder]
                    for i, cylinder in enumerate(_CYLINDERS)
                ]
            )
            radius, z = planar_fk(np.asarray(links), angles)
            swing = float(rng.uniform(-math.pi, math.pi))
            azimuth = swing + swing_offset_rad
            x = radius * math.cos(azimuth) + float(rng.normal(0.0, noise_m))
            y = radius * math.sin(azimuth) + float(rng.normal(0.0, noise_m))
            samples.append(
                {
                    "q": {"swing": swing, **q},
                    "tip_m": [x, y, z + float(rng.normal(0.0, noise_m))],
                }
            )
    return {
        "machine": "synthetic-selftest",
        "samples": samples,
        "ground_truth": {
            "links_m": list(links),
            "affine": [list(pair) for pair in affine],
            "swing_offset_rad": swing_offset_rad,
        },
    }


def fit_machine(
    payload: dict[str, Any],
    *,
    init_links: tuple[float, float, float] = (6.4, 2.9, 1.2),
    init_affine: tuple[tuple[float, float], ...] | None = None,
    tolerance_m: float = 0.05,
    multistart: int = 6,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit links + per-cylinder affine maps; returns machine-file fields + diagnostics.

    Gauss-Newton is local: from a far or unlucky init it can stall in a mirror/
    fold basin. ``multistart`` runs the solver from the caller's init (derive it
    from the machine template — the realistic case) plus sign-flipped and
    randomly perturbed variants, and keeps the lowest-residual result.
    """
    samples = [
        {
            "q": {k: float(v) for k, v in s["q"].items()},
            "tip_m": [float(x) for x in s["tip_m"]],
        }
        for s in payload["samples"]
    ]
    if len(samples) < 9:
        raise ValueError("need >= 9 samples (>= 3 per cylinder) to fit 9 parameters")
    for sample in samples:
        missing = [name for name in (*_CYLINDERS, "swing") if name not in sample["q"]]
        if missing:
            raise ValueError(f"sample missing joints {missing}: {sample['q'].keys()}")
    if init_affine is None:
        # The template's hydraulic_conversion block is the natural init; the
        # bare fallback is a guess that multistart compensates for.
        init_affine = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
    init = np.array(
        [*init_links, *[v for pair in init_affine for v in pair]], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    starts = [init]
    flipped = init.copy()
    flipped[3:9:2] *= -1.0
    flipped[4:9:2] *= -1.0
    starts.append(flipped)
    for _ in range(max(0, multistart - len(starts))):
        perturbed = init.copy()
        perturbed[0:3] *= rng.uniform(0.6, 1.4, size=3)
        perturbed[3:9:2] += rng.uniform(-0.8, 0.8, size=3)
        perturbed[4:9:2] *= rng.uniform(0.5, 1.5, size=3)
        starts.append(perturbed)
    best_error = math.inf
    params = init
    for start in starts:
        candidate = _gauss_newton(samples, start)
        error = float(np.median(np.abs(_residuals(candidate, samples))))
        if error < best_error:
            best_error, params = error, candidate
    residual = _residuals(params, samples)
    links = params[0:3]
    affine = params[3:9].reshape(3, 2)

    errors = np.abs(residual.reshape(-1, 2)).max(axis=1)
    median_error = float(np.median(errors))
    swing_offsets = [
        math.atan2(
            math.sin(math.atan2(s["tip_m"][1], s["tip_m"][0]) - s["q"]["swing"]),
            math.cos(math.atan2(s["tip_m"][1], s["tip_m"][0]) - s["q"]["swing"]),
        )
        for s in samples
    ]
    swing_offset = math.atan2(
        float(np.mean(np.sin(swing_offsets))), float(np.mean(np.cos(swing_offsets)))
    )
    return {
        "links_m": [round(float(v), 4) for v in links],
        "cylinder_affine_rad_per_m": {
            name: {
                "base_rad": round(float(affine[i, 0]), 6),
                "slope_rad_per_m": round(float(affine[i, 1]), 6),
            }
            for i, name in enumerate(_CYLINDERS)
        },
        "swing_offset_rad": round(swing_offset, 6),
        "median_residual_m": round(median_error, 4),
        "max_residual_m": round(float(errors.max()), 4),
        "calibrated": median_error <= tolerance_m,
        "tolerance_m": tolerance_m,
        "samples": len(samples),
    }


def machine_file_from_fit(
    fit: dict[str, Any],
    template: dict[str, Any],
    *,
    stroke_bounds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Fill a machine_map-shaped file from a fit result + recorded stroke bounds."""
    if not fit.get("calibrated"):
        raise ValueError(
            f"fit residual {fit['median_residual_m']} m exceeds tolerance "
            f"{fit['tolerance_m']} m — NOT writing a calibrated machine file"
        )
    machine = json.loads(json.dumps(template))  # deep copy
    machine["calibrated"] = True
    machine["links_m"] = fit["links_m"]
    diagram = {j["name"]: j for j in machine["joints"]}
    for name, low_high in stroke_bounds.items():
        spec = diagram[name]
        if "hydraulic_conversion" not in spec:
            raise ValueError(
                f"template joint {name!r} has no hydraulic_conversion block"
            )
        affine = fit["cylinder_affine_rad_per_m"][name]
        angles_deg = [
            math.degrees(affine["base_rad"] + affine["slope_rad_per_m"] * bound)
            for bound in (low_high[0], low_high[1])
        ]
        spec["hydraulic_conversion"].update(
            {
                "min_displacement_m": float(low_high[0]),
                "max_displacement_m": float(low_high[1]),
                "min_angle_deg": round(min(angles_deg), 3),
                "max_angle_deg": round(max(angles_deg), 3),
                "poly_coeffs": None,
            }
        )
    swing = diagram.get("swing")
    if swing is not None:
        swing["offset_rad"] = fit["swing_offset_rad"]
    return machine


def load_samples(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
