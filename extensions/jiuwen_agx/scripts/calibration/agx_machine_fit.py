# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AGX 挖掘机标定：扫缸采样 → 拟合 → machine_agx365.json（license 恢复后跑）。

记录方法（AGX 侧，需许可证；齿尖坐标与关节值在场景里读）：

1. swing 保持 0 附近，逐个液压缸在其行程内取 ≥5 个等分点；另外给 swing
   取 ≥5 个不同角（其余保持），用于回转零位；
2. 每个采样点记录四关节当前值（原生单位）与齿尖在**策略坐标系**的位置
   （x 前 / y 左 / z 上，米），写成 JSON：

       {"machine": "agx_excavator_365", "samples": [
           {"q": {"swing": 0.0, "boom": -0.2, "arm": 0.1, "bucket": -0.3},
            "tip_m": [3.1, 0.02, 0.4]}, ...]}

3. 拟合：median 残差 ≤ 0.05 m 才算标定成功（`--tol` 可调）；成功后写出
   calibrated=true 的机器文件，并打印建议给他们的 `links_m`（进他们 config 的
   geometry_signature——会强制他们重造数据重训）。

没有许可证时可用 `--selftest` 验证拟合器本身（合成数据往返恢复）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from jiuwen_agx.machine_fit import (
    fit_machine,
    load_samples,
    machine_file_from_fit,
    make_synthetic_samples,
)

__all__ = ["main"]

_CYLINDERS = ("boom", "arm", "bucket")


def _init_affine_from_template(
    template: dict,
) -> tuple[tuple[float, float], ...] | None:
    """The template's displacement<->angle endpoints are the natural fit init."""
    try:
        affine = []
        for name in _CYLINDERS:
            spec = next(j for j in template["joints"] if j["name"] == name)
            conv = spec["hydraulic_conversion"]
            lo_d = float(conv["min_displacement_m"])
            hi_d = float(conv["max_displacement_m"])
            lo_a = math.radians(float(conv["min_angle_deg"]))
            hi_a = math.radians(float(conv["max_angle_deg"]))
            slope = (hi_a - lo_a) / (hi_d - lo_d)
            affine.append((lo_a - slope * lo_d, slope))
        return tuple(affine)
    except (KeyError, StopIteration, ZeroDivisionError, TypeError):
        return None


def _parse_strokes(text: str | None) -> dict[str, tuple[float, float]] | None:
    if not text:
        return None
    bounds: dict[str, tuple[float, float]] = {}
    for part in text.split(";"):
        if not part.strip():
            continue
        name, _, span = part.partition("=")
        low, _, high = span.partition(":")
        bounds[name.strip()] = (float(low), float(high))
    return bounds


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--samples", help="calibration samples JSON (see module docstring)"
    )
    parser.add_argument(
        "--template",
        default=str(
            Path(__file__).resolve().parents[2]
            / "configs"
            / "machine_agx365.template.json"
        ),
        help="machine file template to fill",
    )
    parser.add_argument("--out", help="where to write the calibrated machine file")
    parser.add_argument(
        "--strokes",
        help="override stroke bounds, e.g. 'boom=-0.7:0.4;arm=-0.88:0.8;bucket=-1.0:0.312' "
        "(default: observed min/max per cylinder in the samples)",
    )
    parser.add_argument(
        "--tol", type=float, default=0.05, help="median residual tolerance, m"
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="fit synthetic samples and print recovery",
    )
    args = parser.parse_args(argv)

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))

    if args.selftest:
        ground_truth = {
            "links": (6.4, 2.9, 1.2),
            "affine": ((0.4, 0.9), (-1.1, 1.4), (0.2, -1.0)),
        }
        payload = make_synthetic_samples(ground_truth["links"], ground_truth["affine"])
        fit = fit_machine(payload, init_links=(6.4, 2.9, 1.2), tolerance_m=args.tol)
        print(
            "[fit] selftest ground truth :",
            ground_truth["links"],
            ground_truth["affine"],
        )
        print("[fit] recovered links_m     :", fit["links_m"])
        print("[fit] recovered affine      :", fit["cylinder_affine_rad_per_m"])
        print(
            f"[fit] residual median {fit['median_residual_m']} m / max {fit['max_residual_m']} m"
            f" | calibrated={fit['calibrated']}"
        )
        if not fit["calibrated"]:
            sys.exit("[fit] SELFTEST FAILED: solver did not reach tolerance")
        return

    if not args.samples or not args.out:
        parser.error("--samples and --out are required (or use --selftest)")
    payload = load_samples(args.samples)
    fit = fit_machine(
        payload,
        init_affine=_init_affine_from_template(template),
        tolerance_m=args.tol,
    )
    strokes = _parse_strokes(args.strokes)
    if strokes is None:
        observed: dict[str, list[float]] = {}
        for sample in payload["samples"]:
            for name in ("boom", "arm", "bucket"):
                observed.setdefault(name, []).append(float(sample["q"][name]))
        strokes = {name: (min(v), max(v)) for name, v in observed.items()}
        print("[fit] stroke bounds from samples:", strokes)
    machine = machine_file_from_fit(
        fit,
        template,
        stroke_bounds={k: (float(v[0]), float(v[1])) for k, v in strokes.items()},
    )
    out = Path(args.out)
    out.write_text(json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fit] wrote {out} (calibrated=true)")
    print(f"[fit] links_m for THEIR config: {fit['links_m']}")
    print(
        f"[fit] residual median {fit['median_residual_m']} m / max {fit['max_residual_m']} m"
    )
    print(
        "[fit] NOTE: 他们的 geometry_signature 会因新 links_m 拒绝旧数据——重造数据重训；"
        "我们侧重跑 lerobot_to_goalact_h5.py 生成正式训练集"
    )


if __name__ == "__main__":
    main()
