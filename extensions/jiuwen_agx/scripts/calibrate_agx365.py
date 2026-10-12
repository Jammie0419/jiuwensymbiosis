# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AGX 365 标定 CLI：OpenPLX 纯几何扫描 → 标定表 + machine_agx365.json（真尺）。

不需要 AGX 许可证：几何全部来自 ``data/models/365_model.openplx``（导出文本），
符号由已采集的 112 条数据回放选定，自检全过才写 calibrated=true 的机器文件
（fail-closed；任何一项不过就停下并报告，绝不强行放行）。

    python scripts/calibrate_agx365.py                  # 只用几何：画标定表 + 报告
    python scripts/calibrate_agx365.py --data data/raw/run_all
        # 再加数据回放：选符号、推算 pick_z/dump_z、写 machine file

产物：
    configs/calibration/agx365_cylinders.json   扫描原表（含 disp_to_theta）
    configs/machine_agx365.json                 真尺（--data 且自检全过时）
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from jiuwen_agx.cylinder_calibration import (
    AGX365_LAYOUT,
    OPENPLX_DEFAULT,
    build_machine_file,
    fit_poly_table,
    geometry_report,
    load_openplx,
    replay_episode_tip,
    scan_machine,
    select_signs,
    stroke_window,
)

__all__ = ["main"]

_EXT_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_LINKS = (7.800, 3.600, 2.273)
_EXPECTED_LINKS_TOL_M = 0.005


def _load_episodes(data_dir: Path):
    episodes = []
    for path in sorted(glob.glob(str(data_dir / "ep_*_teacher.npz"))):
        d = np.load(path, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        episodes.append(
            (
                d["joints_after"],
                d["scoop"].astype(bool),
                meta.get("goal", {}),
            )
        )
    return episodes


def _tool_heights(joint_scans, sign_map, episodes, links, pivot_z):
    """Tip z (pivot frame) at the scoop load / unload moments, per episode."""
    load_z, unload_z = [], []
    for joints_after, scoop, _goal in episodes:
        rep = replay_episode_tip(joint_scans, sign_map, joints_after, links, pivot_z)
        z = rep["z_m"]
        starts = np.flatnonzero(scoop[1:] & ~scoop[:-1])
        ends = np.flatnonzero(~scoop[1:] & scoop[:-1])
        if len(starts):
            v = z[int(starts[0])]
            if v is not None:
                load_z.append(v)
        if len(ends):
            v = z[int(ends[0])]
            if v is not None:
                unload_z.append(v)
    return load_z, unload_z


def _monotone(values) -> bool:
    return all(values[i + 1] >= values[i] for i in range(len(values) - 1)) or all(
        values[i + 1] <= values[i] for i in range(len(values) - 1)
    )


def _keyframe_compare(scan_result, sign_map, template_path: Path) -> list[str]:
    """Old placeholder vs the true ruler, on the teacher's four dig keyframes."""
    lines = []
    try:
        old = json.loads(template_path.read_text(encoding="utf-8"))
        old_specs = {j["name"]: j for j in old["joints"]}
    except (OSError, ValueError):
        old_specs = None
    keyframes = {
        "ready": {"boom": 0.10, "arm": -0.82, "bucket": -0.65},
        "dig": {"boom": -0.25, "arm": 0.30, "bucket": -0.70},
        "curl": {"boom": 0.02, "arm": 0.70, "bucket": 0.10},
        "dump": {"boom": 0.38, "arm": -0.40, "bucket": -0.80},
    }
    for name in ("boom", "arm", "bucket"):
        sc = scan_result["joints"][name]["scan"]

        def angle_at(native, sc=sc):
            inc = sc["disp_m"][-1] >= sc["disp_m"][0]
            ts = sc["theta_deg"] if inc else sc["theta_deg"][::-1]
            vs = sc["disp_m"] if inc else sc["disp_m"][::-1]
            theta = float(np.interp(native, vs, ts))
            return float(np.interp(theta, sc["theta_deg"], sc["angle_deg"]))

        def old_angle(native, name=name, old_specs=old_specs):
            if old_specs is None:
                return None
            conv = old_specs[name]["hydraulic_conversion"]
            lo_d, hi_d = conv["min_displacement_m"], conv["max_displacement_m"]
            lo_a, hi_a = conv["min_angle_deg"], conv["max_angle_deg"]
            return lo_a + (native - lo_d) / (hi_d - lo_d) * (hi_a - lo_a)

        parts = []
        for label, value in keyframes.items():
            new = angle_at(sign_map[name] * value[name])
            old_v = old_angle(value[name])
            parts.append(
                f"{label} {value[name]:+.2f} -> {new:+.1f} deg"
                + (f" (old {old_v:+.1f})" if old_v is not None else "")
            )
        lines.append(f"  {name:6s}: " + " | ".join(parts))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--openplx", default=OPENPLX_DEFAULT)
    parser.add_argument(
        "--data",
        default=None,
        help="teacher npz directory for the sign/replay checks "
        "(e.g. data/raw/run_all); without it only the geometric scan runs",
    )
    parser.add_argument(
        "--out",
        default=str(_EXT_ROOT / "configs" / "calibration" / "agx365_cylinders.json"),
    )
    parser.add_argument(
        "--machine-out",
        default=str(_EXT_ROOT / "configs" / "machine_agx365.json"),
    )
    parser.add_argument(
        "--template",
        default=str(_EXT_ROOT / "configs" / "machine_agx365.template.json"),
    )
    parser.add_argument("--samples", type=int, default=401)
    args = parser.parse_args(argv)

    print(f"[calib] parsing {args.openplx}")
    geo = load_openplx(args.openplx)
    print(f"[calib] bodies={len(geo['bodies'])} constraints={len(geo['constraints'])}")
    report = geometry_report(geo)
    print(
        "[calib] geometry report:\n" + json.dumps(report, indent=1, ensure_ascii=False)
    )

    scan = scan_machine(geo, AGX365_LAYOUT, n=args.samples)
    links = scan["links_m"]
    pivot_z = scan["joints"]["boom"]["mechanism"]["pivot"][2]

    failures: list[str] = []
    for name, want in zip(("boom", "arm", "bucket"), _EXPECTED_LINKS, strict=True):
        got = links[("boom", "arm", "bucket").index(name)]
        if abs(got - want) > _EXPECTED_LINKS_TOL_M:
            failures.append(
                f"link {name} measured {got:.4f} m, expected {want:.4f} (tol {_EXPECTED_LINKS_TOL_M})"
            )

    for name in ("boom", "arm", "bucket"):
        joint = scan["joints"][name]
        res = joint["mechanism"]["psi_residual_max"]
        if res > 5e-3:
            failures.append(
                f"{name}: barrel/pin residual {res:.2e} > 5e-3 (mechanism model broken)"
            )
        if not joint["mechanism"]:
            failures.append(f"{name}: no mechanism info")

    print("\n[calib] scan summary (our convention; sign chosen below):")
    for name in ("boom", "arm", "bucket"):
        joint = scan["joints"][name]
        sc = joint["scan"]
        lo, hi = joint["stroke_range_m"]
        print(
            f"  {name:<6s} stroke [{lo:+.3f}, {hi:+.3f}] "
            f"(enabled={joint['stroke_range_enabled']})  "
            f"disp over scan [{min(sc['disp_m']):+.3f}, {max(sc['disp_m']):+.3f}]  "
            f"angle over scan [{min(sc['angle_deg']):+.1f}, {max(sc['angle_deg']):+.1f}] deg  "
            f"monotonic_disp={joint['monotonic_disp']}"
        )

    sign_map = {"boom": 1.0, "arm": 1.0, "bucket": 1.0}
    sign_info = None
    replay_stats = None
    pick_z = dump_z = None
    if args.data:
        data_dir = Path(args.data)
        episodes = _load_episodes(data_dir)
        if not episodes:
            print(f"[calib] ERROR: no ep_*_teacher.npz under {data_dir}")
            return 2
        print(
            f"\n[calib] replaying {len(episodes)} recorded episodes for the sign check"
        )
        sign_map, sign_info = select_signs(
            scan["joints"],
            [(ja, sc) for ja, sc, _ in episodes],
            links,
            pivot_z,
            verbose=True,
        )
        print(f"[calib] picked signs {sign_map} (margin {sign_info['margin']})")
        if sign_info["margin"] <= 0:
            failures.append("sign selection has no margin — replay cannot discriminate")
        best_frac = float(sign_info["scores"][sign_info["best"]])
        if best_frac > 0.15:
            failures.append(
                f"best sign combo still violates physics on {best_frac:.1%} of frames"
            )
        replay_stats = {"violation_fraction": best_frac, "sign_scores": sign_info}

        load_z, unload_z = _tool_heights(
            scan["joints"], sign_map, episodes, links, pivot_z
        )
        if not load_z or not unload_z:
            failures.append("replay produced no scoop load/unload moments")
        else:
            pick_z = float(np.median(load_z))
            dump_z = float(np.median(unload_z))
            replay_stats["load_z_world_m"] = {
                "n": len(load_z),
                "median": round(pick_z, 4),
                "p10": round(float(np.percentile(load_z, 10)), 4),
                "p90": round(float(np.percentile(load_z, 90)), 4),
            }
            replay_stats["unload_z_world_m"] = {
                "n": len(unload_z),
                "median": round(dump_z, 4),
                "p10": round(float(np.percentile(unload_z, 10)), 4),
                "p90": round(float(np.percentile(unload_z, 90)), 4),
            }
            # the machine file speaks the POLICY frame (the cerebellum FK's origin
            # is the boom pivot) — subtract the pivot height above grade.
            pick_z -= pivot_z
            dump_z -= pivot_z
            replay_stats["pick_z_policy_m"] = round(pick_z, 4)
            replay_stats["dump_z_policy_m"] = round(dump_z, 4)
            print(
                f"[calib] tip z at load med={pick_z + pivot_z:.2f} m "
                f"[{np.percentile(load_z, 10):.2f}, {np.percentile(load_z, 90):.2f}] | "
                f"unload med={dump_z + pivot_z:.2f} m "
                f"[{np.percentile(unload_z, 10):.2f}, {np.percentile(unload_z, 90):.2f}] "
                f"(world) -> policy pick_z={pick_z:.2f} dump_z={dump_z:.2f}"
            )

    # ---- per-joint tables, windows, monotonicity and fit quality
    calibration = {
        "machine": scan["machine"],
        "source": args.openplx,
        "note": "theta = rotation of the child about the joint axis from the assembly pose; "
        "disp = AGX Prismatic reading our convention; angle = policy joint angle "
        "(a1 elevation above horizontal, a2/a3 relative, up-positive)",
        "links_m": [round(v, 5) for v in links],
        "boom_pivot_world_m": [round(v, 5) for v in scan["boom_pivot_world_m"]],
        "geometry_report": report,
        "sign_map": sign_map,
        "sign_selection": sign_info,
        "replay": replay_stats,
        "joints": {},
    }
    for name in ("boom", "arm", "bucket"):
        joint = scan["joints"][name]
        sc = joint["scan"]
        lo, hi = joint["stroke_range_m"]
        sign = sign_map[name]
        window = stroke_window(joint, (sign * lo, sign * hi))
        if window is None:
            failures.append(f"{name}: stroke range not reachable in the scan window")
            continue
        thetas = sc["theta_deg"]
        disp = sc["disp_m"]
        angles = sc["angle_deg"]
        grid = np.linspace(window[0], window[1], 201)
        d_win = np.interp(grid, thetas, disp)
        a_win = np.interp(grid, thetas, angles)
        if not _monotone(list(d_win)) or not _monotone(list(a_win)):
            failures.append(f"{name}: window table is not monotone")
        d_native = (sign * d_win).tolist()
        order = np.argsort(d_native)
        d_sorted = [d_native[i] for i in order]
        a_sorted = [a_win[i] for i in order]
        lo_a = float(np.interp(lo, d_sorted, a_sorted))
        hi_a = float(np.interp(hi, d_sorted, a_sorted))
        linear_err = max(
            abs(np.interp(d, [lo, hi], [lo_a, hi_a]) - a)
            for d, a in zip(d_sorted, a_sorted, strict=True)
        )
        fit = fit_poly_table(d_sorted, a_sorted, degree=5)
        samples = []
        for k in range(11):
            dq = lo + (hi - lo) * k / 10
            samples.append(
                {
                    "disp_m": round(dq, 5),
                    "angle_deg": round(float(np.interp(dq, d_sorted, a_sorted)), 3),
                }
            )
        calibration["joints"][name] = {
            "disp_range_m": [round(lo, 4), round(hi, 4)],
            "window_theta_deg": [round(window[0], 4), round(window[1], 4)],
            "cyl_zero_length_m": round(joint["mechanism"]["pin_to_barrel_m"], 5),
            "scan": {
                "theta_deg": [round(t, 3) for t in thetas],
                "disp_m": [round(x, 5) for x in disp],
                "angle_deg": [round(x, 3) for x in angles],
            },
            "disp_to_theta": samples,
            "angle_at_stroke_ends_deg": [round(lo_a, 4), round(hi_a, 4)],
            "linear_max_err_deg": round(linear_err, 4),
            "poly5_max_err_deg": round(fit["max_err_deg"], 4),
            "poly5_monotone": fit["monotone"],
            "poly5_increasing": fit["increasing"],
            "mechanism": {
                k: v
                for k, v in joint["mechanism"].items()
                if k not in ("pivot", "e_pin", "b0")
            },
        }
        print(
            f"  {name:<6s} window theta [{window[0]:+.1f}, {window[1]:+.1f}] deg | "
            f"angles at stroke ends [{lo_a:+.2f}, {hi_a:+.2f}] deg | "
            f"linear err {linear_err:.3f} deg vs poly5 err {fit['max_err_deg']:.3f} deg "
            f"(monotone={fit['monotone']}, inc={fit['increasing']})"
        )

    print("\n[calib] old placeholder vs true ruler on the teacher keyframes:")
    for line in _keyframe_compare(scan, sign_map, Path(args.template)):
        print(line)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(calibration, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[calib] wrote {out_path}")

    if failures:
        print("\n[calib] FAIL-CLOSED — machine file NOT written. Failures:")
        for f in failures:
            print(f"  - {f}")
        return 1

    if args.data:
        template = json.loads(Path(args.template).read_text(encoding="utf-8"))
        machine = build_machine_file(
            scan, sign_map, template, pick_z_m=pick_z, dump_z_m=dump_z
        )
        machine_path = Path(args.machine_out)
        machine_path.write_text(
            json.dumps(machine, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[calib] wrote calibrated machine file {machine_path}")
        for name, details in machine["_calibration_details"].items():
            print(f"  {name}: {json.dumps(details, ensure_ascii=False)}")
    else:
        print(
            "[calib] geometry-only run: pass --data data/raw/run_all to emit the machine file"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
