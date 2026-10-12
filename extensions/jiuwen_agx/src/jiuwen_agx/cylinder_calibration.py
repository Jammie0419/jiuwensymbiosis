# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AGX 挖掘机液压缸-关节标定：OpenPLX 纯几何扫描（不需要 AGX 许可证）。

给出一台挖掘机（当前：AGX 365）的「液压缸位移 ↔ 策略关节角」换算表：
解析 ``.openplx`` 导出文本里的刚体/铰点/液压缸安装点，绕每个关节轴扫描
姿态，算出每个缸的 Prismatic 读数对应的关节角。三个关节各自的算法：

- **boom / arm**：缸的活塞杆直接铰接在两个连杆上，绕关节轴扫描子刚体，
  解出缸筒的随动转角 ψ，再按轴向投影得到位移；
- **bucket**：铲斗是四连杆（杆端接 EndToBucketConnector，经 RigidBody15
  曲柄回到斗杆），先用圆-圆交点闭合回路求杆销位置，再走与上面相同的
  ψ/位移流程。

与 ``machine_fit.py``（数据驱动拟合）互补：本模块只需要模型文件。

关键约定（与 ``configs/machine_agx365.template.json`` 的字段一一对应）：
- 关节角 a1/a2/a3 与小脑组 ``kinematics.py`` 的 FK 一致：a1 = 动臂相对
  水平面的仰角，a2/a3 为相对上一节的转角，上抬为正；
- 位移 0 = 模型装配姿态；符号以 AGX 原生读数为准，由采集数据回放选定
  （``select_signs``：``AGX_reading = sign * q_ours``）；
- 单位换算只此一处，训练与部署共用（五红线之一）。
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "AGX365_LAYOUT",
    "OPENPLX_DEFAULT",
    "load_openplx",
    "geometry_report",
    "scan_machine",
    "stroke_window",
    "select_signs",
    "replay_episode_tip",
    "fit_poly_table",
    "build_machine_file",
]

OPENPLX_DEFAULT = "E:/AGX-2.42.2.1/data/models/365_model.openplx"

# ---------------------------------------------------------------------------- layout
# 365 的机构布局（名字全部取自 365_model.openplx；换机型=换这份表 + 重新标定）。
AGX365_LAYOUT: dict[str, Any] = {
    "machine": "agx_excavator_365",
    "system_layout": {
        # up / forward 从这里推导：up = CabinHinge 轴（取 +z 侧），
        # forward = 装配姿态下动臂连杆方向的水平投影。
        "up_mc": ("ChassieBody", "CabinHinge_mc"),
        "boom_link": ("Boom", "BoomHinge_mc", "Boom", "StickHinge_mc"),
    },
    "edge_begin": ("Bucket", "CuttingEdgeBegin"),
    "edge_end": ("Bucket", "CuttingEdgeEnd"),
    "joints": {
        "boom": {
            "parent": "ChassieBody",
            "pivot": ("ChassieBody", "BoomHinge_mc"),
            "child_points": {
                "rod_pin": ("StartCylinder1", "Hinge23_mc"),
                "link_a": ("Boom", "BoomHinge_mc"),
                "link_b": ("Boom", "StickHinge_mc"),
            },
            "barrel": ("StartCylContainer1", "Hinge1_mc", "ArmPrismatic1_mc"),
            "rod_mc": ("StartCylinder1", "ArmPrismatic1_mc"),
            "stroke_range": "ArmPrismatic1_range",
            "angle_mode": "absolute",
        },
        "arm": {
            "parent": "Boom",
            "pivot": ("Boom", "StickHinge_mc"),
            "child_points": {
                "rod_pin": ("CylinderMid", "Hinge8_mc"),
                "link_a": ("Stick", "StickHinge_mc"),
                "link_b": ("Stick", "BucketHinge_mc"),
            },
            "barrel": ("CylContainerMid", "Hinge6_mc", "StickPrismatic_mc"),
            "rod_mc": ("CylinderMid", "StickPrismatic_mc"),
            "stroke_range": "StickPrismatic_range",
            "angle_mode": "relative",
        },
        "bucket": {
            "parent": "Stick",
            "pivot": ("Stick", "BucketHinge_mc"),
            "child_points": {
                "link_a": ("Bucket", "BucketHinge_mc"),
                "edge_begin": ("Bucket", "CuttingEdgeBegin"),
                "edge_end": ("Bucket", "CuttingEdgeEnd"),
                "hinge32": ("Bucket", "Hinge32_mc"),
            },
            "barrel": ("CylContainerEnd", "Hinge9_mc", "BucketPrismatic_mc"),
            "rod_mc": ("CylinderEnd", "BucketPrismatic_mc"),
            "stroke_range": "BucketPrismatic_range",
            "angle_mode": "relative",
            "four_bar": {
                "fixed_a": ("Stick", "Hinge34_mc"),
                "fixed_rod_mc": ("EndToBucketConnector", "Hinge2_mc"),
                "child_c": "hinge32",
            },
        },
    },
}


# ---------------------------------------------------------------------------- parsing
def _build_blocks(lines: list[str]) -> list[dict[str, Any]]:
    out = []
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)(?:const\s+)?(`[^`]+`|\S+) is ([A-Za-z0-9_.]+):\s*$", line)
        if m:
            out.append(
                {
                    "name": m.group(2).strip("`"),
                    "cls": m.group(3),
                    "indent": len(m.group(1)),
                    "i": i,
                }
            )
    return out


def _fields_of(lines: list[str], block: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    stack: list[tuple[int, str]] = []
    j = block["i"] + 1
    while j < len(lines):
        line = lines[j]
        if line.strip():
            indent = len(line) - len(line.lstrip())
            if indent <= block["indent"]:
                break
            while stack and stack[-1][0] >= indent:
                stack.pop()
            m = re.match(r"^\s*([A-Za-z0-9_.]+):\s+(.*?)\s*$", line)
            if m:
                key = ".".join([s[1] for s in stack] + [m.group(1)])
                out[key] = m.group(2)
            else:
                nm = re.match(r"^\s*([A-Za-z0-9_.]+):\s*$", line)
                if nm:
                    stack.append((indent, nm.group(1)))
        j += 1
    return out


def _f(value: str | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def load_openplx(path: str | Path) -> dict[str, Any]:
    """Parse bodies (pose + mate connectors with axes), constraints and ranges."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    blocks = _build_blocks(lines)

    body_blocks = [b for b in blocks if b["cls"] == "Physics3D.Bodies.RigidBody"]
    bodies: dict[str, Any] = {}
    for k, block in enumerate(body_blocks):
        fl = _fields_of(lines, block)
        end = body_blocks[k + 1]["i"] if k + 1 < len(body_blocks) else len(lines)
        mcs: dict[str, Any] = {}
        for other in blocks:
            if other["i"] <= block["i"] or other["i"] >= end:
                continue
            if other["cls"] != "Physics3D.Interactions.MateConnector":
                continue
            mf = _fields_of(lines, other)
            mcs[other["name"]] = {
                "pos": [_f(mf.get(f"position.{a}")) for a in "xyz"],
                "axis": [_f(mf.get(f"main_axis.{a}")) for a in "xyz"],
            }
        bodies[block["name"]] = {
            "pos": [
                _f(fl.get(f"kinematics.local_transform.position.{a}")) for a in "xyz"
            ],
            "quat": [
                _f(fl.get(f"kinematics.local_transform.rotation.{a}")) for a in "xyzw"
            ],
            "mc": mcs,
        }

    constraints: dict[str, Any] = {}
    ranges: dict[str, Any] = {}
    for block in blocks:
        cls = block["cls"]
        if cls in ("Physics3D.Interactions.Hinge", "Physics3D.Interactions.Prismatic"):
            fl = _fields_of(lines, block)
            raw = fl.get("connectors", "").replace("`", "")
            connectors = []
            for part in raw.strip("[]").split(","):
                part = part.strip()
                if part:
                    body_name, _, mc_name = part.rpartition(".")
                    connectors.append((body_name, mc_name))
            constraints[block["name"]] = {"cls": cls, "connectors": connectors}
        elif cls in (
            "Physics3D.Interactions.LinearRange",
            "Physics3D.Interactions.RotationalRange",
        ):
            fl = _fields_of(lines, block)
            start = _f(fl.get("start"))
            end = _f(fl.get("end"))
            ranges[block["name"]] = {
                "enabled": fl.get("enabled", "").strip().lower() == "true",
                "start": start if start is not None else -math.inf,
                "end": end if end is not None else math.inf,
            }
    return {"bodies": bodies, "constraints": constraints, "ranges": ranges}


# ---------------------------------------------------------------------------- vectors
def _sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _add(a, b):
    return [a[i] + b[i] for i in range(3)]


def _mul(a, s):
    return [x * s for x in a]


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(v):
    return math.sqrt(_dot(v, v))


def _unit(v):
    n = _norm(v)
    if n < 1e-12:
        raise ValueError(f"degenerate vector {v}")
    return [x / n for x in v]


def _rot(v, axis, ang):
    """Rodrigues rotation of vector ``v`` about unit ``axis``."""
    c, s = math.cos(ang), math.sin(ang)
    return _add(
        _add(_mul(v, c), _mul(_cross(axis, v), s)), _mul(axis, _dot(axis, v) * (1 - c))
    )


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _quat_rotate(quat: list[float], v: list[float]) -> list[float]:
    x, y, z, w = (float(q) for q in quat)
    m = [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ]
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


class _World:
    """World-frame accessors over a parsed openplx geometry."""

    def __init__(self, geo: dict[str, Any]) -> None:
        self.geo = geo

    def point(self, body: str, mc: str) -> list[float]:
        b = self.geo["bodies"][body]
        if b["pos"][0] is None or b["quat"][0] is None:
            raise ValueError(f"body {body} lacks kinematics")
        return _add(_quat_rotate(b["quat"], b["mc"][mc]["pos"]), b["pos"])

    def direction(self, body: str, mc: str) -> list[float]:
        b = self.geo["bodies"][body]
        return _unit(_quat_rotate(b["quat"], b["mc"][mc]["axis"]))

    @staticmethod
    def rotate_point(p, pivot, axis, ang):
        return _add(pivot, _rot(_sub(p, pivot), axis, ang))


# ---------------------------------------------------------------------------- scan core
def _barrel_residual(e_pin, a_bar, m0, e_dir, b_point, psi):
    direction = _rot(e_dir, a_bar, psi)
    anchor = _add(e_pin, _rot(_sub(m0, e_pin), a_bar, psi))
    return _norm(_cross(direction, _sub(b_point, anchor)))


def _solve_barrel_psi(e_pin, a_bar, m0, e_dir, b_point, prev_psi, span=0.8, step=0.002):
    """Continuity-tracked 1-D solve of the barrel swivel angle.

    The barrel rotates about its own pin; on the valid pose the rod pin sits on
    the cylinder axis line, i.e. the residual vanishes. Coarse scan + ternary
    refinement, always started around the previous pose to stay on the branch.
    """
    best_psi, best_val = prev_psi, float("inf")
    n = int(span / step)
    for k in range(-n, n + 1):
        psi = prev_psi + k * step
        val = _barrel_residual(e_pin, a_bar, m0, e_dir, b_point, psi)
        if val < best_val:
            best_val, best_psi = val, psi
    lo, hi = best_psi - step, best_psi + step
    for _ in range(50):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if _barrel_residual(e_pin, a_bar, m0, e_dir, b_point, m1) < _barrel_residual(
            e_pin, a_bar, m0, e_dir, b_point, m2
        ):
            hi = m2
        else:
            lo = m1
    psi = 0.5 * (lo + hi)
    return psi, _barrel_residual(e_pin, a_bar, m0, e_dir, b_point, psi)


def _circle_intersect(p1, p2, r1, r2, normal, prev):
    """Two intersections of circles (p1,r1)-(p2,r2) in the plane ⊥ normal."""
    n = _unit(normal)
    ab = _sub(p2, p1)
    d = _norm(ab)
    if d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9 or d < 1e-12:
        raise ValueError(
            f"circles do not intersect (d={d:.4f}, r1={r1:.4f}, r2={r2:.4f})"
        )
    e1 = _unit(ab)
    e2 = _cross(n, e1)
    x = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)
    h = math.sqrt(max(0.0, r1 * r1 - x * x))
    base = _add(p1, _mul(e1, x))
    a = _add(base, _mul(e2, h))
    b = _sub(base, _mul(e2, h))
    if prev is None:
        return a, b
    return (a, b) if _norm(_sub(a, prev)) <= _norm(_sub(b, prev)) else (b, a)


def _scan_joint(
    geo: dict[str, Any],
    spec: dict[str, Any],
    *,
    theta_lo_deg: float,
    theta_hi_deg: float,
    n: int,
):
    """Sweep one joint's child rotation; return θ keyed states + mechanism info."""
    w = _World(geo)
    pivot = w.point(*spec["pivot"])
    axis = w.direction(*spec["pivot"])

    barrel_body, barrel_pin_mc, barrel_prism_mc = spec["barrel"]
    e_pin = w.point(barrel_body, barrel_pin_mc)
    a_bar = w.direction(barrel_body, barrel_pin_mc)
    m0 = w.point(barrel_body, barrel_prism_mc)
    e_dir = w.direction(barrel_body, barrel_prism_mc)
    if _dot(e_dir, _sub(m0, e_pin)) < 0:
        e_dir = _mul(e_dir, -1.0)
    rod_body, rod_prism_mc = spec["rod_mc"]
    m0_rod = w.point(rod_body, rod_prism_mc)
    if _norm(_sub(m0_rod, m0)) > 1e-3:
        raise ValueError(
            f"prismatic anchors do not coincide: "
            f"{barrel_body}.{barrel_prism_mc} vs {rod_body}.{rod_prism_mc} "
            f"({_norm(_sub(m0_rod, m0)):.4f} m apart)"
        )

    pts = {key: w.point(*ref) for key, ref in spec["child_points"].items()}
    four_bar = spec.get("four_bar")
    if four_bar:
        a_fixed = w.point(*four_bar["fixed_a"])
        b0 = w.point(*four_bar["fixed_rod_mc"])
        c0 = pts[four_bar["child_c"]]
        ra = _norm(_sub(b0, a_fixed))
        rc = _norm(_sub(b0, c0))
    else:
        a_fixed = c0 = ra = rc = None
        b0 = pts["rod_pin"]

    sigma_r = -1.0 if _dot(_sub(m0, b0), e_dir) < 0 else 1.0
    rod_mc_offset = _norm(_sub(m0, b0))

    step = (theta_hi_deg - theta_lo_deg) / (n - 1)
    thetas = np.arange(theta_lo_deg, theta_hi_deg + step / 2, step)
    state: dict[float, Any] = {}
    res_max = 0.0
    # Two legs, each continuing from the assembly pose θ=0 — a single pass that
    # crosses θ=0 would drag the barrel-swivel prediction far from its branch.
    legs = [
        [float(t) for t in thetas if t >= 0.0],
        [float(t) for t in thetas if t < 0.0][::-1],
    ]
    for leg in legs:
        psi, prev_b = 0.0, b0
        for theta_deg in leg:
            theta = math.radians(theta_deg)
            rotated = {
                key: _World.rotate_point(p, pivot, axis, theta)
                for key, p in pts.items()
            }
            if four_bar:
                b, _ = _circle_intersect(
                    a_fixed, rotated[four_bar["child_c"]], ra, rc, axis, prev_b
                )
            else:
                b = rotated["rod_pin"]
            psi, res = _solve_barrel_psi(e_pin, a_bar, m0, e_dir, b, psi)
            res_max = max(res_max, res)
            e_psi = _rot(e_dir, a_bar, psi)
            m_psi = _add(e_pin, _rot(_sub(m0, e_pin), a_bar, psi))
            mr = _add(b, _mul(e_psi, sigma_r * rod_mc_offset))
            q = _dot(_sub(mr, m_psi), e_psi)
            prev_b = b
            state[round(theta_deg, 6)] = (q, psi, res, rotated)

    info = {
        "pivot": pivot,
        "e_pin": e_pin,
        "b0": b0,
        "sigma_r": sigma_r,
        "rod_mc_offset": rod_mc_offset,
        "pin_to_barrel_m": _norm(_sub(m0, e_pin)),
        "pin_to_rod_m": _norm(_sub(b0, e_pin)),
        "four_bar": (
            {
                "ra": ra,
                "rc": rc,
                "rod_len": _norm(_sub(b0, a_fixed)),
                "crank_len": _norm(_sub(c0, pts["link_a"])),
            }
            if four_bar
            else None
        ),
        "psi_residual_max": res_max,
    }
    return state, info


def scan_machine(
    geo: dict[str, Any],
    layout: dict[str, Any] = AGX365_LAYOUT,
    *,
    theta_lo_deg: float = -100.0,
    theta_hi_deg: float = 100.0,
    n: int = 401,
):
    """Full three-joint scan + policy-angle curves + mechanism report."""
    w = _World(geo)

    up = w.direction(*layout["system_layout"]["up_mc"])
    if _dot(up, [0.0, 0.0, 1.0]) < 0:
        up = _mul(up, -1.0)
    boom = layout["system_layout"]["boom_link"]
    v_boom0 = _sub(w.point(boom[2], boom[3]), w.point(boom[0], boom[1]))
    f = _unit(_sub(v_boom0, _mul(up, _dot(v_boom0, up))))

    def elevation(v):
        return math.atan2(_dot(v, up), _dot(v, f))

    def link_dir(body_a, mc_a, body_b, mc_b):
        return _sub(w.point(body_b, mc_b), w.point(body_a, mc_a))

    phi_arm0 = elevation(link_dir("Stick", "StickHinge_mc", "Stick", "BucketHinge_mc"))
    edge_begin = w.point(*layout["edge_begin"])
    edge_end = w.point(*layout["edge_end"])
    edge_mid = _mul(_add(edge_begin, edge_end), 0.5)
    links_m = [
        _norm(link_dir("Boom", "BoomHinge_mc", "Boom", "StickHinge_mc")),
        _norm(link_dir("Stick", "StickHinge_mc", "Stick", "BucketHinge_mc")),
        _norm(_sub(edge_mid, w.point("Bucket", "BucketHinge_mc"))),
    ]

    result: dict[str, Any] = {
        "machine": layout["machine"],
        "links_m": links_m,
        "boom_pivot_world_m": w.point(*boom[:2]),
        "joints": {},
    }

    for name in ("boom", "arm", "bucket"):
        spec = layout["joints"][name]
        state, info = _scan_joint(
            geo, spec, theta_lo_deg=theta_lo_deg, theta_hi_deg=theta_hi_deg, n=n
        )
        thetas = sorted(state)
        disp = [state[t][0] for t in thetas]

        def angle_at(rotated, name=name):
            if name == "boom":
                return math.degrees(
                    elevation(_sub(rotated["link_b"], rotated["link_a"]))
                )
            if name == "arm":
                return math.degrees(
                    _wrap_pi(
                        elevation(_sub(rotated["link_b"], rotated["link_a"]))
                        - elevation(
                            link_dir("Boom", "BoomHinge_mc", "Boom", "StickHinge_mc")
                        )
                    )
                )
            v_edge = _sub(
                _mul(_add(rotated["edge_begin"], rotated["edge_end"]), 0.5),
                rotated["link_a"],
            )
            return math.degrees(_wrap_pi(elevation(v_edge) - phi_arm0))

        angles = [angle_at(state[t][3]) for t in thetas]
        mono_d = all(disp[i + 1] >= disp[i] for i in range(len(disp) - 1)) or all(
            disp[i + 1] <= disp[i] for i in range(len(disp) - 1)
        )
        mono_a = all(angles[i + 1] >= angles[i] for i in range(len(angles) - 1)) or all(
            angles[i + 1] <= angles[i] for i in range(len(angles) - 1)
        )
        stroke = geo["ranges"][spec["stroke_range"]]
        result["joints"][name] = {
            "scan": {"theta_deg": thetas, "disp_m": disp, "angle_deg": angles},
            "stroke_range_m": [float(stroke["start"]), float(stroke["end"])],
            "stroke_range_enabled": bool(stroke["enabled"]),
            "monotonic_disp": bool(mono_d),
            "monotonic_angle": bool(mono_a),
            "mechanism": info,
        }
    return result


# ---------------------------------------------------------------------------- windows
def _interp(theta_a, val_a, target, increasing):
    """θ at a target value; ``val_a`` monotone in ``theta_a`` (ascending θ)."""
    ts = theta_a if increasing else theta_a[::-1]
    vs = val_a if increasing else val_a[::-1]
    if target < vs[0] - 1e-9 or target > vs[-1] + 1e-9:
        return None
    lo, hi = 0, len(vs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if vs[mid] <= target:
            lo = mid
        else:
            hi = mid
    if vs[hi] == vs[lo]:
        return ts[lo]
    frac = (target - vs[lo]) / (vs[hi] - vs[lo])
    return ts[lo] + frac * (ts[hi] - ts[lo])


def stroke_window(joint_scan: dict[str, Any], stroke_ours: tuple[float, float]):
    """θ interval whose (our-convention) displacement stays in ``stroke_ours``."""
    thetas = joint_scan["scan"]["theta_deg"]
    disp = joint_scan["scan"]["disp_m"]
    increasing = disp[-1] >= disp[0]
    t_a = _interp(thetas, disp, stroke_ours[0], increasing)
    t_b = _interp(thetas, disp, stroke_ours[1], increasing)
    if t_a is None or t_b is None:
        return None
    return (min(t_a, t_b), max(t_a, t_b))


def _table_over_window(joint_scan, window_theta, samples=201):
    thetas = joint_scan["scan"]["theta_deg"]
    disp = joint_scan["scan"]["disp_m"]
    angles = joint_scan["scan"]["angle_deg"]
    grid = np.linspace(window_theta[0], window_theta[1], samples)
    d = np.interp(grid, thetas, disp)
    a = np.interp(grid, thetas, angles)
    return d.tolist(), a.tolist()


# ---------------------------------------------------------------------------- fit
def fit_poly_table(disp_m, angle_deg, degree: int = 5):
    """Fit angle_deg as a polynomial of the normalised displacement dn∈[0,1]."""
    d = np.asarray(disp_m, dtype=np.float64)
    a = np.asarray(angle_deg, dtype=np.float64)
    if d[-1] == d[0]:
        raise ValueError("degenerate displacement table")
    dn = (d - d[0]) / (d[-1] - d[0])
    coeffs = np.polyfit(dn, a, degree)
    grid = np.linspace(0.0, 1.0, 4001)
    fit = np.polyval(coeffs, grid)
    err = float(np.max(np.abs(fit - np.interp(grid, dn, a))))
    diffs = np.diff(fit)
    mono = bool(np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9))
    return {
        "coeffs": [float(c) for c in coeffs],
        "max_err_deg": err,
        "monotone": mono,
        "increasing": bool(fit[-1] >= fit[0] - 1e-9),
    }


# ---------------------------------------------------------------------------- replay
def _joint_theta_from_disp(joint_scan, native_disp, sign):
    thetas = joint_scan["scan"]["theta_deg"]
    disp = joint_scan["scan"]["disp_m"]
    increasing = disp[-1] >= disp[0]
    return _interp(thetas, disp, sign * float(native_disp), increasing)


def replay_episode_tip(
    joint_scans: dict[str, Any],
    sign_map: dict[str, float],
    joints_after: np.ndarray,
    links_m,
    pivot_z_m: float,
):
    """FK tip (world z, horizontal radius) per recorded frame, policy conventions."""
    curves = {}
    for name in ("boom", "arm", "bucket"):
        curves[name] = (
            joint_scans[name]["scan"]["theta_deg"],
            joint_scans[name]["scan"]["angle_deg"],
        )
    idx = {"boom": 1, "arm": 2, "bucket": 3}
    theta_series = {}
    for name in ("boom", "arm", "bucket"):
        theta_series[name] = [
            _joint_theta_from_disp(joint_scans[name], v, sign_map[name])
            for v in joints_after[:, idx[name]]
        ]
    L1, L2, L3 = links_m
    z_out, r_out, ok = [], [], []
    for k in range(len(joints_after)):
        t = {name: theta_series[name][k] for name in ("boom", "arm", "bucket")}
        if any(t[name] is None for name in t):
            z_out.append(None)
            r_out.append(None)
            ok.append(False)
            continue
        a1, a2, a3 = (
            math.radians(float(np.interp(t[name], *curves[name])))
            for name in ("boom", "arm", "bucket")
        )
        radius = (
            L1 * math.cos(a1) + L2 * math.cos(a1 + a2) + L3 * math.cos(a1 + a2 + a3)
        )
        height = (
            L1 * math.sin(a1) + L2 * math.sin(a1 + a2) + L3 * math.sin(a1 + a2 + a3)
        )
        r_out.append(radius)
        z_out.append(height + pivot_z_m)
        ok.append(True)
    return {"z_m": z_out, "radius_m": r_out, "ok": ok}


def select_signs(
    joint_scans,
    episodes,
    links_m,
    pivot_z_m: float,
    ground_tol=0.8,
    z_max=7.5,
    verbose=False,
):
    """Pick each cylinder's AGX reading sign (2^3 combos) by physical replay.

    ``AGX_reading = sign * q_ours`` with q_ours the scanned value. A wrong sign
    makes the replayed tip height contradict the recorded scoop cycle: the tip
    must sit near the ground when the scoop loads, must not fly, and must not
    dig deep below ground.
    """
    scores: dict[tuple[float, float, float], float] = {}
    for sb in (1.0, -1.0):
        for sa in (1.0, -1.0):
            for sk in (1.0, -1.0):
                bad = total = 0
                for joints_after, scoop in episodes:
                    rep = replay_episode_tip(
                        joint_scans,
                        {"boom": sb, "arm": sa, "bucket": sk},
                        joints_after,
                        links_m,
                        pivot_z_m,
                    )
                    z = rep["z_m"]
                    for value in z:
                        total += 1
                        if value is None or value > z_max or value < -ground_tol:
                            bad += 1
                    for k in range(1, len(scoop)):
                        if scoop[k] and not scoop[k - 1] and z[k] is not None:
                            total += 1
                            if abs(z[k]) > ground_tol + 0.2:
                                bad += 1
                frac = bad / max(1, total)
                scores[(sb, sa, sk)] = frac
                if verbose:
                    print(
                        f"  signs boom={sb:+.0f} arm={sa:+.0f} bucket={sk:+.0f}: "
                        f"violation {frac:.4f}"
                    )
    ranked = sorted(scores, key=lambda k: scores[k])
    best = ranked[0]
    second = ranked[1]
    margin = scores[second] - scores[best]
    sign_map = {"boom": best[0], "arm": best[1], "bucket": best[2]}
    return sign_map, {
        "scores": {
            "|".join(f"{v:+.0f}" for v in k): round(v, 5) for k, v in scores.items()
        },
        "best": "|".join(f"{v:+.0f}" for v in best),
        "second": "|".join(f"{v:+.0f}" for v in second),
        "margin": round(margin, 5),
    }


# ---------------------------------------------------------------------------- machine file
def build_machine_file(
    scan_result,
    sign_map,
    template: dict[str, Any],
    pick_z_m: float,
    dump_z_m: float,
    poly_degree: int = 5,
    poly_err_deg: float = 0.2,
):
    """Assemble the calibrated machine file (schema = machine_agx365.template.json)."""
    out = copy.deepcopy(template)
    out["calibrated"] = True
    out["_doc"] = (
        "AGX 365 真尺（OpenPLX 纯几何扫描，scripts/calibrate_agx365.py 生成；"
        "符号由采集数据回放选定）。训练与部署共用本文件（五红线）。"
    )
    out["links_m"] = [round(v, 4) for v in scan_result["links_m"]]
    out["pick_z_m"] = float(pick_z_m)
    out["dump_z_m"] = float(dump_z_m)
    specs = {j["name"]: j for j in out["joints"]}
    details: dict[str, Any] = {}
    for name in ("boom", "arm", "bucket"):
        joint = scan_result["joints"][name]
        stroke_lo, stroke_hi = joint["stroke_range_m"]
        sign = sign_map[name]
        window = stroke_window(joint, (sign * stroke_lo, sign * stroke_hi))
        if window is None:
            raise ValueError(
                f"{name}: stroke range not reachable within the scan window"
            )
        d_ours, angles = _table_over_window(joint, window)
        d_native = [sign * d for d in d_ours]  # AGX reading equals sign*q_ours
        order = np.argsort(d_native)
        d_sorted = [d_native[i] for i in order]
        a_sorted = [angles[i] for i in order]
        lo_a = float(np.interp(stroke_lo, d_sorted, a_sorted))
        hi_a = float(np.interp(stroke_hi, d_sorted, a_sorted))
        linear_err = max(
            abs(np.interp(d, [stroke_lo, stroke_hi], [lo_a, hi_a]) - a)
            for d, a in zip(d_sorted, a_sorted, strict=True)
        )
        fit = fit_poly_table(d_sorted, a_sorted, degree=poly_degree)
        use_poly = bool(
            fit["max_err_deg"] <= poly_err_deg
            and fit["max_err_deg"] <= 0.5 * linear_err
            and fit["monotone"]
        )
        # np.polyfit is descending-degree; the machine-file schema stores
        # ascending powers (angle_deg = sum(coeff_i * dn**i)) — reverse.
        conv = {
            "min_displacement_m": round(stroke_lo, 6),
            "max_displacement_m": round(stroke_hi, 6),
            "min_angle_deg": round(lo_a, 4),
            "max_angle_deg": round(hi_a, 4),
            "poly_coeffs": (
                [round(c, 6) for c in reversed(fit["coeffs"])] if use_poly else None
            ),
        }
        specs[name].update(
            {
                "native_unit": "m",
                "sign": 1.0,
                "offset_rad": 0.0,
                "hydraulic_conversion": conv,
            }
        )
        details[name] = {
            "agx_reading_sign_vs_scan": sign,
            "window_theta_deg": [round(window[0], 4), round(window[1], 4)],
            "angle_at_stroke_ends_deg": [round(lo_a, 4), round(hi_a, 4)],
            "linear_max_err_deg": round(linear_err, 4),
            "poly_max_err_deg": round(fit["max_err_deg"], 4),
            "poly_monotone": fit["monotone"],
            "poly_increasing": fit["increasing"],
            "poly_used": use_poly,
            "angle_increases_with_disp": bool(a_sorted[-1] >= a_sorted[0]),
        }
    out["_calibration_details"] = details
    return out


# ---------------------------------------------------------------------------- report
def geometry_report(
    geo: dict[str, Any], layout: dict[str, Any] = AGX365_LAYOUT
) -> dict:
    """Key assembly measurements (audit evidence for the calibration report)."""
    w = _World(geo)
    report: dict[str, Any] = {
        "bodies": len(geo["bodies"]),
        "constraints": len(geo["constraints"]),
    }
    pivot = {
        "boom": ("ChassieBody", "BoomHinge_mc"),
        "arm": ("Boom", "StickHinge_mc"),
        "bucket": ("Stick", "BucketHinge_mc"),
    }
    report["pivots_m"] = {
        name: [round(v, 5) for v in w.point(*ref)] for name, ref in pivot.items()
    }
    report["swing_axis"] = [
        round(v, 5) for v in w.direction(*layout["system_layout"]["up_mc"])
    ]
    edge_mid = _mul(
        _add(w.point(*layout["edge_begin"]), w.point(*layout["edge_end"])), 0.5
    )
    report["edge_mid_m"] = [round(v, 5) for v in edge_mid]
    four_bar = layout["joints"]["bucket"]["four_bar"]
    a_fixed = w.point(*four_bar["fixed_a"])
    b0 = w.point(*four_bar["fixed_rod_mc"])
    c0 = w.point("Bucket", "Hinge32_mc")
    d = w.point("Stick", "BucketHinge_mc")
    report["four_bar_m"] = {
        "crank_hinge34_m": [round(v, 5) for v in a_fixed],
        "pin_b_m": [round(v, 5) for v in b0],
        "hinge32_m": [round(v, 5) for v in c0],
        "bucket_hinge_m": [round(v, 5) for v in d],
        "crank_len": round(_norm(_sub(b0, a_fixed)), 5),
        "coupler_len": round(_norm(_sub(c0, b0)), 5),
        "rocker_len": round(_norm(_sub(d, c0)), 5),
    }
    e9 = w.point("Stick", "Hinge9_mc")
    report["bucket_cyl_initial_m"] = round(_norm(_sub(b0, e9)), 5)
    m0 = w.point("CylContainerEnd", "BucketPrismatic_mc")
    report["prismatic_anchor_lateral_offset_m"] = {
        "boom": round(
            abs(
                w.point("StartCylContainer1", "ArmPrismatic1_mc")[1]
                - w.point("ChassieBody", "Hinge1_mc")[1]
            ),
            5,
        ),
        "bucket": round(_norm(_cross(_sub(m0, e9), _unit(_sub(b0, e9)))), 5),
    }
    return report
