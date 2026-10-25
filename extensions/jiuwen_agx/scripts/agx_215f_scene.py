# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""215F（SWE215F / describe_215i）AGX 场景工厂。

在 **AGX 自带 Python** 里执行（agxGym 的依赖与仓库 venv 不兼容），由
``agx_bridge_server.py --machine 215f`` 或 ``agx_viewer_bridge_215f.agxPy`` 调用。
本文件也能单独跑做自检::

    %AGX_DIR%\\python-x64\\python.exe scripts\\agx_215f_scene.py --steps 60

场景 = 山河 215F 交付包的环境类 ``Excavator215FEnv``（可变形 agxTerrain 土壤 +
Shovel 注册 + 接触材料 + 265 kg 岩石），四关节：

    swing  -> base_to_shangche   (Hinge, rad, 上车回转；底盘固定、无履带)
    boom   -> dongbi_joint       (Hinge, rad, ±90°)
    arm    -> dougan_joint       (Hinge, rad, 0..127°)
    bucket -> chandou_joint      (Hinge, rad, 0..186°)

**副本选择**：``sh`` 目录里的原件是 Linux / AGX 2.39 版，本机跑不动；
``deploy/215f`` 里那份是同一交付的本机适配副本（相对路径 + 2.39/2.42 API
兼容垫片）。默认用适配副本，``JIUWEN_215F_ENV_ROOT`` 可指到别处。

环境变量（都有缺省值，只有换机器/换土堆时才需要设）::

    JIUWEN_215F_ENV_ROOT     环境类所在目录（默认 deploy\\215f\\215F_AGX_MINIMAL）
    JIUWEN_215F_URDF_DIR     describe_215i 目录（默认 <env_root>\\..\\describe_215i）
    JIUWEN_215F_PYLIBS       部署包依赖（gymnasium 等；默认 <env_root>\\..\\pylibs）
    JIUWEN_215F_PILE         土堆 "top,x0,x1,y0,y1,ramp"，米
                             （默认 1.95,3.4,6.6,0.4,3.6,0.5）
    JIUWEN_215F_NO_PILE=1    不铺土堆（平地对照）
    JIUWEN_215F_FORCE_KNM    力矩上限 kN·m "swing,boom,arm,bucket"（默认 80,400,250,150）
    JIUWEN_215F_MACHINE      对端身份名（默认 excavator215f）
    JIUWEN_215F_LOADED_KG    载料阈值（kg；默认交给桥接的 50）
    JIUWEN_215F_VERBOSE=1    打印环境详细日志

为什么抬高力矩：桥接的位置锁从 Motor1D 继承力上限，URDF 的占位 effort（动臂
50 kN·m）撑不住伸出的工作装置——小脑组实测动臂以 −40°/s 下坠；他们的 215F
映射（agx_mapping_215f.json）给的是 swing 80 / boom 400 / arm 250 / bucket 150 kN·m。
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["build", "pile_heights", "pile_spec", "force_ranges", "main"]

DEFAULT_ENV_ROOT = r"E:\AGX-2.42.2.1\deploy\215f\215F_AGX_MINIMAL"
DEFAULT_MACHINE_NAME = "excavator215f"
DEFAULT_FORCE_KNM: dict[str, float] = {
    "swing": 80.0,
    "boom": 400.0,
    "arm": 250.0,
    "bucket": 150.0,
}
DEFAULT_PILE: dict[str, Any] = {
    "top": 1.95,
    "x": (3.4, 6.6),
    "y": (0.4, 3.6),
    "ramp": 0.5,
}
JOINTS_VOCAB: tuple[str, ...] = ("swing", "boom", "arm", "bucket")
# 词表关节名 -> (URDF 关节名, 是否在 env.joint_hinges 里)
JOINT_SPEC: dict[str, tuple[str, bool]] = {
    "swing": ("base_to_shangche", False),
    "boom": ("dongbi_joint", True),
    "arm": ("dougan_joint", True),
    "bucket": ("chandou_joint", True),
}
TERRAIN_RES = 128
TERRAIN_SIZE_M = 20.0
BASE_LINK = "base_link"
# Fallbacks; the machine json is the source of truth when present.
TIP_LOCAL_M: tuple[float, float, float] = (-0.8236, 0.0029, 1.2164)
POLICY_ORIGIN_M: tuple[float, float, float] = (0.093, -0.0425, 1.8125)


def _machine_data() -> dict[str, Any]:
    """Per-machine geometry/IK metadata (also used by the IK loader)."""
    path = Path(__file__).resolve().parents[1] / "configs" / "machine_swe215f.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fnum(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[agx_215f] WARNING: bad {name}={raw!r}, using {default}")
        return default


def pile_spec() -> dict[str, Any] | None:
    """土堆参数（``None`` = 不铺）。``JIUWEN_215F_PILE`` 覆盖默认值。"""
    if os.environ.get("JIUWEN_215F_NO_PILE", "").strip() in ("1", "true", "yes"):
        return None
    raw = os.environ.get("JIUWEN_215F_PILE", "").strip()
    if not raw:
        return dict(DEFAULT_PILE)
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 6:
        print(f"[agx_215f] WARNING: JIUWEN_215F_PILE needs 6 numbers, got {raw!r}")
        return dict(DEFAULT_PILE)
    top, x0, x1, y0, y1, ramp = (float(p) for p in parts)
    return {"top": top, "x": (x0, x1), "y": (y0, y1), "ramp": ramp}


def pile_heights(spec: dict[str, Any]) -> list[float]:
    """128×128 高度场：土堆 = 带斜坡边缘的矩形（照交付场景的同一公式）。"""
    cell = TERRAIN_SIZE_M / TERRAIN_RES
    half = TERRAIN_RES // 2
    x0, x1 = float(spec["x"][0]), float(spec["x"][1])
    y0, y1 = float(spec["y"][0]), float(spec["y"][1])
    top, ramp = float(spec["top"]), max(float(spec["ramp"]), 1e-6)
    heights: list[float] = []
    for j in range(TERRAIN_RES):
        y = (j - half) * cell
        for i in range(TERRAIN_RES):
            x = (i - half) * cell
            hx = min((x - x0) / ramp, (x1 - x) / ramp, 1.0)
            hy = min((y - y0) / ramp, (y1 - y) / ramp, 1.0)
            frac = max(0.0, min(1.0, min(hx, hy)))
            heights.append(top * frac)
    return heights


def force_ranges() -> dict[str, float]:
    """各关节力矩上限（kN·m）。``JIUWEN_215F_FORCE_KNM`` = 4 个数或 name=value。"""
    ranges = dict(DEFAULT_FORCE_KNM)
    raw = os.environ.get("JIUWEN_215F_FORCE_KNM", "").strip()
    if not raw:
        return ranges
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if all("=" in p for p in parts):
        for part in parts:
            key, _, value = part.partition("=")
            ranges[key.strip()] = float(value)
        return ranges
    if len(parts) == 4:
        for key, value in zip(JOINTS_VOCAB, parts, strict=True):
            ranges[key] = float(value)
        return ranges
    print(f"[agx_215f] WARNING: JIUWEN_215F_FORCE_KNM needs 4 numbers, got {raw!r}")
    return ranges


def _try_set_force_range(holder: Any, value: float) -> bool:
    """给 Motor1D / Lock1D 设对称力限；两种 SWIG 重载都试（RangeReal 优先）。"""
    if holder is None:
        return False
    setter = getattr(holder, "setForceRange", None)
    if setter is None:
        return False
    import agx

    errors: list[str] = []
    for args in ((agx.RangeReal(-value, value),), (-value, value)):
        try:
            setter(*args)
            return True
        except Exception as exc:  # noqa: BLE001 - 试下一个重载
            errors.append(str(exc))
    print(f"[agx_215f] WARNING: setForceRange failed: {'; '.join(errors[:2])}")
    return False


def _find_hinge(sim: Any, name: str) -> Any:
    """按名字找铰（场景里 ``getConstraints()`` 返回基类包装，需要 asHinge 转型）。"""
    for constraint in sim.getConstraints():
        if constraint.getName() != name:
            continue
        hinge = getattr(constraint, "asHinge", lambda: None)()
        if hinge is not None:
            return hinge
        if type(constraint).__name__ == "Hinge":
            return constraint
    raise ValueError(
        f"[agx_215f] 场景里找不到铰 {name!r}（URDF 装配异常？可用名见桥接 inventory）"
    )


def build() -> dict[str, Any]:
    """搭好 215F 带土场景，返回桥接 ``attach_machine`` 认的句柄 dict。

    键：sim / joints / units / machine_name / terrain_truth / load_mass_fn /
    loaded_threshold_kg / env（env 只给 viewer 渲染和保活，桥接不读）。
    """
    env_root = os.environ.get("JIUWEN_215F_ENV_ROOT", DEFAULT_ENV_ROOT)
    parent = os.path.dirname(env_root)
    urdf_dir = os.environ.get(
        "JIUWEN_215F_URDF_DIR", os.path.join(parent, "describe_215i")
    )
    pylibs = os.environ.get("JIUWEN_215F_PYLIBS", os.path.join(parent, "pylibs"))
    verbose = os.environ.get("JIUWEN_215F_VERBOSE", "").strip() in ("1", "true", "yes")

    for path in (env_root, pylibs):
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    # 交付环境自己也认这个变量（deploy 副本的适配点之一）
    os.environ.setdefault("AGX_215F_URDF_DIR", urdf_dir)
    if not os.path.exists(os.path.join(urdf_dir, "urdf", "describe_215i_fixed.urdf")):
        raise FileNotFoundError(
            f"[agx_215f] URDF 不在 {urdf_dir!r} —— 用 JIUWEN_215F_URDF_DIR 指向 "
            "describe_215i 目录（其下应有 urdf/describe_215i_fixed.urdf）"
        )

    import agx
    import agxTerrain  # noqa: F401  （环境类要用；顺便验证模块存在）
    from agxGym.envs.excavator215f_env_final import Excavator215FEnv

    print(
        f"[agx_215f] env={env_root}\n[agx_215f] urdf={urdf_dir}\n[agx_215f] pylibs={pylibs}",
        flush=True,
    )
    env = Excavator215FEnv(urdf_dir=urdf_dir, verbose=verbose)
    env.reset(seed=0)
    sim = env.sim

    # The delivery URDF imports with fixToWorld=True: the root is held by a
    # world LockJoint. Release it and make the root kinematic so the whole
    # machine can be repositioned (the terrain stays world-fixed). This is not
    # articulated-track physics; it is the "move into range" primitive.
    base = sim.getRigidBody(BASE_LINK)
    if base is None:
        raise RuntimeError(f"[agx_215f] scene has no {BASE_LINK!r} rigid body")
    for constraint in sim.getConstraints():
        body0 = constraint.getBodyAt(0)
        body1 = constraint.getBodyAt(1)
        name0 = body0.getName() if body0 is not None else None
        name1 = body1.getName() if body1 is not None else None
        if (name0 == BASE_LINK and name1 is None) or (
            name1 == BASE_LINK and name0 is None
        ):
            constraint.setEnable(False)
            print(
                "[agx_215f] released root fixToWorld constraint "
                f"{constraint.getName()!r} ({type(constraint).__name__})",
                flush=True,
            )
            break
    else:
        print(
            "[agx_215f] WARNING: no world constraint found on base_link; "
            "kinematic base move may fight the constraint",
            flush=True,
        )
    base.setMotionControl(agx.RigidBody.KINEMATICS)

    rest_pose = {
        name: round(math.degrees(float(hinge.getAngle())), 2)
        for name, hinge in env.joint_hinges.items()
    }
    print(
        f"[agx_215f] scene built; dt={float(sim.getTimeStep()):.5f} s, "
        f"rest pose {rest_pose}",
        flush=True,
    )

    joints: dict[str, Any] = {}
    for vocab, (urdf_name, in_env) in JOINT_SPEC.items():
        if in_env:
            hinge = env.joint_hinges.get(urdf_name)
            if hinge is None:
                raise ValueError(f"[agx_215f] env.joint_hinges 缺 {urdf_name!r}")
        else:
            hinge = _find_hinge(sim, urdf_name)
        joints[vocab] = hinge
    units = dict.fromkeys(JOINT_SPEC, "rad")

    # 桥接的位置锁继承 Motor1D 力限；URDF 占位 effort 撑不住带载动臂。
    for vocab, knm in force_ranges().items():
        hinge = joints.get(vocab)
        if hinge is None:
            continue
        _try_set_force_range(hinge.getMotor1D(), knm * 1000.0)
        _try_set_force_range(hinge.getLock1D(), knm * 1000.0)

    spec = pile_spec()
    cell = TERRAIN_SIZE_M / TERRAIN_RES
    half = TERRAIN_RES // 2

    def _height_at_world(x_m: float, y_m: float) -> float | None:
        i = int(round(float(x_m) / cell)) + half
        j = int(round(float(y_m) / cell)) + half
        if not (0 <= i < TERRAIN_RES and 0 <= j < TERRAIN_RES):
            return None
        hf = env.ground.getHeightField()
        try:
            return float(hf.getHeight(i, j))
        except Exception:  # noqa: BLE001 - fall back to the other AGX accessor
            getter = getattr(env.ground, "getHeight", None)
            if getter is None:
                return None
            try:
                return float(getter(agx.Vec2i(i, j)))
            except Exception:  # noqa: BLE001
                return None

    world_center: tuple[float, float] | None = None
    pile_volume_m3 = 0.0
    if spec is not None:
        heights = agx.RealVector()
        for value in pile_heights(spec):
            heights.append(float(value))
        env.ground.setHeights(heights)
        cx = 0.5 * (spec["x"][0] + spec["x"][1])
        cy = 0.5 * (spec["y"][0] + spec["y"][1])
        area = (spec["x"][1] - spec["x"][0]) * (spec["y"][1] - spec["y"][0])
        world_center = (cx, cy)
        pile_volume_m3 = round(0.5 * area * float(spec["top"]), 2)  # 斜坡棱柱近似
        print(
            f"[agx_215f] pile: x {spec['x']} y {spec['y']} top {spec['top']:.2f} m "
            f"ramp {spec['ramp']:.2f} m -> world center {world_center}",
            flush=True,
        )
    else:
        print("[agx_215f] pile: skipped (JIUWEN_215F_NO_PILE)", flush=True)

    data = _machine_data()
    tip_local = agx.Vec3(*[float(v) for v in data.get("tip_local_m", TIP_LOCAL_M)])
    origin = tuple(
        float(v) for v in data.get("policy_origin_in_base_m", POLICY_ORIGIN_M)
    )

    def base_pose_fn() -> dict[str, float]:
        pos = base.getPosition()
        yaw = float(base.getRotation().getAsEulerAngles().z())
        return {
            "x_m": float(pos.x()),
            "y_m": float(pos.y()),
            "z_m": float(pos.z()),
            "yaw_rad": yaw,
        }

    # ---- kinematic whole-machine move: ANIMATED, not teleported ----------------
    # An instant relocation both looks like a jump in the viewer AND physically
    # dumps the bucket: soil is bulk material, it stays where the bucket was.
    # Moving the assembly in small per-sim-step increments lets contact carry
    # the load — the same mechanism that lets the bucket dig at all.
    base_move_goal: dict[str, Any] | None = None  # 活动中的整机移动目标
    base_speed_mps = _fnum("JIUWEN_215F_BASE_MPS", 0.5)
    base_turn_rps = _fnum("JIUWEN_215F_BASE_RPS", 0.25)
    _SIM_DT = 1.0 / 60.0

    def _machine_bodies() -> list[tuple[str, Any]]:
        """The whole URDF mechanism. The delivery env snapshots this name list
        BEFORE the rock/terrain exist, so the terrain is never in it."""
        cache = getattr(env, "_direct_pose_cache", None)
        names = (
            list(cache.keys())
            if isinstance(cache, dict) and cache
            else [
                BASE_LINK,
                "shangche_Link",
                "dongbi_Link",
                "dougan_Link",
                "chandou_Link",
            ]
        )
        out = []
        for name in names:
            rb = sim.getRigidBody(name)
            if rb is not None:
                out.append((name, rb))
        return out

    def _assembly_pose_at(
        goal: dict[str, Any], u: float
    ) -> list[tuple[str, Any, Any, Any]]:
        """Exact rigid poses of every machine body at progress ``u`` ∈ [0, 1]:
        rotate the start poses about the base's start origin by ``u·dyaw``,
        then translate by ``u·step``. Rigid by construction — every joint angle
        (swing included) is preserved at every instant."""
        angle = goal["dyaw"] * u
        c, s = math.cos(angle), math.sin(angle)
        rz = agx.Quat(angle, agx.Vec3(0.0, 0.0, 1.0)) if angle != 0.0 else None
        pos0 = goal["base_pos0"]
        step = goal["step"]
        tx, ty, tz = step.x() * u, step.y() * u, step.z() * u
        poses = []
        for name, rb, p0, r0 in goal["bodies"]:
            rel_x = float(p0.x()) - float(pos0.x())
            rel_y = float(p0.y()) - float(pos0.y())
            pos = agx.Vec3(
                float(pos0.x()) + c * rel_x - s * rel_y + tx,
                float(pos0.y()) + s * rel_x + c * rel_y + ty,
                float(p0.z()) + tz,
            )
            rot = r0 if rz is None else rz.leftMult(r0)
            poses.append((name, rb, pos, rot))
        return poses

    def _finish_base_move(goal: dict[str, Any]) -> None:
        """Snap the exact end poses, still the loads, and resync the caches."""
        nonlocal base_move_goal
        zero = agx.Vec3(0.0, 0.0, 0.0)
        cache = getattr(env, "_direct_pose_cache", None)
        for name, rb, pos, rot in _assembly_pose_at(goal, 1.0):
            rb.setPosition(pos)
            rb.setRotation(rot)
            rb.setVelocity(zero)
            rb.setAngularVelocity(zero)
            if isinstance(cache, dict):
                cache[name] = (pos, rot)
        # The shovel proxy follows the bucket (the same resync the delivery
        # env's _restore_direct_pose does after moving bodies around).
        proxy = getattr(env, "shovel_proxy_body", None)
        bucket = getattr(env, "bucket_body", None)
        if proxy is not None and bucket is not None:
            try:
                proxy.setTransform(bucket.getTransform())
            except Exception:  # noqa: BLE001 - fall back to a pose copy
                proxy.setPosition(bucket.getPosition())
                proxy.setRotation(bucket.getRotation())
            proxy.setVelocity(zero)
            proxy.setAngularVelocity(zero)
        if base_move_goal is goal:
            base_move_goal = None

    def move_base(
        dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0
    ) -> dict[str, float]:
        """Start a kinematic whole-machine move: rotate, then translate in the
        new frame — ANIMATED over sim time (see ``base_step_fn``). Returns the
        PREDICTED end pose; the move itself completes asynchronously and the
        bridge reports progress through ``base_state`` (viewer/pump mode) or
        steps the sim itself (headless)."""
        nonlocal base_move_goal
        pos = base.getPosition()
        yaw = float(base.getRotation().getAsEulerAngles().z())
        dyaw = float(dyaw_rad)
        new_yaw = (yaw + dyaw + math.pi) % (2.0 * math.pi) - math.pi
        # World-frame step: the body-frame (dx, dy) expressed in the NEW heading.
        c2, s2 = math.cos(new_yaw), math.sin(new_yaw)
        step = agx.Vec3(
            c2 * float(dx_m) - s2 * float(dy_m),
            s2 * float(dx_m) + c2 * float(dy_m),
            0.0,
        )
        goal: dict[str, Any] = {
            "bodies": [
                (name, rb, rb.getPosition(), rb.getRotation())
                for name, rb in _machine_bodies()
            ],
            "base_pos0": pos,
            "step": step,
            "dyaw": dyaw,
            "t0": float(sim.getTimeStamp()),
        }
        goal["duration"] = max(
            math.hypot(step.x(), step.y()) / max(base_speed_mps, 1e-6),
            abs(dyaw) / max(base_turn_rps, 1e-6),
        )
        goal["deadline"] = (
            goal["t0"] + goal["duration"] + max(1.0, 0.5 * goal["duration"])
        )
        if not goal["bodies"]:
            # No cache (unexpected): at least relocate the base itself.
            base.setPosition(
                agx.Vec3(
                    float(pos.x()) + step.x(), float(pos.y()) + step.y(), float(pos.z())
                )
            )
            base.setRotation(agx.Quat(new_yaw, agx.Vec3(0.0, 0.0, 1.0)))
        elif goal["duration"] < 0.05:
            _finish_base_move(goal)  # degenerate nudge: apply instantly
        else:
            if base_move_goal is not None:
                print(
                    "[agx_215f] WARNING: 上一条整机移动还没走完就被覆盖 —— "
                    "客户端应当轮询 base_state 等到位",
                    flush=True,
                )
            base_move_goal = goal
        return {
            "x_m": float(pos.x()) + step.x(),
            "y_m": float(pos.y()) + step.y(),
            "z_m": float(pos.z()),
            "yaw_rad": new_yaw,
        }

    def base_step_fn(sim_time_s: float) -> None:
        """Advance the animated move ONE sim step (the bridge calls this from its
        per-step tick, before stepping). Smoothstep ease-in/out; positions are
        set analytically, velocities are the finite difference so contacts (the
        soil in the bucket) see the real motion."""
        goal = base_move_goal
        if goal is None:
            return
        if sim_time_s >= goal["deadline"]:
            _finish_base_move(goal)
            return
        u = min(max((sim_time_s - goal["t0"]) / goal["duration"], 0.0), 1.0)
        s = u * u * (3.0 - 2.0 * u)  # smoothstep
        if u >= 1.0:
            _finish_base_move(goal)
            return
        u2 = min(u + _SIM_DT / goal["duration"], 1.0)
        s2 = u2 * u2 * (3.0 - 2.0 * u2)
        spin_z = goal["dyaw"] * (s2 - s) / _SIM_DT
        now_poses = _assembly_pose_at(goal, s)
        next_poses = _assembly_pose_at(goal, s2)
        for (_, rb, p, r), (_, _rb2, p2, _r2) in zip(
            now_poses, next_poses, strict=True
        ):
            rb.setPosition(p)
            rb.setRotation(r)
            rb.setVelocity(
                agx.Vec3(
                    (p2.x() - p.x()) / _SIM_DT,
                    (p2.y() - p.y()) / _SIM_DT,
                    (p2.z() - p.z()) / _SIM_DT,
                )
            )
            rb.setAngularVelocity(agx.Vec3(0.0, 0.0, spin_z))

    def base_busy_fn(sim_time_s: float) -> dict[str, Any]:
        goal = base_move_goal
        if goal is None:
            return {"busy": False, "remaining_s": 0.0}
        remaining = max(0.0, goal["t0"] + goal["duration"] - sim_time_s)
        return {"busy": True, "remaining_s": remaining}

    def surface_fn(x_m: float, y_m: float) -> float | None:
        """Soil height at a BASE-frame (x, y); converts to world through base."""
        world = base.getFrame().transformPointToWorld(
            agx.Vec3(float(x_m), float(y_m), 0.0)
        )
        return _height_at_world(world.x(), world.y())

    def tip_fn() -> dict[str, Any]:
        """Bucket-tip pose in world and policy frames."""
        if env.bucket_body is None:
            raise RuntimeError("[agx_215f] scene has no bucket_body")
        world = env.bucket_body.getFrame().transformPointToWorld(tip_local)
        local = base.getFrame().transformPointToLocal(world)
        return {
            "world_m": [float(world.x()), float(world.y()), float(world.z())],
            "policy_m": [
                float(local.x()) - origin[0],
                float(local.y()) - origin[1],
                float(local.z()) - origin[2],
            ],
            "base_link": BASE_LINK,
        }

    def terrain_truth_fn() -> list[dict[str, Any]]:
        """The HIGHEST remaining point on the pile, not its original center.

        Digging turns the center into a pit; reporting the fixed center makes
        the brain aim at the pit bottom, where the bucket is stopped by the
        pit wall. A small grid over the footprint follows the material as the
        pile is consumed.
        """
        if spec is None and world_center is None:
            return []
        probe: list[tuple[float, float]] = []
        if spec is not None:
            x0, x1 = float(spec["x"][0]), float(spec["x"][1])
            y0, y1 = float(spec["y"][0]), float(spec["y"][1])
            # Stay a little inside the ramp edge; a point on the exact edge is
            # reachable but usually has no material under the teeth.
            ix, iy = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
            xs = [x0 + ix + f * (x1 - x0 - 2 * ix) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
            ys = [y0 + iy + f * (y1 - y0 - 2 * iy) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
            probe = [(x, y) for x in xs for y in ys]
        best: tuple[float, float, float] | None = None
        cx0, cy0 = world_center if world_center is not None else (0.0, 0.0)
        for x, y in probe:
            height = _height_at_world(x, y)
            if height is None:
                continue
            if best is None or height > best[2] + 1e-3:
                best = (x, y, height)
            elif abs(height - best[2]) <= 1e-3:
                # Flat top: prefer the point nearest the pile center so the
                # first dig is not aimed at a ramp corner.
                if (x - cx0) ** 2 + (y - cy0) ** 2 < (best[0] - cx0) ** 2 + (
                    best[1] - cy0
                ) ** 2:
                    best = (x, y, height)
        if best is None:
            if world_center is None:
                return []
            cx, cy = world_center
            height = _height_at_world(cx, cy)
            best = (cx, cy, 0.0 if height is None else height)
        cx, cy, z = best
        local = base.getFrame().transformPointToLocal(agx.Vec3(cx, cy, 0.0))
        return [
            {
                "name": "soil_pile_215f",
                "x_m": round(float(local.x()), 3),
                "y_m": round(float(local.y()), 3),
                "z_m": round(float(z), 3),
                "volume_m3": pile_volume_m3,
            }
        ]

    terrain_truth = terrain_truth_fn()

    def load_mass_kg() -> float:
        """铲斗内土壤质量（内容口径，与 365 桥接同款）。"""
        aggregate = getattr(env.shovel, "getSoilParticleAggregate", None)
        if aggregate is not None:
            try:
                return float(aggregate().getTotalAggregateMass())
            except Exception:  # noqa: BLE001 - 退回扰动口径
                pass
        return float(env._get_shovel_dynamic_mass())  # noqa: SLF001

    threshold_raw = os.environ.get("JIUWEN_215F_LOADED_KG", "").strip()
    print(
        f"[agx_215f] bucket={env.bucket_body.getName() if env.bucket_body else None} "
        f"shovel={env.shovel is not None} soil_mass={load_mass_kg():.4f} kg",
        flush=True,
    )
    return {
        "sim": sim,
        "joints": joints,
        "units": units,
        "machine_name": os.environ.get("JIUWEN_215F_MACHINE", DEFAULT_MACHINE_NAME),
        "terrain_truth": terrain_truth,
        "terrain_truth_fn": terrain_truth_fn,
        "load_mass_fn": load_mass_kg,
        "loaded_threshold_kg": (float(threshold_raw) if threshold_raw else None),
        "tip_fn": tip_fn,
        "surface_fn": surface_fn,
        "base": {
            "body": base,
            "pose_fn": base_pose_fn,
            "move_fn": move_base,
            "step_fn": base_step_fn,
            "busy_fn": base_busy_fn,
        },
        "env": env,  # viewer 渲染/保活用；桥接不读
    }


def main() -> None:
    """自检入口：搭场景、打印句柄摘要，可选步进观察（--steps N）。"""
    import argparse

    parser = argparse.ArgumentParser(description="215F 场景工厂自检")
    parser.add_argument("--steps", type=int, default=0, help="再步进 N 步后退出")
    args = parser.parse_args()

    handle = build()
    print("[agx_215f] handle:")
    for key, value in handle.items():
        if key in ("sim", "env"):
            print(f"  {key:<20} {type(value).__name__}")
        elif key == "joints":
            for vocab, hinge in value.items():
                rng = hinge.getRange1D().getRange()
                print(
                    f"  joint {vocab:<7} {hinge.getName():<18} "
                    f"angle={float(hinge.getAngle()):+8.3f} "
                    f"range=[{float(rng.lower()):+8.3f}, {float(rng.upper()):+8.3f}] rad"
                )
        else:
            print(f"  {key:<20} {value}")
    if args.steps > 0:
        sim = handle["sim"]
        target = float(sim.getTimeStamp()) + args.steps * float(sim.getTimeStep())
        while float(sim.getTimeStamp()) < target:
            sim.stepForward()
        print(f"[agx_215f] stepped to t={float(sim.getTimeStamp()):.2f} s")
        print(f"[agx_215f] soil_mass={float(handle['load_mass_fn']()):.4f} kg")


if __name__ == "__main__":
    main()
