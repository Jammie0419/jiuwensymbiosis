# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Planar FK/IK for a 4-DOF excavator (swing yaw + boom/arm/bucket in one plane).

自包含的运动学（不依赖小脑组的 `act_trajectory` 包）。模型已在 215F 上标定：
静止位姿下 `forward()` 与 AGX 实测铲尖差 **6 mm**（2026-10-13）。

约定（与 `agx_mapping_215f.json` 同源）：

    q = [yaw, boom, arm, bucket]        策略系弧度
    q_policy = sign · q_native + offset_rad      （每机器文件给 sign/offset）
    forward(q) = 铲齿尖在**策略系**的位置
                 策略系原点 = 动臂铰点在 base_link 里的位置，轴向同 base_link
    三段连杆：boom 销距 / arm 销距 / 铲斗销距（齿尖），角度自上而下累加
    tool_pitch(q) = boom + arm + bucket（铲斗在世界系里的俯仰）

执行限位默认用机器文件里的策略限位；上层可以用 YAML 的原生关节限位覆盖
（:meth:`set_execution_limits_native`），避免把“策略推理限位”当成机械硬限位。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["ExcavatorIk", "machine_file_path"]

_JOINT_ORDER = ("swing", "boom", "arm", "bucket")


def machine_file_path(path: str | Path) -> Path:
    """定位每机器标定 json：绝对路径直接用；相对路径依次试 CWD 与扩展包 configs/。"""
    candidate = Path(path)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    ext_root = Path(__file__).resolve().parents[2]
    tries = [
        Path.cwd() / candidate,
        ext_root / candidate,
        ext_root / "configs" / candidate,
        ext_root.parents[1] / candidate,
    ]
    for option in tries:
        if option.is_file():
            return option
    raise FileNotFoundError(
        f"machine file {path!r} 不在任何候选位置: {[str(p) for p in tries]}"
    )


class ExcavatorIk:
    """4-DOF excavator kinematics: yaw + planar 3-link chain."""

    def __init__(
        self,
        *,
        links_m: list[float],
        offsets_rad: list[float],
        signs: list[float] | None = None,
        limits_policy: list[tuple[float, float]] | None = None,
        max_step_policy: list[float] | None = None,
        joint_order: tuple[str, ...] = _JOINT_ORDER,
        origin_in_base_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        bucket_open_native: float = 0.0,
    ) -> None:
        self.links = np.asarray(links_m, dtype=np.float64)
        self.offsets = np.asarray(offsets_rad, dtype=np.float64)
        self.signs = np.asarray(
            signs if signs is not None else [1.0] * 4, dtype=np.float64
        )
        if self.links.shape != (3,):
            raise ValueError("links_m 需要 3 个正数（boom/arm/bucket 销距，米）")
        if np.any(self.links <= 0) or not np.isfinite(self.links).all():
            raise ValueError("links_m 必须有限且为正")
        if self.offsets.shape != (4,) or self.signs.shape != (4,):
            raise ValueError("offsets_rad / signs 需要 4 个值（swing/boom/arm/bucket）")
        self.joint_order = tuple(joint_order)
        self.origin_in_base = tuple(float(v) for v in origin_in_base_m)
        # Preferred dump/open direction in the bucket joint's native unit.
        self.bucket_open_native = float(bucket_open_native)
        self.limits = (
            np.asarray(limits_policy, dtype=np.float64)
            if limits_policy is not None
            else np.tile(np.array([-math.pi, math.pi]), (4, 1))
        )
        self.max_step = np.asarray(
            max_step_policy if max_step_policy is not None else [0.3] * 4,
            dtype=np.float64,
        )

    # ------------------------------------------------------------ 装载
    @classmethod
    def from_machine_file(cls, path: str | Path) -> "ExcavatorIk":
        """从每机器标定 json 读（`links_m` + `joints[].sign/offset_rad` + `dig_ik`）。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExcavatorIk":
        order = tuple(data.get("policy_joint_order") or _JOINT_ORDER)
        joints = {str(j["name"]): j for j in (data.get("joints") or [])}
        missing = [name for name in order if name not in joints]
        if missing:
            raise ValueError(f"machine file 缺少关节定义: {missing}")
        units = {str(joints[name].get("native_unit", "rad")) for name in order}
        if units != {"rad"}:
            raise ValueError(
                "ExcavatorIk only supports all-radian machines; "
                f"machine file declares native_unit={sorted(units)}"
            )
        ik_cfg = dict(data.get("dig_ik") or {})
        origin = tuple(
            float(v) for v in (data.get("policy_origin_in_base_m") or (0.0, 0.0, 0.0))
        )
        return cls(
            links_m=[float(v) for v in data["links_m"]],
            offsets_rad=[float(joints[name]["offset_rad"]) for name in order],
            signs=[float(joints[name].get("sign", 1.0)) for name in order],
            limits_policy=[
                tuple(float(v) for v in pair)
                for pair in ik_cfg.get("policy_limits_rad", [])
            ]
            or None,
            max_step_policy=[
                float(v) for v in ik_cfg.get("policy_max_step_rad", [])
            ]
            or None,
            joint_order=order,
            origin_in_base_m=origin,
            bucket_open_native=float(ik_cfg.get("bucket_open_native", 0.0)),
        )

    def set_execution_limits_native(
        self, limits: dict[str, tuple[float, float]]
    ) -> None:
        """Use the machine's native joint limits as hard IK execution limits."""
        rows: list[tuple[float, float]] = []
        for i, name in enumerate(self.joint_order):
            if name not in limits:
                raise ValueError(f"missing native limit for joint {name!r}")
            lo, hi = limits[name]
            a = float(lo) * self.signs[i] + self.offsets[i]
            b = float(hi) * self.signs[i] + self.offsets[i]
            rows.append((min(a, b), max(a, b)))
        self.limits = np.asarray(rows, dtype=np.float64)

    # ------------------------------------------------------------ 换算
    def native_to_policy(self, native: dict[str, float]) -> np.ndarray:
        return np.asarray(
            [
                float(native[name]) * self.signs[i] + self.offsets[i]
                for i, name in enumerate(self.joint_order)
            ],
            dtype=np.float64,
        )

    def policy_to_native(self, q: Any) -> dict[str, float]:
        q = np.asarray(q, dtype=np.float64)
        return {
            name: float((q[i] - self.offsets[i]) / self.signs[i])
            for i, name in enumerate(self.joint_order)
        }

    # ------------------------------------------------------------ 正/逆运动学
    def forward(self, q: Any) -> np.ndarray:
        yaw, boom, arm, bucket = (float(v) for v in np.asarray(q, dtype=np.float64))
        angles = np.array([boom, boom + arm, boom + arm + bucket])
        radius = float(self.links @ np.cos(angles))
        z = float(self.links @ np.sin(angles))
        return np.array([radius * math.cos(yaw), radius * math.sin(yaw), z])

    def tool_pitch(self, q: Any) -> float:
        return float(np.sum(np.asarray(q, dtype=np.float64)[1:]))

    def jacobian(self, q: Any) -> np.ndarray:
        yaw, boom, arm, bucket = (float(v) for v in np.asarray(q, dtype=np.float64))
        angles = np.array([boom, boom + arm, boom + arm + bucket])
        sin_a, cos_a = np.sin(angles), np.cos(angles)
        radius = float(self.links @ cos_a)
        dr = np.array(
            [
                -float(self.links @ sin_a),
                -float(self.links[1:] @ sin_a[1:]),
                -self.links[2] * sin_a[2],
            ]
        )
        dz = np.array(
            [
                float(self.links @ cos_a),
                float(self.links[1:] @ cos_a[1:]),
                self.links[2] * cos_a[2],
            ]
        )
        jac = np.zeros((3, 4), dtype=np.float64)
        jac[:, 0] = [-radius * math.sin(yaw), radius * math.cos(yaw), 0.0]
        for k in range(3):
            jac[:, k + 1] = [dr[k] * math.cos(yaw), dr[k] * math.sin(yaw), dz[k]]
        return jac

    def pose_jacobian(self, q: Any) -> np.ndarray:
        return np.vstack([self.jacobian(q), np.array([0.0, 1.0, 1.0, 1.0])])

    @staticmethod
    def wrap_angle(value: float) -> float:
        return float(math.atan2(math.sin(value), math.cos(value)))

    def step_ik(
        self,
        q: Any,
        target_xyz: Any,
        damping: float,
        *,
        target_pitch: float | None = None,
        pitch_weight: float = 0.2,
    ) -> np.ndarray:
        """一步阻尼最小二乘 IK（限步长、限位），返回**下一拍**的 q（不是最终解）。"""
        q = np.asarray(q, dtype=np.float64)
        if target_pitch is None:
            error = np.asarray(target_xyz, dtype=np.float64) - self.forward(q)
            jac = self.jacobian(q)
        else:
            error = np.r_[
                np.asarray(target_xyz, dtype=np.float64) - self.forward(q),
                pitch_weight * self.wrap_angle(target_pitch - self.tool_pitch(q)),
            ]
            jac = self.pose_jacobian(q)
            jac[-1] *= pitch_weight
        dq = jac.T @ np.linalg.solve(
            jac @ jac.T + damping**2 * np.eye(len(error)), error
        )
        dq = np.clip(dq, -self.max_step, self.max_step)
        return np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])

    def step_ik_fixed(
        self,
        q: Any,
        target_xyz: Any,
        joint_index: int,
        value: float,
        damping: float,
    ) -> np.ndarray:
        """One IK step with one joint pinned (used for the open-bucket dump).

        With the bucket fixed at its open angle, yaw + boom + arm are exactly the
        3 DOF needed to put the tip at a 3D point.
        """
        q = np.asarray(q, dtype=np.float64).copy()
        q[joint_index] = float(value)
        error = np.asarray(target_xyz, dtype=np.float64) - self.forward(q)
        jac = self.jacobian(q)
        cols = [i for i in range(4) if i != joint_index]
        j3 = jac[:, cols]
        dq = j3.T @ np.linalg.solve(j3 @ j3.T + damping**2 * np.eye(3), error)
        step = np.zeros(4, dtype=np.float64)
        step[cols] = np.clip(dq, -self.max_step[cols], self.max_step[cols])
        return np.clip(q + step, self.limits[:, 0], self.limits[:, 1])
