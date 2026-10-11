# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""每机器标定文件的加载与单位换算（适配器与数据转换器共用）。

三层分离里的"机器层"：``goal-act-v2`` 的模型世界是"四个关节角（弧度）"，
本框架的机器世界是每台挖掘机自己的原生单位（AGX 365：1 个 Hinge 弧度 +
3 个 Prismatic 米）。换算参数（sign/offset、位移↔等效角、goal 的 z 高度）
全部住在一份 ``machine_<机型>.json`` 里（schema 见
``configs/machine_agx365.template.json``），换机型只换文件，代码零改动。

标定状态 ``calibrated`` 是数据质量门：转换器默认拒绝未标定文件（防止米值
被当角度写进训练数据）；适配器允许加载但打警告（链路干跑用）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

__all__ = ["JointMap", "MachineMap", "load_machine_file"]

_BISECTION_STEPS = 48


def load_machine_file(path: str | Path) -> dict[str, Any]:
    """Read a machine calibration JSON, validating the pieces every user needs."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "joints" not in data:
        raise ValueError(
            f"machine file {path} must be a JSON object with a 'joints' list"
        )
    order = data.get("policy_joint_order")
    if not isinstance(order, list) or not order:
        raise ValueError(
            f"machine file {path} must declare a non-empty 'policy_joint_order'"
        )
    names = [str(j.get("name", "")) for j in data["joints"]]
    missing = [name for name in order if name not in names]
    if missing:
        raise ValueError(
            f"machine file {path} lacks joints {missing} from policy_joint_order"
        )
    return data


class JointMap:
    """One joint's native unit <-> the policy's angle space (radians).

    ``native_unit == "rad"`` (a revolute joint): ``q_policy = sign * native +
    offset_rad`` directly. ``native_unit == "m"`` (a hydraulic cylinder): the
    displacement goes through the machine file's conversion — linear between
    (min_displacement_m -> min_angle_deg) and (max_displacement_m ->
    max_angle_deg), or an optional degree-valued polynomial over the
    normalised displacement ``dn = (d - min_d) / (max_d - min_d)``:
    ``angle_deg = sum(coeff_i * dn**i)``. The polynomial inverse is monotone
    bisection, so any monotone fit works.
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        self.name = str(spec["name"])
        self.sign = float(spec.get("sign", 1.0))
        if self.sign == 0.0:
            raise ValueError(f"joint {self.name!r}: sign must be non-zero")
        self.offset_rad = float(spec.get("offset_rad", 0.0))
        self.native_unit = str(spec.get("native_unit", "rad"))
        self.conv = spec.get("hydraulic_conversion")
        if self.native_unit not in ("rad", "m"):
            raise ValueError(f"joint {self.name!r}: native_unit must be 'rad' or 'm'")
        if self.native_unit == "m":
            if not isinstance(self.conv, dict):
                raise ValueError(
                    f"joint {self.name!r}: native_unit 'm' needs a 'hydraulic_conversion' block"
                )
            required = (
                "min_displacement_m",
                "max_displacement_m",
                "min_angle_deg",
                "max_angle_deg",
            )
            missing = [key for key in required if key not in self.conv]
            if missing:
                raise ValueError(
                    f"joint {self.name!r}: hydraulic_conversion missing {missing}"
                )
            if self.conv["max_displacement_m"] <= self.conv["min_displacement_m"]:
                raise ValueError(
                    f"joint {self.name!r}: displacement range must be increasing"
                )

    # ---- per-joint conversion -------------------------------------------------
    def to_policy(self, native: float) -> float:
        """Native unit (m or rad) -> policy angle in radians."""
        value = float(native)
        if self.native_unit == "rad":
            angle_rad = value
        else:
            angle_rad = math.radians(self._displacement_to_angle_deg(value))
        return self.sign * angle_rad + self.offset_rad

    def to_native(self, policy_angle_rad: float) -> float:
        """Policy angle in radians -> native unit (m or rad)."""
        angle_rad = (float(policy_angle_rad) - self.offset_rad) / self.sign
        if self.native_unit == "rad":
            return angle_rad
        return self._angle_deg_to_displacement(math.degrees(angle_rad))

    # ---- displacement <-> angle ----------------------------------------------
    def _displacement_to_angle_deg(self, displacement_m: float) -> float:
        conv = self.conv
        lo_d = float(conv["min_displacement_m"])
        hi_d = float(conv["max_displacement_m"])
        dn = (displacement_m - lo_d) / (hi_d - lo_d)
        coeffs = conv.get("poly_coeffs")
        if coeffs:
            return float(sum(float(c) * dn**i for i, c in enumerate(coeffs)))
        lo_a = float(conv["min_angle_deg"])
        hi_a = float(conv["max_angle_deg"])
        return lo_a + dn * (hi_a - lo_a)

    def _angle_deg_to_displacement(self, angle_deg: float) -> float:
        conv = self.conv
        lo_d = float(self.conv["min_displacement_m"])
        hi_d = float(conv["max_displacement_m"])
        coeffs = conv.get("poly_coeffs")
        if not coeffs:
            lo_a = float(conv["min_angle_deg"])
            hi_a = float(conv["max_angle_deg"])
            if hi_a == lo_a:
                raise ValueError(
                    f"joint {self.name!r}: angle range must be non-degenerate"
                )
            dn = (angle_deg - lo_a) / (hi_a - lo_a)
            return lo_d + dn * (hi_d - lo_d)
        # Monotone polynomial: bisection on the normalised displacement.
        low, high = 0.0, 1.0
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (low + high)
            value = sum(float(c) * mid**i for i, c in enumerate(coeffs))
            if value < angle_deg:
                low = mid
            else:
                high = mid
        return lo_d + 0.5 * (low + high) * (hi_d - lo_d)


class MachineMap:
    """The whole body: joint order + per-joint maps + goal heights."""

    def __init__(self, data: dict[str, Any], *, source: str | Path = "?") -> None:
        self.data = data
        self.source = str(source)
        self.machine = str(data.get("machine", "unknown"))
        self.calibrated = bool(data.get("calibrated", False))
        self.joint_order = tuple(str(name) for name in data["policy_joint_order"])
        specs = {str(j["name"]): j for j in data["joints"]}
        self.joints = {name: JointMap(specs[name]) for name in self.joint_order}
        self.pick_z_m = float(data.get("pick_z_m", 0.0))
        self.dump_z_m = float(data.get("dump_z_m", 0.0))

    @classmethod
    def load(cls, path: str | Path) -> MachineMap:
        return cls(load_machine_file(path), source=path)
