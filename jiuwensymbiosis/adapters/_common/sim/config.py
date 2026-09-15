# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SimMachineConfig — shared configuration for every simulated machine.

One config class for the whole sim family; a per-machine package subclasses it
only to add that machine's work-cycle parameters and friendlier defaults (see
``adapters/agx_excavator/config.py``). Everything here is machine-agnostic:
which backend, which joints, what the undercarriage may do per command, where
the terrain truth comes from.

``from_dict`` accepts the FULL config YAML dict (the nested ``env.cfg.low_level``
shape real YAMLs use) or a flat dict — the piper convention.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

KNOWN_BACKENDS = ("mock", "inprocess", "remote")


@dataclass
class SimMachineConfig:
    """Configuration for one simulated machine (AGX-backed)."""

    # ==================== 基本信息 ====================
    name: str = "sim_machine"

    # ==================== 仿真器连接 ====================
    # mock: 内存模拟（离线开发/测试，零依赖）；inprocess: 本进程 import agx（阶段B）；
    # remote: TCP 桥接 scripts/agx_bridge_server.py（可跑在 AGX 自带 Python 里）。
    backend: Literal["mock", "inprocess", "remote"] = "mock"
    scene_path: str | None = None  # [选填-仅 inprocess] 加载的 .agx 场景文件
    host: str = "127.0.0.1"  # [选填-仅 remote] 桥接服务地址
    port: int = 9700  # [选填-仅 remote] 桥接服务端口
    startup_timeout_s: float = 60.0  # 建立仿真连接的超时
    move_timeout_s: float = 30.0  # 单次关节/底盘命令的默认到位超时

    # ==================== 关节 ====================
    # 关节名（链序）；"键顺序 = q 索引顺序"。每台机器的 YAML 必填。
    joint_names: tuple[str, ...] = ()
    # 关节单位：deg / rad。必须与 joint_limits、观测值一致（会进世界状态）。
    joint_units: str = "deg"
    # 关节软限位 {名: (低, 高)}（SafetyRail 读取；None = 不做限位检查）。
    joint_limits: dict[str, tuple[float, float]] | None = None
    # home 关节角（安全姿态）；None = home 不动。
    home_joints: dict[str, float] | None = None

    # ==================== 履带/行走底盘 ====================
    has_base: bool = False  # True 才声明 motion.base（navigate_relative 等）
    # 底盘每命令包络 (最大位移 m, 最大转角 rad)（SafetyRail 读取；None = 不限）。
    base_step_limits: tuple[float, float] | None = None

    # ==================== 地形真值 ====================
    terrain_enabled: bool = True  # False 则不声明 sensing.terrain（get_terrain 不可见）
    # mock 后端报告的料堆 [{name, x_m, y_m, volume_m3}]；AGX 后端读仿真真值，忽略此项。
    terrain_piles: list[dict[str, Any]] | None = None

    # ==================== 能力扩展 ====================
    # 机器族工作能力（如 "motion.excavator"），写进实例能力集合（须已在
    # KNOWN_CAPABILITIES 注册）。通用能力（motion.joint/base、sensing.terrain、
    # vision.*）由上面的开关推导，不要写在这里。
    extra_capabilities: tuple[str, ...] = ()

    # ==================== 相机（阶段B） ====================
    camera_enabled: bool = False  # True 才声明 vision.* 并要求后端出帧
    camera_resolution: tuple[int, int] = (640, 480)

    # ========================================================================
    #  Loaders — framework contract, do NOT modify lightly
    # ========================================================================

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimMachineConfig:
        """Accept the full config YAML dict (nested ``env.cfg.low_level``) or a flat one.

        Only keys matching dataclass field names are used; extra keys are
        silently ignored.
        """
        env = data.get("env") if isinstance(data.get("env"), dict) else None
        cfg_node = env.get("cfg") if isinstance(env, dict) else None
        ll = cfg_node.get("low_level") if isinstance(cfg_node, dict) else None
        if isinstance(ll, dict) and ll:
            kw: dict[str, Any] = {k: v for k, v in ll.items() if not k.startswith("_")}
        else:
            kw = dict(data)

        if str(kw.get("backend", "mock")) not in KNOWN_BACKENDS:
            raise ValueError(f"{cls.__name__}: backend must be one of {KNOWN_BACKENDS}, got {kw.get('backend')!r}")
        if kw.get("joint_units") not in (None, "deg", "rad"):
            raise ValueError(f"{cls.__name__}: joint_units must be 'deg' or 'rad', got {kw['joint_units']!r}")

        if "joint_names" in kw and isinstance(kw["joint_names"], list):
            kw["joint_names"] = tuple(str(n) for n in kw["joint_names"])
        if "extra_capabilities" in kw and isinstance(kw["extra_capabilities"], list):
            kw["extra_capabilities"] = tuple(str(c) for c in kw["extra_capabilities"])
        if "base_step_limits" in kw and isinstance(kw["base_step_limits"], (list, tuple)):
            if len(kw["base_step_limits"]) == 2:
                kw["base_step_limits"] = (float(kw["base_step_limits"][0]), float(kw["base_step_limits"][1]))
            else:
                kw["base_step_limits"] = None
        if "camera_resolution" in kw and isinstance(kw["camera_resolution"], list):
            kw["camera_resolution"] = tuple(kw["camera_resolution"])
        if "joint_limits" in kw:
            kw["joint_limits"] = _normalise_limits(kw["joint_limits"])
        if "home_joints" in kw and isinstance(kw["home_joints"], dict):
            kw["home_joints"] = {str(k): float(v) for k, v in kw["home_joints"].items()}

        valid = {f.name for f in dataclasses.fields(cls)}
        clean = {k: v for k, v in kw.items() if k in valid}
        return cls(**clean)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SimMachineConfig:
        """Load config from a YAML file (the full nested YAML or a flat one)."""
        path = Path(path).resolve()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


def _normalise_limits(raw: Any) -> dict[str, tuple[float, float]] | None:
    """Coerce YAML ``{名: [低, 高]}`` into ``{名: (float, float)}``; drop bad rows."""
    if not isinstance(raw, dict):
        return None
    normalised: dict[str, tuple[float, float]] = {}
    for k, v in raw.items():
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            continue
        try:
            normalised[str(k)] = (float(v[0]), float(v[1]))
        except (TypeError, ValueError):
            continue
    return normalised or None
