# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AgxExcavatorConfig — the excavator's slice of the shared sim config.

Only what makes an excavator AN EXCAVATOR lives here: the work-tool reach
envelope and the dig-cycle keyframe tuning. Everything else (backend, joints,
limits, undercarriage, terrain, camera) comes from
:class:`~jiuwen_agx.sim.config.SimMachineConfig`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from jiuwen_agx.sim.config import SimMachineConfig

__all__ = ["AgxExcavatorConfig"]


def _joint_frame(value: Any, *, where: str, allowed: set[str]) -> dict[str, float]:
    """Validate one optional work-cycle frame: {joint: finite native value}."""
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{where} must be a non-empty mapping of joint -> value")
    frame: dict[str, float] = {}
    for key, raw in value.items():
        name = str(key)
        if name not in allowed:
            raise ValueError(f"{where}: unknown joint {name!r} (allowed: {sorted(allowed)})")
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}.{name} must be a number, got {raw!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{where}.{name} must be finite, got {raw!r}")
        frame[name] = number
    return frame


@dataclass
class AgxExcavatorConfig(SimMachineConfig):
    """履带挖掘机（回转 + 动臂 + 斗杆 + 铲斗）配置。"""

    # --- 覆盖通用默认：挖掘机出厂即 4 工作关节 + 履带底盘
    name: str = "agx_excavator"
    joint_names: tuple[str, ...] = ("swing", "boom", "arm", "bucket")
    has_base: bool = True
    base_step_limits: tuple[float, float] | None = (1.0, 0.7)  # 每命令 ≤1 m / ≤0.7 rad
    # 占位限位（deg）：接入 AGX 模型后按探针实测改；须覆盖 dig_cycle_tuning
    # 的全部关键帧（driver 会在执行前校验）。swing 视回转支承行程。
    joint_limits: dict[str, tuple[float, float]] | None = field(
        default_factory=lambda: {
            "swing": (-180.0, 180.0),
            "boom": (-45.0, 60.0),
            "arm": (-135.0, 60.0),
            "bucket": (-160.0, 40.0),
        }
    )
    # 安全收拢姿态（走行/转场）。
    home_joints: dict[str, float] | None = field(
        default_factory=lambda: {
            "swing": 0.0,
            "boom": -20.0,
            "arm": 25.0,
            "bucket": 10.0,
        }
    )

    # ==================== 挖掘工作参数 ====================
    # 铲齿可达半径包络（基座系，米）：dig 前置校验，超出即拒绝并提示先走底盘。
    reach_min_m: float = 1.0
    reach_max_m: float = 6.0
    # swing 关节的角度单位（"deg" 或 "rad"）。AGX 模型的回转是弧度、液压缸是
    # 米（混合单位，joint_units 留空），此时 swing_unit 必须填 "rad"。
    swing_unit: str = "deg"
    # 关键帧微调，键集见 work.DEFAULT_DIG_TUNING；YAML 里只写要改的键。
    # 注意：boom/arm/bucket 关键帧的值 = 该关节的原生单位（deg 机体是角度，
    # AGX 液压缸机体是米——local.yaml 里按探针实测填米值）。
    dig_cycle_tuning: dict[str, float] | None = None

    # 可选的自然挖掘段；不写 = 保持 365 原来的六段行为。
    #   dig_travel:      起步抬升姿态（先离地再摆转，避免沿途拖土）
    #   dig_cut_path:    切入段逐点（[{boom,arm,bucket}, ...]，原生单位），
    #                    替代 dig/curl 两帧，形成"拖切"而不是两点一跳
    #   dig_carry:       收斗举升姿态（满载搬运，保持斗角，避免摆转时撒土）
    #   dig_dump_dwell_s:卸料驻留秒数（让土流光）
    dig_travel: dict[str, float] | None = None
    dig_cut_path: list[dict[str, float]] | None = None
    dig_carry: dict[str, float] | None = None
    dig_dump_dwell_s: float = 0.0

    # ==================== 3D 位姿挖掘（可选） ====================
    # keyframe: 现有六段/自然段；ik: 用 3D 铲尖位姿 + IK 执行（215F）。
    # 3D 坐标是当前基座系（REP-103，米）；z 是铲尖高度，不是地面高度。
    dig_mode: str = "keyframe"
    dig_ik_machine: str | None = None  # machine_<机型>.json（IK 几何/符号/offset）
    reach_z_min_m: float | None = None
    reach_z_max_m: float | None = None
    # IK waypoint 参数
    approach_gap_m: float = 0.35  # 接近点在目标上方的余量
    bite_m: float = 0.25  # 从大脑给的 dig_z 再向下压的深度
    cut_length_m: float = 1.0  # 拖切沿半径向内的长度
    # 拖切终点离基座的最近半径（米）：挖点半径 - cut_length_m 必须 >= 这个值。
    # 0 = 不检查（保持 365 keyframe 行为）。215F 实测：拖到 2.1 m 时 arm 关节已到
    # 下限 0，IK 会原地卡在 0.45 m；挖点太近时先退底盘或换远点。
    dig_min_drag_radius_m: float = 0.0
    carry_lift_m: float = 1.0  # 收斗后抬升高度
    # 铲斗角（原生 rad）：入土/拖切时张开（大角），收斗时往机身内侧合拢
    # （小角），卸料时再往外翻（大角）。0.0 是往内翻的一端，不是卸料方向。
    dig_bucket_open_rad: float = 1.5
    dig_bucket_curl_rad: float = 0.1
    dump_bucket_open_rad: float | None = None  # 卸料外翻角；None = 机器文件
    # 未显式给 dump_z_m 时，卸点高度 = 该点地形高度 + 这个余量。地面高度直接
    # 卸料时铲斗张不开（tip 在开斗状态下到不了低点），所以默认抬高。
    dump_clearance_m: float = 3.0
    ik_tolerance_m: float = 0.05
    # 卸点到达容差：远侧卸点（摆转接近 180°）的模型残差可能到 ~0.2 m，
    # 按挖点同样 0.10 m 会“到不了就抛错”，连开斗都执行不到。卸点单独放宽；
    # 到达后仍会原地外翻开斗并检查载料。
    dump_tolerance_m: float = 0.25
    # 接触段（press/drag/curl）容差：土硬时铲尖压不到命令深度，按实际入土
    # 深度继续拖切，不把“咬不深”当成“到不了大脑给的点”。
    contact_tolerance_m: float = 0.25
    ik_max_iters: int = 80
    ik_stall_iters: int = 8

    # ==================== 学习策略（policy.act 能力） ====================
    # 配置了才声明 policy.act 并暴露 act_exec；未配置 = 词表里没有这个动作。
    # name 对应 jiuwen_agx.policy.POLICIES 的注册名（"act"=lerobot ACT checkpoint，
    # "fake"=无依赖假策略，测试/演示用）；其余键由各策略类自取（act 要 ckpt/device）。
    # 注意：from_dict 会静默丢弃未知键——这个字段必须存在，YAML 的 policy 节才生效。
    policy: dict[str, Any] | None = None
    # act_exec 执行参数：单拍下发超时（数据采集与执行用同一常数，见接入计划 §4.5）。
    policy_beat_timeout_s: float = 1.0
    # act_exec 兜底拍数上限：超过即判失败（取"实测教师拍数 × 2"，先给占位值）。
    policy_max_beats: int = 300

    @classmethod
    def from_dict(cls, data):  # type: ignore[override]
        cfg = super().from_dict(data)
        if cfg.swing_unit not in ("deg", "rad"):
            raise ValueError(
                f"{cls.__name__}: swing_unit must be 'deg' or 'rad', got {cfg.swing_unit!r}"
            )
        if cfg.dig_cycle_tuning:
            cfg.dig_cycle_tuning = {
                str(k): float(v) for k, v in cfg.dig_cycle_tuning.items()
            }
        allowed = set(cfg.joint_names)
        if cfg.dig_travel is not None:
            cfg.dig_travel = _joint_frame(cfg.dig_travel, where="dig_travel", allowed=allowed)
        if cfg.dig_carry is not None:
            cfg.dig_carry = _joint_frame(cfg.dig_carry, where="dig_carry", allowed=allowed)
        if cfg.dig_cut_path is not None:
            if not isinstance(cfg.dig_cut_path, list):
                raise ValueError(
                    f"{cls.__name__}: dig_cut_path must be a list of joint frames, "
                    f"got {type(cfg.dig_cut_path).__name__}"
                )
            cfg.dig_cut_path = [
                _joint_frame(frame, where=f"dig_cut_path[{index}]", allowed=allowed)
                for index, frame in enumerate(cfg.dig_cut_path)
            ]
        cfg.dig_dump_dwell_s = float(cfg.dig_dump_dwell_s)
        if cfg.dig_dump_dwell_s < 0:
            raise ValueError(
                f"{cls.__name__}: dig_dump_dwell_s must be >= 0, got {cfg.dig_dump_dwell_s}"
            )
        cfg.dig_mode = str(cfg.dig_mode).strip().lower()
        if cfg.dig_mode not in ("keyframe", "ik"):
            raise ValueError(
                f"{cls.__name__}: dig_mode must be 'keyframe' or 'ik', got {cfg.dig_mode!r}"
            )
        if cfg.dig_ik_machine is not None:
            cfg.dig_ik_machine = str(cfg.dig_ik_machine)
        if cfg.dig_mode == "ik":
            if not cfg.dig_ik_machine:
                raise ValueError(
                    f"{cls.__name__}: dig_mode='ik' needs dig_ik_machine "
                    "(a machine_<id>.json)"
                )
            # The IK cycle seeds itself from the calibrated ready/dig frames;
            # the generic defaults are 365 keyframes in degrees, not native rad.
            needed = (
                "ready_boom_deg",
                "ready_arm_deg",
                "ready_bucket_deg",
                "dig_boom_deg",
                "dig_arm_deg",
                "dig_bucket_deg",
            )
            missing = [
                key for key in needed if key not in (cfg.dig_cycle_tuning or {})
            ]
            if missing:
                raise ValueError(
                    f"{cls.__name__}: dig_mode='ik' needs native-rad "
                    f"dig_cycle_tuning keys {missing}"
                )
        for name in (
            "approach_gap_m",
            "bite_m",
            "cut_length_m",
            "dig_min_drag_radius_m",
            "carry_lift_m",
            "dig_bucket_open_rad",
            "dig_bucket_curl_rad",
            "dump_clearance_m",
            "ik_tolerance_m",
            "dump_tolerance_m",
            "contact_tolerance_m",
        ):
            setattr(cfg, name, float(getattr(cfg, name)))
        for name in ("ik_max_iters", "ik_stall_iters"):
            setattr(cfg, name, int(getattr(cfg, name)))
        if cfg.dump_clearance_m < 0:
            raise ValueError(
                f"{cls.__name__}: dump_clearance_m must be >= 0, got {cfg.dump_clearance_m}"
            )
        if cfg.dig_min_drag_radius_m < 0:
            raise ValueError(
                f"{cls.__name__}: dig_min_drag_radius_m must be >= 0, "
                f"got {cfg.dig_min_drag_radius_m}"
            )
        if cfg.dump_tolerance_m < 0:
            raise ValueError(
                f"{cls.__name__}: dump_tolerance_m must be >= 0, "
                f"got {cfg.dump_tolerance_m}"
            )
        if cfg.contact_tolerance_m < 0:
            raise ValueError(
                f"{cls.__name__}: contact_tolerance_m must be >= 0, "
                f"got {cfg.contact_tolerance_m}"
            )
        if cfg.dig_bucket_curl_rad > cfg.dig_bucket_open_rad:
            raise ValueError(
                f"{cls.__name__}: dig_bucket_curl_rad "
                f"({cfg.dig_bucket_curl_rad}) must be <= dig_bucket_open_rad "
                f"({cfg.dig_bucket_open_rad}): the cut runs open, then curls in"
            )
        for name in ("reach_z_min_m", "reach_z_max_m", "dump_bucket_open_rad"):
            raw = getattr(cfg, name)
            if raw is not None:
                setattr(cfg, name, float(raw))
        if (
            cfg.reach_z_min_m is not None
            and cfg.reach_z_max_m is not None
            and cfg.reach_z_min_m > cfg.reach_z_max_m
        ):
            raise ValueError(
                f"{cls.__name__}: reach_z_min_m must be <= reach_z_max_m"
            )
        if cfg.policy is not None:
            if (
                not isinstance(cfg.policy, dict)
                or not str(cfg.policy.get("name", "")).strip()
            ):
                raise ValueError(
                    f"{cls.__name__}: policy must be a mapping with a non-empty "
                    f"'name' (a jiuwen_agx.policy.POLICIES registration), got {cfg.policy!r}"
                )
            cfg.policy = {str(k): v for k, v in cfg.policy.items()}
        cfg.policy_beat_timeout_s = float(cfg.policy_beat_timeout_s)
        cfg.policy_max_beats = int(cfg.policy_max_beats)
        return cfg
