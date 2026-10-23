# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""关节空间平滑轨迹 —— "丝滑"运动的生成端（采集级）。

为什么需要它：工作循环（dig）现在是**逐关键帧阻塞到位**——每段都等机器停稳
再发下一段，速度在每个关键帧掉到 0；用这种参考采集，策略学到的是"停—走—停"
的节奏，拍与拍之间的位移还忽大忽小（到位处几乎为 0）。

这里把同一个循环表达成一条 **C¹ 连续的关节空间曲线**：

  * 三次 Hermite 样条（Catmull-Rom 切线，时间非均匀）穿过所有关键帧，
    并对每一段做 Fritsch–Carlson 单调限幅（同 PCHIP）——段内不过冲、不抖；
  * 端点切线可置零（``rest_ends``）= 平稳起步、平稳停稳；
  * 回转关节按**最短方向**解缠（±π 处不跨圈），不然一转到底会反向绕一圈；
  * 全局时间伸缩把每关节峰值速度压到给定上限（只放慢、不加速）；
  * 可选关节限位裁剪（越界帧夹回并报告），与驱动层的限位守卫同一口径。

产出 = 定频帧序列（默认 60 Hz = AGX 仿真步），交给桥接 ``play_trajectory``
按仿真步播放；同一时钟下的 ``record_*`` 采样就是采集数据本身——参考、执行、
记录三者同一条时间线。

纯 numpy、无 AGX/jiuwensymbiosis 依赖，单测覆盖（tests/test_trajectory.py）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["Trajectory", "build_trajectory", "peak_speeds", "unwrap_revolute"]


@dataclass(frozen=True)
class Trajectory:
    """定频帧序列 + 取证指标。``frames`` 列序 = 调用方给的关节顺序。"""

    times_s: np.ndarray  # (N,) 每帧的仿真时间（从 0 起）
    frames: np.ndarray  # (N, D) 关节目标（原生单位）
    rate_hz: float
    time_scale: float  # 时间伸缩因子（>1 = 为限速整体放慢）
    peak_speed_raw: np.ndarray  # (D,) 限速前的每关节峰值速度 |dq/dt|
    clipped: int = 0  # 被限位夹回的帧数（0 = 未触发）


def unwrap_revolute(values: np.ndarray, joints: Sequence[int]) -> np.ndarray:
    """把回转关节的序列解缠成"相邻差在 ±π 内"的连续序列（走最短方向）。"""
    out = np.array(values, dtype=np.float64, copy=True)
    for j in joints:
        column = out[:, j]
        for k in range(1, len(column)):
            delta = (column[k] - column[k - 1] + math.pi) % (2.0 * math.pi) - math.pi
            column[k] = column[k - 1] + delta
    return out


def _hermite_tangents(
    times: np.ndarray, values: np.ndarray, rest_ends: bool, monotone: bool = True
) -> np.ndarray:
    """Catmull-Rom 切线（时间非均匀中心差分；端点按 rest_ends 取零或单侧）。

    ``monotone=True`` 时再套 Fritsch–Carlson 单调限幅（同 PCHIP）：段内不过冲、
    "该停住的关节"（相邻关键帧相同）切线自动归零——否则中心差分会让已经到位的
    关节继续往前冲一点再退回来（215F 实测：挖点方位多转 0.1 rad 再回）。
    """
    m = np.zeros_like(values)
    if len(times) < 2:
        return m
    if len(times) == 2:
        slope = (values[1] - values[0]) / (times[1] - times[0])
        m[0] = 0.0 if rest_ends else slope
        m[1] = 0.0 if rest_ends else slope
        return m
    for k in range(1, len(times) - 1):
        m[k] = (values[k + 1] - values[k - 1]) / (times[k + 1] - times[k - 1])
    first = (values[1] - values[0]) / (times[1] - times[0])
    last = (values[-1] - values[-2]) / (times[-1] - times[-2])
    m[0] = 0.0 if rest_ends else first
    m[-1] = 0.0 if rest_ends else last
    if not monotone:
        return m
    for k in range(len(times) - 1):
        span = float(times[k + 1] - times[k])
        delta = (values[k + 1] - values[k]) / span  # (D,)
        flat = np.abs(delta) <= 1e-12  # 这一段该关节不动 → 两端切线必须为 0
        safe = np.where(flat, 1.0, delta)
        alpha = np.where(flat, 0.0, m[k] / safe)
        beta = np.where(flat, 0.0, m[k + 1] / safe)
        alpha = np.where(alpha < 0.0, 0.0, alpha)  # 与段斜率反向 = 必然过冲
        beta = np.where(beta < 0.0, 0.0, beta)
        norm = alpha * alpha + beta * beta
        shrink = np.where(norm > 9.0, 3.0 / np.sqrt(np.where(norm > 9.0, norm, 1.0)), 1.0)
        m[k] = np.where(flat, 0.0, shrink * alpha * delta)
        m[k + 1] = np.where(flat, 0.0, shrink * beta * delta)
    return m


def _sample_segment(
    q0: np.ndarray,
    q1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    span: float,
    s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """一段 Hermite：返回 (q(s), dq/dt(s))；span = 该段时长（秒）。"""
    s2 = s * s
    s3 = s2 * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    q = (
        h00[:, None] * q0
        + (h10 * span)[:, None] * m0
        + h01[:, None] * q1
        + (h11 * span)[:, None] * m1
    )
    dh00 = 6 * s2 - 6 * s
    dh10 = 3 * s2 - 4 * s + 1
    dh01 = -6 * s2 + 6 * s
    dh11 = 3 * s2 - 2 * s
    dq = (dh00[:, None] * q0 + dh01[:, None] * q1) / span + dh10[:, None] * m0 + dh11[
        :, None
    ] * m1
    return q, dq


def peak_speeds(
    waypoints: np.ndarray,
    durations_s: Sequence[float],
    rest_ends: bool = True,
    samples: int = 40,
    monotone: bool = True,
) -> np.ndarray:
    """每关节的峰值速度 |dq/dt|（对每段密集采样取最大，限速伸缩的判据）。"""
    times = np.concatenate([[0.0], np.cumsum(np.asarray(durations_s, dtype=np.float64))])
    tangents = _hermite_tangents(times, waypoints, rest_ends, monotone=monotone)
    peak = np.zeros(waypoints.shape[1], dtype=np.float64)
    for k in range(len(times) - 1):
        span = float(times[k + 1] - times[k])
        s = np.linspace(0.0, 1.0, max(int(samples), 2))
        _, dq = _sample_segment(waypoints[k], waypoints[k + 1], tangents[k], tangents[k + 1], span, s)
        peak = np.maximum(peak, np.abs(dq).max(axis=0))
    return peak


def build_trajectory(
    waypoints: Any,
    durations_s: Sequence[float],
    *,
    rate_hz: float = 60.0,
    rest_ends: bool = True,
    monotone: bool = True,
    vmax: Any | None = None,
    unwrap_joints: Sequence[int] = (),
    limits: Any | None = None,
    max_time_scale: float = 8.0,
) -> Trajectory:
    """关键帧 + 分段时长 → 定频帧序列（默认 60 Hz，配合桥接的按仿真步播放）。

    ``waypoints``: (M, D) 关节关键帧（原生单位），或 [dict, ...]（需给 ``names``，
    见 :func:`waypoints_from_dicts`）。``durations_s``: 长度 M-1 的分段时长（秒）。
    ``vmax``: (D,) 每关节速度上限（rad/s 或 m/s）；超过则整体放慢（只减不增）。
    ``unwrap_joints``: 需要按最短方向解缠的列（回转关节）。
    ``limits``: (D, 2) 关节限位；越界帧夹回并计数（与驱动层限位守卫同口径）。
    ``monotone``: 段内单调限幅（默认开）——关掉会回到纯 Catmull-Rom（有过冲）。
    """
    q = np.asarray(waypoints, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] < 2:
        raise ValueError("waypoints 需要 (M>=2, D) 的二维数组")
    durations = np.asarray(durations_s, dtype=np.float64)
    if durations.shape != (q.shape[0] - 1,):
        raise ValueError(
            f"durations_s 需要 {q.shape[0] - 1} 段（M-1），收到 {durations.shape}"
        )
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("durations_s 必须全为有限正数")
    if rate_hz <= 0:
        raise ValueError("rate_hz 必须为正")

    if unwrap_joints:
        q = unwrap_revolute(q, list(unwrap_joints))

    raw_peak = peak_speeds(q, durations, rest_ends=rest_ends, monotone=monotone)
    scale = 1.0
    if vmax is not None:
        vmax_arr = np.asarray(vmax, dtype=np.float64)
        if vmax_arr.shape != (q.shape[1],):
            raise ValueError(f"vmax 需要 {q.shape[1]} 个值，收到 {vmax_arr.shape}")
        over = raw_peak / np.maximum(vmax_arr, 1e-9)
        if over.max() > 1.0:
            scale = float(min(over.max(), float(max_time_scale)))
    durations = durations * scale

    times = np.concatenate([[0.0], np.cumsum(durations)])
    tangents = _hermite_tangents(times, q, rest_ends, monotone=monotone)

    frames: list[np.ndarray] = []
    sample_times: list[float] = []
    step = 1.0 / float(rate_hz)
    for k in range(len(times) - 1):
        span = float(times[k + 1] - times[k])
        count = max(int(math.ceil(span / step)), 1)
        s = np.arange(count, dtype=np.float64) / count  # 不含段终点，避免重复帧
        seg_q, _ = _sample_segment(q[k], q[k + 1], tangents[k], tangents[k + 1], span, s)
        frames.append(seg_q)
        sample_times.append(times[k] + s * span)
    frames.append(q[-1][None, :])
    sample_times.append(np.array([times[-1]]))

    frame_arr = np.vstack(frames)
    clipped = 0
    if limits is not None:
        lim = np.asarray(limits, dtype=np.float64)
        if lim.shape != (q.shape[1], 2):
            raise ValueError(f"limits 需要 (D, 2)，收到 {lim.shape}")
        out = np.clip(frame_arr, lim[:, 0], lim[:, 1])
        clipped = int(np.count_nonzero(np.any(np.abs(out - frame_arr) > 1e-12, axis=1)))
        frame_arr = out
    return Trajectory(
        times_s=np.concatenate(sample_times),
        frames=frame_arr,
        rate_hz=float(rate_hz),
        time_scale=scale,
        peak_speed_raw=raw_peak,
        clipped=clipped,
    )


def waypoints_from_dicts(
    frames: Sequence[dict[str, float]], names: Sequence[str]
) -> np.ndarray:
    """关键帧 dict 列表（按 ``names`` 取列）→ (M, D) 数组；缺键即报错。"""
    rows = []
    for index, frame in enumerate(frames):
        missing = [name for name in names if name not in frame]
        if missing:
            raise ValueError(f"关键帧 #{index} 缺关节 {missing}")
        rows.append([float(frame[name]) for name in names])
    return np.asarray(rows, dtype=np.float64)
