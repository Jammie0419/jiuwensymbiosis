# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""平滑轨迹生成的回归（不需要 AGX）。

被测的是"丝滑"这件事的可验证性质，不是手感：
  - 帧序列定频、首末帧等于关键帧、段内无过冲（单调限幅生效）；
  - 峰值速度受 vmax 约束（超了就整体放慢，且只慢不冲）；
  - 相邻拍速度连续（无速度跳变 = 无顿挫）；
  - 回转关节按最短方向解缠。
"""

from __future__ import annotations

import numpy as np
import pytest

from jiuwen_agx.trajectory import build_trajectory, peak_speeds, unwrap_revolute


# ------------------------------------------------------------------ trajectory


class TestBuildTrajectory:
    def test_rate_endpoints_and_no_overshoot(self):
        waypoints = [[0.0, 0.0], [1.0, -0.5], [1.0, -0.5], [2.0, 0.25]]
        traj = build_trajectory(waypoints, [1.0, 1.0, 1.0], rate_hz=100.0)
        assert traj.times_s[0] == pytest.approx(0.0)
        assert traj.times_s[-1] == pytest.approx(3.0)
        assert np.allclose(traj.frames[0], waypoints[0])
        assert np.allclose(traj.frames[-1], waypoints[-1])
        # 段内不过冲：每列不超出该列关键帧的取值范围
        low = np.min(np.asarray(waypoints, dtype=float), axis=0)
        high = np.max(np.asarray(waypoints, dtype=float), axis=0)
        assert np.all(traj.frames >= low - 1e-12)
        assert np.all(traj.frames <= high + 1e-12)
        # 定频：相邻帧时间间隔恒定
        assert np.allclose(np.diff(traj.times_s), 1 / 100.0, atol=1e-9)

    def test_velocity_continuity(self):
        """相邻拍的速度差必须小（无速度跳变 = 采集数据里没有顿挫）。"""
        waypoints = [[0.0], [1.0], [1.3], [2.4]]
        traj = build_trajectory(waypoints, [0.8, 1.2, 0.6], rate_hz=200.0)
        speed = np.diff(traj.frames[:, 0]) * traj.rate_hz
        jumps = np.abs(np.diff(speed))
        assert jumps.max() < 0.05 * max(speed.max(), 1e-9) + 1e-9

    def test_vmax_slows_never_speeds(self):
        waypoints = [[0.0], [1.0]]
        fast = build_trajectory(waypoints, [0.5], rate_hz=100.0)
        slow = build_trajectory(waypoints, [0.5], rate_hz=100.0, vmax=[0.5])
        assert fast.time_scale == pytest.approx(1.0)
        assert slow.time_scale > 1.0
        # 峰值速度压到上限内（留一点数值余量）
        peak = np.abs(np.diff(slow.frames[:, 0])).max() * slow.rate_hz
        assert peak <= 0.5 * 1.05
        # 帧数随放慢变多，形状不变（首末仍等于关键帧）
        assert slow.frames.shape[0] > fast.frames.shape[0]
        assert np.allclose(slow.frames[0], waypoints[0])
        assert np.allclose(slow.frames[-1], waypoints[-1])

    def test_unwrap_revolute_takes_short_way(self):
        values = np.array([[3.0], [-3.0]])
        wrapped = unwrap_revolute(values, [0])
        # 3.0 → -3.0 的物理短路径是继续向正方向走 0.283 rad
        assert wrapped[1, 0] == pytest.approx(3.0 + (2 * np.pi - 6.0), abs=1e-9)
        traj = build_trajectory([[3.0], [-3.0]], [1.0], rate_hz=50.0, unwrap_joints=[0])
        assert traj.frames[:, 0].max() < 3.5  # 不会反向绕一整圈

    def test_limits_clip_and_report(self):
        traj = build_trajectory(
            [[0.0], [1.0]], [1.0], rate_hz=50.0, limits=[[-0.5, 0.5]]
        )
        assert traj.clipped > 0
        assert traj.frames.min() >= -0.5 - 1e-12 and traj.frames.max() <= 0.5 + 1e-12

    def test_bad_inputs_refused(self):
        with pytest.raises(ValueError, match="durations_s"):
            build_trajectory([[0.0], [1.0]], [1.0, 1.0])
        with pytest.raises(ValueError, match="正数"):
            build_trajectory([[0.0], [1.0]], [0.0])
        with pytest.raises(ValueError, match="vmax"):
            build_trajectory([[0.0], [1.0]], [1.0], vmax=[1.0, 1.0])
        with pytest.raises(ValueError, match="rate_hz"):
            build_trajectory([[0.0], [1.0]], [1.0], rate_hz=0)

    def test_peak_speeds_matches_actual_peak(self):
        waypoints = np.array([[0.0], [1.0]])
        durations = [1.0]
        predicted = peak_speeds(waypoints, durations)[0]
        traj = build_trajectory(waypoints, durations, rate_hz=500.0)
        actual = np.abs(np.diff(traj.frames[:, 0]) * traj.rate_hz).max()
        assert predicted == pytest.approx(actual, rel=0.02)
