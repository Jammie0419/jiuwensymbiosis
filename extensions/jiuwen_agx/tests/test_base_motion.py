# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""底盘到位可观测性（base_state）—— 相对位移不能互相覆盖。

踩过的坑（2026-09-21，AGX 挖掘机）：pump 模式下底盘命令是"发了就走"，桥接设一个
停车时刻立刻返回。9 条 ``rotate_base(0.7)``（规划器按 0.7 rad/命令上限拆出来的）
在 0.13 s 内连发，每条都**覆盖**上一条的停车时刻，实际只走了一次 1.4 s 行驶
≈ 0.7 rad（约 40°），而不是 2π —— 用户看到的是"让它转一圈，只转了一点点"。

修法：桥接暴露 ``base_state``，客户端轮询到停车（与关节到位轮询同一套路）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

_BRIDGE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "agx_bridge_server.py"


def _load_bridge_module():
    spec = importlib.util.spec_from_file_location(
        "agx_bridge_server_under_test", _BRIDGE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pump_adapter(now_s: float = 0.0, deadline: float | None = None):
    """AgxSceneAdapter（pump 模式）+ 一个假仿真时钟（不加载 AGX）。"""
    module = _load_bridge_module()
    adapter = module.AgxSceneAdapter([], {}, None, mode="viewer")
    clock = {"t": now_s}
    adapter._sim = SimpleNamespace(getTimeStamp=lambda: clock["t"])  # noqa: SLF001
    if deadline is not None:
        adapter._drive_goal = {  # noqa: SLF001
            "yaw_target": 1.0,
            "xy_target": None,
            "deadline": deadline,
        }
    return module, adapter, clock


class TestBaseState:
    def test_idle_when_no_goal(self):
        _, adapter, _ = _pump_adapter()
        assert adapter.base_state() == {"busy": False, "remaining_s": 0.0}

    def test_busy_while_yaw_error_remains(self):
        _, adapter, _ = _pump_adapter(now_s=10.0, deadline=40.0)
        state = adapter.base_state()
        assert state["busy"] is True  # 目标 yaw=1.0，当前 0.0（无 AGX 时位姿按 0 报）
        assert state["remaining_s"] > 0.0

    def test_expired_deadline_is_not_busy(self):
        """超时后不能再报 busy —— 否则客户端白等到自己的超时。"""
        _, adapter, _ = _pump_adapter(now_s=100.0, deadline=40.0)
        assert adapter.base_state()["busy"] is False

    def test_demo_scene_has_no_motion(self):
        """演示假机（无仿真）永远不忙，客户端不会白等。"""
        module = _load_bridge_module()
        demo = module.DemoSceneAdapter(["swing"])
        assert demo.base_state() == {"busy": False, "remaining_s": 0.0}


class TestBaseTrackCommand:
    """闭环判据（纯函数）：方向必须让 **yaw 误差减小**。

    实测符号（2026-09-21）：左=+v,右=-v 会让 yaw **减小**；所以"增大 yaw"要
    左=-v, 右=+v —— 旧代码用的是反的那个配对。
    """

    def test_positive_yaw_error_turns_left_track_backward(self):
        module, _ = _load_bridge_module(), None
        reached, left, right = module.base_track_command(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}, {"yaw_target": 0.7, "xy_target": None}
        )
        assert reached is False
        assert left < 0 and right > 0

    def test_negative_yaw_error_reverses(self):
        module = _load_bridge_module()
        _reached, left, right = module.base_track_command(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}, {"yaw_target": -0.7, "xy_target": None}
        )
        assert left > 0 and right < 0

    def test_reached_within_tolerance(self):
        module = _load_bridge_module()
        reached, left, right = module.base_track_command(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.69},
            {"yaw_target": 0.7, "xy_target": None},
        )
        assert reached is True and left == 0.0 and right == 0.0

    def test_yaw_first_then_drive(self):
        """先转后走：yaw 没到位时不前进。"""
        module = _load_bridge_module()
        reached, left, right = module.base_track_command(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            {"yaw_target": 1.0, "xy_target": (2.0, 0.0)},
        )
        assert reached is False and left < 0 < right  # 还在转

    def test_drive_phase_goes_straight(self):
        module = _load_bridge_module()
        reached, left, right = module.base_track_command(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
            {"yaw_target": None, "xy_target": (2.0, 0.0)},
        )
        assert reached is False and left > 0 and right > 0 and left == right

    def test_reached_xy_within_tolerance(self):
        module = _load_bridge_module()
        reached, _left, _right = module.base_track_command(
            {"x_m": 1.98, "y_m": 0.0, "yaw_rad": 0.0},
            {"yaw_target": None, "xy_target": (2.0, 0.0)},
        )
        assert reached is True

    def test_yaw_error_wraps(self):
        """目标 -3.0、当前 +3.0：真实差距是 wrap(-6.0)=+0.283 rad，不是 -6 rad。"""
        module = _load_bridge_module()
        _reached, left, right = module.base_track_command(
            {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 3.0},
            {"yaw_target": -3.0, "xy_target": None},
        )
        # 方向：小幅正向修正（不是反向大转）
        assert left < 0 < right
        # 速度未饱和 → 误差被当成 0.28 rad（若没 wrap，|err|=6 会顶到上限）
        assert right < module.BASE_MAX_TRACK_SPEED


class TestBaseMotionReply:
    """pump 模式的底盘回复必须带 async，客户端才知道要等到位。"""

    def test_pump_mode_reply_is_async_with_eta(self):
        module, adapter, _ = _pump_adapter(now_s=10.0, deadline=40.0)
        session = module.BridgeSession(adapter)
        reply = session._base_motion_reply({"dx_m": 0.0, "dyaw_rad": 0.7})  # noqa: SLF001
        assert reply["async"] is True
        assert reply["eta_s"] > 0.0
        assert reply["result"]["dyaw_rad"] == 0.7

    def test_headless_reply_has_no_async(self):
        """headless 模式本来就阻塞到走完，不需要客户端轮询。"""
        module = _load_bridge_module()
        adapter = module.AgxSceneAdapter([], {}, None, mode="headless")
        session = module.BridgeSession(adapter)
        reply = session._base_motion_reply({"dx_m": 1.0, "dyaw_rad": 0.0})  # noqa: SLF001
        assert "async" not in reply

    def test_base_state_command_is_routed(self):
        module, adapter, _ = _pump_adapter()
        session = module.BridgeSession(adapter)
        resp = session.handle({"v": module.PROTOCOL_VERSION, "cmd": "base_state"})
        assert resp["ok"] is True and resp["busy"] is False

    def test_base_pose_command_is_routed(self):
        module, adapter, _ = _pump_adapter()
        session = module.BridgeSession(adapter)
        resp = session.handle({"v": module.PROTOCOL_VERSION, "cmd": "base_pose"})
        assert resp["ok"] is True and set(resp["pose"]) == {"x_m", "y_m", "yaw_rad"}
