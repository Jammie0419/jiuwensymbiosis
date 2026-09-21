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


def _pump_adapter(now_s: float = 0.0, stop_at: float | None = None):
    """AgxSceneAdapter（pump 模式）+ 一个假仿真时钟（不加载 AGX）。"""
    module = _load_bridge_module()
    adapter = module.AgxSceneAdapter([], {}, None, mode="viewer")
    clock = {"t": now_s}
    adapter._sim = SimpleNamespace(getTimeStamp=lambda: clock["t"])  # noqa: SLF001
    if stop_at is not None:
        adapter._drive_plan = ([], stop_at)  # noqa: SLF001
    return module, adapter, clock


class TestBaseState:
    def test_idle_when_no_plan(self):
        _, adapter, _ = _pump_adapter()
        assert adapter.base_state() == {"busy": False, "remaining_s": 0.0}

    def test_busy_reports_remaining_time(self):
        _, adapter, _ = _pump_adapter(now_s=10.0, stop_at=11.4)
        state = adapter.base_state()
        assert state["busy"] is True
        assert abs(state["remaining_s"] - 1.4) < 1e-6

    def test_elapsed_plan_is_not_busy(self):
        """停车时刻已过（泵还没清标志）也不能报 busy —— 否则客户端白等到超时。"""
        _, adapter, _ = _pump_adapter(now_s=12.0, stop_at=11.4)
        assert adapter.base_state()["busy"] is False

    def test_demo_scene_has_no_motion(self):
        """演示假机（无仿真）永远不忙，客户端不会白等。"""
        module = _load_bridge_module()
        demo = module.DemoSceneAdapter(["swing"])
        assert demo.base_state() == {"busy": False, "remaining_s": 0.0}


class TestBaseMotionReply:
    """pump 模式的底盘回复必须带 async，客户端才知道要等到位。"""

    def test_pump_mode_reply_is_async_with_eta(self):
        module, adapter, _ = _pump_adapter(now_s=10.0, stop_at=11.4)
        session = module.BridgeSession(adapter)
        reply = session._base_motion_reply({"dx_m": 0.0, "dyaw_rad": 0.7})  # noqa: SLF001
        assert reply["async"] is True
        assert abs(reply["eta_s"] - 1.4) < 1e-6
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
