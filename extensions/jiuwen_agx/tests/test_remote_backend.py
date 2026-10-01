# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""RemoteSimBackend protocol tests — the layer that will face the real AGX bridge.

Two harnesses:
  1. An in-process threaded bridge (reuses ``BridgeSession`` from
     ``scripts/agx_bridge_server.py`` via file-location import, like test_smoke).
  2. A scripted canned-response fake, for version-mismatch / error paths the
     demo bridge cannot produce.
Plus one subprocess end-to-end test: backend <-> ``agx_bridge_server.py --demo``.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from jiuwen_agx.agx_excavator import build_agx_excavator_session
from jiuwen_agx.sim.backend import RemoteSimBackend

_REPO = Path(__file__).resolve().parents[1]
_EXT = Path(__file__).resolve().parents[1]
_BRIDGE_PATH = _EXT / "scripts" / "agx_bridge_server.py"


def _load_bridge_module():
    spec = importlib.util.spec_from_file_location(
        "agx_bridge_server_under_test", _BRIDGE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# In-process threaded bridge serving BridgeSession
# ---------------------------------------------------------------------------
class _ThreadedBridge:
    """Serve one connection at a time with the REAL BridgeSession logic."""

    def __init__(self, scene) -> None:
        module = _load_bridge_module()
        self._session = module.BridgeSession(scene)
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        self._server.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except TimeoutError:
                continue
            with conn:
                reader = conn.makefile("r", encoding="utf-8", newline="\n")
                writer = conn.makefile("w", encoding="utf-8", newline="\n")
                for line in reader:
                    if self._stop.is_set():
                        return
                    try:
                        request = json.loads(line)
                    except json.JSONDecodeError:
                        response = {"v": 1, "ok": False, "error": "bad json"}
                    else:
                        if request.get("cmd") == "bye":
                            break
                        response = self._session.handle(request)
                    writer.write(json.dumps(response) + "\n")
                    writer.flush()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._server.close()


class _ScriptedFake:
    """Canned responses for paths the demo bridge cannot produce."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = iter(responses)
        self.seen: list[str] = []  # 收到的命令序列（断言"有没有轮询"）
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        self._server.settimeout(2)
        try:
            conn, _ = self._server.accept()
        except TimeoutError:
            return
        with conn:
            reader = conn.makefile("r", encoding="utf-8", newline="\n")
            writer = conn.makefile("w", encoding="utf-8", newline="\n")
            for line in reader:
                try:
                    self.seen.append(str(json.loads(line).get("cmd", "")))
                except json.JSONDecodeError:
                    self.seen.append("<bad json>")
                try:
                    response = next(self._responses)
                except StopIteration:
                    return
                writer.write(json.dumps(response) + "\n")
                writer.flush()


@pytest.fixture()
def bridge():
    module = _load_bridge_module()
    scene = module.DemoSceneAdapter(["swing", "boom", "arm", "bucket"])
    server = _ThreadedBridge(scene)
    yield server
    server.close()


def _backend(port: int) -> RemoteSimBackend:
    cfg = SimpleNamespace(
        host="127.0.0.1", port=port, startup_timeout_s=5.0, move_timeout_s=5.0
    )
    return RemoteSimBackend(cfg)


# ---------------------------------------------------------------------------
class TestRemoteBackendProtocol:
    def test_open_ping_and_version_check(self, bridge):
        backend = _backend(bridge.port)
        backend.open()
        backend.close()

    def test_version_mismatch_refused(self):
        fake = _ScriptedFake([{"v": 99, "ok": True, "pong": True}])
        backend = _backend(fake.port)
        with pytest.raises(RuntimeError, match="version"):
            backend.open()
        backend.close()

    def test_error_response_raises(self):
        # open() 现在会先 ping、再 inventory（认对端身份，见 _report_peer），
        # 所以脚本里三条响应：ping → inventory → 出错的那条。
        fake = _ScriptedFake(
            [
                {"v": 1, "ok": True, "pong": True},
                {"v": 1, "ok": False, "error": "inventory unsupported"},
                {"v": 1, "ok": False, "error": "machine on fire"},
            ]
        )
        backend = _backend(fake.port)
        backend.open()
        with pytest.raises(RuntimeError, match="machine on fire"):
            backend.read_joints()
        backend.close()

    def test_open_warns_when_peer_is_demo_machine(self, bridge, caplog):
        """连到内存演示机必须在 open() 就喊出来——"连上了但不动"的头号原因。"""
        backend = _backend(bridge.port)
        with caplog.at_level("WARNING", logger="jiuwen_agx.sim.backend"):
            backend.open()
        backend.close()
        assert any(
            "demo_excavator" in rec.message and "演示机" in rec.message
            for rec in caplog.records
        )

    def test_open_ignores_unknown_inventory(self, caplog):
        """老桥不认 inventory 时只记 debug，不能挡住连接。"""
        fake = _ScriptedFake(
            [
                {"v": 1, "ok": True, "pong": True},
                {"v": 1, "ok": False, "error": "unknown command 'inventory'"},
            ]
        )
        backend = _backend(fake.port)
        with caplog.at_level("WARNING", logger="jiuwen_agx.sim.backend"):
            backend.open()  # 不抛
        backend.close()
        assert not [r for r in caplog.records if r.levelno >= 30]

    def test_joint_round_trip(self, bridge):
        backend = _backend(bridge.port)
        backend.open()
        state = backend.send_joint_targets({"swing": 42.0}, timeout_s=5.0)
        assert state["swing"] == 42.0
        assert backend.read_joints()["swing"] == 42.0
        backend.close()

    def test_unmentioned_joints_hold(self, bridge):
        backend = _backend(bridge.port)
        backend.open()
        backend.send_joint_targets({"swing": 10.0, "boom": 20.0}, timeout_s=5.0)
        backend.send_joint_targets({"swing": 30.0}, timeout_s=5.0)
        joints = backend.read_joints()
        assert joints["swing"] == 30.0
        assert joints["boom"] == 20.0

    def test_navigate_and_terrain_and_scoop(self, bridge):
        backend = _backend(bridge.port)
        backend.open()
        assert backend.navigate_relative(0.5, 0.1, timeout_s=5.0) == {
            "dx_m": 0.5,
            "dyaw_rad": 0.1,
        }
        assert backend.navigate_arc(2.0, 0.3, timeout_s=5.0) == {
            "radius_m": 2.0,
            "dyaw_rad": 0.3,
        }
        piles = backend.read_terrain()
        assert piles and {"name", "x_m", "y_m", "volume_m3"} <= set(piles[0])
        assert backend.scoop_state() is False
        backend.mark_scoop(True)
        assert backend.scoop_state() is True
        backend.close()

    def test_grab_frames_is_none_without_camera(self, bridge):
        backend = _backend(bridge.port)
        backend.open()
        assert backend.grab_frames() is None
        backend.close()

    def test_requires_open(self):
        backend = _backend(1)
        with pytest.raises(RuntimeError, match="open"):
            backend.read_joints()

    def test_inventory_command(self, bridge):
        backend = _backend(bridge.port)
        backend.open()
        # inventory is additive in v1; the demo bridge implements it
        report = backend._call({"cmd": "inventory"})
        assert report["ok"] is True
        machines = report["machines"]
        assert machines and machines[0]["name"] == "demo_excavator"
        joint_names = [j["name"] for j in machines[0]["joints"]]
        assert joint_names == ["swing", "boom", "arm", "bucket"]
        backend.close()

    def test_bridge_restart_breaks_and_reports(self, bridge):
        backend = _backend(bridge.port)
        backend.open()
        bridge.close()  # server gone
        with pytest.raises(ConnectionError):
            backend.read_joints()


# ---------------------------------------------------------------------------
class TestEndToEndViaDemoBridge:
    """Full stack: session -> RemoteSimBackend -> agx_bridge_server.py --demo."""

    @pytest.fixture()
    def demo_port(self):
        # pick a free port by binding then releasing
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        proc = subprocess.Popen(
            [sys.executable, str(_BRIDGE_PATH), "--demo", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # wait for the port to accept connections
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                sock.close()
                break
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError("demo bridge exited early") from None
                time.sleep(0.1)
        yield port
        proc.terminate()
        proc.wait(timeout=5)

    def test_full_stack_dig_cycle(self, demo_port):
        session = build_agx_excavator_session.from_dict(
            {"env": {"cfg": {"low_level": {"backend": "remote", "port": demo_port}}}}
        )
        with session:
            piles = session.api.get_terrain()["piles"]
            assert piles, "demo bridge must report terrain"
            pile = piles[0]
            result = session.api.dig(pile["x_m"], pile["y_m"], 3.0, 0.5)
            assert result["ok"] is True
            moved = session.api.navigate_relative(0.5, dyaw_rad=0.2)
            assert moved["dx_m"] == 0.5

    def test_probe_bridge_mode_against_demo(self, demo_port):
        """The --bridge probe path, exercised through the same code the colleague runs."""
        probe_spec = importlib.util.spec_from_file_location(
            "agx_probe_under_test", _REPO / "scripts" / "agx_scene_probe.py"
        )
        assert probe_spec is not None and probe_spec.loader is not None
        probe = importlib.util.module_from_spec(probe_spec)
        probe_spec.loader.exec_module(probe)
        report: dict = {}

        def capture_render(report_data):
            report.update(report_data)
            return None

        original_render = probe._render_report
        probe._render_report = capture_render
        try:
            rc = probe.probe_bridge("127.0.0.1", demo_port, timeout_s=5.0)
        finally:
            probe._render_report = original_render
        assert rc == 0
        machines = report["machines"]
        assert machines[0]["name"] == "demo_excavator"


class TestBaseMotionCompletion:
    """pump 模式的底盘命令是"发了就走"：客户端必须轮询 base_state 等到位。

    否则连发的相对命令互相覆盖停车时刻（实测 9×0.7 rad 只走出 0.7 rad）。
    """

    def test_async_reply_is_polled_until_idle(self):
        fake = _ScriptedFake(
            [
                {"v": 1, "ok": True, "pong": True},          # ping
                {"v": 1, "ok": False, "error": "no inventory"},  # open() 的身份查询
                {"v": 1, "ok": True, "result": {"dx_m": 0.0, "dyaw_rad": 0.7}, "async": True, "eta_s": 1.4},
                {"v": 1, "ok": True, "busy": True, "remaining_s": 1.2},
                {"v": 1, "ok": True, "busy": False, "remaining_s": 0.0},
            ]
        )
        backend = _backend(fake.port)
        backend.open()
        result = backend.navigate_relative(0.0, 0.7, timeout_s=5.0)
        backend.close()
        assert result == {"dx_m": 0.0, "dyaw_rad": 0.7}
        assert fake.seen == ["ping", "inventory", "navigate_relative", "base_state", "base_state"]

    def test_sync_reply_is_not_polled(self):
        fake = _ScriptedFake(
            [
                {"v": 1, "ok": True, "pong": True},
                {"v": 1, "ok": False, "error": "no inventory"},
                {"v": 1, "ok": True, "result": {"dx_m": 1.0, "dyaw_rad": 0.0}},
            ]
        )
        backend = _backend(fake.port)
        backend.open()
        backend.navigate_relative(1.0, 0.0, timeout_s=5.0)
        backend.close()
        assert fake.seen == ["ping", "inventory", "navigate_relative"]

    def test_old_bridge_without_base_state_does_not_wait(self):
        """老桥不认 base_state：只记 debug，命令结果照常返回（向后兼容）。"""
        fake = _ScriptedFake(
            [
                {"v": 1, "ok": True, "pong": True},
                {"v": 1, "ok": False, "error": "no inventory"},
                {"v": 1, "ok": True, "result": {"dx_m": 0.0, "dyaw_rad": 0.3}, "async": True},
                {"v": 1, "ok": False, "error": "unknown command 'base_state'"},
            ]
        )
        backend = _backend(fake.port)
        backend.open()
        result = backend.navigate_relative(0.0, 0.3, timeout_s=5.0)
        backend.close()
        assert result == {"dx_m": 0.0, "dyaw_rad": 0.3}
