# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Bridge robustness tests — arbitrary scenes/models must not crash the bridge.

Pins the fixes found in the post-decouple review:
  - BridgeSession.handle wraps scene exceptions into ok:False responses
    (a raising scene method used to kill the pump/viewer loop)
  - inventory tolerates constraints without Motor1D (arbitrary .agx scenes)
  - _as_actuator SWIG downcast (base Constraint -> Hinge/Prismatic, Lock -> None)
  - scene-file mode never leaks DEMO_PILES as terrain truth
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXT = Path(__file__).resolve().parents[1]
_BRIDGE = _EXT / "scripts" / "agx_bridge_server.py"


def _load_bridge():
    spec = importlib.util.spec_from_file_location("agx_bridge_server_robust", _BRIDGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def bridge_module():
    return _load_bridge()


class _RaisingScene:
    """Scene adapter whose every command explodes (simulates a bad model)."""

    pump_mode = False

    def send_joint_targets(self, targets, timeout_s):
        raise RuntimeError("boom from scene")

    def read_joints(self):
        return {}


class TestHandleExceptionGuard:
    def test_scene_exception_becomes_ok_false_response(self, bridge_module):
        session = bridge_module.BridgeSession(_RaisingScene())
        resp = session.handle(
            {"v": 1, "cmd": "move_joints", "targets": {"swing": 1.0}, "timeout_s": 1.0}
        )
        assert resp["ok"] is False
        assert "boom from scene" in resp["error"]

    def test_unknown_command_still_reported(self, bridge_module):
        session = bridge_module.BridgeSession(_RaisingScene())
        resp = session.handle({"v": 1, "cmd": "teleport"})
        assert resp["ok"] is False and "unknown command" in resp["error"]


class TestInventoryWithoutMotor:
    def test_motorless_constraint_does_not_crash_inventory(self, bridge_module):
        adapter = bridge_module.AgxSceneAdapter(["swing"], {}, None, mode="headless")
        adapter._sim = object()  # bypass _require_sim (no AGX in unit test)

        class _Stub:
            def getName(self):
                return "PlainHinge"

            def getAngle(self):
                return 0.25

            def getMotor1D(self):
                return None  # the arbitrary-scene case: no motor attached

            def getLock1D(self):
                raise AssertionError(
                    "lock read must happen only when motor exists? no — but must not crash"
                )

        adapter._constraints["swing"] = [_Stub()]
        adapter._units["swing"] = "rad"
        adapter._ranges["swing"] = (-180.0, 180.0)
        report = adapter.inventory()
        joint = report["machines"][0]["joints"][0]
        assert joint["has_motor"] is False
        assert joint["force_range"] is None


class TestAsActuatorDowncast:
    def test_base_constraint_downcasts_to_hinge(self, bridge_module):
        hinge = SimpleNamespace(getAngle=lambda: 0.5)

        class _Base:
            def asHinge(self):
                return hinge

            def asPrismatic(self):
                return None

        assert bridge_module.AgxSceneAdapter._as_actuator(_Base()) is hinge

    def test_lock_constraints_return_none(self, bridge_module):
        class _Lock:
            def asHinge(self):
                return None

            def asPrismatic(self):
                return None

        assert bridge_module.AgxSceneAdapter._as_actuator(_Lock()) is None

    def test_already_typed_constraint_passes_through(self, bridge_module):
        typed = type("Hinge", (), {})()  # __name__ == "Hinge" → 直接透传
        assert bridge_module.AgxSceneAdapter._as_actuator(typed) is typed


class TestSceneTruthIsolation:
    def test_scene_file_mode_never_leaks_demo_piles(self, bridge_module):
        adapter = bridge_module.AgxSceneAdapter(
            ["swing"], {}, scene_path="x.agx", mode="headless"
        )
        assert adapter._piles, "demo default present before load"
        adapter._reset_scene_truth()
        assert adapter._piles == []
        assert adapter.read_terrain() == []
