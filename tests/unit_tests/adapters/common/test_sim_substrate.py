# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SimMachine substrate tests — config / mock backend / driver / env / api gating."""

from __future__ import annotations

import math

import pytest

from jiuwensymbiosis.adapters._common.sim.api import SimMachineApi
from jiuwensymbiosis.adapters._common.sim.backend import (
    MockSimBackend,
    create_backend,
)
from jiuwensymbiosis.adapters._common.sim.config import SimMachineConfig
from jiuwensymbiosis.adapters._common.sim.driver import SimMachineDriver
from jiuwensymbiosis.adapters._common.sim.env import SimMachineEnv
from jiuwensymbiosis.tools.builder import list_tool_meta

JOINTS = ("swing", "boom", "arm", "bucket")


def _cfg(**overrides) -> SimMachineConfig:
    defaults: dict = {
        "name": "t",
        "joint_names": JOINTS,
        "joint_limits": dict.fromkeys(JOINTS, (-100.0, 100.0)),
        "home_joints": {"swing": 0.0},
        "has_base": True,
        "base_step_limits": (1.0, 0.7),
    }
    defaults.update(overrides)
    return SimMachineConfig(**defaults)


def _driver(**overrides) -> SimMachineDriver:
    driver = SimMachineDriver(_cfg(**overrides), backend=MockSimBackend(_cfg(**overrides)))
    driver.connect()
    return driver


# ============================================================================ config
class TestConfig:
    def test_nested_env_cfg_low_level_dict(self):
        cfg = SimMachineConfig.from_dict(
            {
                "adapter": "x",
                "env": {"cfg": {"low_level": {"name": "m", "joint_names": ["a", "b"], "has_base": True}}},
                "model": {"api_base": "x"},
            }
        )
        assert cfg.name == "m"
        assert cfg.joint_names == ("a", "b")
        assert cfg.has_base is True

    def test_normalisation_and_unknown_keys_ignored(self):
        cfg = SimMachineConfig.from_dict(
            {
                "joint_names": ["j1", "j2"],
                "base_step_limits": [2, 0.5],
                "camera_resolution": [320, 240],
                "joint_limits": {"j1": [-1, 1], "bad": [1]},
                "nonexistent_field": 1,
            }
        )
        assert cfg.joint_names == ("j1", "j2")
        assert cfg.base_step_limits == (2.0, 0.5)
        assert cfg.camera_resolution == (320, 240)
        assert cfg.joint_limits == {"j1": (-1.0, 1.0)}
        assert not hasattr(cfg, "nonexistent_field")

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            SimMachineConfig.from_dict({"backend": "teleport"})

    def test_bad_joint_units_rejected(self):
        with pytest.raises(ValueError, match="joint_units"):
            SimMachineConfig.from_dict({"joint_units": "gradians"})


# ============================================================================ mock backend
class TestMockBackend:
    def test_open_and_close_idempotent(self):
        backend = MockSimBackend(SimMachineConfig())
        backend.open()
        backend.open()
        backend.close()
        backend.close()
        assert [e["cmd"] for e in backend.move_log].count("open") == 1

    def test_commands_require_open(self):
        backend = MockSimBackend(SimMachineConfig())
        with pytest.raises(RuntimeError, match="open"):
            backend.read_joints()

    def test_joint_targets_update_state_and_log(self):
        backend = MockSimBackend(SimMachineConfig(joint_names=JOINTS))
        backend.open()
        state = backend.send_joint_targets({"swing": 15.0}, timeout_s=1.0)
        assert state["swing"] == 15.0
        assert backend.move_log[-1] == {"cmd": "move_joints", "targets": {"swing": 15.0}}

    def test_terrain_and_scoop_truth(self):
        backend = MockSimBackend(SimMachineConfig(), piles=[{"name": "p", "x_m": 1.0, "y_m": 2.0, "volume_m3": 3.0}])
        backend.open()
        assert backend.read_terrain() == [{"name": "p", "x_m": 1.0, "y_m": 2.0, "volume_m3": 3.0}]
        assert backend.scoop_state() is False
        backend.mark_scoop(True)
        assert backend.scoop_state() is True

    def test_grab_frames_is_none_never_invented(self):
        backend = MockSimBackend(SimMachineConfig())
        backend.open()
        assert backend.grab_frames() is None


def test_create_backend_unknown_name_lists_registered():
    with pytest.raises(ValueError, match="mock"):
        create_backend(SimMachineConfig(backend="teleport"))


# ============================================================================ driver
class TestDriver:
    def test_partial_command_holds_unmentioned_joints(self):
        driver = _driver()
        driver.move_joints_blocking({"swing": 10.0, "boom": 20.0})
        driver.move_joints_blocking({"swing": 30.0})
        state = driver.get_joint_positions()
        assert state["swing"] == 30.0
        assert state["boom"] == 20.0  # held

    def test_unknown_joint_refused(self):
        driver = _driver()
        with pytest.raises(ValueError, match="unknown joint"):
            driver.move_joints_blocking({"elbow": 1.0})

    def test_non_finite_refused(self):
        driver = _driver()
        with pytest.raises(ValueError, match="non-finite"):
            driver.move_joints_blocking({"swing": math.nan})

    def test_joint_target_outside_configured_limits_refused(self):
        driver = _driver()
        with pytest.raises(ValueError, match="outside configured"):
            driver.move_joints_blocking({"swing": 500.0})
        # boundary values are inclusive
        driver.move_joints_blocking({"swing": 100.0, "boom": -100.0})

    def test_empty_command_refused(self):
        driver = _driver()
        with pytest.raises(ValueError, match="empty"):
            driver.move_joints_blocking({})

    def test_base_motion_needs_has_base(self):
        driver = _driver(has_base=False)
        with pytest.raises(NotImplementedError, match="has_base"):
            driver.navigate_relative(1.0)

    def test_navigate_delegates_and_ignores_strafe(self):
        driver = _driver()
        result = driver.navigate_relative(0.8, dy_m=5.0, dyaw_rad=0.3)
        assert result == {"dx_m": 0.8, "dyaw_rad": 0.3}
        assert driver.backend.move_log[-1]["cmd"] == "navigate_relative"

    def test_home_moves_to_home_joints(self):
        driver = _driver()
        driver.move_joints_blocking({"swing": 90.0})
        driver.home()
        assert driver.get_joint_positions()["swing"] == 0.0

    def test_requires_connect(self):
        driver = SimMachineDriver(_cfg(), backend=MockSimBackend(_cfg()))
        with pytest.raises(RuntimeError, match="connect"):
            driver.get_joint_positions()


# ============================================================================ env
class TestEnv:
    def test_capabilities_follow_config(self):
        assert SimMachineEnv(_cfg()).capabilities == frozenset({"motion.joint", "motion.base", "sensing.terrain"})
        assert SimMachineEnv(_cfg(has_base=False, terrain_enabled=False)).capabilities == frozenset({"motion.joint"})
        camera_env = SimMachineEnv(_cfg(camera_enabled=True, has_base=False, terrain_enabled=False))
        assert "vision.camera" in camera_env.capabilities
        assert "motion.base" not in camera_env.capabilities

    def test_low_level_binds_exactly_once(self):
        env = SimMachineEnv(_cfg())
        driver = object()
        env.low_level = driver
        assert env.low_level is driver
        with pytest.raises(AttributeError, match="already bound"):
            env.low_level = object()

    def test_connect_is_idempotent_and_disconnect_unbinds(self):
        env = SimMachineEnv(_cfg())
        env.connect()
        first = env.low_level
        env.connect()
        assert env.low_level is first
        env.disconnect()
        assert env.low_level is None
        env.disconnect()  # idempotent

    def test_connect_opens_a_prebound_driver(self):
        """Regression: a driver bound via the low_level setter must still get
        connect() called — the simulator connection was silently skipped."""
        env = SimMachineEnv(_cfg())
        driver = SimMachineDriver(_cfg(), backend=MockSimBackend(_cfg()))
        env.low_level = driver
        env.connect()
        assert driver._connected is True  # backend opened through the driver
        env.disconnect()
        assert driver._connected is False

    def test_observation_best_effort_when_backend_breaks(self):
        env = SimMachineEnv(_cfg())
        env.connect()
        env.low_level._backend.read_joints = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        obs = env.get_observation()
        assert obs.joints is None  # survived
        assert obs.extra["joint_units"] == "deg"

    def test_observation_carries_joints_in_order_and_terrain(self):
        env = SimMachineEnv(_cfg())
        env.connect()
        env.low_level._backend.send_joint_targets({"boom": 5.0}, timeout_s=1.0)
        obs = env.get_observation()
        assert obs.joints == [0.0, 5.0, 0.0, 0.0]
        assert obs.extra["terrain"][0]["name"]

    def test_holding_payload_maps_scoop_truth(self):
        env = SimMachineEnv(_cfg())
        env.connect()
        assert env.holding_payload is False
        env.driver.mark_scoop(True)
        assert env.holding_payload is True

    def test_joint_limits_ordered_and_read_only(self):
        env = SimMachineEnv(_cfg())
        assert list(env.joint_limits) == list(JOINTS)
        with pytest.raises(AttributeError, match="read-only"):
            env.joint_limits = {}

    def test_base_step_limits_read_only(self):
        env = SimMachineEnv(_cfg())
        assert env.base_step_limits == (1.0, 0.7)
        with pytest.raises(AttributeError, match="read-only"):
            env.base_step_limits = (2.0, 2.0)

    def test_home_dispatches_to_driver(self):
        env = SimMachineEnv(_cfg())
        env.connect()
        env.low_level._backend.send_joint_targets({"swing": 45.0}, timeout_s=1.0)
        env.home()
        assert env.get_joint_positions()["swing"] == 0.0


# ============================================================================ api / tool gating
class TestToolGating:
    def _tools(self, **overrides) -> set[str]:
        cfg = _cfg(**overrides)
        env = SimMachineEnv(cfg)
        env.connect()
        return {entry["name"] for entry in list_tool_meta(SimMachineApi(env), env=env)}

    def test_full_machine_emits_all_generic_tools(self):
        assert self._tools() == {
            "home",
            "move_joint",
            "get_joint_positions",
            "navigate_relative",
            "rotate_base",
            "drive_arc",
            "get_terrain",
        }

    def test_static_machine_drops_base_and_keeps_terrain_gate_decidable(self):
        tools = self._tools(has_base=False)
        assert "navigate_relative" not in tools
        assert "get_terrain" in tools

    def test_terrain_disabled_drops_get_terrain(self):
        tools = self._tools(terrain_enabled=False)
        assert "get_terrain" not in tools
        assert "move_joint" in tools

    def test_effective_caps_are_the_api_env_intersection(self):
        # api.capabilities is the api-side derivation; the TOOL gate is api ∩ env,
        # so a machine with base/terrain switched off never sees those tools (the
        # emission tests above pin the tool sets themselves).
        cfg = _cfg(has_base=False, terrain_enabled=False)
        env = SimMachineEnv(cfg)
        api = SimMachineApi(env)
        assert api.capabilities & env.capabilities == frozenset({"motion.joint"})
