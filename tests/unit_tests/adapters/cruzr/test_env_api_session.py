# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for Cruzr env/api/session structure."""

from __future__ import annotations

import pytest

from jiuwensymbiosis.adapters.cruzr.api import CruzrApi
from jiuwensymbiosis.adapters.cruzr.config import CruzrConfig
from jiuwensymbiosis.adapters.cruzr.env import CruzrEnv
from jiuwensymbiosis.adapters.cruzr.session import build_cruzr_session
from jiuwensymbiosis.tools.builder import list_tool_meta


class _FakeLowLevel:
    def __init__(self):
        self.calls = []
        self.joints = {"L_shoulder_pitch_joint": 0.0}

    def raise_arm_blocking(self, **kwargs):
        self.calls.append(("raise_arm_blocking", kwargs))
        return {"ok": True, **kwargs}

    def home(self, **kwargs):
        self.calls.append(("home", kwargs))
        return {"ok": True, **kwargs}

    def move_joints_blocking(self, *args, **kwargs):
        self.calls.append(("move_joints_blocking", args, kwargs))
        return {"ok": True}

    def get_joint_positions(self):
        return dict(self.joints)

    def close(self):
        self.calls.append(("close",))


def _connected_env() -> CruzrEnv:
    env = CruzrEnv(CruzrConfig())
    env._inner = _FakeLowLevel()
    env._connected = True
    return env


class TestCruzrEnv:
    def test_capabilities(self):
        env = CruzrEnv(CruzrConfig())
        assert "motion.joint" in env.capabilities
        assert "motion.cartesian" not in env.capabilities

    def test_observation_uses_joint_state(self):
        env = _connected_env()
        obs = env.get_observation()
        assert obs.extra["joint_positions"]["L_shoulder_pitch_joint"] == 0.0

    def test_home_refuses_instead_of_calling_the_bring_up_primitive(self):
        # driver.home(arm=...) moves ONE shoulder-pitch joint — answering the safe-posture
        # contract with it would sweep the arms across a leaned-forward torso.
        env = _connected_env()
        with pytest.raises(NotImplementedError, match="CruzrApi.home"):
            env.home()
        assert env.low_level.calls == []


class TestCruzrApi:
    def test_tool_methods_are_exposed(self):
        api = CruzrApi(_connected_env())
        names = {m["name"] for m in list_tool_meta(api)}
        # move_joint 现在按「关节名 -> 位置」寻址，省略的关节保持不动——这对一台
        # 双臂 + 腰 + 升降、没有规范下标序的本体是有意义的，所以它也实现这条共享动作。
        # 词表不再为了迁就本体而分叉成两个名字（见 tests/unit_tests/test_vocabulary_forks.py）。
        assert "move_joint" in names
        assert "move_named_joint" in names

    def test_raising_an_arm_is_move_named_joint_not_a_cruzr_only_tool(self):
        """"抬左臂"就是把一个具名关节开到某个角度——共享词表已经能说这件事，
        本体专有工具说的是同一件事，却只在这台机器人上成立。"""
        names = {m["name"] for m in list_tool_meta(CruzrApi(_connected_env()))}
        assert not {"raise_left_arm", "raise_right_arm", "lower_left_arm", "lower_right_arm", "raise_arm"} & names

    def test_move_named_joint_dispatches(self):
        """Through ``move_joints_blocking``: it is the encoding this body speaks, and the only
        one that can carry ``ramp_duration_s``. The old union-typed ``move_joint_blocking``
        silently mapped a bare 1-element list onto the default arm's shoulder — exactly the
        drift the shared vocabulary exists to prevent — and is gone."""
        env = _connected_env()
        api = CruzrApi(env)
        api.move_named_joint("L_shoulder_pitch_joint", 0.5)
        assert env.low_level.calls[0][0] == "move_joints_blocking"
        assert env.low_level.calls[0][1][0] == {"L_shoulder_pitch_joint": 0.5}


class TestCruzrSession:
    def test_builder_from_dict(self):
        session = build_cruzr_session.from_dict({"name": "cruzr_test"})
        assert session.name == "cruzr_test"
        assert isinstance(session.env, CruzrEnv)
        assert isinstance(session.api, CruzrApi)
        assert session.extra_globals["cruzr_cfg"].name == "cruzr_test"


class TestCruzrVisionWiring:
    def test_vision_capabilities(self):
        env = CruzrEnv(CruzrConfig())
        assert "vision.detection" in env.capabilities
        assert "vision.camera" in env.capabilities
        assert "vision.depth" in env.capabilities

    def test_vision_actions_exposed_via_session(self):
        session = build_cruzr_session.from_dict({"name": "cruzr_vis"})
        names = {m["name"] for m in list_tool_meta(session.api)}
        assert "locate_for_grasp" in names

    def test_api_gets_calib_path_from_cfg(self):
        session = build_cruzr_session.from_dict({"camera_calib_path": "/tmp/c.json"})
        assert session.api._camera_calib_path == "/tmp/c.json"
