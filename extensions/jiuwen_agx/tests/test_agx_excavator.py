# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AgxExcavator thin-package tests: config slice, dig-cycle geometry, api bindings."""

from __future__ import annotations

import math

import pytest
from jiuwen_agx.agx_excavator import build_agx_excavator_session
from jiuwen_agx.agx_excavator.api import AgxExcavatorApi
from jiuwen_agx.agx_excavator.config import AgxExcavatorConfig
from jiuwen_agx.agx_excavator.env import AgxExcavatorEnv
from jiuwen_agx.agx_excavator.work import REQUIRED_JOINTS, execute_dig_cycle

from jiuwensymbiosis.tools.builder import list_tool_meta


# ============================================================================ config
class TestConfig:
    def test_excavator_defaults(self):
        cfg = AgxExcavatorConfig()
        assert cfg.joint_names == ("swing", "boom", "arm", "bucket")
        assert cfg.has_base is True
        assert cfg.name == "agx_excavator"
        assert set(cfg.joint_limits or {}) == set(cfg.joint_names)

    def test_yaml_overlay_merges_with_defaults(self):
        cfg = AgxExcavatorConfig.from_dict(
            {
                "env": {
                    "cfg": {
                        "low_level": {
                            "backend": "mock",
                            "dig_cycle_tuning": {"dig_boom_deg": -50.5},
                        }
                    }
                },
            }
        )
        assert cfg.backend == "mock"
        assert cfg.dig_cycle_tuning == {"dig_boom_deg": -50.5}
        # untouched defaults survive
        assert cfg.joint_names == ("swing", "boom", "arm", "bucket")
        assert cfg.reach_max_m == 6.0

    def test_env_adds_work_capability(self):
        env = AgxExcavatorEnv(AgxExcavatorConfig())
        assert "motion.excavator" in env.capabilities
        assert env.capabilities == frozenset(
            {"motion.joint", "motion.base", "sensing.terrain", "motion.excavator"}
        )


# ============================================================================ work cycle
class _SpyDriver:
    """Records keyframes and scoop writes; satisfies what work.py touches."""

    def __init__(self, joint_names=REQUIRED_JOINTS):
        self.joint_names = tuple(joint_names)
        self._joints = dict.fromkeys(self.joint_names, 0.0)
        self._scoop = False
        self.moves: list[dict[str, float]] = []

    def move_joints_blocking(self, targets, **kwargs):
        self._joints.update(targets)
        self.moves.append(dict(targets))
        return dict(self._joints)

    def scoop_state(self) -> bool:
        return self._scoop

    def mark_scoop(self, loaded: bool) -> None:
        self._scoop = bool(loaded)


class TestDigCycle:
    def test_swing_is_atan2_of_the_ground_point(self):
        driver = _SpyDriver()
        execute_dig_cycle(driver, dig_x_m=-2.0, dig_y_m=1.0, dump_x_m=3.0, dump_y_m=0.0)
        expected = math.degrees(math.atan2(1.0, -2.0))
        assert driver.moves[0] == {"swing": pytest.approx(expected)}
        assert {
            "swing": pytest.approx(math.degrees(math.atan2(0.0, 3.0)))
        } == driver.moves[4]

    def test_swing_offset_is_applied_and_normalised(self):
        driver = _SpyDriver()
        # offset rotates the whole dig plane; atan2(-2,0)=-90 + (-45) = -135
        execute_dig_cycle(
            driver,
            dig_x_m=0.0,
            dig_y_m=-2.0,
            dump_x_m=0.0,
            dump_y_m=2.0,
            tuning={"swing_offset_deg": -45.0},
        )
        assert driver.moves[0] == {"swing": pytest.approx(-135.0)}
        # wrap-around: atan2(2,0)=90 + 135 = 225 -> normalised to -135
        execute_dig_cycle(
            driver,
            dig_x_m=0.0,
            dig_y_m=2.0,
            dump_x_m=0.0,
            dump_y_m=2.0,
            tuning={"swing_offset_deg": 135.0},
        )
        assert driver.moves[0] == {"swing": pytest.approx(-135.0)}

    def test_cycle_shape_and_scoop_truth(self):
        driver = _SpyDriver()
        result = execute_dig_cycle(
            driver, dig_x_m=-2.0, dig_y_m=1.0, dump_x_m=3.0, dump_y_m=0.5
        )
        assert len(driver.moves) == 6  # 对准/就位/下铲/收斗/摆转/卸料
        assert result["volume_m3"] > 0
        assert result["cycle_s"] >= 0.0
        assert driver.scoop_state() is False  # cycle ends unloaded

    def test_tuning_overlay_changes_keyframes_and_volume(self):
        driver = _SpyDriver()
        result = execute_dig_cycle(
            driver,
            dig_x_m=-2.0,
            dig_y_m=1.0,
            dump_x_m=3.0,
            dump_y_m=0.5,
            tuning={"bucket_volume_m3": 9.9, "dig_boom_deg": -50.0},
        )
        assert result["volume_m3"] == 9.9
        assert driver.moves[2]["boom"] == pytest.approx(-50.0)

    def test_refuses_point_outside_reach_annulus(self):
        driver = _SpyDriver()
        with pytest.raises(ValueError, match="annulus"):
            execute_dig_cycle(
                driver, dig_x_m=20.0, dig_y_m=0.0, dump_x_m=3.0, dump_y_m=0.0
            )
        with pytest.raises(ValueError, match="annulus"):
            execute_dig_cycle(
                driver, dig_x_m=0.0, dig_y_m=0.0, dump_x_m=3.0, dump_y_m=0.0
            )

    def test_refuses_when_still_loaded(self):
        driver = _SpyDriver()
        driver.mark_scoop(True)
        with pytest.raises(ValueError, match="loaded"):
            execute_dig_cycle(
                driver, dig_x_m=-2.0, dig_y_m=1.0, dump_x_m=3.0, dump_y_m=0.5
            )

    def test_refuses_body_missing_required_joints(self):
        driver = _SpyDriver(joint_names=("swing", "boom"))
        with pytest.raises(ValueError, match="needs joints"):
            execute_dig_cycle(
                driver, dig_x_m=-2.0, dig_y_m=1.0, dump_x_m=3.0, dump_y_m=0.5
            )

    def test_refuses_non_finite_point(self):
        driver = _SpyDriver()
        with pytest.raises(ValueError, match="finite"):
            execute_dig_cycle(
                driver, dig_x_m=math.nan, dig_y_m=1.0, dump_x_m=3.0, dump_y_m=0.5
            )


# ============================================================================ api / session
class TestApi:
    def _built(self):
        session = build_agx_excavator_session.from_dict({})
        return session

    def test_from_empty_dict_builds_a_session(self):
        session = self._built()
        assert isinstance(session.env, AgxExcavatorEnv)
        assert isinstance(session.api, AgxExcavatorApi)

    def test_tools_include_the_work_cycle(self):
        session = self._built()
        names = {
            entry["name"] for entry in list_tool_meta(session.api, env=session.env)
        }
        assert "dig" in names
        assert {"move_joint", "navigate_relative", "get_terrain", "home"} <= names

    def test_dig_success_and_failure_shapes(self):
        session = self._built()
        with session:
            good = session.api.dig(-2.0, 1.0, 3.0, 0.5)
            assert good["ok"] is True
            assert set(good) == {"ok", "volume_m3", "cycle_s"}
            bad = session.api.dig(20.0, 0.0, 3.0, 0.5)
            assert bad["ok"] is False
            assert "annulus" in bad["error"]

    def test_dig_refuses_second_scoop_while_loaded_mid_cycle_is_guarded_by_work(self):
        session = self._built()
        with session:
            # one cycle ends unloaded, so back-to-back digs are legal (matches the contract:
            # requires payload.clear, provides payload.clear)
            assert session.api.dig(-2.0, 1.0, 3.0, 0.5)["ok"] is True
            assert session.api.dig(-2.0, 1.0, 3.0, 0.5)["ok"] is True

    def test_get_terrain_reports_mock_truth(self):
        session = self._built()
        with session:
            piles = session.api.get_terrain()["piles"]
            assert piles and {"name", "x_m", "y_m", "volume_m3"} <= set(piles[0])
