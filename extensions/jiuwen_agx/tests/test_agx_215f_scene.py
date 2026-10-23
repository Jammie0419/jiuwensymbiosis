# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""215F 接入的回归测试（不需要 AGX）。

覆盖三件事：
  * 场景工厂的纯函数（土堆高度场/力矩表）；
  * 桥接的 ``attach_machine`` 契约（关节/单位/量程/身份/地形/载料）；
  * 215F 配置和 ``has_base: false`` 的能力门控。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from jiuwensymbiosis.tools.builder import list_tool_meta

from jiuwen_agx.agx_excavator.api import AgxExcavatorApi
from jiuwen_agx.agx_excavator.config import AgxExcavatorConfig
from jiuwen_agx.agx_excavator.env import AgxExcavatorEnv

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "agx_excavator"
    / "agx_excavator.215f.yaml"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bridge():
    return _load_module(
        "agx_bridge_server_215f_test", _SCRIPTS / "agx_bridge_server.py"
    )


def _factory():
    return _load_module("agx_215f_scene_test", _SCRIPTS / "agx_215f_scene.py")


class _FakeRange:
    def __init__(self, low: float, high: float) -> None:
        self._low = low
        self._high = high

    def lower(self) -> float:
        return self._low

    def upper(self) -> float:
        return self._high


class _FakeHinge:
    def __init__(self, name: str, low: float = -3.14, high: float = 3.14) -> None:
        self._name = name
        self._low = low
        self._high = high

    def getName(self) -> str:
        return self._name

    def getAngle(self) -> float:
        return 0.0

    def getRange1D(self):
        return SimpleNamespace(getRange=lambda: _FakeRange(self._low, self._high))

    def getMotor1D(self):
        return None

    def getLock1D(self):
        return None


def _joints():
    return {name: _FakeHinge(name) for name in ("swing", "boom", "arm", "bucket")}


def _attached_adapter(mass_kg: float = 0.0):
    module = _bridge()
    adapter = module.AgxSceneAdapter(
        ["swing", "boom", "arm", "bucket"], {}, None, mode="headless"
    )
    adapter.attach_machine(
        SimpleNamespace(),
        _joints(),
        units=dict.fromkeys(("swing", "boom", "arm", "bucket"), "rad"),
        machine_name="excavator215f",
        terrain_truth=[
            {
                "name": "soil_pile_215f",
                "x_m": 5.0,
                "y_m": 2.0,
                "z_m": 1.95,
                "volume_m3": 9.98,
            }
        ],
        load_mass_fn=lambda: mass_kg,
        loaded_threshold_kg=50.0,
    )
    return module, adapter


class TestSceneFactoryPure:
    def test_default_pile_spec_and_heights(self):
        factory = _factory()
        spec = factory.pile_spec()
        assert spec == factory.DEFAULT_PILE
        heights = factory.pile_heights(spec)
        assert len(heights) == factory.TERRAIN_RES * factory.TERRAIN_RES
        # 土堆中心（世界 5.0, 2.0）应接近顶面
        cell = factory.TERRAIN_SIZE_M / factory.TERRAIN_RES
        half = factory.TERRAIN_RES // 2
        i = round(5.0 / cell) + half
        j = round(2.0 / cell) + half
        assert heights[j * factory.TERRAIN_RES + i] == pytest.approx(spec["top"])
        assert heights[0] == 0.0

    def test_default_force_ranges(self):
        factory = _factory()
        assert factory.force_ranges() == factory.DEFAULT_FORCE_KNM


class TestAttachMachine:
    def test_transfers_joints_units_machine_truth_and_load(self):
        _, adapter = _attached_adapter(mass_kg=123.0)
        assert set(adapter._constraints) == {"swing", "boom", "arm", "bucket"}
        assert set(adapter._units.values()) == {"rad"}
        assert adapter.read_terrain() == [
            {
                "name": "soil_pile_215f",
                "x_m": 5.0,
                "y_m": 2.0,
                "z_m": 1.95,
                "volume_m3": 9.98,
            }
        ]
        assert adapter.bucket_mass_kg() == 123.0
        assert adapter.scoop_state() is True

        inventory = adapter.inventory()
        machine = inventory["machines"][0]
        assert machine["name"] == "excavator215f"
        assert machine["loaded_threshold_kg"] == 50.0
        assert machine["bucket_mass_kg"] == 123.0
        assert machine["terrain"][0]["name"] == "soil_pile_215f"
        assert len(machine["joints"]) == 4

    def test_below_threshold_is_clear(self):
        _, adapter = _attached_adapter(mass_kg=1.0)
        assert adapter.scoop_state() is False

    def test_factory_scene_does_not_inherit_demo_piles(self):
        module = _bridge()
        adapter = module.AgxSceneAdapter(["swing", "boom", "arm", "bucket"], {}, None)
        adapter.attach_machine(
            SimpleNamespace(),
            _joints(),
            units=dict.fromkeys(("swing", "boom", "arm", "bucket"), "rad"),
            machine_name="excavator215f",
            terrain_truth=None,
        )
        assert adapter.read_terrain() == []

    def test_missing_joint_is_a_readable_error(self):
        module = _bridge()
        adapter = module.AgxSceneAdapter(["swing", "boom"], {}, None)
        with pytest.raises(ValueError, match="joint 'boom'"):
            adapter.attach_machine(SimpleNamespace(), {"swing": _FakeHinge("swing")})

    def test_unknown_machine_is_a_readable_error(self):
        module = _bridge()
        adapter = module.AgxSceneAdapter(["swing"], {}, None, machine="nope")
        with pytest.raises(ValueError, match="unknown --machine"):
            adapter.load()


class Test215Config:
    def test_yaml_is_all_radian_with_kinematic_base(self):
        cfg = AgxExcavatorConfig.from_yaml(_CONFIG)
        assert cfg.joint_names == ("swing", "boom", "arm", "bucket")
        assert cfg.joint_units == "rad"
        assert cfg.swing_unit == "rad"
        assert cfg.has_base is True
        assert cfg.base_step_limits == (1.0, 0.7)

    def test_all_six_frames_are_inside_joint_limits(self):
        cfg = AgxExcavatorConfig.from_yaml(_CONFIG)
        tuning = cfg.dig_cycle_tuning or {}
        limits = cfg.joint_limits or {}
        for prefix in ("ready", "dig", "curl", "dump"):
            for joint in ("boom", "arm", "bucket"):
                value = float(tuning[f"{prefix}_{joint}_deg"])
                low, high = limits[joint]
                assert low <= value <= high, (
                    f"{prefix}_{joint}={value} outside {limits[joint]}"
                )

    def test_natural_stages_parse_and_stay_inside_limits(self):
        cfg = AgxExcavatorConfig.from_yaml(_CONFIG)
        assert cfg.dig_travel is not None and cfg.dig_travel["bucket"] == 0.9
        assert cfg.dig_carry is not None and cfg.dig_carry["bucket"] == 0.449709
        assert cfg.dig_dump_dwell_s == 1.5
        cut_path = cfg.dig_cut_path or []
        assert len(cut_path) == 4
        limits = cfg.joint_limits or {}
        for frame in cut_path:
            for joint, value in frame.items():
                low, high = limits[joint]
                assert low <= value <= high, (
                    f"cut_path {joint}={value} outside {limits[joint]}"
                )

    def test_kinematic_base_capability_and_tools(self):
        cfg = AgxExcavatorConfig.from_yaml(_CONFIG)
        env = AgxExcavatorEnv(cfg)
        api = AgxExcavatorApi(env)
        assert "motion.base" in api.capabilities
        names = {entry["name"] for entry in list_tool_meta(api, env=env)}
        assert {"navigate_relative", "rotate_base", "drive_arc"} <= names
        assert {"dig", "move_joint", "get_terrain", "home"} <= names
