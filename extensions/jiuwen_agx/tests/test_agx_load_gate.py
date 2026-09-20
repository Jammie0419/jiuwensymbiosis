# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""载料判定（payload.held）的回归测试 —— 这个坑值得钉住。

AGX Terrain 的铲斗 aggregate 质量**空斗贴地时也不为 0**（接触区土壤算在内），
阈值取小（曾经是 1.0 kg）会让世界状态永久报 payload.held，于是 `dig` 的前置
条件 payload.clear 永不成立，规划器每次都失败：

    dig requires ['payload.clear'] but the state here is ['payload.held']

判据：空斗/残留 → clear；一斗沙土（数百公斤）→ held。
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


def _adapter_with_shovel(mass_kg: float | None):
    """AgxSceneAdapter + 一个只回答 aggregate 质量的假铲斗（None = 没有铲斗）。"""
    module = _load_bridge_module()
    adapter = module.AgxSceneAdapter(
        ["swing", "boom", "arm", "bucket"], {}, None, mode="headless"
    )
    if mass_kg is not None:
        adapter._shovel = SimpleNamespace(  # noqa: SLF001 - 白盒：直接注入测量源
            getSoilParticleAggregate=lambda: SimpleNamespace(
                getTotalAggregateMass=lambda: mass_kg
            )
        )
    return module, adapter


class TestLoadedMassThreshold:
    def test_empty_bucket_is_clear(self):
        _, adapter = _adapter_with_shovel(0.0)
        assert adapter.bucket_mass_kg() == 0.0
        assert adapter.scoop_state() is False

    def test_contact_residue_is_clear(self):
        """实测本底：空斗贴地/刚倒完会残留 0~1 kg —— 必须是 clear。"""
        _, adapter = _adapter_with_shovel(1.0)
        assert adapter.scoop_state() is False

    def test_full_bucket_is_held(self):
        """一斗沙土（约 0.6 m3）是数百公斤量级 —— 必须是 held。"""
        _, adapter = _adapter_with_shovel(600.0)
        assert adapter.scoop_state() is True

    def test_threshold_is_well_above_the_noise_floor(self):
        module, _ = _adapter_with_shovel(None)
        assert module.LOADED_MASS_THRESHOLD_KG >= 10.0, (
            "阈值必须远高于空斗本底（实测 0~1 kg），否则 dig 会被永久判为"
            "'斗里有料'而不可规划"
        )

    def test_measurement_wins_over_mark_scoop(self):
        """AGX 后端的载料是真值（测质量），mark_scoop 只是协议兼容，不得覆盖它。"""
        _, adapter = _adapter_with_shovel(0.5)
        adapter.mark_scoop(True)
        assert adapter.scoop_state() is False, "写接口不能盖过实测"

    def test_memory_machine_still_uses_mark_scoop(self):
        """没有铲斗测量源的场景（--scene 加载的任意 .agx）才回落到内存标志。"""
        _, adapter = _adapter_with_shovel(None)
        assert adapter.bucket_mass_kg() == 0.0
        assert adapter.scoop_state() is False
        adapter.mark_scoop(True)
        assert adapter.scoop_state() is True
