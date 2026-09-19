# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Registration-chain tests — importing jiuwen_agx must wire everything up.

These pin the extension contract: capabilities into KNOWN_CAPABILITIES, specs
into the shared ACTIONS vocabulary, the skill into the catalogue, and the
adapter builder discoverable through the entry-points seam.
"""

from __future__ import annotations

import jiuwen_agx  # noqa: F401  (import IS the registration)

from jiuwensymbiosis._extensions import discover_adapter_builders
from jiuwensymbiosis.agent.fast.registry import DEFAULT_REGISTRY
from jiuwensymbiosis.api.actions import ACTIONS
from jiuwensymbiosis.env.base import KNOWN_CAPABILITIES


def test_capabilities_registered():
    assert "motion.excavator" in KNOWN_CAPABILITIES
    assert "sensing.terrain" in KNOWN_CAPABILITIES


def test_actions_registered_into_shared_vocabulary():
    assert "dig" in ACTIONS
    assert "get_terrain" in ACTIONS
    assert ACTIONS["dig"].capability == "motion.excavator"
    assert ACTIONS["get_terrain"].capability == "sensing.terrain"


def test_skill_registered():
    assert "excavate" in [s["name"] for s in DEFAULT_REGISTRY.catalogue()]


def test_entry_point_discovery():
    builders = discover_adapter_builders()
    assert "agx_excavator" in builders
    builder = builders["agx_excavator"]
    assert hasattr(builder, "from_dict") and hasattr(builder, "from_yaml")


def test_registration_is_idempotent():
    """Re-running the registration chain must not duplicate or crash."""
    import importlib

    from jiuwen_agx.actions import DIG, GET_TERRAIN

    import jiuwensymbiosis.api.actions as core_actions

    before = dict(core_actions.ACTIONS)
    core_actions.register_actions(DIG, GET_TERRAIN)  # same specs again
    assert dict(core_actions.ACTIONS) == before  # name-keyed overwrite, no dupes
    importlib.reload(jiuwen_agx.contracts)  # zero-dependency module reloads cleanly


def test_dig_spec_contract_intact():
    spec = ACTIONS["dig"]
    assert spec.required_params == ("dig_x_m", "dig_y_m", "dump_x_m", "dump_y_m")
    assert spec.requires == ("payload.clear",)
    assert spec.provides == ("payload.clear",)
    assert spec.invalidates_locations is True
    fields = spec.result_schema().get("properties", {})
    assert {"ok", "volume_m3", "cycle_s", "error"} <= set(fields)
