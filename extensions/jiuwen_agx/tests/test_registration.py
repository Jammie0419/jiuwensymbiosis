# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Registration-chain tests — importing jiuwen_agx must wire everything up.

These pin the extension contract: capabilities into KNOWN_CAPABILITIES, specs
into the shared ACTIONS vocabulary, the skill into the catalogue, and the
adapter builder discoverable through the entry-points seam.
"""

from __future__ import annotations

from jiuwensymbiosis._extensions import discover_adapter_builders
from jiuwensymbiosis.agent.fast.registry import DEFAULT_REGISTRY
from jiuwensymbiosis.api.actions import ACTIONS
from jiuwensymbiosis.env.base import KNOWN_CAPABILITIES

import jiuwen_agx
from jiuwen_agx.agx_excavator.config import AgxExcavatorConfig


def test_capabilities_registered():
    assert "motion.excavator" in KNOWN_CAPABILITIES
    assert "sensing.terrain" in KNOWN_CAPABILITIES
    assert "policy.act" in KNOWN_CAPABILITIES


def test_actions_registered_into_shared_vocabulary():
    assert "dig" in ACTIONS
    assert "get_terrain" in ACTIONS
    assert "act_exec" in ACTIONS
    assert ACTIONS["dig"].capability == "motion.excavator"
    assert ACTIONS["get_terrain"].capability == "sensing.terrain"
    assert ACTIONS["act_exec"].capability == "policy.act"
    # RecoveryRail claims tools by tag — act_exec must carry the same motion
    # tag dig does, or a failed learned cycle skips the recovery path.
    assert "motion" in ACTIONS["act_exec"].tags


def test_skill_registered():
    names = [s["name"] for s in DEFAULT_REGISTRY.catalogue()]
    assert "excavate" in names
    assert "excavate_act" in names


def test_entry_point_discovery():
    builders = discover_adapter_builders()
    assert "agx_excavator" in builders
    builder = builders["agx_excavator"]
    assert hasattr(builder, "from_dict") and hasattr(builder, "from_yaml")


def test_registration_is_idempotent():
    """Re-running the registration chain must not duplicate or crash."""
    import importlib

    import jiuwensymbiosis.api.actions as core_actions

    from jiuwen_agx.actions import DIG, GET_TERRAIN

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


def test_act_exec_contract_intact():
    spec = ACTIONS["act_exec"]
    assert spec.required_params == ("dig_x_m", "dig_y_m", "dump_x_m", "dump_y_m")
    assert spec.requires == ("payload.clear",)
    assert spec.provides == ("payload.clear",)
    assert spec.invalidates_locations is True
    fields = spec.result_schema().get("properties", {})
    # Deliberately NO volume_m3: the client backend reports no measured volume
    # (docs/act-integration-plan.md §4.4) — a plan binding <bind>.volume_m3
    # must be rejected, not handed a fiction.
    assert {"ok", "beats", "cycle_s", "error"} <= set(fields)
    assert "volume_m3" not in fields


def test_resource_keys_identify_the_command_endpoint():
    """Runtime admission needs the endpoint identity (upstream contract)."""
    from jiuwen_agx.agx_excavator.session import _agx_resource_keys

    cfg = AgxExcavatorConfig()  # mock default
    assert _agx_resource_keys(cfg) == ("agx-mock",)
    remote = AgxExcavatorConfig(backend="remote", host="127.0.0.1", port=9700)
    assert _agx_resource_keys(remote) == ("agx-bridge:127.0.0.1:9700",)
    scene = AgxExcavatorConfig(backend="inprocess", scene_path="scene/exc.agx")
    assert _agx_resource_keys(scene) == ("agx-scene:scene/exc.agx",)
