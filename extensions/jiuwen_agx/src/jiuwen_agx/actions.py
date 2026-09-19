# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""jiuwen_agx ActionSpecs — simulator work cycles & terrain truth.

Declared HERE (not in core ``api/actions.py``) and pushed into the shared
vocabulary through ``register_actions`` at package import. Locally declared
specs are first-class: ``@implements`` binds them, the fast planner plans them
(the action index reads the api instance's ``__tool_meta__``), and
``jiuwensymbiosis-actions --config`` shows them. Only the body-agnostic
``--vocabulary`` listing is unaware of them until this package is imported —
which any session build does via the entry-point discovery.
"""

from __future__ import annotations

from jiuwen_agx.contracts import DigFailure, DigResult, TerrainScan
from jiuwensymbiosis.api.actions import ActionSpec
from jiuwensymbiosis.api.decorators import UnknownCapability  # noqa: F401  (re-export convenience)

__all__ = ["DIG", "GET_TERRAIN"]

# NOTE: registering the capabilities these specs gate happens in jiuwen_agx/__init__.py
# BEFORE this module is imported (ActionSpec validates capability membership at
# construction time).


DIG = ActionSpec(
    name="dig",
    description=(
        "Execute one excavator dig-and-dump cycle: swing to face the dig ground point, scoop "
        "one bucket of material, swing to the dump point, release. Both points are ground "
        "coordinates in the base frame, in METRES (REP-103), and must lie inside this body's "
        "reachable annulus — if a spot is out of reach, drive the undercarriage there first "
        "(navigate_relative), then dig. Refuses while the bucket is still loaded; dump or home "
        "first. Read get_terrain for where the material actually is — never guess coordinates."
    ),
    capability="motion.excavator",
    params=("dig_x_m", "dig_y_m", "dump_x_m", "dump_y_m"),
    required_params=("dig_x_m", "dig_y_m", "dump_x_m", "dump_y_m"),
    param_schema={
        "dig_x_m": {
            "type": "number",
            "description": "dig ground point X in base frame (metres)",
        },
        "dig_y_m": {
            "type": "number",
            "description": "dig ground point Y in base frame (metres)",
        },
        "dump_x_m": {
            "type": "number",
            "description": "dump ground point X in base frame (metres)",
        },
        "dump_y_m": {
            "type": "number",
            "description": "dump ground point Y in base frame (metres)",
        },
    },
    requires=("payload.clear",),
    provides=("payload.clear",),
    invalidates=("body.home",),
    invalidates_locations=True,  # digging reshapes the terrain: prior pile observations go stale
    result=DigResult | DigFailure,
    tags=("motion",),
)

GET_TERRAIN = ActionSpec(
    name="get_terrain",
    description=(
        "Read the terrain truth this body's environment reports: material piles as {name, x_m, "
        "y_m, volume_m3} ground points in the base frame (metres, REP-103). Zero-cost "
        "observation — call it before dig instead of guessing where material is, and after "
        "digging to see what moved."
    ),
    capability="sensing.terrain",
    params=(),
    result=TerrainScan,
    tags=("sensing",),
)
