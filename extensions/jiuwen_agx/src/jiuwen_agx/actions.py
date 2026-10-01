# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""jiuwen_agx ActionSpecs — simulator work cycles, terrain truth, policy actions.

Declared HERE (not in core ``api/actions.py``) and pushed into the shared
vocabulary through ``register_actions`` at package import. Locally declared
specs are first-class: ``@implements`` binds them, the fast planner plans them
(the action index reads the api instance's ``__tool_meta__``), and
``jiuwensymbiosis-actions --config`` shows them. Only the body-agnostic
``--vocabulary`` listing is unaware of them until this package is imported —
which any session build does via the entry-point discovery.
"""

from __future__ import annotations

from jiuwensymbiosis.api.actions import ActionSpec
from jiuwensymbiosis.api.decorators import (
    UnknownCapability,  # noqa: F401  (re-export convenience)
)

from jiuwen_agx.contracts import (
    ActFailure,
    ActResult,
    DigFailure,
    DigResult,
    TerrainScan,
)

__all__ = ["ACT_EXECUTE", "DIG", "GET_TERRAIN"]

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

# NOTE: params are deliberately IDENTICAL to dig (same names, same units, same
# annulus pre-check) so the planner can swap between the scripted and the
# learned cycle without re-deriving anything; the SKILL.md is where "when to
# use which" lives. tags=("motion",) matters: RecoveryRail claims motion-tagged
# tools, so a failed act_exec gets the same recovery path as dig.
ACT_EXECUTE = ActionSpec(
    name="act_exec",
    description=(
        "Execute one learned-policy dig-and-dump cycle: a trained ACT model watches the "
        "joint angles and the two ground points and produces the whole motion, so the "
        "trajectory adapts to the goal instead of replaying fixed keyframes. Both points "
        "are ground coordinates in the base frame, in METRES (REP-103), inside this body's "
        "reachable annulus — drive the undercarriage closer first if a spot is out of reach. "
        "Refuses while the bucket is still loaded. Read get_terrain for where the material "
        "actually is — never guess coordinates. Use dig instead when you want the "
        "well-tested scripted cycle; use act_exec when the goal or material layout is "
        "unusual and the learned trajectory may generalise better."
    ),
    capability="policy.act",
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
    result=ActResult | ActFailure,
    tags=("motion",),
)
