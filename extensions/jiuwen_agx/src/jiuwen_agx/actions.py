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
        "Execute one excavator dig-and-dump cycle. The dig/dump points are bucket-tip poses "
        "in the base frame, in METRES (REP-103: +x is the machine's forward, +y its left, "
        "-y its right). x/y are required; z is optional — a body "
        "with an IK work cycle uses it as the bucket-tip height, while a keyframe-only body "
        "uses its calibrated depth and ignores z. dig_pitch_deg/dump_pitch_deg are optional "
        "tool pitches; yaw follows atan2(y, x). The cut runs with the bucket open and curls "
        "inward; the dump point is reached while carrying the bucket closed, then the bucket "
        "opens OUTWARD there. The cut drags inward by about 1 m, so the dig point must leave "
        "room for it: keep the dig radius >= about 4 m (the drag endpoint must stay reachable); "
        "if the pile is closer, move the undercarriage back first and re-read get_terrain. "
        "The dump point must be high/far enough to reach while carrying "
        "(roughly 4.5-5.5 m out and 3-4 m high); a low/near point is refused with a readable "
        "error. If dump_z_m is omitted the body uses the terrain height there plus its "
        "configured dump clearance. The points must be reachable; if a spot is out of reach, "
        "move the undercarriage (navigate_relative) first, re-read get_terrain, then dig. "
        "Refuses while the bucket is still loaded; dump or home first. Read get_terrain for "
        "where the material actually is — never guess coordinates."
    ),
    capability="motion.excavator",
    params=(
        "dig_x_m",
        "dig_y_m",
        "dig_z_m",
        "dump_x_m",
        "dump_y_m",
        "dump_z_m",
        "dig_pitch_deg",
        "dump_pitch_deg",
    ),
    required_params=("dig_x_m", "dig_y_m", "dump_x_m", "dump_y_m"),
    param_schema={
        "dig_x_m": {
            "type": "number",
            "description": "dig bucket-tip X in base frame (metres)",
        },
        "dig_y_m": {
            "type": "number",
            "description": "dig bucket-tip Y in base frame (metres)",
        },
        "dig_z_m": {
            "type": "number",
            "description": "dig bucket-tip Z in base frame (metres); optional for keyframe-only bodies",
        },
        "dump_x_m": {
            "type": "number",
            "description": "dump bucket-tip X in base frame (metres)",
        },
        "dump_y_m": {
            "type": "number",
            "description": "dump bucket-tip Y in base frame (metres)",
        },
        "dump_z_m": {
            "type": "number",
            "description": "dump bucket-tip Z in base frame (metres); optional",
        },
        "dig_pitch_deg": {
            "type": "number",
            "description": "dig tool pitch in degrees; optional, defaults to flat",
        },
        "dump_pitch_deg": {
            "type": "number",
            "description": "dump tool pitch in degrees; optional, defaults to a downward dump",
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
        "y_m, z_m, volume_m3} points in the base frame (metres, REP-103). z_m is the pile "
        "surface height at that point. Zero-cost observation — call it before dig instead of "
        "guessing where material is, and after digging (or after moving the base) to see what "
        "moved and where the pile now is in the new base frame."
    ),
    capability="sensing.terrain",
    params=(),
    produces_location=True,
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
        "Execute one learned-policy dig-and-dump cycle on bucket-tip poses in the base frame, "
        "in METRES (REP-103). x/y are required; the policy supplies its own calibrated "
        "dig/dump depths. The body opens the bucket fully at the dump. The points must be "
        "reachable; move the undercarriage first if needed, then re-read get_terrain. "
        "Refuses while the bucket is still loaded. Use dig for the scripted cycle; use "
        "act_exec when the goal or material layout is unusual and the learned trajectory may "
        "generalise better."
    ),
    capability="policy.act",
    params=(
        "dig_x_m",
        "dig_y_m",
        "dump_x_m",
        "dump_y_m",
    ),
    required_params=("dig_x_m", "dig_y_m", "dump_x_m", "dump_y_m"),
    param_schema={
        "dig_x_m": {
            "type": "number",
            "description": "dig bucket-tip X in base frame (metres)",
        },
        "dig_y_m": {
            "type": "number",
            "description": "dig bucket-tip Y in base frame (metres)",
        },
        "dump_x_m": {
            "type": "number",
            "description": "dump bucket-tip X in base frame (metres)",
        },
        "dump_y_m": {
            "type": "number",
            "description": "dump bucket-tip Y in base frame (metres)",
        },
    },
    requires=("payload.clear",),
    provides=("payload.clear",),
    invalidates=("body.home",),
    invalidates_locations=True,  # digging reshapes the terrain: prior pile observations go stale
    result=ActResult | ActFailure,
    tags=("motion",),
)
