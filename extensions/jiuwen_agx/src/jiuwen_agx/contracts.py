# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""jiuwen_agx action result types — the simulation terrain & work-cycle contracts.

Mirrors the role of ``jiuwensymbiosis.contracts`` but owned by this extension:
``api/actions.py`` (the specs declared here) and the adapter implementations
both read these shapes. Keep this module dependency-free.
"""

from __future__ import annotations

from typing import Literal, TypedDict

__all__ = [
    "PileInfo",
    "TerrainScan",
    "DigResult",
    "DigFailure",
]


class PileInfo(TypedDict):
    """One material pile as the simulator's terrain truth reports it.

    Ground coordinates in the BASE frame, in metres (REP-103) — the same frame
    ``navigate_relative`` speaks, so a planner can go from a pile to a dig spot
    without unit conversions.
    """

    name: str
    x_m: float
    y_m: float
    volume_m3: float


class TerrainScan(TypedDict):
    """Success shape returned by ``get_terrain``.

    ``ok`` is required: the fast runner treats a bound step's return as a
    usable detection only when ``ok`` is truthy (same convention as the other
    sensing results).
    """

    ok: Literal[True]
    piles: list  # list[PileInfo]


class DigFailure(TypedDict):
    """Failure shape returned by ``dig``."""

    ok: Literal[False]
    error: str


class DigResult(TypedDict, total=False):
    """Success shape returned by ``dig`` — one completed dig-and-dump cycle.

    ``volume_m3`` is the volume the simulator (or the nominal bucket figure, on
    a mock) reports as actually moved; ``total=False`` because a backend that
    cannot measure it may omit it.
    """

    ok: Literal[True]
    volume_m3: float
    cycle_s: float
