# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""TerrainDriver — the terrain-truth driver protocol slice (extension-owned).

Same role as the core optional-driver protocols (VisionDriver etc.), owned by
this extension because ``sensing.terrain`` is registered by this extension.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["TerrainDriver"]


@runtime_checkable
class TerrainDriver(Protocol):
    """Optional terrain-truth slice — the simulator seam for earthmoving bodies
    (capability ``sensing.terrain``; a sim backend reports it, real hardware
    rarely can)."""

    def read_terrain(self) -> list[dict[str, Any]]:
        """Material piles as {name, x_m, y_m, volume_m3} ground points in the
        base frame, in metres (REP-103). Empty list = no piles reported."""
