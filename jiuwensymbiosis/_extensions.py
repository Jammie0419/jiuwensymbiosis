# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Out-of-tree extension discovery — the entry-point seam.

An extension package declares adapter builders under the group
``jiuwensymbiosis.adapters`` in ITS OWN pyproject::

    [project.entry-points."jiuwensymbiosis.adapters"]
    agx_excavator = "jiuwen_agx.agx_excavator:build_agx_excavator_session"

The referenced object must be a session builder exposing ``.from_dict`` /
``.from_yaml`` (what ``adapters/_common/builder.make_builder`` produces).
Loading it also runs the extension package's own registration chain
(capabilities, actions, skills) — that is the documented contract, so by the
time a caller holds the builder, the vocabulary knows the extension's actions.

Used as the fallback by all three adapter-resolution sites: the terminal
runner (``examples/run_task.py``), the introspection CLI
(``introspect.build_session``), and the GUI registry.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib.metadata import entry_points
from typing import Any, Callable

logger = logging.getLogger(__name__)

ADAPTERS_GROUP = "jiuwensymbiosis.adapters"


@lru_cache(maxsize=1)
def discover_adapter_builders() -> dict[str, Callable[..., Any]]:
    """Return ``{name: builder}`` for every extension-declared adapter (cached).

    A builder that fails to import is skipped with a warning rather than
    breaking every other adapter.
    """
    builders: dict[str, Callable[..., Any]] = {}
    for ep in entry_points(group=ADAPTERS_GROUP):
        try:
            builders[ep.name] = ep.load()
        except Exception as exc:  # noqa: BLE001 - one bad extension must not sink the rest
            logger.warning("extension adapter %r failed to load: %s", ep.name, exc)
    return builders
