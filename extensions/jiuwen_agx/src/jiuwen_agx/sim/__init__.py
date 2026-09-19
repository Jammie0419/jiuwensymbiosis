# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Simulator substrate — shared by every simulated machine adapter.

Adding a simulated machine (AGX or otherwise) = a thin adapter package that
reuses everything here; see ``docs/zh/how-to/add-sim-machine.md`` for the recipe.
Adapters import these submodules directly; importing this package stays
side-effect free.
"""

from jiuwen_agx.sim.api import SimMachineApi
from jiuwen_agx.sim.backend import (
    BACKENDS,
    InProcessAgxBackend,
    MockSimBackend,
    RemoteSimBackend,
    SimBackend,
    create_backend,
    register_backend,
)
from jiuwen_agx.sim.config import SimMachineConfig
from jiuwen_agx.sim.driver import SimMachineDriver
from jiuwen_agx.sim.env import SimMachineEnv

__all__ = [
    "BACKENDS",
    "InProcessAgxBackend",
    "MockSimBackend",
    "RemoteSimBackend",
    "SimBackend",
    "SimMachineApi",
    "SimMachineConfig",
    "SimMachineDriver",
    "SimMachineEnv",
    "create_backend",
    "register_backend",
]
