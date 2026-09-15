# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Simulator substrate — shared by every simulated machine adapter.

Adding a simulated machine (AGX or otherwise) = a thin adapter package that
reuses everything here; see ``docs/zh/how-to/add-sim-machine.md`` for the recipe.
Adapters import these submodules directly; importing this package stays
side-effect free.
"""

from jiuwensymbiosis.adapters._common.sim.api import SimMachineApi
from jiuwensymbiosis.adapters._common.sim.backend import (
    BACKENDS,
    InProcessAgxBackend,
    MockSimBackend,
    RemoteSimBackend,
    SimBackend,
    create_backend,
    register_backend,
)
from jiuwensymbiosis.adapters._common.sim.config import SimMachineConfig
from jiuwensymbiosis.adapters._common.sim.driver import SimMachineDriver
from jiuwensymbiosis.adapters._common.sim.env import SimMachineEnv

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
