# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session builder: YAML → ready-to-connect excavator session (no registration
call anywhere — the naming convention IS the registry)."""

from jiuwensymbiosis.adapters._common.builder import make_builder

from jiuwen_agx.agx_excavator.api import AgxExcavatorApi
from jiuwen_agx.agx_excavator.config import AgxExcavatorConfig
from jiuwen_agx.agx_excavator.env import AgxExcavatorEnv


def _agx_resource_keys(cfg: AgxExcavatorConfig) -> tuple[str, ...]:
    """Reserve the simulator's command endpoint in the domain its config names.

    remote → the TCP bridge endpoint; inprocess → the loaded scene file; mock →
    a machine-local nominal key (an in-memory sim owns no shared resource, but
    runtime admission still wants the session identified — without it the
    config must carry a top-level ``physical_device_id`` to be admissible).
    """
    if cfg.backend == "remote":
        return (f"agx-bridge:{cfg.host}:{cfg.port}",)
    if cfg.backend == "inprocess":
        return (f"agx-scene:{cfg.scene_path or 'default'}",)
    return ("agx-mock",)


build_agx_excavator_session = make_builder(
    AgxExcavatorConfig,
    AgxExcavatorEnv,
    AgxExcavatorApi,
    resource_keys=_agx_resource_keys,
)
