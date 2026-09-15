# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Session builder: YAML → ready-to-connect excavator session (no registration
call anywhere — the naming convention IS the registry)."""

from jiuwensymbiosis.adapters._common.builder import make_builder
from jiuwensymbiosis.adapters.agx_excavator.api import AgxExcavatorApi
from jiuwensymbiosis.adapters.agx_excavator.config import AgxExcavatorConfig
from jiuwensymbiosis.adapters.agx_excavator.env import AgxExcavatorEnv

build_agx_excavator_session = make_builder(AgxExcavatorConfig, AgxExcavatorEnv, AgxExcavatorApi)
