# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""The two trace layers share one key: ``TraceEntry.runner_step``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jiuwensymbiosis.agent.step_context import runner_step
from jiuwensymbiosis.agent.trace import TraceRail
from tests.helpers import FakeCtx, FakeInputs, make_mock_session


@pytest.fixture
def trace_rail(tmp_path):
    return TraceRail(make_mock_session(), workspace=str(tmp_path), save_frames=False)


async def _run_step(rail: TraceRail, *, tool_name: str = "goto_xyzr", tool_args=None):
    ctx = FakeCtx(FakeInputs(tool_name=tool_name, tool_args=tool_args or {}))
    await rail.before_tool_call(ctx)
    ctx.inputs.tool_result = {"ok": True}
    await rail.after_tool_call(ctx)
    return ctx


async def test_entry_records_runner_step_from_contextvar(trace_rail):
    await trace_rail.before_invoke(FakeCtx(conversation_id="c", query="q"))

    with runner_step(7):
        await _run_step(trace_rail)

    entry = trace_rail.trace.entries[-1]
    assert entry.runner_step == 7
    assert entry.to_dict()["runner_step"] == 7


async def test_entry_runner_step_is_none_outside_a_step(trace_rail):
    await trace_rail.before_invoke(FakeCtx(conversation_id="c", query="q"))
    await _run_step(trace_rail)

    assert trace_rail.trace.entries[-1].runner_step is None


async def test_last_trace_path_is_exposed_after_finalize(trace_rail, tmp_path):
    await trace_rail.before_invoke(FakeCtx(conversation_id="ep", query="q"))
    await _run_step(trace_rail)

    path = trace_rail.finalize()

    assert path is not None
    assert trace_rail.last_trace_path == path
    assert Path(path).is_file()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert data["entries"][-1]["runner_step"] is None

    # A new invoke must not inherit the previous run's file reference.
    await trace_rail.before_invoke(FakeCtx(conversation_id="ep2", query="q"))
    assert trace_rail.last_trace_path is None
