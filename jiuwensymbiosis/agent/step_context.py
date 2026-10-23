# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Propagate the runner's current step id into tool-call hooks.

The fast path executes a plan step through ``run_sequence`` → executor →
``ability_manager`` → ``TraceRail``. The runner knows the step id; TraceRail
records one entry per tool call and does not. A context variable bridges them
without widening the ``Executor`` signature (which every fake executor in the
tests relies on).

The binding wrapper must sit **inside** the cancellation wrapper: a GUI cancel
path dispatches the call on a fresh thread, so the context must be set on that
thread for ``asyncio.run`` inside ``ability_exec`` to inherit it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

__all__ = ["bind_executor", "current_runner_step", "runner_step"]

_CURRENT_STEP: ContextVar[int | None] = ContextVar("jiuwensymbiosis_runner_step", default=None)


def current_runner_step() -> int | None:
    """The runner step id active in this context, or ``None`` outside a step."""
    return _CURRENT_STEP.get()


@contextmanager
def runner_step(step_id: int | None) -> Iterator[None]:
    """Bind ``step_id`` for the duration of the block (nested-safe)."""
    token = _CURRENT_STEP.set(step_id)
    try:
        yield
    finally:
        _CURRENT_STEP.reset(token)


def bind_executor(
    executor: Callable[[str, dict[str, Any]], dict[str, Any]],
    state: Mapping[str, Any],
) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Wrap ``executor`` so each call carries ``state["current_step"]``.

    ``state`` is the runner's shared mutable dict; the wrapper reads it at call
    time, so the runner only has to assign ``current_step`` before dispatch.
    """

    def _run(op: str, params: dict[str, Any]) -> dict[str, Any]:
        step_id = state.get("current_step")
        with runner_step(step_id if isinstance(step_id, int) else None):
            return executor(op, params)

    return _run
