# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the runner-side planning-level events (step / replan)."""

from __future__ import annotations

import types

import jiuwensymbiosis.agent.fast.runner as runner_module
from jiuwensymbiosis.agent.fast.runner import run_sequence
from jiuwensymbiosis.agent.fast.sequence import ActionStep
from jiuwensymbiosis.agent.step_context import current_runner_step


class _Api:
    """Minimal action surface; records the runner step seen by each op."""

    def __init__(self):
        self.seen: list[int | None] = []
        self.recovered = False

    def ok_op(self, value: int = 1):
        self.seen.append(current_runner_step())
        return {"ok": True, "value": value}

    def fail_op(self):
        self.seen.append(current_runner_step())
        raise ValueError("boom")

    def release_effector(self):
        self.recovered = True
        return {"ok": True}

    def home(self):
        return {"ok": True}


def _session(api):
    return types.SimpleNamespace(api=api, env=None)


def _index(api):
    return {
        "ok_op": api.ok_op,
        "fail_op": api.fail_op,
        "release_effector": api.release_effector,
        "home": api.home,
    }


def test_step_events_carry_resolved_params_and_contextvar():
    api = _Api()
    steps = [ActionStep("ok_op", {"value": 3}), ActionStep("fail_op", {})]
    events: list[dict] = []

    result = run_sequence(_session(api), steps, action_index=_index(api), trace_sink=events.append)

    assert result["ok"] is False
    step_events = [event for event in events if event["type"] == "step"]
    assert [event["step_id"] for event in step_events] == [0, 1]
    assert step_events[0]["ok"] is True
    assert step_events[0]["op"] == "ok_op"
    assert step_events[0]["params"] == {"value": 3}
    assert step_events[0]["result"] == {"ok": True, "value": 3}
    assert step_events[1]["ok"] is False
    assert step_events[1]["op"] == "fail_op"
    assert "ValueError: boom" in step_events[1]["error"]
    assert step_events[1]["recovery_managed"] is False
    # The executor wrapper bound the runner id in this thread.
    assert api.seen == [0, 1]
    assert api.recovered is True  # the failure path still retreats


def test_business_failure_emits_replan_event_and_continues():
    api = _Api()
    steps = [ActionStep("fail_op", {}), ActionStep("ok_op", {"value": 2})]
    events: list[dict] = []
    reasons: list[str] = []

    def replan(_world, why):
        reasons.append(why)
        return [ActionStep("ok_op", {"value": 9})]

    result = run_sequence(
        _session(api),
        steps,
        action_index=_index(api),
        replan=replan,
        max_replans=1,
        trace_sink=events.append,
    )

    assert result["ok"] is True
    assert result["replans"] == 1
    assert reasons and "fail_op 失败" in reasons[0]
    replans = [event for event in events if event["type"] == "replan"]
    assert len(replans) == 1
    event = replans[0]
    assert event["trigger"] == "business_failure"
    assert event["at_step"] == 0
    assert event["ok"] is True
    assert event["budget"] == {"used": 1, "max": 1}
    assert [step["op"] for step in event["old_remaining"]] == ["ok_op"]
    assert [step["op"] for step in event["new_steps"]] == ["ok_op"]
    assert [step["params"] for step in event["new_steps"]] == [{"value": 9}]


def test_failed_replan_is_recorded_and_original_failure_stands():
    api = _Api()
    steps = [ActionStep("fail_op", {})]
    events: list[dict] = []

    def replan(_world, _why):
        raise RuntimeError("planner down")

    result = run_sequence(
        _session(api),
        steps,
        action_index=_index(api),
        replan=replan,
        max_replans=1,
        trace_sink=events.append,
    )

    assert result["ok"] is False
    assert result["replans"] == 0
    replans = [event for event in events if event["type"] == "replan"]
    assert len(replans) == 1
    assert replans[0]["ok"] is False
    assert "planner down" in replans[0]["error"]
    assert replans[0]["new_steps"] == []


def test_empty_replan_is_recorded():
    api = _Api()
    steps = [ActionStep("fail_op", {})]
    events: list[dict] = []

    result = run_sequence(
        _session(api),
        steps,
        action_index=_index(api),
        replan=lambda _world, _why: [],
        max_replans=1,
        trace_sink=events.append,
    )

    assert result["ok"] is False
    replans = [event for event in events if event["type"] == "replan"]
    assert replans[0]["ok"] is False
    assert replans[0]["error"] == "re-planner returned no steps"


def test_sink_failure_never_changes_the_run():
    api = _Api()
    steps = [ActionStep("ok_op", {})]

    def boom(_event):
        raise RuntimeError("sink down")

    result = run_sequence(_session(api), steps, action_index=_index(api), trace_sink=boom)

    assert result["ok"] is True
    assert result["steps_done"] == 1


def test_without_sink_result_is_unchanged_and_replans_is_zero():
    api = _Api()
    result = run_sequence(_session(api), [ActionStep("ok_op", {})], action_index=_index(api))

    assert result["ok"] is True
    assert result["replans"] == 0
    assert result["env_keys"] == []


def test_drift_kinds_are_structured(monkeypatch):
    meta = types.SimpleNamespace(requires=("payload.clear",), consumes_location=False)

    monkeypatch.setattr(runner_module, "_self_state_drift", lambda *_args: "bucket is loaded")
    assert runner_module._drift(ActionStep("dig", {}), {"dig": meta}, None, {}) == (
        "self_state",
        "bucket is loaded",
    )

    monkeypatch.setattr(runner_module, "_self_state_drift", lambda *_args: None)
    monkeypatch.setattr(runner_module, "_location_drift", lambda *_args: "sensing is stale")
    assert runner_module._drift(ActionStep("dig", {}), {"dig": meta}, None, {}) == (
        "location",
        "sensing is stale",
    )

    assert runner_module._drift(ActionStep("unknown", {}), {"dig": meta}, None, {}) is None
