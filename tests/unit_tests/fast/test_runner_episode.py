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
    assert result["failed_replans"] == 0
    assert reasons and "fail_op 失败" in reasons[0]
    replans = [event for event in events if event["type"] == "replan"]
    assert len(replans) == 1
    event = replans[0]
    assert event["trigger"] == "business_failure"
    assert event["at_step"] == 0
    assert event["ok"] is True
    # The budget reports the counters AS THEY WERE when the attempt started —
    # a success increments afterwards, a failure never does.
    assert event["budget"] == {"used": 0, "max": 1, "failed": 0}
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
    # A failed compile burns the failed budget, never the success budget.
    assert result["failed_replans"] == 1
    replans = [event for event in events if event["type"] == "replan"]
    assert len(replans) == 1
    assert replans[0]["ok"] is False
    assert "planner down" in replans[0]["error"]
    assert replans[0]["new_steps"] == []


def test_failed_replans_have_their_own_cap(monkeypatch):
    """Location drift + a planner that always fails: each failed compile costs one
    failed-budget slot; once that cap is hit no more LLM attempts are made, but
    the steps still dispatch (a plan we cannot improve still beats none)."""
    api = _Api()
    calls = []

    def replan(_world, _why):
        calls.append(1)
        raise RuntimeError("planner down")

    monkeypatch.setattr(
        runner_module,
        "_drift",
        lambda *_a: ("location", "sensing is stale"),
    )
    steps = [ActionStep("ok_op", {}), ActionStep("ok_op", {}), ActionStep("ok_op", {})]
    result = run_sequence(
        _session(api),
        steps,
        action_index=_index(api),
        replan=replan,
        max_replans=2,
        max_failed_replans=2,
    )

    assert result["ok"] is True  # the steps themselves succeeded
    assert result["replans"] == 0
    assert result["failed_replans"] == 2
    assert len(calls) == 2  # capped — the third drift does not buy an LLM round trip


def test_predicted_failure_does_not_buy_a_second_replan(monkeypatch):
    """self_state drift + a failed re-plan + the step then really fails: the
    step is still DISPATCHED (a best-effort drift belief must never take away an
    execution — the action's own check is the arbiter), but the confirmed
    prediction does not buy another LLM round trip (observed live: +137 s of
    compile that produced nothing)."""
    api = _Api()
    events: list[dict] = []
    attempts: list[str] = []

    def replan(_world, why):
        attempts.append(why)
        raise RuntimeError("planner down")

    monkeypatch.setattr(
        runner_module,
        "_drift",
        lambda step, *_a: ("self_state", "dump_soil requires payload.held") if step.op == "fail_op" else None,
    )
    steps = [ActionStep("ok_op", {"value": 1}), ActionStep("fail_op", {})]
    result = run_sequence(
        _session(api),
        steps,
        action_index=_index(api),
        replan=replan,
        trace_sink=events.append,
    )

    assert result["ok"] is False
    assert result["replans"] == 0
    assert result["failed_replans"] == 1
    # The step really ran (belief is not an interlock) and really failed...
    assert api.seen == [0, 1]
    failed = [record for record in result["steps"] if record["op"] == "fail_op"]
    assert failed and failed[0]["ok"] is False
    # ...but exactly ONE replan attempt was made — for the drift, never a
    # second business-failure pass over the same contradiction.
    assert len(attempts) == 1
    replans = [event for event in events if event["type"] == "replan"]
    assert len(replans) == 1 and replans[0]["ok"] is False


def test_replan_recurrence_is_rejected_not_reexecuted():
    """The re-plan re-emitted the just-failed step unchanged and first — it must
    NOT be adopted and re-executed (observed live: the same unreachable dump
    point ran twice and spent the whole budget)."""
    api = _Api()
    steps = [ActionStep("fail_op", {})]
    events: list[dict] = []

    def replan(_world, _why):
        return [ActionStep("fail_op", {})]  # byte-identical recurrence

    result = run_sequence(
        _session(api),
        steps,
        action_index=_index(api),
        replan=replan,
        trace_sink=events.append,
    )

    assert result["ok"] is False
    assert result["replans"] == 0
    assert result["failed_replans"] == 1
    # The doomed step ran ONCE, never twice.
    assert api.seen == [0]
    replans = [event for event in events if event["type"] == "replan"]
    assert any(event.get("trigger") == "recurrence" and not event["ok"] for event in replans)


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
