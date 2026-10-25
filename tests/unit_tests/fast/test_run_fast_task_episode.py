# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""End-to-end wiring tests for the fast path's planning-level trace."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from jiuwensymbiosis.agent import ModelSpec, RobotAgentConfig
from jiuwensymbiosis.agent.episode_trace import load_episode
from jiuwensymbiosis.agent.fast import PlanResult
from jiuwensymbiosis.agent.run import run_fast_task
from tests.unit_tests.fast.test_run_fast_task import (
    _FIXED_SEQUENCE,
    _install_fast_agent_doubles,
    _make_session,
)


def _cfg(tmp_path: Path, **overrides) -> RobotAgentConfig:
    cfg = RobotAgentConfig()
    cfg.model_spec = ModelSpec()
    cfg.exec_mode = "fastagent"
    cfg.workspace = str(tmp_path)
    cfg.enable_visual_feedback = False
    cfg.enable_skill = True
    cfg.enable_tracing = True
    cfg.enable_episode_trace = True
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _run(monkeypatch, session, cfg, *, query="pick the box", conv_id="ep-test", plan=None):
    _install_fast_agent_doubles(monkeypatch, session)
    result = plan if plan is not None else PlanResult(sequence=_FIXED_SEQUENCE, tier="skill", skills=("visual_pick",))
    patcher = (
        mock.patch("jiuwensymbiosis.agent.fast.plan_task", side_effect=result)
        if isinstance(result, list)
        else mock.patch("jiuwensymbiosis.agent.fast.plan_task", return_value=result)
    )
    with patcher:
        with session:
            return run_fast_task(session, query, cfg, conversation_id=conv_id)


def _episode_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "episodes").glob("*.jsonl"))


def test_episode_records_plan_steps_and_outcome(tmp_path, monkeypatch):
    session = _make_session()
    cfg = _cfg(tmp_path)

    result = _run(monkeypatch, session, cfg)

    files = _episode_files(tmp_path)
    assert len(files) == 1
    assert result["episode_path"] == str(files[0])
    doc = load_episode(files[0])
    assert doc.episode["query"] == "pick the box"
    assert doc.episode["mode"] == "fastagent"
    assert doc.episode["versions"]["harness"] == "v0"
    assert doc.episode["versions"]["exec_mode"] == "fastagent"
    assert len(doc.plan_attempts) == 1
    assert doc.plan_attempts[0]["trigger"] == "initial"
    assert doc.plan_attempts[0]["tier"] == "skill"
    assert doc.plan_attempts[0]["skills"] == ["visual_pick"]
    assert [step["op"] for step in doc.steps] == [item["op"] for item in _FIXED_SEQUENCE]
    assert all(step["ok"] for step in doc.steps)
    assert doc.outcome["ok"] is True
    assert doc.outcome["steps_done"] == len(_FIXED_SEQUENCE)
    assert doc.outcome["replans"] == 0
    assert doc.outcome["attempts"] == 1
    assert doc.outcome["final_tier"] == "skill"
    assert doc.outcome["trace_path"] is not None
    assert Path(doc.outcome["trace_path"]).is_file()

    # The tool-level entries carry the same runner ids as the episode steps.
    trace = json.loads(Path(doc.outcome["trace_path"]).read_text(encoding="utf-8"))
    assert [entry["runner_step"] for entry in trace["entries"]] == [step["step_id"] for step in doc.steps]


def test_episode_records_a_replan_attempt(tmp_path, monkeypatch):
    session = _make_session()
    cfg = _cfg(tmp_path)

    def boom_goto(*_args, **_kwargs):
        raise ValueError("out of reach")

    monkeypatch.setattr(session.api, "goto_xyzr", boom_goto, raising=False)
    plans = [
        PlanResult(sequence=_FIXED_SEQUENCE, tier="skill", skills=("visual_pick",)),
        PlanResult(sequence=[{"op": "home"}], tier="action", reason="after failure"),
    ]

    result = _run(monkeypatch, session, cfg, plan=plans)

    assert result["ok"] is True
    assert result["replans"] == 1
    doc = load_episode(_episode_files(tmp_path)[0])
    assert [attempt["trigger"] for attempt in doc.plan_attempts] == ["initial", "replan"]
    assert doc.plan_attempts[1]["tier"] == "action"
    replans = doc.replans
    assert len(replans) == 1
    assert replans[0]["trigger"] == "business_failure"
    assert replans[0]["ok"] is True
    assert replans[0]["at_step"] is not None
    failed = [step for step in doc.steps if not step["ok"]]
    assert failed and failed[0]["op"] == "goto_xyzr"
    assert doc.outcome["replans"] == 1
    assert doc.outcome["attempts"] == 2
    assert doc.outcome["final_tier"] == "action"


def test_failed_run_homes_as_cleanup(tmp_path, monkeypatch):
    """失败收场兜底：任务失败后尽力 home，别把 carry 位形留给下一条任务。"""
    session = _make_session()
    cfg = _cfg(tmp_path)

    def boom_goto(*_args, **_kwargs):
        raise ValueError("out of reach")

    monkeypatch.setattr(session.api, "goto_xyzr", boom_goto, raising=False)
    # One plan, returned for every re-plan too: the run must end in failure.
    plan = PlanResult(sequence=_FIXED_SEQUENCE, tier="skill", skills=("visual_pick",))

    result = _run(monkeypatch, session, cfg, plan=plan)

    assert result["ok"] is False
    assert result["cleanup"] == "home_ok"
    doc = load_episode(_episode_files(tmp_path)[0])
    cleanup_steps = [step for step in doc.steps if step.get("cleanup")]
    assert len(cleanup_steps) == 1
    assert cleanup_steps[0]["op"] == "home" and cleanup_steps[0]["ok"] is True
    assert doc.outcome["cleanup"] == "home_ok"
    assert doc.outcome["failed_replans"] >= 0  # reported for traceability


def test_cleanup_home_skipped_while_holding_payload(tmp_path, monkeypatch):
    session = _make_session()
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(session.env, "holding_payload", True, raising=False)

    def boom_goto(*_args, **_kwargs):
        raise ValueError("out of reach")

    monkeypatch.setattr(session.api, "goto_xyzr", boom_goto, raising=False)
    plan = PlanResult(sequence=_FIXED_SEQUENCE, tier="skill", skills=("visual_pick",))

    result = _run(monkeypatch, session, cfg, plan=plan)

    assert result["ok"] is False
    # 带载不盲收（直接 home 会把料洒在机器上）。
    assert result["cleanup"] == "skipped_holding_payload"
    doc = load_episode(_episode_files(tmp_path)[0])
    assert not [step for step in doc.steps if step.get("cleanup")]


def test_no_episode_file_when_disabled(tmp_path, monkeypatch):
    session = _make_session()
    cfg = _cfg(tmp_path, enable_episode_trace=False)

    result = _run(monkeypatch, session, cfg)

    assert result["episode_path"] is None
    assert not (tmp_path / "episodes").exists()


def test_planning_failure_still_writes_an_outcome(tmp_path, monkeypatch):
    session = _make_session()
    cfg = _cfg(tmp_path)
    _install_fast_agent_doubles(monkeypatch, session)

    with mock.patch("jiuwensymbiosis.agent.fast.plan_task", side_effect=RuntimeError("no plan")):
        with session:
            result = run_fast_task(session, "pick", cfg, conversation_id="boom")

    assert result["ok"] is False
    assert result["reason"].startswith("compile_failed")
    doc = load_episode(_episode_files(tmp_path)[0])
    assert doc.plan_attempts[0]["ok"] is False
    assert "no plan" in doc.plan_attempts[0]["error"]
    assert doc.outcome["ok"] is False
    assert doc.outcome["error"] == "compile_failed: no plan"


def test_missing_model_spec_is_recorded(tmp_path):
    session = _make_session()
    cfg = _cfg(tmp_path, model_spec=None)

    with session:
        result = run_fast_task(session, "pick", cfg, conversation_id="no-model")

    assert result["reason"] == "no_model_spec"
    doc = load_episode(_episode_files(tmp_path)[0])
    assert doc.outcome["ok"] is False
    assert doc.outcome["error"] == "no_model_spec"
