# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the planning-level replay CLI."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from jiuwensymbiosis.cli_episode import episode_replay


def _write_episode(path: Path) -> None:
    events = [
        {
            "type": "episode",
            "schema": "episode/1",
            "seq": 1,
            "episode_id": "e1",
            "query": "dig one bucket",
            "robot": "agx_excavator",
            "mode": "fastagent",
            "workspace": "/w",
            "versions": {"harness": "v0"},
        },
        {
            "type": "plan_attempt",
            "seq": 2,
            "attempt": 1,
            "trigger": "initial",
            "tier": "skill",
            "skills": ["excavate"],
            "validated_steps": ["get_terrain", "dig"],
            "ok": True,
        },
        {"type": "step", "seq": 3, "step_id": 0, "op": "get_terrain", "ok": True, "duration_s": 0.01},
        {
            "type": "step",
            "seq": 4,
            "step_id": 1,
            "op": "dig",
            "ok": False,
            "duration_s": 0.02,
            "error": "ValueError: out of reach",
        },
        {
            "type": "replan",
            "seq": 5,
            "at_step": 1,
            "trigger": "business_failure",
            "why": "dig 失败",
            "ok": True,
            "budget": {"used": 1, "max": 2},
            "old_remaining": [{"op": "home"}],
            "new_steps": [{"op": "navigate_relative"}],
        },
        {
            "type": "outcome",
            "seq": 6,
            "ok": True,
            "steps_done": 2,
            "replans": 1,
            "attempts": 2,
            "final_tier": "skill",
            "trace_path": "/w/traces/x.json",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )


def test_text_replay_prints_the_plan_timeline(tmp_path):
    path = tmp_path / "ep.jsonl"
    _write_episode(path)
    out = StringIO()

    assert episode_replay(str(path), out=out) == 0

    text = out.getvalue()
    assert "[plan #1] ok" in text
    assert "skills=['excavate']" in text
    assert "[s0] ok" in text
    assert "[s1] FAIL dig" in text
    assert "ValueError: out of reach" in text
    assert "[replan @s1] ok" in text
    assert "navigate_relative" in text
    assert "[outcome] ok=True" in text
    assert "tool trace: /w/traces/x.json" in text


def test_json_replay_returns_the_aggregated_events(tmp_path):
    path = tmp_path / "ep.jsonl"
    _write_episode(path)
    out = StringIO()

    assert episode_replay(str(path), out=out, as_json=True) == 0

    data = json.loads(out.getvalue())
    assert data["episode"]["episode_id"] == "e1"
    assert len(data["plan_attempts"]) == 1
    assert len(data["steps"]) == 2
    assert len(data["replans"]) == 1
    assert data["outcome"]["ok"] is True
    assert data["bad_lines"] == []


def test_missing_file_returns_one(tmp_path):
    assert episode_replay(str(tmp_path / "missing.jsonl"), out=StringIO()) == 1


def test_file_without_readable_events_returns_one(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")

    assert episode_replay(str(path), out=StringIO()) == 1
