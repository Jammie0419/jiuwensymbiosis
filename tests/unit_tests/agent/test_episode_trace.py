# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Tests for the planning-level (episode) trace writer/reader."""

from __future__ import annotations

import json
from pathlib import Path

from jiuwensymbiosis.agent.episode_trace import (
    EPISODE_SCHEMA,
    EpisodeRecorder,
    NullEpisodeRecorder,
    emit_safely,
    episode_path,
    iter_episode,
    load_episode,
)


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_recorder_writes_header_and_monotonic_seq(tmp_path):
    path = tmp_path / "ep.jsonl"
    with EpisodeRecorder(
        path,
        episode_id="task-1",
        query="dig one bucket",
        robot="agx_excavator",
        mode="fastagent",
        workspace=str(tmp_path),
        versions={"harness": "v0"},
        capabilities=["motion.excavator"],
    ) as rec:
        rec.intent(ok=True, intent={"targets": ["soil"]})
        rec.step(step_id=0, op="get_terrain", ok=True, result={"piles": []})
        rec.outcome(ok=True, steps_done=1)

    events = _events(path)
    assert [event["seq"] for event in events] == [1, 2, 3, 4]
    assert [event["type"] for event in events] == ["episode", "intent", "step", "outcome"]
    assert all(event["schema"] == EPISODE_SCHEMA for event in events)
    header = events[0]
    assert header["episode_id"] == "task-1"
    assert header["run_token"] == path.stem
    assert header["query"] == "dig one bucket"
    assert header["versions"] == {"harness": "v0"}
    assert header["capabilities"] == ["motion.excavator"]


def test_recorder_sanitizes_non_finite_and_objects(tmp_path):
    path = tmp_path / "ep.jsonl"

    class Payload:
        def __init__(self):
            self.value = 3

    with EpisodeRecorder(path, episode_id="e") as rec:
        rec.plan_attempt(
            trigger="initial",
            raw=[
                {
                    "op": "dig",
                    "params": {
                        "nan": float("nan"),
                        "inf": float("inf"),
                        "labels": frozenset({"soil"}),
                        "payload": Payload(),
                    },
                }
            ],
        )
        rec.outcome(ok=True)

    # json.loads succeeded for every line, i.e. strict JSON (no NaN literal).
    events = _events(path)
    params = events[1]["raw"][0]["params"]
    assert params["nan"] is None
    assert params["inf"] is None
    assert params["labels"] == ["soil"]
    assert params["payload"] == {"value": 3}


def test_recorder_is_best_effort_on_write_failure(tmp_path):
    path = tmp_path / "ep.jsonl"
    rec = EpisodeRecorder(path, episode_id="e")

    class BrokenHandle:
        def write(self, _line):
            raise OSError("disk full")

        def flush(self):
            return None

        def close(self):
            return None

    rec._fh = BrokenHandle()
    rec.step(step_id=0, op="dig", ok=True)  # must not raise
    rec.step(step_id=1, op="home", ok=True)  # no-op once broken
    rec.close()

    assert rec._broken is True
    assert len(_events(path)) == 1  # only the header made it to disk


def test_recorder_writes_outcome_on_exception(tmp_path):
    path = tmp_path / "ep.jsonl"
    try:
        with EpisodeRecorder(path, episode_id="e"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    doc = load_episode(path)
    assert doc.outcome["ok"] is False
    assert doc.outcome["error"] == "RuntimeError: boom"


def test_reader_skips_bad_lines_and_keeps_the_rest(tmp_path):
    path = tmp_path / "ep.jsonl"
    path.write_text(
        '{"type":"episode","seq":1}\nnot json\n\n[1, 2]\n{"type":"outcome","ok":true,"seq":5}\n',
        encoding="utf-8",
    )

    doc = load_episode(path)
    assert [event["type"] for event in doc.events] == ["episode", "outcome"]
    assert doc.bad_lines == (2, 4)
    assert [event["type"] for event in iter_episode(path)] == ["episode", "outcome"]


def test_reader_tolerates_invalid_utf8(tmp_path):
    path = tmp_path / "ep.jsonl"
    path.write_bytes(b'{"type":"episode","seq":1}\n\xff\xfe\n{"type":"outcome","ok":true}\n')

    doc = load_episode(path)
    assert [event["type"] for event in doc.events] == ["episode", "outcome"]
    assert doc.bad_lines == (2,)


def test_episode_path_uses_episodes_dir_and_sanitizes_id(tmp_path):
    path = episode_path(tmp_path, None, "task/with:bad*chars")
    assert path.parent == tmp_path / "episodes"
    assert path.name.startswith("task_with_bad_chars_")
    assert path.suffix == ".jsonl"

    override = episode_path(tmp_path, tmp_path / "custom", "e")
    assert override.parent == tmp_path / "custom"


def test_null_recorder_is_a_noop(tmp_path):
    rec = NullEpisodeRecorder()
    rec.intent(ok=True)
    rec.world({"tokens": []})
    rec.plan_attempt(tier="skill")
    rec.step(step_id=0, op="x", ok=True)
    rec.replan(at_step=0, trigger="business_failure", why="x", ok=False)
    rec.outcome(ok=False)
    rec.close()
    assert rec.path is None
    assert not list(tmp_path.iterdir())


def test_emit_safely_swallows_sink_failure():
    def boom(_event):
        raise RuntimeError("sink down")

    emit_safely(boom, {"type": "step"})  # must not raise
    emit_safely(None, {"type": "step"})
