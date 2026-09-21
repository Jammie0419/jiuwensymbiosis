# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Phase A: LLM① task parser — natural-language task → structured intent."""

from jiuwensymbiosis.agent.fast import planner


def test_parse_task_structured(monkeypatch):
    monkeypatch.setattr(
        planner,
        "_chat",
        lambda *a, **k: '{"targets":["white box"],"destination":"table behind",'
        '"mode":"continuous","count":10,"intent":"carry"}',
    )
    out = planner.parse_task("把10个白箱子搬到身后桌子上", api_base="x", model_name="m")
    assert out["targets"] == ["white box"]
    assert out["count"] == 10
    assert out["mode"] == "continuous"
    assert out["destination"] == "table behind"
    assert out["intent"] == "carry"


def test_parse_task_defaults_on_bad_json(monkeypatch):
    monkeypatch.setattr(planner, "_chat", lambda *a, **k: "not json at all")
    out = planner.parse_task("抓杯子", api_base="x", model_name="m")
    # graceful, well-shaped default
    assert isinstance(out, dict)
    assert isinstance(out.get("targets"), list)
    assert out.get("count") is None
    assert out.get("mode") in ("single", "multi", "continuous")


def test_parse_task_fills_missing_fields(monkeypatch):
    # LLM returns only targets; parser fills the rest with safe defaults.
    monkeypatch.setattr(planner, "_chat", lambda *a, **k: '{"targets": ["red cup"]}')
    out = planner.parse_task("拿起红杯子", api_base="x", model_name="m")
    assert out["targets"] == ["red cup"]
    assert out["destination"] is None
    assert out["count"] is None
    assert out["mode"] == "single"


def test_default_timeout_suits_a_reasoning_model(monkeypatch):
    """默认读超时不能按快模型给。

    实测（glm-5，2026-09-21）：默认 20 s 时连续两次 `attempt N/3 failed: The read
    operation timed out`，第三次才回——白等 40 s 还只是意图解析。慢端点可以在
    YAML 里调 agent.intent_timeout_s。
    """
    seen: dict = {}

    def _fake_chat(*args, **kwargs):
        seen.update(kwargs)
        return "{}"

    monkeypatch.setattr(planner, "_chat", _fake_chat)
    planner.parse_task("抓杯子", api_base="x", model_name="m")
    assert seen["timeout_s"] >= 60.0


def test_timeout_is_forwarded(monkeypatch):
    seen: dict = {}

    def _fake_chat(*args, **kwargs):
        seen.update(kwargs)
        return "{}"

    monkeypatch.setattr(planner, "_chat", _fake_chat)
    planner.parse_task("抓杯子", api_base="x", model_name="m", timeout_s=123.0)
    assert seen["timeout_s"] == 123.0
