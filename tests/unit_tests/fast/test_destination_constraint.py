# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""User-stated destination direction ("倒到左边3米处"): parse → constrain → check.

The failure this guards against: the task said "dump to the left", the compiler
planned ``dump_y_m = -1`` (the machine's RIGHT — it copied the example point from
an error message), the dig succeeded geometrically, and nothing could catch the
mismatch because direction was never part of any contract. These tests pin the
three layers: deterministic parsing of the free-text destination, constraint
rendering into both planning tiers, and the compile-loop check that turns a
wrong side into a compile retry instead of a physical dig.
"""

from __future__ import annotations

import json

from jiuwensymbiosis.agent.fast import planner


# --------------------------------------------------------------------------- parse
def test_destination_side_parsed_from_english_and_chinese():
    assert planner._parse_destination_side("left 3 meters") == "left"
    assert planner._parse_destination_side("倒到左边3米处") == "left"
    assert planner._parse_destination_side("right side") == "right"
    assert planner._parse_destination_side("table behind") == "behind"
    # A bare referent ("beside") states no side: nothing to enforce.
    assert planner._parse_destination_side("beside") is None
    assert planner._parse_destination_side(None) is None


def test_destination_distance_parsed():
    assert planner._parse_destination_distance("left 3 meters") == 3.0
    assert planner._parse_destination_distance("左边3米处") == 3.0
    assert planner._parse_destination_distance("left 2.5 m") == 2.5
    # No unit → no distance: a bare "left 3" must not invent one.
    assert planner._parse_destination_distance("left 3") is None
    assert planner._parse_destination_distance("beside") is None


def test_validate_parsed_fills_destination_fields():
    out = planner._validate_parsed({"destination": "left 3 meters"})
    assert out["destination_side"] == "left"
    assert out["destination_distance_m"] == 3.0

    out = planner._validate_parsed({"destination": "beside"})
    assert out["destination_side"] is None
    assert out["destination_distance_m"] is None


# --------------------------------------------------------------------------- render
def test_constraint_block_rendered_for_stated_sides():
    text = planner._format_destination_constraint("left", 3.0)
    assert "【用户方向约束】" in text
    assert "机身左侧" in text and "dump_y_m > 0" in text
    assert "3" in text

    assert planner._format_destination_constraint(None) == ""
    # ahead/behind render (no sign to check, but the intent is stated)
    assert "机身正前方" in planner._format_destination_constraint("ahead")


# --------------------------------------------------------------------------- check
def test_side_violation_is_reported():
    data = [{"op": "dig", "params": {"dump_x_m": 5.0, "dump_y_m": -1.0, "dump_z_m": 3.5}}]
    err = planner._check_destination_side(data, "left")
    assert err is not None
    assert "dump_y_m=-1" in err and "左侧" in err


def test_side_satisfied_passes():
    left = [{"op": "dig", "params": {"dump_y_m": 2.5}}]
    right = [{"op": "dig", "params": {"dump_y_m": -1.0}}]
    assert planner._check_destination_side(left, "left") is None
    assert planner._check_destination_side(right, "right") is None


def test_bindings_and_unprovable_sides_are_skipped():
    # A binding resolves at runtime; the check must not reject what it cannot prove.
    bound = [{"op": "dig", "params": {"dump_y_m": "terrain.piles[0].y_m"}}]
    assert planner._check_destination_side(bound, "left") is None
    # ahead/behind have no provable y-sign.
    ahead = [{"op": "dig", "params": {"dump_y_m": -1.0}}]
    assert planner._check_destination_side(ahead, "ahead") is None
    assert planner._check_destination_side([], "left") is None


def test_plans_that_rotate_the_base_skip_the_sign_check():
    """站点式移动会转整车：转向后"左"的参考系已变，最终帧符号证明不了方向。

    目标在左 10 m 的正确计划是 rotate(+90°) → 前进 → 卸点 ≈ (5, 0)（正前方），
    符号校验若照查会误杀；参考系锚定留给后续，v1 先跳过。
    """
    rotated = [
        {"op": "rotate_base", "params": {"dyaw_rad": 1.5708}},
        {"op": "navigate_relative", "params": {"dx_m": 1.0}},
        {"op": "dig", "params": {"dump_y_m": 0.0}},
    ]
    assert planner._check_destination_side(rotated, "left") is None


# --------------------------------------------------------------------------- loop
def _dig_index():
    return {"dig": lambda **kwargs: None}


def _patch_chat(monkeypatch, replies):
    calls: list[str] = []

    def fake_chat(system, user, **kwargs):
        calls.append(user)
        return replies.pop(0)

    monkeypatch.setattr(planner, "_chat", fake_chat)
    return calls


def test_compile_loop_renders_constraint_and_retries_on_violation(monkeypatch):
    bad = [
        {
            "op": "dig",
            "params": {
                "dig_x_m": 5.0,
                "dig_y_m": 2.0,
                "dump_x_m": 5.0,
                "dump_y_m": -1.0,
                "dump_z_m": 3.5,
            },
        }
    ]
    good = [
        {
            "op": "dig",
            "params": {
                "dig_x_m": 5.0,
                "dig_y_m": 2.0,
                "dump_x_m": 4.5,
                "dump_y_m": 2.5,
                "dump_z_m": 3.5,
            },
        }
    ]
    calls = _patch_chat(monkeypatch, [json.dumps(bad), json.dumps(good)])

    out = planner.compose_actions(
        "把土堆挖一斗倒到左边3米处",
        action_index=_dig_index(),
        allowed_ops=_dig_index(),
        api_base="http://x",
        model_name="m",
        destination_side="left",
        destination_distance_m=3.0,
    )

    assert out == good
    assert len(calls) == 2, "a side violation must cost one compile retry, not a dig"
    # First prompt states the constraint; the retry carries the correction.
    assert "【用户方向约束】" in calls[0] and "机身左侧" in calls[0]
    assert "方向不满足用户约束" in calls[1] and "3" in calls[1]


def test_compile_loop_accepts_first_try_when_side_ok(monkeypatch):
    good = [{"op": "dig", "params": {"dump_y_m": 2.5}}]
    calls = _patch_chat(monkeypatch, [json.dumps(good)])

    out = planner.compose_actions(
        "把土堆挖一斗倒到左边3米处",
        action_index=_dig_index(),
        allowed_ops=_dig_index(),
        api_base="http://x",
        model_name="m",
        destination_side="left",
    )

    assert out == good
    assert len(calls) == 1
