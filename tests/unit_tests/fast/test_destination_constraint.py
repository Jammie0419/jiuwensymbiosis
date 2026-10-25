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
import math

import pytest

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


# --------------------------------------------------------------------------- base limits
def test_over_limit_rotate_is_rejected_with_split_hint():
    plan = [{"op": "rotate_base", "params": {"dyaw_rad": 1.5708}}]
    err = planner._check_base_limits(plan, (1.0, 0.7))
    assert err is not None
    assert "1.5708" in err and "0.7" in err and "拆" in err


def test_over_limit_navigate_is_rejected():
    plan = [{"op": "navigate_relative", "params": {"dx_m": 3.0, "dy_m": 4.0}}]
    err = planner._check_base_limits(plan, (1.0, 0.7))
    assert err is not None
    assert "5.00" in err and "拆" in err


def test_within_limits_and_other_ops_pass():
    plan = [
        {"op": "rotate_base", "params": {"dyaw_rad": -0.7}},
        {"op": "navigate_relative", "params": {"dx_m": 1.0, "dyaw_rad": 0.3}},
        {"op": "dig", "params": {"dump_y_m": 2.0}},
    ]
    assert planner._check_base_limits(plan, (1.0, 0.7)) is None


def test_no_limits_disables_the_check():
    plan = [{"op": "rotate_base", "params": {"dyaw_rad": 3.0}}]
    assert planner._check_base_limits(plan, None) is None


def test_compile_loop_retries_on_over_limit_move(monkeypatch):
    over = [
        {"op": "rotate_base", "params": {"dyaw_rad": 1.5708}},
        {"op": "home"},
    ]
    fixed = [
        {"op": "rotate_base", "params": {"dyaw_rad": 0.7}},
        {"op": "rotate_base", "params": {"dyaw_rad": 0.7}},
        {"op": "rotate_base", "params": {"dyaw_rad": 0.1708}},
        {"op": "home"},
    ]
    calls = _patch_chat(monkeypatch, [json.dumps(over), json.dumps(fixed)])

    out = planner.compose_actions(
        "把土堆挖一斗倒到左边10米处",
        action_index={"rotate_base": lambda **k: None, "home": lambda **k: None},
        allowed_ops={"rotate_base": lambda **k: None, "home": lambda **k: None},
        api_base="http://x",
        model_name="m",
        base_step_limits=(1.0, 0.7),
    )

    assert out == fixed
    assert len(calls) == 2
    assert "超过底盘单条上限" in calls[1]


def test_compile_max_tokens_allows_long_station_plans(monkeypatch):
    captured = {}

    def fake_chat(system, user, **kwargs):
        captured.update(kwargs)
        return json.dumps([{"op": "home"}])

    monkeypatch.setattr(planner, "_chat", fake_chat)
    planner.compose_actions(
        "把土堆挖一斗倒到左边10米处",
        action_index={"home": lambda **k: None},
        allowed_ops={"home": lambda **k: None},
        api_base="http://x",
        model_name="m",
    )
    # 11 步计划曾按 1500 截断；重规划的 24 步计划在 3000 仍截断 → 6000。
    assert captured["max_tokens"] >= 6000


# --------------------------------------------------------------------------- bearing / universal leg
def test_bearing_parsed_from_compound_and_explicit_angles():
    import math

    def deg(b):
        return round(math.degrees(b), 1)
    assert deg(planner._parse_destination_bearing("3 meters at 45 degrees right front")) == -45.0
    assert deg(planner._parse_destination_bearing("往正右前方45度角移动3米")) == -45.0
    assert deg(planner._parse_destination_bearing("左后方移动2米")) == 135.0
    assert deg(planner._parse_destination_bearing("右边30度")) == -30.0
    # Plain four-way sides stay with the offset map (None here).
    assert planner._parse_destination_bearing("left 3 meters") is None
    assert planner._parse_destination_bearing("behind 3 m") is None


def test_validate_parsed_fills_bearing():
    out = planner._validate_parsed({"destination": "右前方45度移动3米"})
    assert out["destination_side"] == "right"
    assert out["destination_distance_m"] == 3.0
    assert out["destination_bearing_rad"] == pytest.approx(-math.pi / 4)


def test_anchor_uses_bearing_over_side():
    from jiuwensymbiosis.agent.fast.planner import destination_world_anchor

    ax, ay = destination_world_anchor("right", 3.0, {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}, -math.pi / 4)
    assert ax == pytest.approx(2.12, abs=0.01) and ay == pytest.approx(-2.12, abs=0.01)
    # Without a bearing the plain-side offset (0, -3) still applies.
    assert destination_world_anchor("right", 3.0, {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}) == (0.0, -3.0)


def test_move_constraint_renders_the_universal_leg():
    """The 45-degree move task that failed live: the anchor is right, and the
    transcription numbers (2 x -0.3927, 3 x 1.0) are the answer the model
    could not derive on its own."""
    text = planner._format_destination_constraint(
        "right",
        3.0,
        [2.1213, -2.1213],
        {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        (1.0, 0.7),
        None,
        None,
        "move",
    )
    assert "整机移动" in text and "终点世界坐标 (2.12, -2.12)" in text
    assert "dyaw_rad=-0.3927" in text and "3 条 navigate_relative" in text and "dx_m=1.00" in text
    # The dig-flavoured recipe must NOT leak into a move task.
    assert "dig_soil" not in text and "卸点" not in text and "dump_soil" not in text


def test_move_target_check_validates_the_end_pose():
    base = {"x_m": -3.0, "y_m": 0.0, "yaw_rad": 0.0}
    anchor = [-0.8787, -2.1213]
    good = [{"op": "rotate_base", "params": {"dyaw_rad": -0.3927}}] * 2 + [
        {"op": "navigate_relative", "params": {"dx_m": 1.0}}
    ] * 3
    assert planner._check_destination_anchor(good, anchor, base, 3.0, move_target=True) is None
    wrong_sign = [{"op": "rotate_base", "params": {"dyaw_rad": 0.3927}}] * 2 + [
        {"op": "navigate_relative", "params": {"dx_m": 1.0}}
    ] * 3
    err = planner._check_destination_anchor(wrong_sign, anchor, base, 3.0, move_target=True)
    assert err is not None and "终点" in err and "4.24" in err


def test_compile_failure_carries_attempt_history(monkeypatch):
    import pytest as _pytest

    def fake_chat(_s, _u, **_k):
        return "not json at all"

    monkeypatch.setattr(planner, "_chat", fake_chat)
    with _pytest.raises(RuntimeError, match="历次"):
        planner.compose_actions(
            "task",
            action_index={"home": lambda **k: None},
            allowed_ops={"home": lambda **k: None},
            api_base="http://x",
            model_name="m",
            attempts=2,
        )


# --------------------------------------------------------------------------- world anchor
def test_destination_world_anchor_pinned_to_task_start_pose():
    from jiuwensymbiosis.agent.fast.planner import destination_world_anchor

    # "左 10 米" against the start pose stays ONE fixed world point.
    assert destination_world_anchor("left", 10.0, {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}) == (0.0, 10.0)
    # Same words, machine facing +Y: left is now −X, anchor = (2−10, 1).
    ax, ay = destination_world_anchor("left", 10.0, {"x_m": 2.0, "y_m": 1.0, "yaw_rad": 1.5708})
    assert ax == pytest.approx(-8.0, abs=1e-3) and ay == pytest.approx(1.0, abs=1e-3)
    # Unknown side / distance / pose → None (caller falls back to y-sign check).
    assert destination_world_anchor(None, 10.0, {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}) is None
    assert destination_world_anchor("left", None, {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}) is None
    assert destination_world_anchor("left", 10.0, None) is None


def test_constraint_block_renders_the_world_anchor():
    text = planner._format_destination_constraint("left", 10.0, [0.0, 10.0])
    assert "世界坐标目标点 (0.00, 10.00)" in text
    assert "固定不变" in text


def test_recipe_computes_the_full_station_move():
    """The worked example the model transcribes instead of doing geometry:
    左 10 米 from the origin → station (0,5), turn π/2 split 3×0.5236, 5×1 m.
    No dig point known → the dump leg is still fully computed."""
    text = planner._destination_recipe([0.0, 10.0], {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}, (1.0, 0.7))
    assert "先挖后运" in text
    assert "S=(0.00, 5.00)" in text
    assert "θ=+1.5708" in text
    assert "3 条 rotate_base" in text and "dyaw_rad=+0.5236" in text
    assert "5 条 navigate_relative" in text and "dx_m=1.00" in text
    assert "dump_x_m=5, dump_y_m=0, dump_z_m=3.5" in text


def test_recipe_says_no_move_when_anchor_already_in_envelope():
    text = planner._destination_recipe([3.0, 0.0], {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}, (1.0, 0.7))
    # Too-close dump points are pushed OUT along the bearing into the 4.4 m band.
    assert "不动" in text and "(4.40, 0.00, 3.5)" in text


# --------------------------------------------------------------- three cases
def _reach():
    return (1.0, 6.0, 3.0, 1.0)


def test_case1_both_reachable_zero_movement_compound_dig():
    """The live 3 m run: pile (5,2) r=5.39 ✓, anchor (0,3) r=3 ✓ ⇒ NO movement —
    the planner had driven 2.5 m first and broke the drag constraint instead."""
    text = planner._destination_recipe(
        [0.0, 3.0],
        {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        (1.0, 0.7),
        dig_point=(5.0, 2.0, 1.95),
        dig_reach=_reach(),
    )
    assert "零移动" in text and "不要任何 rotate_base / navigate_relative" in text
    assert "dig_x_m=5, dig_y_m=2, dig_z_m=1.95" in text
    # dump pushed out to the 4.4 m band along the same (0,1) bearing.
    assert "dump_x_m=0.00, dump_y_m=4.40, dump_z_m=3.5" in text


def test_case2_dig_here_then_haul():
    text = planner._destination_recipe(
        [0.0, 10.0],
        {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        (1.0, 0.7),
        dig_point=(5.0, 2.0, 1.95),
        dig_reach=_reach(),
    )
    assert "先挖后运" in text
    assert "dig_soil(dig_x_m=5, dig_y_m=2, dig_z_m=1.95" in text
    assert "不要先移动" in text
    assert "S=(0.00, 5.00)" in text and "dump_x_m=5, dump_y_m=0" in text


def test_case3_dig_too_close_face_pile_and_retreat():
    """太近（r=2.69 < 4）：正对土堆转向 + 带符号后退，挖点保持正前方 (5.2, 0)。"""
    text = planner._destination_recipe(
        [0.0, 3.0],
        {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        (1.0, 0.7),
        dig_point=(2.5, 1.0, 1.95),
        dig_reach=_reach(),
    )
    assert "太近" in text and "先就位再挖" in text
    assert "正对土堆转向 θ=+0.3805" in text  # atan2(1, 2.5)
    assert "后退 2.51 m" in text and "dx_m=-0.84" in text  # 2.69−5.2, split 3
    assert "dig_x_m=5.20, dig_y_m=0" in text
    # The anchor (0,3) is within 5.5 m of the retreat station → dump in place.
    assert "不动" in text


def test_case3_dig_too_far_approaches_then_hauls():
    text = planner._destination_recipe(
        [0.0, 10.0],
        {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        (1.0, 0.7),
        dig_point=(0.0, 11.0, 1.95),  # 11 m out: unreachable
        dig_reach=_reach(),
    )
    assert "超出可达环带" in text and "前进" in text
    assert "dig_x_m=5.20, dig_y_m=0" in text
    # From the dig station the anchor (0,10) is ~1 m away → dump in place after dig.
    assert "dig_soil" in text


def test_constraint_block_carries_the_recipe_upfront():
    """First attempt, before any violation: the recipe rides along so the very
    first plan can be right (the model proved it cannot derive the geometry)."""
    text = planner._format_destination_constraint(
        "left", 10.0, [0.0, 10.0], {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}, (1.0, 0.7)
    )
    assert "照抄解法" in text and "S=(0.00, 5.00)" in text


def test_correction_carries_the_recipe_for_retries():
    text = planner._destination_correction(
        "left", 10.0, [0.0, 10.0], {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}, (1.0, 0.7)
    )
    assert "照抄解法" in text and "3 条 rotate_base" in text


def test_retries_take_temperature_to_escape_repetition(monkeypatch):
    """temperature=0 retries differ ONLY by the appended correction — a
    systematically wrong derivation repeats itself until attempts run out
    (observed live: 4 identical anchor violations). The first attempt stays
    exact; retries carry heat."""
    temps: list[float] = []
    good = [{"op": "home"}]

    def fake_chat(_system, _user, **kwargs):
        temps.append(float(kwargs.get("temperature", 0.0)))
        return json.dumps(good if len(temps) > 1 else [{"op": "dig", "params": {"dump_y_m": -1.0}}])

    monkeypatch.setattr(planner, "_chat", fake_chat)
    planner.compose_actions(
        "把土堆挖一斗倒到左边3米处",
        action_index={"dig": lambda **k: None, "home": lambda **k: None},
        allowed_ops={"dig": lambda **k: None, "home": lambda **k: None},
        api_base="http://x",
        model_name="m",
        destination_side="left",
    )

    assert temps[0] == 0.0
    assert all(t >= 0.2 for t in temps[1:])


def test_anchor_check_replays_motion_and_accepts_a_station_plan():
    """左 10 米 → 锚点 (0,10)。正确计划：转到方位、前进 5 m、正前方 5 m 卸料。"""
    base = {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}
    plan = [
        *[{"op": "rotate_base", "params": {"dyaw_rad": 0.7}} for _ in range(2)],
        {"op": "rotate_base", "params": {"dyaw_rad": 0.1708}},
        *[{"op": "navigate_relative", "params": {"dx_m": 1.0}} for _ in range(5)],
        {"op": "dump_soil", "params": {"dump_x_m": 5.0, "dump_y_m": 0.0}},
        {"op": "home"},
    ]
    assert planner._check_destination_anchor(plan, [0.0, 10.0], base, 10.0) is None


def test_anchor_check_flags_the_observed_drift_plan():
    """上一条真机轨迹的实际形态：转 1.4 rad 后前进 10.5 m、再往侧向 5 m 卸——
    落点距锚点 ~7.5 m，必须打回并给出基座系应瞄的坐标。"""
    base = {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}
    plan = [
        *[{"op": "rotate_base", "params": {"dyaw_rad": 0.7}} for _ in range(2)],
        *[{"op": "navigate_relative", "params": {"dx_m": 1.0}} for _ in range(10)],
        {"op": "navigate_relative", "params": {"dx_m": 0.5}},
        {"op": "dump_soil", "params": {"dump_x_m": 0.0, "dump_y_m": 5.0}},
    ]
    err = planner._check_destination_anchor(plan, [0.0, 10.0], base, 10.0)
    assert err is not None
    assert "偏了" in err and "基座系约为" in err


def test_anchor_check_skips_bindings_and_missing_pose():
    bound = [{"op": "dump_soil", "params": {"dump_x_m": "bind.x", "dump_y_m": 0.0}}]
    assert planner._check_destination_anchor(bound, [0.0, 10.0], {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}, 10.0) is None
    literal = [{"op": "dump_soil", "params": {"dump_x_m": 5.0, "dump_y_m": 0.0}}]
    assert planner._check_destination_anchor(literal, [0.0, 10.0], None, 10.0) is None


def test_compile_loop_uses_the_anchor_check_before_the_sign_check(monkeypatch):
    """带锚点时，转向计划也要过方向校验（y 符号校验对它本来是盲的）。"""
    drifted = [
        {"op": "rotate_base", "params": {"dyaw_rad": 0.7}},
        {"op": "rotate_base", "params": {"dyaw_rad": 0.7}},
        {"op": "dump_soil", "params": {"dump_x_m": 0.0, "dump_y_m": 5.0}},
    ]
    fixed = [
        *[{"op": "rotate_base", "params": {"dyaw_rad": 0.7}} for _ in range(2)],
        {"op": "rotate_base", "params": {"dyaw_rad": 0.1708}},
        *[{"op": "navigate_relative", "params": {"dx_m": 1.0}} for _ in range(5)],
        {"op": "dump_soil", "params": {"dump_x_m": 5.0, "dump_y_m": 0.0}},
        {"op": "home"},
    ]
    calls = _patch_chat(monkeypatch, [json.dumps(drifted), json.dumps(fixed)])

    out = planner.compose_actions(
        "把土堆挖一斗倒到左边10米处",
        action_index={
            "rotate_base": lambda **k: None,
            "navigate_relative": lambda **k: None,
            "dump_soil": lambda **k: None,
            "home": lambda **k: None,
        },
        allowed_ops={
            "rotate_base": lambda **k: None,
            "navigate_relative": lambda **k: None,
            "dump_soil": lambda **k: None,
            "home": lambda **k: None,
        },
        api_base="http://x",
        model_name="m",
        base_step_limits=(1.0, 0.7),
        destination_world=[0.0, 10.0],
        base_pose={"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0},
        destination_distance_m=10.0,
    )

    assert out == fixed
    assert len(calls) == 2
    # The retry prompt carries the fixed world point and how to aim at it.
    assert "世界坐标点固定为 (0.00, 10.00)" in calls[1]


# --------------------------------------------------------------------------- point hard checks
def test_check_points_rejects_the_historical_radius_ten_dump():
    """第一次真机失败的形态：把"10 米"当基座系卸点半径（10.005 m）。"""
    plan = [
        {"op": "dig", "params": {"dig_x_m": 5.0, "dig_y_m": 2.0, "dump_x_m": 5.0, "dump_y_m": 8.66, "dump_z_m": 3.5}}
    ]
    err = planner._check_points(plan, (1.0, 6.0, 3.0, 1.0))
    assert err is not None and "卸点半径 10.00" in err and "闭斗卸料带" in err


def test_check_points_rejects_a_too_close_dig_point():
    """3e9c1a1e 的失败形态：移动后挖点半径 2.30 m，低于拖切下限 4 m。"""
    plan = [{"op": "dig_soil", "params": {"dig_x_m": 2.3, "dig_y_m": 0.0, "dig_z_m": 1.9}}]
    err = planner._check_points(plan, (1.0, 6.0, 3.0, 1.0))
    assert err is not None and "太近" in err and "拖切" in err


def test_check_points_rejects_dump_height_out_of_band():
    plan = [{"op": "dump_soil", "params": {"dump_x_m": 4.5, "dump_y_m": 0.0, "dump_z_m": 1.0}}]
    err = planner._check_points(plan, (1.0, 6.0, 3.0, 1.0))
    assert err is not None and "卸料高度带" in err


def test_check_points_accepts_the_good_compound_dig():
    plan = [
        {
            "op": "dig",
            "params": {
                "dig_x_m": 5.0,
                "dig_y_m": 2.0,
                "dig_z_m": 1.95,
                "dump_x_m": 0.0,
                "dump_y_m": 4.4,
                "dump_z_m": 3.5,
            },
        },
        {"op": "home"},
    ]
    assert planner._check_points(plan, (1.0, 6.0, 3.0, 1.0)) is None


def test_check_points_skips_bindings_and_missing_reach():
    bound = [{"op": "dig_soil", "params": {"dig_x_m": "terrain.piles[0].x_m", "dig_y_m": 0.0}}]
    assert planner._check_points(bound, (1.0, 6.0, 3.0, 1.0)) is None
    literal = [{"op": "dig_soil", "params": {"dig_x_m": 2.0, "dig_y_m": 0.0}}]
    assert planner._check_points(literal, None) is None


def test_compile_loop_retries_on_a_bad_point(monkeypatch):
    """点位错只花一次编译重试，不花一次物理挖掘。"""
    bad = [
        {"op": "dig", "params": {"dig_x_m": 5.0, "dig_y_m": 2.0, "dump_x_m": 5.0, "dump_y_m": 8.66, "dump_z_m": 3.5}}
    ]
    good = [
        {"op": "dig", "params": {"dig_x_m": 5.0, "dig_y_m": 2.0, "dump_x_m": 0.0, "dump_y_m": 4.4, "dump_z_m": 3.5}}
    ]
    calls = _patch_chat(monkeypatch, [json.dumps(bad), json.dumps(good)])

    out = planner.compose_actions(
        "把土堆挖一斗",
        action_index={"dig": lambda **k: None},
        allowed_ops={"dig": lambda **k: None},
        api_base="http://x",
        model_name="m",
        dig_reach=(1.0, 6.0, 3.0, 1.0),
    )

    assert out == good
    assert len(calls) == 2 and "卸点半径" in calls[1]


# --------------------------------------------------------------------------- rotation direction
def test_spin_plan_is_rejected_with_shortest_direction_hint():
    """The observed failure: a re-plan without odometry emitted 15×0.7+0.17 ≈ 611°."""
    plan = [
        *[{"op": "rotate_base", "params": {"dyaw_rad": 0.7}} for _ in range(15)],
        {"op": "rotate_base", "params": {"dyaw_rad": 0.17}},
        {"op": "home"},
    ]
    err = planner._check_rotation_direction(plan)
    assert err is not None
    assert "半圈" in err and "π" in err
    # Fires the moment the run crosses π: 0.7×5 = 3.5 rad ≡ −2.78 rad the
    # other way — the correction must say so.
    assert "3.5" in err and "-2.78" in err


def test_half_circle_split_passes():
    plan = [
        {"op": "rotate_base", "params": {"dyaw_rad": 0.7}},
        {"op": "rotate_base", "params": {"dyaw_rad": 0.7}},
        {"op": "rotate_base", "params": {"dyaw_rad": 0.1708}},
        {"op": "home"},
    ]
    assert planner._check_rotation_direction(plan) is None


def test_exactly_pi_with_split_noise_passes():
    # 0.7×4 + 0.316 ≈ π − 0.0004: inside the 0.05 tolerance.
    plan = [
        *[{"op": "rotate_base", "params": {"dyaw_rad": 0.7}} for _ in range(4)],
        {"op": "rotate_base", "params": {"dyaw_rad": 0.316}},
    ]
    assert planner._check_rotation_direction(plan) is None


def test_navigates_between_rotations_are_separate_runs():
    """Two station legs each turning 2.1 rad the same way is legitimate: the
    navigate in between ends the in-place run. No false positive."""
    plan = [
        *[{"op": "rotate_base", "params": {"dyaw_rad": 0.7}} for _ in range(3)],
        {"op": "navigate_relative", "params": {"dx_m": 1.0}},
        *[{"op": "rotate_base", "params": {"dyaw_rad": 0.7}} for _ in range(3)],
    ]
    assert planner._check_rotation_direction(plan) is None


def test_direction_reversal_resets_the_run():
    plan = [
        *[{"op": "rotate_base", "params": {"dyaw_rad": 0.7}} for _ in range(3)],
        *[{"op": "rotate_base", "params": {"dyaw_rad": -0.7}} for _ in range(3)],
    ]
    assert planner._check_rotation_direction(plan) is None


def test_single_over_pi_rotate_flagged_even_without_limits():
    # The rotation guard is geometric, not machine-config dependent.
    plan = [{"op": "rotate_base", "params": {"dyaw_rad": 6.0}}]
    err = planner._check_rotation_direction(plan)
    assert err is not None and "半圈" in err


def test_binding_dyaw_is_not_provable_and_resets_the_run():
    plan = [
        {"op": "rotate_base", "params": {"dyaw_rad": "bind.bearing"}},
        *[{"op": "rotate_base", "params": {"dyaw_rad": 0.7}} for _ in range(3)],
    ]
    assert planner._check_rotation_direction(plan) is None


def test_compile_loop_retries_on_spin_plan(monkeypatch):
    spin = [
        *[{"op": "rotate_base", "params": {"dyaw_rad": 0.7}} for _ in range(15)],
        {"op": "home"},
    ]
    shortest = [
        {"op": "rotate_base", "params": {"dyaw_rad": -0.7}},
        {"op": "rotate_base", "params": {"dyaw_rad": -0.7}},
        {"op": "rotate_base", "params": {"dyaw_rad": -0.5}},
        {"op": "home"},
    ]
    calls = _patch_chat(monkeypatch, [json.dumps(spin), json.dumps(shortest)])

    out = planner.compose_actions(
        "把土堆挖一斗倒到左边10米处",
        action_index={"rotate_base": lambda **k: None, "home": lambda **k: None},
        allowed_ops={"rotate_base": lambda **k: None, "home": lambda **k: None},
        api_base="http://x",
        model_name="m",
        base_step_limits=(1.0, 0.7),
    )

    assert out == shortest
    assert len(calls) == 2
    # Each 0.7 rad step is within the per-command cap — only the rotation
    # guard could catch this, and its correction must reach the retry prompt.
    assert "半圈" in calls[1] and "最短方向" in calls[1]


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
