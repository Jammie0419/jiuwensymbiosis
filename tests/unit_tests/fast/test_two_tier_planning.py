# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Planning must survive an empty skill library, and a plan must survive the world moving.

Three properties, one loop. First: the skill library is a shortcut, not a
prerequisite — strip every SKILL.md and the planner still has to derive a working
sequence from the action contracts alone, because that is the only way a body can
do something nobody wrote a skill for. Second: having a second tier must not cost
anything when the first one works — a task the library covers is **one** LLM call,
so the fallback is paid for only when it is used. Third: the sequence was planned
against a belief, so when the measured state contradicts the next step the runner
must re-plan from what is actually true rather than drive on or give up.

The LLM (``planner._chat``) is monkeypatched, so what is under test is the routing
between the tiers and the runner's drift handling — never the model.
"""

from __future__ import annotations

import json
from typing import Any

from jiuwensymbiosis.agent.fast import planner
from jiuwensymbiosis.agent.fast.runner import direct_executor, run_sequence
from jiuwensymbiosis.agent.fast.sequence import parse_sequence
from jiuwensymbiosis.api.actions import ActionSpec, implements
from jiuwensymbiosis.api.base import BaseRobotApi
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation
from jiuwensymbiosis.tools.robot_control_tool import _build_action_index

_CAPS = {"vision.detection", "grasp.parallel", "motion.base"}


class _Env(BaseRobotEnv):
    capabilities = frozenset(_CAPS)
    name = "fake"

    def __init__(self) -> None:
        self.holding_payload: bool | None = None

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def home(self) -> None: ...

    def get_observation(self) -> RobotObservation:
        return RobotObservation(pose={"x": 0.0, "y": 0.0, "z": 0.0}, joints=[0.0])


class _Api(BaseRobotApi):
    capability = set(_CAPS)

    @implements(
        ActionSpec(name="locate", description="find an object", produces_location=True, capability="vision.detection")
    )
    def locate(self, object_name: str) -> dict:
        return {"ok": True, "position": [100.0, 0.0, 50.0]}

    @implements(
        ActionSpec(
            name="approach",
            description="drive up to the sensed object",
            consumes_location=True,
            capability="motion.base",
            tags=("motion",),
        )
    )
    def approach(self, target: Any = None) -> dict:
        return {"ok": True}

    @implements(
        ActionSpec(
            name="drive",
            description="drive the base somewhere else",
            invalidates_locations=True,
            capability="motion.base",
            tags=("motion",),
        )
    )
    def drive(self) -> dict:
        return {"ok": True}

    @implements(
        ActionSpec(
            name="grip",
            description="close the gripper",
            requires=("payload.clear",),
            provides=("payload.held",),
            capability="grasp.parallel",
            tags=("grasp",),
        )
    )
    def grip(self) -> dict:
        return {"ok": True}

    @implements(
        ActionSpec(
            name="release",
            description="open the gripper",
            provides=("payload.clear",),
            capability="grasp.parallel",
            tags=("grasp",),
        )
    )
    def release(self) -> dict:
        return {"ok": True}


class _Session:
    def __init__(self) -> None:
        self.env = _Env()
        self.api = _Api(self.env)


def _index() -> dict[str, Any]:
    s = _Session()
    return _build_action_index(s.api, env=s.env)


_PICK_MD = {
    "name": "visual_pick",
    "description": "视觉引导抓取",
    "capabilities": ["grasp.parallel"],
    "requires": ["payload.clear"],
    "provides": ["payload.held"],
    "markdown": "# visual_pick\n\n| 步骤 | 动作 |\n|---|---|\n| 1 | locate |\n| 2 | approach |\n| 3 | grip |",
}
_PICK_SEQ = [
    {"op": "locate", "params": {"object_name": "crate"}, "bind": "crate"},
    {"op": "approach"},
    {"op": "grip"},
]


def _patch_chat(monkeypatch, *, compiled=None, composed=None):
    """Route each tier's call by its system prompt; ``None`` = that tier is offline.

    Every entry in ``calls`` is one LLM round trip, so the list length is what the
    "one call when the library covers it" assertions read.
    """
    calls: list[tuple[str, str]] = []

    def fake_chat(system, user, **kwargs):
        if system is planner._COMPILER_SYSTEM:  # noqa: SLF001 - asserting on tier routing
            tier, reply = "compiler", compiled
        else:
            tier, reply = "composer", composed
        calls.append((tier, user))
        if reply is None:
            raise RuntimeError(f"{tier} offline")
        return json.dumps(reply)

    monkeypatch.setattr(planner, "_chat", fake_chat)
    return calls


def _plan(**overrides):
    index = _index()
    kwargs: dict[str, Any] = {
        "query": "把箱子抓起来",
        "skills_md": [_PICK_MD],
        "action_index": index,
        "allowed_ops": index,  # the index, so the contract checks are live (see parse_sequence)
        "api_base": "http://x",
        "model_name": "m",
        "api_capabilities": sorted(_CAPS),
        "world_tokens": ["payload.clear"],
        "attempts": 1,
    }
    kwargs.update(overrides)
    return planner.plan_task(**kwargs)


# --------------------------------------------------------------------------- #
# Tier 2: no skill library at all
# --------------------------------------------------------------------------- #
def test_an_empty_library_still_yields_an_executable_sequence(monkeypatch):
    calls = _patch_chat(monkeypatch, composed=_PICK_SEQ)
    result = _plan(skills_md=[])
    assert result.tier == "action"
    assert [t for t, _ in calls] == ["composer"]  # no library to consult at all
    assert [s["op"] for s in result.sequence] == ["locate", "approach", "grip"]


def test_tier_2_plans_from_the_contracts_with_no_skill_text(monkeypatch):
    # An empty array from tier 1 is its way of saying "nothing here fits" — the one
    # decidable signal that hands the task down, and it costs no extra round trip.
    calls = _patch_chat(monkeypatch, compiled=[], composed=_PICK_SEQ)
    result = _plan()
    assert result.tier == "action" and "visual_pick" in result.reason
    assert [t for t, _ in calls] == ["compiler", "composer"]
    prompt = calls[-1][1]
    assert "视觉引导抓取" not in prompt  # the library is out of the picture entirely
    assert "locate" in prompt and "payload.clear" in prompt  # contracts are all it gets


def test_a_skill_whose_preconditions_cannot_hold_routes_to_tier_2(monkeypatch):
    # Already holding something, so grip's requires=[payload.clear] cannot hold and
    # the expansion is rejected by the state machine, not by anyone's opinion.
    calls = _patch_chat(
        monkeypatch,
        compiled=_PICK_SEQ,
        composed=[{"op": "release"}, *_PICK_SEQ],  # tier 2 is free to empty the gripper first
    )
    result = _plan(world_tokens=["payload.held"], attempts=2)
    assert result.tier == "action" and "payload.clear" in result.reason
    assert [t for t, _ in calls] == ["compiler", "compiler", "composer"]  # 2 attempts, then down


def test_a_skill_that_cannot_be_expanded_routes_to_tier_2(monkeypatch):
    calls = _patch_chat(
        monkeypatch,
        compiled=[{"op": "teleport"}],  # not an action this body has
        composed=_PICK_SEQ,
    )
    result = _plan()
    assert result.tier == "action" and "visual_pick" in result.reason
    assert [t for t, _ in calls] == ["compiler", "composer"]


# --------------------------------------------------------------------------- #
# Tier 1: the library covers it — and having a tier 2 costs nothing
# --------------------------------------------------------------------------- #
def test_a_covering_library_stops_at_tier_1_in_one_call(monkeypatch):
    calls = _patch_chat(
        monkeypatch,
        compiled=_PICK_SEQ,
        composed=None,  # entering tier 2 would raise, which is the assertion
    )
    result = _plan()
    assert result.tier == "skill" and result.skills == ("visual_pick",)
    assert [t for t, _ in calls] == ["compiler"]  # picking and expanding are one inference
    prompt = calls[0][1]
    assert "视觉引导抓取" in prompt and "locate" in prompt  # library and contracts, together


# --------------------------------------------------------------------------- #
# Runtime: the world drifts away from what the plan assumed
# --------------------------------------------------------------------------- #
def _executor_for(session: _Session, log: list[str]):
    """Dispatch ops, and let ``locate`` be where the world turns out to be wrong."""

    def run(op: str, params: dict) -> dict:
        log.append(op)
        if op == "locate":
            session.env.holding_payload = True  # something is already in the gripper
            return {"ok": True, "result": {"ok": True, "position": [100.0, 0.0, 50.0]}}
        if op == "release":
            session.env.holding_payload = False
            return {"ok": True, "result": {"ok": True}}
        return {"ok": True, "result": {"ok": True}}

    return run


def _drifting_run(replan, *, max_replans: int = 2):
    session = _Session()
    log: list[str] = []
    steps = parse_sequence(
        [{"op": "locate", "params": {"object_name": "crate"}, "bind": "crate"}, {"op": "grip"}],
        allowed_ops=_index(),
    )
    res = run_sequence(
        session,
        steps,
        executor=_executor_for(session, log),
        action_index=_index(),
        replan=replan,
        max_replans=max_replans,
    )
    return res, log


def test_drift_replans_instead_of_running_the_stale_step():
    seen: list[str] = []

    def replan(world, why):
        seen.append(why)
        assert "payload.held" in world.tokens
        return parse_sequence([{"op": "release"}, {"op": "grip"}], allowed_ops=_index())

    res, log = _drifting_run(replan)
    assert res["ok"] is True
    assert log == ["locate", "release", "grip"]  # the gripper was emptied before grasping
    assert len(seen) == 1 and "payload.clear" in seen[0]
    replanned = [s for s in res["steps"] if s["op"] == "<replan>"]
    assert len(replanned) == 1 and "grip" in replanned[0]["result"]["reason"]


def test_without_a_replanner_the_step_still_dispatches():
    # The drift check reads a best-effort belief, not a safety interlock: with no
    # better plan available it must not take away a run that would have happened.
    res, log = _drifting_run(None)
    assert res["ok"] is True and log == ["locate", "grip"]
    assert all(s["op"] != "<replan>" for s in res["steps"])


def test_replanning_is_capped():
    attempts: list[str] = []

    def replan(world, why):
        attempts.append(why)
        # a DIFFERENT step each call — an identical re-emission would be
        # rejected as a recurrence before ever touching the cap.
        if len(attempts) > 1:
            return parse_sequence([{"op": "grip"}], allowed_ops=_index())
        return parse_sequence(
            [
                {"op": "locate", "params": {"object_name": "crate"}, "bind": "again"},
                {"op": "grip"},
            ],
            allowed_ops=_index(),
        )

    res, log = _drifting_run(replan, max_replans=2)
    assert len(attempts) == 2  # then it stops asking and runs what it has
    assert res["ok"] is True and "grip" in log


def test_a_failing_replanner_leaves_the_original_plan_alone():
    def replan(world, why):
        raise RuntimeError("planner offline")

    res, log = _drifting_run(replan)
    assert res["ok"] is True and log == ["locate", "grip"]


def test_a_business_failure_replans_with_the_exact_reason():
    """A tool reporting ok=False is a planning error; replan instead of aborting.

    Measured failure (215F): the brain planned a dump point 8.06 m from the
    base while the reach annulus ends at 6 m. ``dig`` correctly refused it, but
    the runner ended the whole task. The exact reason must reach the re-planner
    (which can choose a reachable point or move the base) and the task continue.
    """
    session = _Session()
    log: list[str] = []
    attempts: list[str] = []
    grip_calls = {"n": 0}

    def executor(op: str, params: dict) -> dict:
        log.append(op)
        if op == "grip":
            grip_calls["n"] += 1
            if grip_calls["n"] == 1:
                return {"ok": False, "reason": "dump point outside reach"}
        if op == "locate":
            return {"ok": True, "result": {"ok": True, "position": [100.0, 0.0, 50.0]}}
        return {"ok": True, "result": {"ok": True}}

    def replan(world, why):
        attempts.append(why)
        return parse_sequence([{"op": "release"}, {"op": "grip"}], allowed_ops=_index())

    steps = parse_sequence(
        [
            {"op": "locate", "params": {"object_name": "crate"}, "bind": "crate"},
            {"op": "grip"},
        ],
        allowed_ops=_index(),
    )
    res = run_sequence(
        session,
        steps,
        executor=executor,
        action_index=_index(),
        replan=replan,
        max_replans=1,
    )
    assert res["ok"] is True
    assert len(attempts) == 1 and "dump point outside reach" in attempts[0]
    assert log == ["locate", "grip", "release", "grip"]
    assert [s["op"] for s in res["steps"] if s["op"] != "<replan>" and s["ok"]] == ["locate", "release", "grip"]


# --------------------------------------------------------------------------- #
# Runtime: the sensing the next step acts on was invalidated after the plan was written
# --------------------------------------------------------------------------- #
def _sensing_run(replan, *, moved_out_of_band: bool, acting_step: dict | None = None):
    """Sense, then act on that sensing — optionally with the body moving in between.

    The move happens INSIDE the locate call rather than as a step, because a move the
    plan contains is caught at plan time. What only the runner can see is one it does
    not contain: a body that drives itself into range, a recovery retreat, a re-planned
    remainder. The executor dispatches through ``direct_executor`` so the real
    ``ExecutionMemory`` bookkeeping runs. It moves ONCE, so a re-plan can converge.
    """
    session = _Session()
    log: list[str] = []
    idx = _index()
    direct = direct_executor(session.api)
    moves_left = [1] if moved_out_of_band else [0]

    def run(op: str, params: dict) -> dict:
        log.append(op)
        result = direct(op, params)
        if op == "locate" and moves_left[0]:
            moves_left[0] -= 1
            direct("drive", {})
        return result

    steps = parse_sequence(
        [
            {"op": "locate", "params": {"object_name": "crate"}, "bind": "crate"},
            acting_step or {"op": "approach"},
        ],
        allowed_ops=idx,
    )
    res = run_sequence(session, steps, executor=run, action_index=idx, replan=replan)
    return res, log


def test_an_out_of_band_move_replans_before_acting_on_the_stale_sensing():
    # Plan time signed this sequence off — and was right to: nothing in it moves the
    # body. Only the runner can see that something did.
    seen: list[str] = []

    def replan(world, why):
        seen.append(why)
        return parse_sequence(
            [{"op": "locate", "params": {"object_name": "crate"}}, {"op": "approach"}],
            allowed_ops=_index(),
        )

    res, log = _sensing_run(replan, moved_out_of_band=True)
    assert res["ok"] is True
    assert log == ["locate", "locate", "approach"]  # re-sensed from where it now stands
    assert len(seen) == 1 and "approach" in seen[0]


def test_sensing_that_still_stands_is_left_alone():
    def replan(world, why):
        raise AssertionError(f"re-planned with a valid sensing in hand: {why}")

    res, log = _sensing_run(replan, moved_out_of_band=False)
    assert res["ok"] is True and log == ["locate", "approach"]


def test_a_step_naming_its_own_source_is_not_checked():
    # Its value is already resolved in the runner's bindings, and an op that binds a
    # result without a location-producing contract would make an empty memory look
    # like staleness we cannot prove. Deliberately out of scope for the runtime check.
    def replan(world, why):
        raise AssertionError(f"re-planned a step that names its source: {why}")

    res, log = _sensing_run(
        replan,
        moved_out_of_band=True,
        acting_step={"op": "approach", "params": {"target": "crate.position"}},
    )
    assert res["ok"] is True and log == ["locate", "approach"]
