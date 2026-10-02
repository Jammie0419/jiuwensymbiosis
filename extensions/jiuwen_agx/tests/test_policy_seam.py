# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Policy seam tests (M1): registration, capability gating, act_exec semantics.

The seam contract under test (``jiuwen_agx/policy.py`` + ``act_exec``):

- ``policy.act`` is a registered capability; ``act_exec`` enters the shared
  vocabulary through the same four-step chain as ``dig`` (incl. the
  ``tags=("motion",)`` RecoveryRail hook).
- The capability — and therefore the tool — appears ONLY when the machine
  config carries a ``policy:`` section (env-side gating: api ∩ env).
- The act_exec loop: preconditions first (shared with dig), policy chunks as
  absolute joint targets, driver validation as the per-beat guard (a rejected
  target is a readable ActFailure, not an exception), and the scoop truth
  going loaded→unloaded as the success criterion with ``policy_max_beats``
  as the backstop.
- No test here needs torch/lerobot: the ActPolicy registration entry exists
  (its constructor imports lazily), and construction failures surface as
  readable act_exec failures.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from jiuwensymbiosis.api.actions import planner_vocabulary
from jiuwensymbiosis.env.base import KNOWN_CAPABILITIES
from jiuwensymbiosis.tools.builder import list_tool_meta

from jiuwen_agx.actions import ACT_EXECUTE
from jiuwen_agx.agx_excavator import build_agx_excavator_session
from jiuwen_agx.agx_excavator.api import AgxExcavatorApi
from jiuwen_agx.agx_excavator.config import AgxExcavatorConfig
from jiuwen_agx.agx_excavator.env import AgxExcavatorEnv
from jiuwen_agx.policy import POLICIES, FakePolicy

# A legal dig/dump pair inside the default reach annulus [1, 6] m.
GOAL = (-2.0, 1.0, 3.0, 0.5)
JOINTS = ("swing", "boom", "arm", "bucket")


# ============================================================================ stubs
class _StubDriver:
    """Duck-typed driver covering exactly what act_exec + preconditions touch."""

    def __init__(
        self,
        *,
        scoop_script: list[bool] | None = None,
        reject: Any = None,
    ) -> None:
        self.joint_names = JOINTS
        self._joints = dict.fromkeys(self.joint_names, 0.0)
        self._scoop_script = list(scoop_script or [])
        self._reject = reject  # callable(targets) -> error str | None
        self.moves: list[dict[str, float]] = []

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self._joints)

    def move_joints_blocking(
        self, targets: dict[str, float], *, timeout_s: float | None = None
    ) -> dict[str, float]:
        if self._reject is not None and (reason := self._reject(dict(targets))):
            raise ValueError(reason)
        self._joints.update(targets)
        self.moves.append(dict(targets))
        return dict(self._joints)

    def scoop_state(self) -> bool:
        if self._scoop_script:
            return self._scoop_script.pop(0)
        return False

    def mark_scoop(self, loaded: bool) -> None:
        """Unused by act_exec; present for precondition parity with dig."""


class _ChunkPolicy:
    """Scripted policy: fixed-size chunks of tiny in-limit absolute steps."""

    def __init__(
        self,
        policy_cfg: dict[str, Any],
        *,
        joint_names: tuple[str, ...] = (),
        **_: Any,
    ) -> None:
        self._chunk = max(1, int(policy_cfg.get("chunk", 2)))
        self._names = tuple(joint_names)

    def reset(self) -> None:
        pass

    def predict(self, obs: dict[str, Any]) -> list[dict[str, float]]:
        current = obs.get("joints") or {}
        return [
            {name: float(current.get(name, 0.0)) + 0.01 for name in self._names}
            for _ in range(self._chunk)
        ]


class _BadBeatPolicy:
    """Scripted policy whose second beat carries an out-of-limit boom target."""

    def __init__(
        self,
        policy_cfg: dict[str, Any],
        *,
        joint_names: tuple[str, ...] = (),
        **_: Any,
    ) -> None:
        self._names = tuple(joint_names)
        self._bad = float(policy_cfg.get("bad_boom", 2000.0))
        self._emitted = False

    def reset(self) -> None:
        pass

    def predict(self, obs: dict[str, Any]) -> list[dict[str, float]]:
        current = obs.get("joints") or {}
        beat = {name: float(current.get(name, 0.0)) for name in self._names}
        if self._emitted:
            return [beat]
        self._emitted = True
        bad = dict(beat)
        bad["boom"] = self._bad
        return [beat, bad]


def _api(
    cfg: AgxExcavatorConfig | None = None, driver: _StubDriver | None = None
) -> AgxExcavatorApi:
    env = AgxExcavatorEnv(cfg or AgxExcavatorConfig())
    if driver is not None:
        env.low_level = driver
    return AgxExcavatorApi(env)


# ============================================================================ registration
class TestRegistration:
    def test_policy_act_capability_registered(self):
        assert "policy.act" in KNOWN_CAPABILITIES

    def test_act_exec_in_shared_vocabulary_only_for_its_capability(self):
        assert "act_exec" in planner_vocabulary(["policy.act"])
        assert "act_exec" not in planner_vocabulary(["motion.excavator"])

    def test_act_exec_carries_motion_tag_for_recovery_rail(self):
        # RecoveryRail claims tools by tag; without "motion" a failed act_exec
        # would silently skip the payload-aware recovery path dig enjoys.
        assert "motion" in ACT_EXECUTE.tags

    def test_act_exec_contract_mirrors_dig(self):
        dig = planner_vocabulary(["motion.excavator"])["dig"]
        assert ACT_EXECUTE.params == dig.params
        assert ACT_EXECUTE.requires == dig.requires == ("payload.clear",)
        assert ACT_EXECUTE.invalidates_locations is True

    def test_act_policy_registered_without_importing_torch(self):
        # The registry entry must exist on a machine with no learning stack;
        # importing the class (not constructing it) never touches torch.
        assert "act" in POLICIES
        assert "fake" in POLICIES


# ============================================================================ gating
class TestCapabilityGating:
    def test_default_config_hides_act_exec(self):
        env = AgxExcavatorEnv(AgxExcavatorConfig())
        assert "policy.act" not in env.capabilities
        assert "act_exec" not in _toolnames(AgxExcavatorConfig())

    def test_policy_config_exposes_act_exec(self):
        cfg = AgxExcavatorConfig(policy={"name": "fake"})
        env = AgxExcavatorEnv(cfg)
        assert "policy.act" in env.capabilities
        names = _toolnames(cfg)
        assert "act_exec" in names
        assert "dig" in names

    def test_yaml_policy_section_round_trips(self):
        cfg = AgxExcavatorConfig.from_dict(
            {"env": {"cfg": {"low_level": {"policy": {"name": "fake"}}}}}
        )
        assert cfg.policy == {"name": "fake"}
        assert "policy.act" in AgxExcavatorEnv(cfg).capabilities

    def test_yaml_policy_without_name_is_rejected(self):
        with pytest.raises(ValueError, match="name"):
            AgxExcavatorConfig.from_dict(
                {"env": {"cfg": {"low_level": {"policy": {"ckpt": "x"}}}}}
            )

    def test_api_advertises_policy_act_only_when_configured(self):
        # Without a configured policy the api withholds the claim, so session
        # builds do not trip the capability-mismatch alarm (the env side never
        # declares it either; the tool gate outcome is identical).
        assert "policy.act" not in _api(AgxExcavatorConfig()).capabilities
        assert "policy.act" in _api(AgxExcavatorConfig(policy={"name": "fake"})).capabilities

    def test_session_builder_respects_the_gate(self):
        gated = build_agx_excavator_session.from_dict({})
        open = build_agx_excavator_session.from_dict(
            {"env": {"cfg": {"low_level": {"policy": {"name": "fake"}}}}}
        )
        assert "act_exec" not in _names_of(gated)
        assert "act_exec" in _names_of(open)


# ============================================================================ FakePolicy
class TestFakePolicy:
    def test_chunks_walk_toward_home_and_land_on_it(self):
        limits = {
            "swing": (-3.0, 3.0),
            "boom": (-1.0, 1.0),
            "arm": (-1.0, 1.0),
            "bucket": (-1.0, 1.0),
        }
        home = {"swing": 0.5, "boom": -0.5, "arm": 0.5, "bucket": 0.5}
        policy = FakePolicy(
            {"beats": 4}, joint_names=JOINTS, joint_limits=limits, home_joints=home
        )
        obs = {"joints": {"swing": 2.0, "boom": 0.0, "arm": 0.0, "bucket": 0.0}}
        chunk = policy.predict(obs)
        assert len(chunk) == 4
        for beat in chunk:
            assert set(beat) == set(JOINTS)
            assert all(math.isfinite(v) for v in beat.values())
        assert chunk[-1] == pytest.approx(home)  # final beat reaches the target

    def test_target_is_clipped_into_joint_limits(self):
        limits = {
            "swing": (-1.0, 1.0),
            "boom": (-1.0, 1.0),
            "arm": (-1.0, 1.0),
            "bucket": (-1.0, 1.0),
        }
        policy = FakePolicy(
            {"beats": 3},
            joint_names=JOINTS,
            joint_limits=limits,
            home_joints={"swing": 99.0, "boom": -99.0, "arm": 0.0, "bucket": 0.0},
        )
        chunk = policy.predict({"joints": dict.fromkeys(JOINTS, 0.0)})
        for beat in chunk:
            assert -1.0 <= beat["swing"] <= 1.0
            assert -1.0 <= beat["boom"] <= 1.0


# ============================================================================ act_exec
class TestActExec:
    def _cfg(self, policy: dict[str, Any], **kw: Any) -> AgxExcavatorConfig:
        return AgxExcavatorConfig(policy=policy, **kw)

    def test_success_on_scoop_loaded_then_unloaded(self):
        # -- register a scripted policy for this test only
        POLICIES["test_chunk"] = _ChunkPolicy
        try:
            # script slot 0 is consumed by the shared precondition check
            # (bucket must be clear to start); beats 1-5 then walk
            # False → False → True → True → False = one full scooped cycle.
            driver = _StubDriver(scoop_script=[False, False, False, True, True, False])
            api = _api(
                self._cfg({"name": "test_chunk", "chunk": 4}, policy_max_beats=50),
                driver,
            )
            result = api.act_exec(*GOAL)
        finally:
            POLICIES.pop("test_chunk", None)
        assert result["ok"] is True
        assert result["beats"] == 5
        assert result["cycle_s"] >= 0.0
        assert len(driver.moves) == 5

    def test_policy_output_rejected_by_driver_is_a_readable_failure(self):
        POLICIES["test_bad"] = _BadBeatPolicy
        try:
            driver = _StubDriver(
                reject=lambda t: (
                    f"joint boom target {t['boom']} outside limits"
                    if abs(t.get("boom", 0.0)) > 100
                    else None
                )
            )
            api = _api(self._cfg({"name": "test_bad"}), driver)
            result = api.act_exec(*GOAL)
        finally:
            POLICIES.pop("test_bad", None)
        assert result["ok"] is False
        assert "rejected at beat 1" in result["error"]
        assert "outside limits" in result["error"]

    def test_max_beats_backstop_fails_never_loaded_cycle(self):
        POLICIES["test_chunk"] = _ChunkPolicy
        try:
            api = _api(
                self._cfg({"name": "test_chunk", "chunk": 2}, policy_max_beats=6),
                _StubDriver(),
            )
            result = api.act_exec(*GOAL)
        finally:
            POLICIES.pop("test_chunk", None)
        assert result["ok"] is False
        assert "incomplete after 6 beats" in result["error"]
        assert "never loaded" in result["error"]

    def test_refuses_point_outside_annulus_before_any_policy_or_motion(self):
        api = _api(self._cfg({"name": "fake"}), _StubDriver())
        result = api.act_exec(20.0, 0.0, 3.0, 0.5)
        assert result["ok"] is False
        assert "annulus" in result["error"]

    def test_refuses_while_bucket_loaded(self):
        api = _api(self._cfg({"name": "fake"}), _StubDriver(scoop_script=[True]))
        result = api.act_exec(*GOAL)
        assert result["ok"] is False
        assert "loaded" in result["error"]

    def test_unknown_policy_name_is_a_readable_failure(self):
        api = _api(self._cfg({"name": "no_such_policy"}), _StubDriver())
        result = api.act_exec(*GOAL)
        assert result["ok"] is False
        assert "unknown policy" in result["error"]
        assert "fake" in result["error"]  # lists the registered names

    def test_act_policy_construction_without_torch_surfaces_as_failure(self):
        # The registry entry exists, but constructing it without the learning
        # stack must degrade to a readable act_exec failure, never raise past
        # the action boundary.
        api = _api(self._cfg({"name": "act"}), _StubDriver())
        result = api.act_exec(*GOAL)
        assert result["ok"] is False
        assert "policy unavailable" in result["error"]

    def test_fake_policy_runs_the_mock_session_until_the_backstop(self):
        # End-to-end on the real mock backend: FakePolicy beats are legal, the
        # loop runs, and the cycle honestly reports failure (mock scoop truth
        # never turns loaded) — the failure-shape half of the M1 acceptance.
        session = build_agx_excavator_session.from_dict(
            {
                "env": {
                    "cfg": {
                        "low_level": {
                            "policy": {"name": "fake", "beats": 3},
                            "policy_max_beats": 12,
                        }
                    }
                }
            }
        )
        with session:
            result = session.api.act_exec(*GOAL)
        assert result["ok"] is False
        assert "incomplete after 12 beats" in result["error"]


# ============================================================================ helpers
def _toolnames(cfg: AgxExcavatorConfig) -> set[str]:
    api = _api(cfg)
    return {entry["name"] for entry in list_tool_meta(api, env=api.env)}


def _names_of(session: Any) -> set[str]:
    return {entry["name"] for entry in list_tool_meta(session.api, env=session.env)}
