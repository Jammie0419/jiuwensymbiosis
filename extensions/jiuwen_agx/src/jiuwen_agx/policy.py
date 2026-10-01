# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Policy seam — the plug point between the action vocabulary and learned
policies ("小脑").

Same registry pattern as ``sim/backend.py`` (BACKENDS + ``register_backend``):
a module-level dict, populated by the ``@register_policy`` decorator at class
definition time. What a policy IS, contractually:

- constructed as ``cls(policy_cfg, *, joint_names, joint_limits, home_joints)``
  where ``policy_cfg`` is the raw ``policy:`` mapping from the machine YAML
  (each policy consumes the keys it knows — ``ckpt`` / ``device`` / …) and the
  keyword context comes from the machine config so every policy sees the same
  body truth. Construction must be CHEAP-side-effect-free enough to retry, and
  heavy imports (torch / lerobot) belong INSIDE ``__init__`` — this module is
  importable on a machine with no learning stack installed.
- ``reset()`` — called once at the start of every ``act_exec`` invocation.
- ``predict(obs) -> list[dict[str, float]]`` — ONE ACTION CHUNK: a list of
  absolute joint targets, each a ``{joint_name: native_unit_value}`` dict.
  Chunk models (ACT) return the whole block from a single forward; non-chunk
  policies return a one-element list. The executor walks the list beat by beat
  through ``move_joints_blocking`` — the same throat every motion uses, so
  driver validation and joint limits apply to policy output for free.

``obs`` keys: ``{"joints": {name: value}, "goal": {act params}, "image": …}``.
The registry never imports torch; the only bundled policy here is FakePolicy,
a dependency-free scripted walk used by tests, smoke runs and demos.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

__all__ = ["POLICIES", "FakePolicy", "Policy", "register_policy"]

POLICIES: dict[str, type] = {}


def register_policy(name: str):
    """Register a policy class under ``name`` (the YAML ``policy.name`` value)."""

    def deco(cls: type) -> type:
        POLICIES[str(name)] = cls
        return cls

    return deco


@runtime_checkable
class Policy(Protocol):
    """The seam every learned policy plugs into (see the module docstring)."""

    def reset(self) -> None:
        """Forget episode state (action queues, ensemblers) before a new task."""

    def predict(self, obs: dict[str, Any]) -> list[dict[str, float]]:
        """Return one action chunk for the given observation."""


@register_policy("fake")
class FakePolicy:
    """Dependency-free scripted policy: walk the joints to ``home_joints``.

    Exists so the whole seam (capability gating, act_exec loop, failure paths,
    driver validation of policy output) is exercisable without torch/lerobot or
    a trained checkpoint — the M1 acceptance vehicle and a bring-up tool.
    """

    def __init__(
        self,
        policy_cfg: dict[str, Any],
        *,
        joint_names: tuple[str, ...] = (),
        joint_limits: dict[str, tuple[float, float]] | None = None,
        home_joints: dict[str, float] | None = None,
    ) -> None:
        self._names = tuple(joint_names)
        limits = joint_limits or {}
        # Target posture: the configured home pose, clipped into the limits
        # (a home key outside the configured travel would be refused per beat).
        self._target = {
            name: float(home_joints.get(name, 0.0)) if home_joints else 0.0
            for name in self._names
        }
        for name, value in self._target.items():
            if name in limits:
                low, high = limits[name]
                self._target[name] = min(max(value, low), high)
        self._beats = max(1, int(policy_cfg.get("beats", 10)))

    def reset(self) -> None:
        """Stateless: nothing to forget between tasks."""

    def predict(self, obs: dict[str, Any]) -> list[dict[str, float]]:
        """Interpolate from the observed joints toward the target posture."""
        current = dict(obs.get("joints") or {})
        chunk: list[dict[str, float]] = []
        for step in range(1, self._beats + 1):
            frac = step / self._beats
            beat: dict[str, float] = {}
            for name in self._names:
                start = float(current.get(name, 0.0))
                goal = self._target.get(name, start)
                value = start + (goal - start) * frac
                if not math.isfinite(value):  # paranoia: never emit NaN/inf
                    value = goal
                beat[name] = value
            chunk.append(beat)
        return chunk
