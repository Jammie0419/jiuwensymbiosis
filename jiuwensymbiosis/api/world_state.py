# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""``WorldState`` — what is true right now, in the planner's own vocabulary.

A planner can only derive an order if it knows where it is starting from. This
assembles that starting point from three sources, in order of authority:

1. **Observation** — what the env can actually measure (``holding_payload``,
   pose, joints). Authoritative: it overrides belief.
2. **Belief** — tokens established by the actions run so far
   (``ExecutionMemory.self_state``). Covers what no sensor reports, such as
   whether a carried payload has been raised to travel height.
3. **Proprioception + scene** — pose / joints / extra from the env observation,
   and sensed locations from ``ExecutionMemory``. Not tokens; context the planner
   reasons over directly.

Everything is best-effort: a body that cannot report something simply omits it,
and an omitted token means *unknown*, never *false*. That distinction matters —
``parse_sequence`` disables self-state checking when the state is unknown rather
than rejecting plans against a guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from jiuwensymbiosis.api.memory import ExecutionMemory
from jiuwensymbiosis.api.state import ruled_out_by

logger = logging.getLogger(__name__)


def _joint_names(env: Any) -> tuple[str, ...] | None:
    """The body's joint names (chain order), or None when it has not stated them."""
    try:
        names = getattr(env, "joint_names", None)
    except Exception as exc:  # noqa: BLE001 - a broken property must not sink the snapshot
        logger.debug("[world_state] joint_names unreadable: %s", exc)
        return None
    return tuple(str(n) for n in names) if names else None


def _observed_payload(env: Any) -> frozenset[str]:
    """``payload.held`` / ``payload.clear`` from the env, or empty when unreported."""
    try:
        value = getattr(env, "holding_payload", None)
    except Exception as exc:
        logger.debug("[world_state] holding_payload unreadable: %s", exc)
        return frozenset()
    if not isinstance(value, bool):
        return frozenset()
    return frozenset({"payload.held"} if value else {"payload.clear"})


def current_tokens(session: Any) -> frozenset[str]:
    """The self-state tokens true right now — belief corrected by observation.

    Split out of :meth:`WorldState.snapshot` because the runner re-reads this
    before every step: it touches nothing but ``env.holding_payload``, whereas a
    full snapshot grabs an observation. Never raises.
    """
    api = getattr(session, "api", None)
    memory = getattr(api, "memory", None)
    believed = memory.self_state if isinstance(memory, ExecutionMemory) else frozenset()
    observed = _observed_payload(getattr(session, "env", None))
    # Observation wins, but only over what it actually contradicts: ``holding_payload`` is a
    # bool, so it can rule out "empty" or "holding" and says nothing about ``payload.stowed``.
    ruled_out = ruled_out_by(observed)
    return frozenset({t for t in believed if t not in ruled_out} | observed)


def _with_reachability(locations: list[dict[str, Any]], api: Any) -> list[dict[str, Any]]:
    """Annotate each sensed location with whether the body can reach it FROM WHERE IT IS NOW.

    This is the channel that stays fresh — it is re-read on every re-plan, unlike the pre-run
    scene — so without this the planner's up-to-date positions carried no reach information at
    all and its reach-annotated positions were stale. A verdict of ``None`` means the judge
    could not decide, and the key is then OMITTED: unknown is not "out of reach".
    """
    if "planning.reachability" not in (getattr(api, "capabilities", None) or frozenset()):
        return locations
    for loc in locations:
        position = loc.get("position_mm")
        if not position:
            continue
        try:
            verdict = api.check_reachable({"position": position})
        except Exception as exc:
            logger.debug("[world_state] check_reachable(%s) failed: %s", loc.get("referent"), exc)
            continue
        if verdict is not None:
            loc["reachable"] = bool(verdict)
    return locations


def _safe_attr(obj: Any, name: str) -> Any:
    """Read an optional env property without letting a body's getter sink the snapshot."""
    try:
        return getattr(obj, name, None)
    except Exception as exc:
        logger.debug("[world_state] %s unreadable: %s", name, exc)
        return None


def _reach_prior(api: Any) -> dict[str, Any] | None:
    """The body's URDF reach model, when it has one. ``None`` = it cannot say."""
    if "planning.reachability" not in (getattr(api, "capabilities", None) or frozenset()):
        return None
    try:
        prior = api.describe_reach()
    except Exception as exc:
        logger.debug("[world_state] describe_reach failed: %s", exc)
        return None
    return prior if isinstance(prior, dict) else None


# What each goto_xyzr orientation policy means, spelled out in the prompt: the policy NAME
# alone does not tell a planner that "preserve" will not point the tool down.
_ORIENTATION_HINTS = {
    "preserve": "沿用当前倾角，不会自动朝下；要朝下必须显式传 top_down",
    "top_down": "工具朝下",
    "grasp": "本体标定的抓取倾角",
}


@dataclass(frozen=True)
class WorldState:
    """A snapshot of the robot and its surroundings, for planning."""

    tokens: frozenset[str] = frozenset()
    locations: list[dict[str, Any]] = field(default_factory=list)
    pose: dict[str, Any] | None = None
    joints: list[float] | None = None
    # "deg" / "rad" / None. Carried beside the numbers because the numbers alone are
    # ambiguous: 1.5 is a nudge in degrees and 86 degrees in radians.
    joint_units: str | None = None
    # Which orientation_policy goto_xyzr applies when the caller omits it. The tool schema
    # can only say "there is a default" — implements() runs before any config exists.
    default_orientation_policy: str | None = None
    # The body's reach envelope, so a plan can be built inside it instead of being bounced
    # off SafetyRail one step at a time. workspace_bounds/z_min_safe are the CONFIG-stated
    # Cartesian box (a fixed arm has one; a mobile body does not); reach_prior is the
    # URDF-derived reach model, for bodies that ship a URDF. Either may be absent.
    workspace_bounds: tuple[float, float, float, float] | None = None
    z_min_safe: float | None = None
    # Per-command base caps (max_step_m, max_turn_rad) — a mobile body's SafetyRail
    # envelope. Stated at plan time for the same reason as workspace_bounds: "转一圈"
    # must be split into several rotate_base calls instead of being rejected mid-run.
    base_step_limits: tuple[float, float] | None = None
    # Joint names in the same order as ``joints`` above. The MOVE_JOINT contract tells
    # the planner to read them here ("never invent one") — without them a plan that
    # wants "zero the arm" emits move_joint({joint_1: 0, ...}) and dies on step one.
    joint_names: tuple[str, ...] | None = None
    reach_prior: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()

    @classmethod
    def snapshot(cls, session: Any) -> WorldState:
        """Read the current state off a live session. Never raises."""
        env = getattr(session, "env", None)
        api = getattr(session, "api", None)
        memory = getattr(api, "memory", None)
        memory = memory if isinstance(memory, ExecutionMemory) else ExecutionMemory()
        tokens = current_tokens(session)

        pose = joints = None
        extra: dict[str, Any] = {}
        try:
            obs = env.get_observation() if env is not None else None
        except Exception as exc:
            logger.debug("[world_state] get_observation failed: %s", exc)
            obs = None
        if obs is not None:
            pose = obs.pose
            joints = obs.joints
            # rgb / depth are large arrays and never part of a prompt; keep the rest.
            extra = {k: v for k, v in (obs.extra or {}).items() if not hasattr(v, "shape")}
        # The work-tool reach annulus lives on the machine config (excavator dig);
        # state it so the planner picks reachable dig/dump points instead of
        # discovering the violation when the action runs.
        cfg = getattr(env, "cfg", None)
        for key in (
            "reach_min_m",
            "reach_max_m",
            "reach_z_min_m",
            "reach_z_max_m",
            "dig_min_drag_radius_m",
            "cut_length_m",
        ):
            value = getattr(cfg, key, None)
            if isinstance(value, (int, float)):
                extra[key] = float(value)
        dig_mode = getattr(cfg, "dig_mode", None)
        if isinstance(dig_mode, str) and dig_mode:
            extra["dig_mode"] = dig_mode

        caps = tuple(sorted(getattr(env, "capabilities", None) or ()))
        return cls(
            tokens=tokens,
            locations=_with_reachability(memory.describe()["locations"], api),
            pose=pose,
            joints=joints,
            joint_units=getattr(env, "joint_units", None),
            default_orientation_policy=getattr(env, "default_orientation_policy", None),
            workspace_bounds=_safe_attr(env, "workspace_bounds"),
            z_min_safe=_safe_attr(env, "z_min_safe"),
            base_step_limits=_safe_attr(env, "base_step_limits"),
            joint_names=_joint_names(env),
            reach_prior=_reach_prior(api),
            extra=extra,
            capabilities=caps,
        )

    def describe(self) -> dict[str, Any]:
        """JSON-able form for the planner prompt and the ``state`` CLI."""
        return {
            "tokens": sorted(self.tokens),
            "locations": self.locations,
            "pose": self.pose,
            "joints": self.joints,
            "joint_units": self.joint_units,
            "default_orientation_policy": self.default_orientation_policy,
            "workspace_bounds": list(self.workspace_bounds) if self.workspace_bounds else None,
            "z_min_safe": self.z_min_safe,
            "base_step_limits": list(self.base_step_limits) if self.base_step_limits else None,
            "joint_names": list(self.joint_names) if self.joint_names else None,
            "reach_prior": self.reach_prior,
            "extra": self.extra,
            "capabilities": list(self.capabilities),
        }

    def as_prompt_block(self) -> str:
        """Render the snapshot as the 【当前状态】 block of a planning prompt.

        Empty string when nothing is known, so the caller can concatenate freely
        without emitting a header over an empty body.
        """
        lines: list[str] = []
        if self.tokens:
            lines.append("状态：" + ", ".join(sorted(self.tokens)))
        if self.pose:
            pose = ", ".join(f"{k}={float(v):.0f}" for k, v in self.pose.items() if isinstance(v, (int, float)))
            if pose:
                lines.append(f"位姿：{pose}")
        base_pose = self.extra.get("base_pose")
        if isinstance(base_pose, dict):
            bx, by, byaw = (
                base_pose.get("x_m"),
                base_pose.get("y_m"),
                base_pose.get("yaw_rad"),
            )
            if all(isinstance(v, (int, float)) for v in (bx, by, byaw)):
                lines.append(
                    f"底盘位姿：x={float(bx):.2f}m, y={float(by):.2f}m, "
                    f"yaw={float(byaw):.3f}rad（REP-103，yaw 正=左转；到目标朝向的转向量"
                    "=wrap(目标方位−yaw)，取绝对值 ≤ π 的最短方向，不得连转超过半圈）"
                )
        envelope: list[str] = []
        if self.workspace_bounds:
            xmin, ymin, xmax, ymax = self.workspace_bounds
            envelope.append(f"XY 可达框 x∈[{xmin:.0f},{xmax:.0f}] y∈[{ymin:.0f},{ymax:.0f}]mm")
        if self.z_min_safe is not None:
            envelope.append(f"Z 下限 {self.z_min_safe:.0f}mm")
        if self.base_step_limits:
            max_step, max_turn = self.base_step_limits
            envelope.append(
                f"底盘单命令上限 平移≤{max_step:g}m/转动≤{max_turn:g}rad"
                "（更远或更大的动作要拆成多条 navigate_relative / rotate_base）"
            )
        reach_min = self.extra.get("reach_min_m")
        reach_max = self.extra.get("reach_max_m")
        if isinstance(reach_min, (int, float)) and isinstance(reach_max, (int, float)):
            envelope.append(
                f"挖掘可达半径 {reach_min:g}–{reach_max:g} m"
                "（dig 的挖点/卸点都必须以基座为原点落在这个环带内；"
                "不要简单按同方向加距离）"
            )
        reach_z_min = self.extra.get("reach_z_min_m")
        reach_z_max = self.extra.get("reach_z_max_m")
        if isinstance(reach_z_min, (int, float)) and isinstance(reach_z_max, (int, float)):
            envelope.append(f"挖掘 Z 范围 {reach_z_min:g}–{reach_z_max:g} m")
        min_drag = self.extra.get("dig_min_drag_radius_m")
        cut_length = self.extra.get("cut_length_m")
        if isinstance(min_drag, (int, float)) and min_drag > 0 and isinstance(cut_length, (int, float)):
            envelope.append(
                f"挖点拖切限制 向机身拖 {cut_length:g} m，拖切终点半径须 ≥ {min_drag:g} m"
                f"（即挖点半径 ≥ {min_drag + cut_length:g} m；太近先退底盘再重读地形）"
            )
        if envelope:
            # Stated at plan time so a target outside it is never planned, instead of being
            # rejected by SafetyRail one step into the run.
            lines.append("工作范围：" + "，".join(envelope) + "（超出即被安全护栏拒绝，请在范围内规划）")
        if self.reach_prior:
            lines.append(f"可达模型：{self.reach_prior}")
        dig_mode = self.extra.get("dig_mode")
        if dig_mode:
            lines.append(f"挖掘方式：{dig_mode}")
        terrain = self.extra.get("terrain")
        if isinstance(terrain, list) and terrain:
            piles = []
            for pile in terrain:
                if not isinstance(pile, dict):
                    continue
                parts = [str(pile.get("name", "?"))]
                for key in ("x_m", "y_m", "z_m"):
                    value = pile.get(key)
                    if isinstance(value, (int, float)):
                        parts.append(f"{key[:-2]}={float(value):.2f}")
                piles.append(" ".join(parts))
            if piles:
                lines.append("地形真值：" + "；".join(piles))
        if self.default_orientation_policy:
            hint = _ORIENTATION_HINTS.get(self.default_orientation_policy, "")
            suffix = f"（{hint}）" if hint else ""
            lines.append(f"goto_xyzr 省略 orientation_policy 时用：{self.default_orientation_policy}{suffix}")
        if self.joints:
            # Name the unit, or say it is unknown — never let a bare number imply one.
            unit = self.joint_units or "单位未声明"
            names = self.joint_names
            if names and len(names) == len(self.joints):
                # Names belong here: MOVE_JOINT's contract says to read them from the world
                # state, and a bare vector leaves the planner inventing 'joint_1'.
                body = ", ".join(f"{n}={float(j):.2f}" for n, j in zip(names, self.joints, strict=True))
            else:
                body = ", ".join(f"{float(j):.2f}" for j in self.joints)
            lines.append(f"关节({unit})：{body}")
        for loc in self.locations:
            pos = loc.get("position_mm")
            pos_s = f"[{', '.join(f'{float(c):.0f}' for c in pos)}]mm" if isinstance(pos, (list, tuple)) else "?"
            reach = loc.get("reachable")
            # Only stated when the body could actually decide; absent means unknown.
            reach_s = "" if reach is None else ("，当前位置够得着" if reach else "，当前位置够不着，需先移动过去")
            lines.append(f"已感知 {loc['referent']}：{pos_s}（{loc['sensed_by']}，{loc['age_s']}s 前{reach_s}）")
        if not lines:
            return ""
        # The base-frame convention is static knowledge, but it only matters when
        # there is something to ground: state it beside the numbers it interprets.
        # Without it, "倒到左边" has no answer — nothing says which sign is left.
        lines.insert(0, "坐标系：基座系（REP-103）——+X 机身正前方，+Y 机身左侧，−Y 机身右侧，+Z 向上")
        return "【当前状态】\n" + "\n".join(f"  {line}" for line in lines) + "\n\n"
