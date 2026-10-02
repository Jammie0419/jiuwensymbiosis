# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Demonstration recording for ACT training — the M2 data pipeline's logic.

Everything here is deliberately FREE of lerobot/torch so the collector runs on
the AGX machine and every rule is unit-testable. Layout (see
docs/act-data-collection-design.md):

- :class:`RecordingDriverProxy` — wraps a real driver and intercepts
  ``move_joints_blocking``: each call (one dig keyframe transition) is split
  into ``sub_beats`` interpolated beats, each sent with
  ``beat_timeout_s`` (the SAME constant deployment uses,
  ``cfg.policy_beat_timeout_s``), the final beat at the caller's own timeout so
  the keyframe is never under-shot. Every beat records (measured before,
  measured after, scoop truth, wall seconds).
- :func:`sample_goal` — seeded, area-uniform dig point over the reachable
  annulus; dump point 2–4 m away, re-sampled until it also fits the annulus.
- :class:`Episode` — one dig-and-dump cycle as arrays + auditable metadata;
  npz round-trip with a format-version gate.
- :func:`admission_reason` — the converter's four gates (version / no-mock /
  goal / success) as one pure function.
- :func:`segment_keyboard` — cut a continuous keyboard-poll stream into
  candidate episodes at scoop unloaded→loaded→unloaded boundaries.

The success criterion everywhere is the scoop truth going loaded→unloaded
within the episode — the same measured mass threshold ``dig`` and ``act_exec``
rely on.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "FORMAT_VERSION",
    "GOAL_KEYS",
    "Episode",
    "RecordingDriverProxy",
    "admission_reason",
    "sample_goal",
    "segment_keyboard",
    "unit_conflict",
]

FORMAT_VERSION = 1
GOAL_KEYS: tuple[str, ...] = ("dig_x_m", "dig_y_m", "dump_x_m", "dump_y_m")
_TASK = "excavate"
_META_KEY = "meta"
_MIN_BEATS = 10


# ============================================================================ proxy
class RecordingDriverProxy:
    """Driver wrapper that sub-steps keyframe transitions into beats.

    Satisfies exactly the surface ``execute_dig_cycle`` touches (joint_names,
    scoop_state, mark_scoop, move_joints_blocking) plus home so a chain can
    reposition between cycles. Motion calls are recorded; reads are forwarded.
    """

    def __init__(
        self, inner: Any, *, sub_beats: int = 20, beat_timeout_s: float = 1.0
    ) -> None:
        self._inner = inner
        self._sub = max(1, int(sub_beats))
        self._beat_timeout_s = float(beat_timeout_s)
        self.joint_names = tuple(inner.joint_names)
        self._last: dict[str, float] = dict(inner.get_joint_positions())
        # One entry per beat: (before, after, scoop, seconds) — plain dicts so
        # Episode can stack them without knowing the driver's vocabulary.
        self.beats: list[tuple[dict[str, float], dict[str, float], bool, float]] = []

    # -- forwarded reads / writes (execute_dig_cycle's surface)
    def get_joint_positions(self) -> dict[str, float]:
        return dict(self._last)

    def scoop_state(self) -> bool:
        return bool(self._inner.scoop_state())

    def mark_scoop(self, loaded: bool) -> None:
        self._inner.mark_scoop(bool(loaded))

    def home(self) -> None:
        self._inner.home()
        self._last = dict(self._inner.get_joint_positions())

    # -- intercepted motion
    def move_joints_blocking(
        self, targets: dict[str, float], *, timeout_s: float | None = None
    ) -> dict[str, float]:
        """Split one keyframe transition into recorded beats.

        The final beat re-sends the exact target with the caller's own timeout
        (usually the config default) so the keyframe is fully reached; the
        interpolated beats before it run at ``beat_timeout_s`` each.
        """
        wanted = {str(k): float(v) for k, v in dict(targets).items()}
        for i in range(1, self._sub + 1):
            final = i == self._sub
            frac = i / self._sub
            step = {
                name: self._last.get(name, 0.0)
                + (value - self._last.get(name, 0.0)) * frac
                for name, value in wanted.items()
            }
            before = dict(self._last)
            started = time.perf_counter()
            after = self._inner.move_joints_blocking(
                step, timeout_s=None if final else self._beat_timeout_s
            )
            seconds = time.perf_counter() - started
            self._last.update(after)
            self.beats.append((before, dict(self._last), self.scoop_state(), seconds))
        return dict(self._last)

    def clear_beats(self) -> None:
        """Forget recorded beats (start of a new episode within a chain)."""
        self.beats.clear()


# ============================================================================ goal sampler
def sample_goal(
    rng: np.random.Generator,
    *,
    reach_min_m: float,
    reach_max_m: float,
    dump_dist_range: tuple[float, float] = (2.0, 4.0),
    max_tries: int = 200,
) -> dict[str, float]:
    """One (dig, dump) pair: area-uniform dig point in the annulus, dump point
    ``dump_dist_range`` metres away (random bearing) that still fits the
    annulus — mirroring what the SKILL.md tells the planner to choose."""
    for _ in range(max_tries):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        radius = math.sqrt(rng.uniform(reach_min_m**2, reach_max_m**2))
        dig_x, dig_y = radius * math.cos(angle), radius * math.sin(angle)
        dump_angle = rng.uniform(0.0, 2.0 * math.pi)
        dump_dist = rng.uniform(*dump_dist_range)
        dump_x, dump_y = (
            dig_x + dump_dist * math.cos(dump_angle),
            dig_y + dump_dist * math.sin(dump_angle),
        )
        if reach_min_m <= math.hypot(dump_x, dump_y) <= reach_max_m:
            return {
                "dig_x_m": dig_x,
                "dig_y_m": dig_y,
                "dump_x_m": dump_x,
                "dump_y_m": dump_y,
            }
    raise RuntimeError(
        f"could not sample a valid (dig, dump) pair within {max_tries} tries — "
        "is the annulus large enough for the dump distance range?"
    )


# ============================================================================ episode
@dataclass
class Episode:
    """One dig-and-dump cycle: arrays + the metadata that makes it auditable."""

    source: str  # "teacher" | "keyboard"
    backend: str
    goal: dict[str, float] | None
    joint_names: tuple[str, ...]
    joint_units: str | None
    swing_unit: str
    beat_timeout_s: float
    fps_nominal: int
    joints_before: np.ndarray  # (N, joints) float64
    joints_after: np.ndarray  # (N, joints) float64
    scoop: np.ndarray  # (N,) bool
    beat_seconds: np.ndarray  # (N,) float32
    sub_beats: int | None = None  # teacher only
    seed: int | None = None  # teacher only
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    @property
    def n_beats(self) -> int:
        return int(self.joints_before.shape[0])

    def slice(self, start: int, stop: int) -> Episode:
        """A sub-interval of this episode with the same metadata."""
        return Episode(
            source=self.source,
            backend=self.backend,
            goal=self.goal,
            joint_names=self.joint_names,
            joint_units=self.joint_units,
            swing_unit=self.swing_unit,
            beat_timeout_s=self.beat_timeout_s,
            fps_nominal=self.fps_nominal,
            joints_before=self.joints_before[start:stop],
            joints_after=self.joints_after[start:stop],
            scoop=self.scoop[start:stop],
            beat_seconds=self.beat_seconds[start:stop],
            sub_beats=self.sub_beats,
            seed=self.seed,
            created_at=self.created_at,
        )

    def successful(self) -> bool:
        """Scoop truth went loaded→unloaded within the episode (D4/D8's hard
        criterion — the same measured mass threshold dig relies on)."""
        return self.n_beats > 0 and bool(self.scoop.any()) and not bool(self.scoop[-1])

    def frames(self) -> tuple[np.ndarray, np.ndarray]:
        """(states, actions) as (N, joints) float32 — the lerobot frame pair.

        state_t = measured before beat t; action_t = measured after beat t.
        float32 exactly here: this is the single dtype gate before lerobot's
        exact validation (design doc P5).
        """
        return (
            self.joints_before.astype(np.float32),
            self.joints_after.astype(np.float32),
        )

    def write_npz(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "format_version": FORMAT_VERSION,
            "source": self.source,
            "backend": self.backend,
            "robot": "agx_excavator",
            "joint_names": list(self.joint_names),
            "joint_units": self.joint_units,
            "swing_unit": self.swing_unit,
            "goal": self.goal,
            "seed": self.seed,
            "beat_timeout_s": self.beat_timeout_s,
            "sub_beats": self.sub_beats,
            "fps_nominal": self.fps_nominal,
            "created_at": self.created_at,
        }
        np.savez_compressed(
            path,
            joints_before=self.joints_before,
            joints_after=self.joints_after,
            scoop=self.scoop,
            beat_seconds=self.beat_seconds,
            **{_META_KEY: json.dumps(meta, ensure_ascii=False)},
        )
        return path

    @classmethod
    def read_npz(cls, path: str | Path) -> Episode:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data[_META_KEY]))
            if int(meta.get("format_version", -1)) != FORMAT_VERSION:
                raise ValueError(
                    f"{path.name}: format_version {meta.get('format_version')!r} != {FORMAT_VERSION} "
                    "— re-record or upgrade the converter"
                )
            return cls(
                source=str(meta["source"]),
                backend=str(meta["backend"]),
                goal=meta.get("goal"),
                joint_names=tuple(meta["joint_names"]),
                joint_units=meta.get("joint_units"),
                swing_unit=str(meta["swing_unit"]),
                beat_timeout_s=float(meta["beat_timeout_s"]),
                fps_nominal=int(meta["fps_nominal"]),
                joints_before=np.asarray(data["joints_before"], dtype=np.float64),
                joints_after=np.asarray(data["joints_after"], dtype=np.float64),
                scoop=np.asarray(data["scoop"], dtype=bool),
                beat_seconds=np.asarray(data["beat_seconds"], dtype=np.float32),
                sub_beats=meta.get("sub_beats"),
                seed=meta.get("seed"),
                created_at=str(meta.get("created_at", "")),
            )

    @classmethod
    def from_proxy(
        cls,
        proxy: RecordingDriverProxy,
        *,
        source: str,
        backend: str,
        goal: dict[str, float] | None,
        joint_units: str | None,
        swing_unit: str,
        beat_timeout_s: float,
        fps_nominal: int,
        sub_beats: int | None = None,
        seed: int | None = None,
        final_scoop: bool | None = None,
    ) -> Episode:
        names = list(proxy.joint_names)
        before = np.asarray(
            [[b[n] for n in names] for b, _, _, _ in proxy.beats], dtype=np.float64
        )
        after = np.asarray(
            [[a[n] for n in names] for _, a, _, _ in proxy.beats], dtype=np.float64
        )
        scoop = np.asarray([s for _, _, s, _ in proxy.beats], dtype=bool)
        seconds = np.asarray([d for _, _, _, d in proxy.beats], dtype=np.float32)
        if final_scoop is not None and scoop.size:
            # The episode's TERMINAL scoop state: on measured backends the mass
            # already reads unloaded during the dump beats, but on flag-based
            # backends (mock) the cycle writes mark_scoop(False) only AFTER the
            # last motion — without this override every mock episode would read
            # "never dumped".
            scoop[-1] = bool(final_scoop)
        return cls(
            source=source,
            backend=backend,
            goal=goal,
            joint_names=tuple(names),
            joint_units=joint_units,
            swing_unit=swing_unit,
            beat_timeout_s=beat_timeout_s,
            fps_nominal=fps_nominal,
            joints_before=before,
            joints_after=after,
            scoop=scoop,
            beat_seconds=seconds,
            sub_beats=sub_beats,
            seed=seed,
        )


# ============================================================================ converter gates
def unit_conflict(first: Episode, other: Episode) -> str | None:
    """Dataset-level unit mismatch between two episodes, or None.

    The MEAN_STD statistics are computed per dataset: one episode recorded with
    swing in radians beside one in degrees would train a policy whose outputs
    are systematically wrong in BOTH unit systems at deployment. This is a
    collector-config bug (design doc P4), so the converter fails the whole run
    instead of filtering episodes.
    """
    if other.joint_names != first.joint_names:
        return f"joint_names {other.joint_names} != {first.joint_names}"
    if other.joint_units != first.joint_units:
        return f"joint_units {other.joint_units!r} != {first.joint_units!r}"
    if other.swing_unit != first.swing_unit:
        return f"swing_unit {other.swing_unit!r} != {first.swing_unit!r}"
    return None


def admission_reason(
    ep: Episode, *, allow_mock: bool = False, min_beats: int = _MIN_BEATS
) -> str | None:
    """Why this episode must NOT enter the training set, or None if it may.

    The four gates from the design doc §5, in refusal order: mock pollution,
    missing goal, incomplete cycle, degenerate length.
    """
    if ep.backend == "mock" and not allow_mock:
        return "mock data must not enter a training dataset (use --allow-mock only for pipeline self-tests)"
    if ep.goal is None:
        return "episode has no goal (keyboard stream not segmented/confirmed)"
    if any(not math.isfinite(float(ep.goal[key])) for key in GOAL_KEYS):
        return "episode goal contains non-finite coordinates"
    if not ep.successful():
        if not len(ep.scoop) or not bool(ep.scoop.any()):
            return "cycle incomplete: bucket never loaded"
        return "cycle incomplete: bucket never dumped"
    if ep.n_beats < min_beats:
        return f"too short: {ep.n_beats} beats < {min_beats}"
    return None


# ============================================================================ keyboard segmentation
def segment_keyboard(ep: Episode) -> list[Episode]:
    """Cut a continuous keyboard stream into candidate episodes.

    A cycle starts at the first unloaded tick after a loaded stretch and ends
    at the next unloaded tick after loading — i.e. boundaries fall on
    loaded→unloaded transitions. Segments inherit ``goal=None`` (the operator
    never stated one); the converter's goal gate drops them until the
    goal-confirmation enhancement lands (design doc §4).
    """
    if ep.source != "keyboard":
        raise ValueError("segment_keyboard is for keyboard streams only")
    segments: list[Episode] = []
    start = 0
    was_loaded = False
    for i, loaded in enumerate(ep.scoop.tolist()):
        if was_loaded and not loaded:
            segments.append(ep.slice(start, i + 1))
            start = i + 1
            was_loaded = False
        elif loaded:
            was_loaded = True
    if start < ep.n_beats and was_loaded:
        # trailing loaded stretch never dumped → keep as a (failing) candidate
        segments.append(ep.slice(start, ep.n_beats))
    return segments
