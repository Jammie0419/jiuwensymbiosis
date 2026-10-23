# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Planning-level (episode) trace — one JSONL file per task run.

This is the upper layer of the two-layer trace described in
``extensions/jiuwen_agx/docs/agent-loop-and-evolution.md`` (方向一):

* ``TraceRail`` (``agent/trace.py``) answers *what each tool call did*;
* this module answers *why the plan was chosen, how it was repaired, and what
  the task outcome was*, and references the tool-level file through
  ``outcome.trace_path`` / ``TraceEntry.runner_step``.

Design constraints:

* **Append-only JSONL.** One event per line, flushed as it happens, so a crash
  still leaves the events written so far. A reader skips a corrupt line and
  keeps going; a bad line never discards the batch.
* **Best effort.** Tracing is observability, not part of the task. A write or
  serialisation failure logs once and disables the recorder; it never raises
  into the runner (which would trigger its recovery path).
* **Dependency-light.** Standard library + ``utils/json_safe.py`` only, so the
  runner can import the sink type without pulling in openjiuwen.

The event schema is ``episode/1``; see ``docs/zh/reference/episode-trace.md``.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jiuwensymbiosis.utils.json_safe import json_safe

__all__ = [
    "EPISODE_SCHEMA",
    "EpisodeDocument",
    "EpisodeRecorder",
    "NullEpisodeRecorder",
    "TraceSink",
    "emit_safely",
    "episode_path",
    "iter_episode",
    "load_episode",
]

logger = logging.getLogger(__name__)

EPISODE_SCHEMA = "episode/1"

# Strict-JSON limits. World snapshots and plan sequences are small; these only
# stop a pathological value (a huge terrain list, a megabyte string) from
# turning one event into a file-sized line.
_MAX_STRING = 16_000
_MAX_ITEMS = 1_000
_MAX_DEPTH = 12

# A sink receives one event dict. The recorder's ``emit`` is the production
# implementation; tests use ``list.append``.
TraceSink = Callable[[dict[str, Any]], None]


def _utc_now() -> str:
    """UTC ISO-8601 with millisecond precision (stable, sortable)."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _safe_component(value: Any, *, limit: int = 64) -> str:
    """Sanitise a value for use as a filename component."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return cleaned[:limit] or "episode"


def episode_path(
    workspace: str | Path,
    episode_dir: str | Path | None,
    episode_id: str,
) -> Path:
    """Return the JSONL path for one episode.

    ``episode_dir`` wins when given (used verbatim, like ``trace_dir``);
    otherwise ``<workspace>/episodes``. The filename carries a timestamp, a
    microsecond component and the pid, so repeated runs under the same
    conversation id never overwrite each other.
    """
    base = Path(episode_dir) if episode_dir else Path(workspace) / "episodes"
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    return base / (f"{_safe_component(episode_id)}_{stamp}_{now.microsecond:06d}_{os.getpid()}.jsonl")


class EpisodeRecorder:
    """Append-only writer for one episode.

    The first line is the ``episode`` header (written by ``__init__``); every
    later call appends one event. Use it as a context manager so an aborted run
    still gets an ``outcome`` line.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        episode_id: str,
        query: str | None = None,
        robot: str | None = None,
        mode: str = "fastagent",
        workspace: str | None = None,
        versions: Mapping[str, Any] | None = None,
        capabilities: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.episode_id = episode_id
        self._lock = threading.Lock()
        self._seq = 0
        self._broken = False
        self._closed = False
        self._outcome_written = False
        self._plan_attempts = 0
        self.last_plan_tier: str | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self.emit(
            {
                "type": "episode",
                "episode_id": episode_id,
                "run_token": self.path.stem,
                "query": query,
                "robot": robot,
                "mode": mode,
                "workspace": workspace,
                "versions": dict(versions or {}),
                "capabilities": sorted(str(cap) for cap in (capabilities or ())),
                "started_at": _utc_now(),
            }
        )

    # ------------------------------------------------------------------ writing
    def emit(self, event: Mapping[str, Any]) -> None:
        """Append one event; never raises.

        A failure (disk full, a value that slipped past ``json_safe``) disables
        the recorder after one warning. Continuing to retry would add a warning
        per step without producing data.
        """
        if self._closed or self._broken:
            return
        record = dict(event)
        record.setdefault("type", "event")
        with self._lock:
            self._seq += 1
            record["schema"] = EPISODE_SCHEMA
            record["seq"] = self._seq
            record["ts"] = _utc_now()
            safe = json_safe(
                record,
                max_depth=_MAX_DEPTH,
                max_string=_MAX_STRING,
                max_items=_MAX_ITEMS,
                finite_only=True,
                truncate_strings=True,
            )
            try:
                line = json.dumps(safe, ensure_ascii=False, allow_nan=False)
                self._fh.write(line + "\n")
                self._fh.flush()
            except (OSError, TypeError, ValueError) as exc:
                self._broken = True
                logger.warning(
                    "[episode] %s: write failed, episode trace disabled: %s",
                    self.path,
                    exc,
                )
                return
        if record.get("type") == "outcome":
            self._outcome_written = True

    # --------------------------------------------------------------- convenience
    def intent(
        self,
        *,
        ok: bool,
        intent: Mapping[str, Any] | None = None,
        error: str | None = None,
        duration_s: float | None = None,
    ) -> None:
        self.emit(
            {
                "type": "intent",
                "ok": bool(ok),
                "intent": dict(intent or {}) if intent is not None else None,
                "error": error,
                "duration_s": duration_s,
            }
        )

    def world(self, state: Mapping[str, Any], *, kind: str = "initial") -> None:
        """Record a world snapshot (``WorldState.describe()``)."""
        self.emit({"type": "world", "kind": kind, "state": dict(state)})

    def plan_attempt(
        self,
        *,
        trigger: str = "initial",
        attempt: int | None = None,
        tier: str | None = None,
        skills: Sequence[str] | None = None,
        reason: str = "",
        raw: Any = None,
        validated_steps: Sequence[str] | None = None,
        ok: bool = True,
        error: str | None = None,
        duration_s: float | None = None,
        at_step: int | None = None,
    ) -> None:
        """Record one ``plan_task`` call (initial plan or a re-plan)."""
        number = int(attempt) if attempt is not None else self._plan_attempts + 1
        self._plan_attempts = max(self._plan_attempts, number)
        if ok and tier:
            self.last_plan_tier = str(tier)
        self.emit(
            {
                "type": "plan_attempt",
                "attempt": number,
                "trigger": trigger,
                "tier": tier,
                "skills": list(skills or ()),
                "reason": reason,
                "raw": raw,
                "validated_steps": list(validated_steps or ()),
                "ok": bool(ok),
                "error": error,
                "duration_s": duration_s,
                "at_step": at_step,
            }
        )

    def step(
        self,
        *,
        step_id: int,
        op: str,
        params: Mapping[str, Any] | None = None,
        bind: str | None = None,
        ok: bool,
        result: Any = None,
        error: str | None = None,
        error_code: str | None = None,
        recovery_managed: bool = False,
        duration_s: float | None = None,
        tokens_after: Sequence[str] | None = None,
        loop_iter: int | None = None,
        cancelled: bool = False,
    ) -> None:
        self.emit(
            {
                "type": "step",
                "step_id": step_id,
                "op": op,
                "params": dict(params or {}),
                "bind": bind,
                "ok": bool(ok),
                "result": result,
                "error": error,
                "error_code": error_code,
                "recovery_managed": bool(recovery_managed),
                "duration_s": duration_s,
                "tokens_after": list(tokens_after) if tokens_after is not None else None,
                "loop_iter": loop_iter,
                "cancelled": bool(cancelled),
            }
        )

    def replan(
        self,
        *,
        at_step: int | None,
        trigger: str,
        why: str,
        ok: bool,
        error: str | None = None,
        budget: Mapping[str, Any] | None = None,
        old_remaining: Sequence[Mapping[str, Any]] | None = None,
        new_steps: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.emit(
            {
                "type": "replan",
                "at_step": at_step,
                "trigger": trigger,
                "why": why,
                "ok": bool(ok),
                "error": error,
                "budget": dict(budget or {}),
                "old_remaining": list(old_remaining or ()),
                "new_steps": list(new_steps or ()),
            }
        )

    def outcome(self, *, ok: bool, **fields: Any) -> None:
        self.emit({"type": "outcome", "ok": bool(ok), **fields})

    @property
    def plan_attempts(self) -> int:
        return self._plan_attempts

    # ---------------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Write a fallback outcome if needed, then close the file (idempotent)."""
        if self._closed:
            return
        if not self._outcome_written:
            self.outcome(ok=False, error="aborted")
        self._closed = True
        try:
            self._fh.close()
        except OSError as exc:
            logger.warning("[episode] %s: close failed: %s", self.path, exc)

    def __enter__(self) -> EpisodeRecorder:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if not self._outcome_written:
            if exc is not None:
                self.outcome(ok=False, error=f"{exc_type.__name__}: {exc}")
            else:
                self.outcome(ok=False, error="aborted")
        self.close()
        return False


class NullEpisodeRecorder:
    """No-op recorder used when ``enable_episode_trace`` is off."""

    path: Path | None = None
    episode_id: str = ""
    last_plan_tier: str | None = None
    plan_attempts: int = 0

    def emit(self, event: Mapping[str, Any]) -> None:
        return None

    def intent(self, **fields: Any) -> None:
        return None

    def world(self, state: Mapping[str, Any], **fields: Any) -> None:
        return None

    def plan_attempt(self, **fields: Any) -> None:
        return None

    def step(self, **fields: Any) -> None:
        return None

    def replan(self, **fields: Any) -> None:
        return None

    def outcome(self, **fields: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> NullEpisodeRecorder:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def emit_safely(sink: TraceSink | None, event: Mapping[str, Any]) -> None:
    """Call ``sink`` without letting a trace failure escape into execution."""
    if sink is None:
        return
    try:
        sink(dict(event))
    except Exception as exc:  # noqa: BLE001 - observability must never break a run
        logger.warning("[episode] trace sink failed: %s", exc)


# --------------------------------------------------------------------- reading
@dataclass(frozen=True)
class EpisodeDocument:
    """Parsed episode file plus the line numbers that could not be read."""

    path: Path
    events: tuple[dict[str, Any], ...]
    bad_lines: tuple[int, ...] = ()

    def _of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("type") == event_type]

    @property
    def episode(self) -> dict[str, Any]:
        found = self._of_type("episode")
        return found[0] if found else {}

    @property
    def intents(self) -> list[dict[str, Any]]:
        return self._of_type("intent")

    @property
    def worlds(self) -> list[dict[str, Any]]:
        return self._of_type("world")

    @property
    def plan_attempts(self) -> list[dict[str, Any]]:
        return self._of_type("plan_attempt")

    @property
    def steps(self) -> list[dict[str, Any]]:
        return self._of_type("step")

    @property
    def replans(self) -> list[dict[str, Any]]:
        return self._of_type("replan")

    @property
    def outcome(self) -> dict[str, Any]:
        found = self._of_type("outcome")
        return found[-1] if found else {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "intents": self.intents,
            "worlds": self.worlds,
            "plan_attempts": self.plan_attempts,
            "steps": self.steps,
            "replans": self.replans,
            "outcome": self.outcome,
            "bad_lines": list(self.bad_lines),
        }


def _iter_raw(path: str | Path) -> Iterator[tuple[int, dict[str, Any] | None]]:
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("[episode] %s:%d: skipping bad line: %s", path, lineno, exc)
                yield lineno, None
                continue
            if not isinstance(event, dict):
                logger.warning(
                    "[episode] %s:%d: skipping non-object line (%s)",
                    path,
                    lineno,
                    type(event).__name__,
                )
                yield lineno, None
                continue
            yield lineno, event


def iter_episode(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield every readable event in an episode file, skipping corrupt lines."""
    for _lineno, event in _iter_raw(path):
        if event is not None:
            yield event


def load_episode(path: str | Path) -> EpisodeDocument:
    """Parse an episode file into an ``EpisodeDocument``.

    Bad lines are recorded in ``bad_lines`` rather than raised: the point of
    the format is that a truncated final line (a crash mid-write) does not make
    the earlier events unreadable.
    """
    events: list[dict[str, Any]] = []
    bad: list[int] = []
    for lineno, event in _iter_raw(path):
        if event is None:
            bad.append(lineno)
        else:
            events.append(event)
    return EpisodeDocument(path=Path(path), events=tuple(events), bad_lines=tuple(bad))
