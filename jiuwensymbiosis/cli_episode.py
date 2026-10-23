# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Replay a planning-level (episode) trace: ``jiuwensymbiosis-episode <file.jsonl>``.

The tool-level replay in ``cli.py`` answers "what did each tool call do". This
command answers the planning questions: which tier produced the plan, why a
step failed, what the re-planner changed, and how the episode ended.

It is deliberately a separate entry point: the two formats have different event
types and timelines, and evolving this one must not change the existing
``jiuwensymbiosis-replay`` output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TextIO

from jiuwensymbiosis.agent.episode_trace import EpisodeDocument, load_episode

__all__ = ["episode_main", "episode_replay"]

logger = logging.getLogger(__name__)


def _top_summary(doc: EpisodeDocument) -> str:
    episode = doc.episode
    outcome = doc.outcome
    parts = [
        f"query: {episode.get('query')}",
        f"robot={episode.get('robot')}  mode={episode.get('mode')}",
        f"workspace={episode.get('workspace')}",
    ]
    versions = episode.get("versions")
    if versions:
        parts.append(f"versions={json.dumps(versions, ensure_ascii=False)}")
    if doc.bad_lines:
        parts.append(f"bad_lines={list(doc.bad_lines)}")
    if outcome:
        parts.append(
            f"outcome: ok={outcome.get('ok')} steps={outcome.get('steps_done')} replans={outcome.get('replans')}"
        )
    return "\n".join(parts)


def episode_replay(path: str, *, out: TextIO = sys.stdout, as_json: bool = False) -> int:
    """Print one episode as a text timeline (or JSON); return a process exit code."""
    try:
        doc = load_episode(path)
    except OSError as exc:
        logger.error("episode not found: %s", exc)
        return 1
    if not doc.events:
        logger.error("episode has no readable events: %s", path)
        return 1
    if as_json:
        json.dump(doc.as_dict(), out, ensure_ascii=False, indent=2, default=str)
        out.write("\n")
        return 0

    print(f"=== Episode: {Path(path).name} ===", file=out)
    print(_top_summary(doc), file=out)
    print("", file=out)

    for event in doc.events:
        kind = event.get("type")
        if kind == "intent":
            if event.get("ok"):
                targets = (event.get("intent") or {}).get("targets") or []
                print(f"[intent] ok  targets={targets}", file=out)
            else:
                print(f"[intent] FAILED: {event.get('error')}", file=out)
        elif kind == "world":
            state = event.get("state") or {}
            print(
                f"[world] tokens={state.get('tokens')}  capabilities={state.get('capabilities')}",
                file=out,
            )
        elif kind == "plan_attempt":
            status = "ok" if event.get("ok") else "FAILED"
            skills = event.get("skills") or []
            steps = event.get("validated_steps") or []
            print(
                f"[plan #{event.get('attempt')}] {status}  trigger={event.get('trigger')}  "
                f"tier={event.get('tier')}  skills={skills}  steps={len(steps)}",
                file=out,
            )
            if event.get("error"):
                print(f"         error: {event['error']}", file=out)
        elif kind == "step":
            mark = "ok  " if event.get("ok") else "FAIL"
            duration = event.get("duration_s")
            suffix = f"  {duration:.2f}s" if isinstance(duration, (int, float)) else ""
            line = f"  [s{event.get('step_id')}] {mark} {event.get('op')}{suffix}"
            if event.get("cancelled"):
                line += "  (cancelled)"
            print(line, file=out)
            if event.get("error"):
                print(f"         error: {event['error']}", file=out)
        elif kind == "replan":
            budget = event.get("budget") or {}
            status = "ok" if event.get("ok") else "FAILED"
            print(
                f"[replan @s{event.get('at_step')}] {status}  trigger={event.get('trigger')}  "
                f"budget={budget.get('used')}/{budget.get('max')}",
                file=out,
            )
            print(f"         why: {event.get('why')}", file=out)
            if event.get("error"):
                print(f"         error: {event['error']}", file=out)
            new_steps = event.get("new_steps") or []
            if new_steps:
                ops = [step.get("op") or f"loop:{step.get('loop')}" for step in new_steps]
                print(f"         new: {ops}", file=out)
        elif kind == "outcome":
            print(
                f"[outcome] ok={event.get('ok')}  steps={event.get('steps_done')}  "
                f"replans={event.get('replans')}  attempts={event.get('attempts')}  "
                f"final_tier={event.get('final_tier')}",
                file=out,
            )
            if event.get("failure_summary"):
                print(f"         failures: {event['failure_summary']}", file=out)
            if event.get("trace_path"):
                print(f"         tool trace: {event['trace_path']}", file=out)
            if event.get("error"):
                print(f"         error: {event['error']}", file=out)
    return 0


def episode_main() -> int:
    """Console-script entry: ``jiuwensymbiosis-episode <file.jsonl> [--json]``."""
    parser = argparse.ArgumentParser(
        prog="jiuwensymbiosis-episode",
        description="Replay a planning-level (episode) JSONL trace.",
    )
    parser.add_argument("episode_path", help="Path to an episodes/*.jsonl file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the parsed episode as JSON instead of a text timeline.",
    )
    args = parser.parse_args()
    return episode_replay(args.episode_path, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(episode_main())
