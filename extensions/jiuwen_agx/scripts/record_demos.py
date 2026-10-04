# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""record_demos — collect ACT demonstration data on the AGX machine.

Teacher mode (default): the existing scripted dig cycle acts as the demo
source — each keyframe transition is sub-stepped into beats by
``RecordingDriverProxy`` and recorded as (measured before, measured after)
pairs. No lerobot/torch needed here; the npz output is converted on the
training machine by ``npz_to_lerobot.py``.

    python scripts/record_demos.py --config <remote.yaml> --out data/raw/run001 \
        --chains 10 --cycles-per-chain 10 --seed 7

Keyboard mode: with the bridge started as JIUWEN_KEYBOARD=1, records a
continuous spectator stream (goal-less; the converter drops goal-less
episodes — see docs/act-data-collection-design.md §4):

    python scripts/record_demos.py --config <remote.yaml> --out data/raw/kbd001 \
        --source keyboard --seconds 120

Mock backends are refused unless --allow-mock (pipeline self-tests only —
mock joints must never enter a training dataset, design doc P6).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

from jiuwen_agx.agx_excavator.config import AgxExcavatorConfig
from jiuwen_agx.agx_excavator.record import (
    Episode,
    RecordingDriverProxy,
    sample_depth_scale,
    sample_goal,
)
from jiuwen_agx.agx_excavator.work import execute_dig_cycle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--config", required=True, help="machine YAML (remote backend for real data)"
    )
    parser.add_argument("--out", required=True, help="output directory for npz files")
    parser.add_argument("--source", choices=("teacher", "keyboard"), default="teacher")
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="nominal fps metadata (beats are arrival-paced)",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="permit the mock backend (self-tests only)",
    )
    # teacher
    parser.add_argument(
        "--chains",
        type=int,
        default=10,
        help="home-to-home chains of consecutive cycles",
    )
    parser.add_argument("--cycles-per-chain", type=int, default=10)
    parser.add_argument(
        "--sub-beats",
        type=int,
        default=20,
        help="interpolated beats per keyframe transition",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="goal-sampling seed (default: random)"
    )
    # keyboard
    parser.add_argument(
        "--seconds", type=float, default=120.0, help="keyboard: stream duration"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="keyboard: spectator poll period (s)",
    )
    return parser


def _collect_teacher(args: argparse.Namespace, cfg: AgxExcavatorConfig) -> None:
    from jiuwen_agx.agx_excavator import build_agx_excavator_session

    if cfg.backend == "mock" and not args.allow_mock:
        sys.exit(
            "refusing to collect on the mock backend: mock joints must never enter a "
            "training dataset. Point --config at a remote (real bridge) config, or pass "
            "--allow-mock for a pipeline self-test whose output stays out of training."
        )
    session = build_agx_excavator_session.from_yaml(args.config)
    seed = int(args.seed if args.seed is not None else np.random.SeedSequence().entropy)
    rng = np.random.default_rng(seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = kept = 0
    print(f"[record] backend={cfg.backend} seed={seed} out={out}")
    with session:
        driver = session.env.driver
        proxy = RecordingDriverProxy(
            driver, sub_beats=args.sub_beats, beat_timeout_s=cfg.policy_beat_timeout_s
        )
        for _chain in range(args.chains):
            proxy.home()  # refresh the proxy's measured state after repositioning
            for _ in range(args.cycles_per_chain):
                goal = sample_goal(
                    rng, reach_min_m=cfg.reach_min_m, reach_max_m=cfg.reach_max_m
                )
                depth_scale = sample_depth_scale(
                    rng,
                    dig_radius_m=math.hypot(goal["dig_x_m"], goal["dig_y_m"]),
                    reach_min_m=cfg.reach_min_m,
                    reach_max_m=cfg.reach_max_m,
                )
                proxy.clear_beats()
                execute_dig_cycle(
                    proxy,
                    dig_x_m=goal["dig_x_m"],
                    dig_y_m=goal["dig_y_m"],
                    dump_x_m=goal["dump_x_m"],
                    dump_y_m=goal["dump_y_m"],
                    tuning=cfg.dig_cycle_tuning,
                    reach_min_m=cfg.reach_min_m,
                    reach_max_m=cfg.reach_max_m,
                    swing_unit=cfg.swing_unit,
                    depth_scale=depth_scale,
                )
                episode = Episode.from_proxy(
                    proxy,
                    source="teacher",
                    backend=cfg.backend,
                    goal=goal,
                    joint_units=cfg.joint_units,
                    swing_unit=cfg.swing_unit,
                    beat_timeout_s=cfg.policy_beat_timeout_s,
                    fps_nominal=args.fps,
                    sub_beats=args.sub_beats,
                    seed=seed,
                    # Terminal scoop truth: the cycle clears its flag/measured
                    # mass after the last motion beat, so read once more now.
                    final_scoop=driver.scoop_state(),
                    depth_scale=depth_scale,
                )
                path = episode.write_npz(out / f"ep_{total:04d}_teacher.npz")
                mark = "ok " if episode.successful() else "BAD"
                kept += bool(episode.successful())
                print(
                    f"[record] {mark} ep_{total:04d} beats={episode.n_beats:4d} "
                    f"depth={depth_scale:.2f} "
                    f"cycle_s={episode.beat_seconds.sum():6.1f} -> {path.name}"
                )
                total += 1
    print(
        f"[record] done: {kept}/{total} episodes passed the success criterion -> {out}"
    )


def _collect_keyboard(args: argparse.Namespace, cfg: AgxExcavatorConfig) -> None:
    from jiuwen_agx.agx_excavator import build_agx_excavator_session

    session = build_agx_excavator_session.from_yaml(args.config)
    with session:
        driver = session.env.driver
        print(
            f"[record] keyboard spectator: polling every {args.poll_interval}s for "
            f"{args.seconds}s — drive the machine (Ctrl+C to stop early)"
        )
        polls: list[tuple[dict[str, float], bool]] = []
        started = time.perf_counter()
        try:
            while time.perf_counter() - started < args.seconds:
                polls.append((driver.get_joint_positions(), bool(driver.scoop_state())))
                time.sleep(
                    max(
                        0.0,
                        args.poll_interval
                        - (time.perf_counter() - started) % args.poll_interval,
                    )
                )
        except KeyboardInterrupt:
            print("[record] interrupted — finalising the stream")
        if len(polls) < 3:
            sys.exit("[record] stream too short to save")
        names = list(cfg.joint_names)
        before = np.asarray(
            [[p[0][n] for n in names] for p in polls[:-1]], dtype=np.float64
        )
        after = np.asarray(
            [[p[0][n] for n in names] for p in polls[1:]], dtype=np.float64
        )
        scoop = np.asarray([p[1] for p in polls[1:]], dtype=bool)
        seconds = np.full(len(polls) - 1, args.poll_interval, dtype=np.float32)
        episode = Episode(
            source="keyboard",
            backend=cfg.backend,
            goal=None,  # operator never stated one; converter drops goal-less episodes
            joint_names=tuple(names),
            joint_units=cfg.joint_units,
            swing_unit=cfg.swing_unit,
            beat_timeout_s=cfg.policy_beat_timeout_s,
            fps_nominal=max(1, round(1.0 / args.poll_interval)),
            joints_before=before,
            joints_after=after,
            scoop=scoop,
            beat_seconds=seconds,
        )
        out = Path(args.out)
        path = episode.write_npz(out / "keyboard_stream.npz")
        print(f"[record] stream saved: {len(polls) - 1} ticks -> {path}")


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg = AgxExcavatorConfig.from_yaml(args.config)
    if args.source == "teacher":
        _collect_teacher(args, cfg)
    else:
        _collect_keyboard(args, cfg)


if __name__ == "__main__":
    main()
