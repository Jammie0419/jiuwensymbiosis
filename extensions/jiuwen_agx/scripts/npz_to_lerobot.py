# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""npz_to_lerobot — convert recorded demonstrations into a LeRobotDataset.

Runs on the TRAINING machine (needs lerobot; the AGX-side collector does not).
Applies the converter's four gates (format/no-mock/goal/success — see
docs/act-data-collection-design.md §5) and writes the exact feature contract
the ACT training expects: observation.state [4] + observation.environment_state
[4] (the goal) + action [4], pure Parquet (use_videos=False, Windows-friendly).

The training machine does NOT need the jiuwensymbiosis core installed: the
recording logic is loaded from the extension's own source tree (record.py
depends on numpy only), so a bare venv with lerobot plus a copy of the
extensions/jiuwen_agx directory is enough (see docs/act-runbook.md).

    python scripts/npz_to_lerobot.py --raw data/raw/run001 \
        --root data/lerobot --repo-id local/agx_dig_demos --fps 10
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np

_TASK = "excavate"


def _load_record_module() -> ModuleType:
    """Load the recording logic with or without the core package installed.

    Preferred path is the normal package import (recording machine, core
    installed). On a bare training machine the import would drag in
    jiuwensymbiosis — fall back to loading ``record.py`` directly from the
    extension source tree; that module depends on numpy only.
    """
    try:
        from jiuwen_agx.agx_excavator import record

        return record
    except Exception:
        record_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "jiuwen_agx"
            / "agx_excavator"
            / "record.py"
        )
        if not record_path.exists():
            sys.exit(
                "cannot import jiuwen_agx and no local record.py found — run this "
                "script from a copy of the extensions/jiuwen_agx directory (see "
                "docs/act-runbook.md)"
            )
        spec = importlib.util.spec_from_file_location(
            "jiuwen_agx_record_standalone", record_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # register BEFORE exec: the @dataclass decorator resolves its owning
        # module through sys.modules during class creation
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


record = _load_record_module()
GOAL_KEYS = record.GOAL_KEYS
Episode = record.Episode
admission_reason = record.admission_reason
segment_keyboard = record.segment_keyboard
unit_conflict = record.unit_conflict

_TASK = "excavate"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--raw", required=True, help="directory of recorded .npz episodes"
    )
    parser.add_argument(
        "--root", required=True, help="dataset root (must not already exist)"
    )
    parser.add_argument("--repo-id", default="local/agx_dig_demos")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-beats", type=int, default=10)
    parser.add_argument(
        "--allow-mock", action="store_true", help="pipeline self-tests only"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="keep at most N episodes"
    )
    return parser


def _features(joint_names: tuple[str, ...]) -> dict:
    """The lerobot feature contract; `names` drives every downstream tool."""
    vector = {
        "dtype": "float32",
        "shape": (len(joint_names),),
        "names": list(joint_names),
    }
    return {
        "observation.state": dict(vector),
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (len(GOAL_KEYS),),
            "names": list(GOAL_KEYS),
        },
        "action": dict(vector),
    }


def _candidates(path: Path) -> list[Episode]:
    """Episodes to consider from one npz (keyboard streams are segmented)."""
    episode = Episode.read_npz(path)
    if episode.source == "keyboard":
        return segment_keyboard(episode)
    return [episode]


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        sys.exit(
            f"lerobot is not installed ({exc}) — run this on the training environment "
            "(pip install -e extensions/jiuwen_agx[policy] or the dedicated training venv)"
        )

    raw = Path(args.raw)
    files = sorted(raw.glob("*.npz"))
    if not files:
        sys.exit(f"no .npz episodes under {raw}")
    root = Path(args.root) / args.repo_id.replace("/", "_")
    if root.exists():
        sys.exit(f"refusing to overwrite an existing dataset root: {root}")

    first = Episode.read_npz(files[0])
    features = _features(first.joint_names)

    # Dataset-level unit gate (design doc P4's enforcement): a dataset mixing
    # unit systems (e.g. swing in rad from one config, deg from another) would
    # train a policy whose outputs are systematically wrong at deployment. This
    # is a collector bug, not a per-episode filter — fail loudly, never skip.
    for path in files[1:]:
        other = Episode.read_npz(path)
        conflict = unit_conflict(first, other)
        if conflict is not None:
            sys.exit(
                f"{path.name}: {conflict} — mixed unit systems corrupt the whole "
                "dataset; re-record with one machine config"
            )

    dataset = LeRobotDataset.create(
        args.repo_id,
        fps=args.fps,
        features=features,
        root=root,
        robot_type="agx_excavator",
        use_videos=False,
    )

    kept = 0
    skipped: Counter[str] = Counter()
    beats_hist: list[int] = []
    for path in files:
        for episode in _candidates(path):
            if episode.joint_names != first.joint_names:
                skipped["joint_names mismatch with the first episode"] += 1
                continue
            reason = admission_reason(
                episode, allow_mock=args.allow_mock, min_beats=args.min_beats
            )
            if reason is not None:
                skipped[reason.split(":")[0]] += 1
                continue
            if args.limit is not None and kept >= args.limit:
                break
            states, actions = episode.frames()
            env_state = np.asarray(
                [episode.goal[key] for key in GOAL_KEYS], dtype=np.float32
            )
            for t in range(episode.n_beats):
                dataset.add_frame(
                    {
                        "observation.state": states[t],
                        "observation.environment_state": env_state,
                        "action": actions[t],
                        "task": _TASK,
                    }
                )
            dataset.save_episode()
            beats_hist.append(episode.n_beats)
            kept += 1
        if args.limit is not None and kept >= args.limit:
            break
    dataset.finalize()

    print(f"[convert] dataset: {root}  episodes: {kept}  frames: {sum(beats_hist)}")
    if beats_hist:
        print(
            f"[convert] beats/episode: min={min(beats_hist)} median={sorted(beats_hist)[len(beats_hist) // 2]} max={max(beats_hist)}"
        )
    if skipped:
        print("[convert] skipped:")
        for reason, count in skipped.most_common():
            print(f"  {count:4d}  {reason}")


if __name__ == "__main__":
    main()
