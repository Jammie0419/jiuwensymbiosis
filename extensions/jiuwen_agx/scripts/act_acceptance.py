# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ACT 部署验收：留出目标点回放 + dig 同点对照（runbook 阶段八 / 阶段六门槛 3）。

每条"留出点"是一次完整挖卸循环的 goal 四元组，两条硬条件：

1. **训练没见过**：与 ``--exclude-raw`` 里每个训练 episode 的 (挖点, 卸点) 相比，
   挖点或卸点至少一侧相距 ≥ ``--min-distance``（不允许整对都贴得近——
   那样就是复读训练样本）。半径分近/中/远三档轮转、方位每 60° 扇区最多 2 个，
   覆盖不塌到一角。
2. **判据与部署一致**：成功 = 铲斗质量真值 loaded→unloaded（``act_exec``/``dig``
   内部同一判据）；失败原因按 error 文本分类入表。

用法（真桥接，需先起 start_agx_bridge.bat 并配好 policy 节）::

    python scripts/act_acceptance.py \
        --config configs/agx_excavator/agx_excavator.local.yaml \
        --exclude-raw data/raw/run_all --points 10 --seed 7 --also-dig \
        --out data/acceptance/act_v1.json

mock 干跑（不用桥接，验证真 checkpoint 的加载+推理链路；mock 斗永远不载料，
``max_beats`` 兜底失败是**预期**）::

    python scripts/act_acceptance.py --config <mock+policy 配置> --points 1 --allow-mock
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from jiuwen_agx.agx_excavator.config import AgxExcavatorConfig
from jiuwen_agx.agx_excavator.record import Episode, sample_goal
from jiuwen_agx.agx_excavator.work import DEFAULT_DIG_TUNING

__all__ = ["main", "select_heldout_goals"]

# 半径档（在 cfg.reach_min_m/reach_max_m 内侧留余量），近/中/远轮转
_BANDS: tuple[tuple[float, float], ...] = ((1.5, 3.0), (3.0, 4.5), (4.5, 5.9))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        required=True,
        help="machine YAML (remote backend for real acceptance)",
    )
    parser.add_argument(
        "--points", type=int, default=10, help="number of held-out goal points"
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="held-out goal sampling seed"
    )
    parser.add_argument(
        "--exclude-raw",
        default=None,
        help="raw npz dir of the training set (data/raw/run_all) — held-out goals keep away from it",
    )
    parser.add_argument(
        "--min-distance", type=float, default=0.5, help="min keep-away distance, metres"
    )
    parser.add_argument(
        "--also-dig",
        action="store_true",
        help="run the scripted dig teacher on the SAME points (comparison table)",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="permit the mock backend (dry-run only — mock truth never loads the bucket)",
    )
    parser.add_argument(
        "--out", default=None, help="write the full result table as JSON here"
    )
    return parser


# ---------------------------------------------------------------- held-out goals
def _training_goals(raw_dir: str | Path) -> list[dict[str, float]]:
    goals: list[dict[str, float]] = []
    for path in sorted(Path(raw_dir).glob("*.npz")):
        episode = Episode.read_npz(path)
        if episode.goal:
            goals.append({str(k): float(v) for k, v in episode.goal.items()})
    return goals


def _novel_enough(
    goal: dict[str, float], training: list[dict[str, float]], min_dist: float
) -> bool:
    """True when no training pair is close on BOTH the dig and the dump point."""
    for t in training:
        dd = math.hypot(goal["dig_x_m"] - t["dig_x_m"], goal["dig_y_m"] - t["dig_y_m"])
        du = math.hypot(
            goal["dump_x_m"] - t["dump_x_m"], goal["dump_y_m"] - t["dump_y_m"]
        )
        if dd < min_dist and du < min_dist:
            return False
    return True


def select_heldout_goals(
    rng: np.random.Generator,
    cfg: AgxExcavatorConfig,
    training: list[dict[str, float]],
    *,
    count: int,
    min_dist: float,
) -> list[dict[str, float]]:
    """Stratified, keep-away-sampled goal points for acceptance replay."""
    chosen: list[dict[str, float]] = []
    sectors: dict[int, int] = {}
    tries = 0
    while len(chosen) < count and tries < 50_000:
        tries += 1
        goal = sample_goal(
            rng, reach_min_m=cfg.reach_min_m, reach_max_m=cfg.reach_max_m
        )
        radius = math.hypot(goal["dig_x_m"], goal["dig_y_m"])
        low, high = _BANDS[len(chosen) % len(_BANDS)]
        if not (low <= radius < high):
            continue
        sector = int(
            (math.atan2(goal["dig_y_m"], goal["dig_x_m"]) + math.pi) // (math.pi / 3)
        )
        if sectors.get(sector, 0) >= 2:
            continue
        if not _novel_enough(goal, training, min_dist):
            continue
        sectors[sector] = sectors.get(sector, 0) + 1
        chosen.append(goal)
    if len(chosen) < count:
        raise SystemExit(
            f"could only find {len(chosen)}/{count} held-out goals in {tries} tries — "
            "loosen --min-distance or shrink --exclude-raw"
        )
    return chosen


# ---------------------------------------------------------------- bucket hygiene
def _dump_residual(
    session: object, cfg: AgxExcavatorConfig, *, timeout_s: float = 90.0
) -> bool:
    """Physically empty a loaded bucket after a failed cycle.

    On the real bridge the scoop truth is MEASURED (aggregate particle mass >
    threshold) — ``mark_scoop`` is only a protocol-compat flag there, so the
    only way to clear it is to actually pour: drive the dump keyframe (bucket
    wide open) and wait for the mass to fall. Without this, the next point is
    refused by ``check_cycle_preconditions`` ("bucket is still loaded").
    """
    driver = session.env.driver  # type: ignore[attr-defined]
    if not driver.scoop_state():
        return True
    tuning = {**DEFAULT_DIG_TUNING, **(cfg.dig_cycle_tuning or {})}
    dump_frame = {
        "boom": tuning["dump_boom_deg"],
        "arm": tuning["dump_arm_deg"],
        "bucket": tuning["dump_bucket_deg"],
    }
    try:
        driver.move_joints_blocking(dump_frame, timeout_s=30.0)
    except ValueError:
        pass  # posture refused: the mass poll below still decides
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline and driver.scoop_state():
        time.sleep(1.0)
    driver.mark_scoop(False)  # no-op on the AGX truth path; clears the mock flag
    return not driver.scoop_state()


# ---------------------------------------------------------------- one cycle
def _run_cycle(
    api: object, goal: dict[str, float], *, method: str, index: int
) -> dict[str, object]:
    started = time.perf_counter()
    if method == "act_exec":
        result = api.act_exec(**goal)  # type: ignore[attr-defined]
    else:
        result = api.dig(**goal)  # type: ignore[attr-defined]
    wall_s = time.perf_counter() - started
    row: dict[str, object] = {"method": method, "index": index, **goal}
    if result.get("ok"):
        # act_exec promises {ok, beats, cycle_s}; dig promises {ok, volume_m3,
        # cycle_s} — beats simply is not part of dig's contract.
        row.update(ok=True, cycle_s=float(result["cycle_s"]))
        if "beats" in result:
            row["beats"] = int(result["beats"])
        if "volume_m3" in result:
            row["volume_m3"] = float(result["volume_m3"])
        beats = f"beats={row['beats']:4d} " if "beats" in row else ""
        print(
            f"[accept] #{index:02d} {method:8s} ok  {beats}"
            f"cycle_s={row['cycle_s']:6.1f} dig=({goal['dig_x_m']:+.2f},{goal['dig_y_m']:+.2f}) "
            f"dump=({goal['dump_x_m']:+.2f},{goal['dump_y_m']:+.2f})",
            flush=True,
        )
    else:
        row.update(
            ok=False, error=str(result.get("error", "?")), wall_s=round(wall_s, 1)
        )
        print(
            f"[accept] #{index:02d} {method:8s} BAD wall_s={wall_s:6.1f} "
            f"error={row['error']}",
            flush=True,
        )
    return row


def _summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method in ("act_exec", "dig"):
        subset = [r for r in rows if r["method"] == method]
        if not subset:
            continue
        oks = [r for r in subset if r["ok"]]
        beats = [r["beats"] for r in oks if "beats" in r]
        summary[method] = {
            "success": len(oks),
            "total": len(subset),
            "rate": round(len(oks) / len(subset), 3),
            "cycle_s_p50": round(
                float(statistics.median([r["cycle_s"] for r in oks])), 1
            )
            if oks
            else 0.0,
            "beats_p50": float(statistics.median(beats)) if beats else 0.0,
        }
    return summary


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cfg = AgxExcavatorConfig.from_yaml(args.config)
    if cfg.backend == "mock" and not args.allow_mock:
        sys.exit(
            "refusing to run acceptance on the mock backend: its scoop truth never loads, "
            "so every cycle is a foregone BAD. Point --config at a remote (real bridge) "
            "config, or pass --allow-mock for a dry-run of the loading/inference chain."
        )

    training = _training_goals(args.exclude_raw) if args.exclude_raw else []
    if args.exclude_raw and not training:
        sys.exit(f"--exclude-raw {args.exclude_raw} holds no episodes with a goal")
    rng = np.random.default_rng(args.seed)
    goals = select_heldout_goals(
        rng, cfg, training, count=args.points, min_dist=args.min_distance
    )
    print(
        f"[accept] config={args.config} backend={cfg.backend} points={len(goals)} "
        f"seed={args.seed} trained-on={len(training)} pairs min_dist>={args.min_distance}m",
        flush=True,
    )

    from jiuwen_agx.agx_excavator import build_agx_excavator_session

    session = build_agx_excavator_session.from_yaml(args.config)
    rows: list[dict[str, object]] = []
    with session:
        for index, goal in enumerate(goals, start=1):
            for method in ("act_exec", "dig") if args.also_dig else ("act_exec",):
                rows.append(_run_cycle(session.api, goal, method=method, index=index))
                if not _dump_residual(session, cfg):
                    print(
                        "[accept] ABORT: could not empty the bucket after a failed cycle",
                        flush=True,
                    )
                    rows.append(
                        {
                            "method": "abort",
                            "index": index,
                            "reason": "bucket never emptied",
                        }
                    )
                    break
            else:
                continue
            break

    summary = _summarize(rows)
    print("[accept] === summary ===")
    print(
        f"[accept] {'method':9s} {'success':>7s} {'rate':>6s} {'cycle_s p50':>12s} {'beats p50':>10s}"
    )
    for method, s in summary.items():
        beats_cell = f"{s['beats_p50']:>10.0f}" if s["beats_p50"] else f"{'n/a':>10s}"
        print(
            f"[accept] {method:9s} {s['success']:>3.0f}/{s['total']:<3.0f} {s['rate']:>6.0%} "
            f"{s['cycle_s_p50']:>12.1f} {beats_cell}"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": args.config,
            "backend": cfg.backend,
            "seed": args.seed,
            "min_distance_m": args.min_distance,
            "training_pairs": len(training),
            "goals": goals,
            "rows": rows,
            "summary": summary,
        }
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[accept] wrote {out}")


if __name__ == "__main__":
    main()
