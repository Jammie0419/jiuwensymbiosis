# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""我们的采集数据 → 小脑组 goal-act-v2 训练格式（episode_*.h5）。

两边格式差异的完整映射见 ``docs/cerebellum-interface-draft.md`` §8；本脚本是
"格式打通"的实物：以后他们在 AGX 采集也走 ``record_demos.py``，再经这里转换。

字段映射（源 = ``data/raw/<run>/*.npz``，目标 = 他们 real 载入器的 episode_*.h5）：

- ``joints_before/after``（4 关节，**原生单位**：swing 弧度 + 三缸米，到位节拍）
  → ``observations/state[:, 6:10]``（**度**）与 ``joint_rad``（弧度）。
  原生→角的换算走 ``--machine``（``jiuwen_agx.machine_map``）；**未标定的
  机器文件默认拒绝**（米值被当角度是会静默污染训练数据的，fail-closed），
  只有 ``--allow-uncalibrated`` 的干跑才放行并在文件属性里标红。
- **时间轴默认"拍 = 帧"**（一条 120 拍循环 = 120 帧，control_frequency=10 为
  名义值）：他们的训练帧本来就是"步"（IK 生成器每步运动受 max_joint_step
  限幅 ≤2.5°，评估也是逐步语义），而我们的"拍"正是"到位一步"——两者同构。
  若把事情做反（重采样到细粒度 10Hz），模型会学出 ~0.2°/步的增量，部署时
  我们的拍循环要上千拍才走完一个循环（max_beats=300 直接失败）。
  ``--resample-hz`` 可显式要求等间隔重采样（默认关闭）。
- 相邻（拍/帧）差分 → ``actions/joint_increment`` (T,16,4)。**写成度**：他们
  载入器用"99 分位 >0.5 即视为度"的启发式自动识别，度制在正常与异常（大盘摆）
  两种幅度下都会被正确识别；弧度制在盘摆跳变时可能被误判。同时写
  ``annotations/unit_joint_increment = "deg"`` 备他们改成显式读取。
- goal 4 个地面坐标 → ``annotations/goal``（精确值）与 ``goal``（同构镜像），
  z 取机器文件的 ``pick_z_m``/``dump_z_m``。他们现有载入器只会从 FK 伪推断
  目标；获得直读需要他们在 ``infer_real_episode`` 里加几行（对接文档任务项）。
- ``scoop`` 载料真值 → ``phase``，默认 **two 模式**（未载料 → 2、载料 → 4）：
  与部署侧适配器的相位估计规则**完全同源**（train/inference 一致性优先于
  标签丰富度）。``--phase-mode scoop`` 给 0..7 自然分段（仅当适配器由固定
  2/4 升级到同规则估计时才用）；``time`` 为他们的时间八等分兜底。
- 成功判据与采集端一致（装料后卸掉），不完整循环整条剔除并计入 ``skipped``。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from jiuwen_agx.agx_excavator.record import Episode
from jiuwen_agx.machine_map import MachineMap

__all__ = ["main", "convert_episode"]

_DEFAULT_SLOTS = (6, 7, 8, 9)  # their config.joint_state_indices (degrees channels)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", required=True, help="npz directory (data/raw/run_all)")
    parser.add_argument(
        "--machine",
        required=True,
        help="per-machine calibration JSON (configs/machine_agx365.template.json shows the schema)",
    )
    parser.add_argument(
        "--out", required=True, help="output directory for episode_*.h5"
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="nominal control frequency recorded in attrs (frames are beats by default)",
    )
    parser.add_argument(
        "--resample-hz",
        type=float,
        default=None,
        help="opt-in: resample the beat timeline onto a uniform grid at this rate "
        "(default: off — frames ARE beats, see module docstring)",
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--joint-slots",
        default=",".join(str(i) for i in _DEFAULT_SLOTS),
        help="state35 columns holding the joints (their config.joint_state_indices)",
    )
    parser.add_argument(
        "--phase-mode",
        choices=("two", "scoop", "time"),
        default="two",
        help="two: not-loaded -> 2, loaded -> 4 (matches the deployment adapter's "
        "estimator exactly); scoop: natural 0..7 segments; time: their 8-split fallback",
    )
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="dry-run only: convert with an UNCALIBRATED machine file (output is marked; do NOT train on it)",
    )
    return parser


def _resample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Column-wise linear resample of (T, D) samples at `times` onto `grid`."""
    out = np.empty((len(grid), values.shape[1]), dtype=np.float64)
    for column in range(values.shape[1]):
        out[:, column] = np.interp(grid, times, values[:, column])
    return out


def _hold_resample(
    times: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Step-hold resample (last sample at or before each grid point)."""
    index = np.searchsorted(times, grid, side="right") - 1
    index = np.clip(index, 0, len(values) - 1)
    return values[index]


def _phase_from_two(scoop: np.ndarray) -> np.ndarray:
    """Two-label phase that mirrors the deployment adapter's estimator exactly.

    The adapter can only see the measure truth at inference time, so it feeds
    phase 2 (approach/dig, active goal = pick) until the bucket loads and 4
    (swing, active goal = dump) after. Training labels must use the same rule
    or the phase embedding would be conditioned on a distribution the model
    never sees in deployment.
    """
    return np.where(scoop, 4, 2).astype(np.int64)


def _phase_from_scoop(scoop: np.ndarray) -> np.ndarray:
    """Their 8-phase labels from the measured loaded/unloaded truth."""
    phases = np.zeros(len(scoop), dtype=np.int64)
    loaded = np.flatnonzero(scoop)
    if len(loaded) == 0:
        return phases
    first, last = int(loaded[0]), int(loaded[-1])
    if first > 0:  # approach .. lift, quartered
        edges = np.linspace(0, first, 5, dtype=int)
        for phase_id in range(4):
            phases[edges[phase_id] : edges[phase_id + 1]] = phase_id
    span = last - first + 1
    split = first + (span + 1) // 2
    phases[first : min(split, last)] = 4  # swing
    phases[min(split, last) : last] = 5  # approach dump
    phases[last] = 6  # dump moment (last loaded sample)
    phases[last + 1 :] = 7  # retreat / trailing unloaded stretch
    return phases


def _phase_from_time(length: int) -> np.ndarray:
    return np.minimum(np.arange(length) * 8 // max(length, 1), 7).astype(np.int64)


def _action_chunks(
    diffs_rad: np.ndarray, length: int, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Their sliding-window future increments + pad mask (see their generation.py).

    ``diffs_rad[t]`` is the motion leaving frame ``t`` (there are ``length`` or
    ``length - 1`` of them, depending on whether the source series carries an
    extra trailing pose); frame ``t`` charges toward the future ``diffs[t:]``.
    """
    available = diffs_rad.shape[0]
    chunks = np.zeros((length, chunk_size, diffs_rad.shape[1]), dtype=np.float32)
    pad = np.ones((length, chunk_size), dtype=np.uint8)
    for t in range(length):
        count = min(chunk_size, available - t)
        if count > 0:
            chunks[t, :count] = np.rad2deg(
                diffs_rad[t : t + count]
            )  # DEGREES on purpose (see module docstring)
            pad[t, :count] = 0
    return chunks, pad


def convert_episode(
    episode: Episode,
    machine: MachineMap,
    *,
    slots: tuple[int, ...],
    chunk_size: int,
    phase_mode: str,
    resample_hz: float | None = None,
) -> dict[str, np.ndarray]:
    """Build the H5 payload for one episode (caller writes it).

    Default timeline is ``frames = beats``: the state series is the measured
    pose at each beat's start, the increments are the per-beat motions, and the
    phase comes straight off the per-beat scoop truth — the same quantities the
    deployment loop reproduces one beat at a time. ``resample_hz`` opts into a
    uniform grid (linear interpolation; scoop step-held).
    """
    if tuple(episode.joint_names) != machine.joint_order:
        raise ValueError(
            f"episode joint order {episode.joint_names} != machine order {machine.joint_order}"
        )
    # Poses along the native timeline: initial pose + the measurement after each beat.
    poses = np.vstack([episode.joints_before[:1], episode.joints_after])  # (T+1, 4)
    scoop_frames = np.asarray(episode.scoop, dtype=bool)  # (T,)

    if resample_hz is None:
        frames_native = poses[
            : len(scoop_frames)
        ]  # (T, 4): frame t = pose at beat t's start
        extra_tail = poses[
            len(scoop_frames) :
        ]  # keeps the last beat's motion in the diffs
        scoop_used = scoop_frames
    else:
        times = np.concatenate(
            [[0.0], np.cumsum(np.asarray(episode.beat_seconds, dtype=np.float64))]
        )
        grid = np.arange(0.0, times[-1] + 1e-9, 1.0 / float(resample_hz))
        frames_native = _resample(times, poses, grid)  # (N, 4)
        extra_tail = np.empty((0, poses.shape[1]))
        scoop_used = (
            _hold_resample(
                times, np.concatenate([[False], scoop_frames]).astype(np.float64), grid
            )
            > 0.5
        )

    joint_rad = np.empty_like(frames_native)
    for column, name in enumerate(machine.joint_order):
        jmap = machine.joints[name]
        joint_rad[:, column] = [
            jmap.to_policy(value) for value in frames_native[:, column]
        ]
    tail_rad = np.empty_like(extra_tail)
    for column, name in enumerate(machine.joint_order):
        jmap = machine.joints[name]
        tail_rad[:, column] = [jmap.to_policy(value) for value in extra_tail[:, column]]
    diffs = np.diff(np.vstack([joint_rad, tail_rad]), axis=0)  # (frames, 4)

    length = joint_rad.shape[0]
    state = np.zeros((length, 35), dtype=np.float32)
    state[:, list(slots)] = np.rad2deg(joint_rad).astype(np.float32)
    goal_row = np.asarray(
        [
            episode.goal["dig_x_m"],
            episode.goal["dig_y_m"],
            machine.pick_z_m,
            episode.goal["dump_x_m"],
            episode.goal["dump_y_m"],
            machine.dump_z_m,
        ],
        dtype=np.float32,
    )
    goal = np.tile(goal_row, (length, 1))
    if phase_mode == "two":
        phase = _phase_from_two(scoop_used)
    elif phase_mode == "scoop":
        phase = _phase_from_scoop(scoop_used)
    else:
        phase = _phase_from_time(length)
    chunks, pad = _action_chunks(diffs, length, chunk_size)
    return {
        "observations/state": state,
        "actions/joint_increment": chunks,
        "annotations/action_pad": pad,
        "annotations/valid": np.ones(length, dtype=bool),
        "annotations/goal": goal,
        "annotations/phase": phase,
        "joint_rad": joint_rad.astype(np.float32),
        "goal": goal,
        "phase": phase,
    }


def _write_h5(
    path: Path, payload: dict[str, np.ndarray], attrs: dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for name, array in payload.items():
            handle.create_dataset(name, data=array, compression="gzip")
        for key, value in attrs.items():
            handle.attrs[key] = value


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    machine = MachineMap.load(args.machine)
    if not machine.calibrated and not args.allow_uncalibrated:
        sys.exit(
            f"machine file {args.machine} is NOT calibrated: the m->angle mapping is a "
            "placeholder, and converting with it would silently poison the training "
            "data. Calibrate first (see configs/machine_agx365.template.json), or pass "
            "--allow-uncalibrated for a dry run whose output must not be trained on."
        )
    if not machine.calibrated:
        print(
            "[convert] WARNING: uncalibrated dry run — output marked calibrated=false, "
            "do NOT feed it to training",
            flush=True,
        )
    slots = tuple(int(part) for part in args.joint_slots.split(",") if part.strip())
    files = sorted(Path(args.raw).glob("*.npz"))
    if not files:
        sys.exit(f"--raw {args.raw} holds no .npz episodes")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("episode_*.h5")):
        sys.exit(
            f"{out_dir} already holds episode_*.h5 — refusing to overwrite, pick a fresh --out"
        )

    kept = 0
    skipped: Counter[str] = Counter()
    phases_seen: Counter[int] = Counter()
    travel_deg: list[np.ndarray] = []
    step_p999: list[np.ndarray] = []
    limits_min: np.ndarray | None = None
    limits_max: np.ndarray | None = None
    for path in files:
        episode = Episode.read_npz(path)
        if not (episode.scoop.any() and not episode.scoop[-1]):
            skipped["cycle incomplete"] += 1
            continue
        if not episode.goal:
            skipped["episode has no goal"] += 1
            continue
        try:
            payload = convert_episode(
                episode,
                machine,
                slots=slots,
                chunk_size=int(args.chunk_size),
                phase_mode=args.phase_mode,
                resample_hz=args.resample_hz,
            )
        except ValueError as exc:
            skipped[f"{exc}"] += 1
            continue
        name = f"episode_{kept:06d}.h5"
        attrs = {
            "source": "jiuwen_agx_agx365",
            "source_episode": path.name,
            "control_frequency": float(args.hz),
            "timeline": "beats"
            if args.resample_hz is None
            else f"uniform_{float(args.resample_hz):g}hz",
            "chunk_size": int(args.chunk_size),
            "history_length": 10,
            "machine": machine.machine,
            "machine_file": machine.source,
            "calibrated": int(machine.calibrated),
            "unit_joint_increment": "deg",
            "unit_state_joint_slots": "deg",
            "phase_mode": args.phase_mode,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        _write_h5(out_dir / name, payload, attrs)
        phases_seen.update(int(p) for p in np.unique(payload["phase"]))
        joint_deg = np.rad2deg(payload["joint_rad"]).astype(np.float64)
        increments = np.abs(payload["actions/joint_increment"]).astype(np.float64)
        travel_deg.append(np.abs(np.diff(joint_deg, axis=0)).sum(axis=0))
        step_p999.append(
            np.percentile(increments.reshape(-1, increments.shape[-1]), 99.9, axis=0)
        )
        low, high = joint_deg.min(axis=0), joint_deg.max(axis=0)
        limits_min = low if limits_min is None else np.minimum(limits_min, low)
        limits_max = high if limits_max is None else np.maximum(limits_max, high)
        kept += 1

    timeline = (
        "beats"
        if args.resample_hz is None
        else f"uniform_{float(args.resample_hz):g}hz"
    )
    print(
        f"[convert] dataset: {out_dir}  episodes: {kept}  (from {len(files)} npz)  "
        f"timeline={timeline} phase_mode={args.phase_mode}"
    )
    if skipped:
        print("[convert] skipped:")
        for reason, count in skipped.most_common():
            print(f"      {count:4d}  {reason}")
    print(f"[convert] phases seen: {dict(sorted(phases_seen.items()))}")
    if kept and limits_min is not None and limits_max is not None:
        # Their training config must adopt AGX beat-scale values or the closed
        # loop stalls: their inference clips every step to config
        # max_joint_step_deg (the current checkpoints declare ~2.5 deg), while
        # our beats legitimately travel up to ~20+ deg per joint.
        step_cap = np.ceil(np.median(np.stack(step_p999), axis=0) * 2.0) / 2.0
        suggested = {
            "machine": machine.machine,
            "from_dataset": str(out_dir),
            "timeline": timeline,
            "note": (
                "重训时请把 configs/<machine>.json 的这两组值设为本文件输出："
                "joint_limits_deg=数据实际角域（要余量自行外扩）；max_joint_step_deg="
                "每拍运动 p99.9 向上取整到 0.5°（步长设小：推理端每步 clip，偏小会把"
                "部署循环压慢数倍；偏大则失去保护）。"
            ),
            "joint_limits_deg": np.stack([limits_min, limits_max], axis=1)
            .round(1)
            .tolist(),
            "max_joint_step_deg": step_cap.round(1).tolist(),
            "cycle_travel_deg_p50": np.median(np.stack(travel_deg), axis=0)
            .round(0)
            .tolist(),
        }
        suggested_path = out_dir / "suggested_config.json"
        suggested_path.write_text(
            json.dumps(suggested, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[convert] suggested AGX365 config -> {suggested_path}")
        print(f"[convert]   joint_limits_deg     = {suggested['joint_limits_deg']}")
        print(f"[convert]   max_joint_step_deg   = {suggested['max_joint_step_deg']}")
        print(
            f"[convert]   cycle travel p50 (deg/joint) = {suggested['cycle_travel_deg_p50']}"
        )
        print(
            "[convert]   现 ckpt 的 max_joint_step_deg 是他们的 IK 步长（~2.5°/步），"
            "按它部署会把循环压到 1/8 步速；重训必须采用上面的建议值"
        )
    if not machine.calibrated:
        print(
            "[convert] NOTE: uncalibrated dry run — regenerate after calibration before training"
        )


if __name__ == "__main__":
    main()
