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
- 到位节拍 → **固定 ``--hz`` 等间隔**时间轴（逐关节线性重采样；他们的
  加速度/jerk 损失与时钟检查都要求等间隔）。
- 相邻帧差分 → ``actions/joint_increment`` (T,16,4)。**写成度**：他们载入器
  用"99 分位 >0.5 即视为度"的启发式自动识别，度制在正常与异常（大盘摆）两种
  幅度下都会被正确识别；弧度制在盘摆跳变时可能被误判。同时写
  ``annotations/unit_joint_increment = "deg"`` 备他们改成显式读取。
- goal 4 个地面坐标 → ``annotations/goal``（精确值）与 ``goal``（同构镜像），
  z 取机器文件的 ``pick_z_m``/``dump_z_m``。他们现有载入器只会从 FK 伪推断
  目标；获得直读需要他们在 ``infer_real_episode`` 里加几行（对接文档任务项）。
- ``scoop`` 载料真值 → ``phase``：未载料段四等分为 0..3，载料段分 4/5、
  末帧 6，卸载后 7（``--phase-mode time`` 退回时间八等分）。
- 成功判据与采集端一致（装料后卸掉），不完整循环整条剔除并计入 ``skipped``。
"""

from __future__ import annotations

import argparse
import datetime as dt
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
        help="their control frequency (uniform timeline)",
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--joint-slots",
        default=",".join(str(i) for i in _DEFAULT_SLOTS),
        help="state35 columns holding the joints (their config.joint_state_indices)",
    )
    parser.add_argument(
        "--phase-mode",
        choices=("scoop", "time"),
        default="scoop",
        help="scoop: phase from the measured carry truth; time: their 8-split fallback",
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
    joint_rad: np.ndarray, chunk_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Their sliding-window future increments + pad mask (see their generation.py)."""
    length = joint_rad.shape[0]
    diffs = np.diff(joint_rad, axis=0)  # (T-1, 4) radians
    chunks = np.zeros((length, chunk_size, joint_rad.shape[1]), dtype=np.float32)
    pad = np.ones((length, chunk_size), dtype=np.uint8)
    for t in range(length):
        count = min(chunk_size, length - 1 - t)
        if count > 0:
            chunks[t, :count] = np.rad2deg(
                diffs[t : t + count]
            )  # DEGREES on purpose (see module docstring)
            pad[t, :count] = 0
    return chunks, pad


def convert_episode(
    episode: Episode,
    machine: MachineMap,
    *,
    slots: tuple[int, ...],
    hz: float,
    chunk_size: int,
    phase_mode: str,
) -> dict[str, np.ndarray]:
    """Build the H5 payload for one episode (caller writes it)."""
    if tuple(episode.joint_names) != machine.joint_order:
        raise ValueError(
            f"episode joint order {episode.joint_names} != machine order {machine.joint_order}"
        )
    # Native timeline: state after each beat, plus the initial pose at t=0.
    times = np.concatenate(
        [[0.0], np.cumsum(np.asarray(episode.beat_seconds, dtype=np.float64))]
    )
    native = np.vstack([episode.joints_before[:1], episode.joints_after])  # (T+1, 4)
    scoop = np.concatenate([[False], np.asarray(episode.scoop, dtype=bool)])
    grid = np.arange(0.0, times[-1] + 1e-9, 1.0 / hz)

    native_u = _resample(times, native, grid)  # (N, 4) native units
    scoop_u = _hold_resample(times, scoop.astype(np.float64), grid) > 0.5

    joint_rad = np.empty_like(native_u)
    for column, name in enumerate(machine.joint_order):
        jmap = machine.joints[name]
        joint_rad[:, column] = [jmap.to_policy(value) for value in native_u[:, column]]

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
    phase = (
        _phase_from_scoop(scoop_u)
        if phase_mode == "scoop"
        else _phase_from_time(length)
    )
    chunks, pad = _action_chunks(joint_rad, chunk_size)
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
                hz=float(args.hz),
                chunk_size=int(args.chunk_size),
                phase_mode=args.phase_mode,
            )
        except ValueError as exc:
            skipped[f"{exc}"] += 1
            continue
        name = f"episode_{kept:06d}.h5"
        attrs = {
            "source": "jiuwen_agx_agx365",
            "source_episode": path.name,
            "control_frequency": float(args.hz),
            "chunk_size": int(args.chunk_size),
            "history_length": 10,
            "machine": machine.machine,
            "machine_file": machine.source,
            "calibrated": int(machine.calibrated),
            "unit_joint_increment": "deg",
            "unit_state_joint_slots": "deg",
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        _write_h5(out_dir / name, payload, attrs)
        phases_seen.update(int(p) for p in np.unique(payload["phase"]))
        kept += 1

    print(f"[convert] dataset: {out_dir}  episodes: {kept}  (from {len(files)} npz)")
    if skipped:
        print("[convert] skipped:")
        for reason, count in skipped.most_common():
            print(f"      {count:4d}  {reason}")
    print(f"[convert] phases seen: {dict(sorted(phases_seen.items()))}")
    if not machine.calibrated:
        print(
            "[convert] NOTE: uncalibrated dry run — regenerate after calibration before training"
        )


if __name__ == "__main__":
    main()
