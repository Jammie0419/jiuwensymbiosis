# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""M2 data-pipeline tests: proxy sub-stepping, goal sampling, Episode
round-trip, converter gates, keyboard segmentation. All lerobot-free —
the collection logic must be testable on the AGX machine (design doc §1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jiuwen_agx.agx_excavator.record import (
    FORMAT_VERSION,
    GOAL_KEYS,
    Episode,
    RecordingDriverProxy,
    admission_reason,
    sample_goal,
    segment_keyboard,
    unit_conflict,
)

JOINTS = ("swing", "boom", "arm", "bucket")
GOAL = {"dig_x_m": -2.0, "dig_y_m": 1.0, "dump_x_m": 3.0, "dump_y_m": 0.5}


class _FakeDriver:
    """Records every send with its timeout; joints jump straight to targets."""

    def __init__(self, scoop_script: list[bool] | None = None) -> None:
        self.joint_names = JOINTS
        self._positions = dict.fromkeys(JOINTS, 0.0)
        self._scoop_script = list(scoop_script or [])
        self._scoop = False
        self.sent: list[tuple[dict[str, float], float | None]] = []

    def get_joint_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def move_joints_blocking(
        self, targets: dict[str, float], *, timeout_s: float | None = None
    ) -> dict[str, float]:
        self.sent.append((dict(targets), timeout_s))
        self._positions.update(targets)
        return dict(self._positions)

    def scoop_state(self) -> bool:
        if self._scoop_script:
            return self._scoop_script.pop(0)
        return self._scoop

    def mark_scoop(self, loaded: bool) -> None:
        self._scoop = bool(loaded)

    def home(self) -> None:
        self._positions = dict.fromkeys(JOINTS, 0.0)


# ============================================================================ proxy
class TestRecordingDriverProxy:
    def test_transition_becomes_sub_beats_plus_exact_final(self):
        driver = _FakeDriver()
        proxy = RecordingDriverProxy(driver, sub_beats=4, beat_timeout_s=0.5)
        proxy.move_joints_blocking({"swing": 8.0})
        assert len(proxy.beats) == 4
        # intermediate beats carry the beat timeout; the final keeps the caller's
        assert [t for _, t in driver.sent[:3]] == [0.5, 0.5, 0.5]
        assert driver.sent[-1][1] is None
        # final beat targets the exact keyframe value
        assert driver.sent[-1][0]["swing"] == pytest.approx(8.0)
        # interpolation is monotone towards the target
        swings = [s["swing"] for s, _ in driver.sent]
        assert swings == sorted(swings)
        assert swings[0] < 8.0

    def test_beats_record_measured_before_and_after(self):
        driver = _FakeDriver()
        proxy = RecordingDriverProxy(driver, sub_beats=2, beat_timeout_s=0.5)
        proxy.move_joints_blocking({"swing": 4.0})
        (before1, after1, scoop1, seconds1) = proxy.beats[0]
        (before2, after2, scoop2, seconds2) = proxy.beats[1]
        assert before1 == dict.fromkeys(JOINTS, 0.0)
        assert after1["swing"] == pytest.approx(2.0)
        assert before2["swing"] == pytest.approx(
            2.0
        )  # chain: before of beat n = after of n-1
        assert after2["swing"] == pytest.approx(4.0)
        assert scoop1 is scoop2 is False
        assert seconds1 >= 0.0 and seconds2 >= 0.0
        assert proxy.get_joint_positions()["swing"] == pytest.approx(4.0)

    def test_reads_and_scoop_writes_are_forwarded(self):
        driver = _FakeDriver()
        driver.mark_scoop(True)
        proxy = RecordingDriverProxy(driver, sub_beats=2, beat_timeout_s=0.5)
        assert proxy.scoop_state() is True
        proxy.mark_scoop(False)
        assert driver.scoop_state() is False

    def test_home_refreshes_the_measured_state(self):
        driver = _FakeDriver()
        proxy = RecordingDriverProxy(driver, sub_beats=2, beat_timeout_s=0.5)
        proxy.move_joints_blocking({"swing": 8.0})
        proxy.home()
        assert proxy.get_joint_positions() == dict.fromkeys(JOINTS, 0.0)

    def test_clear_beats_starts_a_fresh_episode(self):
        proxy = RecordingDriverProxy(_FakeDriver(), sub_beats=2, beat_timeout_s=0.5)
        proxy.move_joints_blocking({"swing": 1.0})
        assert len(proxy.beats) == 2
        proxy.clear_beats()
        assert proxy.beats == []


# ============================================================================ goal sampler
class TestSampleGoal:
    def test_deterministic_under_a_seed(self):
        a = sample_goal(np.random.default_rng(7), reach_min_m=1.0, reach_max_m=6.0)
        b = sample_goal(np.random.default_rng(7), reach_min_m=1.0, reach_max_m=6.0)
        assert a == b

    def test_hundred_samples_respect_annulus_and_dump_distance(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            goal = sample_goal(rng, reach_min_m=1.0, reach_max_m=6.0)
            assert set(goal) == set(GOAL_KEYS)
            dig = math.hypot(goal["dig_x_m"], goal["dig_y_m"])
            dump = math.hypot(goal["dump_x_m"], goal["dump_y_m"])
            assert 1.0 <= dig <= 6.0
            assert 1.0 <= dump <= 6.0
            dist = math.hypot(
                goal["dump_x_m"] - goal["dig_x_m"], goal["dump_y_m"] - goal["dig_y_m"]
            )
            assert 2.0 <= dist <= 4.0


# ============================================================================ episode
def _episode(**overrides) -> Episode:
    n = 10
    fields = {
        "source": "teacher",
        "backend": "remote",
        "goal": GOAL,
        "joint_names": JOINTS,
        "joint_units": None,
        "swing_unit": "rad",
        "beat_timeout_s": 1.0,
        "fps_nominal": 10,
        "joints_before": np.zeros((n, 4)),
        "joints_after": np.ones((n, 4)),
        "scoop": np.array(
            [False, False, True, True, True, False, False, False, False, False]
        ),
        "beat_seconds": np.full(n, 0.5, dtype=np.float32),
    }
    fields.update(overrides)
    return Episode(**fields)


class TestEpisode:
    def test_success_requires_loaded_then_unloaded(self):
        assert _episode().successful() is True
        assert (
            _episode(scoop=np.zeros(6, dtype=bool)).successful() is False
        )  # never loaded
        assert (
            _episode(scoop=np.array([False, True, True, True, True, True])).successful()
            is False
        )  # never dumped
        assert _episode(scoop=np.array([True] * 6)).successful() is False

    def test_frames_are_float32_pairs(self):
        states, actions = _episode().frames()
        assert states.dtype == np.float32 and actions.dtype == np.float32
        assert states.shape == actions.shape == (10, 4)

    def test_final_scoop_overrides_the_last_beat_reading(self):
        driver = _FakeDriver(scoop_script=[True, True])  # both beats read loaded
        proxy = RecordingDriverProxy(driver, sub_beats=2, beat_timeout_s=0.5)
        proxy.move_joints_blocking({"swing": 4.0})
        # mock-style truth: the flag clears only AFTER the last motion beat
        driver.mark_scoop(False)
        without = Episode.from_proxy(
            proxy,
            source="teacher",
            backend="mock",
            goal=GOAL,
            joint_units=None,
            swing_unit="rad",
            beat_timeout_s=0.5,
            fps_nominal=10,
        )
        assert bool(without.scoop[-1]) is True and without.successful() is False
        corrected = Episode.from_proxy(
            proxy,
            source="teacher",
            backend="mock",
            goal=GOAL,
            joint_units=None,
            swing_unit="rad",
            beat_timeout_s=0.5,
            fps_nominal=10,
            final_scoop=driver.scoop_state(),
        )
        assert bool(corrected.scoop[-1]) is False and corrected.successful() is True

    def test_npz_round_trip_preserves_arrays_and_metadata(self, tmp_path):
        path = _episode(sub_beats=20, seed=7).write_npz(tmp_path / "ep.npz")
        loaded = Episode.read_npz(path)
        assert loaded.source == "teacher"
        assert loaded.backend == "remote"
        assert loaded.goal == GOAL
        assert loaded.joint_names == JOINTS
        assert loaded.sub_beats == 20 and loaded.seed == 7
        np.testing.assert_array_equal(loaded.joints_before, _episode().joints_before)
        np.testing.assert_array_equal(loaded.scoop, _episode().scoop)

    def test_version_gate_rejects_foreign_files(self, tmp_path):
        import json

        path = _episode().write_npz(tmp_path / "ep.npz")
        with np.load(path, allow_pickle=False) as data:
            blob = {k: data[k] for k in data.files if k != "meta"}
            meta = json.loads(str(data["meta"]))
        meta["format_version"] = FORMAT_VERSION + 99
        blob["meta"] = json.dumps(meta)
        np.savez_compressed(tmp_path / "foreign.npz", **blob)
        with pytest.raises(ValueError, match="format_version"):
            Episode.read_npz(tmp_path / "foreign.npz")


# ============================================================================ converter gates
class TestAdmission:
    def test_good_episode_passes(self):
        assert admission_reason(_episode()) is None

    def test_mock_is_refused_unless_explicitly_allowed(self):
        assert "mock" in admission_reason(_episode(backend="mock"))
        assert admission_reason(_episode(backend="mock"), allow_mock=True) is None

    def test_goalless_is_refused(self):
        assert "no goal" in admission_reason(_episode(goal=None))

    def test_incomplete_cycles_are_refused(self):
        never_loaded = _episode(scoop=np.zeros(6, dtype=bool))
        never_dumped = _episode(scoop=np.array([False] + [True] * 9))
        assert "never loaded" in admission_reason(never_loaded)
        assert "never dumped" in admission_reason(never_dumped)

    def test_short_episodes_are_refused(self):
        assert "too short" in admission_reason(
            _episode(
                joints_before=np.zeros((5, 4)),
                joints_after=np.ones((5, 4)),
                scoop=np.array([False, False, True, True, False]),
                beat_seconds=np.full(5, 0.5, dtype=np.float32),
            )
        )


# ============================================================================ dataset-level unit gate
class TestUnitConflict:
    def test_same_units_pass(self):
        assert unit_conflict(_episode(), _episode()) is None

    def test_mixed_swing_units_are_named(self):
        # vary ONE fingerprint field so the reported conflict is unambiguous
        conflict = unit_conflict(_episode(), _episode(swing_unit="deg"))
        assert conflict is not None and "swing_unit" in conflict

    def test_mixed_joint_units_are_named(self):
        conflict = unit_conflict(_episode(), _episode(joint_units="deg"))
        assert conflict is not None and "joint_units" in conflict

    def test_joint_name_mismatch_is_named(self):
        conflict = unit_conflict(
            _episode(), _episode(joint_names=("j1", "j2", "j3", "j4"))
        )
        assert conflict is not None and "joint_names" in conflict


# ============================================================================ keyboard segmentation
class TestSegmentKeyboard:
    def test_stream_is_cut_at_loaded_then_unloaded_boundaries(self):
        stream = _episode(
            source="keyboard",
            goal=None,
            scoop=np.array(
                [False, False, True, True, False, False, False, True, False, False],
                dtype=bool,
            ),
        )
        segments = segment_keyboard(stream)
        assert [s.n_beats for s in segments] == [
            5,
            4,
        ]  # idx 0-4 (dump tick 4), idx 5-8 (dump tick 8)
        assert all(s.goal is None for s in segments)

    def test_trailing_loaded_stretch_is_kept_as_a_failing_candidate(self):
        stream = _episode(
            source="keyboard",
            goal=None,
            scoop=np.array([False, True, True, True], dtype=bool),
        )
        segments = segment_keyboard(stream)
        assert len(segments) == 1 and segments[0].successful() is False

    def test_teacher_episodes_are_rejected(self):
        with pytest.raises(ValueError, match="keyboard"):
            segment_keyboard(_episode())
