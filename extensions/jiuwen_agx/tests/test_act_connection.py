# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ACT connection verification: the whole chain, no training required.

Builds a random-init checkpoint via make_smoke_checkpoint.py (the lerobot
pretrained_model layout), then drives it through OUR deployment path:
ActPolicy adapter → predict → act_exec → move_joints_blocking guardrails →
termination. Random weights produce meaningless targets — what is under test
is the CONNECTION, not the policy quality (docs/act-runbook.md 阶段八 mock 冒烟).

Skips automatically on machines without torch/lerobot (pytest.importorskip).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="ACT connection tests need torch+lerobot installed")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_smoke_checkpoint.py"
JOINTS = ("swing", "boom", "arm", "bucket")
GOAL = {"dig_x_m": -2.0, "dig_y_m": 1.0, "dump_x_m": 3.0, "dump_y_m": 0.5}


def _build_checkpoint(out: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "make_smoke_checkpoint_under_test", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(["--out", str(out), "--seed", "7"])


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("act_smoke_ckpt") / "pretrained_model"
    _build_checkpoint(out)
    return out


def test_checkpoint_layout_and_adapter_predict(ckpt_dir: Path) -> None:
    # the pretrained_model layout lerobot's own loading path expects
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        assert (ckpt_dir / name).exists(), f"missing {name}"
    from jiuwen_agx.policy_act import ActPolicy

    policy = ActPolicy({"ckpt": str(ckpt_dir), "device": "cpu"}, joint_names=JOINTS)
    policy.reset()
    chunk = policy.predict({"joints": dict.fromkeys(JOINTS, 0.0), "goal": GOAL})
    assert len(chunk) == 100  # one full action chunk
    for beat in chunk:
        assert set(beat) == set(JOINTS)
        assert all(v == v for v in beat.values())  # finite (no NaN)


def test_act_exec_drives_act_policy_end_to_end_on_mock(ckpt_dir: Path) -> None:
    from jiuwen_agx.agx_excavator import build_agx_excavator_session

    session = build_agx_excavator_session.from_dict(
        {
            "env": {
                "cfg": {
                    "low_level": {
                        "policy": {"name": "act", "ckpt": str(ckpt_dir)},
                        "policy_max_beats": 30,
                    }
                }
            }
        }
    )
    with session:
        result = session.api.act_exec(**GOAL)
    # Mock scoop truth never loads, so the backstop MUST terminate the cycle —
    # and the message proves 30 policy-generated targets reached the driver.
    assert result["ok"] is False
    assert "incomplete after 30 beats" in result["error"]
    assert "never loaded" in result["error"]
