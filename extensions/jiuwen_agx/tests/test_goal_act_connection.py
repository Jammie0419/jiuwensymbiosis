# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""goal-act-v2 连接验证：小脑组模型 → 适配器 → act_exec 全链路，无需 AGX/训练。

镜像 test_act_connection.py 的套路：模型权重产出无意义的目标不重要——被测的
是 CONNECTION（加载、单位换算、单目标/拍、驱动护栏、兜底终止）。

依赖小脑组工件（checkpoint + act_trajectory 源码），没有就整模块 skip：
默认探测本机已知路径，可用环境变量覆盖——``JIUWEN_GOALACT_SRC``（含
``act_trajectory`` 包的目录）、``JIUWEN_GOALACT_CKPT``（goal-act-v2 .pt）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="goal-act-v2 connection tests need torch installed")

_DEFAULT_SRC = Path(r"E:/AGX-2.42.2.1/act_trajectory/act_trajectory/src")
_DEFAULT_CKPT = Path(
    r"E:/AGX-2.42.2.1/act_trajectory/act_trajectory/outputs/geometry_smoke_20261010/swe600f_model.pt"
)

JOINTS = ("swing", "boom", "arm", "bucket")
GOAL = {"dig_x_m": 1.5, "dig_y_m": 2.5, "dump_x_m": -1.0, "dump_y_m": 1.2}


def _their_src() -> Path:
    return Path(os.environ.get("JIUWEN_GOALACT_SRC", _DEFAULT_SRC))


def _their_ckpt() -> Path:
    return Path(os.environ.get("JIUWEN_GOALACT_CKPT", _DEFAULT_CKPT))


pytestmark = pytest.mark.skipif(
    not (
        (_their_src() / "act_trajectory" / "inference.py").exists()
        and _their_ckpt().exists()
    ),
    reason="cerebellum (goal-act-v2) artifacts not present on this machine",
)


@pytest.fixture()
def machine_file(tmp_path: Path) -> Path:
    """Uncalibrated placeholder machine file — chain checks only (see adapter docstring)."""
    payload = {
        "machine": "unit-test",
        "calibrated": False,
        "policy_joint_order": list(JOINTS),
        "pick_z_m": 0.0,
        "dump_z_m": 0.0,
        "joints": [
            {"name": "swing", "native_unit": "rad", "sign": 1, "offset_rad": 0.0},
            *[
                {
                    "name": name,
                    "native_unit": "m",
                    "sign": 1,
                    "offset_rad": 0.0,
                    "hydraulic_conversion": {
                        "min_displacement_m": 0.5,
                        "max_displacement_m": 1.5,
                        "min_angle_deg": -35.0 if name == "boom" else -120.0,
                        "max_angle_deg": 75.0 if name == "boom" else 60.0,
                        "poly_coeffs": None,
                    },
                }
                for name in ("boom", "arm", "bucket")
            ],
        ],
    }
    path = tmp_path / "machine.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_adapter_predict_returns_single_finite_target(machine_file: Path) -> None:
    from jiuwen_agx.policy_goal_act import GoalActV2Policy

    policy = GoalActV2Policy(
        {
            "ckpt": str(_their_ckpt()),
            "src": str(_their_src()),
            "machine": str(machine_file),
        },
        joint_names=JOINTS,
    )
    policy.reset()
    chunk = policy.predict(
        {"joints": dict.fromkeys(JOINTS, 0.0), "goal": GOAL, "scoop": False}
    )
    # one target per call: our beat loop then re-observes every beat, which is
    # exactly the model's own per-cycle replan loop.
    assert len(chunk) == 1
    assert set(chunk[0]) == set(JOINTS)
    assert all(v == v for v in chunk[0].values())  # finite (no NaN)


def test_act_exec_drives_goal_act_on_mock(machine_file: Path) -> None:
    from jiuwen_agx.agx_excavator import build_agx_excavator_session

    session = build_agx_excavator_session.from_dict(
        {
            "env": {
                "cfg": {
                    "low_level": {
                        "policy": {
                            "name": "goal_act_v2",
                            "ckpt": str(_their_ckpt()),
                            "src": str(_their_src()),
                            "machine": str(machine_file),
                        },
                        "policy_max_beats": 8,
                    }
                }
            }
        }
    )
    with session:
        result = session.api.act_exec(**GOAL)
    # Mock scoop truth never loads, so the backstop MUST terminate the cycle —
    # and the message proves 8 policy-generated targets reached the driver.
    assert result["ok"] is False
    assert "incomplete after 8 beats" in result["error"]
    assert "never loaded" in result["error"]
