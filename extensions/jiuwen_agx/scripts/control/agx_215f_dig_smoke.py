# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""215F 一斗验收 —— 和 365 一样走 ``dig``，读桥接真值。

前提：先起 215F 桥接（``start_agx_bridge_215f.bat`` 或
``agx_bridge_server.py --machine 215f``），再运行::

    python scripts/control/agx_215f_dig_smoke.py \
        --config extensions/jiuwen_agx/configs/agx_excavator/agx_excavator.215f.yaml

默认挖点 (5.0, 2.0)（土堆中心）、卸点 (3.0, 0.0)。脚本打印：
inventory 身份/关节/土堆/空斗质量 → dig 结果 → 卸料后残留质量。
"""

from __future__ import annotations

import argparse
import sys
import time

from jiuwen_agx.agx_excavator import build_agx_excavator_session
from jiuwen_agx.agx_excavator.config import AgxExcavatorConfig


def _inventory(session: object) -> dict:
    """Read the bridge inventory (diagnostic only; older bridges may not answer)."""
    backend = getattr(getattr(session.env, "driver", None), "backend", None)
    call = getattr(backend, "_call", None)
    if call is None:
        return {}
    try:
        return call({"cmd": "inventory"})
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"[215f-smoke] inventory unavailable: {exc}", flush=True)
        return {}


def _print_inventory(inventory: dict) -> None:
    for machine in inventory.get("machines") or []:
        print(
            f"[215f-smoke] machine={machine.get('name')} "
            f"bucket_mass_kg={machine.get('bucket_mass_kg')} "
            f"loaded_threshold_kg={machine.get('loaded_threshold_kg')} "
            f"terrain={machine.get('terrain')}",
            flush=True,
        )
        for joint in machine.get("joints") or []:
            print(
                f"[215f-smoke]   joint {joint.get('name'):<7} "
                f"{joint.get('type'):<10} unit={joint.get('unit')} "
                f"range={joint.get('range')} angle={joint.get('angle')}",
                flush=True,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="215F dig smoke test")
    parser.add_argument(
        "--config",
        default="extensions/jiuwen_agx/configs/agx_excavator/agx_excavator.215f.yaml",
    )
    parser.add_argument("--dig-x", type=float, default=5.0)
    parser.add_argument("--dig-y", type=float, default=2.0)
    parser.add_argument(
        "--dig-z",
        type=float,
        default=None,
        help="bucket-tip dig height; default = terrain surface at the dig point",
    )
    parser.add_argument("--dump-x", type=float, default=5.0)
    parser.add_argument("--dump-y", type=float, default=-1.0)
    parser.add_argument(
        "--dump-z",
        type=float,
        default=3.5,
        help="bucket-tip dump height; must be reachable while carrying the closed bucket",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="permit the mock backend (mock truth never loads the bucket)",
    )
    args = parser.parse_args(argv)

    cfg = AgxExcavatorConfig.from_yaml(args.config)
    if cfg.backend == "mock" and not args.allow_mock:
        sys.exit(
            "refusing to run on the mock backend: its scoop truth never loads. "
            "Point --config at the 215F remote config, or pass --allow-mock."
        )

    session = build_agx_excavator_session.from_yaml(args.config)
    with session:
        _print_inventory(_inventory(session))
        print(
            f"[215f-smoke] dig=({args.dig_x}, {args.dig_y}, {args.dig_z}) "
            f"dump=({args.dump_x}, {args.dump_y}, {args.dump_z})",
            flush=True,
        )
        result = session.api.dig(
            dig_x_m=args.dig_x,
            dig_y_m=args.dig_y,
            dump_x_m=args.dump_x,
            dump_y_m=args.dump_y,
            dig_z_m=args.dig_z,
            dump_z_m=args.dump_z,
        )
        print(f"[215f-smoke] dig result: {result}", flush=True)
        for _ in range(10):
            if not session.env.driver.scoop_state():
                break
            time.sleep(0.5)
        print(
            f"[215f-smoke] after dump: scoop_state={session.env.driver.scoop_state()}",
            flush=True,
        )
        _print_inventory(_inventory(session))


if __name__ == "__main__":
    main()
