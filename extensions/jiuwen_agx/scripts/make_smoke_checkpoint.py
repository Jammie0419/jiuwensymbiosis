# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""make_smoke_checkpoint — prove the ACT connection chain WITHOUT training.

Builds a randomly-initialised ACT checkpoint in the exact lerobot
``pretrained_model`` layout our deploy side loads (config.json + weights +
pre/post processors), then re-loads it through the same path ActPolicy uses.
The weights are random: outputs are meaningless motion targets — the point is
that the WHOLE chain executes (checkpoint format → from_pretrained → predict
→ act_exec → driver guardrails). NOT for real task use; swap in a trained
checkpoint afterwards (docs/act-runbook.md 阶段五/八).

Requires lerobot (pip install "lerobot[training]==0.6.1", or the CPU-only
torch variant first — see act-runbook.md 阶段三).

    python scripts/make_smoke_checkpoint.py --out data/ckpt/smoke --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--out", default="data/ckpt/smoke", help="checkpoint output dir"
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        import torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.act.processor_act import make_act_pre_post_processors
    except ImportError as exc:
        sys.exit(
            f"lerobot/torch not installed ({exc}) — install first, e.g. CPU-only:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            '  pip install "lerobot[training]==0.6.1"\n'
            "(see docs/act-runbook.md 阶段三)"
        )

    from lerobot.configs.types import FeatureType, PolicyFeature

    torch.manual_seed(args.seed)
    # Feature contract = the dataset spec (act-data-collection-design.md §5):
    # 4 joints in, 4 goal dims conditioning, 4-dim absolute joint targets out.
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (4,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (4,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
        chunk_size=100,
        n_action_steps=100,
        device="cpu",
    )
    policy = ACTPolicy(config)  # random init — smoke only, not a trained model
    pre, post = make_act_pre_post_processors(config, dataset_stats=None)

    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite an existing checkpoint dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    pre.save_pretrained(out)
    post.save_pretrained(out)

    # Re-load through the SAME path ActPolicy uses at deployment (plan §4.3).
    reloaded_cfg = ACTConfig.from_pretrained(out)
    reloaded = ACTPolicy.from_pretrained(out, config=reloaded_cfg)
    reloaded_pre, reloaded_post = make_pre_post_processors(
        policy_cfg=reloaded_cfg, pretrained_path=out
    )
    batch = {
        "observation.state": torch.zeros(1, 4),
        "observation.environment_state": torch.zeros(1, 4),
    }
    with torch.inference_mode():
        chunk = reloaded.predict_action_chunk(reloaded_pre(batch))
        chunk = reloaded_post(chunk)
    print(
        f"[smoke] checkpoint written & reload-verified: {out}\n"
        f"[smoke] predict_action_chunk -> shape {tuple(chunk.shape)}\n"
        "[smoke] next: local.yaml policy {name: act, ckpt: <this dir>} → mock 冒烟\n"
        "[smoke] (random weights = meaningless targets; only the CHAIN is under test)"
    )


if __name__ == "__main__":
    main()
