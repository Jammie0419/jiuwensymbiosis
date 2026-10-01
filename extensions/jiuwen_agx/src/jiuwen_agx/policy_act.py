# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""ActPolicy — the lerobot ACT checkpoint translated onto the Policy seam.

Registered as ``"act"`` (YAML ``policy: {name: act, ckpt: ..., device: ...}``).
This module imports torch/lerobot ONLY inside :meth:`ActPolicy.__init__`, so
importing the package (and seeing the registry entry) never requires the
learning stack; a missing stack surfaces as a readable act_exec failure instead
of an import error.

Checkpoint contract (see docs/act-integration-plan.md §3.6): a lerobot
``pretrained_model`` directory holding config.json + model.safetensors + the
saved pre/post processors. The observation mapping is fixed here:

- ``obs["joints"]`` → ``observation.state``            (order = joint_names)
- ``obs["goal"]``   → ``observation.environment_state`` (ACT refuses state-only
  input; the goal conditions the trajectory — this is what makes it generalise
  over dig/dump points at all)

and the prediction is ONE ``predict_action_chunk`` call (not the per-step
``select_action`` queue — our beats are arrival-paced, see plan §4.3), wrapped
in the checkpoint's own pre/post processors so normalization statistics travel
with the weights. Mirrors the official rollout loop
(``lerobot/rollout/inference/sync.py``).
"""

from __future__ import annotations

from typing import Any

from jiuwen_agx.policy import register_policy

__all__ = ["ActPolicy"]

# The goal vector layout — fixed here AND in the dataset recorder/converters;
# changing one side without the other silently scrambles conditioning.
GOAL_ORDER: tuple[str, ...] = ("dig_x_m", "dig_y_m", "dump_x_m", "dump_y_m")


@register_policy("act")
class ActPolicy:
    """lerobot ACT checkpoint behind the Policy seam (torch loaded lazily)."""

    def __init__(
        self,
        policy_cfg: dict[str, Any],
        *,
        joint_names: tuple[str, ...] = (),
        joint_limits: dict[str, tuple[float, float]] | None = None,
        home_joints: dict[str, float] | None = None,
    ) -> None:
        del (
            joint_limits,
            home_joints,
        )  # the checkpoint self-describes; body truth unused
        import torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        ckpt = policy_cfg.get("ckpt")
        if not ckpt:
            raise ValueError(
                "policy 'act' needs 'ckpt' in the YAML policy section "
                "(a lerobot pretrained_model directory from training)"
            )
        self._device = str(policy_cfg.get("device", "cpu"))
        self._torch = torch
        # Component order fixed by the dataset features (see GOAL_ORDER note);
        # joint_names IS the observation.state order recorded at training time.
        self._joint_names = tuple(joint_names)
        self._act_cfg = ACTConfig.from_pretrained(ckpt)
        self._policy = ACTPolicy.from_pretrained(ckpt, config=self._act_cfg)
        self._pre, self._post = make_pre_post_processors(
            policy_cfg=self._act_cfg, pretrained_path=ckpt
        )

    def reset(self) -> None:
        """Clear the lerobot-side episode state (action queue / ensembler)."""
        self._policy.reset()

    def predict(self, obs: dict[str, Any]) -> list[dict[str, float]]:
        """One ACT action chunk as absolute per-joint targets (native units)."""
        torch = self._torch
        joints = obs.get("joints") or {}
        goal = obs.get("goal") or {}
        state = [float(joints[name]) for name in self._joint_names]
        env_state = [float(goal[name]) for name in GOAL_ORDER]
        batch = {
            "observation.state": torch.tensor(
                state, dtype=torch.float32, device=self._device
            ).unsqueeze(0),
            "observation.environment_state": torch.tensor(
                env_state, dtype=torch.float32, device=self._device
            ).unsqueeze(0),
        }
        with torch.inference_mode():
            chunk = self._policy.predict_action_chunk(self._pre(batch))
            chunk = self._post(chunk)  # unnormalize → native units, moved to cpu
        rows = chunk.squeeze(0).detach().cpu().tolist()
        return [dict(zip(self._joint_names, row, strict=True)) for row in rows]
