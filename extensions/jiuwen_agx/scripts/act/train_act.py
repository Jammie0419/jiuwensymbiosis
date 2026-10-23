# coding: utf-8
"""lerobot-train wrapper that survives WinError 1314 (runbook 阶段三, path B).

Every checkpoint save updates ``checkpoints/last`` with a symlink
(``lerobot.common.train_utils.update_last_checkpoint``). Without the
SeCreateSymbolicLink privilege — Windows without Developer Mode — that raises
``[WinError 1314]`` and kills the run at the first ``--save_freq`` step.

This wrapper swaps the function for one that keeps the save itself and only
skips the convenience link: numbered checkpoints land normally, and resume
still works through an explicit ``--resume --config_path <checkpoint>``
(which resolves a path, never the ``last`` link). Everything else — argv
parsing, config, training — is lerobot's own code, forwarded unchanged.

Usage (identical CLI to ``lerobot-train``):

    python scripts/act/train_act.py --dataset.repo_id=... --policy.type=act ...
"""

from __future__ import annotations

import sys

from lerobot.common import train_utils as _train_utils

_original_update_last_checkpoint = _train_utils.update_last_checkpoint


def _update_last_checkpoint_without_symlink(checkpoint_dir):
    try:
        return _original_update_last_checkpoint(checkpoint_dir)
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314:
            raise
        print(
            f"[train_act] WinError 1314: skipped the 'last' symlink; "
            f"checkpoint saved at {checkpoint_dir} "
            f"(resume with --resume --config_path <checkpoint>)",
            file=sys.stderr,
            flush=True,
        )
        return checkpoint_dir


# Must happen BEFORE lerobot.scripts.lerobot_train is imported: it binds the
# name at import time via `from lerobot.common.train_utils import (...)`.
_train_utils.update_last_checkpoint = _update_last_checkpoint_without_symlink

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    main()
