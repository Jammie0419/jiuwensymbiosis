# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Windows launcher for ``examples/run_task.py`` — installs a POSIX fcntl.flock shim.

Why this exists
---------------
``jiuwensymbiosis/runtime/{jobs,resources}.py`` import ``fcntl`` at module top
(Linux-only). The runner's non-piper path does
``from jiuwensymbiosis.runtime.bindings import prepare_binding`` and then enters
``runtime.admission.admitted_session``, so on Windows it dies with
``No module named 'fcntl'`` before any adapter is built — even though the fcntl
surface it uses is tiny (``flock`` / ``LOCK_EX`` / ``LOCK_NB`` on empty lock
files). The official ``jiuwensymbiosis-actions --config`` path hits the same
wall.

This launcher installs a ``fcntl`` module backed by ``msvcrt`` byte-range locks
(the same pattern openjiuwen's own Windows branch uses) and then runs
``examples/run_task.py`` unchanged. Admission therefore still takes REAL
cross-process locks — it does not silently disable device exclusivity.
openjiuwen guards its own import behind ``os.name == "nt"``, so the shim only
ever feeds ``jiuwensymbiosis.runtime``.

The upstream fix belongs in core (guard the import + platform lock fallback);
until then use this launcher on Windows:

    python scripts/run_task_windows.py --config <yaml> --query "<task>"

All arguments are forwarded verbatim to ``examples/run_task.py``.
"""

from __future__ import annotations

import errno
import msvcrt
import os
import runpy
import sys
import types
from pathlib import Path

_LOCK_SH = 1
_LOCK_EX = 2
_LOCK_NB = 4
_LOCK_UN = 8


def _flock(fileobj, operation):  # noqa: ANN001, ANN201 - POSIX flock signature
    """flock(2) on an open file object or raw fd, backed by msvcrt locks."""
    fd = fileobj.fileno() if hasattr(fileobj, "fileno") else int(fileobj)
    position = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        if operation & _LOCK_UN:
            try:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass  # not held — POSIX flock(LOCK_UN) is a no-op then too
            return
        mode = msvcrt.LK_NBLCK if operation & _LOCK_NB else msvcrt.LK_LOCK
        while True:
            try:
                msvcrt.locking(fd, mode, 1)
                return
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    if operation & _LOCK_NB:
                        # resources.acquire() catches BlockingIOError → ResourceBusyError
                        raise BlockingIOError(exc.errno, str(exc)) from exc
                    continue  # blocking flock: retry (LK_LOCK itself waits ~10 s)
                raise
    finally:
        os.lseek(fd, position, os.SEEK_SET)


def _install_fcntl_shim() -> None:
    if "fcntl" in sys.modules:
        return
    shim = types.ModuleType("fcntl")
    shim.LOCK_SH = _LOCK_SH
    shim.LOCK_EX = _LOCK_EX
    shim.LOCK_NB = _LOCK_NB
    shim.LOCK_UN = _LOCK_UN
    shim.flock = _flock
    sys.modules["fcntl"] = shim


def main() -> None:
    _install_fcntl_shim()
    runner = Path(__file__).resolve().parents[1] / "examples" / "run_task.py"
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
