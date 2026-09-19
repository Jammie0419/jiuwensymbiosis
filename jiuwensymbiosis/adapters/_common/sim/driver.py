# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SimMachineDriver — the one driver every simulated machine uses.

Satisfies the capability-sliced protocols in ``env/protocol.py``:

  RobotDriver        close()
  NamedJointDriver   get_joint_positions / move_joints_blocking  (motion.joint)
  BaseDriver         navigate_relative / navigate_arc             (motion.base, cfg-gated)
  CameraDriver       grab_frames                                  (vision.*, stage B)

All simulator specifics live behind the :class:`~.backend.SimBackend` seam; this
class adds only what the protocols need and a simulator cannot decide: joint-name
validation, finite-value checks, and the differential-base strafe rejection. A
per-machine package never subclasses this — it configures it and adds work-cycle
functions beside it.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from jiuwensymbiosis.adapters._common.sim.backend import SimBackend, create_backend
from jiuwensymbiosis.adapters._common.sim.config import SimMachineConfig

logger = logging.getLogger(__name__)

__all__ = ["SimMachineDriver"]


class SimMachineDriver:
    """Drive one simulated machine through a :class:`SimBackend`."""

    def __init__(self, cfg: SimMachineConfig, backend: SimBackend | None = None) -> None:
        """``backend`` injection is the test/simulator seam: None = build from cfg."""
        self._cfg = cfg
        self._backend: SimBackend = backend if backend is not None else create_backend(cfg)
        self._connected = False

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        """Idempotent. Open the simulator (or attach to the running one)."""
        if self._connected:
            return
        self._backend.open()
        self._connected = True
        logger.info("[%s] sim backend %s connected.", self._cfg.name, type(self._backend).__name__)

    def close(self) -> None:
        """Idempotent and safe at any state."""
        if not self._connected:
            return
        try:
            self._backend.close()
        finally:
            self._connected = False

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError(f"{self._cfg.name}: connect() first")

    # ------------------------------------------------------------------ joints (NamedJointDriver)
    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._cfg.joint_names

    def get_joint_positions(self) -> dict[str, float]:
        """Latest joint angles keyed by joint name (the body's joint_units)."""
        self._require_connected()
        return self._backend.read_joints()

    def move_joints_blocking(
        self,
        targets: dict[str, float],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, float]:
        """Move the NAMED joints to absolute targets; unmentioned joints HOLD.

        Unknown names are refused (the vocabulary promises this), non-finite
        values are refused, and the call blocks until the simulator reports
        arrival or the timeout elapses.
        """
        self._require_connected()
        clean: dict[str, float] = {}
        for name, value in dict(targets).items():
            joint = str(name)
            if joint not in self._cfg.joint_names:
                raise ValueError(
                    f"{self._cfg.name}: unknown joint {joint!r}; known joints: {list(self._cfg.joint_names)}"
                )
            position = float(value)
            if not math.isfinite(position):
                raise ValueError(f"{self._cfg.name}: non-finite target for joint {joint!r}: {value!r}")
            limits = self._cfg.joint_limits
            if limits is not None and joint in limits:
                low, high = limits[joint]
                if not (low <= position <= high):
                    raise ValueError(
                        f"{self._cfg.name}: joint {joint!r} target {position} outside configured "
                        f"limits [{low}, {high}] — fix the config/limits or the keyframe tuning"
                    )
            clean[joint] = position
        if not clean:
            raise ValueError(f"{self._cfg.name}: empty joint command")
        return self._backend.send_joint_targets(
            clean, timeout_s=float(timeout_s if timeout_s is not None else self._cfg.move_timeout_s)
        )

    def home(self) -> None:
        """Return to the configured home joint posture. No home_joints = no-op."""
        if not self._cfg.home_joints:
            logger.debug("[%s] home: no home_joints configured, treating as already home", self._cfg.name)
            return
        self.move_joints_blocking(dict(self._cfg.home_joints))

    # ------------------------------------------------------------------ undercarriage (BaseDriver)
    def _require_base(self) -> None:
        if not self._cfg.has_base:
            raise NotImplementedError(
                f"{self._cfg.name}: undercarriage motion needs has_base=True in the config "
                "(the env then declares 'motion.base' and the verbs become reachable)."
            )

    def navigate_relative(
        self, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0, *, timeout_s: float | None = None
    ) -> dict:
        """Turn by ``dyaw_rad`` then advance ``dx_m`` metres (REP-103). Differential:
        a tracked undercarriage cannot strafe, so ``dy_m`` is ignored."""
        self._require_connected()
        self._require_base()
        if abs(float(dy_m)) > 1e-9:
            logger.debug(
                "[%s] navigate_relative: dy_m=%s ignored (differential base cannot strafe)", self._cfg.name, dy_m
            )
        return self._backend.navigate_relative(
            float(dx_m),
            float(dyaw_rad),
            timeout_s=float(timeout_s if timeout_s is not None else self._cfg.move_timeout_s),
        )

    def navigate_arc(self, radius_m: float, dyaw_rad: float, *, timeout_s: float | None = None) -> dict:
        """Drive ONE constant-curvature arc (signed radius, + = left)."""
        self._require_connected()
        self._require_base()
        return self._backend.navigate_arc(
            float(radius_m),
            float(dyaw_rad),
            timeout_s=float(timeout_s if timeout_s is not None else self._cfg.move_timeout_s),
        )

    def rotate_base(self, dyaw_rad: float, *, timeout_s: float | None = None) -> dict:
        """Turn in place — a pure rotate cannot translate, so dx_m is pinned to 0."""
        return self.navigate_relative(0.0, 0.0, dyaw_rad, timeout_s=timeout_s)

    # ------------------------------------------------------------------ terrain / scoop
    def read_terrain(self) -> list[dict[str, Any]]:
        """Terrain truth: material piles (base frame, metres)."""
        self._require_connected()
        return self._backend.read_terrain()

    def scoop_state(self) -> bool:
        """Whether the work tool currently carries material (→ payload.held)."""
        return bool(self._backend.scoop_state())

    def mark_scoop(self, loaded: bool) -> None:
        """Sim-truth write for work cycles; see SimBackend.mark_scoop."""
        self._backend.mark_scoop(loaded)

    # ------------------------------------------------------------------ camera (stage B)
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """One (rgb, depth) pair, or None when this backend has no camera."""
        self._require_connected()
        return self._backend.grab_frames()

    # ------------------------------------------------------------------ introspection
    @property
    def backend(self) -> SimBackend:
        """Read-only access for tests and bring-up scripts."""
        return self._backend
