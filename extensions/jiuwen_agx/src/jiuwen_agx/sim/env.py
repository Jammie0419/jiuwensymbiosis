# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SimMachineEnv — the one Env every simulated machine uses.

The class-level ``capabilities`` is the SUPERSUPERSET the static adapter
validator checks against ``KNOWN_CAPABILITIES``; the instance-level value is
narrowed in :meth:`_capabilities_for_config` from what this machine's config
actually switches on (``has_base`` / ``terrain_enabled`` / ``camera_enabled`` /
``extra_capabilities``) — the so101 config-gating pattern.

A per-machine package subclasses this only when it must ADD a capability the
generic surface cannot know (the excavator adds ``motion.excavator`` for its
work-cycle action); everything else is configuration.
"""

from __future__ import annotations

import logging
from typing import Any

from jiuwen_agx.sim.config import SimMachineConfig
from jiuwen_agx.sim.driver import SimMachineDriver
from jiuwensymbiosis.env.base import BaseRobotEnv, RobotObservation

logger = logging.getLogger(__name__)

__all__ = ["SimMachineEnv"]


class SimMachineEnv(BaseRobotEnv):
    """One env for every simulated machine — capabilities follow the config."""

    # Class-level superset (validator-checked). vision.* activates in stage B
    # via camera_enabled; motion.excavator-style family capabilities are added
    # by per-machine subclasses, never here.
    capabilities = frozenset(
        {
            "motion.joint",
            "motion.base",
            "sensing.terrain",
            "vision.camera",
            "vision.depth",
            "vision.detection",
            "vision.eye_to_hand",
        }
    )
    _ALWAYS_ON = frozenset({"motion.joint"})
    name = "sim_machine"

    def __init__(self, cfg: SimMachineConfig) -> None:
        """Store config; do NOT connect yet (deferred to connect())."""
        self.cfg = cfg
        self._driver: SimMachineDriver | None = None
        self.joint_units = cfg.joint_units or None  # base setter validates deg/rad
        self.capabilities = self._capabilities_for_config()

    # ------------------------------------------------------------------ capabilities
    def _capabilities_for_config(self) -> frozenset[str]:
        caps = set(self._ALWAYS_ON) | set(self.cfg.extra_capabilities)
        if self.cfg.has_base:
            caps.add("motion.base")
        if self.cfg.terrain_enabled:
            caps.add("sensing.terrain")
        if self.cfg.camera_enabled:
            caps.update(
                {
                    "vision.camera",
                    "vision.depth",
                    "vision.detection",
                    "vision.eye_to_hand",
                }
            )
        return frozenset(caps)

    # ------------------------------------------------------------------ driver seam
    @property
    def low_level(self) -> SimMachineDriver | None:
        return self._driver

    @low_level.setter
    def low_level(self, value: SimMachineDriver | None) -> None:
        """Bind a driver before connect() — the seam a smoke test or a harness uses.
        Once one is bound, only connect/disconnect may rebind it."""
        if self._driver is not None:
            raise AttributeError(
                f"{type(self).__name__}.low_level is already bound — connect/disconnect owns rebinding"
            )
        self._driver = value

    @property
    def driver(self) -> SimMachineDriver:
        """The connected driver; RuntimeError when not connected (public accessor
        so api/work-cycle code never reaches into privates)."""
        if self._driver is None:
            raise RuntimeError(f"{type(self).__name__}: not connected — no driver")
        return self._driver

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        if self._driver is not None:
            # Pre-bound via the low_level setter (smoke test / harness seam):
            # still open the simulator connection — connect() is idempotent, so
            # an already-connected driver just returns.
            self._driver.connect()
            return
        driver = SimMachineDriver(self.cfg)
        try:
            driver.connect()
        except Exception:
            try:
                driver.close()
            except Exception as exc:
                logger.warning(
                    "%s: driver.close() after failed connect raised %s",
                    type(self).__name__,
                    exc,
                )
            raise
        self._driver = driver
        self.capabilities = self._capabilities_for_config()

    def disconnect(self) -> None:
        if self._driver is None:
            return
        try:
            self._driver.close()
        finally:
            self._driver = None
            self.capabilities = self._capabilities_for_config()

    # ------------------------------------------------------------------ observation
    def get_observation(self) -> RobotObservation:
        """Best-effort snapshot; never raises on transient backend gaps."""
        joints: list[float] | None = None
        if self._driver is not None:
            try:
                positions = self._driver.get_joint_positions()
                joints = [
                    float(positions[n]) for n in self.cfg.joint_names if n in positions
                ]
            except Exception as exc:
                logger.debug("[%s] read_joints failed: %s", self.cfg.name, exc)
        extra: dict[str, Any] = {"joint_units": self.cfg.joint_units or None}
        if "sensing.terrain" in self.capabilities and self._driver is not None:
            try:
                extra["terrain"] = self._driver.read_terrain()
            except Exception as exc:
                logger.debug("[%s] read_terrain failed: %s", self.cfg.name, exc)
        if self._driver is not None:
            try:
                extra["bucket_loaded"] = self._driver.scoop_state()
            except Exception as exc:
                logger.debug("[%s] scoop_state failed: %s", self.cfg.name, exc)
        return RobotObservation(joints=joints, extra=extra)

    # ------------------------------------------------------------------ required verbs
    def home(self) -> None:
        """HOME is the one unconditional action: back to the home joint posture."""
        self._require_driver().home()

    # ------------------------------------------------------------------ base verbs (base class raises; we delegate)
    def navigate_relative(
        self, dx_m: float, dy_m: float = 0.0, dyaw_rad: float = 0.0
    ) -> dict:
        return self._require_driver().navigate_relative(dx_m, dy_m, dyaw_rad)

    def navigate_arc(self, radius_m: float, dyaw_rad: float) -> dict:
        return self._require_driver().navigate_arc(radius_m, dyaw_rad)

    # ------------------------------------------------------------------ terrain / payload truth
    def read_terrain(self) -> list[dict[str, Any]]:
        """Terrain truth for the get_terrain action (sensing.terrain-gated)."""
        if "sensing.terrain" not in self.capabilities:
            raise NotImplementedError(
                f"{type(self).__name__}: terrain_enabled=False — no terrain truth"
            )
        return self._require_driver().read_terrain()

    def get_joint_positions(self) -> dict[str, float]:
        """Public verb for the get_joint_positions action."""
        return dict(self._require_driver().get_joint_positions())

    @property
    def holding_payload(self) -> bool:
        """WorldState reads this for the payload.* tokens: the bucket carrying
        material IS a held payload (and RecoveryRail must not home blindly)."""
        if self._driver is None:
            return False
        try:
            return self._driver.scoop_state()
        except Exception:
            return False

    # ------------------------------------------------------------------ safety envelopes (read-only)
    @property
    def joint_limits(self) -> dict[str, tuple[float, float]] | None:
        limits = self.cfg.joint_limits
        if limits is None:
            return None
        # Re-insert in joint_names order for stable indexing.
        return {
            name: limits[name] for name in self.cfg.joint_names if name in limits
        } or None

    @joint_limits.setter
    def joint_limits(self, _: dict[str, tuple[float, float]] | None) -> None:
        raise AttributeError(
            "SimMachineEnv.joint_limits is read-only (read from config)"
        )

    @property
    def base_step_limits(self) -> tuple[float, float] | None:
        if not self.cfg.has_base or self.cfg.base_step_limits is None:
            return None
        return (
            float(self.cfg.base_step_limits[0]),
            float(self.cfg.base_step_limits[1]),
        )

    @base_step_limits.setter
    def base_step_limits(self, _: tuple[float, float] | None) -> None:
        raise AttributeError(
            "SimMachineEnv.base_step_limits is read-only (read from config)"
        )
