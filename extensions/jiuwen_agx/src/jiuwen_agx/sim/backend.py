# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""SimBackend — the simulator seam every simulated machine drives through.

Layering: ``SimMachineDriver`` (protocol slices) → ``SimBackend`` (here) → simulator.

The backend speaks MACHINE-AGNOSTIC primitives — joints, undercarriage motion,
terrain truth, scoop truth, frames. Everything body-specific (keyframe work
cycles, reach envelopes) lives in the per-machine adapter package, so a new
simulated machine never touches this layer.

Three implementations ship:

  ``mock``       :class:`MockSimBackend`      in-memory; targets arrive instantly; zero
                                             dependencies — the offline dev/test path
  ``inprocess``  :class:`InProcessAgxBackend` imports ``agx`` in THIS process (stage B:
                                             fill in against the team's scene interface)
  ``remote``     :class:`RemoteSimBackend`    versioned JSON-line protocol over TCP,
                                             paired with ``scripts/agx_bridge_server.py``
                                             which runs inside AGX's own Python

Adding a backend = a class + one ``@register_backend`` decorator — nothing else
changes. Wire protocol version: remote peer and bridge both speak ``{"v": 1, ...}``.
"""

from __future__ import annotations

import base64
import json
import logging
import socket
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "BACKENDS",
    "InProcessAgxBackend",
    "MockSimBackend",
    "RemoteSimBackend",
    "SimBackend",
    "create_backend",
    "register_backend",
]

# Bridge protocol version — bump when a command/response shape changes
# incompatibly; RemoteSimBackend refuses peers announcing a different major.
PROTOCOL_VERSION = 1

# Default mock terrain: one pile to dig from. Ground points, base frame, metres.
DEFAULT_PILES: list[dict[str, Any]] = [
    {"name": "soil_pile", "x_m": -2.0, "y_m": 1.0, "volume_m3": 2.0},
]


@runtime_checkable
class SimBackend(Protocol):
    """What a simulator backend owes the driver — machine-agnostic primitives.

    Blocking calls (``send_joint_targets`` / ``navigate_*``) return when the
    simulator reports arrival or the timeout elapses; they never raise for
    "didn't quite reach" — the returned state tells the caller where it is.
    """

    def open(self) -> None:
        """Establish the simulator connection. Must be idempotent."""

    def close(self) -> None:
        """Release the simulator. Must be idempotent and safe at any state."""

    def read_joints(self) -> dict[str, float]:
        """Current joint angles keyed by joint name (the body's joint_units)."""

    def send_joint_targets(
        self, targets: dict[str, float], *, timeout_s: float
    ) -> dict[str, float]:
        """Command absolute joint targets (unmentioned joints hold), block until
        reached or timeout, return the full joint state after the motion."""

    def navigate_relative(
        self, dx_m: float, dyaw_rad: float, *, timeout_s: float
    ) -> dict:
        """Drive the undercarriage: turn ``dyaw_rad`` then advance ``dx_m`` metres.
        Differential — there is no strafe. Returns what was commanded."""

    def navigate_arc(
        self, radius_m: float, dyaw_rad: float, *, timeout_s: float
    ) -> dict:
        """Drive ONE constant-curvature arc (signed radius, + = left). Returns
        what was commanded."""

    def read_terrain(self) -> list[dict[str, Any]]:
        """Terrain truth: material piles as {name, x_m, y_m, volume_m3} (base
        frame, metres). Empty list = the sim reports no piles."""

    def scoop_state(self) -> bool:
        """Whether the work tool currently carries material (the payload token
        ``payload.held`` maps onto this)."""

    def mark_scoop(self, loaded: bool) -> None:
        """Write the scoop truth. Work cycles call this at keyframe boundaries to
        keep the observed state consistent with commanded motion; a MEASURING
        backend (AGX Terrain volume in the bucket) derives it from the sim and
        may treat this as a no-op."""

    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        """One (rgb, depth) frame pair, or ``None`` when no camera is available.
        Like the smoke-test stub: a backend without a camera returns None rather
        than inventing content."""


# ---------------------------------------------------------------------------
# Backend registry — new backend = class + decorator, nothing else changes.
# ---------------------------------------------------------------------------
BACKENDS: dict[str, type] = {}


def register_backend(name: str):
    """Register a backend class under ``cfg.backend`` name for :func:`create_backend`."""

    def _wrap(cls: type) -> type:
        BACKENDS[name] = cls
        return cls

    return _wrap


def create_backend(cfg: Any) -> SimBackend:
    """Build the backend a config names. Unknown names fail here, loudly."""
    name = str(getattr(cfg, "backend", "mock"))
    cls = BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown sim backend {name!r}; registered: {sorted(BACKENDS)}"
        )
    return cls(cfg)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# mock — offline development and tests
# ---------------------------------------------------------------------------
@register_backend("mock")
class MockSimBackend:
    """In-memory backend: targets arrive instantly, terrain is a fixed pile list.

    Deterministic and dependency-free, so the whole stack above it (driver, env,
    api, rails, planner) runs offline. Every command is recorded in ``move_log``
    for assertions.
    """

    def __init__(
        self,
        cfg: Any,
        *,
        joint_names: tuple[str, ...] | None = None,
        piles: list[dict[str, Any]] | None = None,
    ) -> None:
        self._joint_names = (
            tuple(joint_names)
            if joint_names is not None
            else tuple(getattr(cfg, "joint_names", ()))
        )
        self._joints: dict[str, float] = dict.fromkeys(self._joint_names, 0.0)
        self._piles: list[dict[str, Any]] = [
            dict(p)
            for p in (piles or getattr(cfg, "terrain_piles", None) or DEFAULT_PILES)
        ]
        self._scoop = False
        self._open = False
        self.move_log: list[dict[str, Any]] = []

    # -- lifecycle
    def open(self) -> None:
        if self._open:
            return
        self._open = True
        self.move_log.append({"cmd": "open"})

    def close(self) -> None:
        self._open = False
        self.move_log.append({"cmd": "close"})

    def _require_open(self) -> None:
        if not self._open:
            raise RuntimeError("MockSimBackend: open() first")

    # -- joints
    def read_joints(self) -> dict[str, float]:
        self._require_open()
        return dict(self._joints)

    def send_joint_targets(
        self, targets: dict[str, float], *, timeout_s: float
    ) -> dict[str, float]:
        self._require_open()
        clean = {str(k): float(v) for k, v in targets.items()}
        self._joints.update(clean)
        self.move_log.append({"cmd": "move_joints", "targets": dict(clean)})
        return dict(self._joints)

    # -- undercarriage
    def navigate_relative(
        self, dx_m: float, dyaw_rad: float, *, timeout_s: float
    ) -> dict:
        self._require_open()
        entry = {
            "cmd": "navigate_relative",
            "dx_m": float(dx_m),
            "dyaw_rad": float(dyaw_rad),
        }
        self.move_log.append(entry)
        return {"dx_m": float(dx_m), "dyaw_rad": float(dyaw_rad)}

    def navigate_arc(
        self, radius_m: float, dyaw_rad: float, *, timeout_s: float
    ) -> dict:
        self._require_open()
        entry = {
            "cmd": "navigate_arc",
            "radius_m": float(radius_m),
            "dyaw_rad": float(dyaw_rad),
        }
        self.move_log.append(entry)
        return {"radius_m": float(radius_m), "dyaw_rad": float(dyaw_rad)}

    # -- terrain / scoop
    def read_terrain(self) -> list[dict[str, Any]]:
        self._require_open()
        return [dict(p) for p in self._piles]

    def scoop_state(self) -> bool:
        return self._scoop

    def mark_scoop(self, loaded: bool) -> None:
        self._scoop = bool(loaded)

    # -- camera (stage B: mock has no camera — None, never invented frames)
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        return None


# ---------------------------------------------------------------------------
# inprocess — AGX in this process (stage B skeleton)
# ---------------------------------------------------------------------------
@register_backend("inprocess")
class InProcessAgxBackend:
    """Load the AGX scene in THIS process and drive it through ``agx`` directly.

    Stage-B skeleton: the class, its config surface and its place in the registry
    are final, but the AGX calls are deliberately NOT written yet — they depend
    on the team's scene structure (constraint names, motor interfaces, stepping
    loop). Fill in the ``TODO(AGX)`` blocks against the team's interface; the
    protocol to satisfy is :class:`SimBackend`, and
    ``scripts/agx_bridge_server.py`` carries an isomorphic command loop worth
    mirroring.
    """

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        self._scene: Any = None

    def open(self) -> None:
        try:
            import agx  # noqa: F401  (lazy; AGX ships its own Python env)
        except ImportError as exc:
            raise RuntimeError(
                "InProcessAgxBackend 需要 AGX Python 环境（import agx 失败）。"
                "在 AGX 自带的 Python 里运行，或改用 backend: remote + scripts/agx_bridge_server.py。"
            ) from exc
        # TODO(AGX): 加载 cfg.scene_path 场景；按 cfg.joint_names 找到对应铰链/液压约束
        # （约束名映射放 config.joint_constraint_map）；初始化履带驱动约束。
        raise NotImplementedError(
            "InProcessAgxBackend: 待 AGX 场景接口确认后实现（TODO(AGX) 块）"
        )

    def close(self) -> None:
        self._scene = None

    def read_joints(self) -> dict[str, float]:
        raise RuntimeError("InProcessAgxBackend: not open")

    def send_joint_targets(
        self, targets: dict[str, float], *, timeout_s: float
    ) -> dict[str, float]:
        raise RuntimeError("InProcessAgxBackend: not open")

    def navigate_relative(
        self, dx_m: float, dyaw_rad: float, *, timeout_s: float
    ) -> dict:
        raise RuntimeError("InProcessAgxBackend: not open")

    def navigate_arc(
        self, radius_m: float, dyaw_rad: float, *, timeout_s: float
    ) -> dict:
        raise RuntimeError("InProcessAgxBackend: not open")

    def read_terrain(self) -> list[dict[str, Any]]:
        raise RuntimeError("InProcessAgxBackend: not open")

    def scoop_state(self) -> bool:
        raise RuntimeError("InProcessAgxBackend: not open")

    def mark_scoop(self, loaded: bool) -> None:
        raise RuntimeError("InProcessAgxBackend: not open")

    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        return None


# ---------------------------------------------------------------------------
# remote — bridge over TCP (works today, AGX or bridge --demo on the other end)
# ---------------------------------------------------------------------------
@register_backend("remote")
class RemoteSimBackend:
    """Talk to ``scripts/agx_bridge_server.py`` over a versioned JSON-line protocol.

    One request = one line in, one line out. The bridge may stand in front of a
    real AGX scene OR its built-in ``--demo`` mock, so the remote path is
    testable end-to-end without AGX. Machine-agnostic: one bridge serves every
    simulated machine.
    """

    def __init__(self, cfg: Any) -> None:
        self._host = str(getattr(cfg, "host", "127.0.0.1"))
        self._port = int(getattr(cfg, "port", 9700))
        self._startup_timeout_s = float(getattr(cfg, "startup_timeout_s", 60.0))
        self._default_timeout_s = float(getattr(cfg, "move_timeout_s", 30.0))
        self._sock: socket.socket | None = None
        self._file: Any = None

    # -- lifecycle
    def open(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection(
            (self._host, self._port), timeout=self._startup_timeout_s
        )
        self._sock = sock
        self._file = sock.makefile("rw", encoding="utf-8", newline="\n")
        try:
            resp = self._call({"cmd": "ping"})
        except Exception:
            self.close()
            raise
        peer_version = int(resp.get("v", 0))
        if peer_version != PROTOCOL_VERSION:
            self.close()
            raise RuntimeError(
                f"bridge protocol version mismatch: peer v{peer_version}, client v{PROTOCOL_VERSION}"
            )

    def close(self) -> None:
        file, sock = self._file, self._sock
        self._file = None
        self._sock = None
        try:
            if file is not None:
                file.close()
        finally:
            if sock is not None:
                sock.close()

    def _require_open(self) -> None:
        if self._sock is None or self._file is None:
            raise RuntimeError("RemoteSimBackend: open() first")

    def _call(
        self, cmd: dict[str, Any], *, read_timeout_s: float | None = None
    ) -> dict[str, Any]:
        self._require_open()
        assert self._sock is not None and self._file is not None
        payload = {"v": PROTOCOL_VERSION, **cmd}
        self._file.write(json.dumps(payload) + "\n")
        self._file.flush()
        old_timeout = self._sock.gettimeout()
        if read_timeout_s is not None:
            self._sock.settimeout(max(float(read_timeout_s), 1.0))
        try:
            line = self._file.readline()
        finally:
            if read_timeout_s is not None:
                self._sock.settimeout(old_timeout)
        if not line:
            raise ConnectionError("bridge closed the connection")
        resp = json.loads(line)
        if not resp.get("ok", False):
            raise RuntimeError(
                f"bridge error on {cmd.get('cmd')!r}: {resp.get('error')!r}"
            )
        return resp

    # -- joints
    def read_joints(self) -> dict[str, float]:
        return {
            str(k): float(v) for k, v in self._call({"cmd": "joints"})["joints"].items()
        }

    def send_joint_targets(
        self, targets: dict[str, float], *, timeout_s: float
    ) -> dict[str, float]:
        resp = self._call(
            {
                "cmd": "move_joints",
                "targets": {str(k): float(v) for k, v in targets.items()},
                "timeout_s": float(timeout_s),
            },
            read_timeout_s=float(timeout_s) + 5.0,
        )
        joints = {str(k): float(v) for k, v in resp["joints"].items()}
        if resp.get("async"):
            # viewer 泵模式：桥接立即返回（不能阻塞主线程），客户端轮询到位
            import time as _time

            wanted = {str(k): float(v) for k, v in resp.get("targets", {}).items()}
            deadline = _time.monotonic() + float(timeout_s) + 5.0
            while _time.monotonic() < deadline:
                if all(
                    abs(joints.get(name, 1e9) - value) < 0.01
                    for name, value in wanted.items()
                ):
                    break
                _time.sleep(0.05)
                joints = self.read_joints()
        return joints

    # -- undercarriage
    def navigate_relative(
        self, dx_m: float, dyaw_rad: float, *, timeout_s: float
    ) -> dict:
        resp = self._call(
            {
                "cmd": "navigate_relative",
                "dx_m": float(dx_m),
                "dyaw_rad": float(dyaw_rad),
                "timeout_s": float(timeout_s),
            },
            read_timeout_s=float(timeout_s) + 5.0,
        )
        return dict(resp["result"])

    def navigate_arc(
        self, radius_m: float, dyaw_rad: float, *, timeout_s: float
    ) -> dict:
        resp = self._call(
            {
                "cmd": "navigate_arc",
                "radius_m": float(radius_m),
                "dyaw_rad": float(dyaw_rad),
                "timeout_s": float(timeout_s),
            },
            read_timeout_s=float(timeout_s) + 5.0,
        )
        return dict(resp["result"])

    # -- terrain / scoop
    def read_terrain(self) -> list[dict[str, Any]]:
        return [dict(p) for p in self._call({"cmd": "terrain"})["piles"]]

    def scoop_state(self) -> bool:
        return bool(self._call({"cmd": "scoop"})["loaded"])

    def mark_scoop(self, loaded: bool) -> None:
        self._call({"cmd": "mark_scoop", "loaded": bool(loaded)})

    # -- camera (stage B)
    def grab_frames(self) -> tuple[np.ndarray, np.ndarray] | None:
        resp = self._call({"cmd": "frame"}, read_timeout_s=10.0)
        if not resp.get("rgb_base64"):
            return None
        shape = tuple(int(v) for v in resp["shape"])
        rgb = np.frombuffer(
            base64.b64decode(resp["rgb_base64"]), dtype=np.uint8
        ).reshape(shape)
        depth = np.frombuffer(
            base64.b64decode(resp["depth_base64"]), dtype=np.float32
        ).reshape(shape[:2])
        return rgb, depth
