# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AGX 桥接服务 —— jiuwensymbiosis 仿真适配器（backend: remote）的对端。

**独立脚本，仅用标准库**：它跑在 AGX 自带的 Python 环境里（与本仓库的依赖
不兼容），因此绝不 import jiuwensymbiosis。协议与
``jiuwensymbiosis/adapters/_common/sim/backend.py::RemoteSimBackend`` 一一对应。

用法：
  # 真实 AGX 场景（实现 AgxSceneAdapter 的 TODO(AGX) 方法后）
  python agx_bridge_server.py --scene scene.agx --joint-map swing=YawHinge,boom=BoomHinge

  # 无 AGX 演示/联调：内置内存机器，行为与 MockSimBackend 一致
  python agx_bridge_server.py --demo --port 9700

协议（版本化 JSON 行协议，v=1；一行请求，一行响应）：
  请求  {"v": 1, "cmd": "...", ...}
  响应  {"v": 1, "ok": true, ...}          成功（携带命令各自的载荷字段）
        {"v": 1, "ok": false, "error": ""} 失败
  命令： ping | joints | move_joints{targets,timeout_s}
        navigate_relative{dx_m,dyaw_rad,timeout_s} | navigate_arc{radius_m,dyaw_rad,timeout_s}
        terrain | scoop | mark_scoop{loaded} | frame | bye

所有角度/坐标的单位由使用方约定（jiuwensymbiosis 侧 config 的 joint_units 与
基座系米，REP-103）；桥接层只透传，不做单位换算。
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
from typing import Any

PROTOCOL_VERSION = 1

# 演示机器的地形真值（基座系，米）。
DEMO_PILES: list[dict[str, Any]] = [
    {"name": "soil_pile", "x_m": -2.0, "y_m": 1.0, "volume_m3": 2.0},
]


class AgxSceneAdapter:
    """AGX 场景接入点 —— 把仿真器包成 8 个方法，与 SimBackend 语义一致。

    TODO(AGX)：按组内场景结构实现下列方法。约束名映射通过 --joint-map
    传入（swing=HingeName,...），行走差速由两条履带约束的速度差实现。
    """

    def __init__(self, joint_names: list[str], joint_map: dict[str, str], scene_path: str | None) -> None:
        self.joint_names = joint_names
        self.joint_map = joint_map
        self.scene_path = scene_path
        self._joints: dict[str, float] = dict.fromkeys(joint_names, 0.0)
        self._scoop = False
        self._piles = [dict(p) for p in DEMO_PILES]

    # -- 场景
    def load(self) -> None:
        if self.scene_path:
            # TODO(AGX): import agx; self._scene = agx.loadScene(self.scene_path)
            raise NotImplementedError("AgxSceneAdapter.load：待 AGX 场景接口确认后实现（TODO(AGX) 块）")
        raise ValueError("--demo 或 --scene 必须二选一")

    def step_until_settled(self, timeout_s: float) -> None:
        """步进仿真直到运动到位或超时。TODO(AGX)。demo 实现为即时到位。"""

    # -- 关节
    def read_joints(self) -> dict[str, float]:
        return dict(self._joints)

    def send_joint_targets(self, targets: dict[str, float], timeout_s: float) -> dict[str, float]:
        # TODO(AGX): 对 joint_map 命中的约束写电机目标，然后 step_until_settled(timeout_s)
        self._joints.update(targets)
        return dict(self._joints)

    # -- 履带底盘
    def navigate_relative(self, dx_m: float, dyaw_rad: float, timeout_s: float) -> dict[str, float]:
        # TODO(AGX): 差速 = 左右履带速度差；step_until_settled
        return {"dx_m": dx_m, "dyaw_rad": dyaw_rad}

    def navigate_arc(self, radius_m: float, dyaw_rad: float, timeout_s: float) -> dict[str, float]:
        # TODO(AGX): 常曲率弧线 = 两侧履带不同速度
        return {"radius_m": radius_m, "dyaw_rad": dyaw_rad}

    # -- 地形 / 铲斗
    def read_terrain(self) -> list[dict[str, Any]]:
        # TODO(AGX): 从 AGX Terrain/场景对象读料堆质心与体积
        return [dict(p) for p in self._piles]

    def scoop_state(self) -> bool:
        # TODO(AGX): 由铲斗内物料体积推导
        return self._scoop

    def mark_scoop(self, loaded: bool) -> None:
        self._scoop = bool(loaded)

    # -- 相机（阶段B）
    def grab_frame(self) -> dict[str, Any] | None:
        # TODO(AGX): AGX 渲染一帧 → base64(rgb_jpeg) + base64(depth_f32) + shape
        return None


class DemoSceneAdapter(AgxSceneAdapter):
    """内存演示机器：即时到位，行为与 MockSimBackend 一致（无需 AGX）。"""

    def __init__(self, joint_names: list[str]) -> None:
        super().__init__(joint_names, joint_map={}, scene_path=None)

    def load(self) -> None:
        pass  # 内存机器，无事可做


class BridgeSession:
    """One TCP connection: JSON-line request in, JSON-line response out."""

    def __init__(self, scene: AgxSceneAdapter) -> None:
        self._scene = scene

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        version = int(request.get("v", 0))
        if version != PROTOCOL_VERSION:
            return {"v": PROTOCOL_VERSION, "ok": False, "error": f"protocol version {version} unsupported"}
        cmd = str(request.get("cmd", ""))
        handler = getattr(self, f"_cmd_{cmd}", None)
        if handler is None:
            return {"v": PROTOCOL_VERSION, "ok": False, "error": f"unknown command {cmd!r}"}
        return {"v": PROTOCOL_VERSION, "ok": True, **handler(request)}

    def _num(self, request: dict[str, Any], key: str, default: float) -> float:
        return float(request.get(key, default))

    # -- commands
    def _cmd_ping(self, _request: dict[str, Any]) -> dict[str, Any]:
        return {"pong": True}

    def _cmd_joints(self, _request: dict[str, Any]) -> dict[str, Any]:
        return {"joints": self._scene.read_joints()}

    def _cmd_move_joints(self, request: dict[str, Any]) -> dict[str, Any]:
        targets = {str(k): float(v) for k, v in dict(request.get("targets") or {}).items()}
        joints = self._scene.send_joint_targets(targets, self._num(request, "timeout_s", 30.0))
        return {"joints": joints}

    def _cmd_navigate_relative(self, request: dict[str, Any]) -> dict[str, Any]:
        result = self._scene.navigate_relative(
            self._num(request, "dx_m", 0.0), self._num(request, "dyaw_rad", 0.0), self._num(request, "timeout_s", 30.0)
        )
        return {"result": result}

    def _cmd_navigate_arc(self, request: dict[str, Any]) -> dict[str, Any]:
        result = self._scene.navigate_arc(
            self._num(request, "radius_m", 0.0),
            self._num(request, "dyaw_rad", 0.0),
            self._num(request, "timeout_s", 30.0),
        )
        return {"result": result}

    def _cmd_terrain(self, _request: dict[str, Any]) -> dict[str, Any]:
        return {"piles": self._scene.read_terrain()}

    def _cmd_scoop(self, _request: dict[str, Any]) -> dict[str, Any]:
        return {"loaded": bool(self._scene.scoop_state())}

    def _cmd_mark_scoop(self, request: dict[str, Any]) -> dict[str, Any]:
        self._scene.mark_scoop(bool(request.get("loaded", False)))
        return {}

    def _cmd_frame(self, _request: dict[str, Any]) -> dict[str, Any]:
        frame = self._scene.grab_frame()
        if frame is None:
            return {}
        encoded = dict(frame)
        if "rgb_bytes" in encoded:
            encoded["rgb_base64"] = base64.b64encode(encoded.pop("rgb_bytes")).decode("ascii")
        if "depth_bytes" in encoded:
            encoded["depth_base64"] = base64.b64encode(encoded.pop("depth_bytes")).decode("ascii")
        return encoded


def serve(scene: AgxSceneAdapter, host: str, port: int) -> None:
    scene.load()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(
        f"[agx_bridge] listening on {host}:{port} (protocol v{PROTOCOL_VERSION}, joints={list(scene.joint_names)})",
        flush=True,
    )
    bridge = BridgeSession(scene)
    while True:
        conn, addr = server.accept()
        print(f"[agx_bridge] client {addr} connected", flush=True)
        try:
            with conn:
                reader = conn.makefile("r", encoding="utf-8", newline="\n")
                writer = conn.makefile("w", encoding="utf-8", newline="\n")
                for line in reader:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        request = json.loads(line)
                    except json.JSONDecodeError as exc:
                        response = {"v": PROTOCOL_VERSION, "ok": False, "error": f"bad json: {exc}"}
                    else:
                        if request.get("cmd") == "bye":
                            break
                        response = bridge.handle(request)
                    writer.write(json.dumps(response) + "\n")
                    writer.flush()
        except (ConnectionError, BrokenPipeError) as exc:
            print(f"[agx_bridge] client {addr} dropped: {exc}", flush=True)
        finally:
            print(f"[agx_bridge] client {addr} disconnected", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AGX ↔ jiuwensymbiosis 仿真桥接服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9700)
    parser.add_argument("--demo", action="store_true", help="内置内存演示机器（无需 AGX，联调用）")
    parser.add_argument("--scene", default=None, help="AGX 场景文件（TODO(AGX) 接入后生效）")
    parser.add_argument("--joints", default="swing,boom,arm,bucket", help="关节名（逗号分隔）")
    parser.add_argument("--joint-map", default="", help="关节→AGX约束名映射，如 swing=YawHinge,boom=BoomHinge")
    args = parser.parse_args()

    joint_names = [j.strip() for j in args.joints.split(",") if j.strip()]
    joint_map = dict(part.split("=", 1) for part in args.joint_map.split(",") if "=" in part)
    if args.demo:
        scene = DemoSceneAdapter(joint_names)
    else:
        scene = AgxSceneAdapter(joint_names, joint_map, args.scene)
    serve(scene, args.host, args.port)


if __name__ == "__main__":
    main()
