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
        terrain | scoop | mark_scoop{loaded} | frame | inventory | bye

所有角度/坐标的单位由使用方约定（jiuwensymbiosis 侧 config 的 joint_units 与
基座系米，REP-103）；桥接层只透传，不做单位换算。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
from typing import Any

PROTOCOL_VERSION = 1

# 演示机器的地形真值（基座系，米）。
DEMO_PILES: list[dict[str, Any]] = [
    {"name": "soil_pile", "x_m": -2.0, "y_m": 1.0, "volume_m3": 2.0},
]


class AgxSceneAdapter:
    """AGX 场景接入点 —— 把仿真器包成 8 个方法，与 SimBackend 语义一致。

    两种运行模式：
      headless —— 独立进程：适配器自己拥有仿真循环（send 后 stepTo 到位）。
      viewer   —— agxViewer 的 .agxPy 插件：viewer 拥有步进（实时渲染，
                  用户 RDP 看到的就是被控制的仿真），适配器只设伺服目标并轮询。

    场景两种来源：
      scene_path 加载 .agx 文件（约束按 --joint-map / 自动发现映射）
      excavator=True 程序化搭建 AGX 自带的 365 挖掘机 + 沙地地形
                     （镜像 data/python/agxTerrain/excavator_365_terrain.agxPy）

    关节映射（AGX 自带挖掘机模型，单位混合——hinge 是弧度、液压缸是米）：
      swing  -> cabin_hinge            (Hinge,     rad)
      boom   -> arm_prismatics[0]      (Prismatic, m)
      arm    -> stick_prismatic        (Prismatic, m)
      bucket -> bucket_prismatic       (Prismatic, m)
    joint_map 参数可覆盖任意映射（"swing=CabHinge" 形式）。
    """

    # 挖掘机默认关节映射：名字 -> (模型属性, 索引, 单位)
    EXCAVATOR_JOINT_MAP: dict[str, tuple[str, int, str]] = {
        "swing": ("cabin_hinge", 0, "rad"),
        "boom": ("arm_prismatics", 0, "m"),
        "arm": ("stick_prismatic", 0, "m"),
        "bucket": ("bucket_prismatic", 0, "m"),
    }

    def __init__(
        self,
        joint_names: list[str],
        joint_map: dict[str, str],
        scene_path: str | None,
        *,
        mode: str = "headless",
        build_excavator: bool = False,
    ) -> None:
        self.joint_names = joint_names
        self.joint_map = joint_map
        self.scene_path = scene_path
        self.mode = mode
        self.build_excavator = build_excavator
        # viewer 模式走"每步回调泵"（agxViewer 会冻结后台线程，不能用线程服务）
        self.pump_mode = mode == "viewer"
        self._joints: dict[str, float] = dict.fromkeys(joint_names, 0.0)
        self._scoop = False
        self._piles = [dict(p) for p in DEMO_PILES]
        # AGX 状态（load() 后可用）
        self._sim = None
        self._terrain = None
        self._shovel = None
        self._excavator = None
        self._constraints: dict[str, Any] = {}  # 名字 -> agx 约束对象
        self._units: dict[str, str] = {}  # 名字 -> "rad" | "m"
        self._ranges: dict[str, tuple[float, float]] = {}
        # pump 状态（viewer 模式）
        self._listener = None
        self._conn = None
        self._read_buffer = b""
        self._bridge_session: BridgeSession | None = None
        self._drive_plan: tuple[list, float] | None = None  # (hinges, 停车仿真时刻)

    # -- 场景
    def load(self) -> None:
        if self.build_excavator:
            self._load_excavator_scene()
            return
        if self.scene_path:
            self._load_scene_file(self.scene_path)
            return
        raise ValueError("--demo / --scene / --excavator 必须三选一")

    def _require_sim(self):
        if self._sim is None:
            raise RuntimeError("AgxSceneAdapter: load() first")
        return self._sim

    def _load_scene_file(self, path: str) -> None:
        """加载 .agx 场景文件并自动发现铰链/棱柱约束。"""
        import agx
        import agxSDK

        if self.mode == "viewer":
            from agxPythonModules.utils.environment import simulation

            self._sim = simulation()  # viewer 拥有仿真；agxViewer 已加载场景
        else:
            self._sim = agxSDK.Simulation()
            if hasattr(agx, "loadScene"):
                agx.loadScene(path, self._sim)
            else:  # pragma: no cover - 版本差异兜底
                raise RuntimeError("agx.loadScene 不可用：请改用 --excavator 模式或 agxViewer 插件")
        self._auto_discover()

    def _load_excavator_scene(self) -> None:
        """程序化搭建 AGX 自带挖掘机 + 沙地（镜像官方 excavator_365_terrain.agxPy）。"""
        import agx
        import agxCollide
        import agxSDK
        import agxTerrain

        if self.mode == "viewer":
            from agxPythonModules.utils.environment import simulation

            self._sim = simulation()
        else:
            self._sim = agxSDK.Simulation()
        sim = self._sim

        # ---- 地形（平整沙地，50x50 m，最大挖掘深度 5 m）
        resolution, size = (200, 200), (50.0, 50.0)
        hf = agxCollide.HeightField(resolution[0], resolution[1], size[0], size[1])
        terrain = agxTerrain.Terrain.createFromHeightField(hf, 5.0)
        terrain.loadLibraryMaterial("SAND_1")
        sim.add(terrain)
        self._terrain = terrain

        # ---- 渲染（仅 viewer 模式有 root()）
        if self.mode == "viewer":
            import agxOSG
            from agxPythonModules.utils.environment import root

            renderer = agxOSG.TerrainVoxelRenderer(terrain, root())
            renderer.setRenderHeights(True, agx.RangeReal(-1.25, 1.25))
            renderer.setRenderSoilParticlesMesh(True)
            sim.add(renderer)

        # ---- 挖掘机模型（AGX 自带 365；内部加载自身 .agx 并装配履带/液压缸）
        from agxPythonModules.models.excavators.excavator365 import Excavator365

        # JIUWEN_KEYBOARD=1 时启用官方键盘手动控制（noVNC/RDP 里直接按键开挖掘机）
        keyboard = gamepad = None
        if os.environ.get("JIUWEN_KEYBOARD") == "1":
            keyboard = Excavator365.default_keyboard_settings()
            gamepad = Excavator365.default_gamepad_controls()
            print("[agx_bridge] keyboard control ENABLED (a/z bucket, s/x stick, arrows arm+cabin, PgUp/PgDn tracks)", flush=True)

        excavator = Excavator365(
            gamepad_controls=gamepad,
            keyboard_controls=keyboard,
            use_low_degree_tracks_model=True,
        )
        excavator.setRotation(terrain.getRotation())
        excavator.setPosition(0.0, 10.0, 0.0)
        sim.add(excavator)
        self._excavator = excavator

        # ---- 铲斗 shovel（切割边 + 挖掘设置，与官方脚本一致）
        terrain_shovel = agxTerrain.Shovel(
            excavator.bucket_body,
            excavator.top_edge,
            excavator.cutting_edge,
            excavator.forward_cutting_vector,
        )
        terrain_shovel.getSettings().setVerticalBladeSoilMergeDistance(0.0)
        terrain_shovel.getAdvancedSettings().setNoMergeExtensionDistance(0.1)
        terrain_shovel.getAdvancedSettings().setContactRegionVerticalLimit(0.2)
        terrain_shovel.getAdvancedSettings().setContactRegionThreshold(0.1)
        terrain.getTerrainMaterial().getExcavationContactProperties().setAggregateStiffnessMultiplier(5e-4)
        terrain.getProperties().setMaximumParticleActivationVolume(2)
        sim.add(terrain_shovel)
        self._shovel = terrain_shovel

        self._auto_discover()

    def _auto_discover(self) -> None:
        """把词表关节名映射到 AGX 约束组：--joint-map 优先，否则用挖掘机模型属性。

        一个词表关节可能对应一组约束（365 的 boom = 双液压缸 ArmPrismatic1/2，
        官方键盘控制整组同步驱动）——存为列表，伺服时整组设同一目标。
        """
        exc = self._excavator
        for name in self.joint_names:
            if name in self.joint_map:
                constraint = self._find_constraint_by_name(self.joint_map[name])
                group = [constraint]
                unit = "rad" if type(constraint).__name__ == "Hinge" else "m"
            elif exc is not None and name in self.EXCAVATOR_JOINT_MAP:
                attr, _index, unit = self.EXCAVATOR_JOINT_MAP[name]
                value = getattr(exc, attr)
                group = list(value) if isinstance(value, list) else [value]
            else:
                print(f"[agx_bridge] WARNING: joint {name!r} 无映射（--joint-map 可指定）", flush=True)
                continue
            self._constraints[name] = group
            self._units[name] = unit
            range_real = group[0].getRange1D().getRange()
            lo, hi = float(range_real.lower()), float(range_real.upper())
            if lo == float("-inf"):
                lo, hi = -180.0, 180.0  # 全行程回转：限位按 ±180 报告
            self._ranges[name] = (lo, hi)
            names = [c.getName() for c in group]
            print(f"[agx_bridge] joint {name!r} -> {unit} group={names} range={self._ranges[name]}", flush=True)

    def _find_constraint_by_name(self, constraint_name: str):
        sim = self._require_sim()
        for c in sim.getConstraints():
            if c.getName() == constraint_name:
                return c
        raise ValueError(f"约束 {constraint_name!r} 不在场景中（可用名见 inventory）")

    def _step(self, duration_s: float) -> None:
        """headless 模式推进一步；viewer 模式由 agxViewer 步进，这里只等待。"""
        if self.mode == "headless":
            sim = self._require_sim()
            target = sim.getTimeStamp() + duration_s
            while sim.getTimeStamp() < target:
                sim.stepTo(target)
        else:
            import time

            time.sleep(duration_s)

    def step_until_settled(self, timeout_s: float) -> None:
        """独立推进仿真（headless 模式的等待实现）。"""
        self._step(min(timeout_s, 1 / 60))

    # -- 关节
    def read_joints(self) -> dict[str, float]:
        return {name: float(group[0].getAngle()) for name, group in self._constraints.items()}

    def set_joint_targets(self, targets: dict[str, float]) -> None:
        """立即应用位置伺服目标（Lock1D + 距离自适应阻尼，官方 JointController
        模式）。不等待到达——viewer 模式由仿真每步收敛，headless 由
        send_joint_targets 阻塞轮询。一个词表关节的整组约束设同一目标。"""
        import agx

        for name, target in targets.items():
            group = self._constraints[name]
            for c in group:
                c.getMotor1D().setEnable(False)
                lock = c.getLock1D()
                lock.setEnable(True)
                distance = abs(float(target) - c.getAngle())
                damping = agx.logInterpolate(2 / 100, 1.5, 1 - min(distance * 0.1, 1.0))
                lock.setDamping(max(damping, 2 / 60))
                lock.setForceRange(c.getMotor1D().getForceRange())
                lock.setPosition(float(target))
        self._joints.update({str(k): float(v) for k, v in targets.items()})

    def send_joint_targets(self, targets: dict[str, float], timeout_s: float) -> dict[str, float]:
        """headless 模式：应用伺服目标并阻塞到到位/超时。

        viewer 模式不走这里（agxViewer 拥有步进，且线程会被冻结）——
        BridgeSession 对 pump_mode 场景走异步流，客户端轮询到位。
        """
        import time as _time

        self._require_sim()
        self.set_joint_targets(targets)
        dt = 1 / 60
        deadline = _time.monotonic() + float(timeout_s)
        while _time.monotonic() < deadline:
            if all(
                abs(float(targets[name]) - group[0].getAngle())
                < max(1e-3, 0.01 * (self._ranges[name][1] - self._ranges[name][0]))
                for name, group in self._constraints.items()
                if name in targets
            ):
                break
            self._step(dt)
        return self.read_joints()

    # -- 履带底盘
    def navigate_relative(self, dx_m: float, dyaw_rad: float, timeout_s: float) -> dict[str, float]:
        """履带差速开环控制：Motor1D.setSpeed 驱动驱动轮（官方 set_speed 模式）。

        headless：阻塞走完两段（先转后走）。viewer（pump）：设速度后由 pump()
        按仿真时间自动停车（不能阻塞渲染主线程）。
        """
        exc = self._excavator
        if exc is None:
            raise RuntimeError("navigate_relative 需要 --excavator 场景（履带驱动轮）")
        speed = 1.0  # rad/s（驱动轮角速度）
        hinges = list(exc.sprocket_hinges)

        def _apply(left: float, right: float) -> None:
            for h, s in zip(hinges, (left, right), strict=False):
                h.getLock1D().setEnable(False)
                h.getMotor1D().setEnable(True)
                h.getMotor1D().setSpeed(s)

        def _duration(left: float, right: float, seconds: float) -> None:
            if self.pump_mode:
                # 记录停车计划，pump() 里按仿真时间执行
                self._drive_plan = (list(hinges), self._sim.getTimeStamp() + seconds)
            else:
                _apply(left, right)
                self._step(seconds)
                for h in hinges:
                    h.getMotor1D().setSpeed(0.0)

        turn_rate = 0.5  # rad/s 近似原地转速
        if abs(dyaw_rad) > 1e-4:
            sign = 1.0 if dyaw_rad > 0 else -1.0
            _apply(sign * speed, -sign * speed)
            _duration(sign * speed, -sign * speed, abs(dyaw_rad) / turn_rate)
        wheel_radius = 0.3  # 驱动轮半径近似（米）
        if abs(dx_m) > 1e-4:
            direction = 1.0 if dx_m > 0 else -1.0
            _apply(direction * speed, direction * speed)
            _duration(direction * speed, direction * speed, abs(dx_m) / (speed * wheel_radius))
        if self.pump_mode and abs(dx_m) <= 1e-4 and abs(dyaw_rad) <= 1e-4:
            _apply(0.0, 0.0)
        return {"dx_m": float(dx_m), "dyaw_rad": float(dyaw_rad)}

    def navigate_arc(self, radius_m: float, dyaw_rad: float, timeout_s: float) -> dict[str, float]:
        """常曲率弧线：内外履带速度差（v1 简化为差速时间近似）。"""
        exc = self._excavator
        if exc is None:
            raise RuntimeError("navigate_arc 需要 --excavator 场景")
        speed = 1.0
        wheel_radius = 0.3
        track_width = 2.0  # 两履带间距近似（米）
        arc_len = abs(radius_m * dyaw_rad)
        duration = arc_len / max(speed * wheel_radius, 1e-6)
        v_out = speed if radius_m >= 0 else -speed
        v_in = v_out * (abs(radius_m) - track_width / 2) / max(abs(radius_m) + track_width / 2, 1e-6)
        hinges = list(exc.sprocket_hinges)
        for h, s in zip(hinges, (v_out, v_in), strict=False):
            h.getLock1D().setEnable(False)
            h.getMotor1D().setEnable(True)
            h.getMotor1D().setSpeed(s)
        if self.pump_mode:
            self._drive_plan = (hinges, self._sim.getTimeStamp() + duration)
        else:
            self._step(duration)
            for h in hinges:
                h.getMotor1D().setSpeed(0.0)
        return {"radius_m": float(radius_m), "dyaw_rad": float(dyaw_rad)}

    # -- 地形 / 铲斗
    def read_terrain(self) -> list[dict[str, Any]]:
        if self._terrain is not None:
            # 平整沙地场景：土壤无处不在，报告一个"机身前方 3 m"的代表性可挖点
            #（基座系；确保落在可达环带内，LLM 引用它即得到合法 dig 目标）
            return [{"name": "soil_field", "x_m": 3.0, "y_m": 0.0, "volume_m3": 1.0}]
        return [dict(p) for p in self._piles]

    def scoop_state(self) -> bool:
        if self._shovel is not None:
            mass = float(self._shovel.getSoilParticleAggregate().getTotalAggregateMass())
            return mass > 1.0  # kg 阈值
        return self._scoop

    def mark_scoop(self, loaded: bool) -> None:
        # AGX 真值是测出来的（铲斗内颗粒质量），此写接口仅为协议兼容
        self._scoop = bool(loaded)

    # -- 相机（阶段B）
    def grab_frame(self) -> dict[str, Any] | None:
        # TODO(AGX): AGX 渲染一帧 → base64(rgb_jpeg) + base64(depth_f32) + shape
        return None

    # -- 场景清单（探针/验收用；协议 v1 追加命令，向后兼容）
    def inventory(self) -> dict[str, Any]:
        self._require_sim()  # 未加载即问清单 → 明确报错
        joints = []
        for name, group in self._constraints.items():
            force = group[0].getMotor1D().getForceRange()
            joints.append(
                {
                    "name": name,
                    "constraint": [c.getName() for c in group],
                    "type": type(group[0]).__name__,
                    "angle": float(group[0].getAngle()),
                    "range": list(self._ranges.get(name, (0.0, 0.0))),
                    "unit": self._units.get(name, "rad"),
                    "has_motor": group[0].getMotor1D() is not None,
                    "force_range": [float(force.lower()), float(force.upper())],
                    "lock_enabled": bool(group[0].getLock1D().getEnable()),
                }
            )
        terrain = []
        if self._terrain is not None:
            terrain = [{"name": "soil_field", "x_m": 3.0, "y_m": 0.0, "volume_m3": 1.0}]
        machine_name = "excavator365" if self._excavator is not None else "scene"
        return {"machines": [{"name": machine_name, "joints": joints, "terrain": terrain}]}

    # ------------------------------------------------------------------
    # pump 网络（viewer 模式专用）—— agxViewer 会冻结后台线程，所以网络
    # 收发全部放在 StepEventCallback.pre 的每步回调里非阻塞完成。
    # ------------------------------------------------------------------
    def start_pump(self, host: str, port: int) -> None:
        """绑定非阻塞监听；之后每个仿真步调一次 :meth:`pump`。"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        server.setblocking(False)
        self._listener = server
        self._bridge_session = BridgeSession(self)
        print(f"[agx_bridge] pump listening on {host}:{port} (protocol v{PROTOCOL_VERSION})", flush=True)

    def pump(self) -> None:
        """viewer 每个仿真步调用一次（主线程）：接受连接、读请求、回响应、执行停车计划。"""
        if self._listener is None:
            return
        if self._drive_plan is not None:
            hinges, stop_at = self._drive_plan
            if self._sim.getTimeStamp() >= stop_at:
                for h in hinges:
                    h.getMotor1D().setSpeed(0.0)
                self._drive_plan = None
                print("[agx_bridge] drive plan finished (speeds zeroed)", flush=True)
        if self._conn is None:
            try:
                conn, addr = self._listener.accept()
                conn.setblocking(False)
                self._conn = conn
                print(f"[agx_bridge] client {addr} connected (pump)", flush=True)
            except BlockingIOError:
                return
            except OSError:
                return
        try:
            data = self._conn.recv(65536)
            if not data:
                raise ConnectionError("client closed")
            self._read_buffer += data
        except BlockingIOError:
            pass
        except (ConnectionError, OSError):
            self._close_conn()
            return
        while b"\n" in self._read_buffer and self._conn is not None:
            line, self._read_buffer = self._read_buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                response = {"v": PROTOCOL_VERSION, "ok": False, "error": f"bad json: {exc}"}
            else:
                if request.get("cmd") == "bye":
                    self._close_conn()
                    return
                response = self._bridge_session.handle(request)
            self._send_response(response)

    def _send_response(self, response: dict[str, Any]) -> None:
        try:
            self._conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
        except (ConnectionError, OSError):
            self._close_conn()

    def _close_conn(self) -> None:
        try:
            if self._conn is not None:
                self._conn.close()
                print("[agx_bridge] client disconnected (pump)", flush=True)
        finally:
            self._conn = None
            self._read_buffer = b""


class DemoSceneAdapter(AgxSceneAdapter):
    """内存演示机器：即时到位，行为与 MockSimBackend 一致（无需 AGX）。

    覆写全部状态方法为内存字典实现（基类方法面向真实 AGX，会要求先 load）。
    """

    def __init__(self, joint_names: list[str]) -> None:
        super().__init__(joint_names, joint_map={}, scene_path=None)
        self.move_log: list[dict[str, Any]] = []  # 调试/测试断言用

    def load(self) -> None:
        pass  # 内存机器，无事可做

    def read_joints(self) -> dict[str, float]:
        return dict(self._joints)

    def send_joint_targets(self, targets: dict[str, float], timeout_s: float) -> dict[str, float]:
        self._joints.update({str(k): float(v) for k, v in targets.items()})
        return dict(self._joints)

    def navigate_relative(self, dx_m: float, dyaw_rad: float, timeout_s: float) -> dict[str, float]:
        entry = {"dx_m": float(dx_m), "dyaw_rad": float(dyaw_rad)}
        self.move_log.append({"cmd": "navigate_relative", **entry})
        return entry

    def navigate_arc(self, radius_m: float, dyaw_rad: float, timeout_s: float) -> dict[str, float]:
        entry = {"radius_m": float(radius_m), "dyaw_rad": float(dyaw_rad)}
        self.move_log.append({"cmd": "navigate_arc", **entry})
        return entry

    def read_terrain(self) -> list[dict[str, Any]]:
        return [dict(p) for p in self._piles]

    def scoop_state(self) -> bool:
        return self._scoop

    def mark_scoop(self, loaded: bool) -> None:
        self._scoop = bool(loaded)

    def inventory(self) -> dict[str, Any]:
        limits = {"swing": (-180.0, 180.0), "boom": (-45.0, 60.0), "arm": (-135.0, 60.0), "bucket": (-160.0, 40.0)}
        return {
            "machines": [
                {
                    "name": "demo_excavator",
                    "joints": [
                        {
                            "name": name,
                            "constraint": name,
                            "type": "hinge",
                            "angle": self._joints.get(name, 0.0),
                            "range": list(limits.get(name, (-180.0, 180.0))),
                            "unit": "deg",
                            "has_motor": True,
                        }
                        for name in self.joint_names
                    ],
                    "terrain": [dict(p) for p in self._piles],
                }
            ]
        }


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
        if getattr(self._scene, "pump_mode", False):
            # viewer 泵模式：立即返回，客户端轮询到位（不能阻塞主线程）
            self._scene.set_joint_targets(targets)
            return {"joints": self._scene.read_joints(), "targets": targets, "async": True}
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

    def _cmd_inventory(self, _request: dict[str, Any]) -> dict[str, Any]:
        return self._scene.inventory()


def serve(scene: AgxSceneAdapter, host: str, port: int, *, load: bool = True) -> None:
    """Blocking serve loop: load the scene, then accept clients one at a time."""
    if load:
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


def build_scene_adapter(args: argparse.Namespace) -> AgxSceneAdapter | DemoSceneAdapter:
    """Construct the right scene adapter from CLI arguments (shared with the
    agxViewer plugin launcher)."""
    joint_names = [j.strip() for j in args.joints.split(",") if j.strip()]
    joint_map = dict(part.split("=", 1) for part in args.joint_map.split(",") if "=" in part)
    if args.demo:
        return DemoSceneAdapter(joint_names)
    return AgxSceneAdapter(
        joint_names,
        joint_map,
        args.scene,
        mode=getattr(args, "mode", "headless"),
        build_excavator=bool(getattr(args, "excavator", False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AGX ↔ jiuwensymbiosis 仿真桥接服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9700)
    parser.add_argument("--demo", action="store_true", help="内置内存演示机器（无需 AGX，联调用）")
    parser.add_argument("--scene", default=None, help="AGX 场景文件（.agx；自动发现约束）")
    parser.add_argument(
        "--excavator",
        action="store_true",
        help="程序化搭建 AGX 自带挖掘机+沙地场景（镜像官方 excavator_365_terrain 脚本）",
    )
    parser.add_argument(
        "--mode",
        default="headless",
        choices=["headless", "viewer"],
        help="headless: 自持仿真循环；viewer: agxViewer 插件（viewer 步进+渲染）",
    )
    parser.add_argument("--joints", default="swing,boom,arm,bucket", help="关节名（逗号分隔）")
    parser.add_argument("--joint-map", default="", help="关节→AGX约束名映射，如 swing=YawHinge,boom=BoomHinge")
    args = parser.parse_args()

    scene = build_scene_adapter(args)
    serve(scene, args.host, args.port)


if __name__ == "__main__":
    main()
