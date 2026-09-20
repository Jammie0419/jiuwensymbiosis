# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AGX 桥接状态监视 —— 浏览器里看关节/地形，并可手动设目标。

用途有二：排查"链路到底通没通"（顶部直接标出对端是哪台机器），以及
**手动找姿态**（按原生单位逐个关节设目标，读到满意的一组数就抄进配置的
home_joints / dig_cycle_tuning）。

用法::

    python monitor_ui.py --host 127.0.0.1 --port 9700
    # 浏览器打开 http://127.0.0.1:8050（--ui-port 可改）

注意单位：AGX 365 挖掘机是混合单位——回转（swing）是弧度，三个液压缸
（boom/arm/bucket）是米。面板按对端 inventory 报的单位显示，不换算。
"""

from __future__ import annotations

import argparse
import json
import socket

from nicegui import ui

PROTOCOL_VERSION = 1

# 运行期状态（单进程内的一份全局；页面刷新不丢）
_peer = {"host": "127.0.0.1", "port": 9700}
_inputs: dict[str, ui.number] = {}
_units: dict[str, str] = {}


def send_command(cmd: str, params: dict | None = None, timeout: float = 5.0) -> dict:
    """One request = one connection = one JSON line (bridge protocol v1).

    失败信息写成人话：桥接一次只服务一个客户端，跑着 run_task.py 时这里必然超时。
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((_peer["host"], _peer["port"]))
            request = {"v": PROTOCOL_VERSION, "cmd": cmd}
            if params:
                request.update(params)
            sock.sendall((json.dumps(request) + "\n").encode("utf-8"))
            response = sock.recv(65536).decode("utf-8")
        return json.loads(response.strip())
    except TimeoutError:
        return {
            "ok": False,
            "error": "超时——桥接一次只服务一个客户端，是不是 run_task.py 正连着？",
        }
    except OSError as exc:
        return {"ok": False, "error": f"连不上（{exc}）——桥接起了吗？见 start_agx_bridge.bat"}
    except Exception as exc:  # noqa: BLE001 - 面板永不因断连崩掉
        return {"ok": False, "error": str(exc)}


def _peer_machine_name() -> tuple[str, bool]:
    """(机器名, 是否演示假机)。bridge 不支持 inventory 时返回 ("未知", False)。"""
    resp = send_command("inventory")
    if not resp.get("ok"):
        return "未知（bridge 不支持 inventory）", False
    machines = resp.get("machines") or []
    name = str((machines[0] if machines else {}).get("name", "未知"))
    return name, name.startswith("demo")


@ui.page("/")
async def main_page() -> None:
    ui.label("AGX 挖掘机状态").classes("text-3xl font-bold mb-2")
    peer_label = ui.label("对端：连接中…").classes("text-lg")
    warn_row = ui.row()

    with ui.card().classes("w-full max-w-2xl"):
        ui.label("关节").classes("text-xl font-bold mb-2")
        joints_md = ui.markdown("加载中…")

    with ui.card().classes("w-full max-w-2xl mt-4"):
        ui.label("手动设目标（原生单位）").classes("text-xl font-bold mb-2")
        ui.label("回转 rad；boom/arm/bucket 为米。留空 = 不动该关节。").classes("text-sm opacity-70")
        input_row = ui.row().classes("gap-4")

    with ui.card().classes("w-full max-w-2xl mt-4"):
        ui.label("地形真值").classes("text-xl font-bold mb-2")
        terrain_md = ui.markdown("加载中…")

    def _build_inputs(names: list[str]) -> None:
        """首次拿到关节名后建输入框（每个关节一个）。"""
        if _inputs:
            return
        for name in names:
            unit = _units.get(name, "?")
            number = ui.number(label=f"{name} ({unit})", value=None, step=0.01).classes("w-32")
            _inputs[name] = number

    async def send_targets() -> None:
        targets = {
            name: float(box.value)
            for name, box in _inputs.items()
            if box.value is not None
        }
        if not targets:
            ui.notify("没有填任何目标", type="warning")
            return
        result = send_command("move_joints", {"targets": targets, "timeout_s": 30.0})
        if result.get("ok"):
            ui.notify(f"已发送：{targets}")
        else:
            ui.notify(f"失败：{result.get('error')}", type="negative")

    def fill_from_current() -> None:
        """把当前值填进输入框——找姿态时从现值微调最省事。"""
        result = send_command("joints")
        if not result.get("ok"):
            ui.notify(f"读关节失败：{result.get('error')}", type="negative")
            return
        for name, value in (result.get("joints") or {}).items():
            if name in _inputs:
                _inputs[name].value = round(float(value), 4)
        ui.notify("已填入当前值")

    async def refresh() -> None:
        # 对端身份：先让人看清"连的是不是真 AGX"
        name, is_demo = _peer_machine_name()
        peer_label.set_text(f"对端：{name} @ {_peer['host']}:{_peer['port']}")
        warn_row.clear()
        if is_demo:
            with warn_row:
                ui.label("⚠ 对端是内存演示机（demo），命令不会驱动任何 AGX 场景").classes(
                    "text-red-500 font-bold"
                )

        joints_result = send_command("joints")
        if joints_result.get("ok"):
            joints = joints_result.get("joints") or {}
            _build_inputs(list(joints))
            lines = [
                f"- **{name}**: {float(value):.4f} {_units.get(name, '')}"
                for name, value in joints.items()
            ]
            joints_md.set_content("\n".join(lines) if lines else "无数据")
        else:
            joints_md.set_content(f"❌ 读取失败：{joints_result.get('error')}")

        terrain_result = send_command("terrain")
        if terrain_result.get("ok"):
            piles = terrain_result.get("piles") or []
            if piles:
                terrain_md.set_content(
                    "\n".join(
                        f"- **{p['name']}**: ({p['x_m']} m, {p['y_m']} m), {p['volume_m3']} m³"
                        for p in piles
                    )
                )
            else:
                terrain_md.set_content("对端未报告料堆（真机场景下正常——土壤全程都在）")
        else:
            terrain_md.set_content(f"❌ 读取失败：{terrain_result.get('error')}")

    # units 来自 inventory（真机是 rad/m；demo 假机是 deg）
    inv = send_command("inventory")
    for machine in (inv.get("machines") or []) if inv.get("ok") else []:
        for joint in machine.get("joints") or []:
            _units[str(joint.get("name"))] = str(joint.get("unit", "?"))

    ui.button("发送", on_click=send_targets).props("color=primary")
    ui.button("填入当前值", on_click=fill_from_current)
    ui.button("立即刷新", on_click=refresh)
    ui.timer(2.0, refresh)
    await refresh()


def main() -> None:
    parser = argparse.ArgumentParser(description="AGX 桥接状态监视面板")
    parser.add_argument("--host", default="127.0.0.1", help="桥接地址")
    parser.add_argument("--port", type=int, default=9700, help="桥接端口")
    parser.add_argument("--ui-port", type=int, default=8050, help="本面板端口")
    args = parser.parse_args()
    _peer.update(host=args.host, port=args.port)
    ui.run(port=args.ui_port, title="AGX Excavator Monitor", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
