# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AGX 桥接控制台 —— 浏览器里看关节/地形，手动设目标找姿态。

定位：**标定与排障工具**，不是主控入口（主控是 `run_task.py --query`）。
用途：
  1. 一眼确认"连的是哪台机器"（顶部徽章；demo 假机会变红）——"命令成功但不动"
     最常见的原因就是连错了对端。
  2. 按**原生单位**读关节（AGX 365：回转 rad、三个液压缸 m，不换算、不显示假角度）。
  3. 逐个关节设目标，手动摆到理想姿态后把读数抄进配置的
     `home_joints` / `dig_cycle_tuning`。

用法::

    python monitor_ui.py --host 127.0.0.1 --port 9700
    # 浏览器打开 http://127.0.0.1:8050（--ui-port 可改）

注意：桥接**一次只服务一个客户端**。跑 `run_task.py` 时面板会显示"被占用"，
属预期；等任务跑完再刷新即可。

实现注意：所有页面状态（单位表、输入框引用）都必须是 **main_page 的局部变量** ——
模块级全局会被多次页面加载共享，第二次打开页面就会跳过创建，输入框再也出不来。
"""

from __future__ import annotations

import argparse
import json
import socket
from collections.abc import Iterator
from contextlib import contextmanager

from nicegui import ui

PROTOCOL_VERSION = 1
REFRESH_S = 2.0

# 只有"连哪个桥接"是进程级配置，其余状态一律页面级
_peer = {"host": "127.0.0.1", "port": 9700}


class BridgeError(RuntimeError):
    """连不上 / 被占用 / 对端报错 —— 面板要显示人话，不是 traceback。"""


@contextmanager
def _bridge(timeout: float = 5.0) -> Iterator:
    """一条连接跑多条命令：桥接单客户端，刷新时不要每条命令各开一次连接。"""
    try:
        sock = socket.create_connection((_peer["host"], _peer["port"]), timeout=timeout)
    except TimeoutError as exc:
        raise BridgeError(
            "连接超时——桥接一次只服务一个客户端，是不是 run_task.py 正连着？"
        ) from exc
    except OSError as exc:
        raise BridgeError(
            f"连不上 {_peer['host']}:{_peer['port']}（{exc}）——桥接起了吗？"
            "见 start_agx_bridge.bat"
        ) from exc

    with sock:
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        writer = sock.makefile("w", encoding="utf-8", newline="\n")

        def call(cmd: str, **params) -> dict:
            writer.write(json.dumps({"v": PROTOCOL_VERSION, "cmd": cmd, **params}) + "\n")
            writer.flush()
            try:
                line = reader.readline()
            except TimeoutError as exc:
                raise BridgeError(
                    "读超时——桥接不应答。仿真暂停时泵不跑（看桥接控制台的提示）"
                ) from exc
            if not line:
                raise BridgeError("桥接关闭了连接")
            response = json.loads(line)
            if not response.get("ok"):
                raise BridgeError(str(response.get("error", "桥接返回失败")))
            return response

        yield call


@ui.page("/")
async def main_page() -> None:
    ui.query("body").classes("bg-slate-100")
    # ── 页面级状态（不要提到模块级）────────────────────────────────────
    units: dict[str, str] = {}
    inputs: dict[str, ui.number] = {}

    with ui.column().classes("w-full max-w-3xl mx-auto gap-3 p-4"):
        # 顶栏：标题 + 对端徽章
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("precision_manufacturing").classes("text-2xl text-primary")
                ui.label("AGX 挖掘机控制台").classes("text-xl font-semibold")
            badge = ui.badge("连接中…", color="grey").classes("text-sm px-3 py-1")
        peer_label = ui.label("").classes("text-xs text-slate-500 font-mono")

        alert_row = ui.row().classes("w-full")

        # 关节表
        with ui.card().classes("w-full p-4 shadow-sm rounded-lg"):
            with ui.row().classes("items-center gap-2 mb-1"):
                ui.icon("straighten").classes("text-primary")
                ui.label("关节状态").classes("text-base font-semibold")
                ui.label("原生单位（回转 rad / 液压缸 m）").classes(
                    "text-xs text-slate-400 ml-1"
                )
            joints_table = ui.table(
                columns=[
                    {"name": "joint", "label": "关节", "field": "joint", "align": "left"},
                    {"name": "value", "label": "当前值", "field": "value", "align": "right"},
                    {"name": "unit", "label": "单位", "field": "unit", "align": "left"},
                ],
                rows=[],
                row_key="joint",
            ).classes("w-full").props("flat dense")

        # 手动设目标
        with ui.card().classes("w-full p-4 shadow-sm rounded-lg"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("tune").classes("text-primary")
                ui.label("手动设目标").classes("text-base font-semibold")
            ui.label(
                "留空 = 不动该关节。摆到满意后把读数抄进配置的 home_joints / dig_cycle_tuning。"
            ).classes("text-xs text-slate-400")
            inputs_row = ui.row().classes("gap-3 items-end mt-1")

            def send_targets() -> None:
                targets = {
                    name: float(box.value)
                    for name, box in inputs.items()
                    if box.value is not None
                }
                if not targets:
                    ui.notify("没有填任何目标", type="warning")
                    return
                try:
                    with _bridge() as call:
                        call("move_joints", targets=targets, timeout_s=30.0)
                except BridgeError as exc:
                    ui.notify(str(exc), type="negative")
                else:
                    ui.notify(f"已发送：{targets}", type="positive")

            def fill_from_current() -> None:
                try:
                    with _bridge() as call:
                        joints = call("joints").get("joints") or {}
                except BridgeError as exc:
                    ui.notify(str(exc), type="negative")
                    return
                for name, value in joints.items():
                    if name in inputs:
                        inputs[name].value = round(float(value), 4)
                ui.notify("已填入当前值", type="positive")

            with ui.row().classes("gap-2 mt-1"):
                ui.button("发送", on_click=send_targets, icon="send").props("color=primary")
                ui.button("填入当前值", on_click=fill_from_current, icon="content_paste")

        # 地形
        with ui.card().classes("w-full p-4 shadow-sm rounded-lg"):
            with ui.row().classes("items-center gap-2 mb-1"):
                ui.icon("landscape").classes("text-primary")
                ui.label("地形真值").classes("text-base font-semibold")
            terrain_label = ui.label("加载中…").classes("text-sm whitespace-pre-line")

    def build_inputs(names: list[str]) -> None:
        """首次拿到关节名后建输入框；必须落在控制卡里（否则会跑到页面根部）。"""
        if inputs:
            return
        with inputs_row:
            for name in names:
                inputs[name] = ui.number(
                    label=f"{name} ({units.get(name, '?')})", value=None, step=0.01
                ).classes("w-36")

    def set_badge(text: str, color: str) -> None:
        badge.set_text(text)
        badge.props(f"color={color}")

    def show_error(message: str) -> None:
        alert_row.clear()
        with alert_row:
            with ui.row().classes(
                "w-full items-center gap-2 bg-red-50 text-red-700 rounded-lg px-3 py-2"
            ):
                ui.icon("error_outline")
                ui.label(message).classes("text-sm")

    async def refresh() -> None:
        try:
            with _bridge() as call:
                machines = call("inventory").get("machines") or []
                joints = call("joints").get("joints") or {}
                piles = call("terrain").get("piles") or []
        except BridgeError as exc:
            set_badge("● 未连接", "grey")
            show_error(str(exc))
            return

        machine = machines[0] if machines else {}
        name = str(machine.get("name", "未知"))
        units.update(
            {str(j.get("name")): str(j.get("unit", "?")) for j in machine.get("joints") or []}
        )
        set_badge(f"● {name}", "red-6" if name.startswith("demo") else "green-6")
        peer_label.set_text(
            f"{_peer['host']}:{_peer['port']} · 每 {REFRESH_S:.0f} s 自动刷新"
        )
        if name.startswith("demo"):
            show_error("对端是内存演示机（demo）——命令不会驱动任何 AGX 场景")
        else:
            alert_row.clear()

        build_inputs(list(joints))
        joints_table.rows = [
            {"joint": n, "value": f"{float(v):.4f}", "unit": units.get(n, "")}
            for n, v in joints.items()
        ]
        joints_table.update()
        terrain_label.set_text(
            "\n".join(
                f"• {p['name']}  ({p['x_m']} m, {p['y_m']} m)  {p['volume_m3']} m³"
                for p in piles
            )
            or "对端未报告料堆（真机平整沙地场景正常：土壤全程都在）"
        )

    ui.timer(REFRESH_S, refresh)
    await refresh()


def main() -> None:
    parser = argparse.ArgumentParser(description="AGX 桥接控制台（标定/排障用）")
    parser.add_argument("--host", default="127.0.0.1", help="桥接地址")
    parser.add_argument("--port", type=int, default=9700, help="桥接端口")
    parser.add_argument("--ui-port", type=int, default=8050, help="本面板端口")
    args = parser.parse_args()
    _peer.update(host=args.host, port=args.port)
    ui.run(port=args.ui_port, title="AGX 挖掘机控制台", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
