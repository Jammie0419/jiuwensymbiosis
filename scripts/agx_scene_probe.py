# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""AGX 场景探针 —— 确定 jiuwensymbiosis 适配器能否控制组里的 AGX 模型。

三种模式：

  --direct  （在 Windows 的 AGX Python 里跑，需要 import agx）
      加载场景，遍历所有铰链/棱柱约束与地形对象，输出 JSON 报告并生成
      jiuwensymbiosis 的建议配置片段。这是"模型可识别"的判据。

  --bridge  （在 Linux 控制端跑，不需要 AGX）
      连接正在运行的 agx_bridge_server.py，发 inventory 命令拿场景清单。
      这是"控制链路通"的判据。

  --check-config  （Linux，不需要 AGX）
      拿探针生成的配置片段反向核对：关节名是否齐、限位是否有效、
      关键帧微调是否越限。

仅用标准库；Windows 控制台（GBK）安全——输出不含 emoji，全部 flush。
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from typing import Any

PROTOCOL_VERSION = 1

# dig 工作循环需要的四个关节（与 adapters/agx_excavator/work.py REQUIRED_JOINTS 一致）
EXCAVATOR_JOINTS = ("swing", "boom", "arm", "bucket")

# AGX 内部是 SI（弧度）；config 用 deg 时探针会提示转换
RAD_TO_DEG = 57.29577951308232


def _out(msg: str = "") -> None:
    """GBK-console-safe print: always flush, no non-ASCII punctuation issues."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("gbk", errors="replace").decode("gbk"), flush=True)


# ============================================================================
# --bridge 模式：连桥接服务，验证链路 + 拿场景清单
# ============================================================================
def probe_bridge(host: str, port: int, timeout_s: float) -> int:
    _out(f"连接桥接服务 {host}:{port} ...")
    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
    except OSError as exc:
        _out(f"[FAIL] 连不上桥接服务: {exc}")
        _out("检查: Windows 侧是否已启动 agx_bridge_server.py；防火墙是否放行端口。")
        return 1
    with sock:
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        writer = sock.makefile("w", encoding="utf-8", newline="\n")

        def call(cmd: dict[str, Any]) -> dict[str, Any]:
            writer.write(json.dumps({"v": PROTOCOL_VERSION, **cmd}) + "\n")
            writer.flush()
            line = reader.readline()
            if not line:
                raise ConnectionError("桥接服务关闭了连接")
            return json.loads(line)

        try:
            pong = call({"cmd": "ping"})
        except Exception as exc:
            _out(f"[FAIL] 通信失败: {exc}")
            return 1
        if int(pong.get("v", 0)) != PROTOCOL_VERSION:
            _out(f"[FAIL] 协议版本不匹配: 对端 v{pong.get('v')}, 本工具 v{PROTOCOL_VERSION}")
            return 1
        _out("[OK] 链路通（协议 v1）")

        resp = call({"cmd": "inventory"})
        if not resp.get("ok"):
            _out(f"[FAIL] inventory 被拒绝: {resp.get('error')!r}")
            _out("通常表示桥接服务的 AgxSceneAdapter.inventory 尚未实现（TODO(AGX)）。")
            return 1
        _out("[OK] 场景清单已取回")
        _out("")
        report = resp
        _render_report(report)
        _out("")
        _render_suggested_config(report)
        _out("")
        _save_report(report, "agx_scene_report.json")
    return 0


# ============================================================================
# --direct 模式：在 AGX Python 里跑，遍历场景
# ============================================================================
def probe_direct(scene_path: str) -> int:
    # The probe runs on AGX's bundled Windows Python, which may be older than
    # this repo's floor — the guard is deliberate, not a py311 relic.
    if sys.version_info < (3, 8):  # noqa: UP036
        _out(f"[FAIL] AGX 自带 Python 版本过低: {sys.version}（需要 3.8+）")
        return 1
    try:
        import agx  # noqa: F401
    except ImportError:
        _out("[FAIL] import agx 失败——本脚本必须在 AGX 自带的 Python 环境里运行。")
        _out("提示: 用 AGX 安装目录下的 python.exe 运行，或先运行 AGX 的环境初始化脚本。")
        return 1

    _out(f"AGX Python OK ({sys.version.split()[0]})，加载场景: {scene_path}")
    # TODO(AGX): 下面三段按 AGX 实际 API 填——结构与 --bridge 的 inventory 输出一致，
    # 同事只需把遍历结果装进同样的 dict 形状，两侧工具即共用渲染/生成逻辑。
    #
    #   1) 加载场景: agx.loadScene(scene_path) 或 agxScene = agx.Scene(scene_path)
    #   2) 遍历约束: sim.getConstraints() → 对每个 agx.Hinge / agx.Primitive
    #                记录 name / type / angle / range / hasMotor
    #   3) 遍历地形: sim.getTerrains() → name / 质心 / 包围盒
    _out("[FAIL] --direct 模式的 AGX 遍历代码尚未实现（TODO(AGX) 块）。")
    _out("替代路径: 让桥接服务的 AgxSceneAdapter.inventory 参考本函数注释实现，")
    _out("然后用 --bridge 模式远程取清单——效果相同且无需在 Windows 手动传文件。")
    return 1


# ============================================================================
# --check-config 模式：反向核对配置
# ============================================================================
def check_config(config_path: str) -> int:
    try:
        import yaml  # 探针 --check-config 在 Linux 仓库环境跑，可用 pyyaml
    except ImportError:
        yaml = None  # type: ignore[assignment]
    with open(config_path, encoding="utf-8") as f:
        text = f.read()
    data = yaml.safe_load(text) if yaml else _mini_yaml(text)
    ll = ((data.get("env") or {}).get("cfg") or {}).get("low_level") or {}
    names = list(ll.get("joint_names") or [])
    limits = ll.get("joint_limits") or {}
    tuning = ll.get("dig_cycle_tuning") or {}

    problems: list[str] = []
    missing = [j for j in EXCAVATOR_JOINTS if j not in names]
    if missing:
        problems.append(f"joint_names 缺少挖掘循环必需的关节: {missing}")
    for j in EXCAVATOR_JOINTS:
        if j in names and j not in limits:
            problems.append(f"joint_limits 缺少 {j!r}（driver 无法做限位校验，会拒绝执行）")
    # 关键帧越限检查（与 driver.move_joints_blocking 同规则）
    default_keys = {  # 与 work.DEFAULT_DIG_TUNING 对应的默认关键帧
        "ready_boom_deg": ("boom", 10.0),
        "ready_arm_deg": ("arm", -25.0),
        "ready_bucket_deg": ("bucket", 20.0),
        "dig_boom_deg": ("boom", -35.0),
        "dig_arm_deg": ("arm", 55.0),
        "dig_bucket_deg": ("bucket", -70.0),
        "curl_boom_deg": ("boom", 25.0),
        "curl_arm_deg": ("arm", -55.0),
        "curl_bucket_deg": ("bucket", 40.0),
        "dump_boom_deg": ("boom", 15.0),
        "dump_arm_deg": ("arm", -35.0),
        "dump_bucket_deg": ("bucket", -120.0),
    }
    for key, (joint, default_value) in default_keys.items():
        value = tuning.get(key, default_value)
        lo, hi = limits.get(joint, (float("-inf"), float("inf")))
        if not (lo <= value <= hi):
            problems.append(f"关键帧 {key}={value} 超出 {joint} 限位 [{lo}, {hi}]（执行时会被拒绝）")
    offset = tuning.get("swing_offset_deg", 0.0)
    lo, hi = limits.get("swing", (-180.0, 180.0))
    if lo > -180.0 or hi < 180.0:
        _out(f"[提示] swing 限位 [{lo}, {hi}] 非 ±180 全行程：摆转归一化到 [-180,180)，")
        _out("       若模型只允许部分回转，超出可达面的挖掘点会执行失败——属预期保护。")
    _ = offset

    if problems:
        _out(f"[FAIL] {config_path} 有 {len(problems)} 处问题:")
        for p in problems:
            _out(f"  - {p}")
        return 1
    _out(f"[OK] {config_path}: 关节/限位/关键帧全部自洽，可以执行 dig。")
    return 0


def _mini_yaml(text: str) -> dict[str, Any]:
    """Fallback when pyyaml is missing: only handles the flat low_level keys we need."""
    data: dict[str, Any] = {"env": {"cfg": {"low_level": {}}}}
    ll = data["env"]["cfg"]["low_level"]
    section = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if stripped.startswith("joint_names:"):
            section = None
            ll["joint_names"] = [x.strip() for x in stripped.split("[", 1)[1].rstrip("]").split(",") if x.strip()]
        elif stripped.startswith("joint_limits:") or stripped.startswith("dig_cycle_tuning:"):
            section = "joint_limits" if "limits" in stripped else "tuning"
            ll[section] = {}
        elif section and ":" in stripped and indent >= 8:
            key, _, val = stripped.partition(":")
            try:
                ll[section][key.strip()] = float(val.strip())
            except ValueError:
                pass
        elif indent >= 6 and section is None and ":" in stripped:
            key, _, val = stripped.partition(":")
            ll[key.strip()] = val.strip().strip('"')
    return data


# ============================================================================
# 渲染与保存
# ============================================================================
def _render_report(report: dict[str, Any]) -> None:
    for machine in report.get("machines", []):
        _out(f"机器: {machine.get('name')}")
        for j in machine.get("joints", []):
            rng = j.get("range", [None, None])
            _out(
                f"  关节 {j.get('name'):<10} 约束={j.get('constraint'):<20} "
                f"当前={j.get('angle')} 范围=[{rng[0]}, {rng[1]}] "
                f"单位={j.get('unit')} 电机={'有' if j.get('has_motor') else '无'}"
            )
        for pile in machine.get("terrain", []):
            _out(f"  料堆 {pile.get('name')}: ({pile.get('x_m')}, {pile.get('y_m')}) m, {pile.get('volume_m3')} m3")


def _render_suggested_config(report: dict[str, Any]) -> None:
    _out("# ---- 建议配置片段（复制进 configs/agx_excavator/agx_excavator.local.yaml 的 low_level 段）----")
    for machine in report.get("machines", []):
        joints = machine.get("joints", [])
        if not joints:
            continue
        names = [j.get("name") for j in joints]
        units = {j.get("unit") for j in joints}
        _out("# 关节名（探针实测）")
        _out(f"joint_names: {names}")
        if len(units) == 1:
            unit = next(iter(units))
            _out(f'joint_units: "{unit}"')
        else:
            # 混合单位（挖掘机：回转=rad、液压缸=m）——joint_units 留空表示未声明，
            # 每个关节的原生单位见注释；move_joint 参数按各关节原生单位给值。
            _out("joint_units: null   # 混合单位：见下注释")
        _out("# 限位（探针实测行程，原生单位）")
        _out("joint_limits:")
        for j in joints:
            rng = j.get("range", [0.0, 0.0])
            lo, hi = float(rng[0]), float(rng[1])
            if lo == float("-inf") or hi == float("inf"):
                lo, hi = -180.0, 180.0  # 全行程回转
            _out(f"  {j.get('name')}: [{lo}, {hi}]   # {j.get('unit')}")
        _out("# 约束名映射（桥接 --joint-map / 诊断用）")
        for j in joints:
            _out(f"#   {j.get('name')}: {j.get('constraint')}")
        _out("# 回转零位校准：让模型摆向基座 +X 方向，读 swing 当前角，填其负值")
        _out("# dig_cycle_tuning:")
        _out("#   swing_offset_deg: <探针实测>")


def _save_report(report: dict[str, Any], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _out(f"完整报告已保存: {filename}")


# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="AGX 场景探针：验证 jiuwensymbiosis 能否控制组里的模型")
    parser.add_argument("--bridge", action="store_true", help="连桥接服务取场景清单（Linux 控制端）")
    parser.add_argument("--direct", metavar="SCENE", help="在本机 AGX Python 里直接加载场景盘点（Windows）")
    parser.add_argument("--check-config", metavar="YAML", help="核对一份适配器 YAML 配置是否自洽可执行")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9700)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    modes = [args.bridge, bool(args.direct), bool(args.check_config)].count(True)
    if modes != 1:
        parser.print_help()
        return 2
    if args.bridge:
        return probe_bridge(args.host, args.port, args.timeout)
    if args.direct:
        return probe_direct(args.direct)
    return check_config(args.check_config)


if __name__ == "__main__":
    sys.exit(main())
