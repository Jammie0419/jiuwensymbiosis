# 215F（SWE215F / describe_215i）接入

> 类别：记录。2026-10-20 重组，2026-10-22 补 3D IK / 运动学底盘。
> **目标：像 365 一样操控 215F**——同一条桥接、同一份 YAML 结构、同一个 `dig`
> 动作。215F 额外支持大脑给 3D 铲尖位姿、IK 执行、整机运动学移动；自由空间段
> 走平滑轨迹播放（Phase 4），接触段仍闭环；60 Hz 采集不在当前路径上。

## 和 365 的对应关系

| 环节 | 365 | 215F |
| --- | --- | --- |
| 启动 | `start_agx_bridge.bat` | `start_agx_bridge_215f.bat` |
| 场景来源 | AGX 自带 `Excavator365` | 交付包 `Excavator215FEnv`（Python 环境类） |
| 桥接入口 | `agx_viewer_bridge.agxPy`（`--excavator`） | `agx_viewer_bridge_215f.agxPy`（`build()` + `attach_machine`） |
| 配置 | `agx_excavator.local.yaml` | `configs/agx_excavator/agx_excavator.215f.yaml` |
| 探针 | `agx_scene_probe.py --bridge` | 同左 |
| 动作 | `dig` 六段关键帧 | `dig_mode: ik`（3D 铲尖位姿 + 标定 IK）；关键帧保留为回落 |

## 机器事实

- 四关节，全部 **rad**：
  - `swing = base_to_shangche`（±π，配置收窄）
  - `boom = dongbi_joint`（±90°）
  - `arm = dougan_joint`（0..127.099°）
  - `bucket = chandou_joint`（0..186.102°）
- 交付模型是 fixed-base 简化：5 刚体、4 关节，root 由 LockJoint 固定，**没有履带
  关节**。我们在场景工厂里释放 root 锁、把 `base_link` 设为 `KINEMATICS`，所以
  `has_base: true`、`motion.base` 可用——这是**整机运动学移动**，履带不转、没有
  履带-土壤接触，也不是履带物理。
- 力矩上限：swing 80 / boom 400 / arm 250 / bucket 150 kN·m。桥接的位置锁继承
  `Motor1D` 力限，URDF 的占位 effort 撑不住伸出的工作装置，必须显式抬高。
- 载料真值：Shovel aggregate `getTotalAggregateMass()`，读不到时回落
  `_get_shovel_dynamic_mass()`；阈值 50 kg。
- 土堆真值：`soil_pile_215f @ (5.0, 2.0), 9.98 m³`（顶 1.95 m，x 3.4–6.6，
  y 0.4–3.6，坡 0.5）。
- 场景副本：用 `E:\AGX-2.42.2.1\deploy\215f\215F_AGX_MINIMAL`。`sh` 目录里的
  原件是 Linux / AGX 2.39 版，2.42 上跑不起来。

## 关键帧来源

小脑组参考轨迹 `agx_task_215f.json` 的 `reference_q_rad`，按

```
q_AGX = sign * q_policy - offset_deg
sign   = +1
offset = 0 / +61.8273 / -154.3403 / +216.6141 度
```

换算后抽样：`ready=f20`、`dig=f32`、`curl=f36`；`dump` 用 215F 实测卸料位形。
六个关键帧与 365 的 `work.execute_dig_cycle` 完全同构，`work.py` 不需要机型分支。

## 3D IK 挖掘（`dig_mode: ik`）

- 大脑给两个 3D 铲尖位姿（**当前基座系**，米，REP-103）：`dig_pose=(x,y,z)`、
  `dump_pose=(x,y,z)`；`dig_pitch_deg`/`dump_pitch_deg` 可选，yaw 由 `atan2(y,x)`。
- **配置里没有写死的始末点**：`terrain_piles` 只是 mock 后端/文档的示例；remote
  后端的 `get_terrain` 每次读场景的动态最高点，dig/dump 坐标由大脑按任务规划后
  作为 `dig` 参数传入。`agx_215f_dig_smoke.py` 的默认点只是验收脚本的测试默认
  值，可用 `--dig-x/--dig-y/--dig-z/--dump-x/--dump-y/--dump-z` 覆盖。
- 不给 `dig_z_m` 时取该点地形表面高度；不给 `dump_z_m` 时取该点地形高度 +
  `dump_clearance_m`（默认 3 m）。
- **铲斗方向**（215F 实测）：原生 `bucket=0.0` 是往机身**内侧**翻，`1.5 rad`
  才是往外翻。入土/拖切用张开角 `dig_bucket_open_rad=1.5`；拖切完往内合到
  `dig_bucket_curl_rad=0.1` 收斗；卸料时在卸点从 0.1 往外翻到
  `dump_bucket_open_rad`（默认 2.0；1.5 也能倒空，2.0 翻得更开）。
- **卸点是“收斗状态（0.1）下铲尖所在的位置”**：先带着料到达，再在原地往外
  翻斗。低/近的卸点（例如 `(3,0,0)`）在收斗状态下 3-DOF 手臂到不了，`dig` 会
  直接报错并提示抬高/外移。自然卸点约在半径 4.5–5.5 m、z 3–4 m，例如挖点
  `(5,2,1.95)` 配卸点 `(5,-1,3.5)`。
- 起步先用 ready 关键帧 + 挖点方位就位，避免从上一条命令留下的任意姿态陷入
  IK 局部极小。
- 执行序列：approach → entry → press（z 再下咬 `bite_m`）→ 3 点拖切（斗张开）
  → 原地收斗（1.5→0.1）→ 用 carry 姿态举升 → dump_approach → dump（斗合着）
  → 原地外翻（0.1→1.5）→ 驻留；每点闭环读 `snapshot().tip.policy_m` 纠偏。
  接触段（press/drag）用 `contact_tolerance_m`（默认 25 cm）——土硬时压不到
  命令深度，按实际入土继续拖切；挖点 entry 仍按 `ik_tolerance_m`（215F 实测
  10 cm），卸点按 `dump_tolerance_m`（默认 25 cm）——远侧卸点摆转接近 180°
  时模型残差可达 ~20 cm，按 10 cm 会在开斗前抛错。
- 收斗后检查 `scoop_state()`；没装上料会直接失败，不再把空斗当成功。
- 超差、不可达、或卸完 `scoop_state()` 仍为真 → `ok:false + error`；大脑据此
  移动底盘或换点。自由空间段已走平滑轨迹播放，一斗约 44 s；接触段
  （press/drag）仍逐点闭环。

## 平滑轨迹播放（Phase 4）

- `trajectory.py`：C¹ Hermite/PCHIP 关节空间样条，带限速、限位和回转关节解缠，
  输出 60 Hz 帧序列。
- 桥接：新增 `play_trajectory` / `player_state` / `stop_trajectory`。viewer 模式
  每个仿真步由 `adapter.tick()` 推一帧（客户端轮询到 done），headless 模式由桥接
  阻塞步进；`move_joints` 会先停掉播放器。
- backend/driver：`play_trajectory(frames, rate_hz)` 在下发前校验列数、有限值和
  关节限位；viewer 桥由客户端轮询播放状态。
- IK 工作循环：ready→approach、收斗后的 carry、carry→dump_approach 三段走整段
  平滑轨迹；entry/press/drag 仍闭环（接触段不能被开环轨迹替代）。一斗从
  62–69 s 降到约 44 s，自由空间段不再“一格一格”。

## 用法

```bat
rem 1) 起窗口 + 桥接（和 365 一样，一个 bat）
start_agx_bridge_215f.bat

rem 2) 控制端核对
python extensions\jiuwen_agx\scripts\control\agx_scene_probe.py --bridge --port 9700

rem 3) 真实任务（key 配好后）
python examples\run_task.py ^
  --config extensions\jiuwen_agx\configs\agx_excavator\agx_excavator.215f.yaml ^
  --query "把土堆挖一斗倒旁边"

rem Windows 上 run_task 被 fcntl 卡住时，用同一条后端/同一个 dig 动作的验收脚本
python extensions\jiuwen_agx\scripts\control\agx_215f_dig_smoke.py
```

探针应看到：`machine_name = excavator215f`、4 个关节、单位 rad、载料约 0、
地形真值非空。

## 验收

- `pytest extensions/jiuwen_agx/tests -q`（无 AGX）全绿；365 路径无回归。
- 一斗 `dig`：装料质量跨过 50 kg，卸料后回落到 50 kg 以下。
- 旧实验（带四点拖切）实测装料峰值 **362.2 kg** → 卸料后 **7.7 kg**；本次用
  标准六段帧，实际数字以 `agx_215f_dig_smoke.py` 打印为准。

## 三个坑

1. **力矩**：不抬高 `Motor1D` / `Lock1D` 的力限，带载动臂会被土拖住甚至下坠。
2. **单位**：215F 全 rad。`joint_units: "rad"`、`swing_unit: "rad"`，关键帧键名
   仍沿用历史的 `*_deg`，但值是弧度。
3. **整机移动不是履带物理**：`has_base: true`，但底盘是 `KINEMATICS` 刚体，
   用 `navigate_relative`/`rotate_base` 直接设 transform；没有履带关节、没有
   履带-土壤接触、移动不做碰撞检测。移动前先抬臂/回 home、不要带载，按
   `base_step_limits` 拆成小步。真实履带物理需要新的模型交付。
