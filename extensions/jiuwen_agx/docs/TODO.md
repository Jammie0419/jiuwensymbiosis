# 待办清单 — license 到位后按此推进

> 适用范围已按实际使用方式过滤：**服务器跑桥接 → Windows 浏览器观看 → LLM
> 控制挖掘机/组内模型**。不适配这项工作流的条目已剔除（见文末"已排除项"）。

## A. license 后立即要做的调优（已实现、未验证）

> 放好 `~/.config/agx/agx.lic` → 重启 `agx_viewer_stream.sh` → 按 A1→A5 顺序做。
> 每条的验收标准都写明了，做完一项勾一项。

- [ ] **A1. 确认执行器解锁**
  重启 viewer 后终端不再刷 `License not valid`；浏览器里按 `a`（键盘模式）铲斗有微动。
- [ ] **A2. 伺服参数标定**（`scripts/agx_bridge_server.py` 的 `set_joint_targets`）
  现有阻尼 = 官方 logInterpolate 公式 + 距离自适应，但从未在有力学负载下跑过。
  逐关节试 `move_joint`： swing/boom/arm/bucket 各走全行程，观察是否到位、
  有无超调震荡。不合适就调 `damping = max(damping, 2/60)` 的下限和
  `distance * 0.1` 的插值斜率。
  **验收**：每关节 ±3.6°(0.063 rad)/±1 cm 容差内 30 秒内稳定到位。
- [ ] **A3. 回转零位标定（swing_offset）**
  让模型上车摆到基座 +X 方向，探针读 swing 当前角，取其负值填进
  `configs/agx_excavator/agx_excavator.local.yaml` 的
  `dig_cycle_tuning.swing_offset`（单位 rad）。
  **验收**：`dig` 后挖点方向与画面一致。
- [ ] **A4. 挖掘关键帧调优（dig_cycle_tuning，单位是米）**
  现值是按缸程推算的占位值。按"对准→就位→下铲→收斗→摆转→卸料"逐步观察，
  重点：`dig_arm_deg`（斗杆伸出量，决定吃土深度）、`curl_*`（收斗是否兜住土）、
  `dump_bucket_deg`（倒干净）。改完即生效（config 每次运行重新读）。
  **验收**：一斗 `dig` 后 `scoop_state` 为 true（铲斗颗粒质量 >1 kg），
  倒点处可见土堆。
- [ ] **A5. 可达包络实测修正（reach_min_m / reach_max_m）**
  现值 1~8 m 是估算。在画面里把铲齿摆到最近/最远能挖到的位置，读探针
  swing/boom/arm 角度反算半径，更新 local.yaml。
  **验收**：画面里明显可挖的点，`dig` 不报 "outside the reachable annulus"；
  明显够不着的点，`dig` 正确拒绝。

## B. 缺失能力（适配工作流，按影响排序）

- [ ] **B1. 机器位置感知（影响：高）**
  现状：`navigate_relative` 是开环（速度×时间），底盘走完后的实际位置框架
  不知道——远距离作业后再挖，坐标系全靠推算，会漂。
  做法：桥接增加 `read_pose()`（从 AGX 根刚体读世界位姿，减去初始位姿得
  相对位移），塞进 `get_observation().extra`；`navigate_relative` 改为
  行走中轮询位姿直到位移达标（闭环），替代速度×时间估算。
  **验收**：走 2 m 再探针，报告的位移与画面一致（误差 <5 cm）。
- [ ] **B2. 地形体积真更新（影响：中）**
  现状：挖完土 `get_terrain` 报的 volume 不变，LLM"复查地形"看到假数据，
  判断不了"挖没挖到、装没装满"。
  做法：`read_terrain` 从 agxTerrain 读实际参数（已挖除体积 /
  `getTotalAggregateMass` 换算），让"复查"真的反映变化。
  **验收**：挖一斗前后两次 `get_terrain`，volume 有可见差异。
- [ ] **B3. 轮式导航（影响：中，控制卡车行走时才需要）**
  现状：`navigate_relative` 是履带差速（挖掘机链轮专用），卡车/装载机不能行走。
  做法：把驱动源从 `exc.sprocket_hinges` 泛化为 `--joint-map` 声明的轮组
  （如 `drive_left=Hinge2,Hinge4`、`drive_right=Hinge3,Hinge5`），差速逻辑不变。
  **验收**：卡车场景 `navigate_relative(2.0)` 后画面里卡车前移约 2 m。
- [ ] **B4. 挖掘逐帧失败重试（影响：低）**
  现状：某个关键帧超时就跳下一帧，可能"没挖到也算挖完一斗"。
  做法：`dig` 循环里对超时关键帧重试一次（重新下目标再等），两次不到即
  返回 `DigFailure`，让 LLM 重新规划。
  **验收**：人为把某关键帧目标设成限位外的值，`dig` 返回 ok=False 而非假成功。

## C. 已排除项（不适配当前使用方式，做了也用不上）

| 被剔除项 | 剔除原因 |
| --- | --- |
| 视觉模型进仿真环（AGX 渲染帧→检测服务） | 你的任务用地形真值（get_terrain）即可；仅当导师要求 LLM"看图决策"时再立项 |
| 多会话并行控制两台机器 | 你是单机单人工作流，单桥接分时控制已够 |
| inprocess 后端（本进程 import agx） | 硬性不可行：AGX 绑定只有 Python 3.10，框架要求 3.12——remote 桥接就是最终架构 |
| 探针 `--direct` 模式实现 | `--bridge` 模式已覆盖同一功能；`--direct` 只在同事 Windows 机器脱离本仓库工作时才用 |
| Windows 侧桥接实测 | 你的观看方式是浏览器看服务器画面，桥接跑在服务器；仅在需要借用同事 Windows 机器跑桥接时才需要 |
| 换开源仿真引擎（MuJoCo/Isaac 等） | 组里模型/场景全在 AGX 格式，换引擎等于推翻课题 |
