# 待办清单

> 工作方式（2026-09 起）：**Windows 本机**跑 AGX + 桥接（`scripts\start_agx_bridge.bat`，
> 自带窗口），本仓库 venv 发任务。服务器/浏览器观看那条路已不用。

## 0. 已经跑通的（别再怀疑，也别重做）

- LLM 命令全链路实测通过：`get_terrain → dig(3,0 → 6,0) → home` 三步 `ok`，
  真实 365 挖掘机走完 12 个关键帧，单轮 18.7 s，关节停在 dump 关键帧值上。
- 单位契约：`joint_units: null` + `swing_unit: "rad"`，关键帧取自 AGX 官方
  `digCycle` 并收敛到实测行程内；`--check-config` 通过。
- 载料判定：`LOADED_MASS_THRESHOLD_KG = 50`（空斗贴地本底实测 0~1 kg）。
- 启动方式：AGX 自带 python 直跑插件（`init_app` 自建 ExampleApplication）。
  **不要改回 `agxViewer 插件.agxPy`**——agxViewer 重建进程会丢掉 PATH/PYTHONPATH。

## A. 还需要"看着画面"标定的

- [ ] **A1. 回转零位（swing_offset_deg）**
  `dig` 的摆转目标 = `atan2(挖点) + swing_offset_deg`。让上车摆到基座 +X 方向，
  用 `monitor_ui.py` 读 swing 当前角（rad），取其负值换算成度填进
  `dig_cycle_tuning.swing_offset_deg`。
  **验收**：`dig` 时铲斗朝向画面里的挖点，而不是偏一个固定角。
- [ ] **A2. 关键帧微调（dig_cycle_tuning，原生单位）**
  现值来自官方自动挖掘循环，已收敛进限位；但 `dump_boom_deg: 0.38` 顶在上限
  0.4 附近，`ready_arm_deg: -0.82` 贴近下限 -0.88。逐帧看：吃土够不够深
  （`dig_arm_deg`）、收斗兜不兜得住土（`curl_*`）、倒不倒得干净（`dump_bucket_deg`）。
  **验收**：一斗挖完 `scoop_state` 为 true（`inventory` 的 `bucket_mass_kg` 明显
  大于空斗本底），倒点处画面可见土堆。
- [ ] **A3. 可达包络（reach_min_m / reach_max_m）**
  现值 1~6 m 是估算。把铲齿摆到最近/最远能挖到的位置，读关节角反算半径。
  **验收**：明显可挖的点不报 `outside the reachable annulus`；够不着的点正确拒绝。
- [ ] **A4. 伺服表现（set_joint_targets 的阻尼）**
  现用官方 `logInterpolate` 距离自适应阻尼 + 从 Motor 继承的 forceRange。
  逐关节走全行程，看有无超调/抖动/顶限位异响。
  **验收**：每关节 ±1 cm / ±0.063 rad 容差内 30 s 稳定到位。

## B. 缺失能力（按影响排序）

- [ ] **B1. 机器位置感知（影响：高）**
  `navigate_relative` 是开环（速度×时间），底盘走完后的实际位置框架不知道，
  远距离作业后坐标系会漂。做法：桥接加 `read_pose()`（AGX 根刚体世界位姿减去
  初始位姿）塞进 `get_observation().extra`，`navigate_relative` 改为轮询位移闭环。
  **验收**：走 2 m 后报告的位移与画面一致（误差 <5 cm）。
- [ ] **B2. 地形体积真更新（影响：中）**
  `read_terrain` 的 volume 仍是常量，LLM"复查地形"看不到变化（铲斗质量已有真值，
  见 `inventory.bucket_mass_kg`）。做法：从 agxTerrain 读已挖除体积。
  **验收**：挖一斗前后两次 `get_terrain`，volume 有可见差异。
- [ ] **B3. 轮式导航（影响：中，控制卡车行走时才需要）**
  `navigate_relative` 是履带差速（`exc.sprocket_hinges` 专用）。做法：把驱动源
  泛化为 `--joint-map` 声明的轮组，差速逻辑不变。
  **验收**：卡车场景 `navigate_relative(2.0)` 后画面里卡车前移约 2 m。
- [ ] **B4. 关键帧失败重试（影响：低）**
  某个关键帧超时即跳下一帧，可能"没挖到也算挖完一斗"。做法：超时帧重试一次，
  两次不到返回 `DigFailure`。
  **验收**：人为把关键帧设成限位外的值，`dig` 返回 ok=False 而非假成功。

## C. 已排除项（附原因）

| 被剔除项 | 原因 |
| --- | --- |
| 视觉模型进仿真环（AGX 渲染帧→检测服务） | 任务用地形真值（`get_terrain`）即可；仅当要求 LLM"看图决策"时再立项 |
| 多会话并行控制两台机器 | 单机单人工作流，单桥接分时控制已够（桥接一次只服务一个客户端） |
| inprocess 后端（本进程 import agx） | venv 是 3.12.7、AGX 绑定是精确 3.12.10，跨解释器装不进去；remote 桥接就是长期架构 |
| 换开源仿真引擎（MuJoCo/Isaac 等） | 组里模型/场景全在 AGX 格式，换引擎等于推翻课题 |
| 服务器 + 浏览器观看（x11vnc/noVNC、`agx_viewer_*.sh`） | 已改为 Windows 本机窗口，那条路不再用（脚本保留仅供 Linux 端参考） |
