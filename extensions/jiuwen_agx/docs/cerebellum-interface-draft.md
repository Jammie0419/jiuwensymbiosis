# 大脑-小脑-仿真平台对接草案

> 面向三个组的接口文档（讨论稿 v1，2026-10-01）。目的：把"各组交付什么、
> 在哪交接、谁依赖谁"写清楚，列出小脑组需要回答的对齐问题，并把三种对接
> 情形下各自要改什么提前说明——避免各组按错误假设开发。
>
> 配套阅读：`act-integration-plan.md`（接入计划）、
> `act-data-collection-design.md`（数据格式细节）、
> `architecture-flows.md` 第 10 章（接缝技术细节）。

---

## 1. 三组分工与交接面

```
┌─────────────────────┐      ┌──────────────────────────┐      ┌─────────────────────┐
│  仿真平台组          │      │  大脑组（openjiuwen 编排）│      │  小脑组（策略模型）  │
│                     │      │                          │      │                     │
│ 交付：              │      │ 交付：                   │      │ 交付：              │
│ · 可驱动的仿真场景  │ TCP  │ · 任务编排（LLM 规划）   │ 模型 │ · 训练好的策略模型  │
│ · 状态真值(关节/质量)│ 桥接 │ · 动作契约 + 执行与安全  │ ───► │   (checkpoint 工件) │
│ · 地形/物料仿真     │ ◄──► │ · ★数据采集与部署通道 ★  │      │ · 训练管线          │
│ （宜补：场景重置、  │      │   （采集器/数据集/适配器）│      │                     │
│  超实时步进——若走RL）│      │                          │      │ 需要：示教数据、    │
│                     │      │                          │      │ 成功判据、部署环境  │
└─────────────────────┘      └──────────────────────────┘      └─────────────────────┘
```

**三组的交接面只有两个，而且都已定型：**

1. **仿真 ↔ 大脑：TCP 桥接协议**（已运行）。命令：`joints / move_joints /
   base_pose / terrain / scoop / inventory / frame`；状态真值可查。
2. **大脑 ↔ 小脑：动作契约 + 数据格式**（本文档 §3、§4）。大脑以
   `act_exec(dig_x_m, dig_y_m, dump_x_m, dump_y_m)` 调用小脑能力，以
   LeRobotDataset v3 向小脑供训练数据；小脑以模型工件（checkpoint）回交。

**关键认知：数据采集与部署通道是公共设施，由大脑组提供。** 无论小脑组用
模仿学习还是别的训练方式，"仿真里自动采数据 + 把训好的模型接回仿真执行"
都必须有人建——这两样已建成（§3），小脑组可以直接用，不必重建。

---

## 2. 小脑模型是什么、为什么分层（会议用一页纸）

**本质。** 小脑模型是一个"感知 → 动作"的直接映射函数：输入低层观测
（关节角、目标坐标，未来加图像），输出低层动作（未来 100 拍的关节目标序列）。
中间没有语言、没有符号推理。它毫秒级前向、行为确定，能力**来自数据而非编程**。

**为什么必须分层，不能让 LLM 直接发关节指令：**

| | LLM（大脑） | 策略网络（小脑） |
| --- | --- | --- |
| 单次决策延迟 | 秒级 | 毫秒级 |
| 成本 | 按调用计费 | 本地推理，近零 |
| 数值输出 | 不可靠、需校验 | 输出即张量 |
| 擅长 | 理解意图、选目标、规划步骤 | 连续运动控制 |

大脑把"把左边的土堆挖两斗倒到右边"压缩成四个坐标；小脑把坐标变成一整段
运动。**交接面越窄，三组越独立**——这就是分层的工程理由。

**对"ACT 组在做什么"的判断（供核对）：** "给一批数据 → 训练 → 任务里执行
→ 看效果"是**模仿学习**（imitation learning）的标准流水线；ACT
（Action Chunking with Transformers）正是其代表模型。它与强化学习（RL）的
区别：RL 不给示教数据、给奖励函数、让 agent 自己试错（数据量大几个数量级，
且要求仿真器支持重置/加速）。若小脑组实际走 RL，见 §5 情形 C。

---

## 3. 大脑组已就绪的东西（可直接用，附位置）

| 设施 | 状态 | 位置 / 说明 |
| --- | --- | --- |
| 动作契约 `act_exec` | ✅ 已提交 (3289c00) | `jiuwen_agx/actions.py`：参数与 `dig` 完全同名（四个地面坐标、米、基座系），失败返回 `{"ok": false, "error": 可读原因}` |
| 执行循环 + 安全 | ✅ 已提交 | 策略输出逐拍过 `move_joints_blocking` 驱动校验（限位/非有限值拒绝）；超拍数兜底；失败走 RecoveryRail；成功判据 = 铲斗质量 > 50 kg 的"铲起→卸掉"序列 |
| **数据采集器** | ✅ 已提交 (a8a32b9) | `scripts/act/record_demos.py`：脚本教师自动采（零人工），键盘示教旁观模式；产出带元数据的 npz（关节角对 + 目标 + 铲斗真值 + 单位/来源/种子指纹） |
| **数据集转换器** | ✅ 已提交 | `scripts/act/npz_to_lerobot.py`：npz → LeRobotDataset v3（纯 Parquet，无视频依赖），四道质量闸（禁 mock / 必须有目标 / 循环完整 / 最短拍数）+ 数据集级单位一致性硬门 |
| 数据格式规范 | ✅ 成文 | `act-data-collection-design.md`：帧对 `(state_t, action_t=state_{t+1})`、`observation.environment_state` = 目标坐标、float32、`task="excavate"` |
| 策略适配器（部署侧） | ✅ 骨架就绪 | `jiuwen_agx/policy.py`（注册表+协议）+ `policy_act.py`（lerobot checkpoint 加载→推理→反归一化，torch 惰性导入）；换模型=换一个适配器类 |
| 训练样本量核算 | ✅ | ~126 拍/循环 × 100 循环 ≈ 2700 个 chunk 训练窗口，对 4 维动作的 ACT 充足 |

---

## 4. 需要小脑组回答的 6 个对齐问题

| # | 问题 | 为什么问 |
| --- | --- | --- |
| ① | 算法与框架：ACT？用 lerobot 吗？哪个版本？ | 决定我们的数据格式是否直接可吃（我们按 lerobot 0.6.1 对齐） |
| ② | 模型输入输出契约：观测要哪些通道（关节/图像/目标）？动作是绝对关节目标还是增量？多少维？ | 决定适配器怎么写（§5）；我们当前默认：观测=4 关节角+4 维目标，动作=4 维绝对目标 |
| ③ | 数据谁采、什么格式？ | 若用 LeRobotDataset v3，我们的采集器直供；若自研格式，我们加一个转换（半天级） |
| ④ | 交付物形态：lerobot `pretrained_model` 目录（config+权重+归一化处理器）？还是别的（onnx/服务）？ | 决定部署适配器的加载代码；当前按 lerobot checkpoint 目录实现 |
| ⑤ | 推理跑在哪侧、什么频率？大脑组机器 CPU/GPU？ | 当前按"编排机本地 CPU 推理、逐拍调用"实现，4 维动作毫秒级 |
| ⑥ | 成功判据是否一致？ | 我们用铲斗实测质量 > 50 kg（仿真组真值）；训练数据的成败过滤与此同源，若他们另有判据需对齐 |

---

## 5. 三种对接情形与改动范围

| 情形 | 小脑组做法 | 大脑组要改的 | 仿真组要改的 |
| --- | --- | --- | --- |
| **A. 吃我们的数据集，训 ACT**（最可能） | 用我们的 LeRobotDataset 训练 | **零改动**——只等 checkpoint，填进 YAML `policy: {name: act, ckpt: ...}` | 无 |
| **B. 自研数据格式/模型** | 自采或要我们转格式 | 加一个格式转换（半天级）+ 一个适配器类（~80 行） | 无 |
| **C. 走 RL / 观测要相机 / 输出末端位姿** | 需要环境交互或视觉通道 | 加适配器；观测加 `image` 键（已预留）；输出位姿→关节需逆解（挖掘机无 IK，需仿真组出关节级接口） | 桥接实现 `frame` 出帧；RL 需场景重置+超实时步进 |

**要点：三种情形下，动作契约（act_exec）、执行循环、护栏、成功判据、
LLM 编排全部不动。** 这就是接缝先行的意义——小脑组的任何技术路线变化，
都被限制在"一个适配器文件"的爆炸半径内。

---

## 6. VLA 升级路径（如果未来小脑是视觉-语言-动作模型）

- 观测侧：`obs` 已预留 `"image"` 键；数据集加 `observation.images.<cam>` 特征
  （dtype="image"）即可，ACT 换 VLA（SmolVLA/π0，lerobot 同一注册表）照常训练；
- 桥接侧：需仿真组实现相机出帧（`grab_frame` 当前是占位）；
- 大脑侧：零改动——LLM 看到的仍是 `act_exec` 契约。

## 7. 建议的对接节奏

1. **本周**：小脑组回答 §4 的 6 个问题（一次会或书面）；
2. **随后**：大脑组在真桥接上采 10–20 条样例数据集，把"格式 + 数据量 +
   采集器"实物交给小脑组验证可直接训练；
3. **模型 v1 到位后**：按 §5 情形接入，跑验收（10 个留出目标点 ≥7 成功，
   与脚本 `dig` 出对比表）；
4. **长期**：视觉通道（§6）、RL 备选路线按需另立计划。

---

## 8. 落地版（2026-10-11）：goal-act-v2（小脑组自研模型）接入

> **面向小脑组的交接版说明书见 [`goalact-handoff.md`](goalact-handoff.md)**
> （名字对照、双方接口、命令、验收、现状，可直接转交给小脑组）。

小脑组的 `act_trajectory` 项目（Goal-ACT V2：35 维状态×10 帧历史、13 维目标
特征、8 相位、16 步弧度增量、自定义 PyTorch、HDF5 数据）即 §5 的**情形 B**。
本节的接线已实现：适配器 `policy_goal_act.py`（注册 `goal_act_v2`）、数据
转换器 `scripts/act/lerobot_to_goalact_h5.py`、每机器标定 `machine_map.py`
（schema：`configs/machine_agx365.template.json`）。

### 8.1 三层分离（通用性的落点）

| 层 | 装什么 | 换挖掘机时 |
| --- | --- | --- |
| **通用链路层** | `act_exec` 契约、Policy 接缝（`reset()` + `predict(obs)->目标序列`）、执行循环/护栏/验收器、LLM 编排 | **一行不改** |
| **每机器层** | `machine_<机型>.json`：单位映射（Prismatic 米↔等效角）、sign/offset、goal 高度、**载料判据标定参数** | 只换文件 |
| **每模型层** | 适配器 + checkpoint（`policy_act`=lerobot ACT；`policy_goal_act`=goal-act-v2；未来 VLA） | 只换适配器/模型 |

"铲斗 >50kg"是 AGX 365 **这台身体**判"装了土"的标定值，住在桥接
（`agx_bridge_server.py` 的 `LOADED_MASS_THRESHOLD_KG`），**不在链路契约里**。
链路只认 `driver.scoop_state()` 的布尔语义"未装→装→卸"；换真机就是那台机器
自己的传感器实现（载荷/压力），参数进它的 machine 文件。

### 8.2 模型准入（"什么模型都能接"的三个条件）

1. 能喂进去：当前关节位形 + 目标点——模型想要的任何方言（35 维/历史/相位/度）
   由适配器翻译；
2. 能吐出来：逐拍关节目标——绝对或增量皆可（增量由适配器对实测累加；
   goal-act-v2 适配器每次只返回 1 个目标，使大脑组的拍循环**精确等价于**模型
   自己的"逐步重规划"闭环）；
3. 输出必须过驱动硬闸（限位/非有限）——任何模型都绕不过。

### 8.3 数据字段级映射（`data/raw/<run>/*.npz` → `episode_*.h5`）

| 大脑组的字段 | 小脑组的字段 | 转换规则 |
| --- | --- | --- |
| `joints_before/after`（原生单位：swing 弧度 + 三缸米；到位节拍） | `observations/state[:, 6:10]`（**度**）+ `joint_rad`（弧度） | 经 machine 文件换算；**未标定默认拒绝**（fail-closed，防米当角污染） |
| 拍序列（到位节拍） | **拍 = 帧**（`timeline=beats`，control_frequency=10 名义） | 不重采样（小脑组的训练帧本就是受 max_joint_step 限幅的"步"，与大脑组"拍"同构；重采样到细粒度会让部署循环数千拍跑不完）。`--resample-hz` 可选显式等间隔 |
| 相邻拍差分 | `actions/joint_increment` (T,16,4) | 16 步滑窗 + pad；**写成度**（小脑组载入器的 99% 分位>0.5 嗅探在两极幅度下都判度；另写 `annotations/unit_joint_increment="deg"` 备改显式） |
| `goal` 4 个地面坐标 | `annotations/goal` + `goal` (T,6) | z 取 machine 文件 `pick_z_m/dump_z_m`；小脑组现有载入器仍 FK 伪推断，直读需下方任务② |
| `scoop` 载料真值 | `phase` (T,)，默认 **two 模式：未载料→2、载料→4** | 与部署适配器的相位估计**同源**（train/inference 一致性优先）；`--phase-mode scoop` 给 0..7 自然分段，`time` 为时间八等分兜底 |
| 成功判据（装料后卸掉） | —— | 不完整循环整条剔除，计入 `skipped` |

转换器实测：`run_all` 112 条 → **103 条**（跳过 9 条不完整），小脑组的
`infer_real_episode` 载入器 **103/103 直读通过**（拍=帧时间轴、two 相位模式）。

转换器同时导出 **`suggested_config.json`**（joint_limits_deg=数据实际角域、
max_joint_step_deg=每拍运动 p99.9）——**小脑组重训时的 config 必须采用这组值**：
小脑组推理端把每一步 clip 到 config 的 `max_joint_step_deg`（现 ckpt 是 IK 步长
~2.5°/步），而大脑组的"拍"合法行程可达 20°+/拍；沿用旧值会把部署循环压慢数倍
（大脑组的执行循环每拍重观测、每次预测只走一步）。

### 8.4 小脑组任务清单（按依赖排序）

1. **AGX 365 机器配置 + 重训**：用实测 `links_m`（含铲斗铰点-齿尖，现为 1.2m
   假设）+ 等效角域重造教师数据重训（小脑组的 geometry_signature 门强制）；
   **或**直接吃 §8.3 转换数据微调——那之前需先完成第 3 项的标定；
2. **real 载入器小补丁**（几行）：`annotations/goal`、`annotations/phase`
   存在时直读（现在只会从 FK/液压电流推断伪标签）；
3. **标定协作**（AGX license 恢复后，约半天）：按 `scripts/calibration/agx_machine_fit.py`
   的采样格式（每缸 ≥5 点扫行程 + 齿尖在策略系坐标，见该脚本 docstring）记录
   采样 JSON → 拟合（median 残差 ≤0.05m 才判过）→ 输出 `calibrated: true` 的
   `machine_agx365.json` 与**建议给到小脑组的 `links_m`**（进小脑组 config 的
   geometry_signature，会强制重造数据重训）。拟合器已带合成自测
   （`--selftest` 与 pytest 往返恢复），license 恢复后只需补"扫缸记录"这一环；
4. **修推理交付**：`.venv` 坏链（指向 `D:\anaconda\...`）修复或给
   requirements；确认 `src/act_trajectory` 可被外部 import（yaml `src:` 路径引用）；
5. **补位移↔角多项式逆映射**（`agx_adapter_hydraulic.py` 未实现；大脑组的
   `machine_map.JointMap` 已用单调二分实现，可作参考）；
6. **对齐问题**：phase 供给（推荐 phase-free 变体；否则大脑组用 scoop 真值估计，
   默认 phase_dig=2 / phase_dump=4）、goal 的 z 语义、达标线=10 留出点 ≥7。

### 8.5 使用（接线已就绪）

```yaml
policy:
  name: "goal_act_v2"
  ckpt: "<goal-act-v2>.pt"      # 模型（config/归一化内嵌）
  src: "<project>/src"           # 含 act_trajectory 包
  machine: "machine_agx365.json" # 每机器标定（未标定仅 mock 链路干跑）
```

```bash
# 数据转换（标定前可 --allow-uncalibrated 干跑验证链路，产出禁止入训练）
python scripts/act/lerobot_to_goalact_h5.py --raw data/raw/run_all \
    --machine <machine.json> --out data/goalact_h5/run_all
# 验收（与 lerobot ACT 完全同一把尺）：act_acceptance.py 10 留出点 ≥7 + dig 对照
```

