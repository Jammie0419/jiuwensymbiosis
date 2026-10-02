# ACT 数据采集设计（M2）——格式、机制与合理性论证

> 是 `act-integration-plan.md` 第 5 章的实施级设计。回答三个问题：
> **好数据必须满足什么条件**（D1–D8）、**格式为什么长这样**（npz schema）、
> **每个已知的"数据变坏"路径如何被设计挡住**（P1–P8）。
> 实现落点：`jiuwen_agx/agx_excavator/record.py`（纯逻辑，可单测）+
> `scripts/record_demos.py`（采集 CLI）+ `scripts/npz_to_lerobot.py`（转换 CLI）。

---

## 1. 什么才算"好数据"（八条设计要求）

ACT 训练学到的是映射：`(关节角 state, 目标 goal) → 动作块`。部署时它面对的
输入分布、输出语义、时间节拍，全部必须与采集时一致——**数据质量 = 一致性质量**。

| # | 要求 | 内容 |
| --- | --- | --- |
| D1 | 分布一致 | 训练的观测在部署时必须能**一模一样地**再算出来：关节顺序 = `cfg.joint_names`；单位 = 桥接原生单位（swing 弧度、液压缸米）；goal = 基座系米。目标点分布必须覆盖部署会遇到的输入域（可达环带内分层随机） |
| D2 | 动作可执行 | `action_t` 必须是机器**真实到达过**的绝对关节位形。采集记"下一拍实测关节"，保证每个监督目标都物理可达 |
| D3 | 节拍语义明确 | 拍的时序 = 序号不是秒（执行按"到位"推进）。fps=10 只是元数据名义值。教师关键帧过渡切成 K 个子拍，拍密度均匀 |
| D4 | 回合边界清晰 | 一条 episode = 一次完整挖卸循环。起始姿态有界（链条第一循环从 home 出发，后续从前一循环的卸料姿态出发——与部署的连续作业一致）。失败回合按硬判据整条剔除 |
| D5 | 状态诚实 | `state_t` = 下发前**实测**关节（绝不能记指令值——那等于把答案泄漏进输入，策略学会无视反馈）；`action_t` = 该拍执行后的实测关节 |
| D6 | 可审计 | 每个 npz 自带元数据：格式版本、来源、backend、关节名与单位指纹、goal、随机种子、参数。转换器拒绝缺元数据/版本不符的文件 |
| D7 | 可复现 | 教师目标点来自带种子的随机数，种子进元数据——整个数据集可重新生成 |
| D8 | 量与覆盖 | 100–200 条成功循环；挖掘点在环带内按面积均匀采样，卸料点取"挖掘点外 2–4 米"（与技能文档教给 LLM 的用法一致）且仍在环带内 |

## 2. 数据格式（npz schema，v1）

每个成功循环一个文件：`data/raw/<run>/ep_<序号>_<来源>.npz`。

```python
# 数组（N = 拍数；4 = 关节数；顺序 = joint_names）
joints_before: (N, 4) float64   # 第 t 拍下发前的实测关节（= state_t）
joints_after:  (N, 4) float64   # 第 t 拍执行后的实测关节（= action_t，也是 state_{t+1}）
scoop:         (N, ) bool       # 每拍执行后的铲斗有料状态（成功判据原料）
beat_seconds:  (N, ) float32    # 每拍墙钟耗时（仅存证，不参与训练）

# 元数据（一个 JSON 字符串字段 "meta"）
{
  "format_version": 1,
  "source": "teacher" | "keyboard",
  "backend": "mock" | "remote" | "inprocess",
  "robot": "agx_excavator",
  "joint_names": ["swing", "boom", "arm", "bucket"],
  "joint_units": null,            # null = 混合单位（照抄 cfg.joint_units）
  "swing_unit": "rad",
  "goal": {"dig_x_m":.., "dig_y_m":.., "dump_x_m":.., "dump_y_m":..} | null,
  "seed": 12345,                  # teacher 专用；keyboard 为 null
  "beat_timeout_s": 1.0,          # 与部署共用的同一常数（cfg.policy_beat_timeout_s）
  "sub_beats": 20,                # 教师每过渡的子拍数
  "fps_nominal": 10,
  "created_at": "2026-10-01T12:00:00",
}
```

**为什么是这个形状：**

- `joints_before` / `joints_after` 成对，天然构成转换器的帧：
  `frame_t = (state=joints_before[t], action=joints_after[t])`——lerobot 的
  `observation.state` / `action` 两列直接就位，无需任何二次加工。
- 单位**原样存储**（弧度/米混合）。归一化是训练管线的事（MEAN_STD 自动统计），
  采集层做任何"顺手换算"都只会引入单位漂移（见 P4）。
- float64 落盘、float32 只在转换器生成帧时转一次——lerobot 对帧做精确
  dtype 校验，转换点是唯一的类型关口。
- `scoop` 序列是成功判据的**原始证据**，存下来而不是只在采集时判断——
  转换器可以独立复核，采集逻辑有 bug 时旧数据还能重新筛。

## 3. 教师模式机制（来源 A，零人工）

```
RecordingDriverProxy 包住真 driver：
  move_joints_blocking(targets) 被拦截，切成 K 段：
    前 K-1 段：从上一拍实测位置线性插值到目标，每段以 beat_timeout_s 下发
    第 K 段：精确目标 + 原始超时（关键帧保底到位，不欠行程）
    每段记录 (下发前实测, 执行后实测, scoop_state(), 耗时)
  joint_names / scoop_state / mark_scoop / get_joint_positions / home 原样转发
```

- `execute_dig_cycle` **原封不动**当教师——代理只拦 `move_joints_blocking`，
  教师的全部前置校验（可达环带、斗空）照常生效。
- K 默认 20：一个 6 过渡循环 ≈ 126 拍，与 chunk_size=100 同量级。
- `beat_timeout_s` 读 `cfg.policy_beat_timeout_s`——**采集与部署共用同一个
  常数**，拍节奏定义只有一份。
- 链式采集：`--chains 10 --cycles-per-chain 10` = 100 条；每链开头
  `home()` 归位，链内循环自然衔接（部署时 act_exec 也是从上一循环的
  卸料姿态开始，起始分布一致）。

## 4. 键盘模式机制（来源 B，v1 只旁观）

桥接以 `JIUWEN_KEYBOARD=1` 启动后，键盘在 AGX 进程内直接开挖掘机（不经过
命令通道），采集器以固定间隔轮询 `joints` + `scoop` 纯旁观记录，产出
**连续流** npz（`goal: null`）。

**v1 的明确限制**：键盘流的挖掘/卸料目标点操作员没说话，采集器不知道——
goal-less 的 episode 转换器**默认剔除**（不计入训练集）。键盘数据 v1 只用于
行为多样性分析；把"操作员逐循环确认目标点"的交互做进去是后续增强
（`act-integration-plan.md` §10 决策点，不阻塞 M2 主线）。

## 5. 转换规则（npz → LeRobotDataset）

每条 npz 过四道闸（`record.py: admission_reason()`，纯函数、可单测）：

| 闸 | 条件 | 不过的理由文案 |
| --- | --- | --- |
| 版本 | `format_version == 1` 且元数据齐全 | "unsupported/missing metadata" |
| 来源 | `backend != "mock"`（除非 `--allow-mock`，仅供管线自测） | "mock data must not enter a training dataset" |
| 目标 | `goal` 非空且四键为有限数 | "episode has no goal" |
| 成功 | `scoop` 至少一次 True **且** 最后一拍为 False；拍数 ≥ 10 | "cycle incomplete" / "too short" |

通过后生成帧（全部 `float32`）：

```python
{"observation.state":             state_t,          # (4,)
 "observation.environment_state": [goal 四键],      # (4,) 同一 episode 内恒定
 "action":                        action_t,         # (4,)
 "task": "excavate"}
```

features 定义（lerobot 精确契约，names 列表驱动其全部下游工具）：

```python
{"observation.state":             {"dtype":"float32","shape":(4,),"names":joint_names},
 "observation.environment_state": {"dtype":"float32","shape":(4,),"names":["dig_x_m","dig_y_m","dump_x_m","dump_y_m"]},
 "action":                        {"dtype":"float32","shape":(4,),"names":joint_names}}
```

`LeRobotDataset.create(root=新目录, fps=10, use_videos=False)` → 逐帧
`add_frame` → 每条 `save_episode()` → 结束 `finalize()`。数据集根目录必须
不存在（lerobot 契约），脚本拒绝覆盖已有目录。收尾打印：总数/通过/剔除
（按理由分组）与拍数分布。

## 6. 已知"数据变坏"路径与对应的挡板（P1–P8）

| # | 坑 | 后果 | 挡板 |
| --- | --- | --- | --- |
| P1 | 把**指令值**当 action 存 | 长过渡期动作重复 → 重放时空拍浪费 chunk 预算 | action = 执行后实测（D2）；过渡切成 K 段，每拍都是有效位移 |
| P2 | 把指令值当 state 存 | 输入泄漏，策略学会无视真实反馈 | state = 下发前实测（D5） |
| P3 | 每条都从 home 出发 | 起始分布窄，部署从卸料姿态启动就出分布 | 链式采集（D4） |
| P4 | 单位换算漂移（有人"好心"转 deg） | 部署输出系统性错单位 | 原生单位落盘 + 单位指纹进元数据；转换器对单位指纹做**数据集级硬门**（`unit_conflict`，不一致整批拒绝，绝不静默过滤） |
| P5 | dtype 混乱（float64 帧喂 lerobot） | 精确校验直接拒绝 | 转换点统一 `np.float32`，唯一关口 |
| P6 | mock 数据混入训练集 | 假关节角污染真分布 | 转换器默认拒绝 backend=mock（--allow-mock 仅自测） |
| P7 | 失败/半截循环入库 | 策略学会半途而废 | scoop 序列硬判据 + 最短拍数（第 5 节闸 4） |
| P8 | 采集与部署的拍超时不一致 | 重放节奏系统性偏差 | 两边读同一个 `cfg.policy_beat_timeout_s` |

## 7. 对接核验：这套规范喂给 ACT 训练为什么是对的

本节回答"当前数据规范接入 ACT 有没有问题"——把格式、ACT 的训练数学、部署执行
三边对齐后的结论（2026-10-01 复核）。

1. **滑动窗口与部署重查询天然对齐。** lerobot 训练时从每一帧都能起一个 chunk
   （`delta_timestamps = action[t..t+chunk-1]`，episode 尾部不足的帧由
   `action_is_pad` 掩码剔出损失）——模型学到的是"从**任意中途状态**都能开一段"。
   部署 `act_exec` 每块用尽后重观测再前向，恰好是训练分布内的行为，不存在
   "只会从头开始"的模式。
2. **帧对的代数自洽。** `action_t = state_{t+1}`（相邻帧首尾相接），chunk 内
   第 k 步 = 从 `state_t` 起 k 拍后的实测位形；部署把每个输出当**绝对目标**
   逐拍到位执行，位置控制下轨迹形状保真。
3. **goal 条件化是标准用法。** `observation.environment_state` 每条 episode
   恒定，是标准的目标条件输入；它同时满足 ACT `validate_features` 的硬性
   校验，并且真正起作用——不同 goal 产生不同轨迹，这是模型区别于固定脚本的
   全部价值所在。
4. **fps 只进 delta 换算。** fps=10 → `delta = i/10` 秒，lerobot 的
   `check_delta_timestamps` 精确整除；拍的真实节奏（到位时间）不进训练——
   位置序列才是学习对象，与部署的"到位推进"语义一致。
5. **归一化闭环不经过我方代码。** 数据统计自动累计 → 训练注入前处理器 →
   随 checkpoint 保存 → 部署后处理器反归一化；弧度/米混排的尺度问题全程由
   lerobot 处理，我方只在转换点保证 float32。
6. **样本量核算。** ~126 拍/条 × 100 条 ≈ 2700 个完整 chunk 窗口（另有掩码
   窗口），对 4 维动作的 ACT 充足；不够时加条数比改模型便宜。

**写明的边界（是有意的取舍，不是遗漏）：**

- **观测不含铲斗载荷。** 教师数据里位形+goal 几乎唯一决定循环阶段，载荷引起的
  微小形变已隐含在实测位形中，闭环由真实机器补齐。要显式建模需后端先暴露
  浮点质量（`read_inventory`，可选增强）再重训。
- **观测维度是重训契约。** `observation.state` 恒为 4 维（= `joint_names` 顺序）；
  未来加载荷/底盘位姿 = 改维度 = checkpoint 不兼容，必须重训。
- **拍节奏非墙钟。** 每拍耗时被 `beat_timeout_s` 限制上界但非恒定；准静态挖掘
  可接受。要固定频率伺服需 motion.servo 流式协议（远期，见接入计划 M5）。

## 8. CLI 与验收

```bash
# 采集（需 AGX 桥接在跑；--config 指向 remote 配置）
python scripts/record_demos.py --config configs/agx_excavator/agx_excavator.local.yaml \
    --out data/raw/run001 --chains 10 --cycles-per-chain 10 --seed 7
# 管线自测（mock 机制验证，产物永不入训练集）
python scripts/record_demos.py --config ... --out data/raw/smoke --chains 1 --cycles-per-chain 2 --allow-mock

# 转换（训练机；需 lerobot）
python scripts/npz_to_lerobot.py --raw data/raw/run001 \
    --root data/lerobot --repo-id local/agx_dig_demos --fps 10
```

验收：mock 冒烟全链产出合法 npz 且转换器按预期拒绝 mock；教师模式在真桥接上
产出 ≥100 条通过四道闸的 episode；`lerobot-dataset-viz` 抽查曲线连续无跳变。

## 9. 明确不做（本设计边界）

- 键盘目标点确认交互（v1 剔除 goal-less，见第 4 节）；
- 相机帧录制（M5 视觉版另立）；
- 数据增广/重采样（先看基线够不够）；
- 跨 backend 混合数据集（P6 直接禁止）。
