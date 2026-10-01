# ACT 接入计划 · 系统性审计报告

> 对 `act-integration-plan.md`（v1，2026-10-01）的逐条核验报告。
> 方法：把计划里每条断言拿去对三边源码验证——lerobot 0.6.1（pip 拉取的 wheel，逐文件读）、
> jiuwensymbiosis core（本地）、jiuwen_agx 扩展包（本地）。只采信源码，不采信记忆或文档。
>
> 结论：**计划总体方向成立**（接缝位置、注册链、能力门控、core 零改动、体量估计都经得起
> 源码对质），但有 **3 处错误说法**和 **9 处必须补强的细节**；其中两处不修会直接导致
> 「按计划做出来跑不通」（训练命令必失败、逐拍安全断言是错的）。所有修正已落到计划 v2。

---

## 一、逐条核验表

| # | 计划 v1 的断言 | 结论 | 证据（file:line） |
| --- | --- | --- | --- |
| 1 | ACT 拒绝纯关节角输入，必须有图像或 environment_state | ✅ 成立 | `lerobot/policies/act/configuration_act.py:162-164`；`configs/policies.py:141-148`（env_state 必须精确命名 `observation.environment_state` 且类型 ENV） |
| 2 | 把目标点映射为 environment_state 既过校验又起条件化作用 | ✅ 成立 | `utils/feature_utils.py:168-175`（dataset 特征→policy 特征类型映射）；`modeling_act.py:461-468`（env_state 进编码器 token） |
| 3 | 推理隐变量取零、输出确定 | ✅ 成立 | `modeling_act.py:454-458`（非训练路径 `latent_sample = zeros`） |
| 4 | 归一化/反归一化由 lerobot 处理器完成，统计量随 checkpoint | ✅ 成立 | `policies/act/processor_act.py:29-50`；`processor/factory.py:158-175`；checkpoint 内容 `common/train_utils.py:104-175` |
| 5 | 一次前向出整块（chunk）+ 队列语义 | ✅ 成立，但适配器应改进用法 | `modeling_act.py:93-135`：队列是 `select_action` 内部机制；适配器直接调 `predict_action_chunk` 更贴合「逐拍到位」语义（见修正 A4） |
| 6 | 训练时不需要我方做任何 chunk 构造 | ✅ 成立（v1 未写明，v2 补） | `datasets/factory.py:34-66`：`delta_timestamps` 自动由 `policy.config.action_delta_indices`（=range(chunk_size)）换算为 i/fps；样本自动成 `action (chunk,4)`+`action_is_pad (chunk,)`（`dataset_reader.py:215-232`） |
| 7 | 数据集统计量自动算 | ✅ 成立 | `dataset_writer.py:322-324`（逐 episode 累计）→ `meta/stats.json`（`dataset_metadata.py:680-681`），无独立步骤 |
| 8 | 采集格式：features 表 + add_frame + save_episode | ✅ 方向成立，细节全错漏 | 精确格式见修正 A2；`task` 必填、`timestamp` 禁传（`dataset_writer.py:202-228`）；dtype/shape 精确校验（`feature_utils.py:300-327`）；`create` 的 root 必须不存在（`dataset_metadata.py:829-832`） |
| 9 | 来源 A 录制点「挂在逐帧 move_joints_blocking 调用处」 | ❌ 不可行 | 教师每个关键帧阻塞数秒，按调用点采样每循环只得 ~12 帧；多线程轮询与阻塞式 TCP 请求互斥（后端请求/响应模型）。改为子步进教师（修正 A3） |
| 10 | 终止判据「bucket_mass() > 50 且 at_dump_pose」 | ❌ 双不可行 | 客户端无 float 质量接口：后端只有 `scoop_state() -> bool`（`sim/backend.py:521-522`、`sim/driver.py:196-198`）；float 只看服务器 `inventory`（`agx_bridge_server.py:660-663`）。「at_dump_pose」无此判定。改为 scoop_state 序列判据（修正 A5） |
| 11 | §4.6「SafetyRail 限位检查对逐拍下发生效」 | ❌ 错误 | SafetyRail 只检查固定 tool 名集合：`motion.joint → {move_joint, move_named_joint}` 等（`rails/safety.py:57-64,70`）；`dig`/`act_exec` 不在名单，逐拍 `move_joints_blocking` 是驱动调用不是工具调用。**逐拍真实守卫 = driver 校验**（`sim/driver.py:98-121`：未知关节/非有限/限位 → ValueError）（修正 A6） |
| 12 | ActFailure 与 DigFailure 同形、失败路径一致 | ✅ 成立 | runner `ok = result.get("ok", True)`（`agent/fast/runner.py:644-648`）→ `{"ok": False}` 走 `_raise_executor_failure` 同一路径 |
| 13 | 注册链 `register_capability` + `register_capability_spec` | ✅ 成立，与扩展包现用法一致 | `env/base.py:75-83`；`adapters/_common/capability_spec.py:86-101`；现例见 `jiuwen_agx/__init__.py`（motion.excavator 的同一套四步） |
| 14 | ActionSpec 构造期校验能力成员资格（注册顺序敏感） | ✅ 成立 | `api/decorators.py:128-140`（`UnknownCapability`） |
| 15 | 工具门控 = api∩env，未配置 policy 的机体不出 act_exec | ✅ 成立，且已有现成机制 | `tools/builder.py:54-61,95`；`sim/env.py:62-77`（`_capabilities_for_config` + `extra_capabilities`）；`agx_excavator/env.py:24-25` 现例 |
| 16 | YAML 里加 `policy:` 节即可 | ✳ 陷阱 | `sim/config.py:125`：**未知键静默丢弃**——不先给 `AgxExcavatorConfig` 加字段，YAML 写了等于没写（修正 A7） |
| 17 | 键盘演示 `JIUWEN_KEYBOARD=1` 已有，旁观录制可行 | ✅ 成立 | `scripts/agx_bridge_server.py:320-328`；键盘在 AGX 进程内驱动（不经桥接命令）→ 服务器命令循环可并行响应 `joints` 查询 |
| 18 | 训练命令 `lerobot-train ...` | ❌ 必失败（缺 3 项） | 缺 `--policy.push_to_hub=false`（默认 True，无 repo_id 时 validate 抛错：`configs/train.py:283-289`）；缺训练依赖 `accelerate`（`scripts/lerobot_train.py:231`）；`output_dir` 已存在即报错（`configs/train.py:259-263`）（修正 A1） |
| 19 | Windows 训练环境 | ✳ 有雷 | `common/train_utils.py:96-101`：每次存 checkpoint 都做 `symlink_to`（`last` 链接），Windows 无符号链接特权（WinError 1314，本仓测试已踩过）→ 需开发者模式或包装脚本（修正 A1） |
| 20 | 计划量级 ~250 行 + core 零改动 | ✅ 成立（量级不变，拆分见 v2） | 新增项：录制代理/转换脚本/测试，估 ~350 行，仍全在扩展包内 |
| 21 | 适配器从 ckpt 加载 policy + 处理器 | ✅ 成立，v1 未给精确路径/调用 | checkpoint 目录 = `outputs/.../checkpoints/{step:06d}/pretrained_model/`（`utils/constants.py:49`）；加载 API `policies/factory.py:150-237`（修正 A8） |
| 22 | ACT_EXECUTE 契约四件套 | ✳ 漏一个字段 | v1 未写 `tags`：RecoveryRail 只认领 tags 含 `motion`/`grasp` 的工具（`rails/recovery.py` 模块 docstring）；DIG 带 `tags=("motion",)`（`jiuwen_agx/actions.py:64`）→ ACT_EXECUTE 必须同样带 `tags=("motion",)`（修正 A9） |

图例：✅ 成立 · ✳ 需修正/补强 · ❌ 不成立

---

## 二、改变设计的发现（A1–A9，均落在计划 v2）

**A1 · 训练环境的三个硬前提（v1 §6 不完整，照 v1 命令直接跑必失败）**
a) 安装：`pip install "lerobot[training]==0.6.1"`（`accelerate` 是训练硬依赖；`[training]` 还带 `[dataset]`=`datasets/pandas/pyarrow/av/torchcodec`）。b) 命令补 `--policy.push_to_hub=false` 与全新 `--output_dir`。c) Windows 符号链接：优先开「开发者模式」；不开则用提供的 `scripts/train_act.py` 包装（10 行，patch `update_last_checkpoint` 成复制/忽略）。另：`lerobot` 依赖 `numpy>=2,<2.3`，本机 venv 是 2.5.3——装 lerobot 会把它降到 2.2.x（core 只要求 `>=2`，可接受，但要知道）。

**A2 · 数据集的 features 是精确契约，不是「大概的表」**
`create` 的每个特征按 dtype/shape **精确**校验（float32 必须真是 `np.float32`、形状必须 `(4,)`）。规范写法：
```python
features = {
  "observation.state":               {"dtype": "float32", "shape": (4,), "names": ["swing","boom","arm","bucket"]},
  "observation.environment_state":   {"dtype": "float32", "shape": (4,), "names": ["dig_x_m","dig_y_m","dump_x_m","dump_y_m"]},
  "action":                          {"dtype": "float32", "shape": (4,), "names": ["swing","boom","arm","bucket"]},
}
```
（`names` 不是装饰：`make_robot_action`/`build_dataset_frame`/`lerobot-replay` 全按它取名字。）`add_frame` 每帧必带 `"task"` 字符串、**不能**带 `timestamp`/`frame_index`（自动生成）；`create` 的 root 目录必须**不存在**；录完必须 `finalize()`。

**A3 · 录制改成「子步进教师 + npz 中间层」**
教师逐关键帧阻塞不可采样（发现 #9）→ 录制代理包住 driver，把每个关键帧过渡插值成 K 个子目标，每个子目标走一次 `move_joints_blocking(sub, timeout_s=BEAT_TIMEOUT)`（驱动超时**如实返回当时关节角，不抛异常**——`sim/driver.py:85-131` 已验证），每拍记录 `(发送前实测, 发送后实测)`。
分成两个脚本，**录制机不需要 torch/lerobot**：`record_demos.py`（只产 npz，跑在 AGX 侧）→ `npz_to_lerobot.py`（产 LeRobotDataset，跑在训练机）。键盘模式用同一录制器：固定间隔查 `joints` 旁观采样。两个来源统一约定 **state=本拍实测、action=下一拍实测**（位置序列即轨迹；重放时逐拍到位，形状保真、时间轴由到位速度决定）。

**A4 · 适配器直接用 `predict_action_chunk`，不走 `select_action` 队列**
lerobot 的队列是给「每步一次前向调用」准备的（`modeling_act.py:100-123`）；我们的拍是「到位」节奏，一次性拿整块更直白，也避免 100 拍后才重前向的隐性耦合。调用序列（镜像官方 `rollout/inference/sync.py:109-119`）：观测 dict → `prepare_observation_for_inference(obs, device, task="", robot_type="")` → `preprocessor` → `policy.predict_action_chunk` → `postprocessor` → 每行长 `dict`。`policy.reset()` 每条 act_exec 开始时调用（对无时序集成的配置是清空操作，为将来开集成保契约）。注意输入必须 **np.float32**（float64 会在 Linear 层炸 dtype）。

**A5 · 终止判据 = `scoop_state()` 的布尔序列，不用质量浮点**
远程桥接下 `scoop_state()` 就是服务器端实测「`bucket_mass_kg > 50`」（`agx_bridge_server.py:656-663`），mock 下是 `mark_scoop` 标志——两端语义一致。判据：**观察到 False→True（铲上）→True→False（卸掉）且至少经过一次 True** 即成功；`MAX_BEATS` 兜底判失败。`volume_m3` 实测值客户端拿不到（只在服务器 `inventory`）→ `ActResult` 用 `total=False`，不承诺它（可选增强：给后端加 `read_inventory()` 命令转发）。

**A6 · 逐拍安全：真正的守卫是驱动校验，不是 SafetyRail**
SafetyRail 只在 LLM 工具调用层按名字名单拦截（act_exec/dig 都不在名单）；它的作用体现在**外层**：act_exec 的入参（目标点）由实现自检（照 dig 的可达环带校验），失败返回 ActFailure。**逐拍**守卫来自 `SimMachineDriver.move_joints_blocking`（未知关节/非有限/超限值 → `ValueError`）。act_exec 的实现应把逐拍调用包在 try/except ValueError 里 → 转 `{"ok": False, "error": ...}`（与 dig 捕获域内拒绝的做法一致，`agx_excavator/api.py:55-56`），而不是让异常裸穿到 runner。RecoveryRail 的自动归位会先看 `holding_payload`（铲斗有料不盲目 home）——但前提是工具被它认领（见 A9）。

**A7 · YAML 的 `policy:` 节必须先在 config dataclass 加字段**
`SimMachineConfig.from_dict` 对未知键**静默丢弃**（`sim/config.py:125`）。所以顺序是：`AgxExcavatorConfig` 加 `policy: dict | None = None`（并在 `from_dict` 里校验 name/ckpt/device 类型）→ 再写 YAML（位置与 `dig_cycle_tuning` 同级，都在 `env.cfg.low_level` 下）。env 侧能力开关：`AgxExcavatorEnv._capabilities_for_config` 增加 `if self.cfg.policy: caps.add("policy.act")`。api 侧从 `env.cfg.policy` 里惰性构造 policy 对象（首次 act_exec 调用才 import torch/lerobot）。

**A8 · checkpoint 路径与加载序列定稿**
目录：`outputs/train/<date>/<run>/checkpoints/{step:06d}/pretrained_model/`，内含 `config.json`、`model.safetensors`、`policy_preprocessor.json(+.safetensors)`、`policy_postprocessor.json(+.safetensors)`（`utils/constants.py:49,58-59`；`train_utils.py:104-175`）。加载：
```python
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
cfg    = ACTConfig.from_pretrained(ckpt_dir)
policy = ACTPolicy.from_pretrained(ckpt_dir, config=cfg)   # 读 model.safetensors，自动 eval()
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=ckpt_dir)
```

**A9 · ACT_EXECUTE 必须带 `tags=("motion",)`**
RecoveryRail 按 `ActionSpec.tags` 认领（如 `motion`/`grasp`）；v1 契约草案漏了 tags → 失败后不会被自动恢复路径覆盖。补齐后 act_exec 与 dig 的失败/恢复行为逐位对齐。

---

## 三、架构兼容性与可扩展性评估（用户重点关心的部分）

| 维度 | 评估 | 依据/说明 |
| --- | --- | --- |
| 与 core 的接缝模式一致性 | ✅ | 注册表照 `sim/backend.py`（BACKENDS + register 装饰器）；能力注册用 core 专为扩展包留的 `register_capability`/`register_capability_spec`；动作走与 DIG 完全相同的四步注册链 |
| 能力门控 | ✅ | `policy.act` 走 `extra_capabilities`/`_capabilities_for_config` 现成机制；未配置 policy 的机体词表里不会出现 act_exec（工具门控 api∩env） |
| 失败语义 | ✅（补齐 tags 后） | `{"ok": False}` 与 DigFailure 同路径进 runner；RecoveryRail 认领（tags）；DiagnosisRail 无新分支 |
| 换模型（ACT→DP→VLA） | ✅ | `POLICIES` 注册表 + `Policy` 协议（`reset`/`predict`→list[dict]）；非 chunk 模型返回单元素列表即可；词表/护栏/协议/LLM 全不动 |
| 换机体（别的仿真机/真机） | ✅ | policy.act 是能力名不是机体名；sim 底座通用；任何注册了 policy.act 的机体可复用 |
| 视觉版演进 | ✅ 预留 | obs 加 `"image"` 键即可；数据集加 `observation.images.*`（dtype="image"）时 ACT 自动启用 resnet 骨干；可再细分能力 `policy.act.vision` |
| 依赖隔离 | ✳ 需设计 | 录制（AGX 侧）零新增依赖（npz）；训练放独立 venv（lerobot 会把 numpy 降到 <2.3）；部署（M4）才需要与 core 同 venv——届时锁 `lerobot==0.6.1`，装前留 requirements 快照 |
| 版本演进韧性 | ✳ 部分 | checkpoint 自描述（config+处理器随存），但 0.6.1 的 processor 化 API 与旧版不兼容 → 锁版本；适配器只依赖三个稳定面（`ACTPolicy.from_pretrained`/`predict_action_chunk`/`make_pre_post_processors`） |
| 不引入 core 改动 | ✅ | 全部落在扩展包；core 只被使用（`register_capability` 等既有扩展缝隙） |

另一个已核实、值得写进说明的**边界事实**：lerobot 自己的 rollout 不能直接用——它会丢弃不以 `.pos`/`.vel` 结尾的动作特征（`rollout/context.py:348-358`），且只支持绝对动作策略的同步引擎。这反过来说明「自写逐拍循环」不是偷懒，而是必要。

---

## 四、实施细节补强清单（v1 缺、v2 已补）

1. 子步进录制代理的接口面已核实：只需 `joint_names` / `scoop_state()` / `mark_scoop()` / `move_joints_blocking()` 四个成员（`work.py:105-169`），子步目标 = 从上一拍实测到关键帧目标的线性插值（只插涉及关节）。
2. 每拍 `BEAT_TIMEOUT` 建议起点 1.0 s（记录与执行同一常数），M2 后按实测调整；`MAX_BEATS` 建议 = 实测教师拍数 × 2。
3. 训练样本量：来源 A 自动生成，建议一次采 100–200 条（成本≈半小时级），比 v1 的 50 条更稳。
4. 观测向量顺序必须 = `cfg.joint_names`（swing, boom, arm, bucket），录制与推理同一函数取值，杜绝顺序漂移。
5. 测试矩阵：M1（注册/门控/假策略走 mock/ActFailure 流向），M2（npz→dataset 往返断言 shape/stats/finalize），M4（适配器测试用随机初始化 `ACTConfig` 构造真 policy，无 ckpt 也可测；无 lerobot 环境整测 skip）。
6. 训练侧小事：`--num_workers=0`（小数据集、Windows spawn 省事）；`HF_LEROBOT_HOME`/显式 `--dataset.root` 决定数据落盘位置。

## 五、结论

计划的**架构判断全部成立**：接缝位置、注册链、门控机制、失败语义、core 零改动、施工量级——这些都经得起源码对质。需要修的是**过程级细节**：训练命令（必失败）、录制方式（不可行）、终止判据（不可行）、逐拍安全叙述（错误）四处硬伤，加上数据格式、YAML 陷阱、tags 缺失等九处补强。硬伤都不动摇设计，只影响「按文档操作能否一次做通」。修完后的 v2（同目录 `act-integration-plan.md`）即为此报告的最终产物。
