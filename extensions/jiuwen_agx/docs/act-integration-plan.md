# ACT 模型接入计划（jiuwen_agx 挖掘机）· v2

> 状态：**M1 已实施并提交**（2026-10-01，commit 3289c00）；**M2 采集代码已实施**
> （同日：`agx_excavator/record.py` + `scripts/record_demos.py` + `scripts/npz_to_lerobot.py` +
> `tests/test_record.py`，设计论证见 `act-data-collection-design.md`；真数据待 AGX 桥接实采）；
> M3–M5 待做 · 基于 lerobot 0.6.1 源码审计 · 是
>
> **三组对接**：面向小脑组/仿真组的接口文档见 `cerebellum-interface-draft.md`
>（分工与交接面、给小脑组的 6 个对齐问题、三种对接情形与改动范围、VLA 升级路径）。
> `architecture-flows.md` 第 10 章「小脑接缝」的细化实施版。
>
> **M1 落地记录**（与本文的对应）：`policy.py`（注册表+协议+FakePolicy，§4.2）、
> `policy_act.py`（ActPolicy 惰性适配器，§4.3）、`actions.py` 的 ACT_EXECUTE（含
> `tags=("motion",)`，§4.4）、`config.py` 的 `policy`/`policy_beat_timeout_s`/
> `policy_max_beats` 字段（§4.4 补丁）、`env.py` 门控、`api.py` 的 `act_exec`
> （scoop_state 状态机判据 + 驱动校验拦截，§4.5/§4.6）、`session.py` 补
> `resource_keys`（审计时发现的既有准入缺口）、`skills/excavate_act/SKILL.md`、
> `tests/test_policy_seam.py`。已验证：不配 policy 时 act_exec 不进词表；配置后
> `jiuwensymbiosis-actions` 内省可见（capability=policy.act, tags=[motion]）。
>
> **版本说明**：v1（同日）经系统性源码审计（见同目录 `act-integration-review.md`），
> 修正 3 处错误说法（录制方式、终止判据、逐拍安全叙述）、
> 补齐 9 处过程级细节（训练命令、数据格式、checkpoint 加载、tags、YAML 陷阱等）。
> 本文所有文件行号引用均来自：lerobot 0.6.1 源码（pip wheel）、
> 本仓 core、本仓 `extensions/jiuwen_agx`。凡「已核实」字样 = 审计时对过源码。
>
> 阅读顺序：第 1→3 章讲清 ACT 是什么、lerobot 怎么用（术语先定义后用）；
> 第 4→7 章是接入设计、数据、训练、验收；第 8 章里程碑，第 9 章风险。

---

## 1. 要解决的问题

**现状。** 挖掘机目前只有一条「挖一斗」的路：`dig` 动作（`extensions/jiuwen_agx/src/jiuwen_agx/agx_excavator/work.py`）。
它是一段**手写脚本**——按 6 个关键帧（对准→举臂→下铲→收斗→摆转→卸料）逐帧下发关节角，
帧与帧的数值来自 `DEFAULT_DIG_TUNING` 常量表。它对所有目标点都走同一条固定轨迹，
没有学习能力：目标点偏了、物料形状变了，轨迹不会跟着变。

**目标。** 引入一个**通过示教数据学出来的策略**（policy）作为第二类执行方式，
与 `dig` 并存：

- **术语：策略（policy）**。一个函数：输入当前观测（关节角、目标位置……），
  输出接下来一段时间机器人各关节应该到达的目标。它不是人手写的 if/else，
  而是从数据里训练出来的神经网络。
- **术语：示教数据与模仿学习**。示教数据 = 一批「当时看到什么 → 当时做了什么」的
  记录（让会做的人/脚本做一遍，全程录下来）。模仿学习 = 用这批数据训练网络，
  使网络在新情况下复现这套动作。

**为什么选 ACT。** ACT（Action Chunking with Transformers，论文
arXiv:2304.13705）是模仿学习里最常用、最轻量的模型之一：无相机配置下
约 40M 参数（fp32 约 160 MB）、一次前向推理出整段动作、单卡数小时可训、
CPU 也能推理。我们的场景（4 个关节 + 4 维目标，无相机）正是它最省力的工作点。

**成功标准（全文围绕这三条展开）。**

1. `act_exec`（新动作）在仿真里对**没见过**的挖掘点/卸料点完成挖运循环，
   铲斗载重过 50 kg 阈值（与 `dig` 同一判据，服务器端实测
   `bucket_mass_kg > LOADED_MASS_THRESHOLD_KG`，`agx_bridge_server.py:656-663`）；
2. 护栏、失败路径、可观测性与 `dig` 完全一致（同一条 `move_joints_blocking` 下行通道）；
3. core 包零改动，全部新代码落在 `extensions/jiuwen_agx`。

---

## 2. ACT 原理：只需要懂的四件事

### 2.1 动作块（action chunk）

- **术语：动作块（action chunk）**。模型一次前向输出的不是下一个拍点的动作，
  而是**接下来 C 个拍点的完整序列**（C 叫 chunk_size，ACT 默认 100）。
  好处：一次网络前向摊薄到 C 个执行拍，推理开销小；相邻拍之间天然连贯。
- **术语：拍（beat）**。我们的执行单位：下发一组**绝对关节目标**并等它到位
  （`driver.move_joints_blocking`）。拍与拍之间隔的是「到位时间」不是固定秒数
  ——ACT 的时间轴在我们这里指**拍序号**，不是墙钟时间（详见 4.5 的语义决策）。

### 2.2 网络：一个 CVAE + 一个 Transformer

- **术语：Transformer**。一种神经网络结构，把「一组输入（每个是一段向量）」
  通过注意力机制互相参照后变换成「一组输出」。ACT 用它把观测翻译成动作块。
- **术语：CVAE（条件变分自编码器）**。训练时额外用的一部分网络：把「目标动作块」
  压缩成一个低维向量（隐变量 z），强迫主 Transformer 从 (观测, z) 重建动作
  ——让模型学会「同一观测可能对应几种合理做法」，避免对多种示教风格取平均后
  畏手畏脚。**推理时 z 取全零**（`modeling_act.py:454-458`，已核实），输出确定。

### 2.3 训练目标

损失 = 动作重建误差（L1）+ KL 散度项（kl_weight 默认 10）。公式不用我们改，
lerobot 全部实现好（`modeling_act.py:137-164`）。

### 2.4 推理行为：动作队列

lerobot 的 `select_action()` 内部维护一个队列：第一次查询做一次前向、整块入队；
之后每次调用弹出一个动作，队列空了才再次前向（`modeling_act.py:100-123`，已核实）。
可选的**时序集成**（每拍重新前向、新旧预测指数加权平均）默认关闭。
**我们的适配器不用这个队列**：直接调 `predict_action_chunk()` 一次拿整块
（理由与调用序列见 4.3）。

---

## 3. lerobot 的 ACT 实现：源码地图（0.6.1 已逐文件核实）

[lerobot](https://github.com/huggingface/lerobot) 是 Hugging Face 的机器人学习库，
它的 ACT 是官方论文实现（Tony Zhao 授权）的移植版。以下路径均相对包根 `lerobot/`。

### 3.1 三个源文件

| 文件 | 内容 | 行数 |
| --- | --- | --- |
| `policies/act/configuration_act.py` | `ACTConfig`：全部超参数与输入输出特征声明 | 176 |
| `policies/act/modeling_act.py` | `ACTPolicy`（门面）、`ACT`（网络本体）、`ACTTemporalEnsembler` | 748 |
| `policies/act/processor_act.py` | 前处理/后处理管线（归一化、设备搬运、批处理） | 50 |

### 3.2 ACTConfig 里我们关心的字段

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `chunk_size` | 100 | 一次前向输出的动作拍数 |
| `n_action_steps` | 100 | 每块里实际执行多少拍（≤ chunk_size） |
| `n_obs_steps` | 1 | 只许为 1（源码强制，多帧观测未实现） |
| `dim_model` / `n_encoder_layers` / `n_decoder_layers` | 512 / 4 / 1 | Transformer 规模 |
| `use_vae` / `latent_dim` / `kl_weight` | True / 32 / 10 | CVAE 开关与参数 |
| `vision_backbone` | resnet18 | 有图像输入时才构建；无图像配置完全不加载 |
| `normalization_mapping` | STATE/ACTION: MEAN_STD | 归一化方式，统计量来自数据集，自动处理 |

**输入输出硬约束**（`validate_features()`，`configuration_act.py:162-164`，已核实）：

1. 输入必须至少有一路 `observation.images.*` **或** `observation.environment_state`
   ——**纯关节角是不允许的**。且 `environment_state` 必须是**精确键名**
   且被识别为「环境状态」类型（`configs/policies.py:141-148`；
   `utils/feature_utils.py:168-175` 的映射规则：`observation.environment_state`→ENV，
   其余 `observation.*`→STATE）。这决定了 4.3 的观测映射设计。
2. 输出必须有 `action`。
3. `n_action_steps ≤ chunk_size`；开时序集成时 `n_action_steps` 必须为 1。
4. **`observation.state` 也必须存在**：即使 `use_vae=False`，网络前向仍解引用
   `batch[OBS_STATE]`（`modeling_act.py:456-458`，已核实）。四路特征缺一不可。

### 3.3 batch 键名与我们的观测对照

| 键 | 角色 | 我们的对应物（数据集里就是这个键名） |
| --- | --- | --- |
| `observation.state` | 机器人本体状态（必存） | 4 关节当前值 `[swing, boom, arm, bucket]` |
| `observation.environment_state` | 环境/任务条件（必存其一） | 4 维目标 `[dig_x_m, dig_y_m, dump_x_m, dump_y_m]`（米） |
| `observation.images.<cam>` | 可选相机 | 二期视觉版 |
| `action` | 训练必须 | 下一拍的 4 关节目标 |

单位说明：`swing` 是弧度（CabinHinge），`boom/arm/bucket` 是米（液压缸柱塞行程），
即 AGX 桥接 `joints` 命令的原生单位。**原样记录、原样喂入**，尺度差异交给
MEAN_STD 归一化。观测向量分量顺序必须恒等于 `cfg.joint_names`
（swing, boom, arm, bucket）——录制与推理走同一个取数函数。

### 3.4 数据集：LeRobotDataset v3.0（`datasets/lerobot_dataset.py`）

```python
ds = LeRobotDataset.create(repo_id, fps=10, features={...}, root=..., use_videos=False)
ds.add_frame({...四个特征..., "task": "excavate"})   # 每拍一帧
ds.save_episode()                                    # 每条示教结束
ds.finalize()                                        # 全部录完（不调则读不了！）
```

精确契约（全部已核实，`datasets/feature_utils.py:221-243,300-327`、
`datasets/dataset_writer.py:202-228`、`datasets/dataset_metadata.py:829-832`）：

- `features` 的 dtype/shape **精确校验**（float32 必须真是 `np.float32`，形状必须 `(4,)`）：

  ```python
  features = {
    "observation.state":             {"dtype": "float32", "shape": (4,), "names": ["swing","boom","arm","bucket"]},
    "observation.environment_state": {"dtype": "float32", "shape": (4,), "names": ["dig_x_m","dig_y_m","dump_x_m","dump_y_m"]},
    "action":                        {"dtype": "float32", "shape": (4,), "names": ["swing","boom","arm","bucket"]},
  }
  ```
  `names` 不是装饰：lerobot 的动作回放/帧组装工具都按它取名字。
- `add_frame` 每帧**必须带** `"task"` 字符串；**不许传** `timestamp`/`frame_index`
  （自动生成：`timestamp = frame_index / fps`）。
- `create` 的 root 目录**必须不存在**（`exist_ok=False`）；`use_videos=False` +
  无数码特征 → 纯 Parquet，**无 ffmpeg/视频依赖**（Windows 友好）。
- 统计量（归一化用均值/方差）**自动**：每 `save_episode` 累计一次，
  落在 `meta/stats.json`（`dataset_writer.py:322-324`）——没有独立「算统计」步骤。
- **训练时的 chunk 也自动构造**：训练脚本按 `policy.config.action_delta_indices`
  （= `range(chunk_size)`）生成 `delta_timestamps`（i/fps 秒，`datasets/factory.py:34-66`），
  样本自动成 `action (chunk,4)` + `action_is_pad (chunk,)` 填充掩码
  （`dataset_reader.py:215-232`）。我方零代码。

### 3.5 训练入口：`lerobot-train`（`scripts/lerobot_train.py`）

```bash
lerobot-train \
  --dataset.repo_id=local/agx_dig_demos --dataset.root=<数据目录> \
  --policy.type=act --policy.push_to_hub=false \
  --policy.device=cuda \
  --output_dir=outputs/train/act_agx_v1 \
  --batch_size=8 --num_workers=0 --steps=50000 --save_freq=10000
```

三个**必写**参数（否则直接报错，均已核实）：
`--policy.push_to_hub=false`（默认 True，没有 repo_id 时校验抛错，
`configs/train.py:283-289`）；`--output_dir` 指向**不存在**的目录（已存在即报错，
`configs/train.py:259-263`）；训练依赖 `accelerate`（`lerobot_train.py:231`
要求装 `lerobot[training]`）。Windows 上 `--num_workers=0` 省去 spawn 麻烦。
其余超参用 `--policy.<字段>=值` 覆盖（draccus 机制，`ACTConfig` 注册名 `act`）。

### 3.6 checkpoint 结构与加载（M4 用它）

每次存档目录：`outputs/train/.../checkpoints/{step:06d}/pretrained_model/`
（`utils/constants.py:49`），内含：

```
config.json                                   # ACTConfig 超参（自描述）
model.safetensors                             # 网络权重
policy_preprocessor.json  + step_3_*.safetensors   # 归一化统计量在里面
policy_postprocessor.json + step_0_*.safetensors
```

加载序列（`policies/factory.py:150-237`，已核实）：

```python
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

cfg    = ACTConfig.from_pretrained(ckpt_dir)
policy = ACTPolicy.from_pretrained(ckpt_dir, config=cfg)   # 读权重，自动 eval()
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=ckpt_dir)
```

一个训练与推理**必须**用同一份 checkpoint 目录（权重+config+统计量三者一体），
不允许手工搬统计量。

### 3.7 官方推理循环（`rollout/inference/sync.py`，我们的镜像对象）

```python
obs = prepare_observation_for_inference(obs, device, task="", robot_type="")  # 批维+设备+dtype
obs = preprocessor(obs)              # 归一化
act = policy.predict_action_chunk(obs)   # (1, chunk, 4) 已归一化
act = postprocessor(act)             # 反归一化回原生单位
```

两条已核实的边界事实，说明**为什么我们自己写循环而不是用 lerobot 的 rollout**：
(a) lerobot 的 rollout 会**丢弃**不以 `.pos`/`.vel` 结尾的特征
（`rollout/context.py:348-358`），我们的关节名 `swing/boom/arm/bucket` 会被静默清空；
(b) 它的同步引擎要求绝对动作且按固定节拍推理，不匹配「逐拍到位」语义。
自写循环 + 上述四个调用即可，行为与官方一致。

### 3.8 轻量性结论

- **模型体积**：无相机配置 ≈ 40M 参数（估算：9 层 dim=512 的 Transformer
  + VAE 编码器），fp32 约 160 MB；带 resnet18 视觉骨干再加 ~11M。
- **推理成本**：单块一次前向，CPU 毫秒级到几十毫秒，不是 10 Hz 拍频的瓶颈。
- **训练成本**：4 维动作小任务，2 万~5 万 step、batch 8，单 GPU 小时级。
- **依赖代价**：基础安装自带 torch/torchvision（约 2~3 GB CUDA 版）；
  训练额外要 `accelerate` 与数据集栈（见 6.1）。

---

## 4. 接入设计（jiuwen ↔ ACT 之间）

### 4.1 总览：改动清单

```
LLM 规划层   不动（看到的是新动作 act_exec 的契约，与 dig 无异）
护栏/协议层  不动（策略输出走 move_joints_blocking 同一条咽喉）
jiuwen_agx   + contracts.py: ActResult|ActFailure（与 DigResult 同形）
             + policy.py: 注册表 + Policy 协议（接缝，~40 行；不 import torch）
             + policy_act.py: ActPolicy 适配器（lerobot 翻译层，~90 行；惰性 import）
             + actions.py: ACT_EXECUTE 契约（~30 行，含 tags=("motion",)）
             + agx_excavator/api.py: @implements(ACT_EXECUTE) 薄包（~50 行）
             + agx_excavator/config.py: policy 字段（~10 行）
             + agx_excavator/env.py / __init__.py: 注册链与能力门控（~10 行）
             + skills/excavate_act/SKILL.md（何时用 act_exec 何时用 dig）
             + scripts/record_demos.py（npz 采集，零新依赖，~150 行）
             + scripts/npz_to_lerobot.py（转换，训练机跑，~100 行）
             + tests/test_policy_seam.py / test_record_conversion.py（~120 行）
core         零改动
```

预估总量 ~350 行 Python + 1 个 SKILL.md，全在扩展包内。新增依赖进扩展包
`pyproject.toml` 的**可选 extra**（`[project.optional-dependencies] policy = ["lerobot==0.6.1"]`），
安装扩展包本身不拉 torch。

### 4.2 Policy 接缝（新模块 `jiuwen_agx/policy.py`，不 import torch）

照 `sim/backend.py` 的注册表范式：

```python
POLICIES: dict[str, type] = {}

def register_policy(name: str):
    def deco(cls):
        POLICIES[name] = cls
        return cls
    return deco

class Policy(Protocol):
    def reset(self) -> None: ...
    # 一个动作块 = 若干拍;每拍是 {关节名: 绝对目标值(原生单位)}
    def predict(self, obs: dict) -> list[dict[str, float]]: ...
```

`obs` 键约定（适配器负责翻译成 lerobot 键名）：

```python
obs = {
    "joints": {"swing": .., "boom": .., "arm": .., "bucket": ..},  # get_joint_positions() 原样
    "goal":  {"dig_x_m": .., "dig_y_m": .., "dump_x_m": .., "dump_y_m": ..},  # act_exec 入参
    # "image": <ndarray>   # 二期视觉版增加,接缝本身不排斥
}
```

`predict` 返回**整块列表**而不是逐拍：与 ACT 一次前向出整块对齐；
非 chunk 模型（假策略、脚本策略）返回单元素列表即可——换模型不动执行循环。
未知 policy 名 → 报错列出已注册名与安装提示（extra 未装时给可读原因）。

### 4.3 ActPolicy 适配器（`jiuwen_agx/policy_act.py`）

```python
@register_policy("act")
class ActPolicy:
    def __init__(self, ckpt_dir: str, device: str = "cpu"):
        import torch                                    # 惰性:模块导入不碰 torch
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        # 3.6 的三行加载 (cfg/policy/pre+post),全部存为实例字段

    def reset(self):
        self._policy.reset()                            # 清 lerobot 内部队列(契约对齐)

    def predict(self, obs) -> list[dict[str, float]]:
        import numpy as np, torch
        # 1) obs → lerobot batch(键名见 3.3;顺序 = cfg.joint_names;dtype 必须 float32):
        #      "observation.state"             = np.float32([joints[n] for n in JOINT_ORDER])
        #      "observation.environment_state" = np.float32([goal[n]  for n in GOAL_ORDER])
        # 2) prepare_observation_for_inference(batch, device, task="", robot_type="")
        #      → 批维 + 上设备 + 类型;随后 preprocessor 归一化(不会重复加批维)
        # 3) with torch.inference_mode():
        #      chunk = self._policy.predict_action_chunk(pre(batch))   # (1, C, 4) 已归一化
        #      chunk = post(chunk)                                     # 反归一化回原生单位
        # 4) → [{name: float(chunk[0, i, k]) for k, name in enumerate(JOINT_ORDER)}
        #        for i in range(chunk.shape[1])]
```

必须写对的四个点（全部来自已核实源码）：

1. **goal → `observation.environment_state`**（3.2 约束 1 的硬要求）；四个特征
   在训练 batch 里必须齐全（约束 4）。
2. **dtype 必须 np.float32**：float64 会在网络 Linear 层炸 dtype 不匹配。
3. **用 `predict_action_chunk` 而不是 `select_action`**：后者的 100 拍内部队列
   是为「每步一次调用」设计的，与「逐拍到位」节奏错配；直取整块还顺带绕开
   「不满 100 拍不重前向」的隐性耦合。`reset()` 照调（对未开时序集成的配置是空操作，
   保契约以便未来开时序集成时零改动）。
4. **顺序恒定**：state/action 分量顺序 = `cfg.joint_names`；goal 顺序 = 上表四位，
   训练脚本与适配器共用同一常量。

### 4.4 新动作 `act_exec` 的注册链（照抄 dig 的四步 + 两个补丁）

```python
# ① contracts.py: ActResult | ActFailure(即 runner 的 {"ok":...} 形状,见下)
# ② jiuwen_agx/__init__.py 注册链最前面(顺序敏感,ActionSpec 构造期校验能力):
register_capability("policy.act")
register_capability_spec("policy.act", actions=["act_exec"])   # 供 adapter 校验器 A-10
# ③ actions.py:
ACT_EXECUTE = ActionSpec(
    name="act_exec", capability="policy.act",
    params=("dig_x_m", "dig_y_m", "dump_x_m", "dump_y_m"),   # 与 dig 同参,规划器可互换
    required_params=("dig_x_m", "dig_y_m", "dump_x_m", "dump_y_m"),
    param_schema={...同 DIG 的四项中文说明...},
    requires=("payload.clear",), invalidates=("body.home",),
    invalidates_locations=True,          # 挖掘同样重塑地形
    tags=("motion",),                    # ★ RecoveryRail 按 tags 认领(DIG 同款)
    result=ActResult | ActFailure,
)
register_actions(ACT_EXECUTE)
# ④ agx_excavator/api.py: @implements(ACT_EXECUTE) 的 act_exec → 4.5 的执行循环
```

契约形状（与 DIG 对齐）：

```python
class ActFailure(TypedDict):
    ok: Literal[False]
    error: str

class ActResult(TypedDict, total=False):
    ok: Literal[True]
    beats: int          # 实际执行拍数(规划器可据此比较代价)
    cycle_s: float
    # volume_m3 不承诺:客户端拿不到实测体积(只在服务器 inventory) — total=False 允许省略
```

两个**补丁**（v1 漏掉、审计发现）：

- **`tags=("motion",)`**：RecoveryRail 只认领 tags 含 `motion`/`grasp` 的工具；
  漏写则失败后不进入与 dig 相同的恢复路径。
- **YAML 陷阱**：`SimMachineConfig.from_dict` 对未知键**静默丢弃**
  （`sim/config.py:125`）。必须先在 `AgxExcavatorConfig` 加字段
  `policy: dict | None = None`（`from_dict` 里校验 name/ckpt/device），
  YAML 才认（位置与 `dig_cycle_tuning` 同级，都在 `env.cfg.low_level` 下）：

  ```yaml
  env:
    cfg:
      low_level:
        policy:            # 不配 = 无 policy.act 能力 = act_exec 不进词表
          name: "act"      # POLICIES 注册名
          ckpt: "outputs/train/act_agx_v1/checkpoints/000050/pretrained_model"
          device: "cpu"
  ```

  能力门控：`AgxExcavatorEnv._capabilities_for_config` 增加
  `if self.cfg.policy: caps.add("policy.act")`——未配置的机体词表里没有 act_exec
  （工具门控 api∩env，机制已核实：`tools/builder.py:54-61`）。
  policy 对象**惰性构造**（首次 act_exec 调用才加载权重），会话建立不背 torch 启动成本。

### 4.5 执行循环与节拍语义（关键设计决策）

`act_exec` 实现（伪代码，全部经审计修正）：

```python
def act_exec(self, dig_x_m, dig_y_m, dump_x_m, dump_y_m):
    cfg, driver = ...
    # 0) 域内自检(照 dig):目标在可达环带内、斗空 — 超出返回 ActFailure
    policy = self._get_policy()                      # 惰性构造,失败→ActFailure(原因可读)
    obs0 = {"joints": driver.get_joint_positions(), "goal": {...四个入参...}}
    policy.reset()
    queue, beats, max_beats = deque(policy.predict(obs0)), 0, cfg.policy_max_beats
    was_loaded = False                               # 载重判据的状态机(见下)
    while queue or beats < max_beats:
        if not queue:                                # 块用尽 → 重新观测再前向
            obs = {"joints": driver.get_joint_positions(), "goal": {...}}
            queue = deque(policy.predict(obs))
        target = queue.popleft()
        try:
            driver.move_joints_blocking(target, timeout_s=cfg.policy_beat_timeout_s)
        except ValueError as exc:                    # ★逐拍的真守卫=driver校验(见4.6)
            return {"ok": False, "error": f"策略输出非法(第{beats}拍): {exc}"}
        beats += 1
        loaded = driver.scoop_state() if hasattr(driver, "scoop_state") else False
        if loaded: was_loaded = True
        elif was_loaded:                             # True→False:铲上过、又卸掉了
            return {"ok": True, "beats": beats, "cycle_s": elapsed()}
    return {"ok": False, "error": f"{max_beats}拍内未完成挖卸(铲斗{'有' if was_loaded else '无'}料)"}
```

**终止判据（v1 的「bucket_mass>50 且 at_dump_pose」作废，两条都不可行）**：
客户端没有 float 质量接口（后端只有布尔 `scoop_state()`，`sim/backend.py:521-522`；
float 只在服务器 `inventory`）；也无「卸料姿态」判定。可行且严格的判据是
**布尔状态机**：观察到 `False→True`（铲上）后再 `True→False`（卸掉）才算一次完整循环
——它背后正是服务器实测的 `bucket_mass > 50`（`agx_bridge_server.py:656-663`），
mock 下是 `mark_scoop` 标志，两端语义一致。`max_beats` 兜底失败时把
「有没有铲上过」写进 error 文案，规划器/人一眼可读。

| 决策点 | 取值 | 理由 |
| --- | --- | --- |
| 拍的语义 | 绝对关节目标序列；拍间隔 = 到位时间（不固定秒数） | 位置控制下轨迹形状与时序解耦；与采集端同一语义 |
| 每拍超时 | `policy_beat_timeout_s`，起点 **1.0 s**（记录/执行同一常数） | 驱动超时如实返回、不抛异常；M2 实测后调整 |
| 块用尽策略 | 重新观测 + 重前向（不开时序集成） | 与「n_action_steps=chunk_size + 边界重观测」等价；简单可控 |
| `max_beats` | **= 实测教师拍数 × 2**（M2 后定，暂 300） | 兜底不误杀正常循环 |
| 时序集成 | 先不开 | 挖掘是自由空间运动；需要时配置一行（`temporal_ensemble_coeff=0.01`） |

### 4.6 安全与可观测性（审计后的准确版）

**逐拍的真实守卫是驱动校验，不是 SafetyRail**（v1 此处写错，已修正）：

- SafetyRail 只按**固定工具名名单**拦截 LLM 工具调用：`motion.joint → {move_joint,
  move_named_joint}`、`motion.base → {navigate_relative, rotate_base, drive_arc}` 等
  （`rails/safety.py:57-64,70`）。`dig`/`act_exec` **都不在名单**，逐拍的
  `move_joints_blocking` 是驱动调用而非工具调用——SafetyRail 看不到它们。
- 真实防线在 `SimMachineDriver.move_joints_blocking`：未知关节名、非有限值、
  超出 `joint_limits` → `ValueError`（`sim/driver.py:98-121`，已核实）。
  策略网络不提供任何硬保证，所以这条校验就是策略输出的最后闸门。
  act_exec 循环内 **catch ValueError → ActFailure**（原因带拍号与关节名），
  与 dig 捕获域内拒绝的做法一致（`agx_excavator/api.py:55-56`），
  不让异常裸穿到 runner 的错误码路径。
- **外层**：act_exec 的入参（目标点）由实现自检（可达环带，复用 dig 的校验），
  失败返回 ActFailure；这一层与 dig 行为一致。
- **失败语义**：runner 把动作返回值 `{"ok": False}` 判为步骤失败
  （`runner.py:644-648`，已核实）→ ActFailure 与 DigFailure 走同一条
  停止/重规划路径；加上 `tags=("motion",)` 后 RecoveryRail 同样认领；
  RecoveryRail 归位前会看 `holding_payload`（铲斗有料不盲目 home）。
- **可观测性**：每次 act_exec 的 beats/终态关节进 ActResult；逐拍细节
  （拍了什么、到了哪里）打日志，由 TraceLogHandler 汇入执行轨迹（既有机制）。
- **一个必须遵守的数据一致性纪律**：训练数据与部署必须用**同一份 config**
  （同单位、同关节限位、同 backend）。mock 数据与 remote 数据不得混用
  （真值/占位值单位体系不同）。

---

## 5. 数据采集方案

### 5.1 两个示教来源、两个脚本、零新增依赖

| 来源 | 谁在动 | 采集脚本 | 依赖 |
| --- | --- | --- | --- |
| A. 脚本教师 | 现成 `execute_dig_cycle`（目标点扰动） | `record_demos.py --teacher` | 零（不 import lerobot） |
| B. 键盘示教 | 人按 `JIUWEN_KEYBOARD=1` 键位（官方 365 键位，`agx_bridge_server.py:320-328`） | `record_demos.py --keyboard`（旁观） | 零 |

采集统一产出 **npz**（`state[t]` 四关节 + `goal[4]` + `scoop_state` 序列 + 元数据），
转换用 `npz_to_lerobot.py` 产出 LeRobotDataset。**录制机与训练机解耦**：
AGX 侧不需要 torch/lerobot；训练机不需要 AGX。键盘模式可行性的依据：
键盘在 AGX 进程内驱动机器（不经桥接命令通道），服务器的命令循环可并行响应
`joints` 查询——旁观采样不会阻塞它。

### 5.2 录制器设计（v1 的「挂调用点」方案作废，改为子步进）

**来源 A：子步进教师。** 教师逐关键帧阻塞数秒（每循环仅 6 次阻塞调用），
按调用点采样每循环只得 ~12 帧——不可行。改为**录制代理**包住 driver，
把每个关键帧过渡插值成 K 个子目标（起点 = 上一拍实测位置，终点 = 该关键帧目标，
只插该帧涉及的关节），每个子目标执行一次：

```python
state_after = driver.move_joints_blocking(sub_target, timeout_s=BEAT_TIMEOUT)
# 超时如实返回当时关节角,不抛异常(已核实 sim/driver.py:85-131)
record(state_before, sub_target_executed=state_after)
```

代理只需四个成员（与 `work.py:105-169` 的实际调用面一致）：
`joint_names` / `scoop_state()` / `mark_scoop()` / `move_joints_blocking()`。
K 默认 20/过渡（6 段 ≈ 120 拍/循环，恰好与 chunk_size=100 同量级），M2 实测后调。

**来源 B：旁观采样。** 固定间隔（0.1 s）查 `joints`，记录实测序列；
键盘操纵含失败尝试的回合靠成功判据整条剔除。

**两来源统一记录约定**：`state[t] = 本拍实测关节`，`action[t] = 下一拍实测关节`
（位置序列 = 轨迹；重放时逐拍到位，形状保真、时间轴由到位速度决定）。
来源 A 的子步目标执行后要么到位要么被超时截断，两种情况下「下一拍实测」都是
真实可达的下一个位形——这正是位置控制下最稳的监督信号。

### 5.3 转换：npz → LeRobotDataset（`npz_to_lerobot.py`）

- features 表 = 3.4 的精确三行（names 列表照抄）；`fps=10`（**名义值**：
  fps 只影响 delta 秒数与时间戳元数据，拍的真实节奏由到位速度决定）。
- 每拍 `add_frame(np.float32(state), np.float32(env_state=goal), 下一拍实测, task="excavate")`；
  每条 `save_episode()`；全部结束 `finalize()`。
- 转换器对每条 npz 做成功判据过滤（5.4），不合格整条丢弃并计数报告。
- 抽查：`lerobot-dataset-viz`（Rerun 时间序列视图，无需视频，已核实支持）。

### 5.4 数量与质量判据

| 项 | 建议值 | 说明 |
| --- | --- | --- |
| 条数 | **100–200 条**（v1 的 50 条上调） | 来源 A 自动生成，成本≈半小时级 |
| 目标点分布 | 挖掘点/卸料点在可达环带内分层随机 | 覆盖推理输入域 |
| 硬判据 | `scoop_state` 序列含 `False→True→False`（铲上又卸掉） | 与 4.5 运行时判据同款 |
| 剔除 | 卡死/超时/从未铲上 | 来源 B 的主要用途 |
| 元数据 | 每条记 beat 数（用于定 chunk_size 与 max_beats） | 转换器打印统计 |

---

## 6. 训练方案

### 6.1 环境（Windows 本机）

```bash
pip install "lerobot[training]==0.6.1"    # 含 accelerate/wandb/datasets 栈;自带 torch/torchvision
```

三个已核实的注意点：

1. **numpy 降级**：lerobot 要求 `numpy>=2,<2.3`，本机 venv 现为 2.5.3 →
   安装时会被降到 2.2.x（core 只要求 `>=2`，可接受；但建议**训练用独立 venv**，
   避免动主开发环境的版本）。
2. **Windows 符号链接**：lerobot 每次存 checkpoint 都建 `last` 符号链接
   （`common/train_utils.py:96-101`），无符号链接特权（WinError 1314，本仓测试已踩过）
   会崩。两条路：开 Windows「开发者模式」（推荐，一步到位）；或提供
   `scripts/train_act.py` 包装脚本（10 行：把 `update_last_checkpoint`
   替换为忽略/复制，再转发 CLI）。
3. **torchcodec/av** 在 `[training]` 链路里随 `[dataset]` extra 安装；纯数值数据集
   （无视频特征）不触发其视频路径。若安装报错，备用最小清单：
   `lerobot datasets pandas pyarrow jsonlines accelerate`。

### 6.2 训练配置（起点值）

```bash
lerobot-train \
  --dataset.repo_id=local/agx_dig_demos --dataset.root=E:/lzm/2026/jiuwensymbiosis/extensions/jiuwen_agx/data/agx_dig_demos \
  --policy.type=act --policy.push_to_hub=false --policy.device=cuda \
  --output_dir=outputs/train/act_agx_v1 \
  --batch_size=8 --num_workers=0 --steps=50000 --save_freq=10000
```

- 无图像 → 视觉骨干不构建（`ACT.__init__` 有守卫，已核实）；默认声明留在 config 里无副作用。
- chunk_size 保持 100（默认）；若 M2 实测单循环拍数远超 100，先靠 4.5 的
  「块用尽重前向」承担，不动 chunk。
- 归一化统计量训练开始时自动从数据集取（`dataset.meta.stats`），随 checkpoint 保存。
- 复训/微调：`--resume` + 原 output_dir（此时允许已存在）。

### 6.3 算力与时长的现实预期

| 阶段 | 规模 | 预期 |
| --- | --- | --- |
| 数据生成（来源 A） | 100–200 循环，每循环 ~120 拍 | 脚本自动跑，≤ 1–2 小时（含仿真实时性） |
| 训练 | 50k step, batch 8, 4 维动作 | GPU 小时级；CPU 过夜级 |
| 训练后离线检查 | loss 曲线 + 数据集抽查 + 仿真回放 | 半小时 |

### 6.4 训练质量的三个门槛（进入第 7 章的前置）

1. `l1_loss` 收敛到与动作量纲相称的水平（反归一化后平均每拍偏差 ≪ 关节行程 5%）；
2. **留出目标点**（训练集没出现过的 dig/dump 坐标）仿真回放 3 条完整循环肉眼过关；
3. 归一化统计量抽查：swing（rad）与液压缸（m）各维 std 非退化（>1e-6），
   防止某一维被当成常数学没。

---

## 7. 上线与验收

### 7.1 顺序：mock 先行，真桥接后上

1. **mock 后端 + 假策略**（不 import torch 的 `FakePolicy`：返回往合法关节角插值的块）：
   验证注册链、能力出现（`jiuwensymbiosis-actions --config` 能看到 act_exec）、
   门控（不配 policy 则没有）、ActFailure 流向、逐拍校验拦截——不依赖训练产物。
2. **mock 后端 + 真 checkpoint**：ActPolicy 加载真模型跑通调用链（需要 lerobot 环境）。
3. **真桥接**：端到端 `--query "用 ACT 策略挖一斗运到卸料点"`。

### 7.2 验收标准表

| # | 标准 | 通过线 |
| --- | --- | --- |
| 1 | 留出目标点上完成挖运循环 | 10 个留出点 ≥ 7 个成功（scoop_state 序列判据） |
| 2 | 与 dig 基线对比 | 成功率不要求超过 dig；`cycle_s`/拍数同量级、轨迹无越限 |
| 3 | 非法输出拦截 | 假策略故意输出越限关节目标 → 驱动 ValueError → ActFailure（原因可读），无异常裸穿 |
| 4 | 失败路径 | max_beats 兜底触发 ActFailure → runner 判步失败/重规划；RecoveryRail 认领（tags）且不盲目 home 载料机体 |
| 5 | 能力过滤 | 未配置 policy 的机体上 act_exec 不进词表；`policy.act` 出现在配置了 policy 的机体词表 |
| 6 | 回归 | `pytest tests/unit_tests/`（core）+ `pytest extensions/jiuwen_agx/tests` 全绿；新增测试见 8 章 |

### 7.3 对比实验设计（组会用）

同一批留出目标点，`dig` 与 `act_exec` 各跑 N 次，记录：成功率、`cycle_s`、
拍数、每拍到位偏差（终态关节 vs 下发目标）；（视觉版后）实际挖走体积。
ACT 的价值主张是**目标泛化与轨迹多样性**，不是单点效率——对比表按目标点分组
看成功率分布更有说服力。

---

## 8. 里程碑（对齐 architecture-flows 10.5 的 ①②③④，细化）

| 里程碑 | 交付物 | 验收 | 预估量 |
| --- | --- | --- | --- |
| **M1 接缝 + 假策略** | `policy.py`、`policy_act.py` 骨架（惰性 import）、ActResult 契约、ACT_EXECUTE（含 tags）、config 字段、YAML policy 节、env 门控、SKILL.md、`FakePolicy`、`test_policy_seam.py` | 7.2 #3/#5 + mock 下全链通、ActFailure 流向正确 | ~180 行 + SKILL.md |
| **M2 采集** | `record_demos.py`（子步进代理 + 键盘旁观；npz）、`npz_to_lerobot.py`、`test_record_conversion.py`（小 npz 往返断言） | 100–200 条成功循环落盘；`lerobot-dataset-viz` 抽查；beat 数统计出表 | ~250 行 |
| **M3 训练** | 训练 venv + 6.1/6.2 配置 + `scripts/train_act.py`（若需） | 6.4 三门槛通过 | 配置为主 |
| **M4 上线** | ActPolicy 真 ckpt（3.6 加载序列）、真桥接端到端 | 7.2 #1/#2/#4；`act_exec` 与 `dig` 结果对读 | 适配器补完 ~60 行 |
| **M5（可选）视觉版 + move_chunk** | 桥接 `grab_frame` 实现、`observation.images.wrist` 接入（dtype="image"）、`{"cmd":"move_chunk"}` 批量下发 | 视觉 ACT 上线；块单往返 | 另立计划 |

M1–M4 全部在 `extensions/jiuwen_agx` 内，core 零改动。

---

## 9. 风险与对策

| 风险 | 依据（已核实） | 对策 |
| --- | --- | --- |
| ACT 拒绝纯状态输入 | `configuration_act.py:162-164` | 已按 env_state 条件化设计（4.3）；M1 假策略用同键名，最早暴露 |
| 驱动力守卫之外的越限输出 | 驱动校验是唯一逐拍硬闸（4.6） | 驱动 ValueError → ActFailure；验收 #3 专项测 |
| Windows 训练崩溃：checkpoint 符号链接 | `train_utils.py:96-101` 无条件 `symlink_to` | 开发者模式，或 `train_act.py` 包装（6.1） |
| 装 lerobot 降 numpy 2.5.3→2.2.x | lerobot METADATA：`numpy<2.3,>=2.0` | 训练用独立 venv；部署同 venv 前留 requirements 快照 |
| 训练命令少参数直接报错 | push_to_hub/output_dir/accelerate 三处校验 | 6.2 的命令已含全部必填项 |
| YAML 的 policy 节被静默忽略 | `sim/config.py:125` 未知键丢弃 | 先改 dataclass 再写 YAML（4.4 补丁）；M1 测试断言配置生效 |
| 拍语义错位（录制固定采样 vs 执行到位） | 位置控制时间轴 ≠ 墙钟 | 4.5 定的「绝对目标拍序列」；验收 #1 留出点直接检验 |
| chunk 边界跳变 | 块间独立前向 | 先看回放；明显则块间重叠或开时序集成（配置一行） |
| 示教分布窄（来源 A 单一风格） | 模仿学习固有限制 | 目标点分层随机；不行就上来源 B 补多样性 |
| lerobot 版本演进破坏适配器 | 0.6.1 的 processor 化 API 与旧版不兼容 | 锁 `lerobot==0.6.1`；适配器只依赖三个稳定面（3.6） |
| mock/remote 数据混用 | 单位体系不同 | 数据一致性纪律（4.6 末条）；转换器记录来源 backend 供筛查 |

---

## 10. 待定决策点（开工前拍板）

| # | 决策 | 当前推荐 | 何时定 |
| --- | --- | --- | --- |
| 1 | act_exec 参数集 | 与 dig 同参（规划器零学习成本） | 已定 |
| 2 | `BEAT_TIMEOUT` 具体值 | 1.0 s 起步（记录/执行同一常数） | M2 实测后 |
| 3 | `MAX_BEATS` | = 实测教师拍数 × 2 | M2 后 |
| 4 | 每过渡子步数 K | 20/过渡 | M2 后看 beat 统计 |
| 5 | chunk_size 是否上调 | 保持 100，先靠块用尽重前向 | M2 后 |
| 6 | 是否开时序集成 | 不开（先简后繁） | M4 后按回放质量 |
| 7 | 数据集/训练产物存放位置 | `extensions/jiuwen_agx/data/`（gitignore）+ 训练 outputs 外部目录 | 实施者 |
| 8 | 训练在本机还是搬到 GPU 机器 | 视可用硬件；venv 可整体复制（锁版本） | M3 前 |
| 9 | 视觉版相机方案（AGX 出帧接口） | M5 另立计划 | M4 后 |

## 11. 参考与出处

- ACT 论文：*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*（arXiv:2304.13705）
- lerobot 源码（**v0.6.1**，本文行号据此；审计报告见 `act-integration-review.md` 的完整引用表）：
  `policies/act/{configuration_act,modeling_act,processor_act}.py`、
  `datasets/{lerobot_dataset,dataset_writer,dataset_reader,feature_utils,compute_stats}.py`、
  `scripts/lerobot_train.py`、`common/train_utils.py`、`rollout/inference/sync.py`、
  `rollout/context.py`、`policies/utils.py`、`utils/constants.py`、`processor/factory.py`
- 本仓接缝草案：`extensions/jiuwen_agx/docs/architecture-flows.md` 第 10 章
- 本仓相关实现：`sim/backend.py`（注册表范式）、`sim/env.py`（能力配置化）、
  `agx_excavator/work.py`（脚本教师；driver 调用面）、`sim/driver.py`（输出校验闸）、
  `actions.py`（ActionSpec 注册链示范）、`__init__.py`（注册顺序）
- core 缝隙：`env/base.py:register_capability`、`adapters/_common/capability_spec.py:register_capability_spec`、
  `api/decorators.py:ActionSpec`、`tools/builder.py`（api∩env 门控）、
  `rails/safety.py`（工具名单）、`rails/recovery.py`（tags 认领）、`agent/fast/runner.py`（ok 语义）
