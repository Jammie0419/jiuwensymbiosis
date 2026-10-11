# Goal-ACT 对接说明书（大脑组 ↔ 小脑组）

> 面向双方工程师。目的：**一份文档讲清"数据从哪来、长什么样、两边怎么对上"**，
> 让大脑组（本仓库，AGX 仿真/编排/采集）与小脑组（Goal-ACT V2，模型训练）
> 不需要读对方的代码就能正确交接。
>
> 阅读顺序建议：§0 速查（1 分钟）→ §1 名字对照（解决读不懂的根源）→
> 各自关心的接口章节（小脑组看 §2/§3/§4，大脑组看 §3/§6）。
>
> 文档中的每个字段名都与代码逐字一致（核对过行号，见文内标注）。

---

## §0 一页速查

**连什么**：小脑组训练好的模型（`.pt`）放进大脑组的 `act_exec` 动作里，
在 AGX 仿真里完成"挖一斗→倒掉"的完整循环；验收标准 = 10 个训练没见过的
目标点 ≥7 个成功。

**流程一句话**：大脑组在 AGX 里采数据（npz）→ 转换器产出小脑组能吃的
训练数据（`episode_*.h5`）→ 小脑组训练并交付 `.pt` → 大脑组填一行配置接入
并跑验收。

```
┌─ 大脑组（本仓库，AGX 机器） ────────────────────────────────┐
│                                                             │
│  AGX 仿真 ──采集──► data/raw/*.npz                          │
│                        │                                    │
│                        ▼ 转换器（scripts/lerobot_to_goalact_h5.py）│
│               data/goalact_h5/<run>/episode_*.h5            │
│               ＋ suggested_config.json（他们 config 必须用的值）│
└────────────────────────┬────────────────────────────────────┘
                         ▼ 交付：h5 目录 + suggested_config.json + links_m
┌─ 小脑组 ────────────────────────────────────────────────────┐
│  finetune / 重训 ──► 模型 .pt（goal-act-v2）                  │
└────────────────────────┬────────────────────────────────────┘
                         ▼ 交付：.pt
┌─ 大脑组 ────────────────────────────────────────────────────┐
│  配置 policy: {name: goal_act_v2, ckpt, src, machine}       │
│  ──► act_exec（每拍喂观测、执行模型增量）──► AGX              │
│  验收：scripts/act_acceptance.py（10 留出点 ≥7）              │
└─────────────────────────────────────────────────────────────┘
```

**双方职责一句话**：

| | 大脑组 | 小脑组 |
|---|---|---|
| 负责 | 采集、格式转换、执行/护栏、验收 | 模型训练质量、交付 `.pt` |
| 交付物 | h5 训练数据 + `suggested_config.json` + 标定后的 `links_m` + 验收报告 | goal-act-v2 的 `.pt`（闭环达标后） |

**现在卡在哪**（详见 §6）：AGX 许可证 → 标定 → 小脑组重训 → 验收。
小脑组现有 checkpoint 尚未通过其自带闭环评估（§6.2），需重训后再接。

---

## §1 先对齐名字（读不懂的根源）

同一个东西，两边的叫法不同。**先统一叫法，后面全部按"统一叫法"讲。**

| 说人话（统一叫法） | 大脑组叫法 | 小脑组叫法 |
|---|---|---|
| 走一步动作 | **拍**（beat；一条循环 120 拍） | **步 / 控制周期**（一条循环约 118 步） |
| 挖的位置 | **挖点**（`dig_x_m`, `dig_y_m`） | **pick 点**（`pick_xyz`） |
| 倒土的位置 | **卸点**（`dump_x_m`, `dump_y_m`） | **place 点**（`place_xyz`）——他们不叫 dump |
| 铲斗装没装土 | **`scoop`**（载料真值，>50 kg 判定） | *没有这个概念* |
| 动作进行到哪一段 | *没有这个概念* | **`phase`**（阶段 0..7：接近→下降→挖→抬→转→接近卸→卸→撤） |
| 四个关节 | `swing, boom, arm, bucket`，值为**原生单位**：swing=弧度，三缸=**液压缸位移（米）** | 同名四个关节，值为**转角（度）**（简化平面模型的"等效角"） |
| 模型输出 | 绝对目标（下一拍应站到哪） | 增量（比现在再多转多少） |
| 原始数据文件 | `*.npz` | （不产原始数据） |
| 训练数据文件 | `*.parquet`（lerobot 格式，我们自用） | `episode_*.h5` |
| 模型文件 | `pretrained_model/` 目录 | `*.pt` |
| 位置/几何基准 | AGX 场景（`CabinHinge` + 三个 `*Prismatic`） | 平面三连杆模型（`links_m` 三段长度） |

> ⚠️ **最危险的词：`boom`（同理 `arm`、`bucket`）。**
> 我们说 `boom = -0.13`，意思是"boom 液压缸伸了 **-0.13 米**"；
> 他们看到 `boom = -0.13`，默认意思是"动臂转了 **-0.13 度**"。
> 两个数根本不是一个物理量。
> **一切换算由"标定文件"（machine file）唯一负责**：标定文件给出
> "液压缸位移米 ↔ 等效角"的对照关系，是训练与部署的共享资产（§3.4）。

---

## §2 接口 A：小脑组交模型、大脑组接入

### 2.1 模型文件要求

| 要求 | 说明 |
|---|---|
| 架构版本 | checkpoint 内 `architecture_version == "goal-act-v2"`，否则加载直接报错（他们的 `inference.py` 自己校验） |
| 单文件 | 一个 `.pt`：配置（`config`）与归一化统计（`state_mean/std` 等 6 个 buffer）**都内嵌在文件里**，不需要额外附带文件 |
| 闭环质量 | 见 §6.2——先过小脑组自己的闭环评估，再交付 |

### 2.2 推理契约（我们怎么调用它，小脑组不需要改任何代码）

大脑组的适配器 `src/jiuwen_agx/policy_goal_act.py` 封装了他们的
`ACTPolicy.predict(state35, pick_xyz, place_xyz, phase)`，参数如表：

| 参数 | 内容 | 单位/约定 |
|---|---|---|
| `state35` | 35 维状态；**只有第 6/7/8/9 列是四个关节角**，其余 31 列填 0 | 关节列是**度** |
| `pick_xyz` | 挖点（x, y, z） | 米；z 取标定文件里的地面高度 |
| `place_xyz` | 卸点（x, y, z） | 米；同上 |
| `phase` | 阶段标量 | 我们给 **2**（还没装土，对准挖点）或 **4**（装上了，摆向卸点）——与训练数据的相位规则完全同源（§3.5） |
| 返回 | (16, 4) 的**弧度增量**，已按 `config.max_joint_step_deg` 截断 | 我们**只用第 0 步**：加在当前实测值上 → 换算回 AGX 单位 → 下发 |

**调用节奏**：每拍（每步执行）调用一次 `predict`、只执行返回的第一小步。
这与小脑组自己的"每控制周期重规划、只走一步"完全一致——**等价性由我们实现，
他们零改动**。模型内部维护的 10 帧历史（`history_length=10`）由我们逐拍喂入
自然形成。

### 2.3 大脑组侧的配置（填这份就行）

```yaml
env:
  cfg:
    low_level:
      policy:
        name: "goal_act_v2"
        ckpt: "<模型>.pt"                 # 小脑组交付
        src:  "<他们项目>/src"             # 含 act_trajectory 包的目录
        machine: "data/machine_agx365.json"  # 标定文件（§3.4）
```

### 2.4 两条配置红线（小脑组训练 config 必须遵守）

1. **`max_joint_step_deg` 必须用我们导出的 `suggested_config.json` 里的值。**
   原因：他们的推理端把每一步的输出截断到 `config.max_joint_step_deg`
   （现在旧值 ~2.5°/步），而我们的"拍"合法行程可达 20°+/拍；沿用旧值会把
   部署循环压慢数倍甚至跑不完。
   好消息：这个字段**不在**几何签名里（签名只有 `links_m`、`geometry`、
   `joint_state_indices`、`joint_limits_deg` 四项），改了**不用**重造数据。
2. **`joint_limits_deg` 若要放宽必须"先定再生成数据"。**
   它在几何签名里——训练/评估会因为签名不一致拒绝旧数据，顺序反了就得全部重造。

---

## §3 接口 B：训练数据格式（双方对照）

### 3.1 数据从哪来（谁做哪一步）

| 步骤 | 谁做 | 在哪 | 产出 |
|---|---|---|---|
| 采集（AGX 里自动挖 100+ 条） | 大脑组 | AGX 机器 | `data/raw/run_all/*.npz`（一条循环 = 一个文件） |
| 格式转换 | 大脑组 | AGX 机器 | `data/goalact_h5/<run>/episode_*.h5` ＋ `suggested_config.json` |
| 训练 | 小脑组 | 他们的训练机 | `.pt` |
| 接入 + 验收 | 大脑组 | AGX 机器 | 验收报告（10 点 ≥7） |

**小脑组不需要自己采集**：采集器（`record_demos.py`）和转换器已就绪，
他们拿到的 h5 就是从 AGX 真物理数据来的。

### 3.2 采集侧格式（npz，大脑组内部契约；小脑组可以只看一眼）

一条循环一个文件，120 拍：

| 字段 | 形状 | 含义 |
|---|---|---|
| `joints_before` / `joints_after` | (120, 4) float64 | 每拍前/后实测关节，**原生单位**（swing 弧度 + 三缸米），顺序 `swing, boom, arm, bucket` |
| `scoop` | (120,) bool | 载料真值（>50kg 判定） |
| `beat_seconds` | (120,) float32 | 每拍实际耗时（拍 = 到位的节奏，不是均匀时间） |
| `meta` | JSON 字符串 | `goal` 四坐标（米）、`depth_scale`、`joint_names`、`joint_units=null`（混合单位）、`swing_unit=rad` 等 |

### 3.3 训练侧格式（`episode_*.h5`，小脑组直接吃）

转换器与他们的载入器实测联通（**103/103 条直读通过**）。
字段逐条如下（"他们读吗"列 = 他们现有载入器 `infer_real_episode` 的行为）：

| 数据集路径 | 形状 | 单位 | 含义 | 他们读吗 |
|---|---|---|---|---|
| `observations/state` | (T, 35) float32 | 第 6..9 列**度**，其余 0 | 每拍开始的四关节状态 | ✅ **必需** |
| `actions/joint_increment` | (T, 16, 4) float32 | **度** | 从该拍起未来 16 拍的每拍增量（滑窗 + 补零） | ✅ 可选但请保留（缺了他们自己用状态差分重算） |
| `annotations/action_pad` | (T, 16) uint8 | — | 1 = 该步超出循环末尾（无效） | ❌（他们按长度自己重算，保留无害） |
| `annotations/valid` | (T,) bool | — | 有效帧（我们全 True） | ✅ 可选（缺省全 True） |
| `annotations/goal` | (T, 6) float32 | 米 | **精确**的 pick/dump 坐标（我们采集时就有真值） | ❌ **他们现在忽略**（见 §3.6） |
| `annotations/phase` | (T,) int64 | — | **2/4** 相位标签（与部署时同规则） | ❌ **他们现在忽略**（见 §3.6） |
| `joint_rad` / `goal` / `phase` | 同上的弧度/副本 | — | 便于调试的冗余字段 | ❌（保留无害） |

文件属性（h5 attrs）记录：`source=jiuwen_agx_agx365`、`timeline=beats`、
`control_frequency=10`（名义）、`chunk_size=16`、`history_length=10`、
`machine`、`calibrated`、`unit_joint_increment=deg`、`phase_mode` 等。

文件命名：`episode_000000.h5` 起顺序编号；**他们的载入器只扫一层
`episode_*.h5`（不递归）**，目录名随意（用 `--real-dir` 指过去）。

### 3.4 标定文件（machine file）：训练与部署的共享资产

两边世界的桥。**同一个模型，训练数据与部署执行的换算必须用它同一份。**
（这就是"先标定再训练"的原因：用未标定的占位映射产出的数据，语义是错的。）

内容（`configs/machine_agx365.template.json` 是模板；标定流程见 §4.2）：

| 字段 | 含义 |
|---|---|
| `policy_joint_order` | `["swing","boom","arm","bucket"]`（三处必须一致，见 §5 红线 5） |
| `joints[].native_unit` | `rad`（swing）或 `m`（三缸） |
| `joints[].sign` / `offset_rad` | 回转零位校准 |
| `joints[].hydraulic_conversion` | **位移（米）↔ 等效角（度）** 的映射（线性端点或多项式） |
| `pick_z_m` / `dump_z_m` | 挖点/卸点的地面高度（补进 xyz 的 z） |
| `calibrated` | 只有它 = true，转换器才接受这份文件出正式训练数据（防污染） |

### 3.5 两条关键约定（为什么这么定）

**（a）拍 = 帧（不做时间重采样）。** 小脑组的训练帧本来就是"一次到位
的动作步"（他们的合成数据每步受 `max_joint_step_deg` 限幅），我们的"拍"
正是同样的东西；两边 1:1 对应。若把我们的数据重采样成细粒度时间序列，
模型会学出"每 0.1 秒转 0.2°"的增量，而我们部署循环要几千拍才走完一个循环
（直接超过 300 拍上限失败）。

**（b）相位 = 2/4（不切 8 段）。** 部署时我们能看到的信息只有"铲斗装没装土"
（`scoop` 真值），所以相位规则定为：未装土 → **2**（对准挖点），装上 → **4**
（摆向卸点）。训练数据里的相位标签用同一规则——**训练时怎么标，部署时就怎么给**。

> 附：`actions/joint_increment` 写成**度**是有意为之。他们的载入器用品
> "99 分位 > 0.5 就当度"的启发式自动识别单位；度制在正常和异常幅度下都能被
> 正确识别（弧度制遇到大摆转会被误判）。`annotations/unit_joint_increment="deg"`
> 是给他们改成显式读取的备注。

### 3.6 ⚠️ 一个必须双方知道的现状：goal/phase 标注会被忽略

他们现有载入器**不读** `annotations/goal` 和 `annotations/phase`：
- goal 会用简化 FK 从状态反推（取最"低"点为 pick、最"高"点为 dump）；
- phase 会用时间八等分（与我们部署的 2/4 不同源）。

**两个选项**（推荐 A）：

**A. 加约 6 行直读补丁**（在 `data.py` 的 `infer_real_episode` 里，读取处）
——用我们给的**精确** goal 与同源的 2/4 相位：

```python
# 伪代码示意：优先读我们写好的标注，缺失时才回退到现有的伪推断
if "annotations/goal" in f:
    goal = np.asarray(f["annotations/goal"], dtype=np.float32)
else:
    ...  # 现有的 FK 极值伪推断（保持不动）
...
if "annotations/phase" in f:
    phase = np.asarray(f["annotations/phase"], dtype=np.int64)
else:
    ...  # 现有的液压/时间八等分回退（保持不动）
```

**B. 接受伪推断**：能跑，但 goal 会带上 FK/几何误差、相位与部署不同源——
训练质量与我们验收时的表现都可能打折。

### 3.7 数值走查：一条真实循环的完整数字链

以下取真实采集数据（`run_all` 某条的某一拍），标"示意"的角度值是经标定
文件换算后的形态（现在的标定文件是占位值，标定完成后角度数值会不同，
但**链路形态不变**）：

| 环节 | 数据 | 说明 |
|---|---|---|
| AGX 实测（我们记录的 npz 原值） | swing `-0.00003` 弧度；boom `-0.0097` 米；arm `0.00003` 米；bucket `-0.0004` 米 | **真实数字** |
| → 经标定文件换算 → 写入 h5 的 `observations/state[:, 6:10]` | `[ 0.0, -91.1, -207.5, -248.7 ]`（**度**，示意） | 标定后这组数会变 |
| 同拍 `actions/joint_increment[0, 0]` | 下一拍的每关节变化量（**度**） | 每拍法定上限见 `suggested_config.json` |
| 模型输出（部署时） | 每个关节的弧度增量，例如 swing `+0.026` 弧度 | 已按 config 截断 |
| → 我们加在当前实测上、换算回 AGX 单位 → 下发 | swing `+0.026` 弧度；boom 缸 `-0.02` 米 | 直接作为下一拍目标，过安全限位后执行 |

全程只有**一个**换算点（标定文件），中间没有任何"魔法"。

---

## §4 照抄就能跑的命令

### 4.1 大脑组给你们的交付物

| 交付物 | 说明 |
|---|---|
| `data/goalact_h5/<run>/`（约 100 个 `episode_*.h5`，每个几十 KB） | 直接当 `--real-dir` 用 |
| `data/goalact_h5/<run>/suggested_config.json` | 你们训练 config 里 `joint_limits_deg` / `max_joint_step_deg` 用这里的值 |
| 标定后的 `links_m`（标定完成时一并给） | 填进你们 config 的 `links_m`（会改变几何签名 → 需重造/重训） |

### 4.2 转换命令（大脑组跑，参考）

```bash
cd extensions/jiuwen_agx
python scripts/lerobot_to_goalact_h5.py \
    --raw data/raw/run_all \
    --machine data/machine_agx365.json \
    --out data/goalact_h5/run_all
# 标定前可加 --allow-uncalibrated 只验证链路（产出会标记 calibrated=0，禁止拿去训练）
```

标定（AGX 许可证恢复后，大脑组）：按 `scripts/agx_machine_fit.py` 文档头部的
采样格式记录"扫缸 + 齿尖坐标"，跑一次拟合即可得到 `calibrated: true` 的
标定文件与 `links_m`（拟合器已带合成自测与多起点，残差 >0.05m 会拒绝出文件）。

### 4.3 小脑组训练命令（用我们的数据微调）

```bash
python3 scripts/finetune_real.py \
    --config configs/<你们的 agx365 config>.json \
    --synthetic <你们的合成数据.h5> \
    --real-dir <我们交付的 goalact_h5 目录> \
    --resume outputs/act_pretrained.pt \
    --output outputs/act_agx365_v1.pt
```

注意三件事：
1. `--real-dir` 指向的目录**只扫一层 `episode_*.h5`**——直接把我们的目录或拷贝给它都行；
2. 你们 config 里 `real_data.machine` 与 `geometry.machine` 必须**同名字符串**
   （载入器会校验，不一致直接报错）；
3. `--synthetic` 若换成按 AGX 几何新造的合成数据，先看 §2.4 两条配置红线
   （`joint_limits_deg` 变了要重造数据；`max_joint_step_deg` 用我们的建议值）。

### 4.4 小脑组自测（不依赖 AGX，验证我们给的数据可读）

```bash
python3 -c "
import sys, json; sys.path.insert(0, 'src')
from pathlib import Path
from act_trajectory.data import infer_real_episode
cfg = json.loads(Path('configs/<你们的 config>.json').read_text(encoding='utf-8'))
state, goal, phase, actions, pad, valid = infer_real_episode(
    Path('<我们交付目录>/episode_000000.h5'), cfg)
print(state.shape, actions.shape, phase[:5], goal[0])"
```

### 4.5 交付模型后，大脑组接入 + 验收（参考）

```bash
# 1) 配置（§2.3 的 yaml 填上你们的 .pt 路径）
# 2) mock 冒烟（不连仿真）：确认加载/推理/换算链路
# 3) 真机级验收（AGX 桥接开着）：
python scripts/act_acceptance.py \
    --config configs/agx_excavator/agx_excavator.local.yaml \
    --exclude-raw data/raw/run_all --points 10 --seed 7 --also-dig \
    --out data/acceptance/<name>.json
# 10 个训练没见过的目标点，成功 ≥7 即接入达标
```

---

## §5 一致性红线（5 条）+ 常见问题

**红线**（任何一条破，轻则变慢重则静默出错，验收必挂）：

1. **换算只有一个来源**：米↔角度只走标定文件；训练数据与部署执行必须用**同一份**；
2. **拍 = 步 = 帧**：不重采样、不二次切分步长；
3. **相位同源**：训练标注与部署估计都是 2/4 规则（依赖 §3.6 的选项 A 或 B 明确）；
4. **目标语义**：挖点/卸点（我们）↔ pick/place（他们）= 同一对坐标，z 高度由标定文件定；
5. **关节顺序**：`swing, boom, arm, bucket` 在 npz、h5 的第 6..9 列、标定文件三处一致——
   我们的适配器**对不上会直接报错拒绝**（不会默默出错）。

**FAQ**：

- **问：为什么不能直接吃大脑组的 npz/parquet？**
  答：单位（混合原生 vs 全角度）、时间轴（拍 vs 帧）、动作（绝对 vs 增量）、
  条件（4 坐标 vs 6 坐标+相位）都不同——转换器就是干这个的，半天工作量已免。
- **问：为什么 state 用度、动作却写弧度（其实我们写的是度）？**
  答：他们模型**内部**的 state 槽是度、输出增量是弧度（这是他们的设计）；
  我们的 h5 里 `actions/joint_increment` 写**度**是为了让他们的单位嗅探器稳定识别。
  两边都别改，按现状即可。
- **问：为什么步长上限（`max_joint_step_deg`）不能沿用旧值？**
  答：他们的旧值是合成 IK 的步长（~2.5°）；我们的拍是大步（可达 20°+/拍）。
  推理端按旧值截断＝每一拍只让走一小步，一个循环要多花数倍时间。
- **问：为什么必须先标定才能产正式训练数据？**
  答：不标定，"米↔角度"就是拍脑袋的映射，9 个未知量约等于乱猜——产出的
  训练数据语义错位，模型学到的"角度世界"与部署换算的对不上。

---

## §6 验收与现状

### 6.1 接入验收方式（对任何模型同一把尺）

1. **mock 冒烟**（无需仿真）：验证模型加载、推理、换算、护栏全链路；
2. **真机级验收**：`scripts/act_acceptance.py` ——10 个与训练数据保持距离的
   留出目标点，`act_exec` 成功率 **≥7/10** 为达标；可选同点 `dig` 对照出对比表。
   成功判据 = 铲斗装土（>50kg）后卸掉（与采集/训练过滤同一判据）。

### 6.2 现状（如实）

| 事项 | 状态 |
|---|---|
| 接口/格式 | ✅ 已打通并有实证（转换产物被小脑组载入器 103/103 直读通过；适配器 mock 链路跑通） |
| 小脑组现有 checkpoint | ⚠️ **尚未通过其自带闭环评估**：`outputs/eval_pretrained` 与 `eval_finetuned` 的 `metrics.json` 里 `pick_success=False`、`dump_success=False`，挖点末端误差 ≈1.4 m（容差 0.15 m）、闭环关节 MAE 9~19°（teacher-forcing 时 0.1°）；且为**旧几何**（`links_m=[3.2, 2.4, 1.2]`）。结论：**需在 AGX 域重训、闭环达标后再接入**，避免双方空跑 |
| AGX 许可证 | ⏸ 已过期（我方机器 10-08/09 尚可用，10-10 起暂停）——**标定与真机级验收都等它**；恢复后标定约半天 |
| 依赖链 | license → 标定（大脑组，拟合器已就绪）→ 小脑组重训 → 接入验收 |

### 6.3 小脑组任务清单（按依赖排序）

1. **重训（AGX 域）**：按本文档 §2.4 两条配置红线建 AGX365 config
   （`joint_limits_deg` 用 `suggested_config.json`、`links_m` 用标定好的值）
   → 合成预训练 ＋ 我们交付的 h5 微调 → **闭环达标**后交付 `.pt`；
2. **载入器直读补丁**（§3.6 选项 A，约 6 行，强烈推荐）；
3. venv 修复（可选——我们用 `src:` 路径引用绕过，不修也能接）。

---

## §7 附录

### 7.1 双方文件对照

| 东西 | 大脑组（本仓库） | 小脑组（他们的项目） |
|---|---|---|
| 仓库根 | `E:\jiuwensymbiosis` | `E:\AGX-2.42.2.1\act_trajectory\act_trajectory` |
| 采集器 | `extensions/jiuwen_agx/scripts/record_demos.py` | —（无需自建） |
| 原始数据 | `extensions/jiuwen_agx/data/raw/run_all/*.npz` | — |
| 转换器 | `extensions/jiuwen_agx/scripts/lerobot_to_goalact_h5.py` | — |
| 训练数据（交接物） | `extensions/jiuwen_agx/data/goalact_h5/<run>/` | 拷到他们的 `--real-dir` |
| 建议配置（交接物） | 同上目录里的 `suggested_config.json` | 他们的 `configs/<agx365>.json` |
| 标定 | `src/jiuwen_agx/machine_fit.py` ＋ `scripts/agx_machine_fit.py` | 他们 config 的 `links_m` 用同一结果 |
| 接入适配器 | `src/jiuwen_agx/policy_goal_act.py` | —（无需改代码） |
| 模型（交接物） | `extensions/jiuwen_agx/data/ckpt/<name>/` | `outputs/*.pt` |
| 验收 | `scripts/act_acceptance.py` | — |

### 7.2 小脑组自带 AGX 路线（可选参考）

小脑组项目里已有一套他们自研的 AGX 对接（`agx_validation/`：`policy_server`
＋适配器，协议 `act-agx-v1`，观测 `q_rad`/`ee_m`/可选 `state35`，命令
`q_target_rad`，相位来自教师时间表、每步只走增量第一步）。若他们想在自己
那边先跑 AGX 对照可以继续用；**正式接入以本文档 §2 的路线为准**（走大脑组
的 `act_exec`，护栏/判据/验收统一）。

---

*本说明书由大脑组维护；字段与行为均对照双方代码核对（核对基准：本仓库
`040ace9`/`0ffffd8`/`d3dee22` 及小脑组项目当时工作区）。如有冲突以代码为准，
并请回提给大脑组更新本文档。*
