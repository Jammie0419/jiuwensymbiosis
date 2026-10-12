# Goal-ACT 对接说明书（大白话版）

> 给两个组一起看：大脑组（AGX 仿真、采集、接入、验收）和小脑组（模型训练）。
> 这份文档只讲清楚三件事：**大脑组采的数据长什么样、小脑组训练的数据长什么样、
> 两边怎么对上**。每个字段都有真实数字举例。
> 看不懂的术语先翻 §5「名字对照表」——两边叫法不一样，这是之前读不懂的根源。
> 要动手改代码/写脚本的，看 §8 附录（完整字段表、标定文件、红线）。

**现在的最短结论（2026-10-12 更新）**：量尺**已经标定好**（纯几何扫描，不等许可证），
数据也已用真尺重转（103 条，`data/goalact_h5/run_all_calibrated`，`calibrated=1`，
可直接训练）；交接物齐了——载入器补丁（`docs/patches/goalact_loader_read_annotations.patch`，
已在本机副本验证可干净应用）、AGX365 训练配置样例（`docs/agx365_training_config.json`，
已过小脑组 `validate_geometry` / `check_real_machine` / `Kinematics` 三道校验）、
建议配置数字（同目录 `suggested_config.json`）。剩下两件：
① 小脑组应用补丁 + 建配置 + 用新数据重训；② 大脑组接入 + 10 点验收。
实机复核项（§4.4）挂起等许可证恢复，**不阻塞重训**。

---

## 1. 为什么有三种数据（先搞清这个，后面就不绕了）

因为模型"认识"哪种机器，取决于它用什么数据练过。而世界上有三种"挖掘机身体"：

| | ① 合成数据 | ② 真机数据 | ③ AGX 仿真数据 |
|---|---|---|---|
| **什么机器** | 纸上画的假想机（几个数学公式，没土、没液压） | 真挖掘机（山河 SWE600F，有液压、有土） | 电脑里的仿真机（AGX 物理引擎，有颗粒土） |
| **怎么来的** | 程序自己编题自己解，10 分钟出上万条 | 真机上的传感器录的（液压阀电流等） | 在 AGX 仿真里自动挖，录下来的 |
| **谁产的** | **小脑组**（算的，不用机器） | **小脑组**（要在真机上采） | **只能大脑组产**（AGX 装在大脑组机器上） |
| **解决什么** | 学"挖掘的大致套路" | 学"真机器的力道"（小脑组的部署目标） | 学"仿真机的物理"（大脑组验收模型的考场） |

**接力关系**：①练套路（便宜）→ ②在真机上纠偏（小脑组的活）→ ③在仿真机上
适应（大脑组的活，因为只有大脑组有这台仿真机）。小脑组此前交付的模型只练过
①和②，没在 AGX 这台"机器"上练过——所以在 AGX 仿真里跑不好。**给它③的数据，
它才能在仿真里干活。**

> 这三种不是重复劳动。打个比方：①看教材背套路、②真车实操、③在 VR 模拟器里
> 练——因为最终考试在 VR 模拟器里，所以 VR 的练习素材必须由大脑组录；
> ①②由小脑组自己安排。

**重要说明**：③大脑组**已经采好了**（103 条有效循环，在
`data/raw/run_all/`），**不需要再采**。详见下面。

**"在真机上纠偏"是什么意思**（顺手解释一句）：合成数据里的挖掘机是"理想版"
（瞬时到位、没有摩擦、土只是几何点），真机器有液压延迟、惯性、土质变化——
模型在合成数据上学的动作节奏搬到真机会系统性偏。用真机上录的真实作业数据
继续少量训练（微调，不是重学），把动作校正到这台真机器上，就叫纠偏。
小脑组的 `finetune_real.py` 就是干这个的。**同理，AGX 仿真机和真机之间也有
差距**，所以要在 AGX 数据上做一次"AGX 域纠偏"——这就是大脑组交付③数据的意义。

---

## 2. 一条数据长什么样（真实文件对照）

下面全部用同一条真实循环（ep_0000）的数字。

### 2.1 大脑组采的：一条 = 一个 npz 文件（一条循环 = 120 拍）

文件：`data/raw/run_all/ep_0000_teacher.npz`

| 字段 | 形状 | 真实样例 |
|---|---|---|
| `joints_before` | (120, 4) | 拍 0：`[-0.0000, -0.0097, 0.0000, -0.0004]`；拍 47：`[1.886, -0.085, -0.23, -0.674]` |
| `joints_after` | (120, 4) | 拍 0：`[0.0845, -0.0015, 0.0000, -0.0000]`；拍 47：`[1.886, -0.105, -0.166, -0.678]` |
| `scoop` | (120,) 真假 | 拍 0 = 假；**拍 47 = 真（这一拍装上了土）** |
| `beat_seconds` | (120,) 秒 | 这一拍实际花了多久 |
| `meta`（一小段 JSON） | — | 挖点 `(-1.74, 5.34)`、卸点 `(1.90, 5.46)`、关节名、单位声明 |

**三个关键点**：
1. 单位是**机器原生**的：`swing` 是弧度；`boom/arm/bucket` 是**液压缸伸出去多少米**；
2. 4 列的顺序固定是 `swing, boom, arm, bucket`；
3. "拍"不是固定时间——是"动作走到位"一次。120 拍 = 一个完整循环
   （对准→下挖→收斗→摆转→卸料）。

### 2.2 小脑组训练用的：同一条循环 = 一个 h5 文件（120 帧，帧=拍）

文件：`episode_000000.h5`（由上面那条 npz 转出来的；转换见 §3。下表是**真尺版**数字）

| 字段 | 形状 | 真实样例 |
|---|---|---|
| `observations/state` | (120, 35) | 帧 0 的第 6~9 列 = `[-0.0, 23.2, -89.0, -62.8]`；帧 47 = `[108.1, 19.4, -77.5, -5.7]`——**度**（其余 31 列全 0） |
| `actions/joint_increment` | (120, 16, 4) | 帧 0 的第 0 行 = `[4.842, 0.417, 0.001, -0.037]`：**第 0 拍这一拍里，四个关节各动了多少（度）**；再往下 15 行是未来 15 拍的 |
| `annotations/phase` | (120,) | 帧 0 = **2**（还没装土）；帧 47 = **4**（装上了，该去卸了） |
| `annotations/goal` | (120, 6) | `[-1.74, 5.34, -2.485, 1.90, 5.46, -1.564]` = 挖点 xyz + 卸点 xyz（米，z 是动臂铰点系） |
| `annotations/valid` | (120,) 真假 | 有效帧（大脑组全填"真"） |

### 2.3 对应关系记住三句话就够

1. **大脑组的 `joints_before[47]` = 小脑组的 `state[47]`**（只是米/弧度换成了度）；
2. **`after[47] − before[47]` = `increment[47, 第 0 行]`**（一拍的动作量）；
3. **`scoop[47] = 真` → `phase[47] = 4`**。

其余字段（补零掩码、有效性、文件属性）都是工程细节，附录 §8.1 有全表。

---

## 3. 转换怎么转的（7 步，用拍 47 走一遍）

用拍 47 的真实数字（`before = [1.886, -0.085, -0.23, -0.674]`；已完成标定，下表即真值）：

| 步骤 | 做什么 | 拍 47 的结果 |
|---|---|---|
| 1 | **对齐时间轴**：不重采样。大脑组的"拍"就是小脑组的"帧"，1 拍 = 1 帧 | 拍 47 → 帧 47 |
| 2 | **单位换算**（用"标定文件"这一把尺，全局唯一换算点） | swing：`1.886 弧度 ×57.3 = 108.1 度`；boom：`-0.085 米 → 19.4 度`；arm：`-0.23 米 → -77.5 度`；bucket：`-0.674 米 → -5.7 度` |
| 3 | **写状态**：35 列的矩阵全填 0，只把 4 个角度放进第 6~9 列 | `state[47] = [0,0,...,0, 108.1, 19.4, -77.5, -5.7, 0,...]` |
| 4 | **生成增量**：把每拍的位形变化算出来，从每一帧起存"未来 16 拍"的 | 拍 0：`after−before = [0.0845, 0.0082, ...] → [4.842, 0.417, 0.001, -0.037] 度` |
| 5 | **写相位**：看载料真值，没装土写 2、装上了写 4 | 拍 47 装上了 → `phase[47] = 4` |
| 6 | **写目标**：挖点/卸点补上实测的齿尖高度（z）变成 6 个数，铺满 120 帧 | `[-1.74, 5.34, -2.485, 1.90, 5.46, -1.564]`（z 取 112 条实测在装载/卸载时刻的齿尖高度中位，动臂铰点系） |
| 7 | **收尾**：写文件属性（单位、时间轴、标定状态=1…），顺带导出训练配置要用的建议值 `suggested_config.json` | — |

> 为什么单位换算要先"标定"：AGX 原生数据说"缸伸出 0.085 米"，小脑组模型的世界说
> "臂转了 19.4 度"——米变度靠一把尺量出来。**这把尺已经量好了**：用模型自身的机构
> 几何（铰点、缸锚点）纯几何扫描 + 112 条实采数据回放做物理自检（`scripts/calibrate_agx365.py`，
> 不需要 AGX 许可证）；产出的换算文件 `calibrated: true`，转换器直接放行。

---

## 4. 现在哪里对上了、哪里还不对、还缺什么

### 4.1 已经对上的（不用再动）

- 文件与字段（`episode_*.h5`、`observations/state`、`actions/joint_increment`、
  `annotations/valid`）：形状、命名、类型全对——**小脑组载入器实测 103/103 直读通过**；
- 时间轴（拍=帧）、单位（状态用度、增量写度，小脑组载入器的自动识别实测正确）；
- 相位和目标的设计（2/4 规则、6 维目标）。

### 4.2 三处的现状（2026-10-12：全部闭环）

**① 量尺——已标定（完成，不等许可证）**
- 真尺：`configs/machine_agx365.json`（`calibrated: true`），纯几何扫描生成
  （`scripts/calibrate_agx365.py`：解析 AGX 导出的 OpenPLX 文本 → 四连杆闭环扫缸 →
  112 条实采数据回放做物理自检选符号；**不需要 AGX 许可证**）；
- 结果要点：boom/arm/bucket 用 poly5 换算表（bucket 线性误差 13°→0.11°）；
  `pick_z=-2.485 / dump_z=-1.564 m`（动臂铰点系）；swing 零位=原生读数（待实机复核，见 §4.4）；
- 数据已重转：**`data/goalact_h5/run_all_calibrated`（103 条，`calibrated=1`，可直接训练）**；
  占位版干跑 `run_all_dryrun` 保留作对照，不再使用。

**② 小脑组载入器补丁——已成文（一条命令应用）**
- 补丁：`docs/patches/goalact_loader_read_annotations.patch`（对你们
  `src/act_trajectory/data.py` 的完整 diff；已在本机副本实测：干净应用后
  `phase` 读到 2/4、`goal` 六个数字与标注一致；无标注的数据仍走原逻辑）；
- 这个文件替代了旧版本文档里的"6 行代码样例"（§8.4 保留说明）。

**③ AGX365 训练配置——样例已备好**
- 文件：`docs/agx365_training_config.json`（已过你们的 `validate_geometry` /
  `check_real_machine` / `ExcavatorKinematics` 三道校验；键集与你们 `config.json` 完全一致）；
- 关键值：`links_m [7.8, 3.6, 2.2733]`；`joint_limits_deg` / `max_joint_step_deg`
  取 `run_all_calibrated/suggested_config.json` 实测值；`workspace` 是按大脑组数据
  实测推导的建议值（合成数据用，可调）；机器名两处统一为 `"AGX365"`。

### 4.3 还缺什么（一条线）

```
大脑组 ✅：纯几何标定 → 真尺 → 重转数据（run_all_calibrated）→ 补丁/配置样例/本文档
  → ① 小脑组：应用补丁 + 建 AGX365 配置 + 用 run_all_calibrated 重训（关键路径）
  → ② 小脑组先过自己的闭环自测（pick 0.15 / dump 0.20 m 容差）
  → ③ 大脑组接入 + 10 个留出点验收（脚本现成，≥7 通过）
  → ④（待许可证）实机复核 swing 零位 + 真桥接复测（§4.4）
```

> 大脑组已在本机用你们仓库的**副本**预演过整条链（补丁 → 配置 → 50 轮训练收敛
> → 我们的适配器驱动 ckpt 闭环）：结果与复现命令见 `docs/goalact-rehearsal.md`
> ——你们照做即可，不必从零摸索。

**明确不缺的**：数据（103 条）、尺（已标定并重转）、格式、补丁、配置样例、
接入适配器、验收脚本。

**最大的不确定性仍是模型质量**：小脑组现有的模型连其团队自己的测试都没过
（官方评估里"挖到点/卸到点"都是 False，挖点偏差约 1.4 米、合格线 0.15 米）。
格式和补丁解决"接得对"，模型质量解决"答得对"——后者要靠小脑组重训。

### 4.4 待实机复核项（等 AGX 许可证；不阻塞现在重训）

1. **swing 的世界系零位/符号**：本次纯几何标定有一个无法离线确定的量——
   录制场景把机器摆在 world `(0,10,0)`，而目标点按**原点**环带采样；也就是说
   目标点的**半径**在采集时并未被物理兑现（方向语义有效，半径从未核对过）。
   机器文件里 `swing` 的 `offset_rad` 是唯一旋钮：许可证恢复后做一次
   "驱动到已知位置、瞄准一个已知点、看齿尖落点方位"的核对即可；
   若需修正，重转数据是分钟级。
2. 桥接读一次三缸原生读数，对照 `configs/calibration/agx365_cylinders.json` 的扫描端点。
3. 已知小项：swing 单帧增量 |Δ|>30° 的帧占 **0.26%**（505/197760，分布 19/103 条
   episode）——录制数据自身的跨 ±180° 大回转，新旧数据集一致；你们 swing 周期
   编码 + slew clip 可消化，知悉即可。

---

## 5. 名字对照表（读不懂的根源）

| 说人话 | 大脑组叫法 | 小脑组叫法 |
|---|---|---|
| 走一步动作 | **拍**（一条循环 120 拍） | **步 / 控制周期**（约 118 步） |
| 挖的位置 | **挖点**（`dig_x_m`, `dig_y_m`） | **pick 点** |
| 倒土的位置 | **卸点**（`dump_x_m`, `dump_y_m`） | **place 点**（不叫 dump） |
| 铲斗装没装土 | **`scoop`**（载料真值，>50kg） | *没这个概念* |
| 动作进行到哪段 | *没这个概念* | **`phase`**（0~7 阶段） |
| 四个关节 | 同名，但值是**原生单位**：swing=弧度，三缸=**米** | 同名，但值是**转角（度）** |
| 模型输出 | 绝对目标（下一拍站到哪） | 增量（比现在再多转多少） |
| 数据文件 | `*.npz`（原始） | `episode_*.h5`（训练用） |
| 模型文件 | `pretrained_model/` 目录 | `*.pt` |

> ⚠️ **最危险的词：`boom`（`arm`、`bucket` 同理）。**
> 大脑组数据里的 "boom = -0.085"，意思是**液压缸伸出 -0.085 米**；
> 小脑组模型里的 "boom = -0.085"，意思是**臂转了 -0.085 度**。完全不是一回事。
> **这一切由"标定文件"负责翻译**——它是训练和部署共用的同一把尺。

---

## 6. 双方各自要做什么

**大脑组：**
1. ~~等许可证 → 扫缸标定~~ ✅ **已完成**（纯几何标定，不等许可证；真尺 + 重转数据 +
   补丁 + 配置样例 + 本说明都已在仓库）；
2. ~~重新转换数据 + 导出建议配置~~ ✅ 已交付（`data/goalact_h5/run_all_calibrated`）；
3. 拿到新模型后：填一行配置、跑 10 点验收；
4. （待许可证）实机复核 swing 零位（§4.4）。

**小脑组：**
1. 应用补丁：`docs/patches/goalact_loader_read_annotations.patch`（一条命令，§7）；
2. 用 `docs/agx365_training_config.json` 建 AGX365 配置（拷进你们的 `configs/` 即可）；
3. 用大脑组交付的 `run_all_calibrated` 重训，**先过小脑组自己的闭环测试**
   （pick 0.15 / dump 0.20 m 容差）再交付 `.pt`；
4. （可选）让 `src` 目录可被外部引用（修 venv 或给 requirements）。

**两队一起确认**："AGX 域数据由大脑组采集并交付"这个分工（本来就在三方对接
文档里列为待对齐的问题之一）。

---

## 7. 照抄就能跑的命令

**转换（大脑组跑；已完成，留档）：**
```bash
cd extensions/jiuwen_agx
python scripts/lerobot_to_goalact_h5.py \
    --raw data/raw/run_all \
    --machine configs/machine_agx365.json \
    --out data/goalact_h5/run_all_calibrated
```

**标定（大脑组跑；已完成，留档）：**
```bash
python scripts/calibrate_agx365.py --data data/raw/run_all
# 纯几何 + 数据回放自检；fail-closed，自检不过不写机器文件
```

**应用补丁（小脑组跑）：**
```bash
cd E:/AGX-2.42.2.1/act_trajectory/act_trajectory
git apply -p1 --verbose <大脑组仓库>/extensions/jiuwen_agx/docs/patches/goalact_loader_read_annotations.patch
# 没有 git 的副本用： patch -p1 -i <补丁路径>
```

**训练（小脑组跑）：**
```bash
python3 scripts/finetune_real.py \
    --config configs/agx365.json \
    --synthetic <小脑组的合成数据.h5> \
    --real-dir <大脑组交付的 run_all_calibrated 目录> \
    --resume outputs/act_pretrained.pt \
    --output outputs/act_agx365_v1.pt
```

**自测（小脑组跑，不依赖 AGX；打补丁后应看到 phase 2/4 与真实 goal）：**
```bash
python3 -c "
import sys, json; sys.path.insert(0, 'src')
from pathlib import Path
from act_trajectory.data import infer_real_episode
cfg = json.loads(Path('configs/agx365.json').read_text(encoding='utf-8'))
s, g, ph, a, pad, v = infer_real_episode(Path('<run_all_calibrated 目录>/episode_000000.h5'), cfg)
print(s.shape, a.shape, ph[:5], g[0])"
# 期望: (120, 35) (120, 16, 4) [2 2 2 2 2] [-1.742 5.342 -2.485 1.898 5.463 -1.564]
```

**验收（大脑组跑，拿到模型后）：**
```bash
python scripts/act_acceptance.py \
    --config configs/agx_excavator/agx_excavator.local.yaml \
    --exclude-raw data/raw/run_all --points 10 --seed 7 --also-dig
# 10 个训练没见过的目标点，成功 ≥7 即接入达标
```

---

## 8. 附录：工程细节（给要动手的人）

### 8.1 h5 字段全表（大脑组转换器的产出 = 小脑组载入器实际读的）

| 数据集路径 | 形状/单位 | 含义 | 小脑组载入器读吗 |
|---|---|---|---|
| `observations/state` | (T,35)；第 6~9 列**度**，其余 0 | 每帧的四关节状态 | ✅ **必需** |
| `actions/joint_increment` | (T,16,4)；**度** | 未来 16 帧的每帧增量 | ✅ 可选但保留（缺了会用差分重算） |
| `annotations/valid` | (T,) 真假 | 有效帧 | ✅ 可选（缺省全真） |
| `annotations/goal` | (T,6)；米 | 精确的 pick/place xyz | ✅ 打上 §8.4 补丁后优先读取（之前被忽略） |
| `annotations/phase` | (T,)；2/4 | 相位标签 | ✅ 打上 §8.4 补丁后优先读取（之前被忽略） |
| `annotations/action_pad`、`joint_rad`、`goal`、`phase` | — | 补零掩码/调试用副本 | ❌（保留无害） |

文件属性：`timeline=beats`、`control_frequency=10`（名义）、`chunk_size=16`、
`history_length=10`、`calibrated`、`unit_joint_increment=deg`、`phase_mode` 等。
文件命名 `episode_000000.h5` 起；小脑组载入器只扫一层这个通配符（不递归）。

**单位为什么这么定**：状态用度是小脑组模型的设计；增量写"度"是因为小脑组
载入器用"99% 的数超过 0.5 就当度"这个规则自动识别单位——度制在正常和大动作
两种幅度下都能被正确识别（弧度制遇到大摆转会被误判）。

### 8.2 标定文件（那把"尺"，训练和部署共用）

**真尺：`configs/machine_agx365.json`（`calibrated: true`，2026-10-12 交付）**；
扫描原表 `configs/calibration/agx365_cylinders.json`（401 点 disp↔θ↔关节角）；
schema 模板保留在 `configs/machine_agx365.template.json`。
内容：四个关节的原生单位声明、swing 零位偏移、每缸"位移（米）↔ 角度（度）"的
poly5 换算表、挖/卸点的齿尖高度。`calibrated` 字段只有变成真，转换器才接受。
标定方法：`scripts/calibrate_agx365.py`（解析 AGX 导出的 OpenPLX 文本，纯几何，
不需要许可证；四连杆闭环 + ψ 分支连续跟踪；符号由 112 条实采数据回放选定；
自检含：销距一致性、臂长/四连杆尺寸对拍、装斗时刻齿尖高度）。
注：旧文档提的 `scripts/agx_machine_fit.py`（桥接实测采样 → 拟合）保留，
但其"每缸线性仿射"模型在全行程下表示不了 365 的曲柄滑块非线性
（几何合成样本残差 0.227 m > 其 0.05 门）；如走桥接实测路线，需先给它升级多项式。

### 8.3 五条一致性红线（含 2026-10-12 自查）

1. **换算只有一个来源**：米↔角度只走标定文件，训练和部署用同一份
   —— ✅ 转换器与部署适配器都指向 `configs/machine_agx365.json`
   （适配器 policy 节的 `machine:` 键指同一文件，示例见 `docs/cerebellum-interface-draft.md` §8.5）；
2. **拍 = 步 = 帧**：不重采样、不二次切分 —— ✅ 转换器默认 `timeline=beats`；
3. **相位同源**：训练标签和部署估计都是 2/4 规则 —— ✅ 转换器 `phase_mode=two`，
   适配器 `phase_dig=2/phase_dump=4`；
4. **目标语义**：挖点/卸点 = pick/place，同一对坐标 —— ✅ xy 原样用录制值；
   z 为实测齿尖高度中位（§3 步骤 6 注明口径）；
5. **关节顺序**：`swing, boom, arm, bucket` 三处一致（对不上大脑组的适配器
   **直接报错**，不会默默出错）—— ✅ 数据/机器文件/适配器三处一致。

### 8.4 载入器补丁（优先读大脑组写入的标注）

**补丁文件：`docs/patches/goalact_loader_read_annotations.patch`**（对你们
`src/act_trajectory/data.py` 的真实 diff；应用命令见 §7；已在本机副本验证）。
改动本质（等价于早前说的"6 行"）：

```python
# infer_real_episode 里：读文件时顺带把两列标注读出来
goal_annotation  = np.asarray(f["annotations/goal"],  dtype=np.float32) if "annotations/goal"  in f else None
phase_annotation = np.asarray(f["annotations/phase"], dtype=np.int64)   if "annotations/phase" in f else None
# 原来那两行改成"有标注用标注，没有回退原逻辑"
goal  = goal_annotation  if goal_annotation  is not None else <原有 FK 极值伪推断>
phase = phase_annotation if phase_annotation is not None else <原有液压阈值/时间八等分>
```

打上之后，训练见到的目标/相位与大脑组部署时给的完全同源（phase 2/4、
goal 六个数），这正是"训练条件和部署条件一致"的关键一步。

### 8.5 FAQ

- **为什么不能直接把大脑组的 npz 喂进小脑组管线？** 单位（米/弧度 vs 全角度）、
  动作（绝对 vs 增量）、条件（4 坐标 vs 6 坐标+相位）都不同——转换器就是干
  这个的（已经做好了）。
- **为什么步长上限不能沿用旧值？** 小脑组配置里的旧值是合成数据的步长（约 2.5°）；
  大脑组的"拍"是大步（可达 20°+/拍）。推理端按旧值截断 = 每拍只让走一小步，
  一个循环慢好几倍。用大脑组导出的 `suggested_config.json` 里的值。
- **为什么必须先标定？** 不标定，"米↔度"就是拍脑袋——训练数据语义错位，
  训出来的模型对不上部署换算。

### 8.6 双方路径对照

| 东西 | 大脑组（本仓库 `E:\jiuwensymbiosis`） | 小脑组（`E:\AGX-2.42.2.1\act_trajectory\act_trajectory`） |
|---|---|---|
| 采集器 | `extensions/jiuwen_agx/scripts/record_demos.py` | —（无需自建） |
| 原始数据 | `extensions/jiuwen_agx/data/raw/run_all/*.npz` | — |
| 转换器 | `extensions/jiuwen_agx/scripts/lerobot_to_goalact_h5.py` | — |
| 标定 | `extensions/jiuwen_agx/scripts/calibrate_agx365.py`（真尺见 `configs/machine_agx365.json`） | 臂长用同一结果 |
| 训练数据（交接物） | `extensions/jiuwen_agx/data/goalact_h5/run_all_calibrated/`（calibrated=1） | 拷去当 `--real-dir` |
| 建议配置（交接物） | 同目录 `suggested_config.json` | 建进 `configs/agx365.json` |
| 配置样例（交接物） | `extensions/jiuwen_agx/docs/agx365_training_config.json` | 可直接拷进 `configs/` |
| 载入器补丁（交接物） | `extensions/jiuwen_agx/docs/patches/goalact_loader_read_annotations.patch` | `git apply -p1` 应用到 `src/act_trajectory/data.py` |
| 接入适配器 | `src/jiuwen_agx/policy_goal_act.py` | —（无需改） |
| 验收 | `scripts/act_acceptance.py` | — |

### 8.7 附带说明：小脑组自带的 AGX 路线

小脑组项目里有其自研的 AGX 验证工具（`agx_validation/`，用 TCP 服务把模型
接进 AGX）。它适合小脑组自行跑对照实验；**正式接入以本文档 §6 的路线为准**
（走大脑组的 `act_exec`：护栏、成功判据、验收标准统一）。

---

*本文档由大脑组维护；所有字段名与行为都对照过双方代码。数字举例取自真实
数据文件（ep_0000 循环，真尺版）；§2.2 示例与 §3 走查对应
`data/goalact_h5/run_all_calibrated/episode_000000.h5`。*
