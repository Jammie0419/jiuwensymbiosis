# ACT 流水线操作手册（跨机：采集 → 训练 → 部署）

> 手把手执行手册，命令级细节。适用场景：**训练不在采集机上进行**——
> 在另一台机器（Windows/Linux 均可）训练，训完把模型搬回来部署。
> 配套文档：`act-data-collection-design.md`（数据格式与质量论证）、
> `act-integration-plan.md`（总体计划）、`cerebellum-interface-draft.md`（三组对接）。
>
> 机器角色（2026-10-04 修订：大脑组本机无 AGX license，跑不了 AGX 仿真）：
> **A 机（采集/部署机）**——**有 AGX license 的机器（仿真组环境）**：AGX 仿真 + 桥接 +
> jiuwensymbiosis + jiuwen_agx；采集（阶段一）与部署验收（阶段八）在此执行，
> `scripts/record_demos.py` 零 lerobot 依赖，可整目录移交仿真组运行；
> **大脑组本机（无 AGX）**：开发、连接验证（`make_smoke_checkpoint.py` + mock 后端
> 冒烟，无需 AGX）、数据打包与配置——不承担采集与部署；
> **B 机（训练机）**——另一台：只需 Python ≥ 3.12 + lerobot + 一份扩展包源码拷贝。

```
A 机：启动桥接 → 采 npz ──(拷贝 npz + 扩展包源码)──► B 机：转换 → 训练 → checkpoint
A 机：YAML 填 checkpoint ◄──────────(拷贝 pretrained_model 目录)──────────┘
A 机：内省验证 → mock 冒烟 → 真桥接验收
```

数据流转的两个关键事实（决定了为什么这样分工）：

1. **采集不需要 lerobot**（npz 是自有格式）——A 机不用装 torch；
2. **转换/训练需要 lerobot**——都放 B 机；A 机只在最后**部署**时才装推理侧的
   `[policy]` extra（一次大下载）。

---

## 阶段一：采集（A 机）

**前置：**

1. 桥接已启动：`start_agx_bridge.bat`（默认 `127.0.0.1:9700`，窗口保持开着）；
2. **remote 配置文件**：工厂配置 `configs/agx_excavator/agx_excavator.yaml` 是
   mock 后端，采集必须用 remote。复制为 `agx_excavator.local.yaml`
   （.gitignore 已忽略），按该文件内"接真 AGX"注释块改四处：
   `backend: "remote"`、`joint_units: null`（混合单位）、`swing_unit: "rad"`、
   探针实测的 `joint_limits` / `home_joints` / `dig_cycle_tuning`；
3. 过闸门（必做，配置错会采出废数据）：
   `python scripts/agx_scene_probe.py --check-config configs/agx_excavator/agx_excavator.local.yaml`；
4. 可选但建议：若要做"教师策略多样化"（远挖浅咬/近挖深咬，让 ACT 学到随目标
   变化的策略），**必须在首次采集前**实施——先采后改教师，数据要重采。

**采集命令（100 条成功循环）：**

```bash
cd extensions/jiuwen_agx
python scripts/record_demos.py --config configs/agx_excavator/agx_excavator.local.yaml --out data/raw/run001 --chains 10 --cycles-per-chain 10 --seed 7
```

- 每条打印 `ok/BAD + 拍数 + 循环秒数`；结尾 `done: N/M passed`；
- 每条一个 npz（约几十 KB，100 条 < 10 MB）；`BAD` 行是成功判据没过的，
  转换阶段还会再核一遍，不用手动删；
- **成功判据**：铲斗质量 > 50 kg 的"铲起→卸掉"序列（与部署运行时同一判据）。

## 阶段二：搬运（A 机 → B 机）

拷两样东西（都已 gitignore，U 盘/网盘/scp 任意）：

| 拷什么 | 大小 | 用途 |
| --- | --- | --- |
| `extensions/jiuwen_agx/` 整个目录（源码） | 几 MB | 转换脚本 + 记录逻辑（B 机**不需要**装 jiuwensymbiosis core，脚本会自动退化为直接加载 `record.py`） |
| `data/raw/run001/`（npz 目录） | < 10 MB | 训练数据 |

## 阶段三：B 机训练环境

```bash
# Python ≥ 3.12（lerobot 硬性要求）
python3.12 -m venv .venv-train && source .venv-train/bin/activate   # Linux
# Windows: py -3.12 -m venv .venv-train && .venv-train\Scripts\activate

pip install "lerobot[training]==0.6.1"        # 含 accelerate/datasets 栈 + torch
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('ok')"
lerobot-train --help                           # CLI 可用即就绪
```

平台注记：

- **Linux**：无额外步骤；
- **Windows**：lerobot 每次存 checkpoint 都建符号链接（无特权会 WinError 1314 崩溃）。
  两条路：开 Windows「开发者模式」设置（推荐）；或用包装脚本
  `scripts/train_act.py`（把 `update_last_checkpoint` 替换为忽略后转发 CLI）；
- 版本**锁 0.6.1**：lerobot 的处理器 API 在版本间不兼容，checkpoint 与库版本
  绑定（部署侧 A 机装同版本）。

## 阶段四：转换（B 机）

```bash
cd <拷来的 jiuwen_agx 目录>
python scripts/npz_to_lerobot.py --raw <data/raw/run001 路径> --root data/lerobot --repo-id local/agx_dig_demos --fps 10
```

结尾打印 `dataset: <目录>  episodes: N  frames: M` 与拍数分布——**记下
`dataset:` 打印的路径**，阶段五要用。`skipped:` 列出剔除原因（期望接近 0；
`mock data must not enter` 说明混入了自测数据；`cycle incomplete` 说明采集
时有循环没完成）。

## 阶段五：训练（B 机）

```bash
lerobot-train \
  --dataset.repo_id=local/agx_dig_demos \
  --dataset.root=data/lerobot/local_agx_dig_demos \
  --policy.type=act --policy.push_to_hub=false --policy.device=cuda \
  --output_dir=outputs/train/act_agx_v1 \
  --batch_size=8 --num_workers=0 --steps=50000 --save_freq=10000
```

三个**必写**项（缺了直接报错）：`--policy.push_to_hub=false`（默认要推
HuggingFace Hub，无 repo_id 会校验失败）；`--output_dir` 指向**不存在**的
目录（已存在即拒）；`num_workers=0`（Windows 免 spawn 问题，Linux 可加回 4）。

- GPU（`--policy.device=cuda`）：50k 步小时级；只有 CPU：降到
  `--steps=20000` 过夜跑；
- 产物：`outputs/train/act_agx_v1/checkpoints/{步数}/pretrained_model/`，
  内含 `config.json` + `model.safetensors` + 归一化前/后处理器——**训练与
  部署必须用同一份目录**，不要只拷权重文件；
- 训练中断可 `--resume` 续训（此时 `--output_dir` 允许已存在）。

## 阶段六：训练质量三门槛（两道在 B 机，一道回 A 机）

1. **loss 收敛**（B 机）：训练控制台的 `l1_loss` 应持续下降并走平；
2. **统计量非退化**（B 机）：`meta/stats.json` 里 swing（弧度）与三个液压缸
   （米）各维 `std > 1e-6`——某一维 std≈0 说明它被当成常数学没了；
3. **留出目标点回放**（**A 机**，B 机没有仿真）：挑 3 个训练数据里没出现的
   (挖点, 倒点) 组合，部署后（阶段八）手动调 `act_exec` 肉眼看完整循环——
   注意轨迹是否连贯、有没有中途卡死。

## 阶段七：搬运回（B 机 → A 机）

拷**一个目录**：选定 checkpoint 的 `pretrained_model/`（约 160 MB，fp32）。
放 A 机的 `extensions/jiuwen_agx/data/ckpt/act_v1/`（gitignored）。

## 阶段八：部署与验收（A 机）

1. 装推理侧依赖（一次性，下载较大）：
   `uv pip install -e "extensions/jiuwen_agx[policy]"`（等价于装 lerobot，锁同版本）；
2. 配置：在 `agx_excavator.local.yaml` 的 `low_level` 节加：

   ```yaml
   policy:
     name: "act"
     ckpt: "data/ckpt/act_v1/pretrained_model"   # 相对扩展包根或绝对路径
     device: "cpu"                                # 有卡可 "cuda"
   ```

3. 验证链（由浅入深）：
   - 内省：`jiuwensymbiosis-actions --config <local.yaml>` 能看到 `act_exec`
     （不配 policy 节时必须看不到）；
   - mock 冒烟：`--config` 指 mock 配置 + policy 节，调一次 `act_exec`——
     torch 推理链路跑通即算过（mock 铲斗永远不满足成功判据，`max_beats`
     兜底失败是**预期结果**，看 error 文案可读即可）；
   - 真桥接验收：10 个训练没见过的目标点 ≥ 7 个成功；与 `dig` 出对比表
     （成功率 / 循环秒数 / 拍数）。

---

## 故障排查表

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `ValueError: 'repo_id' argument missing`（训练启动即崩） | 没写 `--policy.push_to_hub=false` | 补上（§阶段五命令已含） |
| `FileExistsError: Output directory ... already exists` | `--output_dir` 已存在 | 换新目录，或加 `--resume` 续训 |
| Windows 训练在第一次存档时崩 `WinError 1314` | checkpoint 符号链接无特权 | 开「开发者模式」或用 `train_act.py` 包装（§阶段三） |
| 转换器报 `mock data must not enter` | npz 里混入了 `--allow-mock` 的自测产物 | 删掉对应 npz 重转 |
| 转换器报 `joint_units/swing_unit ... != ...`（整批拒绝） | 两次采集用了不同单位的机器配置 | 数据集只能来自同一份配置；分数据集转换，或重采 |
| 转换器报 `cycle incomplete` 居多 | 采集时桥接/仿真异常中断 | 看 A 机采集日志的 BAD 行；检查桥接稳定性后重采 |
| `lerobot is not installed`（转换器） | B 机没装 | `pip install "lerobot[training]==0.6.1"`（§阶段三） |
| 部署时 `policy unavailable: No module named 'torch'` | A 机没装 `[policy]` extra | `uv pip install -e "extensions/jiuwen_agx[policy]"`（§阶段八） |
| `act_exec` 返回 `unknown policy 'act'` | 扩展包没装 `[policy]` extra，注册表里只有 fake | 同上 |
| 内省看不到 `act_exec` | 配置里没有 `policy:` 节（这是**设计行为**） | 要用就配上 policy 节 |
| 训练 loss 不降 / 某维输出僵死 | 统计量退化或数据量不足 | 门槛 2 抽查 stats；加采集条数（100→200）比改模型便宜 |

## 与小脑组的关系

若训练由小脑组执行（而非你自己）：本手册**阶段三～七就是他们的操作说明**，
你们组只需交付阶段二的 npz + 扩展包源码，并接收他们回传的 `pretrained_model`
目录走阶段八。数据格式细节与质量判据见 `act-data-collection-design.md`；
接口分工见 `cerebellum-interface-draft.md`。
