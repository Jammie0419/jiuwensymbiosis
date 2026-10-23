# Goal-ACT 本机预演报告（2026-10-12）

**目的**：把「真尺数据 → 小脑组代码训练 → goal-act-v2 ckpt → 大脑组适配器 → act_exec
闭环」这条链在本机完整跑一遍，拿到"链路已通"的直接证据。小脑组原目录零改动：
预演在源码树副本（`extensions/jiuwen_agx/work/goalact_rehearsal/`，gitignore）上进行。

## 装置

| 项 | 内容 |
|---|---|
| 副本 | `work/goalact_rehearsal/`（`act_trajectory` 源码树拷贝，去掉 .venv/outputs） |
| 补丁 | `docs/patches/goalact_loader_read_annotations.patch`（`git apply -p1` 干净应用） |
| 配置 | `docs/agx365_training_config.json` → 副本 `configs/agx365.json` |
| 数据 | `data/goalact_h5/run_all_calibrated`（103 条，真尺版，`calibrated=1`） |
| 环境 | 仓库 `.venv`：torch 2.8.0+cu128 / RTX 4070 SUPER；（matplotlib 未装——训练路径不需要，画图才用） |

## 一、训练（他们的 `train()`，从零训我们的数据）

入口等价于 `scripts/act/train_act.py` 的 `train(cfg, None, output, real_dir=...)`
（无合成预训练——那是小脑组自己的流程；预演只验连通性）：

```
cd extensions/jiuwen_agx/work/goalact_rehearsal
python -u run_rehearsal_train.py
```

结果：

- 50 epochs，**约 13 分钟**（约 16 s/轮的 train+val）；
- best_val_loss **0.1788**（第 49 轮）；验证集 MAE **0.178 / 0.150 / 0.320 / 0.449 度**
  （swing/boom/arm/bucket），val_fk 0.288 m；
- 产物 `outputs/act_agx365_rehearsal.pt`（19 MB，goal-act-v2；内嵌 config
  `machine=AGX365`、`links=[7.8, 3.6, 2.2733]`）；
- 读数：他们的代码能直接读真尺数据并收敛到亚度级 MAE——量纲/符号/时间轴若有一处
  对不上，loss 会炸或平台化，不会安静地低。

## 二、部署闭环（大脑组适配器，mock 后端）

```
python -u run_rehearsal_deploy.py
```

- **A 未改 mock**（scoop 永不载料）：策略生成的每拍目标以 AGX 原生单位到达驱动，
  300 拍兜底按设计终止（`"cycle incomplete after 300 beats ... never loaded"`）
  ——证明 ckpt 加载、MachineMap 换算、拍循环、兜底全通；
- **B 脚本化 scoop**（mock 无真物理，脚本让第 15 次轮询载料、8 拍后卸掉）：
  **`{"ok": true, "beats": 11}`**——`loaded→unloaded` 成功路径跑通；
- 目标量纲抽查：boom 原生目标为米级、落在真尺行程 `[-0.7, 0.4]` 内。

## 三、通了 / 没通 清单

| 环节 | 预演结论 |
|---|---|
| 他们载入器（打补丁）读真尺数据：phase 2/4、goal 六数与标注一致 | ✅（另见 handoff §7 自测命令） |
| 他们的训练代码在这份数据上跑通并收敛 | ✅ |
| 产出的 goal-act-v2 ckpt 被 `policy_goal_act` 适配器加载 | ✅ |
| 适配器→`act_exec` 拍循环（单目标/拍、换算、护栏、成功判据） | ✅ |
| 模型**质量**（闭环 pick/dump 精度、他们的评估线 0.15/0.20 m） | ⬜ 故意不判——预演只证连通性，质量由小脑组用自己的评估流程判 |
| 真桥接（`act_acceptance.py` 10 点） | ⬜ 等 AGX 许可证（handoff §4.4）；demo 桥接是度数假机、scoop 需脚本触发，起不到真物理作用 |

## 复现

`work/` 是 gitignore 的本机目录；两个入口脚本（`run_rehearsal_train.py` /
`run_rehearsal_deploy.py`）随副本保留，重跑只需上述两条命令。给任意新 ckpt
（例如小脑组正式交付件）做同类冒烟时，把脚本里的 `ckpt` 路径换掉即可。
