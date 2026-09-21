# jiuwen_agx — jiuwensymbiosis 扩展包：AGX 仿真机器

通过 jiuwensymbiosis 框架控制 AGX Dynamics 仿真场景里的机器。当前内置：
**履带挖掘机**（回转 + 动臂 + 斗杆 + 铲斗 + 履带行走）+ 共享仿真底座。
AGX 自带模型里的卡车（BedTruck）、装载机、推土机、吊车等都能用同一套底座接入
（见"控制其它 AGX 模型"）。

工作方式：**Windows 本机**跑 AGX（自带窗口）+ 桥接，本仓库 venv 发任务。

## 快速开始（三步）

```bat
rem ① 装一次（editable；core 不在 PyPI，故 --no-deps）
uv pip install -e ./extensions/jiuwen_agx --no-deps

rem ② 起 AGX + 桥接：双击即可（弹出挖掘机 + 沙地窗口，桥接监听 127.0.0.1:9700）
extensions\jiuwen_agx\scripts\start_agx_bridge.bat

rem ③ 另开一个终端发任务（任务不进 config，一律用 --query 给）
.venv\Scripts\python.exe examples\run_task.py ^
  --config extensions\jiuwen_agx\configs\agx_excavator\agx_excavator.local.yaml ^
  --query "挖一斗土倒到旁边"
```

窗口里的挖掘机随即执行：读地形 → 摆转对准 → 下铲 → 收斗 → 摆转 → 卸料 → 归位
（一轮约 19 s）。实测记录：`get_terrain → dig(3,0 → 6,0) → home` 三步 `ok`。

`.local.yaml` 被 gitignore，仓库里没有这份文件——按
`configs/agx_excavator/agx_excavator.yaml` 复制一份再改（AGX 相关的值照下面
"关节单位"一节填）。

## 三条启动路径别搞混（最容易踩的坑）

| 目的 | 命令 | 会动吗 |
| --- | --- | --- |
| **真 AGX**（看画面、演示） | `scripts\start_agx_bridge.bat` | ✅ 窗口里的挖掘机真动 |
| **只验协议/规划链路**（无 AGX） | `scripts\start_demo_bridge.ps1` | ❌ 内存假机，命令"成功"但没物理 |
| **核对场景真值**（只读，不改动） | `python scripts/agx_scene_probe.py --bridge --port 9700` | — 关节名/范围/单位清单 |

## 排障：症状 → 原因

| 症状 | 原因与处理 |
| --- | --- |
| 命令显示成功、画面纹丝不动 | 连到了内存演示机。控制端连上时会打印对端机器名：`demo_excavator` 就是连错了，真机报 `excavator365` |
| 桥接日志出现 `上一条底盘命令还没走完就被新命令覆盖` | 有客户端没等底盘到位就连发命令：pump 模式下底盘命令不阻塞，连发的**相对**位移会互相覆盖（实测 9×0.7 rad 只走出 0.7 rad）。正常路径（`RemoteSimBackend`）会轮询 `base_state` 等到位；自己写脚本连桥接时要照做 |
| 日志刷 `[planner] attempt N/3 failed: The read operation timed out` | 意图解析那次 LLM 调用超时（推理型模型常见）。调大 YAML 里的 `agent.intent_timeout_s`（默认 60 s），慢端点给 90~120 |
| 探针/任务超时，桥接像死了一样 | **AGX 窗口里的仿真被暂停了**（按了空格/暂停键）：泵挂在每仿真步回调上，不步进就不收发网络。桥接控制台会打印 `WARNING: 仿真已暂停 …`，在窗口里按空格/播放键恢复即可 |
| 起桥接时提示「端口 9700 已被占用」 | 已有实例在监听（多半是没关掉的旧 AGX 窗口）。**关掉旧窗口**，或改 `JIUWEN_BRIDGE_PORT` 再启动——两个桥接同时监听同一端口时，客户端连到哪一个是**不确定的**（Windows 允许重复绑定），所以启动前会拒绝 |
| 桥接跑着跑着自己没了 | AGX 窗口被关掉或按了 ESC（`ExampleApplication` 的默认行为：关窗/ESC 即退出 run loop）。重启 `start_agx_bridge.bat` 即可 |
| 规划失败，提示 `dig requires ['payload.clear'] but the state here is ['payload.held']` | 铲斗载料误判。空斗贴地时 AGX Terrain 的 aggregate 本底 0~1 kg，阈值取小了就永久"有料"。阈值见 `agx_bridge_server.py:LOADED_MASS_THRESHOLD_KG`，用 `inventory` 的 `bucket_mass_kg` 校准 |
| 用 `agxViewer 插件.agxPy` 起不来：`No module named 'agxPythonModules'` | agxViewer 会用用户配置文件的环境重建进程，丢掉 PATH/PYTHONPATH，于是加载到别的 Python（本机 Anaconda 3.12.7），而 AGX 模块要求精确 3.12.10。**用 `start_agx_bridge.bat`** |
| `dig` 报 `joint 'X' target ... outside configured limits` | 关键帧与限位单位不一致（回转 rad / 液压缸 m）。跑 `--check-config` 逐条看，见"关节单位" |
| 想手查某个关节现在多少、或手动摆姿态 | `python scripts/monitor_ui.py --port 9700`（浏览器 8050） |

## 关节单位：AGX 365 是混合单位（改限位/关键帧前必读）

| 关节 | AGX 约束 | 单位 |
| --- | --- | --- |
| swing | `CabinHinge` | 弧度 rad |
| boom | `ArmPrismatic1/2`（双缸，整组同步驱动） | 米 m |
| arm | `StickPrismatic` | 米 m |
| bucket | `BucketPrismatic` | 米 m |

所以 config 里 `joint_units` **留空**（`null`，混合单位不能撒谎成 deg/rad），
回转单位由 `swing_unit: "rad"` 单独指明；`joint_limits` 与 `dig_cycle_tuning`
里的 boom/arm/bucket 关键帧一律填**米**（键名沿用历史 `*_deg` 命名，值是原生单位）。

实测行程（`--direct`/`--bridge` 探针一致）：boom `[-0.7, 0.4]`、arm `[-0.88, 0.8]`、
bucket `[-1.0, 0.312]`，swing 是无限位回转（报告按 ±180 兜底）。
关键帧现值取自 AGX 官方自动挖掘循环（`data/python/agxTerrain/excavator_365_terrain.agxpy`
的 `digCycle`），已收敛进限位内侧。

改完先过闸门——它会逐条报出"关键帧越限"这类 dig 执行时才会遇到的问题：

```bash
python scripts/agx_scene_probe.py --check-config configs/agx_excavator/agx_excavator.local.yaml
```

## 三种操控方式

| 方式 | 命令 | 适合 |
| --- | --- | --- |
| **自然语言**（LLM 当司机） | `run_task.py --config ... --query "..."` | 日常演示、任务级控制 |
| **代码直接控制** | `build_agx_excavator_session(...)` → `api.move_joint / dig / navigate_relative` | 精确测试、写自动化脚本 |
| **手动/观察** | `monitor_ui.py`（关节读数 + 逐个设目标）；或启动前 `set JIUWEN_KEYBOARD=1` 用键盘开 | 标定姿态、排查"到底动没动" |

`monitor_ui.py` 是**标定与排障工具**，不是主控入口：面板顶部标出对端是哪台机器
（demo 假机红字警告），关节按原生单位显示（旧版显示成 `°` 是错的），可逐个关节设
目标——找 `home_joints` / `dig_cycle_tuning` 时摆到满意姿态、抄读数即可。
注意桥接**一次只服务一个客户端**：跑 `run_task.py` 时别同时开面板（会互相等超时）。

键盘键位（官方 Excavator365 绑定）：`a/z` 铲斗 收/放 · `s/x` 斗杆 收/伸 ·
`↑/↓` 动臂 升/降 · `←/→` 上车回转 · `PgUp/PgDn/Home/End` 左右履带行走。

## 控制其它 AGX 模型（以卡车 BedTruck 为例）

AGX 自带 `data/models/BedTruck.agx`（17 刚体 / 17 约束）。关键约束（探针实测命名）：
`BedHinge1` / `BedHinge2` 是货箱倾卸铰链，`Hinge2~11` 是车轮悬挂，`Lock8~13` 是锁定
约束（桥接自动跳过）。`.agx` 里的约束以基类包装加载，桥接会按 `asHinge()/asPrismatic()`
下转型。

**立刻可用（无需写代码）**——headless 加载卡车场景 + 映射货箱铰链：

```bat
%AGX_DIR%\python-x64\python.exe extensions\jiuwen_agx\scripts\agx_bridge_server.py ^
  --scene "%AGX_DIR%\data\models\BedTruck.agx" --joints bed --joint-map bed=BedHinge1 --port 9700
```

之后 `move_joint({"bed": 0.6})` 就能倾卸货箱（卡车不需要专属适配器）。
想看画面就在 `start_agx_bridge.bat` 里把 `JIUWEN_BRIDGE_EXCAVATOR` 设为 `0`、
并把 `--scene` 换成卡车场景（`JIUWEN_BRIDGE_SCENE` 见脚本注释）。

**升级成正式 DUMP 动作**（让 LLM 用自然语言指挥，约 150 行）：照核心文档
`docs/zh/how-to/write-an-extension-package.md` 的四条缝——`contracts.py` 加
`DumpResult|DumpFailure`、`actions.py` 加 `DUMP` 契约、注册 `motion.dump` 能力、
薄包 `agx_truck/` 里写关键帧循环并加 entry-point。轮式行走需要把
`navigate_relative` 的驱动轮从"履带链轮"泛化为 `--joint-map` 声明的轮组。

## 多模型同场景 / 自己建的模型

一个桥接实例能控制整个场景里的多台机器——约束在全场景范围查找，把各机器的关节
合进一份 `--joints` + `--joint-map` 即可：

```bat
%AGX_DIR%\python-x64\python.exe extensions\jiuwen_agx\scripts\agx_bridge_server.py ^
  --scene 混合场景.agx --joints swing,boom,arm,bucket,bed ^
  --joint-map swing=CabinHinge,boom=ArmPrismatic1,arm=StickPrismatic,bucket=BucketPrismatic,bed=BedHinge1
```

自己建的模型同理：模型里有命名过的 Hinge / Prismatic，用探针（`--direct` 本地盘点，
或 `--bridge` 远程）确认约束名 → `--joint-map` 映射 → 通用动作
（`move_joint` / `get_joint_positions` / `home`）即刻可用。专属代码（dig/DUMP 这类
工作循环）只在想让 LLM 调用复合动作时才需要。

## 文件与平台

| 文件 | 跑在哪 | 说明 |
| --- | --- | --- |
| `agx_bridge_server.py` | AGX 自带 Python（headless） | 纯标准库；`--demo` 假机 / `--scene` 真实场景；协议 v1 JSON 行 |
| `agx_viewer_bridge.agxPy` | AGX 自带 Python 3.12（自带窗口） | 插件；`init_app` 自建 `ExampleApplication` + 每步回调泵 |
| `start_agx_bridge.bat` | Windows | 设好 PATH/PYTHONHOME/PYTHONPATH 后调 AGX python 跑插件 |
| `start_demo_bridge.ps1` | Windows | 无 AGX 时的协议联调（内存假机，带警告横幅） |
| `monitor_ui.py` | 仓库 venv | NiceGUI 面板：对端身份 / 原生单位关节读数 / 手动设目标 |
| `agx_scene_probe.py` | 两边 | `--direct`（AGX python 本地盘点）/ `--bridge`（远程清单）/ `--check-config`（配置闸门） |

跨平台边界 = 版本化 JSON 行协议 over TCP。跨机控制（AGX 在另一台机器）时把 config
的 `host` 改成那台机器的 IP，启动前设 `JIUWEN_BRIDGE_HOST=0.0.0.0`，并放行防火墙：

```bat
netsh advfirewall firewall add rule name="AGX Bridge" dir=in action=allow protocol=TCP localport=9700
```

## 后端

| backend | 场景 |
| --- | --- |
| `mock` | 内存模拟，零依赖，离线开发/测试（模板 config 的默认） |
| `remote`（推荐） | TCP 桥接；`--demo` 无需 AGX，`--scene` 加载任意 .agx |
| `inprocess` | 本进程 `import agx`——需 venv 与 AGX 的 Python 小版本恰好一致，当前不满足，未实现 |

## 添加下一台仿真机器

见 `docs/add-sim-machine.md`（四步配方 + 每步通过标准）。共享底座全部复用，
每台机器约 150~250 行。

## license

AGX 是商业软件，需要本机许可证：`C:\Users\<你>\AppData\Local\Algoryx\agx\agx.lfx`
（用 AGX 安装目录的 `LicenseManager.bat` 激活/刷新）。本机已激活——实测挖掘机可动。
