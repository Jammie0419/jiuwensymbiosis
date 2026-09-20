# jiuwen_agx — jiuwensymbiosis 扩展包：AGX 仿真机器

通过 jiuwensymbiosis 框架控制 AGX Dynamics 仿真场景里的机器。当前内置：
**履带挖掘机**（回转 + 动臂 + 斗杆 + 铲斗 + 履带行走）+ 共享仿真底座。
**AGX 自带模型动物园里的卡车（BedTruck）、装载机（wheel_loader ×4）、推土机
（bulldozer）、吊车（crane）等都能用同一套底座接入**（见下文"控制卡车"）。

## 安装

```bash
# 前置：jiuwensymbiosis 本体已 editable 安装（见核心仓库 README）
uv pip install -e ./extensions/jiuwen_agx --no-deps   # --no-deps：core 不在 PyPI
```

安装后本体自动被发现：config 顶层 `adapter: agx_excavator` 即可，
run_task / introspect / GUI 通过 entry-points 找到会话构建器。

## Windows 浏览器观看 + 操控（你的日常工作流）

你不需要 RDP，也不需要在 Windows 装任何东西——**浏览器就是显示器**：

**① 服务器上启动直播**（SSH / VS Code 终端里，一次一条命令）：

```bash
cd /mnt/disk2/lzm/jiuwensymbiosis
bash extensions/jiuwen_agx/scripts/agx_viewer_stream.sh
```

脚本会拉起 4 个进程：虚拟屏幕 → agxViewer（挖掘机 + 沙地 + 桥接:9700）→
x11vnc 抓屏 → noVNC 网页代理(:6080)，然后打印一个地址。

**② Windows 浏览器打开**打印的地址（形如 `http://10.157.197.52:6080/vnc.html`），
点左侧 **"连接"**——挖掘机画面实时出现在浏览器里，**鼠标还能直接拖拽旋转视角**
（操作的是 agxViewer 相机）。

**③ 发任务**（另开一个终端，SSH 或 VS Code 都行）：

```bash
cd /mnt/disk2/lzm/jiuwensymbiosis && source .venv/bin/activate
python examples/run_task.py \
  --config extensions/jiuwen_agx/configs/agx_excavator/agx_excavator.local.yaml \
  --query "把土堆的土挖一斗倒到挖点右边3米处"
```

浏览器画面里的挖掘机就会执行：读地形 → 摆转对准 → 下铲 → 收斗 → 摆转 → 倾倒 → 归位。

> 结束观看：服务器终端 `Ctrl+C`（自动清理全部 4 个进程）。
> 局部提醒：VNC 无密码、仅限实验室局域网，不要把 6080 端口暴露公网。

## Windows 本机跑 AGX（本机 = 身体 + 屏幕）

不做无头直播、就想在本机看着挖掘机动时用这条路径：**AGX 仿真 + 窗口 + 桥接
在同一个进程里**，控制端连本机 9700。

```bat
rem 1. 双击（或在任意 cmd 里跑）——它会自己弹 AGX 窗口（365 挖掘机 + 沙地）
extensions\jiuwen_agx\scripts\start_agx_bridge.bat

rem 2. 另开一个普通终端跑控制端（用本仓库 .venv）
.venv\Scripts\python.exe examples\run_task.py ^
  --config extensions\jiuwen_agx\configs\agx_excavator\agx_excavator.local.yaml ^
  --query "挖一斗土倒到旁边"
```

启动脚本做的事：设好 `PATH/PYTHONHOME/PYTHONPATH`（AGX 自带 Python），
再执行 `%AGX_DIR%\python-x64\python.exe agx_viewer_bridge.agxPy`——
`init_app` 会自己建 `agxOSG.ExampleApplication`，窗口与仿真循环都来自它。
`AGX_DIR` 默认 `E:\AGX-2.42.2.1`，装到别处就设这个环境变量。

> **为什么不是 `agxViewer agx_viewer_bridge.agxPy`**：agxViewer 会用*用户配置文件*
> 的环境重建进程，丢掉我们设的 PATH/PYTHONPATH，于是加载到机器上别的 Python
> （本机是 Anaconda 3.12.7），而 AGX 的 Python 模块要求精确 3.12.10，导入直接
> 失败：`No module named 'agxPythonModules'`。这是实测结论，别再往回改。

跨机部署（Windows 跑画面、另一台机器跑大脑）时把控制端 config 里的
`host` 改成 Windows 的局域网 IP，并在 Windows 上放行防火墙 9700/TCP
（启动前设 `JIUWEN_BRIDGE_HOST=0.0.0.0`）。

> `.local.yaml` 被 gitignore，仓库里没有这份文件——按
> `configs/agx_excavator/agx_excavator.yaml` 复制一份再改。

### 三条启动路径别搞混（这是最容易踩的坑）

| 目的 | 命令 | 会动吗 |
| --- | --- | --- |
| **真 AGX**（看画面、演示） | `scripts/start_agx_bridge.bat` | ✅ 窗口里的挖掘机真动 |
| **只验协议/规划链路**（无 AGX） | `scripts/start_demo_bridge.ps1` | ❌ 内存假机，命令"成功"但没物理 |
| **核对场景真值**（不改动任何东西） | `python scripts/agx_scene_probe.py --bridge --port 9700` | — 只读清单（关节名/范围/单位） |

**症状对照**：任务显示成功、但窗口里的机器纹丝不动 —— 十有八九是连到了
内存演示机。控制端连上时会把对端机器名打进日志：看见
`bridge 127.0.0.1:9700 的对端是 'demo_excavator'` 就是连错了，
真机会报 `excavator365`。

另一个曾经踩过的坑：任务直接报"规划失败 / dig 前置条件不满足"。世界状态里的
`payload.held` 来自桥接对铲斗土壤质量的实测，阈值见
`agx_bridge_server.py:LOADED_MASS_THRESHOLD_KG`——空斗贴地时接触区土壤会被算进
aggregate（实测本底 0~1 kg），阈值取小了就会永久判定"斗里有料"，`dig` 永远
不可规划。改阈值前先用 `inventory` 的 `bucket_mass_kg` 读一次实际值。

### 关节单位：AGX 365 是混合单位（改限位前必读）

| 关节 | AGX 约束 | 单位 |
| --- | --- | --- |
| swing | `CabinHinge` | 弧度 rad |
| boom | `ArmPrismatic1/2`（双缸） | 米 m |
| arm | `StickPrismatic` | 米 m |
| bucket | `BucketPrismatic` | 米 m |

因此 config 里 `joint_units` **留空**（`null`，混合单位不能撒谎成 deg/rad），
回转单位由 `swing_unit: "rad"` 单独指明；`joint_limits`、`dig_cycle_tuning`
里的 boom/arm/bucket 关键帧一律填**米**（键名沿用历史 `*_deg` 命名，值是原生单位）。

填之前先跑探针 `--bridge` 拿实测范围，再跑闸门自检：

```bash
python scripts/agx_scene_probe.py --check-config configs/agx_excavator/agx_excavator.local.yaml
```

它会逐条报出"关键帧越限"这类问题——dig 执行时会遇到的拒绝，在这里提前暴露。

想手动摆姿态抄数（填 `home_joints` / `dig_cycle_tuning` 用）：

```bash
python scripts/monitor_ui.py --port 9700        # 浏览器 http://127.0.0.1:8050
```

面板顶部标出对端是哪台机器（demo 会红字警告），可逐个关节设目标并读回原生值；
也可以 `set JIUWEN_KEYBOARD=1` 后用键盘开，面板负责读数。

## 三种操控方式

| 方式 | 命令 | 适合 |
| --- | --- | --- |
| **自然语言**（LLM 当司机） | `run_task.py --config ... --query "..."` | 日常演示、任务级控制 |
| **代码直接控制** | `build_agx_excavator_session(...)` → `api.move_joint/dig/navigate_relative` | 精确测试、写自动化脚本 |
| **键盘手动开** | 启动前 `export JIUWEN_KEYBOARD=1`，然后在 noVNC 画面聚焦后按键 | 手感调试、演示互动 |

键盘键位（官方 Excavator365 绑定）：`a/z` 铲斗 收/放 · `s/x` 斗杆 收/伸 ·
`↑/↓` 动臂 升/降 · `←/→` 上车回转 · `PgUp/PgDn/Home/End` 左右履带行走。

## 控制 AGX 卡车（BedTruck）——现在就能做 + 升级路线

### 现状：底座支持，卡车走"场景文件 + 约束映射"路径

AGX 自带自卸卡车模型：`data/models/BedTruck.agx`（实测 17 个刚体、17 条约束）。
关键约束（探针实测命名）：

| 约束名 | 推测功能 | 可驱动 |
| --- | --- | --- |
| `BedHinge1` / `BedHinge2` | **自卸货箱倾卸铰链**（卡车核心动作） | ✓ Hinge |
| `Hinge2` ~ `Hinge7`、`Hinge9` ~ `Hinge11` | 车轮/悬挂 | ✓ Hinge |
| `Lock8` ~ `Lock13` | 锁定约束 | ✗（`_as_actuator` 自动跳过） |

**注意**：`.agx` 文件里的约束以基类 `Constraint` 包装加载，桥接已做 SWIG
下转型（`asHinge()/asPrismatic()`），Lock 类自动跳过。

### 方式一：立刻控制货箱倾卸（move_joint 路径，无需写任何代码）

```bash
# 服务器上（有 AGX 环境）：headless 加载卡车场景 + 手动映射货箱铰链
source /opt/Algoryx/AGX-2.42.2.1/setup_env.bash
export JIUWEN_SCRIPTS_DIR=$PWD/extensions/jiuwen_agx/scripts
python extensions/jiuwen_agx/scripts/agx_bridge_server.py \
  --scene /opt/Algoryx/AGX-2.42.2.1/data/models/BedTruck.agx \
  --joints bed \
  --joint-map bed=BedHinge1 --port 9700
```

控制端（另一个终端）用 30 行纯标准库客户端即可驱动（卡车不需要专属适配器，
`move_joint` 直接倾卸货箱）：

```python
import json, socket
sock = socket.create_connection(("127.0.0.1", 9700), timeout=10)
f = sock.makefile("rw", encoding="utf-8", newline="\n")

def call(cmd, **kw):
    f.write(json.dumps({"v": 1, "cmd": cmd, **kw}) + "\n"); f.flush()
    return json.loads(f.readline())

print(call(cmd="move_joints", targets={"bed": 0.6}, timeout_s=10))  # 倾卸货箱
print(call(cmd="joints"))                                            # 读角度
```

> 也可在 **Windows 的 AGX 环境里**直接跑同样命令（把 `scripts/` 目录拷过去，
> 命令用 `python agx_bridge_server.py --scene ...`，用 AGX 菜单里自带的 Python）。

### 方式二：浏览器里全程看卡车

与挖掘机完全相同——`agx_viewer_stream.sh` 把 `--scene` 换成卡车场景即可：

```bash
# 编辑 stream 脚本里的 agxViewer 行，或在命令行覆盖：
agxViewer /opt/Algoryx/AGX-2.42.2.1/data/models/BedTruck.agx \
  --attachScript extensions/jiuwen_agx/scripts/agx_viewer_bridge.agxPy
```

（插件通过环境变量 `JIUWEN_BRIDGE_EXCAVATOR=0` 声明"对已加载场景做约束发现"。）

### 方式三：升级成正式的 DUMP 动作（推荐，约 150 行）

让 LLM 能用自然语言指挥卡车（"把货卸在右边"），照核心文档
`docs/zh/how-to/write-an-extension-package.md` 的四条缝：

1. `jiuwen_agx/contracts.py` 加 `DumpResult | DumpFailure`
2. `jiuwen_agx/actions.py` 加 `DUMP = ActionSpec(name="dump", capability="motion.dump", ...)`
3. `core env/base.py` `register_capability("motion.dump")`（扩展包 import 时注册）
4. 薄包 `agx_truck/`：`move_joints_blocking({"bed": 倾卸角})` 的关键帧循环 +
   `register_actions(DUMP)` + entry-point 加一条 `agx_truck`
5. （可选）轮式导航：把 `navigate_relative` 的驱动轮从"履带链轮"泛化为
   "joint-map 声明的轮组速度差"——底座协议不变

### 当前卡车路径的已知限制

- `navigate_relative`（行走）目前实现为**履带差速**（挖掘机链轮专用），卡车
  轮式行走需要上面的第 5 步泛化——底座协议不变，只是桥接实现补充
- `BedTruck.agx` 的约束名是通用的（Hinge2/3/...），**必须用 --joint-map 显式
  指定**哪个是货箱、哪些是驱动轮（探针输出可辅助确认）
- 无 license 时与挖掘机相同：命令通、执行器不动

## 多模型同场景 / 自己建的模型

**同一场景放多台机器**（比如挖掘机 + 卡车）：一个桥接实例就能控制——约束是在
整个场景范围里查找的，把各机器的关节合进一份 `--joints` + `--joint-map`：

```bash
python extensions/jiuwen_agx/scripts/agx_bridge_server.py \
  --scene 混合场景.agx \
  --joints swing,boom,arm,bucket,bed \
  --joint-map swing=CabinHinge,boom=ArmPrismatic1,arm=StickPrismatic,bucket=BucketPrismatic,bed=BedHinge1
```

`move_joint({"bed": 0.6})` 控卡车货箱、`dig(...)` 控挖掘机，同一条连接顺序执行。
**限制**：单桥接同时只接受一个控制连接（两个 LLM 会话要排队/分时）；
`navigate_relative` 行走目前只实现履带差速（挖掘机）——轮式机器的行走见
上方卡车章节第 5 步配方。

**自己建的模型**：只要模型里有你命名过的 Hinge / Prismatic 约束，加载场景后
用探针（`--bridge` 清单）确认约束名 → `--joint-map` 映射 → 全部通用动作
（`move_joint` / `get_joint_positions` / `home`）即刻可用。`.agx` 文件里以基类
包装的约束会自动下转型（Hinge/Prismatic），Lock 类自动跳过。不需要写任何
专属代码——专属代码（dig/DUMP 这类工作循环）只在想让 LLM 用自然语言调用
复合动作时才需要。

## Windows 兼容性

| 文件 | 跑在哪 | Windows 兼容 |
| --- | --- | --- |
| `agx_bridge_server.py` | **Windows**（AGX 自带 Python，headless）或 Linux | ✅ 纯标准库；已审计：无 Unix 专属调用、print 全 ASCII、异常经 JSON 传输（ensure_ascii 转义） |
| `agx_viewer_bridge.agxPy` | **Windows**（AGX 自带 Python 3.12，自带窗口）或 Linux（xvfb） | ✅ 同上；必须用 AGX 自带 python 跑，不能交给 agxViewer（见上文） |
| `start_agx_bridge.bat` / `start_demo_bridge.ps1` | **Windows** | ✅ 前者设好 AGX 环境后调 AGX python；后者纯标准库 mock |
| `agx_scene_probe.py` | 两边 | ✅ `_out()` 带 GBK 控制台兜底 |
| `agx_viewer_live.sh` / `agx_viewer_stream.sh` | **Linux 服务器**（bash + Xvfb + x11vnc） | ❌ 不适用于 Windows——Windows 侧用 `start_agx_bridge.bat` |
| jiuwensymbiosis 核心（框架、LLM、安全护栏） | Linux 服务器 | 与 Windows 无关 |

跨平台边界 = 版本化 JSON 行协议 over TCP。**在 Windows 上启动的推荐姿势**：
`scripts\start_agx_bridge.bat`（真 AGX，自带窗口 + 桥接）或
`python agx_bridge_server.py --scene 场景.agx`（headless，无窗口）。
Windows 防火墙首次运行需放行端口：`netsh advfirewall firewall add rule name="AGX Bridge" dir=in action=allow protocol=TCP localport=9700`。

## 后端

| backend | 场景 |
| --- | --- |
| `mock` | 内存模拟，零依赖，离线开发/测试 |
| `remote`（推荐） | TCP 桥接本脚本；`--demo` 无需 AGX；`--scene` 加载任意 .agx |
| `inprocess` | 本进程 import agx（需 AGX 自带 Python 3.12 环境；骨架就位） |

## 添加下一台仿真机器

见 `docs/add-sim-machine.md` 四步配方（共享底座全部复用，每台约 150~250 行）。

## license 说明

AGX 是商业软件：无 license 时场景/链路/命令全部可用，但**执行器（电机/液压）
被全局禁用**——关节不动。license 文件放 `~/.config/agx/`（服务器）后重启 viewer
即解锁；Windows 侧放到 `%LOCALAPPDATA%\Algoryx\agx\`。
