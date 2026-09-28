# jiuwen_agx 架构与流程:从大模型指令到 AGX,以及小脑模型接入

> 适用读者:要理解或修改本拓展包、要接新的仿真机器、或要接策略模型(小脑)的同学。
> 全文以 commit `c365bc0` 的代码为准——闭环底盘、负载门控、SafetyRail skip 契约均已生效。
> 接新机器的配方见 [add-sim-machine.md](add-sim-machine.md);待办与验收标准见 [TODO.md](TODO.md)。
>
> **本文怎么读**:第 0 章是三分钟速览;第 1~8 章是**全程实录**——严格顺着一次
> 真实运行的时间线(T0 启动 → T7 收尾)逐步展开,每一步都落到具体文件、函数、
> 真实参数,不看省字;第 9~10 章是参考视图(角色分工、core 的作用、四个模型
> 槽、小脑接入方案);附录是差异对照与排障速查。

## 目录

- 第 0 章 · [三分钟版:先看一遍发生了什么](#第-0-章--三分钟版先看一遍发生了什么)
- 第 1 章 · [T0 启动与装配:从敲命令到"身体就绪"](#第-1-章--t0-启动与装配从敲命令到身体就绪)
- 第 2 章 · [T1 任务进来:意图解析与世界快照](#第-2-章--t1-任务进来意图解析与世界快照)
- 第 3 章 · [T2 规划:把人话编译成动作清单](#第-3-章--t2-规划把人话编译成动作清单)
- 第 4 章 · [T3 执行引擎装配:工具、护栏、执行器](#第-4-章--t3-执行引擎装配工具护栏执行器)
- 第 5 章 · [T4 清单执行循环:漂移检查与逐条派发](#第-5-章--t4-清单执行循环漂移检查与逐条派发)
- 第 6 章 · [T5 单步下行全解剖:从 dig 到 AGX 动起来](#第-6-章--t5-单步下行全解剖从-dig-到-agx-动起来)
- 第 7 章 · [T6 上行回流:观测怎么变成世界状态](#第-7-章--t6-上行回流观测怎么变成世界状态)
- 第 8 章 · [T7 收尾:结果、trace 与善后](#第-8-章--t7-收尾结果trace-与善后)
- 第 9 章 · [参考视图:三层角色、比喻与 core 的作用](#第-9-章--参考视图三层角色比喻与-core-的作用)
- 第 10 章 · [小脑(策略模型)接入方案](#第-10-章--小脑策略模型接入方案)
- 附录 A · [与本仓旧文档的差异](#附录-a与本仓旧文档的差异commit-c365bc0)
- 附录 B · [排障速查](#附录-b排障速查)

---

## 第 0 章 · 三分钟版:先看一遍发生了什么

你只说一句话:

```bash
python examples/run_task.py --config …agx_excavator.local.yaml \
  --query "把土堆的土挖一斗倒到右边3米"
```

1. **大模型交出一张清单**(全程它只干这一次活):

   ```json
   [
     {"op": "get_terrain", "params": {}, "bind": "t"},
     {"op": "dig", "params": {"dig_x_m": 3.0, "dig_y_m": 0.0,
                              "dump_x_m": 6.0, "dump_y_m": 0.0}},
     {"op": "home", "params": {}}
   ]
   ```

   注意它给的是**地面坐标(米)**,不是关节角度——大模型根本不知道挖掘机
   有 swing/boom 这些关节。

2. **质检员核对清单**(纯代码):动作名在词表里吗?参数名对吗?`dig` 要求
   "斗空",当前状态里确实是,通过。清单定稿,**到任务结束大模型不再被调用**
   (除非世界和清单对不上,才重叫一次)。

3. **dig 函数把"挖一个点"拆成 6 个关键帧**(顺序、角度全是代码算的):

   | 步骤 | swing(rad) | boom(m) | arm(m) | bucket(m) |
   | --- | --- | --- | --- | --- |
   | ① 对准挖点 atan2(0,3) | 0.0 | — | — | — |
   | ② 举臂就位 | 0.0 | 0.10 | -0.82 | -0.65 |
   | ③ 下铲 | 0.0 | -0.25 | 0.30 | -0.70 |
   | ④ 收斗(装土) | 0.0 | 0.02 | 0.70 | 0.10 |
   | ⑤ 摆到倒点 atan2(0,6) | 0.0 | — | — | — |
   | ⑥ 卸料 | 0.0 | 0.38 | -0.40 | -0.80 |

4. **每个关键帧 = 一行 JSON 发给桥接**(第③帧长这样,swing 弧度、其余米):

   ```json
   {"v":1,"cmd":"move_joints","targets":{"swing":0.0,"boom":-0.25,"arm":0.30,"bucket":-0.70},"timeout_s":30}
   ```

5. **桥接查一张启动时建好的名字表**(`swing`→CabinHinge、`boom`→
   ArmPrismatic1+2 双缸、`arm`→StickPrismatic、`bucket`→BucketPrismatic),
   对每个约束 `Lock1D.setPosition(目标)`,物理引擎把机器拉过去,桥接每步
   `getAngle()` 对比,**到位才回一行 JSON**。AGX 窗口里的挖掘机就动了。

6. 状态回流(铲斗颗粒质量 >50 kg 才算"有料")→ 工头执行下一条 → 全部跑完
   打印结果。实测:`get_terrain → dig(3,0 → 6,0) → home` 三步 ok,约 19 秒。

一句话:**大模型交清单(一次)→ 代码把清单拆成关节角并逐帧发送(其余全部)
→ 桥接查名字表设伺服 → AGX 动。** 下面各章把每一步展开到文件和函数级。

---

## 第 1 章 · T0 启动与装配:从敲命令到"身体就绪"

这一章覆盖:敲下回车之后、大模型被调用之前,程序做完的所有事。

### 1.1 命令行与配置读取(`examples/run_task.py:main`)

1. `clear_proxy_env()` **先于一切 import**——HTTP 代理环境变量会让 httpx 把
   本地 vLLM 的请求也路由去代理,这是血泪教训写成的第一行。
2. `argparse` 解析:`--config`(必填)、`--query`(任务,不在 config 里)、
   `--model/--api-key/--server-url`(临时换大模型)、`--mock/--stepagent`
   (强制慢路径)、`--workspace` 等。
3. `logging.basicConfig` 打到 INFO;`yaml.safe_load` 读入整份 config。
4. 校验任务非空:config 不内置任务,没给 `--query` 就退出(退出码 2)。
5. `--config` 顶层 `adapter: agx_excavator` 决定用哪个适配器。

### 1.2 import 触发的注册链(`src/jiuwen_agx/__init__.py`)

第一次 `import jiuwen_agx`(发生在 entry-point 发现时)依次完成六步,
**顺序不能换**,原因是 ActionSpec 构造期就要校验"能力是否已注册":

| 步骤 | 做了什么 | 在哪个文件 |
| --- | --- | --- |
| ① 注册能力 | `register_capability("motion.excavator")`、`("sensing.terrain")`;`register_capability_spec` 把能力↔动作↔驱动切片绑死 | `__init__.py` |
| ② 契约 | `DigResult/DigFailure/TerrainScan/PileInfo`(TypedDict,零依赖) | `contracts.py` |
| ③ 动作入词表 | `DIG`/`GET_TERRAIN` 两个 ActionSpec + `register_actions(...)` 推进 core 共享词表 | `actions.py` |
| ④ 底座 | `sim/` 五件套类定义(Env/Api/Driver/Config/Backend) | `sim/*` |
| ⑤ 薄包 | 挖掘机四件 + `@implements(DIG)` 绑定 | `agx_excavator/*` |
| ⑥ 技能 | `register_skill_dir(skills/)` 把 excavate 送进 fast 规划器目录 | `skills/excavate/SKILL.md` |

这条链跑完后,`jiuwensymbiosis-actions --config` 里就多出了 `dig` 和
`get_terrain`,core 仓库本身一行没改。

### 1.3 会话构建与五个对象的诞生

适配器解析的**缝隙契约没变**:`discover_adapter_builders()` 扫 entry-point 组
`jiuwensymbiosis.adapters`,拓展包装好后 `agx_excavator →
build_agx_excavator_session` 自动出现(`make_builder(Config, Env, Api)` 生成的
一行,`from_yaml` 即建五对象)。变化在调用方:上游合并引入 `runtime/` 子系统
后,**非 mock 路径的 run_task 经 `runtime.bindings.prepare_binding` 走**——它
校验并冻结配置(解析 typed config、算物理设备/侧车的资源键、workspace 指纹,
产出 `BindingSnapshot`),适配器发现仍走上面同一条 entry-point 缝;mock 路径
保留旧的 `_build_session`。以下为构建出的五对象:

```text
AgxExcavatorConfig   joint_names=(swing,boom,arm,bucket)、has_base=True、
                     base_step_limits=(1.0m, 0.7rad)、joint_limits/home_joints、
                     reach_min/max_m、swing_unit、dig_cycle_tuning(接真机时填
                     探针实测的混合单位值)
AgxExcavatorEnv      类能力超集 ∪ {motion.excavator};实例能力按 config 收窄:
                     motion.joint(恒开)+ motion.base(has_base)
                     + sensing.terrain(terrain_enabled)+ vision.*(camera_enabled)
AgxExcavatorApi      SimMachineApi(通用动作)+ @implements(DIG)
SimMachineDriver     持有一个 SimBackend;所有命令先过它的校验再进后端
SimBackend           按 cfg.backend 三选一:mock / remote(生产) / inprocess(骨架)
```

### 1.4 连接:`with session:` 里发生了什么

`RobotSession.__enter__` → `AgxExcavatorEnv.connect()`(幂等)→
`SimMachineDriver(cfg)` → `create_backend(cfg)`。以 `backend: remote` 为例,
`RemoteSimBackend.open()` 依次:

1. `socket.create_connection((host, 9700), timeout=startup_timeout_s=60)`;
2. 发一行 `{"v":1,"cmd":"ping"}`,校验对端协议版本必须是 v1,不符即断开报错;
3. **对端身份自报**:发 `{"cmd":"inventory"}`,把对端机器打进日志——
   `excavator365` = 真 AGX;`demo_excavator` = 内存假机,打 WARNING
   ("命令会被成功应答,但不会有任何物理动作")。这一步直接消灭
   "命令成功画面不动"的头号误会。

### 1.5 模型配置的读取(两条消费路径,一个事实来源)

`_build_model_spec()`:YAML `model:` 块 → `ModelSpec`,覆盖优先级
`--api-key > $OPENJIUWEN_API_KEY > YAML > 类默认`(默认指向本地 vLLM 的
Qwen3-VL-32B @ 127.0.0.1:8110)。**一个事实来源,两条消费路径**:慢路径经
`build_model()` 包成 openjiuwen Model;快路径规划器直接拿字段发 HTTP。运行时
看启动日志 `model : deepseek-ai/DeepSeek-V3.2 @ <端点>` 确认。

### 1.6 执行模式的决定

`exec_mode = "stepagent" if (--mock or --stepagent) else config 的 exec_mode`。
挖掘机 config 写死 `fastagent`(编译一次、执行零 LLM);`agent:` 块同时声明
`enable_skill: true`(挂 RobotControlTool + SkillUseRail)、
`intent_timeout_s`(LLM① 超时,默认 60s)。随后 `with session:` 内
`run_robot_task(session, query, agent_cfg, conversation_id="task-xxxxxxxx")`。

**T0 结束时的状态**:身体已连上(或 mock 就绪)、词表已注入、护栏还没挂
(挂在 T3)、大模型一次都没被调用。

---

## 第 2 章 · T1 任务进来:意图解析与世界快照

### 2.1 分发

`run_robot_task` 看 `exec_mode == "fastagent"` → `run_fast_task(...)`。

### 2.2 LLM① `parse_task`:把人话解析成结构化意图

- 输入:query 原文;输出:`{targets: [ 土堆 ], references: [...], grounding: {...}}`
  (空间限定词,如"左边的箱子"会记成 `box ← in ← 左边`)。
- 走 `spec.api_base/api_key/model_name` 直接 POST `/chat/completions`,
  读超时 = `agent.intent_timeout_s`(默认 60s,推理型模型建议 90~120)。
- **失败即降级**:解析挂了只打 warning,继续"盲编译"——规划器照样能从
  动作契约推序列,只是少了场景提示。

### 2.3 `_perceive_scene`:有相机才扫,没有就跳过

预感知复用常驻检测器(analyze_scene),前提是 `vision.detection` 在能力集里。
挖掘机 `camera_enabled: false` → 直接返回 None(盲编译)。真值从哪来?下一节。

### 2.4 `WorldState.snapshot`:世界快照逐字段

规划前拍一张快照,**全部经桥接现读,不用缓存**:

| 字段 | 来源 | 挖掘机上的实际值(真机) |
| --- | --- | --- |
| `tokens` | `env.holding_payload` → 桥接 `{"cmd":"scoop"}` → 铲斗颗粒质量 >50 kg? | `payload.clear`(斗空) |
| `joints` | `{"cmd":"joints"}` → 各约束 `getAngle()` | 四个原生单位值(rad/m),**带关节名** |
| `extra.terrain` | `{"cmd":"terrain"}` | `[{name:"soil_field", x_m:3.0, y_m:0.0, volume_m3:1.0}]`(现状:代表点,不随挖掘变) |
| `joint_units` | config | None(混合单位)+ `swing_unit: "rad"` |
| `base_step_limits` | config | (1.0 m, 0.7 rad)——**会进 prompt,规划期就知道包络** |
| `capabilities` | env | `["motion.base","motion.excavator","motion.joint","sensing.terrain"]` |
| `locations` | ExecutionMemory 账本 | 空(还没感知过任何位置) |

---

## 第 3 章 · T2 规划:把人话编译成动作清单

### 3.1 先备料(全部纯代码)

- **动作索引** `_build_action_index(api, planner_only=True)`:扫 Api 的 MRO,
  收集所有 `@implements` 方法,按 `api ∩ env` 门控。挖掘机的可用动作:
  `dig、get_terrain、move_joint、get_joint_positions、navigate_relative、
  rotate_base、drive_arc、home`。这个索引同时是**给大模型看的菜单**和
  **校验器的白名单**——一个东西,两处使用。
- **参数签名** `_action_param_sig`:从函数签名渲染 `dig(dig_x_m, dig_y_m,
  dump_x_m, dump_y_m)`,保证大模型用的参数名和实现一字不差。
- **技能库** `DEFAULT_REGISTRY.skills_markdown()`:全部 SKILL.md,先按能力
  过滤——没有 `motion.excavator` 的机体根本看不到 excavate 技能的正文。

### 3.2 Tier1 编译:一次推理同时完成"选技能 + 展开工作流"

发给大模型的 prompt 由这些块拼成(每块都来自真实数据):

```text
任务:把土堆的土挖一斗倒到右边3米
【当前状态】状态:payload.clear / 关节(rad|米):… / 工作范围:每命令位移≤1.0m、
            转角≤0.7rad(超出会被安全护栏拒绝,请拆成多次)   ← base_step_limits
【能力】motion.base, motion.excavator, motion.joint, sensing.terrain
【可用技能】excavate SKILL.md 正文(读地形→就位→逐斗 dig→复查→收工)
【可用动作】dig(dig_x_m, dig_y_m, dump_x_m, dump_y_m) -> {ok, volume_m3, cycle_s}
            requires: payload.clear;invalidates_locations: true
            get_terrain() -> {ok, piles:[{name,x_m,y_m,volume_m3}]}
            move_joint(targets) / navigate_relative(dx_m,dy_m,dyaw_rad) / …
【特殊动作】无(挖掘机没有 track_detect/track_grasp)
请输出展开后的动作序列 JSON 数组。
```

一次推理吐出第 0 章那张 JSON 数组。**"转一圈会被拆成 9×rotate_base(0.7)"
就发生在这一步**——包络写进了 prompt,规划期就不会规划出越界命令,而不是
等护栏一步一拦。

### 3.3 校验回路:`_compile_loop` 与 `parse_sequence`

大模型吐的数组立刻过 `parse_sequence`,不合格**带着原因打回重写,至多
4 次**。校验逐条是:

1. 每个动作名都在动作索引里(白名单,不是 prompt 里求它别乱写);
2. 参数名 ⊆ 该动作声明的参数(防"target" vs "box" 式的编造);
3. 状态向前模拟:每步 `requires ⊆ 当前状态`(`dig` 要求 `payload.clear`;
   `provides` 往状态里加,`invalidates` 往外删);
4. `<bind>.field` 引用必须由前面某动作的 `returns` 字段真实产出
   (`t.piles[0].x_m` 必须匹配 `get_terrain` 的 TerrainScan 契约);
5. 声明 `consumes_location` 的动作不能在"所有感知已被作废"的状态下运行;
6. 任务自带的空间限定词不能被悄悄丢掉("苹果在抽屉里"不能编译成
   "抓第一个检测到的苹果")。

它**拒绝前置条件不满足的序列,但接受任何合法排序**——顺序是大模型在
契约约束下自己推导的。

### 3.4 Tier2 何时接管

三个**可判定**条件之一(绝不是"模型觉得自己行"):没有技能通过能力过滤 /
模型显式返回空数组 / 重试 4 次仍不合法。Tier2 `compose_actions` 只拿动作
契约(requires/provides/位置新鲜度)推导序列,完全相同的校验。产出
`PlanResult{sequence, tier:"skill"|"action", reason}`——`tier="action"`
的成功序列是新技能的候选蒸馏对象(蒸馏器尚未实现,见 TODO)。

---

## 第 4 章 · T3 执行引擎装配:工具、护栏、执行器

`run_fast_task` 调 `build_robot_agent(session, config)`——和慢路径
**同一个构造函数**,这就是"快慢路径永远不漂"的根:

1. **工具**:`build_robot_tools(api, env, planner_only=True)` 扫 MRO,把每个
   `@implements` 方法包成 openjiuwen `LocalFunction`(外面包一层
   `_recording`,让每次调用的结果流进 ExecutionMemory);`enable_skill`
   再追加 `RobotControlTool`(单入口分发器)。
2. **护栏按能力装配**(`_RailRegistry` 逐条判断):

   | 护栏 | 条件 | 挖掘机 |
   | --- | --- | --- |
   | VisualFeedbackRail | 需要 `vision.camera` | ✘(没相机) |
   | SafetyRail | 任一 motion.* 能力 | ✔(joint + base) |
   | RecoveryRail | 任一 motion/grasp 能力 | ✔(joint) |
   | SkillUseRail | `enable_skill` | ✔ |
   | TraceRail | `enable_tracing`(默认关) | ✘ |

3. **快路径手工初始化** `_prime_fast_agent`:快路径不走 `agent.invoke()`,
   护栏的惰性注册由 `asyncio.run(agent.ensure_initialized())` 手工完成——
   之后每一步动作都会真的穿过护栏。
4. **执行器** `build_ability_executor(agent)`:拿到 `agent.ability_manager`,
   把 `(op, params)` 包成假 ToolCall 送进和 LLM 慢路径同一条分发管线。

---

## 第 5 章 · T4 清单执行循环:漂移检查与逐条派发

`run_sequence(session, steps, executor=…, replan=…)` 主循环:

```text
pending = [get_terrain, dig{3,0→6,0}, home]
while pending:
    step = pending[0]
    ① drift = _drift(step, metas, session, env)
         两份材料:
         a. 自状态:该步 requires(payload.clear)vs 实测 tokens
            —— 只认"实测矛盾","不知道"不算(机体报不了 ≠ 不满足)
         b. 位置新鲜度:该步要用的感知是否已被 invalidates_locations 作废
         矛盾 且 重规划次数 < 2 → ② 重规划;否则记 warning,照常派发
    ② 重规划(唯一会再次调 LLM 的地方):
         重新预感知 + WorldState 重拍 →
         plan_task(f"{query}\n\n(执行中断:{why}。请依据【当前状态】重新规划)")
         → 新序列 parse_sequence 校验 → 替换 pending、清空旧绑定(防旧坐标)
    ③ resolve_params:把 <bind>.field 换成真实值(t.piles[0].x_m → 3.0)
    ④ run_op(op, params) → 第 6 章的单步下行
    ⑤ 记录 out;失败 → 结构化 reason + error_code;RecoveryRail 没接管的
       补 _safe_retreat(先查 holding_payload,斗有料不盲目放料)
    任意一步失败即停(不是静默继续)
```

---

## 第 6 章 · T5 单步下行全解剖:从 dig 到 AGX 动起来

以 `run_op("dig", {dig_x_m:3.0, …})` 为例,一次完整下行有 **7 层**:

### 6.1 伪装成 LLM 的 ToolCall(`agent/fast/ability_exec.py`)

```python
ToolCall(id="fast-dig", name="robot_control",
         arguments='{"action":"dig","params":{"dig_x_m":3.0,…}}')
```

形状和慢路径里大模型吐的完全一样——护栏与分发器对"这是谁发的"无感知,
快慢路径因此共享全部安全设施。交给 `agent.ability_manager.execute(ctx, [tc])`。

### 6.2 SafetyRail:逐项检查与"skip 契约"真拦截

`before_tool_call` 收到的 `tool_args` 是 **JSON 字符串**(先 `_coerce_tool_args`
解析,否则所有运动参数检查会静默空转);识别 `robot_control` 后解包出真正的
action/params;监视表按能力派生(`motion.joint → move_joint/move_named_joint`,
`motion.base → navigate_relative/rotate_base/drive_arc`,另有三个历史常驻)。
逐项:

- `move_joint`:每个关节名已知?值有限?在 `env.joint_limits`(config 的
  混合单位实测值)内?——每项失败各有专属报错文案;
- `navigate_relative/rotate_base/drive_arc`:位移 ≤1.0 m?转角 ≤0.7 rad?

**拒绝走 skip 契约**(不是 raise!框架会捕获回调异常然后照常执行工具——这是
修过的真 bug):置 `ctx.extra["_skip_tool"]=True`,填好失败的
`ToolOutput(error=…, data={"error_code":"safety_rejected"})`。效果:这一次
调用被真正跳过、LLM/运行器看到工具错误、步骤失败。raise 只作为没有框架
上下文时的兜底。

### 6.3 RobotControlTool 分发与记账

护栏放行后,分发器从动作索引找到 `AgxExcavatorApi.dig` 并调用;调用返回后
`record_action` 写 ExecutionMemory:

```text
dig 的契约生效:
  provides = payload.clear     → 自状态记"斗空了"
  invalidates = body.home      → "不在家"了
  invalidates_locations = True → 感知缓存清空(挖掘重塑了地形,旧坐标全作废)
```

### 6.4 dig 的领域校验(返回的是数据,不是异常)

`work.execute_dig_cycle(driver, dig_x_m=3.0, …)` 先做四查,任一失败
**返回 `DigFailure{ok:false, error:"人类可读的原因"}`**(大模型/重规划读得懂,
不是崩溃):

1. 两点坐标有限?
2. 两点半径 ∈ 可达环 [reach_min_m, reach_max_m]=1~6 m?超出会提示
   "navigate_relative the undercarriage closer, then dig";
3. 四个关节 (swing/boom/arm/bucket) 都在?
4. `driver.scoop_state()` 为真(斗里 >50 kg)?——上一斗没倒,先 home。

### 6.5 六个关键帧逐帧执行(真实值,单位混合:rad/m)

```text
① swing=0.0                对准挖点(atan2(0,3);挖 (2,2) 则为 0.785)
② boom=0.10, arm=-0.82, bucket=-0.65        举臂就位(ready_*)
③ boom=-0.25, arm=0.30, bucket=-0.70        下铲(dig_*)
④ boom=0.02, arm=0.70, bucket=0.10         收斗(curl_*);mark_scoop(True)
⑤ swing=0.0                摆到倒点(atan2(0,6))
⑥ boom=0.38, arm=-0.40, bucket=-0.80        卸料(dump_*);mark_scoop(False)
```

每帧调 `driver.move_joints_blocking(targets)`,**阻塞到 AGX 报到位才走下一
帧**。注意 AGX 真值模式下 `mark_scoop` 只是协议兼容写口——"有料"以桥接
实测的铲斗颗粒质量为准,不信任记账。

### 6.6 driver 校验与 JSON 下发

`SimMachineDriver.move_joints_blocking` 二次校验(护栏之外的第二道):
未知关节名拒绝(报错会列出全部已知名)、非有限值拒绝、超配置限位拒绝——
**关键帧与限位单位不一致(把米当度发)就在这里被拦**,报错指明是哪个关节
超了什么范围。通过后 `RemoteSimBackend.send_joint_targets` 拼一行 JSON:

```json
{"v":1,"cmd":"move_joints","targets":{"swing":0.0,"boom":-0.25,"arm":0.30,"bucket":-0.70},"timeout_s":30}
```

发前 `read_timeout_s = timeout_s + 5`;若桥接回 `async:true`(viewer 泵模式,
不能阻塞渲染主线程),客户端每 50ms 发 `{"cmd":"joints"}` 轮询,直到所有目标
关节差 <0.01(按该关节全行程的 1% 计),或超时拿回实况——**协议承诺:阻塞
调用返回的是"运动后的真实状态",没到位不报错,状态自己看**。

### 6.7 桥接内部:名字表是怎么建起来的(`agx_bridge_server.py`)

桥接进程(跑在装 AGX 的机器上)启动时把场景搭好/加载好:

1. **场景**:`--excavator` 程序化搭官方 365 + 沙地(50×50 m 高度场 +
   agxTerrain,切割边 Shovel 绑到铲斗);`--scene x.agx` 加载任意模型;
   `--demo` 起内存假机(零 AGX 联调协议用)。
2. **名字表** `_auto_discover`:遍历场景约束,SWIG 基类包装的用
   `asHinge()/asPrismatic()` 下转型(Lock 类自动跳过——它们不可驱动),
   按官方模型属性对上词表名;`--joint-map swing=CabinHinge,…` 逐项覆盖。
   一个词表名可对应**一组**约束(boom = 双缸,整组同步,与官方键盘一致)。
3. **泵与守卫**:绑定 :9700 前查端口占用(拒绝重复监听——Windows 允许重复
   bind,连到哪个实例是不确定的);viewer 模式挂 `StepEventCallback.pre`
   非阻塞泵(agxViewer 会冻结后台线程,所以不做线程服务),每仿真步收发
   一次;仿真被暂停时打 WARNING(泵挂在不步进的回调上,桥接会"失聪")。
4. **键盘**:`JIUWEN_KEYBOARD=1` 时把官方 365 键位原样打开(数据采集用)。

### 6.8 伺服与收敛:命令如何变成物理动作

`set_joint_targets`:每个目标关节 → 该组每个约束:`Motor1D` 关、`Lock1D`
开,阻尼按"距离越远阻尼越大"的自适应公式(`agx.logInterpolate`,下限
2/60,力范围继承电机)——官方 JointController 的模式。然后:

- **headless 模式**:桥接自己推物理,每步 1/60 s,每步对比
  `|目标 − getAngle()| < max(1e-3, 行程×1%)`,收敛或超时才回话;
- **viewer(pump)模式**:设完目标立即回 `async:true + eta_s`,物理由
  agxViewer 的渲染循环推进,客户端轮询(6.6 节)。

回话内容:`{"v":1,"ok":true,"joints":{…六个帧走完后整机的真实角度…}}`。

### 6.9 行走不走关节伺服:底盘闭环专述

`navigate_relative/rotate_base/drive_arc` 不设 Lock1D,而是:

1. 桥接读 `base_pose()`——底盘刚体的世界 x/y/yaw(欧拉角 z 分量,方向约定
   经实测校准:左履带 +v 右履带 −v 时 yaw 减小,0.22 rad/s);
2. `base_track_command(pose, goal)` 纯函数:先对齐 yaw 再平移,速度正比于
   误差、带下限防土壤卡死——**每仿真步重算,直到真实位姿达标**(实测
   9×rotate_base(0.7) = 353°;旧版"转速×时间"开环曾转出 145° 的反方向);
3. headless 由 `_drive_steps` 阻塞走完;viewer 模式挂 `_drive_goal` 由泵逐步
   执行,回复带 `async+eta_s`,**客户端必须轮询 `{"cmd":"base_state"}` 到
   idle 再发下一条**——连发的相对位移会互相覆盖停车时刻(实测 9 条只走 1
   条)。`RemoteSimBackend` 已内置该轮询;自己写脚本直连桥接时必须照做。

---

## 第 7 章 · T6 上行回流:观测怎么变成世界状态

```text
AGX 真值                     桥接命令            去向
──────────────────────────────────────────────────────────────────
约束角度 getAngle()          {"cmd":"joints"}    观测.joints(带名字)/到位判断
铲斗颗粒质量 bucket_mass_kg  {"cmd":"scoop"}     payload.held/clear token
                             (阈值 50kg;空斗贴地有 0~1kg 本底噪声)
底盘真实位姿 base_pose       {"cmd":"base_pose"} 底盘闭环;框架侧尚未接入观测(见 TODO B1)
地形料堆 read_terrain        {"cmd":"terrain"}   get_terrain 动作
                             (现状:沙地场景报代表点 soil_field(3,0),不随挖掘变——B2)
全场景清单 inventory          {"cmd":"inventory"} 探针/身份自标/阈值校准/关节单位
        │
        ▼  每步执行后 & 每步漂移检查前
WorldState.snapshot(token / 关节 / 包络)  +  ExecutionMemory
(provides/invalidates/位置新鲜度:dig 一挖,旧料堆坐标全部作废)
        │
        ▼
下一步的 _drift(回到第 5 章①)→ 矛盾才重规划,不知道≠否
```

SKILL.md 教的"挖完复查地形"就落在这一环:`get_terrain` 再读一次真值,据此
继续/换堆/收工。当前地形体积是静态的(B2),复查实际有效的是铲斗质量真值;
B2 落地后复查才真正闭环。

---

## 第 8 章 · T7 收尾:结果、trace 与善后

- **结果字典**:`{ok, steps_done, steps:[{i,op,ok,result|reason,error_code}],
  env_keys, sequence(原始 JSON 清单), plan_tier, plan_skills}`。`tier=action`
  的成功序列是新技能候选(蒸馏未实现)。run_task 以 JSON 打印,退出码 0。
- **trace**:`enable_tracing` 开时,TraceRail 在 BEFORE/AFTER_INVOKE 间记录
  每步(输入/输出摘要/成功/时长/观测快照/帧),落 `<workspace>/traces/
  <会话>_<时间>_<pid>.json`,帧存 `frames/<run_token>/`;`jiuwensymbiosis-replay
  <trace.json>` 回放时间线。**这也是小脑模型的演示数据来源(第 10 章)**。
- **善后**:`with session:` 退出 → disconnect(幂等);中途失败时
  RecoveryRail 或 `_safe_retreat` 已按"斗有料不盲放"的原则回到安全态。

---

## 第 9 章 · 参考视图:三层角色、比喻与 core 的作用

### 9.1 三层与比喻

```text
┌────────────────────────────────────────────────────────────────────────┐
│ jiuwensymbiosis core(通用,不知道 AGX 存在)                          │
│  词表 ActionSpec · 规划器 plan_task/parse_sequence · 护栏栈           │
│  状态账本 ExecutionMemory · 技能目录 · trace                          │
│  它只认:动作名 / 参数名 / 状态 token / 限位数值(单位由机体声明)    │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ 四个接口:register_capability /
                               │ register_actions / entry-points /
                               │ register_skill_dir
┌──────────────────────────────▼─────────────────────────────────────────┐
│ jiuwen_agx 拓展包(全部 AGX 知识都在这一层)                           │
│  连接(RemoteSimBackend⇄桥接)· 翻译(名字表)· 真值(质量/地形)  │
│  工艺(dig 关键帧,约 100 行)                                        │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │ agx Python API(仅桥接进程内)
┌──────────────────────────────▼─────────────────────────────────────────┐
│ AGX(物理 + 渲染;license 门控:无 license 命令通但执行器不动)      │
└────────────────────────────────────────────────────────────────────────┘
```

| 现实角色 | 系统里的谁 | 它只干什么 |
| --- | --- | --- |
| 老板 | 你 | 说一句话(`--query`) |
| 秘书 | 大模型(`model:` 块) | 把人话写成带坐标的动作清单(第 3 章) |
| 质检员 | `parse_sequence` | 核对清单能不能执行,不行带原因打回 |
| 工头 | `run_sequence` | 拿着清单逐条干,失败善后(第 5 章) |
| 熟练司机 | `work.py` 的 dig 等 | 把"挖那个点"翻译成 6 个关键帧(第 6 章) |
| 安全员 | SafetyRail | 执行前查限位,越界真拦下(skip 契约) |
| 对讲机+翻译 | 桥接 `agx_bridge_server.py` | 查名字表、设伺服、把真实角度报回来 |
| 挖掘机本体 | AGX 物理场景 | 真正动的东西(含键盘直控) |

### 9.2 接第二台机器:哪层复用、哪层要写

| 层 | 状态 | 第二台机器要做什么 |
| --- | --- | --- |
| SimBackend / Driver / Env / Api / Config | 复用 | 不写 |
| 桥接服务 + 协议 v1 | 复用 | 一台桥接服务所有机器(合用 `--joints`/`--joint-map`) |
| 能力 + ActionSpec + 契约 | 要写 | 机器族能力 + TypedDict + spec(约 30 行) |
| 薄包 config/work/api/env/session | 要写 | 该机参数、工作循环、绑定(约 150 行) |
| 技能 SKILL.md | 要写 | 该机工作流 |

配方与验收标准见 [add-sim-machine.md](add-sim-machine.md)。

### 9.3 四个模型槽

| 槽 | 装什么 | 配置在哪 | 干什么 | 现状 |
| --- | --- | --- | --- | --- |
| 大脑 LLM/VLM | DeepSeek、Qwen3-VL、GPT… | YAML `model:` 块(CLI > 环境变量 > YAML > 默认) | 解析任务、编译序列、失败重规划 | ✅ 在用 |
| 眼睛 检测 | GroundingDINO + SAM2 | 感知 sidecar(`make_detector_sidecar`),不在 `model:` 块 | 相机帧→框/mask/3D | ✅ 在用(有相机的机体;挖掘机未开) |
| 嘴耳 语音 | ASR(funasr 类)+ TTS(ChatTTS) | YAML `voice:` 块 + `--voice` | 唤醒→转写→指令;结果播报 | 可选 |
| 小脑 策略 | ACT / Diffusion Policy / VLA… | 第 10 章的 `@register_policy` 缝 | 观测→关节动作块 | ⛔ 待建 |

---

## 第 10 章 · 小脑(策略模型)接入方案

### 10.1 定位:插在动作契约之下,不替换规划

LLM 决定"用哪个动作、参数是什么";小脑决定"这个动作具体怎么动"。

```text
  LLM 规划层(不变)→ 动作词表层 ──► 护栏(不变)──► Policy 接缝 ★新
                                        │
                        ┌───────────────┴───────────────┐
                        │ dig    = 手写关键帧(零号策略) │
                        │ act_exec = 神经网络(要加的)   │
                        └───────────────┬───────────────┘
                                        ▼
                      SimBackend → TCP → 名字表 → Lock1D → AGX
```

### 10.2 act_exec 运行时循环(与 dig 逐行对照)

```python
# dig(今天):                          # act_exec(要加的):
def execute_dig_cycle(driver, ...):    def execute_act(driver, policy, ...):
    swing = atan2(y, x)                    while not done:
    driver.move_joints_blocking(...)           obs = {"joints": driver.get_joint_positions(),
    …6 个关键帧…                                      # "image": driver.grab_frames(), ←路线二
                                               }
                                               chunk = policy.predict(obs)  # 网络本地推理
                                               for target in chunk:         # 逐拍下发
                                                   driver.move_joints_blocking(target)
                                               done = 成功判据(铲斗质量/超时)
```

网络两端的产物都没有神秘之处——观测就是 `{"cmd":"joints"}` 的返回(加可选
画面),动作块就是一串"下一拍关节目标",形状与第 0 章关键帧表的每一行
一模一样:

```python
[{"swing":0.00,"boom":0.10,"arm":-0.82,"bucket":-0.65},   # 下一拍
 {"swing":0.01,"boom":0.09,"arm":-0.80,"bucket":-0.63},   # 下下拍
 … 共 chunk_size 拍;逐拍喂 move_joints_blocking,发完再取观测、再推理]
```

**下发走的就是 `move_joints_blocking` 同一条咽喉**:driver 校验、SafetyRail、
协议、名字表对策略输出自动生效,零额外安全代码。

### 10.3 接缝与词表扩展(照抄 dig 的注册链,四步)

```python
# ① 契约 contracts.py:ActResult | ActFailure(与 DigResult 同形)
# ② __init__.py 注册链最前面(顺序敏感!):
register_capability("policy.act")
register_capability_spec("policy.act", actions=["act_exec"])
# ③ actions.py:
ACT_EXECUTE = ActionSpec(
    name="act_exec", capability="policy.act",
    params=("task",),                     # 或 object_name / goal 描述
    requires=("payload.clear",), invalidates_locations=True,
    result=ActResult | ActFailure,
)
register_actions(ACT_EXECUTE)
# ④ 薄包 api.py:@implements(ACT_EXECUTE) 的 act_exec,内部跑 10.2 循环

# Policy 接缝(新模块 jiuwen_agx/policy.py,仿 sim/backend.py 的注册表):
POLICIES: dict[str, type] = {}
def register_policy(name): ...                     # @register_policy("act")
class Policy(Protocol):
    def reset(self) -> None: ...
    def predict(self, obs: dict) -> list[dict[str, float]]: ...  # 一个动作块
```

配一个 SKILL.md(`capabilities: [policy.act]`,正文写"何时用 act_exec 何时用
dig"),能力过滤自动把它从不具备的机体上摘掉。**换模型(ACT → Diffusion
Policy → VLA)= 换一个 `@register_policy` 类**,词表、护栏、协议、LLM 全不动。
插不上的模型也说清楚:输出末端位姿的(挖掘机无逆运动学,变不成四关节角)、
输出吸盘/夹爪动作的(机体没有这些执行器)——预训练于机械臂的模型不能拿来
就用,必须自采数据训练。

### 10.4 两条路线与数据闭环

| 路线 | 观测 | 前提 | 结论 |
| --- | --- | --- | --- |
| **一:state-only** | 只吃关节角 | 零(桥接命令都在) | **先做,今天就能通全链** |
| 二:视觉版 | 关节角 + RGB | `grab_frame` 是占位 `return None`,需 AGX 侧实现相机出帧 | 路线一跑通后再做 |

数据闭环(采集机框架已备好):

```text
JIUWEN_KEYBOARD=1 键盘演示(官方 365 键位)+ enable_tracing trace 落盘
   │ 按固定频率记 (观测, 关节角) 对;inventory.bucket_mass_kg 做硬成功判据
   ▼
离线训练(lerobot 的 ACT/Diffusion Policy 实现与训练脚本;chunk_size 取 20~100)
   ▼
ckpt(.pt)→ policy 类加载 → act_exec 上线(先 mock 验收,再上真桥接)
```

### 10.5 分四步实施(每步有验收标准)

| 步骤 | 内容 | 验收 |
| --- | --- | --- |
| ① 接缝 + 假策略 | policy.py 注册表、契约、act_exec 绑定、SKILL.md(约 80 行);mock 后端 + 随机吐合法关节角的假 policy | mock 下全链跑通:能力出现、护栏生效、ActFailure 能被规划器读懂 |
| ② 采数据 | 键盘演示 + trace 记录(画面可后补) | 采够 N 条完整"抓起→放好"演示,帧率/单位一致 |
| ③ 真 policy(state-only) | ckpt 接入 `@register_policy("act")`,真桥接 | 真机完成一次任务;护栏/失败路径与 dig 一致 |
| ④ 视觉版 + 协议增强 | 桥接实现 `grab_frame`;可选 `{"cmd":"move_chunk"}` 批量下发(消除 20 拍=20 往返) | 视觉 ACT 上线;chunk 单往返 |

总预估约 250 行 + 1 个 SKILL.md,**全部在拓展包内,core 零改动**。

---

## 附录 A:与本仓旧文档的差异(commit c365bc0)

| 旧说法(已过时) | 现状 |
| --- | --- |
| SafetyRail 拒绝 = raise ValueError(框架捕获后照常执行工具) | skip 契约真拦截:调用被跳过、步骤失败、`safety_rejected` 错误码 |
| 底盘 = 开环"转速×时间"(转向率/方向靠估,曾转出反方向) | 闭环:`base_pose` 真实位姿 + `base_track_command` 每步纠偏,到位才停(实测 9×0.7rad=353°) |
| pump 模式底盘命令发完即返(连发互相覆盖) | 回 `async+eta_s`,客户端轮询 `base_state` 等到位 |
| 铲斗有料 = 质量 >1 kg | 阈值 **50 kg**(空斗贴地有 0~1 kg 本底),`inventory.bucket_mass_kg` 可校准 |
| Linux noVNC 直播三件套 | 已删;Windows 三件套:`start_agx_bridge.bat` / `start_demo_bridge.ps1` / `monitor_ui.py`(:8050) |
| 规划器不知道底盘包络 | `base_step_limits` 进 WorldState prompt,"转一圈"规划期即拆步 |
| 意图解析固定超时 | `agent.intent_timeout_s` 可配,默认 60s |
| 客户端不知道对端是谁 | `inventory` 自报:`demo_excavator` 打 WARNING,`excavator365` 打 INFO |
| 会话构建直接 `_build_session` | 上游合并(c365bc0→9571058)后:非 mock 路径经 `runtime.bindings.prepare_binding`(校验/冻结成 BindingSnapshot);GUI 工作台拆为独立包 `jiuwensymbiosis_gui`;entry-point 拓展缝隙不变,jiuwen_agx 113 项测试在新代码上全绿 |

## 附录 B:排障速查(详见 README 排障表)

| 症状 | 原因与处理 |
| --- | --- |
| 命令成功、画面不动 | 连到 demo 假机:看连接日志的对端机器名 |
| 转一圈只转一点/方向反 | 已是闭环;仍偏差看桥接日志 `base motion TIMEOUT`(被土挡) |
| `dig requires payload.clear but state is payload.held` | 载料误判:用 inventory 的 bucket_mass_kg 校准阈值 |
| planner attempt N/3 read timed out | 意图解析超时:调大 `agent.intent_timeout_s`(90~120) |
| 探针/任务超时、桥接像死了 | AGX 窗口被暂停:窗口内按空格恢复 |
| 端口 9700 被占用 | 旧 AGX 窗口没关;或 `JIUWEN_BRIDGE_PORT` 换端口(启动器带守卫) |
| `joint 'X' target … outside configured limits` | 关键帧/限位单位不一致(回转 rad、液压缸 m):`--check-config` 逐条核对 |
| agxViewer 加载插件报 `No module named 'agxPythonModules'` | agxViewer 用用户配置重建环境丢了 PATH;用 `start_agx_bridge.bat` |
