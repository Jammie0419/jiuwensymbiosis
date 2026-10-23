# 记录和回放规划级轨迹

> 类别：How-to。

执行轨迹（`TraceRail`）回答“每个工具调用做了什么”。当你要回答的是
“**为什么这么规划、计划怎么被修正、最终结果如何**”时，需要规划级 trace：
每个 episode 一份 JSONL，记录任务、意图、世界状态、每次规划尝试、每步结果、
重规划和最终结果。字段与格式见[规划级轨迹参考](../reference/episode-trace.md)。

本版覆盖 `exec_mode="fastagent"`；慢路径（`stepagent`）暂不产生 plan 事件。

## 一、开启

在任务 YAML 的 `agent:` 块里加两个开关（可以只开一个）：

```yaml
agent:
  exec_mode: fastagent
  # 规划级：episodes/*.jsonl
  enable_episode_trace: true
  # 工具级：traces/*.json（可选；和上一行同时开才能互相引用）
  enable_tracing: true
  # episode_dir: ./episodes    # 覆盖目录（默认 <workspace>/episodes）
```

代码方式等价：

```python
config = RobotAgentConfig(
    exec_mode="fastagent",
    enable_episode_trace=True,
    enable_tracing=True,
)
```

两个开关互相独立：

| | `enable_episode_trace=True` | `enable_episode_trace=False` |
|---|---|---|
| `enable_tracing=True` | 规划级 + 工具级，`outcome.trace_path` 可下钻 | 只有工具级（现状） |
| `enable_tracing=False` | 只有规划级，`trace_path=null` | 零开销（默认） |

## 二、文件在哪

默认 `<workspace>/episodes/`，与 `traces/` 平级。workspace 解析顺序：

```text
显式 config.workspace
  > $JIUWENSYMBIOSIS_WORKSPACE
  > ~/.jiuwensymbiosis/settings.json 的 "workspace"
  > ~/.jiuwensymbiosis/{session.name}_workspace/
```

文件名：`{conversation_id}_{时间戳}_{微秒}_{pid}.jsonl`。`run_fast_task` 的返回字典
多一个 `episode_path` 字段，直接指向本次文件；关闭时它是 `None`。

## 三、看一次运行

```bash
jiuwensymbiosis-episode ~/.jiuwensymbiosis/agx_excavator_workspace/episodes/task-xxx.jsonl
```

输出按时间顺序：

```text
=== Episode: task-8f3a1c2d_....jsonl ===
query: 把土堆挖一斗倒到旁边
robot=agx_excavator  mode=fastagent
outcome: ok=True steps=8 replans=1

[intent] ok  targets=['soil_pile_215f']
[world] tokens=['payload.clear']  capabilities=['motion.excavator']
[plan #1] ok  trigger=initial  tier=skill  skills=['excavate']  steps=3
  [s0] ok   get_terrain  0.02s
  [s1] FAIL dig  0.01s
         error: ValueError: dig point at 6.59 m is outside the reachable annulus ...
[replan @s1] ok  trigger=business_failure  budget=1/2
         why: dig 失败：...
         new: ['navigate_relative', 'get_terrain', 'dig', 'home']
[plan #2] ok  trigger=replan  tier=skill  skills=['excavate']  steps=4
  [s5] ok   dig  48.00s
[outcome] ok=True  steps=8  replans=1  attempts=2  final_tier=skill
         tool trace: /w/traces/task-8f3a1c2d_....json
```

每一段怎么读：

- `[plan #N]`：第 N 次 `plan_task`。`tier=skill` 表示技能库覆盖了任务；
  `tier=action` 表示按动作契约推导。`skills` 是通过能力门的候选技能，
  `steps` 是校验后的动作数。
- `[s<id>]`：一个执行步。`step_id` 与 `run_sequence` 返回的 `steps[i].i` 一致。
  失败的步带 `error`；`recovery_managed` 会在下一行说明是否已由 RecoveryRail 处理。
- `[replan @s<id>]`：`trigger` 说明为什么改计划：
  `self_state`（实测状态否定前置条件）、`location`（感知位置失效）、
  `business_failure`（动作返回 `ok:false`）。`budget=已用/上限`。
  `new` 是替换后的剩余计划。
- `[outcome]`：任务级结果。`attempts` 是规划次数，`replans` 是重规划次数，
  `final_tier` 是最后一次成功规划的 tier；`tool trace` 是工具级 JSON 路径。

需要机器可读形式时加 `--json`：

```bash
jiuwensymbiosis-episode <file.jsonl> --json > episode.json
```

## 四、下钻到工具级

同时开了 `enable_tracing` 时：

1. `outcome.trace_path` 指向 `traces/*.json`；
2. 该 JSON 的每个 entry 多一个 `runner_step`，等于 episode 里对应 `step.step_id`；
3. 用 `jiuwensymbiosis-replay <trace.json> --text` 看工具级时间线（参数、耗时、帧、
   rail 事件、WARNING 日志）。

```bash
# 先看规划：为什么改、改成什么
jiuwensymbiosis-episode episode.jsonl
# 再看执行：那一步的 dig 内部发生了什么
jiuwensymbiosis-replay traces/task-xxx_....json --text
```

## 五、常见问题

**没有生成 episodes 文件**

- 确认 `enable_episode_trace: true` 在 `agent:` 块里，且字段名拼写正确
  （`RobotAgentConfig.from_dict` 对未知键抛 `TypeError`，不会静默忽略）。
- 确认任务走的是 `fastagent`；慢路径本版不写 episode。
- 看日志里是否有 `[fast] episode trace disabled: ...`——目录不可写时会回退 no-op。
- `run_fast_task` 的返回字典里 `episode_path` 是实际路径；`None` 表示没开。

**文件里有 `bad_lines`**

读取器会跳过损坏行（例如进程在写最后一行时崩溃）。其余事件仍完整；
需要时用 `load_episode(path).bad_lines` 查行号。写入端本身是逐行 flush 的，
正常结束不会产生坏行。

**`trace_path` 是 null**

只开了 `enable_episode_trace`，没开 `enable_tracing`。此时规划级信息完整，
但没有工具级明细可下钻。

**`recovery_managed` 为 true**

这一步的失败已经由 RecoveryRail 安全撤退/释放处理，runner 不会为它再触发
重规划（重规划只针对“业务失败”，例如挖点超出可达环带、卸点带料够不到）。
episode 里仍会记录这条失败 step 和 rail 的处理结果。

**重规划次数到了上限**

`run_sequence(max_replans=2)` 默认 2 次，漂移重规划和失败重规划共用。
用完后矛盾只 warning、步骤照常派发；episode 的 `replan.budget` 会显示
`used/max`，`outcome.replans` 是实际成功次数。

## 六、215F 示例

`extensions/jiuwen_agx/configs/agx_excavator/agx_excavator.215f*.yaml` 已开启两个开关，
并把 `agent.workspace` / `episode_dir` / `trace_dir` / `log_dir` 统一指到
`extensions/jiuwen_agx/runtime/`（相对仓库根）。因此 215F 的产物在：

```text
extensions/jiuwen_agx/runtime/episodes/*.jsonl
extensions/jiuwen_agx/runtime/traces/*.json
extensions/jiuwen_agx/runtime/logs/
```

跑完一斗后：

```bash
ls -t extensions/jiuwen_agx/runtime/episodes/*.jsonl | head -1
.venv/Scripts/python.exe -m jiuwensymbiosis.cli_episode \
  extensions/jiuwen_agx/runtime/episodes/<最新>.jsonl
```

一次典型的重规划时间线：`dig` 报 “outside the reachable annulus” → `replan
business_failure` → 新计划插入 `navigate_relative` → 第二次 `dig` 仍失败（卸点带料
够不到）→ 第二次重规划 → 第三次 `dig` 成功 → `outcome.ok=true`。这条链路在
episode JSONL 里可以逐行还原，在工具级 trace 里只能看到工具调用结果。
