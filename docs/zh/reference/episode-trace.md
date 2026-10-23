# 规划级轨迹参考

> 类别：Reference。

本页是**规划级trace（episode trace）**的字段与格式参考。它记录一次任务从「任务 → 解析 →
规划 → 校验 → 执行 → 重规划 → 结果」的完整回合，与工具级的
[执行轨迹](tracing.md)（`TraceRail`）是分层关系：

| 层 | 文件 | 回答的问题 |
|---|---|---|
| 规划级 | `<workspace>/episodes/*.jsonl` | 为什么这么规划、计划怎么被修正、世界怎么变、最终结果如何 |
| 工具级 | `<workspace>/traces/*.json` | 每个工具调用做了什么、结果如何、耗时多少 |

两层通过 `TraceEntry.runner_step` 与 `outcome.trace_path` 互相引用，不重复记录。
本版覆盖 `exec_mode="fastagent"`；慢路径（`stepagent`）只写 `mode` 字段，没有 plan 事件。

## 一、配置

| 字段 | 默认 | 说明 |
|------|------|------|
| `enable_episode_trace` | `False` | 规划级 trace 开关，独立于 `enable_tracing` |
| `episode_dir` | `None` | 覆盖 episode 目录（默认 `<workspace>/episodes`） |
| `enable_tracing` | `False` | 工具级 trace；两者都开时 episode 的 `outcome.trace_path` 才非空，TraceRail entry 才带 `runner_step` |

workspace 的解析顺序与执行轨迹一致（显式 `workspace` > 环境变量 > settings > 默认
`~/.jiuwensymbiosis/{session.name}_workspace/`）。episode 文件名为
`{safe_episode_id}_{YYYYmmdd_HHMMSS}_{微秒}_{pid}.jsonl`，同一 `conversation_id`
重复运行不会覆盖。

## 二、文件格式

一个 episode 一份 **JSONL**：每行一个 JSON 对象，公共信封为
`schema`（固定 `episode/1`）、`seq`（文件内从 1 单调递增）、`ts`（UTC ISO-8601 毫秒）、
`type`。逐行 flush，崩溃时已写的事件仍可读。

### 事件类型

| `type` | 关键字段 | 写入时机 |
|---|---|---|
| `episode` | `episode_id`、`run_token`、`query`、`robot`、`mode`、`workspace`、`versions`、`capabilities`、`started_at` | 打开文件时，第一行 |
| `intent` | `ok`、`intent` 或 `error`、`duration_s` | `parse_task` 之后 |
| `world` | `kind`、`state`（`WorldState.describe()` 原样） | 首次世界状态快照后 |
| `plan_attempt` | `attempt`、`trigger`、`tier`、`skills`、`reason`、`raw`、`validated_steps`、`ok`、`error`、`duration_s` | 每次 `plan_task` 返回或抛错后 |
| `step` | `step_id`、`op`、`params`、`bind`、`ok`、`result` 或 `error`、`error_code`、`recovery_managed`、`duration_s`、`tokens_after`、`loop_iter`、`cancelled` | 每步成功/失败后 |
| `replan` | `at_step`、`trigger`、`why`、`ok`、`error`、`budget`、`old_remaining`、`new_steps` | 每次重规划尝试后 |
| `outcome` | `ok`、`steps_done`、`replans`、`attempts`、`final_tier`、`duration_s`、`trace_path`、`failure_summary`、`error` | episode 结束（正常/异常/取消） |

约定：

- `plan_attempt.trigger` = `initial`（首次规划）或 `replan`（重规划回调里的规划）。
- `replan.trigger` = `self_state`（自状态矛盾）、`location`（感知位置失效）或
  `business_failure`（动作返回 `ok:false`）。
- 重规划时 JSONL 的顺序是「新的 `plan_attempt` → `replan`」，因为回调先算出新计划、
  执行器再记录这次重规划事件；读者按 `seq` 配对即可。
- `step_id` 来自 runner 的序列游标，与 `run_sequence` 返回值里的 `i` 一致；
  它包含 loop detect 和 `<replan>` 记录，不是严格的动作位置。
- 失败 `step` 的 `tokens_after` 是**安全撤退之后**的状态（`duration_s` 不含撤退），
  所以它和紧接着的重规划看到的世界一致；`bucket_loaded` 由 `home` 带上来的料
  会体现为 `payload.held`。
- `outcome.final_tier` 是最后一次成功 `plan_attempt` 的 tier；所有尝试的 tier 见
  `plan_attempt` 行。
- `outcome.trace_path` 仅在 `enable_tracing=True` 时非空。

### 示例

```jsonl
{"schema":"episode/1","seq":1,"ts":"2026-10-23T06:00:00.123+00:00","type":"episode","episode_id":"task-8f3a1c2d","run_token":"task-8f3a1c2d_20261023_140000_123456_4242","query":"把土堆挖一斗倒到旁边","robot":"agx_excavator","mode":"fastagent","workspace":"/w","versions":{"harness":"v0","model":"glm-5","exec_mode":"fastagent","cerebellum":"dig:ik"},"capabilities":["motion.excavator","sensing.terrain"],"started_at":"2026-10-23T06:00:00.123+00:00"}
{"schema":"episode/1","seq":2,"ts":"2026-10-23T06:00:00.200+00:00","type":"intent","ok":true,"intent":{"targets":["soil_pile_215f"]},"error":null,"duration_s":0.08}
{"schema":"episode/1","seq":3,"ts":"2026-10-23T06:00:00.210+00:00","type":"world","kind":"initial","state":{"tokens":["payload.clear"],"joints":[0.0,-0.67,0.68,1.99],"extra":{"reach_min_m":1.0,"reach_max_m":6.0,"dig_mode":"ik"},"capabilities":["motion.excavator"]}}
{"schema":"episode/1","seq":4,"ts":"2026-10-23T06:00:00.300+00:00","type":"plan_attempt","attempt":1,"trigger":"initial","tier":"skill","skills":["excavate"],"reason":"","raw":[{"op":"get_terrain","params":{}},{"op":"dig","params":{"dig_x_m":6.59,"dig_y_m":2.0,"dump_x_m":-5.0,"dump_y_m":0.0}},{"op":"home","params":{}}],"validated_steps":["get_terrain","dig","home"],"ok":true,"error":null,"duration_s":0.09}
{"schema":"episode/1","seq":5,"ts":"2026-10-23T06:00:00.350+00:00","type":"step","step_id":0,"op":"get_terrain","params":{},"bind":null,"ok":true,"result":{"piles":[{"name":"soil_pile_215f","x_m":5.0,"y_m":2.0,"z_m":1.95}]},"error":null,"error_code":null,"recovery_managed":false,"duration_s":0.02,"tokens_after":["payload.clear"],"loop_iter":null,"cancelled":false}
{"schema":"episode/1","seq":6,"ts":"2026-10-23T06:00:00.360+00:00","type":"step","step_id":1,"op":"dig","params":{"dig_x_m":6.59,"dig_y_m":2.0,"dump_x_m":-5.0,"dump_y_m":0.0},"bind":null,"ok":false,"result":null,"error":"ValueError: dig point at 6.59 m is outside the reachable annulus [1.00, 6.00] m","error_code":null,"recovery_managed":false,"duration_s":0.01,"tokens_after":["payload.clear"],"loop_iter":null,"cancelled":false}
{"schema":"episode/1","seq":7,"ts":"2026-10-23T06:00:00.500+00:00","type":"plan_attempt","attempt":2,"trigger":"replan","tier":"skill","skills":["excavate"],"reason":"","raw":[{"op":"navigate_relative","params":{"forward_m":1.0}},{"op":"get_terrain","params":{}},{"op":"dig","params":{"dig_x_m":5.59,"dig_y_m":2.0,"dump_x_m":4.5,"dump_y_m":0.0}},{"op":"home","params":{}}],"validated_steps":["navigate_relative","get_terrain","dig","home"],"ok":true,"error":null,"duration_s":0.14}
{"schema":"episode/1","seq":8,"ts":"2026-10-23T06:00:00.510+00:00","type":"replan","at_step":1,"trigger":"business_failure","why":"dig 失败：ValueError: dig point at 6.59 m is outside the reachable annulus","ok":true,"error":null,"budget":{"used":1,"max":2},"old_remaining":[{"op":"home","params":{},"bind":null}],"new_steps":[{"op":"navigate_relative","params":{"forward_m":1.0},"bind":null},{"op":"get_terrain","params":{},"bind":null},{"op":"dig","params":{"dig_x_m":5.59,"dig_y_m":2.0,"dump_x_m":4.5,"dump_y_m":0.0},"bind":null},{"op":"home","params":{},"bind":null}]}
{"schema":"episode/1","seq":9,"ts":"2026-10-23T06:00:50.000+00:00","type":"step","step_id":5,"op":"dig","params":{"dig_x_m":4.28,"dig_y_m":2.0,"dump_x_m":4.5,"dump_y_m":0.0},"bind":null,"ok":true,"result":{"volume_m3":0.9,"cycle_s":48.0,"mode":"ik","tip_error_m":0.128},"error":null,"error_code":null,"recovery_managed":false,"duration_s":48.0,"tokens_after":["payload.clear"],"loop_iter":null,"cancelled":false}
{"schema":"episode/1","seq":10,"ts":"2026-10-23T06:00:53.000+00:00","type":"outcome","ok":true,"steps_done":8,"replans":1,"attempts":2,"final_tier":"skill","duration_s":53.0,"trace_path":"/w/traces/task-8f3a1c2d_....json","failure_summary":[{"step_id":1,"op":"dig","error":"ValueError: dig point at 6.59 m is outside the reachable annulus"}],"error":null}
```

## 三、读取 API

```python
from jiuwensymbiosis.agent.episode_trace import iter_episode, load_episode

for event in iter_episode("episode.jsonl"):   # 坏行跳过
    ...

doc = load_episode("episode.jsonl")           # EpisodeDocument
doc.episode          # 头部
doc.intents          # list
doc.worlds           # list
doc.plan_attempts    # list
doc.steps            # list
doc.replans          # list
doc.outcome          # 最后一条 outcome（没有则 {}）
doc.bad_lines        # 无法解析的行号
doc.as_dict()        # 聚合结果，供 --json / 下游消费
```

`load_episode` 对损坏行、空行、非 UTF-8 字节都只记 warning 并跳过；一条坏行不会
让整批事件失败。

## 四、回放

```bash
jiuwensymbiosis-episode <episode.jsonl>          # 文本时间线
jiuwensymbiosis-episode <episode.jsonl> --json   # 聚合事件 JSON
```

文本示例：

```text
=== Episode: task-8f3a1c2d_....jsonl ===
query: 把土堆挖一斗倒到旁边
robot=agx_excavator  mode=fastagent
workspace=/w
versions={"harness": "v0", "model": "glm-5"}
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

## 五、与工具级 trace 的引用

- `outcome.trace_path` 指向同一次运行的 `traces/*.json`；开启 `enable_tracing` 时才有。
- 该 JSON 的每个 `TraceEntry` 多一个 `runner_step` 字段，值等于 episode `step.step_id`。
  读者可以按这个键把「规划步」和「工具调用」对上；`track_detect` / `track_grasp` /
  `<replan>` 不对应工具调用，它们的 episode 记录没有 TraceRail 引用。
- 两级的一致性校验建议：`runner_step` 相同的记录，`op`/`tool_name` 应一致，
  `ok`/`success` 应一致；不一致时以 episode 的业务失败原因和 TraceRail 的
  `rail_events` 交叉判断，不要只信一层。

## 六、失败语义

| 场景 | 行为 |
|---|---|
| 目录/文件打不开 | 只 warning，回退 `NullEpisodeRecorder`，任务照常执行 |
| 写入/序列化失败 | 记一次 warning 后停写；不影响执行，后续事件 no-op |
| 中途崩溃/取消 | `__exit__` 补一条 `outcome(ok=false, error=...)` |
| 坏行/非 UTF-8 | 读取时跳过并计入 `bad_lines`，其余事件完整 |
| `enable_tracing=False` | episode 照常写，`trace_path=null`，TraceRail entry 无 `runner_step` |
| `enable_episode_trace=False` | 不创建目录/文件，零开销（`NullEpisodeRecorder`） |

## 七、相关文件

| 文件 | 角色 |
|---|---|
| [jiuwensymbiosis/agent/episode_trace.py](../../../jiuwensymbiosis/agent/episode_trace.py) | 写入器、读取器、`EpisodeDocument` |
| [jiuwensymbiosis/agent/step_context.py](../../../jiuwensymbiosis/agent/step_context.py) | runner step → 工具调用的 contextvar 桥 |
| [jiuwensymbiosis/agent/fast/runner.py](../../../jiuwensymbiosis/agent/fast/runner.py) | `step` / `replan` 事件发射点 |
| [jiuwensymbiosis/agent/run.py](../../../jiuwensymbiosis/agent/run.py) | `episode` / `intent` / `world` / `plan_attempt` / `outcome` 接线 |
| [jiuwensymbiosis/cli_episode.py](../../../jiuwensymbiosis/cli_episode.py) | `jiuwensymbiosis-episode` 回放 |
| [jiuwensymbiosis/utils/json_safe.py](../../../jiuwensymbiosis/utils/json_safe.py) | 两级 trace 共用的 JSON 归一 |
| [tests/unit_tests/agent/test_episode_trace.py](../../../tests/unit_tests/agent/test_episode_trace.py) | 写入/读取/坏行/写入失败测试 |
| [tests/unit_tests/fast/test_runner_episode.py](../../../tests/unit_tests/fast/test_runner_episode.py) | step/replan 事件测试 |
| [tests/unit_tests/fast/test_run_fast_task_episode.py](../../../tests/unit_tests/fast/test_run_fast_task_episode.py) | 快路径端到端接线测试 |
