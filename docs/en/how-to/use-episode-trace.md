# Record and Replay Planning-Level Traces

> Category: How-to.

The execution trace (`TraceRail`) answers "what did each tool call do". To
answer "**why was the plan chosen, how was it repaired, and what was the
outcome**", use the planning-level trace: one JSONL file per episode with the
task, intent, world state, every plan attempt, every step result, re-plans and
the final outcome. The schema lives in
[Planning-Level Trace Reference](../reference/episode-trace.md).

This version covers `exec_mode="fastagent"`; the slow path (`stepagent`) has no
plan events yet.

## 1. Enable it

Add two switches to the `agent:` block (either can be enabled alone):

```yaml
agent:
  exec_mode: fastagent
  # Planning level: episodes/*.jsonl
  enable_episode_trace: true
  # Tool level: traces/*.json (optional; needed for cross-references)
  enable_tracing: true
  # episode_dir: ./episodes    # override (default <workspace>/episodes)
```

Equivalent in code:

```python
config = RobotAgentConfig(
    exec_mode="fastagent",
    enable_episode_trace=True,
    enable_tracing=True,
)
```

The switches are independent:

| | `enable_episode_trace=True` | `enable_episode_trace=False` |
|---|---|---|
| `enable_tracing=True` | Both layers; `outcome.trace_path` links them | Tool level only (current behaviour) |
| `enable_tracing=False` | Planning level only; `trace_path=null` | Zero overhead (default) |

## 2. Where the files are

Default `<workspace>/episodes/`, a sibling of `traces/`. Workspace resolution:

```text
explicit config.workspace
  > $JIUWENSYMBIOSIS_WORKSPACE
  > "workspace" in ~/.jiuwensymbiosis/settings.json
  > ~/.jiuwensymbiosis/{session.name}_workspace/
```

The filename is `{conversation_id}_{timestamp}_{microseconds}_{pid}.jsonl`.
`run_fast_task` returns an `episode_path` pointing at it (`None` when disabled).

## 3. Read one run

```bash
jiuwensymbiosis-episode ~/.jiuwensymbiosis/agx_excavator_workspace/episodes/task-xxx.jsonl
```

```text
=== Episode: task-8f3a1c2d_....jsonl ===
query: dig one bucket
robot=agx_excavator  mode=fastagent
outcome: ok=True steps=8 replans=1

[intent] ok  targets=['soil_pile_215f']
[world] tokens=['payload.clear']  capabilities=['motion.excavator']
[plan #1] ok  trigger=initial  tier=skill  skills=['excavate']  steps=3
  [s0] ok   get_terrain  0.02s
  [s1] FAIL dig  0.01s
         error: ValueError: dig point at 6.59 m is outside the reachable annulus ...
[replan @s1] ok  trigger=business_failure  budget=1/2
         why: dig failed: out of the reachable annulus
         new: ['navigate_relative', 'get_terrain', 'dig', 'home']
[plan #2] ok  trigger=replan  tier=skill  skills=['excavate']  steps=4
  [s5] ok   dig  48.00s
[outcome] ok=True  steps=8  replans=1  attempts=2  final_tier=skill
         tool trace: /w/traces/task-8f3a1c2d_....json
```

How to read it:

- `[plan #N]`: the Nth `plan_task` call. `tier=skill` means the skill library
  covered the task; `tier=action` means it was derived from the action
  contracts. `skills` are the candidates that passed the capability gate;
  `steps` is the validated action count.
- `[s<id>]`: one executed step. `step_id` matches `steps[i].i` in
  `run_sequence`'s result. A failed step carries `error`.
- `[replan @s<id>]`: why the plan changed — `self_state`, `location`, or
  `business_failure` — with the budget and the replacement steps.
- `[outcome]`: task-level result. `attempts` counts plans, `replans` counts
  successful re-plans, `final_tier` is the last successful plan tier.

For a machine-readable form:

```bash
jiuwensymbiosis-episode <file.jsonl> --json > episode.json
```

## 4. Drill down to the tool level

When `enable_tracing` is also on:

1. `outcome.trace_path` points at `traces/*.json`;
2. each entry there has `runner_step` equal to the episode `step.step_id`;
3. `jiuwensymbiosis-replay <trace.json> --text` shows params, duration, frames,
   rail events and WARNING logs.

```bash
jiuwensymbiosis-episode episode.jsonl                    # why the plan changed
jiuwensymbiosis-replay traces/task-xxx_....json --text    # what each call did
```

## 5. Troubleshooting

**No episodes file**

- Check `enable_episode_trace: true` is under `agent:` and spelled exactly
  (`RobotAgentConfig.from_dict` raises `TypeError` on unknown keys).
- Check the task uses `fastagent`; the slow path does not write episodes yet.
- Look for `[fast] episode trace disabled: ...` in the log — an unwritable
  directory falls back to a no-op recorder.
- `run_fast_task(...)["episode_path"]` is the actual path; `None` means disabled.

**`bad_lines` is non-empty**

The reader skipped corrupt lines (for example a crash while writing the last
line). The remaining events are intact; inspect line numbers through
`load_episode(path).bad_lines`.

**`trace_path` is null**

Only `enable_episode_trace` is on. Planning information is complete, but there
is no tool-level detail to drill into.

**`recovery_managed` is true**

RecoveryRail already retreated/released; the runner does not re-plan for these
failures (re-planning is for business failures such as an out-of-envelope dig
point or an unreachable loaded-bucket dump pose). The failed step and the rail
handling are still recorded.

**The re-plan budget ran out**

`run_sequence(max_replans=2)` defaults to 2, shared by drift and business
failures. Once spent, contradictions are logged and the step is dispatched
anyway; `replan.budget` shows `used/max` and `outcome.replans` the actual count.

## 6. 215F example

`extensions/jiuwen_agx/configs/agx_excavator/agx_excavator.215f*.yaml` enable both
switches and point `agent.workspace` / `episode_dir` / `trace_dir` / `log_dir` at
`extensions/jiuwen_agx/runtime/` (relative to the repository root). 215F output
therefore lands in:

```text
extensions/jiuwen_agx/runtime/episodes/*.jsonl
extensions/jiuwen_agx/runtime/traces/*.json
extensions/jiuwen_agx/runtime/logs/
```

After one bucket:

```bash
ls -t extensions/jiuwen_agx/runtime/episodes/*.jsonl | head -1
.venv/Scripts/python.exe -m jiuwensymbiosis.cli_episode \
  extensions/jiuwen_agx/runtime/episodes/<latest>.jsonl
```

A typical timeline: `dig` fails with "outside the reachable annulus" → re-plan
inserts `navigate_relative` → a second `dig` fails because the loaded bucket
cannot reach the dump point → a second re-plan → the third `dig` succeeds and
`outcome.ok=true`. The episode JSONL reconstructs this reasoning chain; the
tool-level trace only shows the call results.
