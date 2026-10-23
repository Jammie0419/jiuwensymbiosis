# Planning-Level Trace Reference

> Category: Reference.

This page documents the **planning-level trace (episode trace)**: one JSONL file
per task run covering "task → parse → plan → validate → execute → re-plan →
outcome". It is the upper layer of the two-layer trace:

| Layer | File | Answers |
|---|---|---|
| Planning | `<workspace>/episodes/*.jsonl` | Why this plan, how it was repaired, how the world changed, what the outcome was |
| Tool | `<workspace>/traces/*.json` | What each tool call did, its result and duration |

The layers reference each other through `TraceEntry.runner_step` and
`outcome.trace_path`; they do not duplicate each other. This version covers
`exec_mode="fastagent"`. The slow path (`stepagent`) only writes `mode` and has
no plan events.

## 1. Configuration

| Field | Default | Meaning |
|------|------|------|
| `enable_episode_trace` | `False` | Planning-level trace switch, independent of `enable_tracing` |
| `episode_dir` | `None` | Override (default `<workspace>/episodes`) |
| `enable_tracing` | `False` | Tool-level trace; only then is `outcome.trace_path` non-null and `runner_step` present on trace entries |

Workspace resolution matches the execution trace (explicit `workspace` >
environment variable > settings > `~/.jiuwensymbiosis/{session.name}_workspace/`).
The filename is `{safe_episode_id}_{YYYYmmdd_HHMMSS}_{microseconds}_{pid}.jsonl`,
so repeated runs with the same `conversation_id` never overwrite each other.

## 2. File format

One episode is one **JSONL** file: each line is a JSON object with the envelope
`schema` (`episode/1`), `seq` (1-based, monotonic), `ts` (UTC ISO-8601 ms) and
`type`. Every line is flushed as it is written, so a crash still leaves the
earlier events readable.

### Event types

| `type` | Key fields | Written when |
|---|---|---|
| `episode` | `episode_id`, `run_token`, `query`, `robot`, `mode`, `workspace`, `versions`, `capabilities`, `started_at` | File open, first line |
| `intent` | `ok`, `intent` or `error`, `duration_s` | After `parse_task` |
| `world` | `kind`, `state` (`WorldState.describe()`) | After the initial world snapshot |
| `plan_attempt` | `attempt`, `trigger`, `tier`, `skills`, `reason`, `raw`, `validated_steps`, `ok`, `error`, `duration_s` | After each `plan_task` call succeeds or fails |
| `step` | `step_id`, `op`, `params`, `bind`, `ok`, `result` or `error`, `error_code`, `recovery_managed`, `duration_s`, `tokens_after`, `loop_iter`, `cancelled` | After each step succeeds or fails |
| `replan` | `at_step`, `trigger`, `why`, `ok`, `error`, `budget`, `old_remaining`, `new_steps` | After each re-plan attempt |
| `outcome` | `ok`, `steps_done`, `replans`, `attempts`, `final_tier`, `duration_s`, `trace_path`, `failure_summary`, `error` | Episode end (normal, exception, cancellation) |

Conventions:

- `plan_attempt.trigger` is `initial` or `replan`.
- `replan.trigger` is `self_state` (measured self-state contradicts `requires`),
  `location` (sensed position invalidated), or `business_failure` (the action
  returned `ok:false`).
- On a re-plan the JSONL order is "new `plan_attempt` → `replan`": the callback
  computes the replacement, then the runner records the re-plan event. Pair them
  by `seq`.
- `step_id` is the runner's sequence cursor and matches `steps[i].i` in
  `run_sequence`'s result. It also covers loop-detect and `<replan>` records, so
  it is not a strict action position.
- A failed step's `tokens_after` is the state **after the safe retreat**
  (`duration_s` excludes the retreat), so it matches what the re-planner sees.
  A bucket loaded by the home motion therefore shows as `payload.held`.
- `outcome.final_tier` is the tier of the last successful plan attempt; all
  attempts are in `plan_attempt` lines.
- `outcome.trace_path` is non-null only when `enable_tracing=True`.

### Example

```jsonl
{"schema":"episode/1","seq":1,"ts":"2026-10-23T06:00:00.123+00:00","type":"episode","episode_id":"task-8f3a1c2d","run_token":"task-8f3a1c2d_20261023_140000_123456_4242","query":"dig one bucket","robot":"agx_excavator","mode":"fastagent","workspace":"/w","versions":{"harness":"v0","model":"glm-5"},"capabilities":["motion.excavator"],"started_at":"2026-10-23T06:00:00.123+00:00"}
{"schema":"episode/1","seq":2,"ts":"2026-10-23T06:00:00.300+00:00","type":"plan_attempt","attempt":1,"trigger":"initial","tier":"skill","skills":["excavate"],"reason":"","raw":[{"op":"get_terrain","params":{}},{"op":"dig","params":{"dig_x_m":6.59,"dig_y_m":2.0}}],"validated_steps":["get_terrain","dig"],"ok":true,"error":null,"duration_s":0.09}
{"schema":"episode/1","seq":3,"ts":"2026-10-23T06:00:00.360+00:00","type":"step","step_id":1,"op":"dig","params":{"dig_x_m":6.59,"dig_y_m":2.0},"bind":null,"ok":false,"result":null,"error":"ValueError: dig point at 6.59 m is outside the reachable annulus [1.00, 6.00] m","error_code":null,"recovery_managed":false,"duration_s":0.01,"tokens_after":["payload.clear"],"loop_iter":null,"cancelled":false}
{"schema":"episode/1","seq":4,"ts":"2026-10-23T06:00:00.510+00:00","type":"replan","at_step":1,"trigger":"business_failure","why":"dig failed: out of the reachable annulus","ok":true,"error":null,"budget":{"used":1,"max":2},"old_remaining":[{"op":"home","params":{},"bind":null}],"new_steps":[{"op":"navigate_relative","params":{"forward_m":1.0},"bind":null},{"op":"home","params":{},"bind":null}]}
{"schema":"episode/1","seq":5,"ts":"2026-10-23T06:00:53.000+00:00","type":"outcome","ok":true,"steps_done":8,"replans":1,"attempts":2,"final_tier":"skill","duration_s":53.0,"trace_path":"/w/traces/task-8f3a1c2d_....json","failure_summary":[{"step_id":1,"op":"dig","error":"ValueError: dig point at 6.59 m is outside the reachable annulus"}],"error":null}
```

## 3. Reader API

```python
from jiuwensymbiosis.agent.episode_trace import iter_episode, load_episode

for event in iter_episode("episode.jsonl"):   # skips bad lines
    ...

doc = load_episode("episode.jsonl")           # EpisodeDocument
doc.episode          # header
doc.intents
doc.worlds
doc.plan_attempts
doc.steps
doc.replans
doc.outcome          # last outcome, {} when absent
doc.bad_lines        # line numbers that could not be parsed
doc.as_dict()        # aggregate for --json / downstream consumers
```

`load_episode` logs a warning and skips corrupt lines, blank lines and invalid
UTF-8; one bad line never discards the batch.

## 4. Replay

```bash
jiuwensymbiosis-episode <episode.jsonl>          # text timeline
jiuwensymbiosis-episode <episode.jsonl> --json   # aggregated JSON
```

```text
=== Episode: task-8f3a1c2d_....jsonl ===
query: dig one bucket
robot=agx_excavator  mode=fastagent
outcome: ok=True steps=8 replans=1

[intent] ok  targets=['soil_pile_215f']
[world] tokens=['payload.clear']  capabilities=['motion.excavator']
[plan #1] ok  trigger=initial  tier=skill  skills=['excavate']  steps=3
  [s1] FAIL dig  0.01s
         error: ValueError: dig point at 6.59 m is outside the reachable annulus ...
[replan @s1] ok  trigger=business_failure  budget=1/2
         why: dig failed: out of the reachable annulus
         new: ['navigate_relative', 'home']
[outcome] ok=True  steps=8  replans=1  attempts=2  final_tier=skill
         tool trace: /w/traces/task-8f3a1c2d_....json
```

## 5. Reference to the tool-level trace

- `outcome.trace_path` points at the `traces/*.json` of the same run (only when
  `enable_tracing` is on).
- Each `TraceEntry` in that JSON has a `runner_step` equal to the episode
  `step.step_id`. `track_detect`, `track_grasp` and `<replan>` do not map to a
  tool call, so their episode records have no trace entry.
- Consistency check: records with the same `runner_step` should agree on
  `op`/`tool_name` and `ok`/`success`. When they disagree, use the episode's
  business failure reason and the trace's `rail_events` together; do not trust
  one layer alone.

## 6. Failure semantics

| Scenario | Behaviour |
|---|---|
| Directory/file cannot be opened | Warning only; falls back to `NullEpisodeRecorder`; the task runs |
| Write/serialisation failure | Logs once, stops writing; execution is unaffected |
| Crash/cancellation mid-run | `__exit__` appends `outcome(ok=false, error=...)` |
| Bad line / invalid UTF-8 | Reader skips it and records `bad_lines`; other events survive |
| `enable_tracing=False` | Episode still written; `trace_path=null`, no `runner_step` |
| `enable_episode_trace=False` | No directory/file; zero overhead (`NullEpisodeRecorder`) |

## 7. Files

| File | Role |
|---|---|
| [jiuwensymbiosis/agent/episode_trace.py](../../../jiuwensymbiosis/agent/episode_trace.py) | Writer, reader, `EpisodeDocument` |
| [jiuwensymbiosis/agent/step_context.py](../../../jiuwensymbiosis/agent/step_context.py) | Contextvar bridge from runner step to tool call |
| [jiuwensymbiosis/agent/fast/runner.py](../../../jiuwensymbiosis/agent/fast/runner.py) | `step` / `replan` emission points |
| [jiuwensymbiosis/agent/run.py](../../../jiuwensymbiosis/agent/run.py) | `episode` / `intent` / `world` / `plan_attempt` / `outcome` wiring |
| [jiuwensymbiosis/cli_episode.py](../../../jiuwensymbiosis/cli_episode.py) | `jiuwensymbiosis-episode` replay |
| [jiuwensymbiosis/utils/json_safe.py](../../../jiuwensymbiosis/utils/json_safe.py) | Shared JSON normalisation for both layers |
