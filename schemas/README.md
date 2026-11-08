# Protocol contracts, version 1.0.0

Use the validator to check a run manifest and its JSONL event log. It checks fields, references between observations/requests/actions, and event order. Use a case runner to execute models or evaluate tasks.

From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python tools/validate_contracts.py --examples
python -m pytest -q
```

To check your own complete run:

```bash
python tools/validate_contracts.py --manifest run.manifest.json --trace run.trace.jsonl
```

`--examples` and `--manifest/--trace` are mutually exclusive. Invalid artifacts return exit status 1 with a diagnostic; command usage errors return 2. The validator only reads artifacts and resolves its bundled schemas locally.

## Manifest

`run-manifest.schema.json` fixes `schema_version` to `1.0.0`. It requires the run identifier, an explicit `synthetic` flag, protocol settings, clock-domain declarations, episode IDs/seeds, and a task budget. Model and benchmark provenance belongs in the corresponding case manifest.

The schedule axis is independent of the clock/delay axis:

| Schedule | Meaning |
| --- | --- |
| `sync` | Sequential observation, request, output and action operation |
| `history_observation` | A synchronous control flow supplies historical observations; requires `history_steps` |
| `async` | The producer declares overlap scheduling |

| Clock | Allowed delay modes |
| --- | --- |
| `virtual` | `zero`, `profile_replay` |
| `realtime` | `measured`, `additive`, `target_total` |

All three schedules can be combined with all five valid clock/delay pairs. `profile_replay` requires `profile_id`. Chunk prediction length and execution length are separate, with execution length no larger than prediction length. Provide producer evidence for actual history offsets and overlap behavior; these are outside the contract checks.

Each `clock_domains` key fixes a `kind`, `process_id` and `monotonic_origin` for the whole run, including episode resets. Use separate IDs across processes/machines unless an alignment protocol establishes a shared origin. Compute durations within a domain.

`budget.clock_domain` selects the simulation or host domain, and `start_event` is explicitly `first_observation_sampled`. A producer must provide an observation timestamp in that domain to calculate elapsed budget time. The optional `host_watchdog` is separate and requires a host-monotonic domain. The validator checks declarations. Budget enforcement is a runtime responsibility; clock alignment is required when sharing an origin across processes.

Optional `metrics` contain `trials`, `successes`, `successful_task_time_mean_ns` and `successful_chunk_count_mean`. Require successes ≤ trials and null means when successes=0. Compute these summaries from episode evidence before validation; this tool checks the supplied fields.

## Trace

`trace-event.schema.json` describes one JSON object per JSONL line. Blank lines are ignored. JSON numbers must be finite; nanosecond timestamps are nonnegative integers. Every event includes:

```text
schema_version, run_id, episode_id, event_seq,
event_type, clock_domain, timestamp_ns
```

`event_seq` strictly increases in serialized run order, with gaps allowed. Timestamps are nondecreasing within each clock domain; origins may differ across domains. Producers supply this ordering.

The validator checks these associations and lifecycle conditions:

- An observation exists before its request is submitted, and both belong to the same episode.
- A request starts before completion. Successful completion assigns a unique chunk. Error/cancelled completion has no chunk.
- A chunk completes successfully before release and is released before action execution. Release refers to its request's chunk and uses the manifest's delay mode/profile ID.
- Each action refers to the chunk's original observation and an index below the declared execution length. It starts once and finishes only after a start.
- Dropped output cannot execute. A pending, started request may complete after being dropped so that a late callback can be recorded.
- Episodes run sequentially. A new episode requires the previous episode's terminal boundary. Episode IDs and observation/request/chunk IDs are unique within a run.
- Each declared episode has exactly one `episode_finished`. New observations/requests, inference starts, releases and action events are rejected after terminal. Completion/drop diagnostics and underrun diagnostics can still be recorded to describe cleanup.

Supported event types are `observation_sampled`, `inference_submitted`, `inference_started`, `inference_completed`, `result_released`, `action_started`, `action_finished`, `queue_underrun`, `result_dropped`, and `episode_finished`. Event-specific payloads are defined by the schema. Unknown properties are rejected to catch misspellings and silent semantic changes.

`result_released.delay_ns` stores the declared delay: modeled total for `profile_replay`, extra delay for `additive`, target total for `target_total`, and zero for no injection. `overshoot_ns` records producer overshoot; `profile_id` is null without a profile. Validation covers types and associations; timing arithmetic belongs to the producer.

Actual `observation_age` uses observation and `action_started` timestamps in the same domain. Actual boundary wait uses action start minus the preceding chunk boundary, including release gating. `inference_completed` does not substitute for `result_released` when injection postpones availability. Compute these quantities from trace timestamps; report paper-model residual delay separately.

Use producer/runtime tests to validate physical dynamics, model quality, clock alignment, injection arithmetic, deadlines, queue capacity, single-flight scheduling, resource drain and metric estimates. Schema extensions require a version decision, example data and acceptance/rejection tests.
