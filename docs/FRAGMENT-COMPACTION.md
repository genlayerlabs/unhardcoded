# Decision-guided compaction preset

`POST /v1/compact` retains its existing single-summary behavior unless the caller
supplies `decision_policy_ir`. With that field, a pure adapter materializes the
[`selective-compaction.json`](../examples/flows/selective-compaction.json) preset
and executes it through the same [typed flow runtime](TYPED-FLOWS.md) exposed by
`flow_ir` on chat completions. All inference, selection and replacement run in
that shared engine. The core and scheduler contain no compaction-specific nodes.

1. Group each assistant tool call with adjacent tool results. Preserve all
   system/developer messages, pinned input and complete recent units verbatim.
2. Prepare bounded batches and static native questions in the flow. The decision
   policy classifies old fragments as `keep`, `summarize` or `archive`.
3. Select only `summarize` records for a JSON generation node. Empty selections
   make no generative calls. Select `archive` records for deterministic removal.
4. Overlay validated summaries and removals, then reassemble in original order.
   Failed, truncated, expanded or invalid summaries preserve the original unit.
   When classification sees only an excerpt, `archive` is not an allowed choice.

| Optional request field | Meaning |
| --- | --- |
| `decision_policy_ir` | Policy for native decision nodes; enables this preset. |
| `target_ratio` | Desired output/input serialized UTF-8 byte ratio, default `0.1`, range `(0,1]`; not a tokenizer count. |
| `pinned_indices` | Original zero-based message indexes to retain; defaults to all user messages. System/developer messages and complete recent units remain protected. |

`policy_ir` selects summary generators. Both policies must include required
provider restrictions. Clients that encode generated observations as user messages
should explicitly pin their real user instructions; the adapter does not infer
instruction provenance from message text.

The response's `compaction` object reports original/output/target bytes,
`target_met`, fragment actions (`start` inclusive, `end` exclusive), preparation
limit reasons, and `flow_fingerprint`. Model failures and fallback details appear
in `x_router.decision_trace.flow_nodes`. The 10% target never overrides protected
or kept evidence; expansion is rejected and the actual result size is reported.

Limits: at most 128 selectable fragments and 32 batches, with at most eight
fragments, one decision and one conditional summary call per batch. Decision
requests fit the native 32 KB ASCII bound; full summary input records fit 50 KB.
The complete flow input stays below 900 KB. All nodes share a 40-second execution
budget (individual decision timeout 7 seconds, summary timeout 20 seconds).
Summary calls allow at most 4,096 output tokens or the caller's smaller limit.
Oversized units, context that cannot fit, and failed decisions retain evidence.

`x_router.cost_usd` includes model calls and any routing decisions; it is null if
an attempted leg has unknown cost. Reported token usage is aggregated, but may be
incomplete when a provider fails. Skipped generation and deterministic operations
cost zero. Responses with no executed nodes omit cost and usage metadata.

This is a stateless transform: **the caller must retain the original transcript**.
`archive` removes active context; the router does not persist an archive. Summary
markers identify original fragments. Unchanged prefix messages stay byte-for-byte
equivalent; compaction can invalidate provider cache entries after the first
change. Summary correctness, cache hits and latency savings need workload-specific
evaluation and are not guaranteed by this preset.

Hermetic coverage: `pytest tests/test_flow_data.py tests/test_fragment_compaction.py
tests/test_compact.py`. The optional live compaction BDD requires an explicit
`DECISION_COMPACTION_POLICY_IR` and can incur provider charges.
