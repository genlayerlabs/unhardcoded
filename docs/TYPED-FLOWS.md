# Typed decision and data flows

`flow_ir` can compose native decisions, deterministic JSON operations and optional
generation. These are general router capabilities: no node knows about agents,
conversation fragments, or a particular decision-model vendor. The core admits
and normalizes the entire finite DAG and every policy before inference.

## Example: classify tickets and draft selected replies

[`ticket-triage.json`](../examples/flows/ticket-triage.json) composes:

```text
input tickets ── decision ── select support ── JSON generation ── overlay ── output
      └──────────────────────────┴────────────────────────────────┘
```

Send that JSON as `flow_ir` to `POST /v1/chat/completions`, with
`messages: []` and `flow_input: {"a":"Application crashes", "b":"Pricing inquiry"}`.
Parse the returned `choices[0].message.content` as JSON. If neither ticket needs
support, the generation node is skipped and the original records are returned.
The example's policies select eligible low-input-price offers; replace them with
your own provider, residency and quality restrictions before real use. Protocol
and JSON-mode requirements still filter eligibility. Decision nodes fail the flow
by default; the example deliberately does not treat an unavailable classifier as
an authoritative decision. Questions are static and identify the supplied IDs.

## Nodes and options

| Kind / operation | Inputs and output |
| --- | --- |
| `decision` | One typed input becomes native decision state; multiple inputs become an ordered array. Declare a routing `policy` and 1–32 `questions` using `choice`, `score`, or `noul`. Output is the validated native answers map. |
| `data` / `project` | One input; `path` is 1–16 object keys. Missing keys fail. |
| `data` / `select` | `[records, answers]`; retain record IDs whose answer object's `field` equals the declared string `equals`. Missing or unmatched answers select nothing. |
| `data` / `overlay` | `[base, replacements, optional removals]`; removals win, absent replacements preserve originals, unknown IDs fail. Optional `min_string_bytes`, `max_string_bytes`, and `only_shrink` reject unsuitable replacements individually. |
| `data` / `union` | Merge 1–32 record maps; duplicate IDs fail. |
| `llm` | Existing generation node, now optionally returning typed JSON. |

Record maps have at most 128 IDs, each 1–128 UTF-8 bytes. The host bounds typed
JSON values and typed flow admission to 1 MiB and depth 32. Input may be an object,
array, or string; internal results also preserve JSON booleans, numbers and null.
Nodes remain a finite DAG; there is no dynamic loop, code evaluation or arbitrary
callback supplied by the caller. Build a bounded graph before submitting it.

`llm.output_format: "json"` requires complete JSON and a JSON-capable provider;
it rejects duplicate keys, non-finite numbers, truncation and tool-call responses.
It does not impose an application schema: downstream operations validate the
shape they need. `context: "inputs"` sends only the node system prompt and its
predecessor data, avoiding inherited conversation history. Without this option,
existing conversation inheritance remains unchanged.

`skip_empty: true` on decision or generation nodes passes through an empty first
input object/array without calling a provider. `on_error: "input"` on data,
decision or generation nodes explicitly preserves the first input on failure and
records a fallback. Choose that behavior only when the downstream graph can
interpret the original input safely; it is not an inferred decision or a successful
model response. Without it, a failed node fails the flow.

Generation supports `max_tokens` (1–4096); model nodes support `timeout_ms`
(1–40000). Typed flows share a 40-second execution budget, including routing and
provider fallbacks. After expiry, further inference is skipped; deterministic
nodes can still assemble declared fallbacks. External cancellation propagates.
Native decision payloads retain their existing 32 KB ASCII-serialized limit.

## Identity, costs and compatibility

The core includes every new semantic option in canonical identity. Existing flows
without these options retain their previous encoding and behavior; a golden
regression test locks the legacy encoding. The core's Lua reference driver and
host scheduler have operation-conformance tests. The host retains lossless JSON
values (including null) rather than passing runtime JSON through Lua tables.

`x_router.decision_trace.flow_nodes` reports node kind, edges, provider metadata,
skips and fallbacks. Pure data and skipped nodes have zero provider cost. A billed
response keeps its cost even if its JSON is rejected. Aggregate cost is null when
any attempted model call has unknown cost; it never silently sums only successful
legs. Token usage aggregates reported usage, which may be incomplete on failures.

## Compacting conversations is a preset

[`selective-compaction.json`](../examples/flows/selective-compaction.json) uses the
same primitives. [`/v1/compact`](FRAGMENT-COMPACTION.md) prepares that graph from
conversation units, then renders its result in message order. It has no separate
inference scheduler. Other applications can submit their own `flow_ir` directly.
