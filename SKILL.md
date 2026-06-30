---
name: sigma-policy-author
description: >-
  Author Σ_pol policies and Σ_flow flows for this LLM policy host. Load this
  file into any assistant and it can generate valid `policy_ir` / `flow_ir`
  terms (as JSON) to POST against the host's OpenAI-compatible endpoint —
  the live model/provider catalog is embedded below so the policies target
  models this host actually serves.
---

# Authoring Σ_pol & Σ_flow for this host

This host routes one OpenAI-compatible call to the best provider for it. **You
do not pick a provider or a model name** — you submit a *policy* (a piece of
data) that says how to choose, and the host evaluates it over its live catalog
(prices, benchmarks, latency, breakers) and picks. Two languages:

- **Σ_pol** (`policy_ir`) — decides *which model serves one call*: a filter
  (who qualifies) → a score (rank the survivors) → a selector (pick / cascade).
- **Σ_flow** (`flow_ir`) — decides *how several calls compose*: a DAG of nodes,
  each node carrying its own Σ_pol policy.

Both are **data**: serializable JSON arrays, hashable, admitted before they run.
There are no loops, no I/O, no side effects — a term *decides*, it does not *do*.

## How to send a policy

`POST /v1/chat/completions` (OpenAI-compatible). Put the term in `policy_ir`
(or `flow_ir`). The `model` field is ignored for selection when `policy_ir` is
present — the policy drives the choice.

```json
{
  "model": "policy:auto",
  "messages": [{"role": "user", "content": "..."}],
  "policy_ir": ["policy",
    ["and", ["meets_req"], ["not", ["is", "disabled"]]],
    ["add", ["scale", 0.7, ["normalize", ["field", "bench_intelligence"]]],
            ["scale", 0.3, ["neg", ["normalize", ["field", "price_out"]]]]],
    ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]
}
```

## The feedback loop — author → preview → run → read

Every step is a request with the **same** `Authorization: Bearer <key>` you used to fetch this guide. You never have to fly blind:

1. **Author** the `policy_ir` (the grammar is below).
2. **Admit & identify — no spend.** `POST /x/policy/normalize` `{policy_ir}` → `{policy_ir, fingerprint, version}`. A `400` here pinpoints what's invalid (unknown op, undeclared field, …) so you fix the term before paying.
3. **Preview the ranking — no spend.** `POST /x/rank` `{policy_ir}` → `{ranked, rejected}`: the candidates this host would admit and how it orders them, plus the ones it filtered out, each with the `reason` it failed. This is how you see *what your policy does* without a single call.
4. **Run it for real.** `POST /v1/chat/completions` with `policy_ir` (or `flow_ir`) + `messages` (the example above). A real call — real spend.
5. **Read how it routed.** The response carries **`x_router`** — your debugger for what actually happened:

   | `x_router` field | what it tells you |
   |---|---|
   | `provider` · `served_model_id` | the model that answered |
   | `served_by` | the *executed route* — the marketplace peer, or the provider for a direct route |
   | `cost_usd` · `price_in` · `price_out` | what the call cost |
   | `policy_fingerprint` | the identity of the policy that ran (matches `/x/policy/normalize`) |
   | `decision_trace` | `ranked` (the candidates considered) **and** `decision_path` (the real fallback attempts — which routes were tried, and `ok` or the error each hit) |
   | `session_acc` | running totals, when the call carries a session |

   `decision_trace.decision_path` is the fallback story — exactly which routes were tried and why it fell through; `served_by` + `cost_usd` say where it landed and what it cost. Refine the term and loop.

**The same loop for a Σ_flow** (a DAG of nodes, each with its own `policy_ir` — see *Σ_flow* below):

1. **Author** the `flow_ir`.
2. **Admit & identify — no spend.** `POST /x/flow/normalize` `{flow_ir}` → `{flow_ir, fingerprint, version}`.
3. **Preview — no spend.** A flow has no single ranking (each node routes on its own policy), so preview a node by sending *its* `policy_ir` to `POST /x/rank`.
4. **Run it for real.** `POST /v1/chat/completions` with `flow_ir` + `messages`.
5. **Read how it routed.** `x_router.decision_trace` carries **`flow_nodes`** — one entry per node with its `node` id, `provider` / `served_by`, tokens, latency, and the node's own `decision_path` (its fallback attempts). That's the per-node debugger: you see which node ran what, where each landed, and any fallback inside a node.

## The Σ_pol term, exactly

> The operators below are the `sigma-pol/v2` signature. The **normative
> grammar is the core spec** (`core/docs/SIGMA-POL.md`) — this guide mirrors it
> for authoring convenience; on a major version bump, regenerate against the
> spec. The **field vocabulary**, by contrast, is injected live from the host
> (see *Field vocabulary* below), so it never drifts from what the host serves.

A policy is a 6-element array — fill the three middle slots, keep the rest as-is:

```
["policy", <Pred>, <Scorer>, <Selector>, ["id"], ["always", {"action":"next_candidate"}]]
            filter    score      pick       xform   fail-plan
```

### `<Pred>` — who qualifies (joined by AND, default-deny)

Always start the filter with the host floor, then AND your conditions:

```json
["and", ["meets_req"], ["not", ["is", "disabled"]], <your conditions...>]
```

| Want | Term |
|---|---|
| Numeric threshold | `["cmp", "<field>", "<rel>", <number>]` — rel ∈ `le lt ge gt eq ne` |
| Boolean is true | `["is", "<bool_field>"]` |
| Boolean is false | `["not", ["is", "<bool_field>"]]` |
| Has a capability | `["has_cap", "supports_tools"]` (model serves it; e.g. `supports_json_mode`) |
| One model family | `["family_eq", "gpt-5.5"]` |
| Set of families | `["or", ["family_eq","gpt-5.5"], ["family_eq","kimi-k2.6"]]` |
| One provider | `["provider_eq", "openrouter"]` — route by *who serves*; set: `provider_in`, exclude: `not`/`provider_not_in` (e.g. drop a marketplace provider) |
| Tier exactly | `["tier_eq", "partner"]` |
| Tier at least | `["min_tier", "marketplace"]` (order `fallback < marketplace < partner`) |
| Specific seller/peer | `["served_by_eq", "<peer-id>"]` — executed route (marketplace peer, or provider for a direct route); set: `served_by_in`, exclude: `served_by_not_in` |
| **In the top N by a benchmark** | `["cmp", "<field>_rank", "le", N]` (e.g. `bench_intelligence_rank`) |
| Either of two | `["or", <predA>, <predB>]` |

> **Top-N is a `cmp` on a `_rank` field**, not a special op — the host
> precomputes catalog ranks (1 = best). The **intersection of two shortlists**
> ("top-5 on intelligence AND top-5 on coding") is just the `and` of two cmps.

### `<Scorer>` — rank the survivors (higher wins)

Score on the **raw observable fields** (the same names the filter gates on; see
*Field vocabulary*) via `["field", "<name>"]`, then weight and sum:

```json
["add", ["scale", 0.6, ["normalize", ["field", "bench_coding"]]],
        ["scale", 0.4, ["neg", ["normalize", ["field", "price_in"]]]]]
```

- `["field", "<name>"]` — a raw field's value: `price_in`, `price_out`,
  `latency_ms`, `tok_s`, `success_rate`, `context`, `bench_intelligence`, … (any
  Num field from the vocabulary).
- `["normalize", base]` — min-max the field across the live population to [0,1]
  (mix fields on different scales only after normalizing).
- `["neg", base]` — invert (`1 − base`), so "lower is better" (cheaper, faster
  latency) scores higher.
- `["scale", <weight>, base]` weights a term; `["add", …]` sums them.
- `["lit", <num>]` a constant; `["clamp", <lo>, <hi>, base]` bounds a score.
- No scoring (pure filter): `["zero"]`.
- Demote breaker-open instead of excluding: wrap the scorer in
  `["gate", ["not", ["is", "breaker_open"]], <scorer>]`.

> **Score on raw fields, not on composite atoms.** The signature also defines
> heuristic scorer atoms (`cost`, `speed`, `quality`, `partner`, `free_credit`)
> that fold fields + request knobs (`max_cost_usd`, `max_latency_ms`, token
> estimates) into one number with fixed host defaults (spec §5.2). They are
> opaque and host-tuned — author with the explicit `["field", …]` form above so
> the ranking is visible and portable.

### `<Selector>` — pick / cascade

- `["argmax"]` — deterministic best (the default; "subzero converges").
- `["top_k", N, ["argmax"]]` — keep the N best as the failover cascade.
- `["sample", <temp>]` — seeded, reproducible stochastic pick (rank-geometric;
  `temp=0` ≡ argmax, larger → more uniform). Used for greybox divergence.

## Worked examples (copy, adjust the numbers)

**Cheapest model that's decent and not over a price ceiling:**
```json
["policy",
  ["and", ["meets_req"], ["not", ["is", "disabled"]],
          ["cmp", "bench_intelligence", "ge", 0.5], ["cmp", "price_out", "le", 10]],
  ["neg", ["normalize", ["field", "price_out"]]],
  ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]
```

**Cheapest in the top-5 on intelligence ∩ top-5 on coding** (the host's Σ_pol
example #1):
```json
["policy",
  ["and", ["meets_req"], ["not", ["is", "disabled"]],
          ["cmp", "bench_intelligence_rank", "le", 5],
          ["cmp", "bench_coding_rank", "le", 5]],
  ["neg", ["normalize", ["field", "price_in"]]],
  ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]
```

**Top-3 by combined benchmarks, as a cascade** (example #2):
```json
["policy",
  ["and", ["meets_req"], ["not", ["is", "disabled"]]],
  ["add", ["scale", 1, ["normalize", ["field", "bench_intelligence"]]],
          ["scale", 1, ["normalize", ["field", "bench_coding"]]],
          ["scale", 1, ["normalize", ["field", "bench_agentic"]]]],
  ["top_k", 3, ["argmax"]], ["id"], ["always", {"action": "next_candidate"}]]
```

**Quality-leaning blend, partners only** (tier gated in the filter, so the score
is just benchmark vs latency):
```json
["policy",
  ["and", ["meets_req"], ["not", ["is", "disabled"]], ["tier_eq", "partner"]],
  ["add", ["scale", 0.7, ["normalize", ["field", "bench_intelligence"]]],
          ["scale", 0.3, ["neg", ["normalize", ["field", "latency_ms"]]]]],
  ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]
```

## Field vocabulary

A policy observes a candidate only through named **fields** (used by `cmp`,
`is`, `field`). The authoritative list — the core vocabulary (`core`, on every
conforming host) plus this host's registered extensions (`host`) — is injected
live below from the host's own schema (`GET /x/fields`), so it always matches
what the host actually serves rather than a copy that can drift.

<!-- FIELD_VOCABULARY -->

**Categorical** attributes, matched by their own ops (not in the table above):
`model_family` (`family_eq`) and `tier` (`tier_eq`, `min_tier`; order
`fallback < marketplace < partner`).

**Benchmarks** (`bench_*`, Num in 0–1) each have a `_rank` companion (1 = best)
for in-top-N gating: a missing benchmark reads as 0 and a missing `_rank` as
huge, so a family without it is correctly outside every top-N. Marketplace-only
families (no OpenRouter data) have empty benchmarks — gate on price/latency for
those.

Defaults when a field is absent are deliberately conservative (prices **+inf**
so a missing price fails a ceiling; `tok_s`/`credits` 0;
`success_rate` 1; bools false) — see *Rules* below.

## Σ_flow — composing several calls

A flow is `["flow", { <id>: <node>, ... }]` with exactly one `input` and one
`output` node; every `llm` node carries a `system` prompt, a `policy` (a full
Σ_pol term), and an `inputs` list of the node ids it consumes. It is a DAG
(acyclic), each node runs once. Edges are pull-model: `b.inputs = ["a"]` means
`a → b`. A node with two inputs is a fusion/synthesizer.

```json
["flow", {
  "u":   {"kind": "input"},
  "a":   {"kind": "llm", "system": "Answer concisely.",
          "policy": ["policy", ["and", ["meets_req"], ["not", ["is","disabled"]]],
                     ["field","bench_intelligence"], ["argmax"], ["id"], ["always", {"action":"next_candidate"}]],
          "inputs": ["u"]},
  "b":   {"kind": "llm", "system": "Answer rigorously, show steps.",
          "policy": ["policy", ["and", ["meets_req"], ["not", ["is","disabled"]]],
                     ["neg",["normalize",["field","price_in"]]], ["argmax"], ["id"], ["always", {"action":"next_candidate"}]],
          "inputs": ["u"]},
  "f":   {"kind": "llm", "system": "Synthesize the single best answer from the drafts.",
          "policy": ["policy", ["and", ["meets_req"], ["not", ["is","disabled"]]],
                     ["add",["scale",0.7,["field","bench_intelligence"]],["scale",0.3,["neg",["normalize",["field","price_in"]]]]], ["argmax"], ["id"], ["always", {"action":"next_candidate"}]],
          "inputs": ["a", "b"]},
  "out": {"kind": "output", "inputs": ["f"]}
}]
```

POST it as `flow_ir`. Optional per-node `template` with `$1,$2,…` overrides how
a multi-input node joins its predecessors' outputs.

## Rules that keep a policy valid

- **Defaults are conservative.** A candidate with no declared price does *not*
  pass a `price_out` ceiling (`price_in/out` default to +inf). Enforce spend
  with a hard `cmp` ceiling on `price_*` in the filter — a scorer only ranks
  softly, it does not bound anything.
- **Score on raw fields, not the composite scorer atoms.** Author scores as
  `["field", "<name>"]` (+ `normalize`/`neg`/`scale`/`add`). Don't use the bare
  `cost` / `speed` / `quality` / `partner` / `free_credit` scorer atoms: they
  bake request knobs and host defaults into one opaque number.
- **Always include `["meets_req"]` and `["not", ["is", "disabled"]]`** in the
  filter — the host's envelope ANDs its own floor on too, so you can only
  *narrow* what the host allows, never widen it.
- **Limits:** term depth ≤ 64, ≤ 4096 nodes; flow ≤ 256 nodes, in-degree ≤ 32.
- **Numbers** must be finite (no NaN/Inf); integers render without a decimal.
- You *can* target a specific seller — `["served_by_eq", "<peer>"]` pins the
  executed route (a marketplace peer, or the provider for a direct route), with
  `served_by_in` / `served_by_not_in` as the set sugar — but prefer gating on
  *properties* over identities: `["cmp", "reputation_score", "ge", 40]` or a
  `success_rate` weight keeps working as peers come and go, whereas a pinned peer
  id rots. Reach for `served_by_eq` for a trusted-peer allowlist, not as the
  default; for the rest, gate on *families and fields*.

---

<!-- LIVE_CATALOG_TABLE -->
*(The live model/provider catalog is injected here when this file is downloaded
from the host's **Catalog** tab. Without it, target the field vocabulary above
and confirm families with `POST /x/rank`.)*
