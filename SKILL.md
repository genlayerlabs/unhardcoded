---
name: sigma-policy-author
description: >-
  Create and debug Unhardcoded router policies and flows. Use when an agent
  needs to turn workload, cost, latency, reliability or provider constraints
  into policy_ir or flow_ir, preview eligible routes, and verify real routing.
---

# Create a routing policy

Use this guide for the normal Unhardcoded router. Download it with the same
consumer key used for inference: `GET /skill`. The dashboard Skills tab serves
it with an admin session. For experiments on an existing policy, fetch
`GET /skill?name=policy-optimization`. These guides do not change Cloud tenant
keys' published-route API contract.

A `policy_ir` is JSON data: a hard filter, a score, a selector, request
transforms and a failure plan. The router evaluates it against current offers.
A `flow_ir` composes several chat calls as a DAG. Neither is executable code.

## 1. Turn intent into a contract

Reuse requirements already given by the user. Ask only for missing choices that
change the result: workload and output type, acceptable quality, permitted
providers/data handling, latency target, price or spend budget, and whether
model versions may change. State assumptions for everything else.

- **Hard constraints belong in the filter:** provider/peer allowlists, exact
  model families, capabilities, context capacity and token-price ceilings.
  Ranking preferences cannot enforce them. Every fallback must satisfy them.
- **Optimize within those constraints:** expected input/output cost, observed
  latency and reliability. Generic model benchmarks are starting signals, not
  evidence of task correctness. A ticket classifier needs labelled tickets;
  a coding agent needs passing task tests.
- **Sensitive data:** allow only explicitly acceptable routes (for example
  `provider_eq` on a configured Bedrock provider). A provider name or tier alone
  proves neither region nor retention/compliance. Check the deployment's
  configuration; apply the same data restrictions to Jev selectors, evaluators,
  summarizers and all flow nodes. Fail closed if no compliant candidate exists.

## 2. Discover and compile without inference spend

All endpoints in this section accept `Authorization: Bearer <consumer-key>`.
Keep credentials in the environment; never put them in a policy or report.

1. Fetch `GET /x/fields`, `GET /x/policy/templates` and `GET /v1/models`.
   Use `GET /v1/models?type=decisions` for Jev. The catalog embedded below is a
   download-time snapshot; refresh before testing. Do not invent field names.
2. Prefer a template from `POST /x/policy/templates/{id}`:

   | Template | Use | Options |
   |---|---|---|
   | `default` | General chat baseline | `{}`; matches the default chat policy |
   | `agent` | Tool agents with context, trust gates and bounded fallbacks | `{}`; also available as `model: "profile:agent"` |
   | `cheapest-family` | Compare providers for one exact family | required `family`; `provider_strategy`: `cost` or `ordered`; optional `provider_order` |
   | `smart-value` | Cost preference within an intelligence shortlist | optional `top_n` |

   For `cheapest-family` and `smart-value`, optional `expected_input_share`
   (0–1), `reliability_floor` (0–1), `max_price_in` and `max_price_out` control
   the tradeoff. Read returned `intent` and actual `policy_ir`; defaults can
   change. Price limits are USD per million tokens, not per-call spend limits.
   `agent` and `default` do not accept overrides. Chat benchmark gates may
   exclude Jev entirely; use a decision-specific policy for decision models.
3. For custom constraints, edit the returned term or author one below.
   `POST /x/policy/normalize` with `{"policy_ir": term}` returns the normalized
   term, `fingerprint` and `version`. This identifies the term; **it does not
   replace live admission**.
4. Send `POST /x/rank` with `policy_ir`, `protocol` (`chat` or `decisions`) and
   `requirements`. Include actual family, context and capability constraints;
   for example `requirements: {"min_context": 32000, "needs": ["tools"]}`.
   Response: `ranked`, `rejected`, `ts`. Fix admission errors before inference.
   Inspect rejected reasons and the whole fallback list, not only rank one.

An empty ranking is a blocked preflight, not a reason to weaken security or
silently substitute a model. Check protocol, exact family vs mutable alias,
capabilities, missing metrics, price ceilings and the host envelope. Discovery
may suppress an offer before ranking, so it need not appear in `rejected`.
Wallet/reputation/cooldown/concurrency gates can prevent advertised free
AntSeed offers from entering the catalog. A policy cannot override host gates.

## 3. Make a bounded real call and inspect the result

A chat example (save the complete object as `request.json`):

```json
{
  "model": "",
  "messages": [{"role": "user", "content": "Reply with the integer: 17 times 23."}],
  "max_tokens": 32,
  "policy_ir": ["policy",
    ["and", ["meets_req"], ["not", ["is", "disabled"]],
      ["cmp", "price_in", "le", 1], ["cmp", "price_out", "le", 5]],
    ["neg", ["normalize", ["field", "price_out"]]],
    ["top_k", 3, ["argmax"]], ["id"],
    ["always", {"action": "next_candidate"}]]
}
```

This is a routing smoke test, not a validated quality policy. After previewing:

```bash
curl --fail-with-body "$ROUTER_URL/v1/chat/completions" \
  -H "Authorization: Bearer $ROUTER_API_KEY" \
  -H 'Content-Type: application/json' --data-binary @request.json
```

Use an empty `model` when the custom policy owns selection. Do not assume a
family or provider pin is ignored when a policy is supplied; request constraints
still matter. Session affinity is not a security boundary or guaranteed pin.

**Decision models (Jev)** use `POST /v1/decisions` (alias `/v1/systemone`), with
`state` and typed `questions`, never chat messages or streaming. A minimal
choice call with a version-constrained, price-bounded Jev cascade:

```json
{
  "state": {"ticket": "I was charged twice."},
  "questions": {"team": {
    "type": "choice", "instructions": "Which team handles this ticket?",
    "criteria": {"billing": "Payments and refunds", "technical": "Software issues"}
  }},
  "policy_ir": ["policy",
    ["and", ["meets_req"], ["not", ["is", "disabled"]],
      ["family_eq", "jev-1.13"],
      ["cmp", "price_in", "le", 0.1], ["cmp", "price_out", "le", 0.1]],
    ["neg", ["normalize", ["field", "price_in"]]],
    ["top_k", 4, ["prefer", ["not", ["is", "breaker_open"]], ["argmax"]]],
    ["set_param", "timeout_ms", 2500], ["always", {"action": "next_candidate"}]]
}
```

Preview that term with `protocol: "decisions"`; chat is the rank default.
Discover actual family names: `jev-latest` is mutable and is not proof of
`jev-1.13`. Add it to an `or` family filter only when version flexibility is
acceptable. Prefer advertised zero-price routes using `prefer` inside the
healthy group, rather than filtering out all paid fallbacks. Validate actual
billing; zero token prices do not establish the absence of other fees.

Decision requests allow `choice`, `score` and `noul`, up to 32 KB, 32 questions
and 32 criteria per question. Typed answers do not guarantee correct decisions.
`first_token_timeout_ms` bounds the complete nonstreamed decision response;
`timeout_ms` bounds an attempt, not the whole multi-attempt request. Leave room
for fallbacks within the server's overall request deadline.

Read the actual response, including errors:

| Evidence | Meaning |
|---|---|
| `x_router.provider`, `served_model_id`, `served_by` | Executed provider, model and actual peer/direct route |
| `x_router.decision_trace.decision_path` | Attempts, failures and fallback outcomes; inspect rather than infer from preview |
| `x_router.cost_usd`, `cost_basis`, `usage` | Reported/estimated winning-attempt cost and tokens; failed-attempt charges may be unknown |
| `x_router.price_in`, `price_out` | USD per million tokens; not the total bill |
| `x_router.policy_fingerprint` | Effective policy identity; host envelope can make this differ from normalization |
| Client elapsed time + checked output | End-to-end latency and task correctness; HTTP 200 alone proves neither quality nor a latency target |

Use bounded test calls within the user's existing authorization and budget.
For streaming, do not replay visible partial output on a fallback. For sustained
optimization, fetch `/skill?name=policy-optimization`. Return the policy JSON,
requirements/assumptions, preview, measured results and unresolved limitations.

## The Σ_pol term, exactly

> The operators below are the `sigma-pol/v2` signature. The **normative
> grammar is the core spec** (`core/docs/SIGMA-POL.md`) — this guide mirrors it
> for authoring convenience; on a major version bump, regenerate against the
> spec. The **field vocabulary**, by contrast, is injected live from the host
> (see *Field vocabulary* below), so it never drifts from what the host serves.

A policy is a 6-element array — the last two slots can set request parameters and failure behavior:

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
- `["prefer", <Pred>, <Selector>]` — strict stable priority: every matching
  candidate precedes every non-match, while the inner selector still orders
  candidates inside both groups. Nest it for lexicographic provider order; an
  outer `prefer(not(is("breaker_open")), ...)` keeps unhealthy routes last.

## Example: provider restriction with explicit cost tradeoff

This example allows only `bedrock`. Replace the identifier only with approved,
live provider IDs. It intentionally fails if none qualifies. Weight input/output
prices according to your workload; this 80/20 split is an example, not a measured
optimum. Preview it and qualify task quality before adoption.

```json
["policy",
  ["and", ["meets_req"], ["not", ["is", "disabled"]],
    ["provider_eq", "bedrock"],
    ["cmp", "price_in", "le", 5], ["cmp", "price_out", "le", 25]],
  ["neg", ["normalize", ["add", ["scale", 0.8, ["field", "price_in"]],
                                ["scale", 0.2, ["field", "price_out"]]]]],
  ["top_k", 3, ["prefer", ["not", ["is", "breaker_open"]], ["argmax"]]],
  ["id"], ["always", {"action": "next_candidate"}]]
```

Normalize the combined token-price estimate when preserving the input/output
ratio matters. Normalizing each price separately before adding them expresses
relative preferences and can change that ratio as the catalog changes.

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

Missing values are not measurements: prices and latency default to **+inf**;
throughput, credits and context to 0; booleans to false. `success_rate` defaults
to **1**, optimistically, so a new unmeasured route may pass a reliability floor.
`quality` and `quality_hint` are not observable fields in this schema.

On this host, route success and mean successful latency use a recent observation
window; they do not measure task correctness or p95. AntSeed reputation is an
additional signal, not an SLA. Advertised concurrency is a capacity limit, not
currently idle slots. Do not invent p95, sample-count or confidence fields when
`/x/fields` does not expose them. Measure these in your evaluation report.

## Σ_flow — composing several calls

Use chat flows only when multiple billable calls are justified. Apply the same
provider/data constraints to every node, including the synthesizer.

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

- **Check missing-value behavior.** A candidate with no declared price does *not*
  pass a `price_out` ceiling (`price_in/out` default to +inf). Bound token
  rates with `cmp` on `price_*`; bound output length and experiment spend
  separately. Include retries, evaluator/selector calls and unknown failed-call
  charges in the budget. A score does not enforce a ceiling.
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
from the host's **Skills** tab or `/skill`. Without it, target the field vocabulary above
and confirm families with `POST /x/rank`.)*
