# Model lines

A policy that names an exact family goes stale the day its successor ships:

```json
["family_eq", "gpt-6.1-sol"]
```

Every caller that copied the term has to be edited and redeployed, and until
someone does, the fleet quietly runs last quarter's model. A **model line**
moves that one fact into the router. The policy names the line; the router
knows which family is its head today:

```json
["is", "line_openai_sol"]
```

The policy bytes, and so the policy id, never change when a line moves.

## What a line is

A line is a named pointer to the family currently **qualified** as the head of
a model series. It is declared in [`model_lines.json`](../model_lines.json):

```json
"openai_sol": {
  "description": "OpenAI general flagship (the -sol tier).",
  "head": "gpt-6.1-sol",
  "match": "^gpt-(\\d(?:\\.\\d+)?)-sol$"
}
```

- `head` is the **seed**: the family the line points at until someone moves it.
- `match` says which family names belong to the series. Its capture groups are
  the version: groups compare in order, each split on `.`/`-` into integers, so
  `gpt-6.2-sol` sorts after `gpt-6.1-sol` and a dated `…-flash-1201` after
  `…-flash-1017`. Tier siblings (`-pro`, `:batch`, vendor-prefixed aliases) do
  not match and are never candidates. Keep the pattern as tight as the real
  naming: marketplace catalogs carry look-alike aliases (`glm-52` for 5.2), and
  a loose `\d+` would rank one above `glm-5.3`. `null` means the line has no
  series to follow and only an operator moves it.

Every declared line becomes a model-group Bool observation field,
`line_<name>`, true for exactly the head family. It composes like any other
field:

```json
["policy",
  ["and", ["meets_req"], ["not", ["is", "disabled"]],
          ["or", ["is", "line_anthropic_sonnet"], ["is", "line_openai_sol"]]],
  ["add", ["scale", 100, ["gate", ["is", "line_anthropic_sonnet"], ["lit", 1]]],
          ["scale", 10,  ["gate", ["is", "line_openai_sol"], ["lit", 1]]],
          ["scale", 0.45, ["neg", ["normalize", ["field", "price_out"]]]]],
  ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]
```

The set of lines is static on purpose. A policy naming an undeclared line fails
admission (`undeclared field 'line_x'`) instead of resolving to "no candidate".

This uses the `fields.lua` extension seam, like `cache_hot`: the engine observes
a Bool and learns nothing about lines. No engine change, no signature change.
Each engine holds the heads as a native table, so the field costs one lookup per
candidate — the same order as `family_eq`.

## Where the head lives

Which family a line points at **now** is operator state in the host store
(`model_line_heads`), with an append-only audit trail (`model_line_events`). A
line with no row serves its seed. Every router replica re-reads the heads on a
short timer (`MODEL_LINES_REFRESH_S`, default 30; `0` pins the boot-time heads),
so a move reaches the whole pool without a reload fan-out or a restart.

Fail-safe directions:

- Store unreachable: replicas keep their last-known heads. They never snap back
  to the seeds mid-outage.
- Definitions file missing: no lines are declared. Invalid: no lines are
  declared and the reason is logged. Either way a policy naming a line is
  refused loudly rather than served from a half-read set.
- A stored head for an undeclared line, or one that is not a plausible family
  name, is ignored.

`MODEL_LINES_FILE` points the router at a different definitions file.

## Moving a head

```
GET  /x/lines                      every line, head, seed, source, recent events
POST /x/lines/{line}/promote       {"family", "expected_current"?, "force"?, "detail"?}
POST /x/lines/{line}/rollback      {"expected_current"?, "expected_previous"?, "detail"?}
```

These are operator surfaces behind the router's internal secret. The dashboard
exposes the same three to an authenticated admin as `/dashboard/api/lines…` and
stamps the audit actor from the session, never from the request body.

Both moves are compare-and-set when you say what you saw. `expected_current`
refuses the move with 409 if the line no longer points there. Send it on a
rollback too: the store picks the rollback target inside its own transaction,
and a rollback retried after a lost response then conflicts instead of undoing
itself. If the dashboard answers 504 `model_line_outcome_unknown`, the request
reached the router and may have been applied: read `/dashboard/api/lines`
before retrying. A family outside the line's own pattern is refused unless
`force` is set.

## Qualification

"Newer" is not "better for this workload". A successor can refuse a prompt the
head answers, blow the caller's deadline, or cost several times more through the
routes that actually serve it. So a line never moves on a version number alone.
[`scripts/qualify_model_lines.py`](../scripts/qualify_model_lines.py) is meant to
run on a schedule, outside the request-serving pods:

1. Probe every line's current head with the prompt suites. This is the baseline
   and a health check of what is live.
2. Find successors by the line's pattern and probe the newest few
   (`max_candidates`) the same way: one family, no fallback, the caller's own
   deadlines. Prompts within a family run sequentially, so the qualifier's own
   burst is never what trips a provider's rate limit.
3. Gate each candidate against the head, on the same prompts in the same run:

   | Gate | Passes when | Knob |
   |---|---|---|
   | answers | share of valid answers inside the deadlines ≥ the overall bar, **and** ≥ the per-suite bar in every suite | `min_pass_rate` (0.9), `min_suite_pass_rate` (0.5) |
   | latency | p90 of successful probes ≤ the absolute bound, **or** ≤ head's p90 × ratio | `max_latency_ms` (15000), `max_latency_ratio` (1.25) |
   | cost | measured cost ≤ head's × ratio, summed over the probes both reported | `max_cost_ratio` (1.1) |

   - Cost is what the router **measured** for those calls, not a list price:
     marketplace routes make list prices a poor predictor of what a call costs.
     It is compared over the same probes on both sides, and a probe that
     returned an unusable answer still counts what it charged.
   - Every gate fails closed. An unknown cost, or too few probes with a cost on
     both sides, qualifies nobody.
   - A head that is merely degraded still serves as the baseline: it is the one
     most worth replacing.
   - `min_pass_rate` is also the bar a head must clear to count as healthy. It is
     below 1 because one transient provider error in a run is noise;
     `consecutive_passes` is what guards against a lucky run, and the per-suite
     bar is what stops a model that fails one whole workload from hiding behind
     the others. Give a one-prompt suite `repeats` if a single miss should not
     decide it.
4. A candidate that qualified on `consecutive_passes` runs in a row (2), all of
   them against the head that is current now, is promoted or proposed, newest
   first. A pass measured against an earlier head does not count.
5. A head that stops passing is always recorded. It is rolled back only in
   `auto`, only if the move that made it current is this qualifier's own
   promotion, and only if the head it replaced still passes. A head an operator
   chose is never moved automatically, and the qualifier never undoes its own
   rollback.
6. A probe stopped by the ingress or the network — the qualifier's own rate
   limit or budget, an overloaded or unreachable router — is **inconclusive**,
   not failed. A family with any such probe is not judged on that run: a
   candidate keeps its streak, and a line whose head was affected is left
   alone.

The run stops with a non-zero exit if the store is unreachable or an audit
event cannot be written: the streak a promotion rests on is read from those
events.

The qualifier reads the same definitions file as the router and writes to the
same store. Run it from the router's image with the same `MODEL_LINES_FILE` (or
none on both). A job with different knobs would promote into the shared store
by its own rules.

Modes (`--mode` / `LINE_QUALIFIER_MODE`):

| Mode | Probes | Audit events | Moves heads |
|---|---|---|---|
| `report` | yes | no | no |
| `propose` (default) | yes | yes, plus one standing `proposed` event | no |
| `auto` | yes | yes | yes |

Start in `propose`: read the proposals in `/x/lines`, approve with a promote,
and switch to `auto` once the proposals match what you would have done by hand.

Probes go through the public ingress with a consumer key
(`LINE_QUALIFIER_API_KEY`), so their spend lands in the call ledger under that
key like any other caller's. Give the key its own budget.

### Prompt suites

A suite is the workload a line has to keep serving. It is a JSON file; pass
files or directories with `--suites`:

```json
{
  "name": "support-triage",
  "request": {"temperature": 0.7, "max_tokens": 4000,
              "response_format": {"type": "json_object"},
              "first_token_timeout_ms": 7000, "timeout_ms": 22000},
  "expect": {"type": "json", "required": {
      "label":      {"enum": ["bug", "billing", "other"]},
      "confidence": {"type": "integer", "min": 0, "max": 100}}},
  "repeats": 1,
  "prompts": ["…", "…"]
}
```

- `request` carries what the real caller sends, deadlines included. A candidate
  that only answers given more time than production allows has not qualified.
- `expect` is `text` (non-empty), `regex`, or `json` with `required` keys
  (`enum`, or `type` with optional `min`/`max`). JSON is read from the first `{`
  to the last `}`, the way lenient callers accept a fenced answer.
- A candidate must pass every suite. Use the caller's real prompt templates;
  keep user-generated content out of the files.

[`evaluations/line_suites/example.json`](../evaluations/line_suites/example.json)
is a runnable sample.

## Limits

- The series pattern is a heuristic for *what to try*, never for what to serve.
  An odd naming scheme costs a wasted probe, not a bad promotion.
- Suites are a sample of the workload. A regression outside them is caught by
  the live reliability fields a policy already gates on (`success_rate`,
  breakers) and by the next run's head check, not prevented.
- One head per line, for every caller. Two workloads that need different heads
  of the same series are two lines.
