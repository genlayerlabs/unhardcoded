---
name: policy-optimization
description: >-
  Improve an existing Unhardcoded routing policy through bounded experiments.
  Use to compare a baseline against candidate policies, diagnose route selection
  or fallback failures, and reduce cost or latency while preserving task quality
  and provider constraints. Produces measured results and a versioned candidate.
---

# Optimize a routing policy

Work against the normal router that issued the consumer key. Fetch this skill
from `GET /skill?name=policy-optimization`; fetch the grammar and authoring
examples from `GET /skill`, using `Authorization: Bearer <consumer-key>` for
both. Dashboard downloads use an admin session. Do not assume these endpoints
are part of Cloud tenant keys' published-route contract.

The outcome is a candidate supported by measurements, or an explicit finding
that the baseline should remain. A lower ranking score or HTTP 200 alone cannot
establish a better policy. The agent runs this workflow; downloading this skill
does not start a background optimizer or change any published route.

## 1. Define success before changing the policy

Read the baseline JSON and existing user requirements. Record:

- Protocol (`chat` or `decisions`), request shape, capabilities/context, model
  version flexibility, provider/peer allowlists and data-handling restrictions.
- Task metric and minimum acceptable quality: labelled classification accuracy,
  passing coding tasks, retrieval relevance, schema plus semantic checks, etc.
  Use asymmetric error costs when relevant (for example unsafe tool approvals).
- Objective: cost per accepted task, end-to-end latency, or reliability, with
  a quality floor. State how ties and acceptable regressions will be decided.
- A finite experiment budget: maximum calls, USD, elapsed time, variants and
  retries. Include baseline, fallback attempts, Jev selection and evaluation
  calls. A per-million-token price ceiling is not a total spend cap.

Reuse existing authorization and budgets. Ask only for missing requirements
that prevent a meaningful or authorized comparison; continue no-spend inspection
while waiting. Do not invent a quality threshold and later present it as agreed.
If no labels or defensible checks exist, report routing/performance results
separately and mark quality unvalidated.

Save the baseline, timestamp, normalized fingerprint, request parameters and
initial live rank snapshot. Never edit the only copy of a working policy.

## 2. Preflight without inference spend

1. Refresh `GET /x/fields`, `GET /v1/models` (add `?type=decisions` for Jev) and,
   if useful, `GET /x/policy/templates`. A field in the schema does not imply
   observations for every route; the embedded catalog is only a snapshot.
2. `POST /x/policy/normalize` with `{"policy_ir": baseline}` identifies the
   term. Live admission happens at `POST /x/rank`, with that term and the
   intended `protocol` and `requirements`.
3. Inspect every eligible fallback and `rejected` reason. Match runtime context,
   tools and model constraints. Do not compare a broad chat preview with a
   restricted decision request. No surviving candidates means no live experiment.

For example, this complete preview explores an exact Jev family and token-price
ceiling, with no inference and no claim of measured quality:

```json
{
  "protocol": "decisions",
  "policy_ir": ["policy",
    ["and", ["meets_req"], ["not", ["is", "disabled"]],
      ["family_eq", "jev-1.13"],
      ["cmp", "price_in", "le", 0.1], ["cmp", "price_out", "le", 0.1]],
    ["neg", ["normalize", ["field", "price_in"]]],
    ["top_k", 4, ["prefer", ["not", ["is", "breaker_open"]], ["argmax"]]],
    ["set_param", "timeout_ms", 2500], ["always", {"action": "next_candidate"}]]
}
```

Save it as `preview.json` and send:

```bash
curl --fail-with-body "$ROUTER_URL/x/rank" \
  -H "Authorization: Bearer $ROUTER_API_KEY" \
  -H 'Content-Type: application/json' --data-binary @preview.json
```

An advertised offer may be missing before admission because discovery,
funding, reputation, reachability or cooldown checks suppressed it. A mutable
`jev-latest` alias is not an exact version. Separate these explanations from
"the policy score was too low". Do not lower global wallet/trust gates, top up,
change provider credentials or relax data restrictions to force a test through.
Raise an operator finding when the host prevents the intended experiment.

With dashboard admin access, `POST /dashboard/api/policy/backtest` accepts
`policy_ir` and a `timeframe` such as `7d`. It reprices historical family groups
against the current ranking and observed token counts. It does not rerun tasks,
reconstruct historical availability or measure counterfactual answer quality.
Report excluded/unroutable groups and missing observations; this endpoint is
not available through a consumer bearer key.

## 3. Establish a baseline and test hypotheses

Use representative cases plus edge cases and a held-out set. Keep request
parameters, output limits, evaluation rubric and input distribution comparable.
For confidential workloads use approved data and providers; the same constraint
applies to any external evaluator or Jev selector.

Start with a small bounded smoke sample, then spend the remaining authorized
budget on paired comparisons. Interleave baseline and candidate requests so
provider load and catalog changes do not systematically favor one. Record
order, time and catalog drift. Repeat noisy cases as the budget permits.
Avoid shared session affinity accidentally forcing both variants onto one route;
keep cache/session treatment equivalent and report warm/cold conditions.

Generate a small number of candidates, each with a specific hypothesis:

| Hypothesis | Candidate change | Evidence required |
|---|---|---|
| Output tokens dominate spend | Adjust input/output price weights to measured usage | Lower cost per accepted task, including output-length changes |
| A route is cheap but often fails | Prefer reliable/healthy routes or exclude a demonstrated bad route | Actual failed attempts, sample count and fallback time |
| Fast routes are hidden by unknown latency | Prefer observed-fast routes while retaining eligible unknown fallbacks | Client latency and adequate successful observations |
| A free decision peer is usable | Prefer zero input/output token prices inside the healthy group | Eligible typed route, actual peer, response and billing; paid fallback retained if allowed |
| Quality shortlist is too broad/narrow | Change benchmark gate or approved family set | Task evaluation and held-out results, not generic benchmark rank alone |
| Retry budget wastes the deadline | Adjust request transforms/failure plan | Recovery rate and full-request latency under representative failures |

Preserve hard requirements in every variant. For price ratios, weight raw input
and output prices before normalizing their sum; separately normalized fields
can change relative economics. `neg` means `1 - score`, not unary minus.
`prefer` imposes strict group priority, so it can dominate a weighted score.
Normalize and preview each candidate before live testing.

## 4. Measure actual execution

Send the same test cases to `POST /v1/chat/completions` or, for Jev,
`POST /v1/decisions` with `state`, `questions` and `policy_ir`. Do not send the
preview wrapper (`protocol`, `requirements`) as a decision inference payload.
Use the authoring guide for complete request examples. Jev supports `choice`,
`score` and `noul`; typed probabilities still need calibration against labels.

Keep a sanitized record per case and variant:

| Record | Source |
|---|---|
| Case ID, variant, timestamps, fixed request settings | Evaluation harness; avoid credentials and unnecessary raw private content |
| Normalized and effective policy IDs | Normalize result and `x_router.policy_fingerprint`; the host envelope may change the latter |
| Actual provider/model/peer | `x_router.provider`, `served_model_id`, `served_by` |
| Attempts and errors | `x_router.decision_trace.decision_path`, HTTP/body errors and client exceptions |
| Tokens, cost, cost basis | `usage`, `x_router.cost_usd`, `cost_basis`; distinguish measured, estimated and unknown |
| End-to-end latency | Client monotonic clock; record timeout/failure latency too |
| Task score and accepted/rejected | Fixed rubric, labels or executable checks, with evaluator version |

`cost_usd` may cover only the winning attempt. Failed-attempt charges are not
necessarily zero. Count known selector/evaluator costs and disclose unknown
costs before claiming savings. Stop before the next batch exceeds the budget;
if charges cannot be bounded, stop spending and report the uncertainty.

Unknown `success_rate` defaults to 1; unknown latency to infinity. These are
engine defaults, not evidence of perfect reliability or infinite measured
latency. Host mean successful latency excludes failed calls and is not p95.
AntSeed reputation and advertised capacity are not task quality or availability
SLAs. Do not invent policy fields for sample count, p95 or confidence intervals;
calculate them in the evaluation report from actual observations.

Use controlled local/staging failures to exercise fallback ordering. Distinguish
simulated fallback coverage from observed live failover. Do not induce production
outages to test recovery. Preserve all provider/model/data constraints through
the cascade; never replay already-visible streamed output as a fresh answer.
A decision model does not stream, so its first-token timeout bounds the complete
response. Per-attempt timeouts do not equal a total cascade deadline.

## 5. Compare, stop and deliver

Compare candidates on the same covered cases. Include failures, timeouts,
unroutable cases and rejected outputs; dropping them biases savings and latency.
Report sample counts, quality and error rate, total known cost, cost per accepted
task, and latency distribution. Report uncertainty; a handful of successes
cannot establish a reliable p95 or production-wide improvement. If no tasks
were accepted, cost per accepted task is undefined, not zero.

Keep candidates that satisfy the agreed quality/security constraints and improve
the objective. Validate the selected candidate on held-out cases before claiming
a win. If differences are noisy, coverage is missing or the budget runs out,
return an inconclusive result and retain the baseline.

Deliver:

- Versioned baseline and candidate JSON, fingerprints and a concise semantic diff.
- Dataset/case IDs, evaluation criteria, sample counts, time window and results
  table, including failed/unknown costs and provider/peer distribution.
- Verified fallback behavior, coverage gaps and a recommendation with evidence.
- If rollout is in scope, concrete promotion/rollback criteria using the saved
  baseline. Publishing, changing a key's route or deploying follows the user's
  existing authorization; preparing an experiment does not authorize those actions.

Rerun qualification when a mutable model alias changes, a new provider becomes
eligible or task performance drifts. Dynamic catalog ranking adapts selection;
it does not automatically prove that new models improve this workload.

## Live field vocabulary

<!-- FIELD_VOCABULARY -->

<!-- LIVE_CATALOG_TABLE -->
