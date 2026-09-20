# Jev-only provider policy qualification — 2026-09-20

Historical baseline, before the decision-routing PR. See ../DECISION-MODELS.md
for the new implementation. Production was not changed by this work.

**Not deployable on the current public router.** The policy is admitted by the
live router and its selection/fallback behavior passes real-engine tests, but
there are no Jev candidates in that router. A real authenticated inference
request returns `503 no_candidates`. This is not a successful live policy test.

## Policy artifact and intended behavior

`policies/jev-value-v1.json` is request-scoped and does not modify
profiles, stored flows, consumer keys or provider configuration.

- Require exact model family `jev-1.13` throughout the cascade. No generic LLM
  fallback. Do not treat the mutable `jev-latest` alias as an independent backend.
- Enforce request capabilities, enabled providers, known nonnegative prices,
  maximum $0.10/M input and output, and at least 0.90 observed success rate.
  The engine defaults an unmeasured success rate to 1; this is not a claim of
  measured reliability. Unknown latency is kept behind measured fast routes.
- Prefer healthy routes; within them prefer observed latency <= 1,500 ms.
- Inside each preference group, rank 70% input-price utility + 30% latency
  utility. Utilities use fixed, clamped scales ($0.10/M and 1,500 ms), so unrelated
  expensive/slow models cannot distort the score. Jev's current OpenRouter
  output price is zero. This is an explicit tradeoff, not proof of a universally
  optimal provider.
- At most four candidates; 1,500 ms first-token and 2,500 ms per-attempt limits;
  failures advance to another eligible Jev candidate. The qualification request
  also has a 7,000 ms overall deadline.
- Existing caller keys and flows are not altered. No deployment was performed.

## Actual public-router test

Base URL `https://router.ygr.ai`, existing operational consumer key retrieved
from AWS SSM in memory. No credential is included in these files.

| Check | Result |
|---|---|
| GET /v1/models | 200; 660 entries; no Jev |
| POST /x/policy/normalize | 200; valid sigma-pol/v2 |
| POST /x/rank | 200; zero candidates; no Jev among 789 rejected entries |
| POST /v1/chat/completions with policy | 503, `no_candidates`; no upstream attempt |

The live deployments still run `v-a9af462`. A read-only query of the AntSeed
`peer_offers` table found zero service names matching Jev, including retained
historical entries. Main `8f72e44` has the separate Cloud decision adapter, but
upgrading alone would not add Jev to the ordinary router execution path.

Raw bounded report: `jev-live-router-20260920.json`.

## Actual OpenRouter upstream test

Used the router's configured OpenRouter credential from SSM, only with synthetic
support-ticket text. These calls went directly to OpenRouter; they did **not**
pass through the public router or exercise its policy.

- `/api/v1/chat/completions` rejects Jev with HTTP 400 and explicitly requires
  `/api/alpha/decisions`.
- Four calls using the existing `JevDecisionProvider` adapter all succeeded and
  chose the expected department. This small smoke test is not a quality benchmark.
- Observed wall-clock adapter latency: 338.8–616.2 ms; median 402.4 ms, from the
  local test runner. Total reported cost of four calls: $0.000059766.
- Returned model: `typesafe/jev-1.13-20260917`.
- Default `/api/v1/models` has no Jev. Querying it with
  `output_modalities=decisions` returns the pinned model and its latest alias.
- The pinned model's public endpoints list contains one backend, TypeSafe,
  priced at $0.042/M input and $0/M output. The alias is not an independent route.

Raw report: `jev-live-openrouter-20260920.json`.

## Automated qualification and remaining prerequisites

39 targeted tests passed: the new seven policy tests plus the existing Jev
adapter and automatic-policy catalog tests. Policy tests use the real Lua engine
and simulated provider responses. They cover admission, price/latency ordering,
unknown/expensive/unreliable/non-Jev exclusions, breaker priority, Jev-only
fallback success, exhaustion, and failure when Jev is absent.

Before calling this policy production-ready:

1. Wire decision-model discovery and the Decisions request/response transport
   into the normal router, with auth/admission/metering and protocol isolation.
   Merely adding the model to the chat catalog would produce upstream 400s.
2. Identify and qualify another real Jev route (another provider or serving
   endpoint). There is currently no AntSeed Jev offer in this deployment and no
   independent second backend in the observed OpenRouter endpoint metadata.
3. Repeat successful consumer-key inference and real provider failover after
   the reviewed integration is deployed. Do not change existing flows to Jev
   merely because the policy normalizes successfully.

Sources checked: https://openrouter.ai/api/v1/models?output_modalities=decisions,
https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints,
https://docs.typesafe.ai/api.
