# Jev decision transport

The host's `decision_providers` interface accepts only named choices and returns
a typed selection. The Jev adapter uses a separate Decisions API, not the normal
provider chat path; it cannot recursively route itself. It is inert unless a
caller explicitly constructs and invokes it.

Verified schema references (2026-09-19):
- [TypeSafe HTTP API](https://docs.typesafe.ai/api): `/v1/systemone`, `state`,
  `questions`, Choice criteria, typed answers and token usage.
- [OpenRouter Decisions SDK](https://openrouter.ai/docs/client-sdks/typescript/sdks/decisions/README):
  `alpha.decisions`, `typesafe/jev-1.13`, state/questions request.
- [Confidence semantics](https://docs.typesafe.ai/confidence): confidence is a
  distribution statistic, not the probability that an individual answer is right.

`tests/test_jev_decision_provider.py` locks the documented wire contract against
a mock HTTP transport. These are contract fixtures, not evidence of a live paid
call. Activation requires a credentialed check and workload calibration.

Endpoints are fixed HTTPS origins. Credentials must be supplied explicitly from
the authorized environment connection; the adapter never reads operator env.
Requests and responses have byte bounds; redirects, duplicate JSON properties,
invalid/unknown choices and malformed distributions are rejected. Missing
probabilities remain missing; confidence and entropy thresholds belong to the
application. The caller supplies an absolute monotonic deadline. Task cancellation
propagates, and at most one explicitly enabled retry shares the same deadline.

Usage costs are provider-reported, not invented. Missing cost or an unmetered
retry yields unknown total cost. Invalid answers retain valid reported usage.
Timeouts can still be billable upstream. No raw state, response or credentials
are logged. This adapter alone does not authorize a choice or reserve a budget.
