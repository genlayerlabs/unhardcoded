# Automatic policy catalog

`auto_policies.py` owns `auto/v1`: twelve workload policies, descriptions and
examples. They compile to ordinary sigma-pol/v2 terms and are admitted and
identified by the existing Lua engine. They do not make network calls.

The initial catalog covers general assistance, conversation, classification,
structured extraction, translation, summarization, long documents, coding,
coding agents, reasoning, creative writing and vision. Different quality/cost
weights and capability requirements express execution preferences, not claims
of measured task quality. Benchmark intelligence is a model observation; it is
not a guarantee that a model solves a particular task.

Every compiled policy retains request requirements, disabled-provider exclusion,
reliability and token-price limits. Unknown prices fail closed. Open breakers
remain last-resort fallbacks. Optional exact model restrictions are hard filters;
otherwise new live candidates may qualify without changing the policy bytes.
Timeout/refusal/stream failure behavior uses the existing sequence algebra.

Published contracts must store the normalized terms, engine identities, catalog
version and descriptions together. Existing revisions must never resolve a
policy by an unversioned mutable name. General is a mandatory fallback; if it
has no eligible candidate at execution time the request fails, never widens.

Token-price ceilings and the output-token cap are not a monthly spending quota.
The output-token cap lives in the frozen host constraints, not the policy term:
the host clamps the caller's limit before execution so a lower caller limit is
preserved. A constant Lua set_param would replace that lower limit. The bundle's
engine_version identifies the IR format; its fingerprint is a diagnostic cache
key, while policy_id is the canonical SHA-256 identity.
Accounting, admission controls and request requirements remain host concerns.
No production behavior changes merely by importing this catalog.

## Cloud runtime

`AUTOMATIC_ROUTING_ENABLED=1` enables automatic contract compilation and decision
calls on the dedicated Cloud data plane. It defaults off. Legacy non-tenant
requests never acquire automatic context. Cloud additionally supplies its live
environment mode on each route resolution. Off and unsampled requests use the
frozen general fallback without contacting Jev; shadow pays for a decision but
keeps that same fallback execution. Active selects only a currently eligible
published policy. Sampling hashes the tenant/environment, configuration and
session (or a fresh request identifier); it never uses process-randomized hashes.

Both API surfaces resolve `model="auto"` (or an omitted model) through the actual
key's default route. Explicit route aliases still undergo the same binding
check. The ingress overwrites internal automatic metadata, policies and timeout
overrides before forwarding. Selection runs inside the request-local tenant
host, after ingress rate/budget/capacity admission, with credentials fetched for
that environment. An assigned OpenRouter key is required; operator keys are
never a fallback.

The twelve frozen terms are admitted and ranked against the actual request
before sending any choices. All variants are recompiled from their fixed
catalog version and constraints to reject altered terms or descriptions. The
selected policy is rechecked immediately before execution. Every attempt keeps
the price/capability constraints and output-token cap, including fallbacks.
The existing total request deadline contains the decision deadline; cancellation
does not start a fallback chain in the background.

The decision sees only the saved use case, up to 8 KB of the latest user text,
and whether tools are requested. No system prompt, tool output, credentials,
image URL or previous conversation is projected. This is an explicit content
sharing choice, not a claim to detect all PII inside text supplied by a user.

Decision and inference cost are shown separately in x_router and summed only
when both are known. Missing usage is unknown, never a fabricated zero. The
bounded activity summary retains the actual executed policy and proposed
selection, latency, uncertainty and fallback reason, not task content. Existing
best-effort ledger admission remains unchanged; this release does not add hard
monthly budget reservations or claim measured savings without a baseline.
