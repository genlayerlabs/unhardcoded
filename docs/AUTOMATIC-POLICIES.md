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
Accounting, admission controls and request requirements remain host concerns.
No production behavior changes merely by importing this catalog.
