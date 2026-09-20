# Decision-guided context compaction

`POST /v1/compact` retains its existing single-summary behavior unless the caller
supplies `decision_policy_ir`. With that field it performs fragment triage:

1. Group each assistant tool call with its adjacent tool results. Preserve all
   system/developer messages, pinned user input, and the recent tail verbatim.
2. Ask the decision policy to classify aged fragments as `keep`, `summarize`, or
   `archive`. Questions are batched, with smaller batches when the decision API's
   32 KB ASCII-serialized request limit requires it. Invalid decisions keep data.
3. Send only `summarize` fragments, with their full evidence and stable IDs, to
   the generative `policy_ir`. Batch summaries within bounded input windows.
4. Validate IDs, text and size; reassemble in original order. A failed, truncated
   or oversized summary retains its original fragment. Excerpt-only decisions
   cannot delete unseen evidence: `archive` becomes `summarize` for those units.

The request adds these optional fields to `messages`, `policy_ir`, `max_tokens`
and `keep_recent`:

| Field | Meaning |
| --- | --- |
| `decision_policy_ir` | Routing policy for the decision model; enables this mode. |
| `target_ratio` | Desired output/input serialized UTF-8 byte ratio, default `0.1`, range `(0,1]`. This is not a tokenizer count. |
| `pinned_indices` | Original zero-based message indexes to retain. Defaults to every user message. System/developer messages and complete recent units are always retained. |

Agents that use the `user` role for generated execution observations should send
the indexes of their actual user instructions. SubZeroClaw does this from its
canonical transcript. No heuristic attempts to distinguish real instructions
from generated observations by inspecting their text.

`compaction` in the response reports original/output/target bytes, `target_met`,
the fragment manifest (`start` inclusive, `end` exclusive), and limit/failure
reasons. The 10% target never overrides protected or explicitly kept content.
When it cannot fit, summaries use a small best-effort budget and the result
reports the actual size. Expansion is rejected. There are at most 128 selectable
fragments, 32 decision calls and 8 summary calls per request; oversized evidence
remains intact. Summary input batches are at most approximately 60 KB and each
summary call has at most 4,096 output tokens (or the caller's smaller limit).

`x_router.compaction_legs` contains each decision/summary leg's routing and cost
metadata, including failures. Top-level `x_router.cost_usd` sums known leg costs;
it is null if any leg's cost is unknown. Usage sums reported tokens and cache reads;
`usage_complete:false` identifies missing leg usage. No-call responses omit cost
and usage. Both routing policies must include any required provider restrictions.

This is a stateless transform: **the caller must retain the original transcript**
before applying it. `archive` removes active context; the router does not create
an archive. A summary includes its fragment ID for reference in the caller's
snapshot manifest. In SubZeroClaw, shell evidence also carries archive call IDs.
Prefix bytes before the first replaced fragment stay unchanged. Compaction can
invalidate cached tokens after that point; it does not guarantee provider cache
hits, factual summary correctness, or a particular savings/latency improvement.

Hermetic coverage: `pytest tests/test_fragment_compaction.py tests/test_compact.py`.
The opt-in live BDD scenario additionally needs the local stack and an explicit
`DECISION_COMPACTION_POLICY_IR` environment value; it may incur provider charges.
