# Provider timing diagnostics

OpenAI-compatible asynchronous calls retain a bounded `provider_diagnostics`
array in `x_router.decision_trace`, including failed attempts. The call ledger
stores the same allowlisted evidence in `calls.routing_summary`, so it remains
available when an ingress replaces a JSON 504 with an HTML error page. No database
migration is needed. Diagnostics do not change request bodies, routing, retries,
deadlines, generation limits or which response content is emitted.

Each entry identifies its attempt, provider and model family and may include:

| Field | Meaning |
|---|---|
| `mode` | `buffered`, `buffered_sse` (first-output-bound buffered request), or `streaming` |
| `phase` | Last observed transport phase: connection pool, connect, TLS, request headers/body, response headers/body; or peer capacity before HTTP starts |
| `request_sent_ms` | Elapsed time when HTTP Core finished sending the request body |
| `response_headers_ms` | Elapsed time when response headers arrived |
| `body_complete_ms` | Elapsed time when HTTP Core completed reading the body |
| `connect_ms`, `tls_ms` | Connection/TLS operation durations, when a connection was created |
| `http_status` | Upstream HTTP status, when headers were observed; an incomplete body can have status 200 and still time out |
| `timeout_source` | Buffered request's total `attempt_deadline`, or HTTPX timeout exception class |
| `first_reasoning_ms`, `first_output_ms` | First nonempty reasoning or content/tool delta observed by the streaming adapter; independent of whether the router has emitted it publicly |
| `elapsed_ms` | Adapter elapsed time; phase timestamps share this origin |
| `upstream_id`, `upstream_provider` | Provider response identifiers when returned; allow correlating with provider support/usage metadata |
| `tokens_reasoning` | Reported reasoning-token count, including explicit zero |
| `requested_timeout_ms`, `requested_max_tokens`, `requested_reasoning_effort`, `requested_reasoning_enabled` | Controls passed to the adapter; not a claim that the upstream honored them |

Reasoning usage is also exposed as standard
`usage.completion_tokens_details.reasoning_tokens` when reported. It is already
part of completion usage, so it is not added to total tokens or charged twice.
Missing counts/timings are unknown, not zero. Reused connections do not emit a
new connect/TLS duration. These fields cannot split an upstream's internal queue
from its computation without upstream metadata.

For a buffered request, `phase=response_headers` with no `response_headers_ms`
means no headers arrived before the failure. `phase=response_body` with an
upstream 200 and no `body_complete_ms` means the response started but was not
fully read before failure. A pooled-connection wait has no send/header events.

Only HTTPX clients expose HTTP Core phase tracing. Custom/mock clients continue
to work without those timing fields. Streaming can record an upstream ID before
completion; buffered JSON may not expose one until the complete response arrives.
An outer router deadline/cancellation can stop execution before an adapter result
is returned; this change does not claim to retain a partially cancelled attempt.
Native tool-call fragments are still accumulated before the final tool-call chunk.
The existing streaming timeout semantics are unchanged.

At most 32 diagnostic entries are retained per router execution. Fields are
allowlisted, strings are capped, and numeric fields must be finite/nonnegative.
Prompts, command arguments, HTTP headers, endpoint URLs, provider error bodies,
and reasoning text are not copied into diagnostics. A response identifier is a
correlation identifier, not an authentication credential.
