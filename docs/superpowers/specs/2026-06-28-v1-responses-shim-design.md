# `/v1/responses` Shim — Design

**Date:** 2026-06-28
**Repo:** `genlayerlabs/unhardcoded` (the router)
**Status:** approved design, ready for implementation plan

## Problem

The router (`router.ygr.ai`) speaks only the OpenAI **chat-completions** wire
(`POST /v1/chat/completions`, plus `/{profile}/v1/chat/completions`, `/v1/compact`,
`/v1/models`, `/healthz`, `/x/*`). It has **no** `POST /v1/responses`.

The OpenAI **Codex CLI** (v0.142.x) speaks ONLY the OpenAI *Responses* API
(`POST /v1/responses`, SSE) — it rejects `wire_api="chat"`. Pointed at the router it
fails with repeated `unexpected status 404 Not Found … url: https://router.ygr.ai/v1/responses`.

This blocks the MMF watchdog's investigator, which shells out to `codex exec` and needs it
to reach the router (so investigations are routed/policy-governed/metered like every other
call, instead of needing a separate raw provider key).

Verified three ways that the route is genuinely absent (not a transient flap): the route
list in `shim.py`, the live `/openapi.json`, and a live A/B (`/v1/responses` → 404 vs
`/v1/chat/completions` → 401, i.e. the chat route exists and auth-gates while the responses
route is unregistered).

## Goal

Add `POST /v1/responses` (and `POST /{profile_name}/v1/responses`) so any Responses-API
client — Codex CLI first — can drive the router. The endpoint reuses the router's existing
contract, routing, policy admission, streaming, and metering; only the wire translation at
the edges is new.

**Acceptance:** `codex exec` against the router returns a real answer end-to-end, and a
tool-call round-trip works.

## Key insight: this is the inverse of `codex_backend.py`

`codex_backend.py` already does the chat↔Responses translation in the **outbound**
direction (router contract → upstream ChatGPT Codex Responses endpoint, and its SSE back to
the router's response shape). This endpoint is its **inbound mirror**:

| `codex_backend.py` (outbound)            | this endpoint (inbound)                        |
|------------------------------------------|------------------------------------------------|
| `_messages_to_input` (messages→input)    | `input_to_messages` (input→messages)           |
| `_to_responses_tools` (nested→flat)      | `tools_to_chat` (flat→nested)                  |
| `_to_responses_tool_choice`              | `tool_choice_to_chat`                          |
| `aggregate_codex_sse` (Responses SSE→resp)| `result_to_responses_*` (resp→Responses obj/SSE)|

The event vocabulary the new SSE emits is exactly the vocabulary `aggregate_codex_sse`
already consumes (`response.output_text.delta`, `response.output_item.added/done`,
`response.function_call_arguments.delta/done`, `response.completed`), so the two stay
mutually consistent.

## Architecture

- **New pure module `responses_api.py`** — all translation as pure, side-effect-free
  functions (unit-tested like `tests/test_codex.py`): request-in translation, the complete
  Responses `response` object builder, and the SSE event encoders.
- **Thin glue in `shim.py`** — two routes (`/v1/responses`, `/{profile_name}/v1/responses`)
  inside `create_app`, reusing `_request_to_contract`, `host.execute_async`,
  `_policy_admission_error`/`_invalid_policy_response`, `_EARLY_FAIL_S`, `_HEARTBEAT_S`,
  `_streaming.HEARTBEAT`, and the per-session meter exactly as the chat path does.

No change to the router core, `llm_router_host.py`, or `codex_backend.py`.

## 1. Request in: Responses → router contract

A `ResponsesRequest` pydantic model (`extra="allow"`, mirroring `ChatRequest`) with the
fields the shim reads: `model`, `input` (str | list), `instructions`, `tools`,
`tool_choice`, `stream`, `max_output_tokens`, `temperature`, `session`. Everything else is
accepted and ignored (`reasoning`, `include`, `store`, `parallel_tool_calls`,
`prompt_cache_key`, `text`, `previous_response_id`, …).

### `input_to_messages(input, instructions) -> list[dict]`

Inverse of `codex_backend._messages_to_input`.

- `instructions` (if non-empty) → a leading `{"role":"system","content":instructions}`.
- `input` is a **string** → `[{"role":"user","content":input}]` (after the system message).
- `input` is a **list** of items, each:
  - `{"role":r,"content":c}` or `{"type":"message","role":r,"content":c}` where `c` is a
    string → `{"role":r,"content":c}`.
  - …where `c` is a content-parts list → concatenate the `text` of each
    `{"type":"input_text"|"output_text"|"text", "text":…}` part into one string
    (non-text parts e.g. `input_image` are dropped — see Out of scope).
  - `{"type":"function_call","call_id":id,"name":n,"arguments":a}` → contributes a
    tool-call `{"id":id,"type":"function","function":{"name":n,"arguments":a}}`. **Consecutive**
    `function_call` items (no non-function item between them) merge onto **one** assistant
    message's `tool_calls` array (chat-completions requires tool_calls grouped per assistant
    turn). An assistant text item immediately preceding the run becomes that message's
    `content`.
  - `{"type":"function_call_output","call_id":id,"output":o}` →
    `{"role":"tool","tool_call_id":id,"content":o}` (coerce `output` to string).
  - `{"type":"reasoning",…}` and any unrecognized item type → skipped.
  - a `role:"developer"` message (Codex sends one — the OpenAI "developer" ≈ system
    role) → normalized to `role:"system"`, since many chat-completions providers accept
    only system/user/assistant/tool. [Added after the live Codex acceptance test.]

Symmetry check (must hold): for the canonical tool-history conversation in
`test_codex.py::test_messages_to_input_translates_tool_call_history`,
`input_to_messages(_messages_to_input(msgs), instructions=None)` reproduces `msgs` up to the
documented normalizations (a leading system message only when `instructions` is set; merged
tool_calls).

### `tools_to_chat(tools) -> list[dict] | None`

Inverse of `codex_backend._to_responses_tools`.

- `{"type":"function","name":n,"description":d?,"parameters":p?}` →
  `{"type":"function","function":{"name":n[,"description":d][,"parameters":p]}}`.
- Any other `type` (e.g. `local_shell`, `web_search`, `custom`) → **dropped**, not
  forwarded: a plain chat-completions provider rejects any tool whose `type != "function"`,
  which would 400 the whole turn (killing even the text answer). Dropping lets the text turn
  through. `dropped_tool_types(tools)` exposes what was discarded so the shim can log it
  (`_handle_responses`), making a needed-but-dropped tool visible. If the live Codex tool set
  turns out to need a native type, add a targeted mapping (revisited during the acceptance
  test). [Updated from the original "pass-through" after the final code review.]
- Empty/None → None.

### `tool_choice_to_chat(tc) -> Any`

Inverse of `codex_backend._to_responses_tool_choice`. Strings (`"auto"`/`"required"`/
`"none"`) pass through; `{"type":"function","name":n}` → `{"type":"function","function":{"name":n}}`;
None → None.

### Contract assembly (in `shim.py`)

Build a synthetic `ChatRequest` from the translated `messages`/`tools`/`tool_choice`/
`temperature`/`session`/`model` and `max_tokens = max_output_tokens`, then call the existing
`_request_to_contract(chatreq, default_profile, default_max_tokens)`. The path-addressed
route overrides `contract["profile"]` with `profile_name`, exactly like
`chat_completions_profiled`.

## 2. Middle: route through the existing engine

`await host.execute_async(contract)`, wrapped in the same try/except that maps a core
admission error (`_policy_admission_error`) to a 400 (`_invalid_policy_response`). Identical
to `_handle_chat`.

## 3. Response out: router result → Responses

### `result_to_responses_object(result, model, *, response_id, subscription_providers, session, owner) -> dict`

The complete non-streaming Responses object AND the source the streamer replays:

```jsonc
{
  "id": "resp_<hex>",
  "object": "response",
  "created_at": <unix>,
  "status": "completed",            // "incomplete" if finish_reason == "length"
  "model": "<raw_model | served_model_id | requested>",
  "output": [
    // present when there is assistant text:
    {"type":"message","id":"msg_<hex>","role":"assistant","status":"completed",
     "content":[{"type":"output_text","text":"<text>","annotations":[]}]},
    // one per tool call:
    {"type":"function_call","id":"fc_<hex>","call_id":"<id>","name":"<n>",
     "arguments":"<args>","status":"completed"}
  ],
  "usage": {"input_tokens":N,"output_tokens":M,"total_tokens":T},  // omitted if unknown
  "x_router": { … same metadata block the chat path emits … }
}
```

- Text item omitted entirely when the result has no text (pure tool-call turn), matching the
  Responses convention; at least one output item is always present.
- `usage` mapped from `tokens_in`/`tokens_out`/`tokens_total`; omitted if all None.
- Per-session metering folded in (same `route_session_meter.observe` call as the chat path)
  so Responses calls show up in the session totals.
- `x_router` reuses the chat path's metadata builder for parity.

### Streaming (`stream:true`) — `responses_sse_events(obj) -> Iterable[str]`

Replays a complete `result_to_responses_object` as a faithful Responses SSE sequence, each
as a **named** event (`event: <type>\n` + `data: <json>\n\n`, the `type` field inside the
JSON matching the event name):

1. `response.created` — `{response: {id, object:"response", status:"in_progress", model}}`
2. for the message item (if any): `response.output_item.added` (the empty message shell) →
   `response.output_text.delta` (the full text as one delta) →
   `response.output_text.done` (`text`) → `response.output_item.done` (the full item)
3. for each function_call item: `response.output_item.added` →
   `response.function_call_arguments.delta` (full args as one delta) →
   `response.function_call_arguments.done` (`arguments`) → `response.output_item.done`
4. `response.completed` — `{response: {id, status, output:[…all items…], usage, model}}`

`delta`/`done` index fields (`output_index`, `content_index`, `item_id`) are kept
self-consistent so a strict client accepts them.

### The route handler (in `shim.py`)

- `stream:false` → return `result_to_responses_object(...)` as JSON
  (or `_openai_error_from_router(result)` on `ok:false`).
- `stream:true` → mirror `_flow_stream`: create `task = asyncio.create_task(host.execute_async(...))`;
  `await asyncio.wait({task}, timeout=_EARLY_FAIL_S)`. If it finished:
  - admission error → 400 JSON; `ok:false` → `_openai_error_from_router` (clean HTTP, since
    nothing was streamed yet);
  - `ok:true` → `StreamingResponse` that yields `response.created`, then the replayed events
    from `responses_sse_events(obj)`.
  If still running past the grace window → `StreamingResponse` that yields `response.created`
  immediately, emits `_streaming.HEARTBEAT` every `_HEARTBEAT_S` while the task runs, then
  the replayed events (or `response.failed` on error/exception).
- `response.failed` event: `{response: {id, status:"failed", error:{message, code}}}`,
  followed by `data: [DONE]` is **not** required (Responses streams end with
  `response.completed`/`response.failed`); no `[DONE]` sentinel is emitted (the Responses API
  does not use it — confirmed against the events `aggregate_codex_sse` consumes, which break
  on `response.completed`, treating `[DONE]` only defensively).

## 4. Out of scope (documented limitations)

- **Server-side conversation state / `previous_response_id`.** Not supported. Codex with
  `store:false` (its default for this path) sends the full `input` each turn, so statelessness
  is correct. A client relying on `previous_response_id` for continuation would lose context —
  documented, not handled.
- **True token-by-token streaming.** The stream is a pseudo-stream of the completed result
  (with heartbeats). Sufficient for `codex exec` (non-interactive). True streaming via the
  existing `streaming_call`/`emit` path is a possible later enhancement; tool-calls would
  still be assembled at the end (they are only in the final result, not the text `emit`
  stream).
- **Non-text input parts** (`input_image`, files). Dropped during input translation.
- **Reasoning output items.** The router's backends don't surface reasoning; none are
  emitted. Codex tolerates their absence.

## Testing

- **Unit (`tests/test_responses_api.py`)** — pure translation, no host:
  `input_to_messages` (string, items, content-parts, function_call merge, function_call_output,
  instructions→system, reasoning/unknown skipped, the round-trip symmetry against
  `_messages_to_input`); `tools_to_chat` + `tool_choice_to_chat` (inverse of the codex_backend
  helpers, incl. pass-through of unknown tool types); `result_to_responses_object` (text-only,
  tool-call-only, text+tools, usage mapping, incomplete status); `responses_sse_events`
  (event order, named events, completed carries output+usage, failed shape).
- **Integration (`tests/test_responses_shim.py`)** — `LLMRouterHost` + mock provider
  responses + `TestClient`, mirroring `test_shim.py`: `POST /v1/responses` non-stream returns
  a well-formed response object with the mocked text; `stream:true` yields a parseable SSE
  sequence ending in `response.completed`; a tool-call mock surfaces a `function_call` output
  item; `POST /{profile}/v1/responses` routes to that profile; a router error maps to the
  right HTTP status; an admission-rejected `policy_ir`-bearing request (if exercised) → 400.
- **Acceptance (manual, gated)** — point Codex CLI at the router
  (`base_url=https://router.ygr.ai/v1`, `wire_api="responses"`) and run `codex exec "reply
  with exactly: hello from codex"`; confirm a real reply and then a tool-using run.

## Rollout

Local TDD in the `unhardcoded` repo; full `pytest tests` green. Deploy is the router's normal
path (not part of this work). No secrets, no env, no core changes. Once live, MMF's
investigator config (`wire_api="responses"`, router base_url + key) works unchanged — the
earlier blocker is removed.
