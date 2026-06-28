# `/v1/responses` Shim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /v1/responses` and `POST /{profile_name}/v1/responses` to the unhardcoded router so OpenAI Responses-API clients (Codex CLI first) can drive it, reusing the router's existing contract, routing, policy admission, streaming, and metering.

**Architecture:** A new pure module `responses_api.py` holds all wire translation (Responses request → chat messages/tools; router result → Responses object + SSE events) — the inbound mirror of `codex_backend.py`. Thin route glue in `shim.py` inside `create_app` reuses `_request_to_contract`, `host.execute_async`, the admission-error → 400 handling, `_EARLY_FAIL_S`/`_HEARTBEAT_S`, `_streaming.HEARTBEAT`, and the per-session meter exactly as the chat path does. No change to the router core, `llm_router_host.py`, or `codex_backend.py`.

**Tech Stack:** Python 3.12, FastAPI, pydantic v2, pytest (`pytest-asyncio`), `fastapi.testclient.TestClient`.

## Global Constraints

- Pure translation lives in `responses_api.py`; it must be side-effect free (no host, no metering, no I/O). The FastAPI glue lives in `shim.py`.
- The SSE event vocabulary emitted MUST match what `codex_backend.aggregate_codex_sse` consumes: `response.created`, `response.output_item.added`, `response.output_text.delta`, `response.output_text.done`, `response.output_item.done`, `response.function_call_arguments.delta`, `response.function_call_arguments.done`, `response.completed`, `response.failed`.
- Responses streams end on `response.completed` / `response.failed`. Do NOT emit a `data: [DONE]` sentinel (the Responses API does not use it).
- `input_to_messages` MUST be the exact inverse of `codex_backend._messages_to_input` up to documented normalizations (leading system message only when `instructions` is set; consecutive `function_call` items merged onto one assistant message's `tool_calls`).
- Tests run from the repo root: `cd /Users/albert/dev/unhardcoded && pytest tests -q`.
- Follow existing repo conventions: `from __future__ import annotations`, module docstring explaining the role, no new dependencies.

---

## File Structure

- **Create `responses_api.py`** — pure translation: `input_to_messages`, `tools_to_chat`, `tool_choice_to_chat`, `result_to_responses_object`, `responses_created_event`, `responses_failed_event`, `responses_sse_events`, `_part_text`, `_new_id`, `_sse`.
- **Create `tests/test_responses_api.py`** — unit tests for the pure module.
- **Modify `shim.py`** — add `ResponsesRequest` model (near `ChatRequest`, ~line 96) and, inside `create_app`, two routes + `_handle_responses` / `_handle_responses_stream` / `_responses_object_with_router` (after `chat_completions_profiled`, ~line 817).
- **Create `tests/test_responses_shim.py`** — integration tests via `LLMRouterHost` + mock responses + `TestClient`.
- **Modify `README.md`** — one line documenting the new endpoint (folded into Task 4).

---

### Task 1: `responses_api.py` — request-in translation

**Files:**
- Create: `responses_api.py` (this task: the request-in half + helpers)
- Test: `tests/test_responses_api.py` (this task: request-in tests)

**Interfaces:**
- Produces: `input_to_messages(input_: Any, instructions: str | None = None) -> list[dict]`; `tools_to_chat(tools: Any) -> list[dict] | None`; `tool_choice_to_chat(tc: Any) -> Any`; `_part_text(content: Any) -> str`.
- Consumes (in tests only): `codex_backend._messages_to_input`, `_to_responses_tools`, `_to_responses_tool_choice` for round-trip symmetry assertions.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_responses_api.py`:

```python
"""Unit tests for responses_api.py — the inbound Responses-API translation
(mirror of codex_backend.py). Pure, no host."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import codex_backend as cb  # noqa: E402
import responses_api as ra  # noqa: E402


# ---- input_to_messages -------------------------------------------------

def test_input_string_becomes_user_message():
    assert ra.input_to_messages("hi") == [{"role": "user", "content": "hi"}]


def test_instructions_become_leading_system_message():
    msgs = ra.input_to_messages("hi", instructions="be terse")
    assert msgs == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_input_items_plain_messages():
    items = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert ra.input_to_messages(items) == items


def test_input_content_parts_are_flattened():
    items = [{"role": "user", "content": [
        {"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}]}]
    assert ra.input_to_messages(items) == [{"role": "user", "content": "ab"}]


def test_message_type_wrapper_is_unwrapped():
    items = [{"type": "message", "role": "user", "content": "hi"}]
    assert ra.input_to_messages(items) == [{"role": "user", "content": "hi"}]


def test_function_call_output_becomes_tool_message():
    items = [{"type": "function_call_output", "call_id": "c1", "output": "42"}]
    assert ra.input_to_messages(items) == [
        {"role": "tool", "tool_call_id": "c1", "content": "42"}]


def test_reasoning_and_unknown_items_skipped():
    items = [{"type": "reasoning", "summary": []},
             {"type": "weird_future_type"},
             {"role": "user", "content": "hi"}]
    assert ra.input_to_messages(items) == [{"role": "user", "content": "hi"}]


def test_consecutive_function_calls_merge_onto_one_assistant_message():
    items = [
        {"role": "user", "content": "weather?"},
        {"type": "function_call", "call_id": "c1", "name": "get_weather",
         "arguments": '{"city":"BCN"}'},
        {"type": "function_call", "call_id": "c2", "name": "get_time",
         "arguments": "{}"},
    ]
    assert ra.input_to_messages(items) == [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"BCN"}'}},
            {"id": "c2", "type": "function",
             "function": {"name": "get_time", "arguments": "{}"}},
        ]},
    ]


def test_assistant_text_then_function_call_merge_into_one_message():
    items = [
        {"role": "assistant", "content": "checking…"},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
    ]
    assert ra.input_to_messages(items) == [
        {"role": "assistant", "content": "checking…", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
    ]


def test_roundtrip_inverse_of_messages_to_input():
    # The canonical tool-history conversation that 400d live (test_codex.py).
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "weather in BCN?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"BCN"}'}},
            {"id": "call_2", "type": "function",
             "function": {"name": "get_time", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "22C sunny"},
        {"role": "tool", "tool_call_id": "call_2", "content": "14:00"},
        {"role": "assistant", "content": "It's 22C."},
        {"role": "user", "content": "and tomorrow?"},
    ]
    items = cb._messages_to_input(msgs)
    # the system message is supplied separately as `instructions` on the way back
    rebuilt = ra.input_to_messages(items, instructions="be terse")
    assert rebuilt == msgs


# ---- tools_to_chat / tool_choice_to_chat -------------------------------

def test_tools_to_chat_nests_function_tools():
    flat = [{"type": "function", "name": "shell", "description": "run",
             "parameters": {"type": "object", "properties": {}}}]
    assert ra.tools_to_chat(flat) == [{"type": "function", "function": {
        "name": "shell", "description": "run",
        "parameters": {"type": "object", "properties": {}}}]


def test_tools_to_chat_is_inverse_of_to_responses_tools():
    nested = [{"type": "function", "function": {
        "name": "shell", "description": "run a shell command",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}]
    flat = cb._to_responses_tools(nested)
    assert ra.tools_to_chat(flat) == nested


def test_tools_to_chat_passes_through_unknown_tool_types():
    flat = [{"type": "local_shell"}, {"type": "web_search"}]
    assert ra.tools_to_chat(flat) == flat


def test_tools_to_chat_empty_is_none():
    assert ra.tools_to_chat(None) is None
    assert ra.tools_to_chat([]) is None


def test_tool_choice_to_chat_passthrough_and_named():
    assert ra.tool_choice_to_chat("auto") == "auto"
    assert ra.tool_choice_to_chat(None) is None
    assert ra.tool_choice_to_chat({"type": "function", "name": "shell"}) == {
        "type": "function", "function": {"name": "shell"}}


def test_tool_choice_to_chat_is_inverse_of_to_responses():
    nested = {"type": "function", "function": {"name": "shell"}}
    flat = cb._to_responses_tool_choice(nested)
    assert ra.tool_choice_to_chat(flat) == nested
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/albert/dev/unhardcoded && pytest tests/test_responses_api.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'responses_api'`.

- [ ] **Step 3: Implement the request-in half of `responses_api.py`**

Create `responses_api.py`:

```python
"""
responses_api.py — inbound OpenAI *Responses* API translation for the shim.

The mirror of codex_backend.py: codex_backend translates the router's chat
contract OUT to the upstream Codex Responses endpoint (and its SSE back); this
module translates a Responses request IN to the chat-completions messages/tools
the router contract understands, and the router's result back OUT to a Responses
`response` object and its SSE event stream.

Pure + side-effect free (the FastAPI glue lives in shim.py), unit-tested like
tests/test_codex.py. The SSE event vocabulary emitted here is exactly the
vocabulary codex_backend.aggregate_codex_sse consumes, so the two stay mutually
consistent.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterable


# ---- request in: Responses -> chat-completions ----------------------------

def _part_text(content: Any) -> str:
    """A Responses content value (str | content-parts list | None) -> text.
    Mirror of codex_backend._content_to_text, reading the Responses `text` field
    of each part (input_text / output_text / text)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text") or "")
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)


def input_to_messages(input_: Any, instructions: str | None = None) -> list[dict]:
    """Responses `input` (+ `instructions`) -> chat-completions `messages`.

    Inverse of codex_backend._messages_to_input:
      - `instructions` (if set) -> a leading system message.
      - a string `input` -> one user message.
      - `{role, content}` / `{type:"message", role, content}` -> a message
        (content-parts flattened to text).
      - `{type:"function_call", call_id, name, arguments}` -> an assistant
        tool_call; consecutive function_calls (and an assistant text item
        directly before them) merge onto ONE assistant message's tool_calls.
      - `{type:"function_call_output", call_id, output}` -> a `tool` message.
      - `{type:"reasoning"}` / unknown item types -> skipped.
    """
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if input_ is None:
        items: list = []
    elif isinstance(input_, str):
        items = [{"role": "user", "content": input_}]
    elif isinstance(input_, list):
        items = input_
    else:
        items = []

    for it in items:
        if not isinstance(it, dict):
            continue
        itype = it.get("type")

        if itype == "function_call":
            tc = {
                "id": it.get("call_id") or "",
                "type": "function",
                "function": {"name": it.get("name") or "",
                             "arguments": it.get("arguments") or "{}"},
            }
            prev = messages[-1] if messages else None
            if prev is not None and prev.get("role") == "assistant":
                prev.setdefault("tool_calls", []).append(tc)
            else:
                messages.append({"role": "assistant", "content": None,
                                 "tool_calls": [tc]})
            continue

        if itype == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": it.get("call_id") or "",
                "content": _part_text(it.get("output")),
            })
            continue

        if itype == "reasoning":
            continue

        role = it.get("role")
        if role is None:
            continue  # unknown item type with no role
        messages.append({"role": role, "content": _part_text(it.get("content"))})

    return messages


def tools_to_chat(tools: Any) -> "list[dict] | None":
    """Responses tools (FLAT {type:"function", name, ...}) -> chat-completions
    tools (NESTED {type:"function", function:{name,...}}). Inverse of
    codex_backend._to_responses_tools. Unknown/native tool types pass through."""
    if not tools:
        return None
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and t.get("name"):
            fn: dict = {"name": t.get("name")}
            if t.get("description") is not None:
                fn["description"] = t["description"]
            if t.get("parameters") is not None:
                fn["parameters"] = t["parameters"]
            out.append({"type": "function", "function": fn})
        else:
            out.append(t)  # native/unknown tool — best-effort pass-through
    return out or None


def tool_choice_to_chat(tc: Any) -> Any:
    """Inverse of codex_backend._to_responses_tool_choice. Strings pass through;
    {type:"function", name} -> {type:"function", function:{name}}."""
    if tc is None or isinstance(tc, str):
        return tc
    if isinstance(tc, dict) and tc.get("type") == "function":
        name = tc.get("name")
        if name is None and isinstance(tc.get("function"), dict):
            name = tc["function"].get("name")
        if name:
            return {"type": "function", "function": {"name": name}}
    return None


# ---- id helpers (shared by the response-out half, Tasks 2-3) --------------

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/albert/dev/unhardcoded && pytest tests/test_responses_api.py -q`
Expected: PASS (all request-in tests green).

- [ ] **Step 5: Commit**

```bash
cd /Users/albert/dev/unhardcoded
git add responses_api.py tests/test_responses_api.py
git commit -m "feat(responses): inbound Responses->chat request translation"
```

---

### Task 2: `responses_api.py` — result → Responses object

**Files:**
- Modify: `responses_api.py` (add `result_to_responses_object`)
- Test: `tests/test_responses_api.py` (add object-builder tests)

**Interfaces:**
- Consumes: `_new_id` (Task 1).
- Produces: `result_to_responses_object(result: dict, requested_model: str = "", *, response_id: str | None = None, created_at: int | None = None, msg_id: str | None = None, now=None) -> dict`. Returns a complete Responses `response` object: `{id, object:"response", created_at, status, model, output:[...], usage?}`. `output` carries a `message` item (only when there is text) and one `function_call` item per tool call. `x_router` is attached later by the shim.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_responses_api.py`:

```python
# ---- result_to_responses_object ----------------------------------------

def _result(text="", tool_calls=None, finish="stop",
            tin=7, tout=3, ttot=10, raw_model="mock-model"):
    return {"ok": True, "response": {
        "text": text, "tool_calls": tool_calls, "finish_reason": finish,
        "tokens_in": tin, "tokens_out": tout, "tokens_total": ttot,
        "raw_model": raw_model}, "chosen": {"served_model_id": "srv"}}


def test_object_text_only():
    obj = ra.result_to_responses_object(_result(text="hello"), "policy:auto",
                                        response_id="resp_x", created_at=111,
                                        msg_id="msg_x")
    assert obj["id"] == "resp_x"
    assert obj["object"] == "response"
    assert obj["created_at"] == 111
    assert obj["status"] == "completed"
    assert obj["model"] == "mock-model"
    assert obj["output"] == [{
        "type": "message", "id": "msg_x", "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "hello", "annotations": []}]}]
    assert obj["usage"] == {"input_tokens": 7, "output_tokens": 3,
                            "total_tokens": 10}


def test_object_tool_calls_only_has_no_message_item():
    tcs = [{"id": "call_1", "type": "function",
            "function": {"name": "shell", "arguments": '{"command":"ls"}'}}]
    obj = ra.result_to_responses_object(_result(text="", tool_calls=tcs),
                                        response_id="r", created_at=1)
    types = [o["type"] for o in obj["output"]]
    assert types == ["function_call"]
    fc = obj["output"][0]
    assert fc["call_id"] == "call_1" and fc["name"] == "shell"
    assert fc["arguments"] == '{"command":"ls"}'
    assert fc["status"] == "completed" and fc["id"].startswith("fc_")


def test_object_text_and_tool_calls():
    tcs = [{"id": "c1", "type": "function",
            "function": {"name": "f", "arguments": "{}"}}]
    obj = ra.result_to_responses_object(_result(text="ok", tool_calls=tcs),
                                        response_id="r", created_at=1)
    assert [o["type"] for o in obj["output"]] == ["message", "function_call"]


def test_object_length_finish_is_incomplete():
    obj = ra.result_to_responses_object(_result(text="x", finish="length"),
                                        response_id="r", created_at=1)
    assert obj["status"] == "incomplete"


def test_object_omits_usage_when_unknown():
    obj = ra.result_to_responses_object(
        _result(text="x", tin=None, tout=None, ttot=None),
        response_id="r", created_at=1)
    assert "usage" not in obj


def test_object_uses_now_when_created_at_absent():
    obj = ra.result_to_responses_object(_result(text="x"), response_id="r",
                                        now=lambda: 555)
    assert obj["created_at"] == 555
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/albert/dev/unhardcoded && pytest tests/test_responses_api.py -k object -q`
Expected: FAIL — `AttributeError: module 'responses_api' has no attribute 'result_to_responses_object'`.

- [ ] **Step 3: Implement `result_to_responses_object`**

Append to `responses_api.py`:

```python
# ---- response out: router result -> Responses object ----------------------

def result_to_responses_object(
    result: dict,
    requested_model: str = "",
    *,
    response_id: str | None = None,
    created_at: int | None = None,
    msg_id: str | None = None,
    now=None,
) -> dict:
    """Router result -> a complete Responses `response` object.

    The single source both the non-stream JSON return and the SSE replay use, so
    they carry identical ids. Pure: x_router + per-session metering are attached
    by the shim. Deterministic when ids/created_at/now are supplied (tests)."""
    resp = result.get("response") or {}
    chosen = result.get("chosen") or {}
    text = resp.get("text") or ""
    tool_calls = resp.get("tool_calls") or []
    finish = resp.get("finish_reason") or "stop"

    model = (resp.get("raw_model") or chosen.get("served_model_id")
             or requested_model or "")
    ts = created_at if created_at is not None else int((now or time.time)())

    output: list[dict] = []
    if text:
        output.append({
            "type": "message",
            "id": msg_id or _new_id("msg"),
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })
    for tc in tool_calls:
        fn = tc.get("function") or {}
        output.append({
            "type": "function_call",
            "id": _new_id("fc"),
            "call_id": tc.get("id") or _new_id("call"),
            "name": fn.get("name") or "",
            "arguments": fn.get("arguments") or "{}",
            "status": "completed",
        })

    obj: dict = {
        "id": response_id or _new_id("resp"),
        "object": "response",
        "created_at": ts,
        "status": "incomplete" if finish == "length" else "completed",
        "model": model,
        "output": output,
    }
    usage = {}
    if resp.get("tokens_in") is not None:
        usage["input_tokens"] = resp["tokens_in"]
    if resp.get("tokens_out") is not None:
        usage["output_tokens"] = resp["tokens_out"]
    if resp.get("tokens_total") is not None:
        usage["total_tokens"] = resp["tokens_total"]
    if usage:
        obj["usage"] = usage
    return obj
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/albert/dev/unhardcoded && pytest tests/test_responses_api.py -q`
Expected: PASS (all request-in + object tests).

- [ ] **Step 5: Commit**

```bash
cd /Users/albert/dev/unhardcoded
git add responses_api.py tests/test_responses_api.py
git commit -m "feat(responses): router-result -> Responses object builder"
```

---

### Task 3: `responses_api.py` — SSE event encoders

**Files:**
- Modify: `responses_api.py` (add `_sse`, `responses_created_event`, `responses_failed_event`, `responses_sse_events`)
- Test: `tests/test_responses_api.py` (add SSE tests)

**Interfaces:**
- Consumes: `result_to_responses_object` output objects.
- Produces:
  - `responses_created_event(obj: dict, seq: int = 0) -> str`
  - `responses_failed_event(response_id: str, message: str, code: str | None = None, seq: int = 0) -> str`
  - `responses_sse_events(obj: dict, start_seq: int = 1) -> Iterable[str]`
  - All emit named SSE frames `event: <type>\ndata: <json>\n\n` with the `type` field inside `data` matching the event name.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_responses_api.py`:

```python
# ---- SSE encoders ------------------------------------------------------

def _parse_sse(frames):
    """[(event_name, data_dict), ...] from a list of 'event:..\\ndata:..\\n\\n'."""
    out = []
    for f in frames:
        lines = [ln for ln in f.split("\n") if ln]
        ev = next(ln[len("event:"):].strip() for ln in lines if ln.startswith("event:"))
        data = next(ln[len("data:"):].strip() for ln in lines if ln.startswith("data:"))
        out.append((ev, json.loads(data)))
    return out


def test_created_event_shape():
    obj = ra.result_to_responses_object(_result(text="x"), response_id="resp_1",
                                        created_at=1)
    (ev, data), = _parse_sse([ra.responses_created_event(obj)])
    assert ev == "response.created"
    assert data["type"] == "response.created"
    assert data["response"]["id"] == "resp_1"
    assert data["response"]["status"] == "in_progress"


def test_failed_event_shape():
    (ev, data), = _parse_sse([ra.responses_failed_event("resp_1", "boom", "server_error")])
    assert ev == "response.failed"
    assert data["response"]["status"] == "failed"
    assert data["response"]["error"]["message"] == "boom"
    assert data["response"]["error"]["code"] == "server_error"


def test_sse_events_text_sequence():
    obj = ra.result_to_responses_object(_result(text="Hello"), response_id="r",
                                        created_at=1, msg_id="msg_1")
    events = _parse_sse(list(ra.responses_sse_events(obj)))
    names = [e for e, _ in events]
    assert names == [
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.output_item.done",
        "response.completed",
    ]
    delta = dict(events)["response.output_text.delta"]
    assert delta["delta"] == "Hello" and delta["item_id"] == "msg_1"
    completed = dict(events)["response.completed"]
    assert completed["response"]["output"] == obj["output"]
    assert completed["response"]["usage"]["total_tokens"] == 10


def test_sse_events_function_call_sequence():
    tcs = [{"id": "call_1", "type": "function",
            "function": {"name": "shell", "arguments": '{"command":"ls"}'}}]
    obj = ra.result_to_responses_object(_result(text="", tool_calls=tcs),
                                        response_id="r", created_at=1)
    events = _parse_sse(list(ra.responses_sse_events(obj)))
    names = [e for e, _ in events]
    assert names == [
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    done = dict(events)["response.function_call_arguments.done"]
    assert done["arguments"] == '{"command":"ls"}'


def test_sse_events_sequence_numbers_increase():
    obj = ra.result_to_responses_object(_result(text="hi"), response_id="r",
                                        created_at=1)
    seqs = [d["sequence_number"] for _, d in _parse_sse(list(ra.responses_sse_events(obj)))]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/albert/dev/unhardcoded && pytest tests/test_responses_api.py -k "sse or created_event or failed_event" -q`
Expected: FAIL — missing `responses_created_event` / `responses_sse_events`.

- [ ] **Step 3: Implement the SSE encoders**

Append to `responses_api.py`:

```python
# ---- response out: Responses SSE events -----------------------------------

def _sse(event: str, data: dict) -> str:
    """One named SSE frame. Codex's parser dispatches on the `type` field inside
    `data`; the `event:` line is emitted too for strict/event-name clients."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def responses_created_event(obj: dict, seq: int = 0) -> str:
    response = {"id": obj.get("id"), "object": "response", "status": "in_progress",
                "model": obj.get("model", ""), "output": []}
    return _sse("response.created", {"type": "response.created",
                                     "sequence_number": seq, "response": response})


def responses_failed_event(response_id: str, message: str,
                           code: str | None = None, seq: int = 0) -> str:
    response = {"id": response_id, "object": "response", "status": "failed",
                "error": {"message": message, "code": code}}
    return _sse("response.failed", {"type": "response.failed",
                                    "sequence_number": seq, "response": response})


def responses_sse_events(obj: dict, start_seq: int = 1) -> Iterable[str]:
    """Replay a complete response object (from result_to_responses_object) as the
    Responses SSE event sequence: per output item an added/delta/done trio, then
    `response.completed` carrying the full object. No `[DONE]` sentinel — the
    Responses API ends on `response.completed`."""
    seq = start_seq

    def _emit(event: str, data: dict) -> str:
        nonlocal seq
        data = {"type": event, "sequence_number": seq, **data}
        seq += 1
        return _sse(event, data)

    for out_index, item in enumerate(obj.get("output") or []):
        if item.get("type") == "message":
            text = (item.get("content") or [{}])[0].get("text", "")
            yield _emit("response.output_item.added",
                        {"output_index": out_index, "item": {**item, "content": []}})
            yield _emit("response.output_text.delta",
                        {"item_id": item["id"], "output_index": out_index,
                         "content_index": 0, "delta": text})
            yield _emit("response.output_text.done",
                        {"item_id": item["id"], "output_index": out_index,
                         "content_index": 0, "text": text})
            yield _emit("response.output_item.done",
                        {"output_index": out_index, "item": item})
        elif item.get("type") == "function_call":
            args = item.get("arguments") or ""
            yield _emit("response.output_item.added",
                        {"output_index": out_index, "item": {**item, "arguments": ""}})
            yield _emit("response.function_call_arguments.delta",
                        {"item_id": item["id"], "output_index": out_index,
                         "delta": args})
            yield _emit("response.function_call_arguments.done",
                        {"item_id": item["id"], "output_index": out_index,
                         "arguments": args})
            yield _emit("response.output_item.done",
                        {"output_index": out_index, "item": item})

    yield _emit("response.completed", {"response": obj})
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /Users/albert/dev/unhardcoded && pytest tests/test_responses_api.py -q`
Expected: PASS (entire pure module green).

- [ ] **Step 5: Commit**

```bash
cd /Users/albert/dev/unhardcoded
git add responses_api.py tests/test_responses_api.py
git commit -m "feat(responses): Responses SSE event encoders"
```

---

### Task 4: `shim.py` — wire the endpoints + integration tests

**Files:**
- Modify: `shim.py` — add `ResponsesRequest` (after `ChatRequest`, ~line 96) and, inside `create_app`, two routes + handlers (after `chat_completions_profiled`, ~line 817)
- Create: `tests/test_responses_shim.py`
- Modify: `README.md` — one line under the endpoints list

**Interfaces:**
- Consumes: `responses_api` (Tasks 1-3); module-level `_request_to_contract`, `_policy_admission_error`, `_invalid_policy_response`, `_openai_error_from_router`, `_executed_cost_usd`, `_trim_trace`, `_compact_suggested`, `_EARLY_FAIL_S`, `_HEARTBEAT_S`; closures `host`, `default_profile`, `default_max_tokens`, `subscription_providers`, `_streaming`, `route_session_meter`; `ChatRequest`, `_session_from_header`.
- Produces: `POST /v1/responses`, `POST /{profile_name}/v1/responses`.

- [ ] **Step 1: Write the failing integration tests**

Create `tests/test_responses_shim.py`:

```python
"""Integration tests for the /v1/responses shim. Mirrors test_shim.py: a real
LLMRouterHost backed by mock provider responses; only outbound HTTP is mocked."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router_host import LLMRouterHost  # noqa: E402
from shim import create_app  # noqa: E402


def _ok(text="hi back", tool_calls=None):
    return {"ok": True, "latency_ms": 10, "response": {
        "text": text, "tool_calls": tool_calls, "finish_reason":
        "tool_calls" if tool_calls else "stop", "tokens_in": 7,
        "tokens_out": 3, "tokens_total": 10, "raw_model": "mock-model-id"}}


def _all_pairs(host):
    cat = host.catalog() or {}
    models = cat.get("models") or {}
    out = []
    for pid, p in (cat.get("providers") or {}).items():
        for fam in (p.get("served_models") or models.keys()):
            out.append((pid, fam))
    return out


@pytest.fixture
def host():
    h = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "core" / "config.example.lua",
        metrics_path=ROOT / "core" / "metrics.example.lua",
        now_ms=lambda: 1_000_000)
    h.init()
    return h


@pytest.fixture
def client(host):
    return TestClient(create_app(host, default_profile="default"))


def _seed(host, resp):
    for prov, fam in _all_pairs(host):
        host.set_mock_response(prov, fam, resp)


def test_responses_nonstream_returns_response_object(client, host):
    _seed(host, _ok("hello from router"))
    r = client.post("/v1/responses", json={
        "model": "policy:auto", "input": "hi", "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    msg = [o for o in body["output"] if o["type"] == "message"][0]
    assert msg["content"][0]["text"] == "hello from router"
    assert body["usage"]["total_tokens"] == 10
    assert body["x_router"]["provider"] is not None


def test_responses_input_items_and_instructions(client, host):
    _seed(host, _ok("ok"))
    r = client.post("/v1/responses", json={
        "model": "", "instructions": "be terse",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "stream": False})
    assert r.status_code == 200
    assert r.json()["output"][0]["content"][0]["text"] == "ok"


def test_responses_stream_emits_completed(client, host):
    _seed(host, _ok("streamed hi"))
    with client.stream("POST", "/v1/responses", json={
            "model": "policy:auto", "input": "hi", "stream": True}) as r:
        assert r.status_code == 200
        body = "".join(chunk for chunk in r.iter_text())
    events = [ln[len("event:"):].strip()
              for ln in body.split("\n") if ln.startswith("event:")]
    assert events[0] == "response.created"
    assert "response.output_text.delta" in events
    assert events[-1] == "response.completed"
    # the completed event carries the assistant text
    datas = [json.loads(ln[len("data:"):].strip())
             for ln in body.split("\n") if ln.startswith("data:")]
    completed = [d for d in datas if d.get("type") == "response.completed"][0]
    msg = [o for o in completed["response"]["output"] if o["type"] == "message"][0]
    assert msg["content"][0]["text"] == "streamed hi"


def test_responses_tool_call_surfaces_function_call_item(client, host):
    tcs = [{"id": "call_1", "type": "function",
            "function": {"name": "shell", "arguments": '{"command":"ls"}'}}]
    _seed(host, _ok(text="", tool_calls=tcs))
    r = client.post("/v1/responses", json={
        "model": "policy:auto", "input": "run ls",
        "tools": [{"type": "function", "name": "shell",
                   "parameters": {"type": "object",
                                  "properties": {"command": {"type": "string"}}}}],
        "stream": False})
    assert r.status_code == 200
    fc = [o for o in r.json()["output"] if o["type"] == "function_call"][0]
    assert fc["name"] == "shell" and fc["call_id"] == "call_1"
    assert fc["arguments"] == '{"command":"ls"}'


def test_responses_profiled_route(client, host):
    _seed(host, _ok("cheap"))
    r = client.post("/cheap_explore/v1/responses", json={
        "model": "ignored", "input": "hi", "stream": False})
    assert r.status_code == 200
    assert r.json()["output"][0]["content"][0]["text"] == "cheap"


def test_responses_router_error_maps_to_status(client, host):
    # no mocks set -> candidates fail -> router returns an error
    r = client.post("/v1/responses", json={
        "model": "policy:auto", "input": "hi", "stream": False})
    assert r.status_code >= 400
    assert r.json()["error"]["type"] == "router_error"
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/albert/dev/unhardcoded && pytest tests/test_responses_shim.py -q`
Expected: FAIL — `404 Not Found` (route not registered) / missing `ResponsesRequest`.

- [ ] **Step 3: Add the `ResponsesRequest` model**

In `shim.py`, after the `ChatRequest` class (before `PolicyRankRequest`, ~line 96) add:

```python
class ResponsesRequest(BaseModel):
    """Permissive OpenAI /v1/responses body. Unknown fields are kept
    (extra="allow") so Responses params the shim does not read (reasoning,
    include, store, parallel_tool_calls, prompt_cache_key, text,
    previous_response_id, …) never break the request."""
    model_config = ConfigDict(extra="allow")

    model: str = ""
    input: Any = None             # str | list[item]
    instructions: str | None = None
    tools: list[dict] | None = None
    tool_choice: Any = None
    stream: bool = False
    max_output_tokens: int | None = None
    temperature: float | None = None
    policy_ir: list | None = None
    session: str | None = None
    caller: str | None = None
```

- [ ] **Step 4: Add the routes + handlers inside `create_app`**

In `shim.py`, immediately after the `chat_completions_profiled` route and its
`_session_from_header` helper (~line 834, before `_handle_chat`), add:

```python
    @app.post("/v1/responses")
    async def responses(req: ResponsesRequest, request: Request):
        _session_from_header(req, request)
        return await _handle_responses(req)

    @app.post("/{profile_name}/v1/responses")
    async def responses_profiled(profile_name: str, req: ResponsesRequest,
                                 request: Request):
        _session_from_header(req, request)
        return await _handle_responses(req, profile_name=profile_name)

    def _responses_object_with_router(result: dict, req: ResponsesRequest,
                                      response_id: str | None = None) -> dict:
        import responses_api as _rapi
        obj = _rapi.result_to_responses_object(result, req.model or "",
                                               response_id=response_id)
        resp = result.get("response") or {}
        chosen = result.get("chosen") or {}
        obj["x_router"] = {
            "provider": chosen.get("provider_id"),
            "model_family": chosen.get("model_family"),
            "served_model_id": chosen.get("served_model_id"),
            "price_in": chosen.get("price_in"),
            "price_out": chosen.get("price_out"),
            "cost_usd": _executed_cost_usd(result, subscription_providers),
            "tokens_cached": resp.get("tokens_cached"),
            "policy_fingerprint": (result.get("trace") or {}).get("policy_fingerprint"),
            "decision_trace": _trim_trace(result.get("trace")),
            "compact": _compact_suggested(resp),
        }
        if req.session:
            acc = route_session_meter.observe(
                req.session,
                tokens_in=resp.get("tokens_in") or 0,
                tokens_out=resp.get("tokens_out") or 0,
                tokens_cached=resp.get("tokens_cached") or 0,
                cost_usd=obj["x_router"]["cost_usd"] or 0.0,
                owner=req.caller)
            obj["x_router"]["session_acc"] = acc
        return obj

    async def _handle_responses(req: ResponsesRequest, profile_name: str | None = None):
        import responses_api as _rapi
        chatreq = ChatRequest(
            model=req.model or "",
            messages=_rapi.input_to_messages(req.input, req.instructions),
            tools=_rapi.tools_to_chat(req.tools),
            tool_choice=_rapi.tool_choice_to_chat(req.tool_choice),
            temperature=req.temperature,
            max_tokens=req.max_output_tokens,
            policy_ir=req.policy_ir,
            session=req.session,
        )
        contract = _request_to_contract(chatreq, default_profile, default_max_tokens)
        if profile_name is not None:
            contract["profile"] = profile_name

        if not req.stream:
            try:
                result = await host.execute_async(contract)
            except Exception as exc:
                admission = _policy_admission_error(exc)
                if admission is None:
                    raise
                return _invalid_policy_response(admission)
            if not result.get("ok"):
                return _openai_error_from_router(result)
            return _responses_object_with_router(result, req)
        return await _handle_responses_stream(contract, req)

    async def _handle_responses_stream(contract: dict, req: ResponsesRequest):
        import responses_api as _rapi
        task = asyncio.create_task(host.execute_async(contract))
        done, _ = await asyncio.wait({task}, timeout=_EARLY_FAIL_S,
                                     return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            try:
                result = task.result()
            except Exception as exc:
                admission = _policy_admission_error(exc)
                if admission is None:
                    raise
                return _invalid_policy_response(admission)
            if not result.get("ok"):
                return _openai_error_from_router(result)
            obj = _responses_object_with_router(result, req)

            async def gen_ready():
                yield _rapi.responses_created_event(obj)
                for ev in _rapi.responses_sse_events(obj):
                    yield ev
            return StreamingResponse(gen_ready(), media_type="text/event-stream")

        async def gen_running():
            rid = _rapi._new_id("resp")
            yield _rapi.responses_created_event({"id": rid, "model": req.model or ""})
            while not task.done():
                await asyncio.wait({task}, timeout=_HEARTBEAT_S)
                if not task.done():
                    yield _streaming.HEARTBEAT
            try:
                result = task.result()
            except Exception as exc:
                admission = _policy_admission_error(exc)
                yield _rapi.responses_failed_event(
                    rid, admission or f"responses error: {exc}",
                    "invalid_policy" if admission else "internal_error")
                return
            if not result.get("ok"):
                err = str(result.get("error") or "router error")
                yield _rapi.responses_failed_event(rid, err, str(result.get("error") or "error"))
                return
            obj = _responses_object_with_router(result, req, response_id=rid)
            for ev in _rapi.responses_sse_events(obj):
                yield ev
        return StreamingResponse(gen_running(), media_type="text/event-stream")
```

- [ ] **Step 5: Run to verify pass**

Run: `cd /Users/albert/dev/unhardcoded && pytest tests/test_responses_shim.py -q`
Expected: PASS (all integration tests).

- [ ] **Step 6: Run the whole suite (no regressions)**

Run: `cd /Users/albert/dev/unhardcoded && pytest tests -q`
Expected: PASS (existing tests unaffected; new tests green).

- [ ] **Step 7: Document the endpoint**

In `README.md`, find the endpoint list (the lines describing `/v1/chat/completions`, `/v1/compact`, `/v1/models`) and add a sibling line:

```markdown
- `POST /v1/responses` (and `POST /{profile}/v1/responses`) — OpenAI *Responses* API
  surface for Responses-only clients (e.g. the Codex CLI). Translates to the same
  routing/policy/metering as chat-completions; streams Responses SSE.
```

- [ ] **Step 8: Commit**

```bash
cd /Users/albert/dev/unhardcoded
git add shim.py tests/test_responses_shim.py README.md
git commit -m "feat(responses): /v1/responses + /{profile}/v1/responses endpoints"
```

---

## Acceptance (manual, after the branch is green — gated, not part of TDD)

Point Codex CLI at the router and confirm end-to-end (this is the original goal):

1. Run the shim locally (or against the live router once deployed).
2. Codex config: `model_provider` base_url `…/v1`, `wire_api = "responses"`, `env_key` for the router key.
3. `codex exec "reply with exactly: hello from codex"` → a real reply (no `404 /v1/responses`).
4. A tool-using run (`codex exec "list the files here"`) → Codex issues a `shell` function call and continues.

If Codex sends a native tool type (e.g. `local_shell`) the router can't route, capture it from the failure and decide (map or instruct Codex to use function tools) — noted as the one open empirical point in the spec.

## Self-Review

- **Spec coverage:** request-in (Task 1), result→object (Task 2), SSE (Task 3), routes + streaming + profiled + metering + error mapping (Task 4), docs (Task 4). Out-of-scope items (previous_response_id, true token streaming, image parts) are intentionally not implemented and documented in the spec.
- **Placeholder scan:** none — every step carries complete code.
- **Type consistency:** `result_to_responses_object` signature, `responses_sse_events`/`responses_created_event`/`responses_failed_event`, `input_to_messages`/`tools_to_chat`/`tool_choice_to_chat`, and `_new_id` are used in Task 4 exactly as defined in Tasks 1-3. `ResponsesRequest` field names (`input`, `instructions`, `max_output_tokens`, `session`, `caller`) match their reads in `_handle_responses` and `_responses_object_with_router`.
