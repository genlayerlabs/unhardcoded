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
        # Codex sends a `developer` role item (the OpenAI "developer" ≈ system
        # role). Many chat-completions providers only accept system/user/
        # assistant/tool, so normalize developer -> system for portability.
        if role == "developer":
            role = "system"
        messages.append({"role": role, "content": _part_text(it.get("content"))})

    return messages


def tools_to_chat(tools: Any) -> "list[dict] | None":
    """Responses tools (FLAT {type:"function", name, ...}) -> chat-completions
    tools (NESTED {type:"function", function:{name,...}}). Inverse of
    codex_backend._to_responses_tools.

    Unknown/native tool types (local_shell, web_search, custom, …) are DROPPED,
    not forwarded: a plain chat-completions provider rejects any tool whose
    type != "function", which would 400 the WHOLE turn (killing even the text
    answer). Dropping lets the text turn through; the shim logs what it dropped
    so a tool the model actually needs is visible (see _handle_responses)."""
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
        # else: native/unknown tool type -> dropped (see docstring)
    return out or None


def dropped_tool_types(tools: Any) -> list[str]:
    """The `type`s tools_to_chat dropped (non-function / nameless-function),
    so the shim can log when a tool the model may need was discarded."""
    if not tools:
        return []
    return [str(t.get("type"))
            for t in tools
            if isinstance(t, dict) and not (t.get("type") == "function" and t.get("name"))]


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


# ---- id helpers (shared by the response-out half) -------------------------

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


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
    # Standard Responses cache field so a client's context accounting treats
    # cache reads correctly (mirrors the chat path's usage.prompt_tokens_details).
    if resp.get("tokens_cached"):
        usage["input_tokens_details"] = {"cached_tokens": resp["tokens_cached"]}
    if usage:
        obj["usage"] = usage
    return obj


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
        frame = _sse(event, {"type": event, "sequence_number": seq, **data})
        seq += 1
        return frame

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
