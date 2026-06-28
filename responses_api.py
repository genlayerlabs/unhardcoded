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


# ---- id helpers (shared by the response-out half) -------------------------

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"
