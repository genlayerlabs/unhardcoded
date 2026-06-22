"""
True streaming for the shim: stream variants of both wire backends, the
api_kind dispatcher for them, and OpenAI chat.completion.chunk encoding.

Contract with the router core (see the 2026-06-10 streaming spec): each
stream backend takes (request, emit) where `emit(text_delta)` is awaited per
content delta, and RETURNS the same complete-response dict the non-streaming
backends return — the core's fallback/retry/breaker/trace machinery consumes
that, untouched. The commit point (first emit) is the shim's business.

Pre-delta failures return classified errors WITHOUT emitting, so the core
falls through to the next candidate exactly as in non-streaming mode.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Awaitable, Callable

from llm_router_host import (
    _classify_from_map,
    _classify_status,
    _err,
    _prepare_openai_call,
)

Emit = Callable[[str], Awaitable[None]]


# ---- openai-compatible stream backend --------------------------------------

async def stream_openai_compatible(
    request: dict,
    emit: Emit,
    *,
    client: Any = None,
    env_get=None,
    extra_headers: dict | None = None,
    timeout_s: float = 45.0,
    token_providers: dict | None = None,
    provider_rules: dict[str, dict] | None = None,
) -> dict:
    import os

    prep, err = _prepare_openai_call(
        request, env_get or os.environ.get, dict(extra_headers or {}),
        timeout_s, token_providers)
    if err is not None:
        return err
    url, body, headers, timeout = prep
    body["stream"] = True
    rules = (provider_rules or {}).get(request.get("provider_id")) or {}

    if client is None:
        import httpx
        client = httpx.AsyncClient()

    t0 = time.monotonic()
    emitted = False
    text_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}
    finish_reason = None
    usage: dict = {}
    raw_model = None

    def _latency() -> int:
        return int((time.monotonic() - t0) * 1000)

    try:
        async with client.stream("POST", url, json=body, headers=headers,
                                 timeout=timeout) as resp:
            if not (200 <= resp.status_code < 300):
                raw = (await resp.aread()).decode("utf-8", "replace")[:500]
                kind = _classify_from_map(raw, rules.get("error_map")) \
                    or _classify_status(resp.status_code, raw)
                return _err(kind, resp.status_code, _latency(), raw)

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if raw_model is None:
                    raw_model = chunk.get("model")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    content = delta.get("content")
                    if content:
                        text_parts.append(content)
                        await emit(content)
                        emitted = True
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        acc = tool_calls_acc.setdefault(idx, {
                            "id": None, "type": "function",
                            "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            acc["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            acc["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            acc["function"]["arguments"] += fn["arguments"]
    except Exception as exc:  # noqa: BLE001 — classified below
        partial = "".join(text_parts)
        if emitted:
            return _err("stream_interrupted", 0, _latency(),
                        f"{type(exc).__name__}: {exc} (partial: {partial[:200]!r})")
        return _err("network_error", 0, _latency(), f"{type(exc).__name__}: {exc}")

    tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)] or None
    text = "".join(text_parts)
    if not text.strip() and not tool_calls:
        return _err("bad_response", 200, _latency(), "empty assistant content")
    return {
        "ok": True,
        "latency_ms": _latency(),
        "response": {
            "text": text,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens"),
            "tokens_total": usage.get("total_tokens"),
            "raw_model": raw_model,
        },
    }


# ---- codex stream backend ----------------------------------------------------

async def stream_codex(
    request: dict,
    emit: Emit,
    *,
    auth,
    client: Any = None,
    base_url: str | None = None,
    timeout_s: float = 120.0,
    extra_headers: dict | None = None,
    observe=None,
) -> dict:
    from codex_backend import (
        CODEX_BASE_URL,
        aggregate_codex_sse,
        build_codex_body,
        build_codex_headers,
    )

    def _notify(status: int, headers=None) -> None:
        if observe is None:
            return
        try:
            hdrs = {k.lower(): v for k, v in dict(headers or {}).items()
                    if re.search(r"ratelimit|usage|quota|percent", k, re.I)}
            observe({"status": status, "headers": hdrs, "ts": int(time.time())})
        except Exception:
            pass

    token = auth.access_token()
    if not token:
        _notify(0)
        return _err("auth_error", 0, 0, "no codex access token (run `codex login`)")

    body = build_codex_body(request)
    headers = build_codex_headers(token, auth.account_id(), extra_headers)
    url = (request.get("base_url") or base_url or CODEX_BASE_URL).rstrip("/") + "/responses"
    timeout = (request.get("timeout_ms") or int(timeout_s * 1000)) / 1000.0

    if client is None:
        import httpx
        client = httpx.AsyncClient()

    t0 = time.monotonic()
    emitted = False
    lines: list[str] = []

    def _latency() -> int:
        return int((time.monotonic() - t0) * 1000)

    try:
        async with client.stream("POST", url, json=body, headers=headers,
                                 timeout=timeout) as resp:
            _notify(resp.status_code, resp.headers)
            if resp.status_code == 401:
                return _err("auth_error", 401, _latency(), "codex token rejected")
            if resp.status_code == 429:
                return _err("rate_limit", 429, _latency(), "codex rate limited")
            if resp.status_code >= 400:
                detail = (await resp.aread()).decode("utf-8", "replace")[:500]
                return _err("server_error", resp.status_code, _latency(), detail)
            async for line in resp.aiter_lines():
                lines.append(line)
                if line.startswith("data:"):
                    data = line[len("data:"):].strip()
                    try:
                        ev = json.loads(data)
                    except ValueError:
                        continue
                    if ev.get("type") == "response.output_text.delta" and ev.get("delta"):
                        await emit(ev["delta"])
                        emitted = True
    except Exception as exc:  # noqa: BLE001
        if emitted:
            partial = aggregate_codex_sse(lines, _latency())
            text = ((partial.get("response") or {}).get("text") or "")[:200]
            return _err("stream_interrupted", 0, _latency(),
                        f"{type(exc).__name__}: {exc} (partial: {text!r})")
        return _err("network_error", 0, _latency(), f"{type(exc).__name__}: {exc}")

    # the battle-tested aggregator builds the complete response for the core
    return aggregate_codex_sse(lines, _latency())


# ---- dispatcher ----------------------------------------------------------------

def make_streaming_dispatcher(default, handlers: dict | None = None):
    """api_kind -> stream backend, mirroring make_api_kind_dispatcher but for
    (request, emit) callables."""
    _handlers = dict(handlers or {})

    async def dispatch(request: dict, emit: Emit) -> dict:
        handler = _handlers.get(request.get("api_kind", "openai_compatible"), default)
        return await handler(request, emit)

    return dispatch


# ---- OpenAI chat.completion.chunk encoding --------------------------------------

DONE_EVENT = "data: [DONE]\n\n"

# An SSE comment line: ignored by OpenAI-compatible clients (no `data:` prefix),
# but it is bytes on the wire, so it resets the idle timers of every hop — the
# 60s ALB idle timeout above all. Emitted periodically while a slow flow runs so
# the connection is never silent long enough to be cut mid-execution (the cause
# of dropped ensemble calls + empty Activity rows).
HEARTBEAT = ": keepalive\n\n"


def new_stream_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _chunk(stream_id: str, model: str, choice: dict, extra: dict | None = None) -> str:
    payload = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model or "",
        "choices": [dict(choice, index=0)],
    }
    if extra:
        payload.update(extra)
    return "data: " + json.dumps(payload) + "\n\n"


def encode_role_chunk(stream_id: str, model: str) -> str:
    return _chunk(stream_id, model, {"delta": {"role": "assistant"}, "finish_reason": None})


def encode_text_chunk(stream_id: str, model: str, text: str) -> str:
    return _chunk(stream_id, model, {"delta": {"content": text}, "finish_reason": None})


def encode_final_chunk(stream_id: str, model: str, finish_reason: str | None,
                       tool_calls, usage: dict | None, x_router: dict | None) -> str:
    delta: dict = {}
    if tool_calls:
        delta["tool_calls"] = tool_calls
    extra: dict = {}
    if usage:
        extra["usage"] = usage
    if x_router:
        extra["x_router"] = x_router
    return _chunk(stream_id, model,
                  {"delta": delta, "finish_reason": finish_reason or "stop"}, extra)


def encode_error_event(message: str, code: str | None = None) -> str:
    return "data: " + json.dumps({"error": {
        "message": message, "type": "router_error", "code": code}}) + "\n\n"
