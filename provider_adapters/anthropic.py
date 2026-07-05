"""Anthropic Messages API provider adapter."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from provider_adapters.common import (
    CACHE_CONTROL,
    AsyncCallProviderHook,
    StreamAcc,
    auth_token,
    drive_http_sse,
    ignore_delta,
    json_args,
    text_from_content,
    _classify_status,
    _elapsed_ms,
    _err,
)


def _openai_messages_to_anthropic(
    messages: list[dict],
) -> tuple[list[dict], str | None]:
    out, system_parts = [], []
    for msg in messages or []:
        role = msg.get("role")
        text = text_from_content(msg.get("content"))
        if role == "system":
            if text:
                system_parts.append(text)
        elif role == "tool":
            # An OpenAI tool result -> an Anthropic tool_result block, keyed
            # by the originating tool_use id. Dropping it (the prior bug)
            # broke every multi-turn tool conversation.
            out.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id"),
                "content": text,
            }]})
        elif role == "assistant":
            # An assistant turn may be text, tool_use, or both. A tool-call-only
            # turn has no text and must NOT be dropped, or the following
            # tool_result references a tool_use that was never sent (API 400).
            blocks: list[dict] = []
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id"),
                    "name": fn.get("name") or "",
                    "input": json_args(fn.get("arguments")),
                })
            if not blocks:
                continue
            # Keep the plain-string shape when it is text-only (what the
            # existing tests pin); use content blocks once tool_use appears.
            if len(blocks) == 1 and blocks[0]["type"] == "text":
                out.append({"role": "assistant", "content": text})
            else:
                out.append({"role": "assistant", "content": blocks})
        elif role == "user":
            if text:
                out.append({"role": "user", "content": text})
    return out, "\n\n".join(system_parts) if system_parts else None


def _openai_tools_to_anthropic(tools: list[dict] | None) -> list[dict] | None:
    out = []
    for tool in tools or []:
        fn = tool.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "description": fn.get("description") or "",
            "input_schema": fn.get("parameters")
            or {"type": "object", "properties": {}},
        })
    return out or None


def _parse_anthropic_response(data: dict, status: int, latency: int) -> dict:
    text_parts, tool_calls = [], []
    for block in data.get("content") or []:
        if block.get("type") == "text" and block.get("text"):
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id"),
                "type": "function",
                "function": {
                    "name": block.get("name") or "",
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })
    text = "".join(text_parts)
    if not text.strip() and not tool_calls:
        return _err("bad_response", status, latency, "empty assistant content")
    usage = data.get("usage") or {}
    return {
        "ok": True,
        "latency_ms": latency,
        "response": {
            "text": text,
            "tool_calls": tool_calls or None,
            "finish_reason": data.get("stop_reason"),
            "tokens_in": usage.get("input_tokens"),
            "tokens_out": usage.get("output_tokens"),
            "tokens_total": (
                (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
                if usage else None
            ),
            # cache reads bill at a fraction — without this the meter (and
            # the cost discount) is blind to the cache the router now
            # requests via cache_control (#74)
            "tokens_cached": usage.get("cache_read_input_tokens"),
            "raw_model": data.get("model"),
        },
    }


def _tag_last_message(messages: list[dict]) -> None:
    """Rolling prompt-cache breakpoint on the LAST message (#74): the next
    call's prefix then matches everything up to and including this turn, so
    an agentic loop re-reads its history from cache instead of re-buying it.
    Two breakpoints total with the system block — well under Anthropic's 4.
    Safe to mutate: `messages` is freshly built by the translator above."""
    if not messages:
        return
    last = messages[-1]
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content,
                            "cache_control": dict(CACHE_CONTROL)}]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1]["cache_control"] = dict(CACHE_CONTROL)


def _anthropic_request(request: dict, token: str, extra_headers: dict) -> tuple[str, dict, dict]:
    messages, system = _openai_messages_to_anthropic(
        request.get("messages") or [])
    body: dict[str, Any] = {
        "model": (request.get("offer") or {}).get("wire_model_id")
        or request["served_model_id"],
        "messages": messages,
        "max_tokens": request.get("max_tokens") or 4096,
    }
    if system:
        # Prompt-cache breakpoint on the system block (#74): anthropic caching
        # is opt-in per request — without markers every call re-pays the whole
        # conversation prefix at full input price (cache reads bill ~10%).
        body["system"] = [{"type": "text", "text": system,
                           "cache_control": dict(CACHE_CONTROL)}]
    _tag_last_message(messages)
    if request.get("temperature") is not None:
        body["temperature"] = request["temperature"]
    tools = _openai_tools_to_anthropic(request.get("tools"))
    if tools:
        body["tools"] = tools
    headers = {
        "Content-Type": "application/json",
        "x-api-key": token,
        "anthropic-version": "2023-06-01",
        **extra_headers,
    }
    url = (
        (request.get("base_url") or "https://api.anthropic.com/v1").rstrip("/")
        + "/messages"
    )
    return url, body, headers


async def stream_anthropic(
    request: dict,
    emit,
    *,
    env_get: Callable[[str], str | None] | None = None,
    timeout_s: float = 30.0,
    extra_headers: dict[str, str] | None = None,
    client: Any = None,
) -> dict:
    """Native Anthropic streaming backend for api_kind='anthropic'."""
    import httpx

    token, error = auth_token(request, env_get or os.environ.get, "ANTHROPIC_API_KEY")
    if error is not None:
        return error
    url, body, headers = _anthropic_request(request, token, dict(extra_headers or {}))
    body["stream"] = True
    timeout = (request.get("timeout_ms") or int(timeout_s * 1000)) / 1000.0
    if client is None:
        client = httpx.AsyncClient()

    async def _on_event(ev: dict, acc: StreamAcc) -> "dict | None":
        etype = ev.get("type")
        if etype == "message_start":
            msg = ev.get("message") or {}
            acc.raw_model = msg.get("model") or acc.raw_model
            acc.usage.update(msg.get("usage") or {})
        elif etype == "content_block_start":
            idx = ev.get("index", 0)
            block = ev.get("content_block") or {}
            if block.get("type") == "tool_use":
                acc.saw_output = True
                if acc.tool_calls is None:
                    acc.tool_calls = {}
                acc.tool_calls[idx] = {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name") or "",
                        "arguments": (
                            json.dumps(block.get("input"))
                            if block.get("input") else ""
                        ),
                    },
                }
        elif etype == "content_block_delta":
            idx = ev.get("index", 0)
            delta = ev.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                acc.saw_output = True
                acc.emitted = True
                acc.text_parts.append(delta["text"])
                await emit(delta["text"])
            elif delta.get("type") == "input_json_delta":
                acc.saw_output = True
                if acc.tool_calls is None:
                    acc.tool_calls = {}
                tc = acc.tool_calls.setdefault(idx, {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                })
                if delta.get("partial_json"):
                    tc["function"]["arguments"] += delta["partial_json"]
            elif str(delta.get("type") or "").endswith("_delta"):
                acc.saw_output = True
        elif etype == "message_delta":
            delta = ev.get("delta") or {}
            acc.finish_reason = delta.get("stop_reason") or acc.finish_reason
            acc.usage.update(ev.get("usage") or {})
        elif etype == "error":
            e = ev.get("error") or {}
            msg = e.get("message") or str(e or ev)
            return _err(_classify_status(acc.status, msg), acc.status,
                        acc.latency(), msg[:500])
        return None

    acc, error = await drive_http_sse(
        client=client, url=url, body=body, headers=headers, timeout=timeout,
        request=request, on_event=_on_event)
    if error is not None:
        return error

    by_index = acc.tool_calls or {}
    tool_calls = [by_index[i] for i in sorted(by_index)] or None
    text = "".join(acc.text_parts)
    if not text.strip() and not tool_calls:
        return _err("bad_response", 200, acc.latency(), "empty assistant content")
    usage = acc.usage
    return {
        "ok": True,
        "latency_ms": acc.latency(),
        "response": {
            "text": text,
            "tool_calls": tool_calls,
            "finish_reason": acc.finish_reason,
            "tokens_in": usage.get("input_tokens"),
            "tokens_out": usage.get("output_tokens"),
            "tokens_total": (
                (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
                if usage else None
            ),
            # cache_read_input_tokens rides message_start's usage and
            # survives the acc merge — read it or the stream path loses it
            "tokens_cached": usage.get("cache_read_input_tokens"),
            "raw_model": acc.raw_model,
        },
    }


def make_anthropic_async_call_provider(
    env_get: Callable[[str], str | None] | None = None,
    timeout_s: float = 30.0,
    extra_headers: dict[str, str] | None = None,
    client: Any = None,
) -> AsyncCallProviderHook:
    """Native Anthropic Messages API backend for api_kind='anthropic'."""
    import httpx

    _env_get = env_get or os.environ.get
    _extra = dict(extra_headers or {})

    async def call(request: dict) -> dict:
        token, error = auth_token(request, _env_get, "ANTHROPIC_API_KEY")
        if error is not None:
            return error
        if request.get("first_token_timeout_ms") is not None:
            return await stream_anthropic(
                request, ignore_delta, env_get=_env_get, timeout_s=timeout_s,
                extra_headers=_extra, client=client)
        url, body, headers = _anthropic_request(request, token, _extra)
        timeout = (request.get("timeout_ms") or int(timeout_s * 1000)) / 1000.0
        t0 = time.monotonic()
        try:
            if client is not None:
                resp = await client.post(url, json=body, headers=headers, timeout=timeout)
            else:
                async with httpx.AsyncClient() as c:
                    resp = await c.post(url, json=body, headers=headers, timeout=timeout)
        except httpx.TimeoutException:
            return _err("timeout", 0, _elapsed_ms(t0), f"POST {url} timed out")
        except (httpx.NetworkError, httpx.RequestError) as e:
            return _err("network_error", 0, _elapsed_ms(t0), str(e))
        latency = _elapsed_ms(t0)
        if not (200 <= resp.status_code < 300):
            return _err(_classify_status(resp.status_code, getattr(resp, "text", "")),
                        resp.status_code, latency, getattr(resp, "text", "")[:500])
        try:
            return _parse_anthropic_response(resp.json(), resp.status_code, latency)
        except Exception as e:
            return _err("bad_response", resp.status_code, latency, f"json parse: {e}")

    return call
