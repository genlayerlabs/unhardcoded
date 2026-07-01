"""Anthropic Messages API provider adapter."""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack
from typing import Any, Callable

from provider_adapters.common import (
    AsyncCallProviderHook,
    auth_token,
    before_first_output,
    first_token_timeout_err,
    first_token_timeout_s,
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
            "raw_model": data.get("model"),
        },
    }


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
        body["system"] = system
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

    t0 = time.monotonic()
    saw_output = False
    emitted = False
    text_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}
    finish_reason = None
    usage: dict = {}
    raw_model = None
    first_timeout_s = first_token_timeout_s(request)

    def _latency() -> int:
        return _elapsed_ms(t0)

    def _saw_output() -> bool:
        return saw_output

    def _timeout_err() -> dict:
        return first_token_timeout_err(first_timeout_s, _latency())

    try:
        async with AsyncExitStack() as stack:
            try:
                resp = await before_first_output(
                    stack.enter_async_context(
                        client.stream("POST", url, json=body, headers=headers,
                                      timeout=timeout)),
                    first_timeout_s, t0, _saw_output)
            except (asyncio.TimeoutError, TimeoutError):
                return _timeout_err()
            if not (200 <= resp.status_code < 300):
                raw = (await resp.aread()).decode("utf-8", "replace")[:500]
                return _err(_classify_status(resp.status_code, raw),
                            resp.status_code, _latency(), raw)

            lines = resp.aiter_lines().__aiter__()
            while True:
                try:
                    line = await before_first_output(
                        lines.__anext__(), first_timeout_s, t0, _saw_output)
                except StopAsyncIteration:
                    break
                except (asyncio.TimeoutError, TimeoutError):
                    if not saw_output:
                        return _timeout_err()
                    raise
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except ValueError:
                    continue
                etype = ev.get("type")
                if etype == "message_start":
                    msg = ev.get("message") or {}
                    raw_model = msg.get("model") or raw_model
                    usage.update(msg.get("usage") or {})
                elif etype == "content_block_start":
                    idx = ev.get("index", 0)
                    block = ev.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        saw_output = True
                        tool_calls_acc[idx] = {
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
                        saw_output = True
                        emitted = True
                        text_parts.append(delta["text"])
                        await emit(delta["text"])
                    elif delta.get("type") == "input_json_delta":
                        saw_output = True
                        acc = tool_calls_acc.setdefault(idx, {
                            "id": None,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if delta.get("partial_json"):
                            acc["function"]["arguments"] += delta["partial_json"]
                    elif str(delta.get("type") or "").endswith("_delta"):
                        saw_output = True
                elif etype == "message_delta":
                    delta = ev.get("delta") or {}
                    finish_reason = delta.get("stop_reason") or finish_reason
                    usage.update(ev.get("usage") or {})
                elif etype == "error":
                    err = ev.get("error") or {}
                    msg = err.get("message") or str(err or ev)
                    return _err(_classify_status(resp.status_code, msg),
                                resp.status_code, _latency(), msg[:500])
    except Exception as exc:  # noqa: BLE001
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
            "tokens_in": usage.get("input_tokens"),
            "tokens_out": usage.get("output_tokens"),
            "tokens_total": (
                (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
                if usage else None
            ),
            "raw_model": raw_model,
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
            async def _ignore_delta(_delta: str) -> None:
                return None

            return await stream_anthropic(
                request, _ignore_delta, env_get=_env_get, timeout_s=timeout_s,
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
