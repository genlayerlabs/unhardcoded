"""Anthropic Messages API provider adapter."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from provider_adapters.common import (
    AsyncCallProviderHook,
    auth_token,
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
            **_extra,
        }
        url = (
            (request.get("base_url") or "https://api.anthropic.com/v1").rstrip("/")
            + "/messages"
        )
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
