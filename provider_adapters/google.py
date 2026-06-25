"""Google Gemini generateContent provider adapter."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable
from urllib.parse import quote

from provider_adapters.common import (
    AsyncCallProviderHook,
    auth_token,
    text_from_content,
    _classify_status,
    _elapsed_ms,
    _err,
)


def _openai_messages_to_gemini(messages: list[dict]) -> tuple[list[dict], dict | None]:
    contents, system_parts = [], []
    for msg in messages or []:
        role = msg.get("role")
        text = text_from_content(msg.get("content"))
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
            continue
        contents.append({
            "role": "model" if role == "assistant" else "user",
            "parts": [{"text": text}],
        })
    system = {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
    return contents, system


def _openai_tools_to_gemini(tools: list[dict] | None) -> list[dict] | None:
    declarations = []
    for tool in tools or []:
        fn = tool.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        declarations.append({
            "name": name,
            "description": fn.get("description") or "",
            "parameters": fn.get("parameters")
            or {"type": "object", "properties": {}},
        })
    return [{"functionDeclarations": declarations}] if declarations else None


def _parse_gemini_response(data: dict, status: int, latency: int) -> dict:
    candidates = data.get("candidates") or []
    if not candidates:
        return _err(
            "content_filter" if data.get("promptFeedback") else "bad_response",
            status,
            latency,
            "no candidates in response",
        )
    cand = candidates[0]
    finish = cand.get("finishReason")
    if finish == "SAFETY":
        return _err("content_filter", status, latency, "blocked by provider filter")
    text_parts, tool_calls = [], []
    for part in ((cand.get("content") or {}).get("parts") or []):
        if part.get("text"):
            text_parts.append(part["text"])
        if part.get("functionCall"):
            fc = part["functionCall"]
            tool_calls.append({
                "id": fc.get("id") or fc.get("name"),
                "type": "function",
                "function": {
                    "name": fc.get("name") or "",
                    "arguments": json.dumps(fc.get("args") or {}),
                },
            })
    text = "".join(text_parts)
    if not text.strip() and not tool_calls:
        return _err("bad_response", status, latency, "empty assistant content")
    usage = data.get("usageMetadata") or {}
    return {
        "ok": True,
        "latency_ms": latency,
        "response": {
            "text": text,
            "tool_calls": tool_calls or None,
            "finish_reason": finish,
            "tokens_in": usage.get("promptTokenCount"),
            "tokens_out": usage.get("candidatesTokenCount"),
            "tokens_total": usage.get("totalTokenCount"),
            "raw_model": data.get("modelVersion"),
        },
    }


def make_google_async_call_provider(
    env_get: Callable[[str], str | None] | None = None,
    timeout_s: float = 30.0,
    extra_headers: dict[str, str] | None = None,
    client: Any = None,
) -> AsyncCallProviderHook:
    """Native Gemini generateContent backend for api_kind='google'."""
    import httpx

    _env_get = env_get or os.environ.get
    _extra = dict(extra_headers or {})

    async def call(request: dict) -> dict:
        token, error = auth_token(request, _env_get, "GEMINI_API_KEY")
        if error is not None:
            return error
        model = (request.get("offer") or {}).get("wire_model_id") \
            or request["served_model_id"]
        model_path = model if str(model).startswith("models/") else f"models/{model}"
        contents, system = _openai_messages_to_gemini(request.get("messages") or [])
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = system
        generation = {}
        if request.get("max_tokens") is not None:
            generation["maxOutputTokens"] = request["max_tokens"]
        if request.get("temperature") is not None:
            generation["temperature"] = request["temperature"]
        if generation:
            body["generationConfig"] = generation
        tools = _openai_tools_to_gemini(request.get("tools"))
        if tools:
            body["tools"] = tools
        base = (
            request.get("base_url")
            or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        url = (
            f"{base}/{quote(model_path, safe='/')}:generateContent"
            f"?key={quote(token, safe='')}"
        )
        headers = {"Content-Type": "application/json", **_extra}
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
            return _parse_gemini_response(resp.json(), resp.status_code, latency)
        except Exception as e:
            return _err("bad_response", resp.status_code, latency, f"json parse: {e}")

    return call
