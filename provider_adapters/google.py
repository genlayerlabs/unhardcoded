"""Google Gemini generateContent provider adapter."""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import AsyncExitStack
from typing import Any, Callable
from urllib.parse import quote

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


def _tool_name_by_id(messages: list[dict]) -> dict:
    """Map each assistant tool_call id to its function name. A later OpenAI
    `tool` result carries only the id, but Gemini's functionResponse keys by
    name, so the name must be recovered from the call that produced the id."""
    out = {}
    for msg in messages or []:
        for tc in msg.get("tool_calls") or []:
            tid, name = tc.get("id"), (tc.get("function") or {}).get("name")
            if tid and name:
                out[tid] = name
    return out


def _tool_response_payload(text: str) -> dict:
    """Gemini functionResponse.response must be an object; wrap a non-object
    result string as {"result": ...}."""
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    return parsed if isinstance(parsed, dict) else {"result": text}


def _openai_messages_to_gemini(messages: list[dict]) -> tuple[list[dict], dict | None]:
    contents, system_parts = [], []
    name_by_id = _tool_name_by_id(messages)
    for msg in messages or []:
        role = msg.get("role")
        text = text_from_content(msg.get("content"))
        if role == "system":
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            # An OpenAI tool result -> a Gemini functionResponse part. Flattening
            # it to plain user text (the prior bug) lost the call linkage and
            # broke multi-turn tool conversations.
            name = name_by_id.get(msg.get("tool_call_id")) or msg.get("name") or "tool"
            contents.append({"role": "user", "parts": [{"functionResponse": {
                "name": name,
                "response": _tool_response_payload(text),
            }}]})
            continue
        if role == "assistant":
            # Carry tool_calls as functionCall parts; a tool-call-only turn has
            # no text and must NOT be dropped.
            parts: list[dict] = []
            if text:
                parts.append({"text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                parts.append({"functionCall": {
                    "name": fn.get("name") or "",
                    "args": json_args(fn.get("arguments")),
                }})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if text:  # user (and any other producer role)
            contents.append({"role": "user", "parts": [{"text": text}]})
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


def _google_request(
    request: dict,
    token: str,
    extra_headers: dict,
    *,
    stream: bool = False,
) -> tuple[str, dict, dict]:
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
    suffix = "streamGenerateContent?alt=sse" if stream else "generateContent"
    # The API key rides the x-goog-api-key header, never the URL query —
    # a `?key=<secret>` URL leaks into error_message/logs/traces on
    # timeout/network errors (§3: keys are never logged or echoed).
    url = f"{base}/{quote(model_path, safe='/')}:{suffix}"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": token,
        **extra_headers,
    }
    return url, body, headers


async def stream_google(
    request: dict,
    emit,
    *,
    env_get: Callable[[str], str | None] | None = None,
    timeout_s: float = 30.0,
    extra_headers: dict[str, str] | None = None,
    client: Any = None,
) -> dict:
    """Native Gemini streamGenerateContent backend for api_kind='google'."""
    import httpx

    token, error = auth_token(request, env_get or os.environ.get, "GEMINI_API_KEY")
    if error is not None:
        return error
    url, body, headers = _google_request(
        request, token, dict(extra_headers or {}), stream=True)
    timeout = (request.get("timeout_ms") or int(timeout_s * 1000)) / 1000.0
    if client is None:
        client = httpx.AsyncClient()

    t0 = time.monotonic()
    saw_output = False
    emitted = False
    text_parts: list[str] = []
    tool_calls: list[dict] = []
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
                    chunk = json.loads(data)
                except ValueError:
                    continue
                raw_model = chunk.get("modelVersion") or raw_model
                if chunk.get("usageMetadata"):
                    usage = chunk["usageMetadata"]
                for cand in chunk.get("candidates") or []:
                    finish_reason = cand.get("finishReason") or finish_reason
                    for part in ((cand.get("content") or {}).get("parts") or []):
                        if part.get("text"):
                            saw_output = True
                            emitted = True
                            text_parts.append(part["text"])
                            await emit(part["text"])
                        if part.get("functionCall"):
                            saw_output = True
                            fc = part["functionCall"]
                            tool_calls.append({
                                "id": fc.get("id") or fc.get("name"),
                                "type": "function",
                                "function": {
                                    "name": fc.get("name") or "",
                                    "arguments": json.dumps(fc.get("args") or {}),
                                },
                            })
    except Exception as exc:  # noqa: BLE001
        partial = "".join(text_parts)
        if emitted:
            return _err("stream_interrupted", 0, _latency(),
                        f"{type(exc).__name__}: {exc} (partial: {partial[:200]!r})")
        return _err("network_error", 0, _latency(), f"{type(exc).__name__}: {exc}")

    text = "".join(text_parts)
    if not text.strip() and not tool_calls:
        return _err("bad_response", 200, _latency(), "empty assistant content")
    return {
        "ok": True,
        "latency_ms": _latency(),
        "response": {
            "text": text,
            "tool_calls": tool_calls or None,
            "finish_reason": finish_reason,
            "tokens_in": usage.get("promptTokenCount"),
            "tokens_out": usage.get("candidatesTokenCount"),
            "tokens_total": usage.get("totalTokenCount"),
            "raw_model": raw_model,
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
        if request.get("first_token_timeout_ms") is not None:
            async def _ignore_delta(_delta: str) -> None:
                return None

            return await stream_google(
                request, _ignore_delta, env_get=_env_get, timeout_s=timeout_s,
                extra_headers=_extra, client=client)
        url, body, headers = _google_request(request, token, _extra)
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
