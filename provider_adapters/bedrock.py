"""Amazon Bedrock Runtime provider adapter."""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Callable

from provider_adapters.common import (
    AsyncCallProviderHook,
    before_first_output,
    first_token_timeout_err,
    first_token_timeout_s,
    ignore_delta,
    json_args,
    text_from_content,
    _elapsed_ms,
    _err,
)


_AUTH_CODES = {
    "AccessDeniedException",
    "ExpiredTokenException",
    "InvalidSignatureException",
    "UnrecognizedClientException",
    "UnauthorizedException",
    "ValidationException:Credentials",
}
_RATE_CODES = {
    "ServiceQuotaExceededException",
    "ThrottlingException",
    "TooManyRequestsException",
}
_UNAVAILABLE_CODES = {
    "ModelNotReadyException",
    "ResourceNotFoundException",
}
_SERVER_CODES = {
    "InternalServerException",
    "ServiceUnavailableException",
}


def _aws_region(request: dict, env_get: Callable[[str], str | None]) -> str:
    return (
        request.get("region")
        or request.get("aws_region")
        or env_get("BEDROCK_REGION")
        or env_get("AWS_REGION")
        or env_get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def _bedrock_model_id(request: dict) -> str:
    return str((request.get("offer") or {}).get("wire_model_id")
               or request["served_model_id"])


def _openai_messages_to_bedrock(
    messages: list[dict],
) -> tuple[list[dict], list[dict] | None]:
    out: list[dict] = []
    system_parts: list[str] = []

    for msg in messages or []:
        role = msg.get("role")
        text = text_from_content(msg.get("content"))
        if role == "system":
            if text:
                system_parts.append(text)
            continue

        if role == "tool":
            tool_use_id = msg.get("tool_call_id") or msg.get("id")
            if not tool_use_id:
                continue
            out.append({
                "role": "user",
                "content": [{
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": text or ""}],
                    },
                }],
            })
            continue

        if role == "assistant":
            content: list[dict] = []
            if text:
                content.append({"text": text})
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name")
                if not name:
                    continue
                content.append({
                    "toolUse": {
                        "toolUseId": tc.get("id") or name,
                        "name": name,
                        "input": json_args(fn.get("arguments")),
                    },
                })
            if content:
                out.append({"role": "assistant", "content": content})
            continue

        if text:
            out.append({"role": "user", "content": [{"text": text}]})

    system = [{"text": "\n\n".join(system_parts)}] if system_parts else None
    return out, system


def _openai_tools_to_bedrock(tools: list[dict] | None) -> dict | None:
    specs = []
    for tool in tools or []:
        fn = tool.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        specs.append({
            "toolSpec": {
                "name": name,
                "description": fn.get("description") or "",
                "inputSchema": {
                    "json": fn.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            },
        })
    return {"tools": specs} if specs else None


def _parse_bedrock_response(data: dict, latency: int) -> dict:
    message = ((data.get("output") or {}).get("message") or {})
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in message.get("content") or []:
        if block.get("text"):
            text_parts.append(block["text"])
        tool = block.get("toolUse")
        if isinstance(tool, dict):
            tool_calls.append({
                "id": tool.get("toolUseId"),
                "type": "function",
                "function": {
                    "name": tool.get("name") or "",
                    "arguments": json.dumps(tool.get("input") or {}),
                },
            })

    text = "".join(text_parts)
    if not text.strip() and not tool_calls:
        return _err("bad_response", 200, latency, "empty assistant content")

    usage = data.get("usage") or {}
    return {
        "ok": True,
        "latency_ms": latency,
        "response": {
            "text": text,
            "tool_calls": tool_calls or None,
            "finish_reason": data.get("stopReason"),
            "tokens_in": usage.get("inputTokens"),
            "tokens_out": usage.get("outputTokens"),
            "tokens_total": usage.get("totalTokens"),
            "tokens_cached": usage.get("cacheReadInputTokens"),
            "raw_model": data.get("modelId"),
        },
    }


def _bedrock_request(request: dict) -> dict[str, Any]:
    messages, system = _openai_messages_to_bedrock(request.get("messages") or [])
    body: dict[str, Any] = {
        "modelId": _bedrock_model_id(request),
        "messages": messages,
    }
    if system:
        body["system"] = system
    inference = {}
    if request.get("max_tokens") is not None:
        inference["maxTokens"] = request["max_tokens"]
    if request.get("temperature") is not None:
        inference["temperature"] = request["temperature"]
    if inference:
        body["inferenceConfig"] = inference
    tools = _openai_tools_to_bedrock(request.get("tools"))
    if tools:
        body["toolConfig"] = tools
    return body


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        err = response.get("Error") or {}
        code = err.get("Code")
        if code:
            return str(code)
    return type(exc).__name__


def _classify_bedrock_error(exc: Exception) -> str:
    code = _error_code(exc)
    if code in _AUTH_CODES:
        return "auth_error"
    if code in _RATE_CODES:
        return "rate_limit"
    if code in _UNAVAILABLE_CODES:
        return "model_unavailable"
    if code in _SERVER_CODES:
        return "server_error"
    if code == "ValidationException":
        msg = str(exc).lower()
        if "isn’t supported" in msg or "isn't supported" in msg \
                or "inference profile" in msg:
            return "model_unavailable"
        if "context" in msg and ("length" in msg or "window" in msg):
            return "context_overflow"
        return "bad_request"
    if "timeout" in code.lower():
        return "timeout"
    return "unknown"


def _next_stream_event(events):
    try:
        return next(events)
    except StopIteration:
        return None


def _bedrock_client(region: str, timeout_s: float):
    import boto3
    from botocore.config import Config
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(read_timeout=timeout_s, connect_timeout=min(timeout_s, 10)),
    )


async def stream_bedrock(
    request: dict,
    emit,
    *,
    env_get: Callable[[str], str | None] | None = None,
    timeout_s: float = 30.0,
    client: Any = None,
    client_factory: Callable[[str], Any] | None = None,
) -> dict:
    """Native Bedrock ConverseStream backend for api_kind='bedrock'."""
    api_kind = request.get("api_kind")
    if api_kind != "bedrock":
        return _err("unsupported_api_kind", 0, 0,
                    f"api_kind={api_kind!r} not supported by Bedrock backend")

    _env_get = env_get or os.environ.get
    region = _aws_region(request, _env_get)
    bedrock = client or (client_factory(region) if client_factory
                         else _bedrock_client(region, timeout_s))
    body = _bedrock_request(request)
    t0 = time.monotonic()
    saw_output = False
    emitted = False
    text_parts: list[str] = []
    tool_calls_acc: dict[int, dict] = {}
    finish_reason = None
    usage: dict = {}
    first_timeout_s = first_token_timeout_s(request)

    def _latency() -> int:
        return _elapsed_ms(t0)

    def _saw_output() -> bool:
        return saw_output

    def _timeout_err() -> dict:
        return first_token_timeout_err(first_timeout_s, _latency())

    def _tool_call(idx: int) -> dict:
        return tool_calls_acc.setdefault(idx, {
            "id": None,
            "type": "function",
            "function": {"name": "", "arguments": ""},
        })

    try:
        try:
            data = await before_first_output(
                asyncio.to_thread(bedrock.converse_stream, **body),
                first_timeout_s, t0, _saw_output)
        except (asyncio.TimeoutError, TimeoutError):
            return _timeout_err()
        events = iter((data or {}).get("stream") or [])
        while True:
            try:
                event = await before_first_output(
                    asyncio.to_thread(_next_stream_event, events),
                    first_timeout_s, t0, _saw_output)
            except (asyncio.TimeoutError, TimeoutError):
                if not saw_output:
                    return _timeout_err()
                raise
            if event is None:
                break
            if "messageStop" in event:
                finish_reason = (event["messageStop"] or {}).get("stopReason") \
                    or finish_reason
                continue
            if "metadata" in event:
                usage = (event["metadata"] or {}).get("usage") or usage
                continue
            if "contentBlockStart" in event:
                payload = event["contentBlockStart"] or {}
                idx = payload.get("contentBlockIndex", 0)
                start = payload.get("start") or {}
                tool = start.get("toolUse")
                if isinstance(tool, dict):
                    saw_output = True
                    acc = _tool_call(idx)
                    acc["id"] = tool.get("toolUseId") or acc["id"]
                    if tool.get("name"):
                        acc["function"]["name"] = tool["name"]
                continue
            if "contentBlockDelta" in event:
                payload = event["contentBlockDelta"] or {}
                idx = payload.get("contentBlockIndex", 0)
                delta = payload.get("delta") or {}
                if delta.get("text"):
                    saw_output = True
                    emitted = True
                    text_parts.append(delta["text"])
                    await emit(delta["text"])
                tool = delta.get("toolUse")
                if isinstance(tool, dict):
                    saw_output = True
                    acc = _tool_call(idx)
                    if tool.get("toolUseId"):
                        acc["id"] = tool["toolUseId"]
                    if tool.get("name"):
                        acc["function"]["name"] = tool["name"]
                    if tool.get("input") is not None:
                        inp = tool["input"]
                        acc["function"]["arguments"] += (
                            inp if isinstance(inp, str) else json.dumps(inp)
                        )
                if delta.get("reasoningContent"):
                    saw_output = True
                continue
            error_key = next((k for k in event if k.endswith("Exception")), None)
            if error_key:
                return _err("server_error", 0, _latency(),
                            str(event.get(error_key) or event)[:500])
    except Exception as exc:  # noqa: BLE001
        if emitted:
            partial = "".join(text_parts)
            return _err("stream_interrupted", 0, _latency(),
                        f"{type(exc).__name__}: {exc} (partial: {partial[:200]!r})")
        return _err(_classify_bedrock_error(exc), 0, _latency(), str(exc)[:500])

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
            "tokens_in": usage.get("inputTokens"),
            "tokens_out": usage.get("outputTokens"),
            "tokens_total": usage.get("totalTokens"),
            "tokens_cached": usage.get("cacheReadInputTokens"),
            "raw_model": body.get("modelId"),
        },
    }


def make_bedrock_async_call_provider(
    env_get: Callable[[str], str | None] | None = None,
    timeout_s: float = 30.0,
    client: Any = None,
    client_factory: Callable[[str], Any] | None = None,
) -> AsyncCallProviderHook:
    """Native Bedrock Converse backend for api_kind='bedrock'."""
    _env_get = env_get or os.environ.get

    def _client(region: str):
        if client is not None:
            return client
        if client_factory is not None:
            return client_factory(region)
        return _bedrock_client(region, timeout_s)

    async def call(request: dict) -> dict:
        api_kind = request.get("api_kind")
        if api_kind != "bedrock":
            return _err("unsupported_api_kind", 0, 0,
                        f"api_kind={api_kind!r} not supported by Bedrock backend")
        if request.get("first_token_timeout_ms") is not None:
            return await stream_bedrock(
                request, ignore_delta, env_get=_env_get, timeout_s=timeout_s,
                client=client, client_factory=client_factory)

        body = _bedrock_request(request)

        region = _aws_region(request, _env_get)
        bedrock = _client(region)
        t0 = time.monotonic()
        try:
            data = await asyncio.to_thread(bedrock.converse, **body)
        except Exception as exc:  # noqa: BLE001
            return _err(_classify_bedrock_error(exc), 0, _elapsed_ms(t0),
                        str(exc)[:500])
        return _parse_bedrock_response(data, _elapsed_ms(t0))

    return call
