"""Amazon Bedrock Runtime provider adapter."""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Callable

from provider_adapters.common import (
    AsyncCallProviderHook,
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
        import boto3
        from botocore.config import Config
        return boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(read_timeout=timeout_s, connect_timeout=min(timeout_s, 10)),
        )

    async def call(request: dict) -> dict:
        api_kind = request.get("api_kind")
        if api_kind != "bedrock":
            return _err("unsupported_api_kind", 0, 0,
                        f"api_kind={api_kind!r} not supported by Bedrock backend")

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
