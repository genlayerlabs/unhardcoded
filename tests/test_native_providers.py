"""
Native provider adapters for provider api_kind values that are not
OpenAI-compatible. No network: clients and responses are faked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router_host import (  # noqa: E402
    make_anthropic_async_call_provider,
    make_google_async_call_provider,
)


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append({
            "url": url,
            "json": json,
            "headers": headers,
            "timeout": timeout,
        })
        return self.response


TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Lookup a value",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
}


@pytest.mark.asyncio
async def test_anthropic_native_handler_translates_openai_shape():
    client = FakeClient(FakeResponse(200, {
        "model": "claude-sonnet-4-6",
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "Use the tool."},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup",
             "input": {"id": "abc"}},
        ],
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }))
    call = make_anthropic_async_call_provider(
        env_get={"ANTHROPIC_TEST_KEY": "anth-key"}.get,
        client=client,
        timeout_s=12,
    )

    result = await call({
        "provider_id": "anthropic",
        "api_kind": "anthropic",
        "auth_env": "ANTHROPIC_TEST_KEY",
        "base_url": "https://api.anthropic.com/v1/",
        "served_model_id": "claude-sonnet-4-6",
        "offer": {"wire_model_id": "claude-sonnet-4-6"},
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
        "tools": [TOOL],
        "temperature": 0.2,
        "max_tokens": 128,
    })

    assert result["ok"] is True
    assert result["response"]["text"] == "Use the tool."
    assert result["response"]["tokens_total"] == 18
    assert result["response"]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"id": "abc"}',
    }

    req = client.requests[0]
    assert req["url"] == "https://api.anthropic.com/v1/messages"
    assert req["headers"]["x-api-key"] == "anth-key"
    assert req["headers"]["anthropic-version"] == "2023-06-01"
    assert req["timeout"] == 12
    assert req["json"]["model"] == "claude-sonnet-4-6"
    assert req["json"]["system"] == "Be terse."
    assert req["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert req["json"]["tools"][0]["input_schema"]["required"] == ["id"]


@pytest.mark.asyncio
async def test_google_native_handler_translates_openai_shape():
    client = FakeClient(FakeResponse(200, {
        "modelVersion": "gemini-3.1-pro-preview",
        "candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [
                {"text": "Done."},
                {"functionCall": {"name": "lookup", "args": {"id": "abc"}}},
            ]},
        }],
        "usageMetadata": {
            "promptTokenCount": 13,
            "candidatesTokenCount": 5,
            "totalTokenCount": 18,
        },
    }))
    call = make_google_async_call_provider(
        env_get={"GEMINI_TEST_KEY": "gem-key"}.get,
        client=client,
        timeout_s=9,
    )

    result = await call({
        "provider_id": "gemini",
        "api_kind": "google",
        "auth_env": "GEMINI_TEST_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "served_model_id": "gemini-3.1-pro-preview",
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        "tools": [TOOL],
        "temperature": 0.1,
        "max_tokens": 64,
    })

    assert result["ok"] is True
    assert result["response"]["text"] == "Done."
    assert result["response"]["raw_model"] == "gemini-3.1-pro-preview"
    assert result["response"]["tokens_total"] == 18
    assert result["response"]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"id": "abc"}',
    }

    req = client.requests[0]
    assert req["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-3.1-pro-preview:generateContent?key=gem-key"
    )
    assert req["timeout"] == 9
    assert req["json"]["systemInstruction"] == {
        "parts": [{"text": "Be terse."}],
    }
    assert req["json"]["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "hello"}]},
    ]
    assert req["json"]["generationConfig"] == {
        "maxOutputTokens": 64,
        "temperature": 0.1,
    }
    assert req["json"]["tools"][0]["functionDeclarations"][0]["name"] == "lookup"


@pytest.mark.asyncio
async def test_native_handlers_require_provider_credentials():
    call = make_google_async_call_provider(env_get={}.get, client=FakeClient(None))

    result = await call({
        "provider_id": "gemini",
        "api_kind": "google",
        "served_model_id": "gemini-3.1-pro-preview",
        "messages": [{"role": "user", "content": "hi"}],
    })

    assert result["ok"] is False
    assert result["error_kind"] == "auth_error"
    assert "GEMINI_API_KEY" in result["error_message"]
