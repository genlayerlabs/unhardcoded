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

from provider_adapters.anthropic import make_anthropic_async_call_provider  # noqa: E402
from provider_adapters.google import make_google_async_call_provider  # noqa: E402


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
        "models/gemini-3.1-pro-preview:generateContent"
    )
    # The key authenticates via header, never the URL query (§3).
    assert req["headers"]["x-goog-api-key"] == "gem-key"
    assert "key=" not in req["url"]
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


# --- multi-turn tool round-trip: the loop the ecosystem's agents actually run.
# An assistant tool-call turn followed by a tool result must reach the provider
# as native tool_use/tool_result (Anthropic) and functionCall/functionResponse
# (Gemini), keyed by the originating call so the conversation can continue.

_TOOL_TURN = [
    {"role": "user", "content": "look up abc"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "lookup",
                                  "arguments": '{"id": "abc"}'}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "value=42"},
]


@pytest.mark.asyncio
async def test_anthropic_native_tool_roundtrip():
    client = FakeClient(FakeResponse(200, {
        "model": "claude-sonnet-4-6", "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }))
    call = make_anthropic_async_call_provider(
        env_get={"ANTHROPIC_API_KEY": "k"}.get, client=client)
    result = await call({
        "served_model_id": "claude-sonnet-4-6",
        "messages": list(_TOOL_TURN), "tools": [TOOL],
    })
    assert result["ok"] is True
    sent = client.requests[0]["json"]["messages"]
    # the tool-call-only assistant turn is preserved as a tool_use block...
    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"][0] == {
        "type": "tool_use", "id": "call_1", "name": "lookup",
        "input": {"id": "abc"}}
    # ...and the result is a tool_result keyed by that same id, not plain text.
    assert sent[2] == {"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": "call_1", "content": "value=42"}]}


@pytest.mark.asyncio
async def test_google_native_tool_roundtrip():
    client = FakeClient(FakeResponse(200, {
        "modelVersion": "gemini-3.1-pro-preview",
        "candidates": [{"finishReason": "STOP",
                        "content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2,
                          "totalTokenCount": 7},
    }))
    call = make_google_async_call_provider(
        env_get={"GEMINI_API_KEY": "k"}.get, client=client)
    result = await call({
        "served_model_id": "gemini-3.1-pro-preview",
        "messages": list(_TOOL_TURN), "tools": [TOOL],
    })
    assert result["ok"] is True
    contents = client.requests[0]["json"]["contents"]
    assert contents[1] == {"role": "model", "parts": [
        {"functionCall": {"name": "lookup", "args": {"id": "abc"}}}]}
    # the result keys by function NAME (recovered from call_1), as Gemini wants.
    assert contents[2] == {"role": "user", "parts": [
        {"functionResponse": {"name": "lookup", "response": {"result": "value=42"}}}]}


class RaisingClient:
    """A client whose POST raises, to exercise the error path."""
    def __init__(self, exc):
        self._exc = exc

    async def post(self, *a, **k):
        raise self._exc


@pytest.mark.asyncio
async def test_google_native_never_leaks_key_in_url_or_errors():
    import httpx
    client = RaisingClient(httpx.TimeoutException("boom"))
    call = make_google_async_call_provider(
        env_get={"GEMINI_API_KEY": "super-secret-key"}.get, client=client)
    result = await call({
        "served_model_id": "gemini-3.1-pro-preview",
        "messages": [{"role": "user", "content": "hi"}],
    })
    # The guard is verified by what it must REJECT: a leaked key. On the error
    # path the secret must appear nowhere in the surfaced error_message.
    assert result["ok"] is False
    assert result["error_kind"] == "timeout"
    assert "super-secret-key" not in result["error_message"]
