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
from provider_adapters.anthropic import stream_anthropic  # noqa: E402
from provider_adapters.bedrock import make_bedrock_async_call_provider  # noqa: E402
from provider_adapters.bedrock import stream_bedrock  # noqa: E402
from provider_adapters.google import make_google_async_call_provider  # noqa: E402
from provider_adapters.google import stream_google  # noqa: E402


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


class FakeStreamResponse:
    def __init__(self, status_code, lines=None, body=b"", open_delay=0, line_delay=0):
        self.status_code = status_code
        self._lines = lines or []
        self._body = body
        self._open_delay = open_delay
        self._line_delay = line_delay
        self.headers = {}

    async def __aenter__(self):
        if self._open_delay:
            import asyncio
            await asyncio.sleep(self._open_delay)
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        if self._line_delay:
            import asyncio
            await asyncio.sleep(self._line_delay)
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body


class FakeStreamClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def stream(self, method, url, json=None, headers=None, timeout=None):
        self.requests.append({
            "method": method,
            "url": url,
            "json": json,
            "headers": headers,
            "timeout": timeout,
        })
        return self.response


class FakeBedrockClient:
    def __init__(self, response, stream_delay=0):
        self.response = response
        self.stream_delay = stream_delay
        self.requests = []
        self.methods = []

    def converse(self, **kwargs):
        self.methods.append("converse")
        self.requests.append(kwargs)
        return self.response

    def converse_stream(self, **kwargs):
        if self.stream_delay:
            import time
            time.sleep(self.stream_delay)
        self.methods.append("converse_stream")
        self.requests.append(kwargs)
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


def _sse(data: dict) -> str:
    return "data: " + json.dumps(data)


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
    # system + last message carry prompt-cache breakpoints (#74): anthropic
    # caching is opt-in per request; the router injects the markers
    assert req["json"]["system"] == [{"type": "text", "text": "Be terse.",
                                      "cache_control": {"type": "ephemeral"}}]
    assert req["json"]["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}]
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
async def test_bedrock_native_handler_translates_openai_shape():
    client = FakeBedrockClient({
        "output": {"message": {"role": "assistant", "content": [
            {"text": "Use the tool."},
            {"toolUse": {"toolUseId": "toolu_1", "name": "lookup",
                         "input": {"id": "abc"}}},
        ]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
    })
    call = make_bedrock_async_call_provider(
        env_get={"AWS_REGION": "us-east-1"}.get,
        client=client,
        timeout_s=12,
    )

    result = await call({
        "provider_id": "bedrock",
        "api_kind": "bedrock",
        "served_model_id": "us.anthropic.claude-sonnet-4-6",
        "offer": {"wire_model_id": "us.anthropic.claude-sonnet-4-6"},
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
    assert req["modelId"] == "us.anthropic.claude-sonnet-4-6"
    assert req["system"] == [{"text": "Be terse."}]
    assert req["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert req["inferenceConfig"] == {"maxTokens": 128, "temperature": 0.2}
    assert req["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["required"] == ["id"]


@pytest.mark.asyncio
async def test_anthropic_native_stream_emits_and_aggregates():
    deltas = []
    client = FakeStreamClient(FakeStreamResponse(200, [
        _sse({"type": "message_start", "message": {
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 11, "cache_read_input_tokens": 6},
        }}),
        _sse({"type": "content_block_delta", "index": 0,
              "delta": {"type": "text_delta", "text": "Use the tool."}}),
        _sse({"type": "content_block_start", "index": 1,
              "content_block": {"type": "tool_use", "id": "toolu_1",
                                "name": "lookup", "input": {}}}),
        _sse({"type": "content_block_delta", "index": 1,
              "delta": {"type": "input_json_delta",
                        "partial_json": '{"id": "abc"}'}}),
        _sse({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
              "usage": {"output_tokens": 7}}),
    ]))

    async def emit(delta):
        deltas.append(delta)

    result = await stream_anthropic({
        "provider_id": "anthropic",
        "api_kind": "anthropic",
        "auth_env": "ANTHROPIC_TEST_KEY",
        "served_model_id": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
    }, emit, env_get={"ANTHROPIC_TEST_KEY": "anth-key"}.get, client=client)

    assert deltas == ["Use the tool."]
    assert result["ok"] is True
    assert result["response"]["text"] == "Use the tool."
    assert result["response"]["tokens_total"] == 18
    # cache_read_input_tokens rides message_start and must survive to the end
    assert result["response"]["tokens_cached"] == 6
    assert result["response"]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"id": "abc"}',
    }
    assert client.requests[0]["json"]["stream"] is True


@pytest.mark.asyncio
async def test_google_native_stream_emits_and_aggregates():
    deltas = []
    client = FakeStreamClient(FakeStreamResponse(200, [
        _sse({"modelVersion": "gemini-3.1-pro-preview",
              "candidates": [{"content": {"parts": [{"text": "Done."}]}}]}),
        _sse({"candidates": [{"finishReason": "STOP",
              "content": {"parts": [
                  {"functionCall": {"name": "lookup", "args": {"id": "abc"}}},
              ]}}],
              "usageMetadata": {
                  "promptTokenCount": 13,
                  "candidatesTokenCount": 5,
                  "totalTokenCount": 18,
                  "cachedContentTokenCount": 4,
              }}),
    ]))

    async def emit(delta):
        deltas.append(delta)

    result = await stream_google({
        "provider_id": "gemini",
        "api_kind": "google",
        "auth_env": "GEMINI_TEST_KEY",
        "served_model_id": "gemini-3.1-pro-preview",
        "messages": [{"role": "user", "content": "hi"}],
    }, emit, env_get={"GEMINI_TEST_KEY": "gem-key"}.get, client=client)

    assert deltas == ["Done."]
    assert result["ok"] is True
    assert result["response"]["text"] == "Done."
    assert result["response"]["raw_model"] == "gemini-3.1-pro-preview"
    assert result["response"]["tokens_total"] == 18
    # Gemini implicit-cache hits (usageMetadata.cachedContentTokenCount)
    assert result["response"]["tokens_cached"] == 4
    assert result["response"]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"id": "abc"}',
    }
    assert client.requests[0]["url"].endswith(
        "/models/gemini-3.1-pro-preview:streamGenerateContent?alt=sse")


@pytest.mark.asyncio
async def test_bedrock_native_stream_emits_and_aggregates():
    deltas = []
    client = FakeBedrockClient({"stream": [
        {"contentBlockDelta": {"contentBlockIndex": 0,
                               "delta": {"text": "Use the tool."}}},
        {"contentBlockStart": {"contentBlockIndex": 1,
                               "start": {"toolUse": {
                                   "toolUseId": "toolu_1",
                                   "name": "lookup",
                               }}}},
        {"contentBlockDelta": {"contentBlockIndex": 1,
                               "delta": {"toolUse": {
                                   "input": '{"id": "abc"}',
                               }}}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 11, "outputTokens": 7,
                                "totalTokens": 18}}},
    ]})

    async def emit(delta):
        deltas.append(delta)

    result = await stream_bedrock({
        "provider_id": "bedrock",
        "api_kind": "bedrock",
        "served_model_id": "us.anthropic.claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
    }, emit, env_get={"AWS_REGION": "us-east-1"}.get, client=client)

    assert deltas == ["Use the tool."]
    assert result["ok"] is True
    assert result["response"]["text"] == "Use the tool."
    assert result["response"]["raw_model"] == "us.anthropic.claude-sonnet-4-6"
    assert result["response"]["tokens_total"] == 18
    assert result["response"]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": '{"id": "abc"}',
    }
    assert client.methods == ["converse_stream"]


@pytest.mark.asyncio
async def test_native_call_providers_use_streaming_when_first_token_timeout_present():
    anthropic = FakeStreamClient(FakeStreamResponse(200, [
        _sse({"type": "content_block_delta", "index": 0,
              "delta": {"type": "text_delta", "text": "a"}}),
    ]))
    call = make_anthropic_async_call_provider(
        env_get={"ANTHROPIC_API_KEY": "k"}.get, client=anthropic)
    result = await call({
        "served_model_id": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "first_token_timeout_ms": 100,
    })
    assert result["ok"] is True
    assert anthropic.requests[0]["json"]["stream"] is True

    google = FakeStreamClient(FakeStreamResponse(200, [
        _sse({"candidates": [{"content": {"parts": [{"text": "g"}]}}]}),
    ]))
    call = make_google_async_call_provider(
        env_get={"GEMINI_API_KEY": "k"}.get, client=google)
    result = await call({
        "served_model_id": "gemini-3.1-pro-preview",
        "messages": [{"role": "user", "content": "hi"}],
        "first_token_timeout_ms": 100,
    })
    assert result["ok"] is True
    assert ":streamGenerateContent?alt=sse" in google.requests[0]["url"]

    bedrock = FakeBedrockClient({"stream": [
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "b"}}},
    ]})
    call = make_bedrock_async_call_provider(
        env_get={"AWS_REGION": "us-east-1"}.get, client=bedrock)
    result = await call({
        "api_kind": "bedrock",
        "served_model_id": "us.anthropic.claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "first_token_timeout_ms": 100,
    })
    assert result["ok"] is True
    assert bedrock.methods == ["converse_stream"]


@pytest.mark.asyncio
async def test_native_streams_enforce_first_token_timeout_before_output():
    async def emit(_delta):
        raise AssertionError("no text should be emitted before timeout")

    anthropic = await stream_anthropic({
        "served_model_id": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "first_token_timeout_ms": 10,
    }, emit, env_get={"ANTHROPIC_API_KEY": "k"}.get,
        client=FakeStreamClient(FakeStreamResponse(
            200,
            [_sse({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "late"}})],
            open_delay=0.05,
        )))
    assert anthropic["error_kind"] == "timeout"

    google = await stream_google({
        "served_model_id": "gemini-3.1-pro-preview",
        "messages": [{"role": "user", "content": "hi"}],
        "first_token_timeout_ms": 10,
    }, emit, env_get={"GEMINI_API_KEY": "k"}.get,
        client=FakeStreamClient(FakeStreamResponse(
            200,
            [_sse({"candidates": [{"content": {"parts": [{"text": "late"}]}}]})],
            line_delay=0.05,
        )))
    assert google["error_kind"] == "timeout"

    bedrock = await stream_bedrock({
        "api_kind": "bedrock",
        "served_model_id": "us.anthropic.claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "hi"}],
        "first_token_timeout_ms": 10,
    }, emit, env_get={"AWS_REGION": "us-east-1"}.get,
        client=FakeBedrockClient({"stream": [
            {"contentBlockDelta": {"contentBlockIndex": 0,
                                   "delta": {"text": "late"}}},
        ]}, stream_delay=0.05))
    assert bedrock["error_kind"] == "timeout"


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
    # ...and the result is a tool_result keyed by that same id, not plain
    # text. As the LAST message it carries the rolling prompt-cache
    # breakpoint (#74) — in agentic loops the newest turn is almost always a
    # tool result, which is precisely where the next call's cached prefix
    # must end.
    assert sent[2] == {"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": "call_1", "content": "value=42",
        "cache_control": {"type": "ephemeral"}}]}


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


@pytest.mark.asyncio
async def test_bedrock_native_tool_roundtrip():
    client = FakeBedrockClient({
        "output": {"message": {"role": "assistant",
                               "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 2, "totalTokens": 7},
    })
    call = make_bedrock_async_call_provider(
        env_get={"AWS_REGION": "us-east-1"}.get, client=client)
    result = await call({
        "api_kind": "bedrock",
        "served_model_id": "us.anthropic.claude-sonnet-4-6",
        "messages": list(_TOOL_TURN), "tools": [TOOL],
    })
    assert result["ok"] is True
    messages = client.requests[0]["messages"]
    assert messages[1] == {"role": "assistant", "content": [{
        "toolUse": {"toolUseId": "call_1", "name": "lookup",
                    "input": {"id": "abc"}}}]}
    assert messages[2] == {"role": "user", "content": [{
        "toolResult": {"toolUseId": "call_1",
                       "content": [{"text": "value=42"}]}}]}


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
