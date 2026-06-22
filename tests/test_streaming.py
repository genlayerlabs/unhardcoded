"""
Unit tests for streaming.py: the stream variants of both backends, the SSE
chunk encoders, and the dispatcher. No network — upstreams are faked.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streaming  # noqa: E402


# ---- fakes ------------------------------------------------------------------

class FakeStreamResponse:
    def __init__(self, status_code, lines=None, body=b"", headers=None):
        self.status_code = status_code
        self._lines = lines or []
        self._body = body
        self.headers = headers or {}

    async def aiter_lines(self):
        for line in self._lines:
            if isinstance(line, Exception):
                raise line
            yield line

    async def aread(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeStreamClient:
    def __init__(self, response):
        self._response = response
        self.requests = []

    def stream(self, method, url, json=None, headers=None, timeout=None):
        self.requests.append({"method": method, "url": url, "json": json})
        return self._response


def _collect(coro):
    return asyncio.run(coro)


OPENAI_REQ = {
    "provider_id": "openrouter",
    "served_model_id": "openai/gpt-5.5",
    "base_url": "https://openrouter.ai/api/v1",
    "auth": {"kind": "none"},
    "messages": [{"role": "user", "content": "hi"}],
}


def _openai_lines(*texts, finish="stop", usage=True):
    lines = []
    for t in texts:
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": t}}]}))
    final = {"choices": [{"delta": {}, "finish_reason": finish}]}
    if usage:
        final["usage"] = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    lines.append("data: " + json.dumps(final))
    lines.append("data: [DONE]")
    return lines


# ---- openai-compatible stream backend ----------------------------------------

def test_stream_openai_compatible_emits_deltas_and_aggregates():
    deltas = []

    async def emit(d):
        deltas.append(d)

    client = FakeStreamClient(FakeStreamResponse(200, lines=_openai_lines("he", "llo")))
    resp = _collect(streaming.stream_openai_compatible(OPENAI_REQ, emit, client=client))
    assert deltas == ["he", "llo"]
    assert resp["ok"] is True
    assert resp["response"]["text"] == "hello"
    assert resp["response"]["finish_reason"] == "stop"
    assert resp["response"]["tokens_total"] == 5
    assert client.requests[0]["json"]["stream"] is True


def test_stream_openai_compatible_pre_delta_error_no_emit():
    deltas = []

    async def emit(d):
        deltas.append(d)

    client = FakeStreamClient(FakeStreamResponse(
        402, body=b'{"error": {"code": "insufficient_deposits"}}'))
    resp = _collect(streaming.stream_openai_compatible(
        OPENAI_REQ, emit, client=client,
        provider_rules={"openrouter": {"error_map": {"insufficient_deposits": "payment_required"}}}))
    assert deltas == []
    assert resp["ok"] is False
    assert resp["error_kind"] == "payment_required"
    assert resp["http_status"] == 402


def test_stream_openai_compatible_mid_stream_failure_after_commit():
    deltas = []

    async def emit(d):
        deltas.append(d)

    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": "par"}}]}),
             RuntimeError("connection reset")]
    client = FakeStreamClient(FakeStreamResponse(200, lines=lines))
    resp = _collect(streaming.stream_openai_compatible(OPENAI_REQ, emit, client=client))
    assert deltas == ["par"]
    assert resp["ok"] is False
    assert resp["error_kind"] == "stream_interrupted"
    assert "par" in resp["error_message"]


def test_stream_openai_compatible_uses_wire_model_id():
    async def emit(d):
        pass

    req = dict(OPENAI_REQ, offer={"wire_model_id": "qwen3-235b-instruct"})
    client = FakeStreamClient(FakeStreamResponse(200, lines=_openai_lines("x")))
    _collect(streaming.stream_openai_compatible(req, emit, client=client))
    assert client.requests[0]["json"]["model"] == "qwen3-235b-instruct"


# ---- codex stream backend -----------------------------------------------------

CODEX_LINES = [
    "data: " + json.dumps({"type": "response.output_text.delta", "delta": "wor"}),
    "data: " + json.dumps({"type": "response.output_text.delta", "delta": "ld"}),
    "data: " + json.dumps({"type": "response.completed", "response": {
        "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}}}),
]


class FakeCodexAuth:
    def access_token(self):
        return "tok"

    def account_id(self):
        return "acct"


def test_stream_codex_emits_output_text_deltas():
    deltas, signals = [], []

    async def emit(d):
        deltas.append(d)

    client = FakeStreamClient(FakeStreamResponse(200, lines=CODEX_LINES,
                                                  headers={"x-codex-primary-used-percent": "12"}))
    resp = _collect(streaming.stream_codex(
        {"served_model_id": "gpt-5.5", "messages": [{"role": "user", "content": "hi"}]},
        emit, auth=FakeCodexAuth(), client=client, observe=lambda s: signals.append(s)))
    assert deltas == ["wor", "ld"]
    assert resp["ok"] is True
    assert resp["response"]["text"] == "world"
    assert resp["response"]["tokens_total"] == 6
    assert signals and signals[0]["headers"].get("x-codex-primary-used-percent") == "12"


def test_stream_codex_pre_delta_429_no_emit():
    deltas = []

    async def emit(d):
        deltas.append(d)

    client = FakeStreamClient(FakeStreamResponse(429))
    resp = _collect(streaming.stream_codex(
        {"served_model_id": "gpt-5.5", "messages": []},
        emit, auth=FakeCodexAuth(), client=client))
    assert deltas == []
    assert resp["error_kind"] == "rate_limit"


# ---- SSE chunk encoding -------------------------------------------------------

def test_sse_chunk_encoding():
    role = streaming.encode_role_chunk("id1", "m")
    text = streaming.encode_text_chunk("id1", "m", "hi")
    final = streaming.encode_final_chunk("id1", "m", "stop", None,
                                          {"prompt_tokens": 1}, {"provider": "openai"})
    for ev in (role, text, final):
        assert ev.startswith("data: ") and ev.endswith("\n\n")
        payload = json.loads(ev[len("data: "):])
        assert payload["object"] == "chat.completion.chunk"
        assert payload["id"] == "id1"
    assert json.loads(text[6:])["choices"][0]["delta"]["content"] == "hi"
    fin = json.loads(final[6:])
    assert fin["choices"][0]["finish_reason"] == "stop"
    assert fin["usage"]["prompt_tokens"] == 1
    assert fin["x_router"]["provider"] == "openai"
    assert streaming.DONE_EVENT == "data: [DONE]\n\n"


# ---- dispatcher ----------------------------------------------------------------

def test_streaming_dispatcher_routes_by_api_kind():
    async def default(request, emit):
        return {"ok": True, "via": "default"}

    async def codex(request, emit):
        return {"ok": True, "via": "codex"}

    dispatch = streaming.make_streaming_dispatcher(default, {"openai_codex": codex})

    async def emit(d):
        pass

    assert _collect(dispatch({"api_kind": "openai_codex"}, emit))["via"] == "codex"
    assert _collect(dispatch({"api_kind": "openai_compatible"}, emit))["via"] == "default"
