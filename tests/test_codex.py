"""
OpenAI ChatGPT-subscription provider (api_kind="openai_codex"): token
read/refresh from auth.json, Responses-API request translation, SSE
aggregation, and the api_kind dispatcher. The live streaming HTTP call is not
exercised (no subscription in CI); everything around it is.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codex_auth import CodexAuth, _extract_tokens, _jwt_exp  # noqa: E402
import codex_backend as cb  # noqa: E402
from llm_router_host import make_api_kind_dispatcher  # noqa: E402


def _jwt(exp: int) -> str:
    def b64(d): return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return f"{b64({'alg': 'none'})}.{b64({'exp': exp})}.sig"


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
    def json(self):
        return self._payload


# ---- codex_auth --------------------------------------------------------

def test_extract_tokens_accepts_nested_and_top_level():
    nested = _extract_tokens({"tokens": {"access_token": "a", "account_id": "acc"}})
    assert nested["access_token"] == "a" and nested["account_id"] == "acc"
    flat = _extract_tokens({"access_token": "b", "refresh_token": "r"})
    assert flat["access_token"] == "b" and flat["refresh_token"] == "r"


def test_jwt_exp_reads_expiry():
    assert _jwt_exp(_jwt(1_900_000_000)) == 1_900_000_000.0
    assert _jwt_exp("not-a-jwt") is None


def test_access_token_reads_from_auth_json(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"tokens": {
        "access_token": _jwt(1_900_000_000), "account_id": "acc-1"}}))
    auth = CodexAuth(p, now=lambda: 1_000_000_000)
    assert auth.access_token().startswith("ey") or "." in auth.access_token()
    assert auth.account_id() == "acc-1"


def test_expired_token_triggers_refresh_and_writeback(tmp_path):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"tokens": {
        "access_token": _jwt(1_000),          # long expired
        "refresh_token": "refresh-abc",
        "account_id": "acc-1",
    }}))
    calls = []
    new_token = _jwt(1_900_000_000)
    def fake_post(url, json):
        calls.append((url, json))
        return _Resp(200, {"access_token": new_token, "refresh_token": "refresh-def"})

    auth = CodexAuth(p, http_post=fake_post, now=lambda: 1_500_000_000)
    tok = auth.access_token()
    assert tok == new_token, "refreshed token returned"
    assert calls and calls[0][1]["grant_type"] == "refresh_token"
    assert calls[0][1]["refresh_token"] == "refresh-abc"
    # written back to disk
    on_disk = json.loads(p.read_text())["tokens"]
    assert on_disk["access_token"] == new_token
    assert on_disk["refresh_token"] == "refresh-def"


def test_missing_auth_json_yields_no_token(tmp_path):
    auth = CodexAuth(tmp_path / "nope.json")
    assert auth.access_token() is None


# ---- codex_backend translation -----------------------------------------

def test_messages_to_input():
    items = cb._messages_to_input([
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ])
    assert items == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
    ]


def test_build_codex_body_uses_responses_shape():
    body = cb.build_codex_body({
        "served_model_id": "gpt-5.5-codex",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
        "temperature": 0.3,
    })
    assert body["model"] == "gpt-5.5-codex"
    assert body["stream"] is True
    assert body["store"] is False
    assert body["instructions"] == "You are a concise assistant."
    assert cb.build_codex_body({"served_model_id": "gpt-5.5-codex", "instructions": "custom"})["instructions"] == "custom"
    assert body["input"][0]["role"] == "user"
    # ChatGPT-account Codex rejects the public Responses max_output_tokens and
    # temperature params.
    assert "max_output_tokens" not in body
    assert "temperature" not in body


def test_build_codex_headers_sets_account_id():
    h = cb.build_codex_headers("tok", "acc-9")
    assert h["Authorization"] == "Bearer tok"
    assert h["chatgpt-account-id"] == "acc-9"
    assert h["Accept"] == "text/event-stream"


def test_aggregate_sse_collects_text_and_usage():
    lines = [
        'data: {"type": "response.output_text.delta", "delta": "Hel"}',
        'data: {"type": "response.output_text.delta", "delta": "lo"}',
        'data: {"type": "response.completed", "response": {"usage": '
        '{"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}}}',
        "data: [DONE]",
    ]
    out = cb.aggregate_codex_sse(lines, latency_ms=12)
    assert out["ok"] is True
    assert out["response"]["text"] == "Hello"
    assert out["response"]["tokens_total"] == 7
    assert out["latency_ms"] == 12


def test_aggregate_sse_maps_failure_to_error():
    lines = ['data: {"type": "response.failed", "response": {"error": "boom"}}']
    out = cb.aggregate_codex_sse(lines, latency_ms=3)
    assert out["ok"] is False
    assert out["error_kind"] == "server_error"
    assert "boom" in out["error_message"]


# ---- dispatcher --------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatcher_routes_by_api_kind():
    seen = {}
    async def default(req): seen["default"] = req; return {"ok": True, "via": "default"}
    async def codex(req): seen["codex"] = req; return {"ok": True, "via": "codex"}

    dispatch = make_api_kind_dispatcher(default=default, handlers={"openai_codex": codex})
    r1 = await dispatch({"api_kind": "openai_compatible"})
    r2 = await dispatch({"api_kind": "openai_codex"})
    assert r1["via"] == "default"
    assert r2["via"] == "codex"


# ── tool forwarding + native function-call aggregation (router tool-call fix) ──
def _sse_tool(*events):
    import json as _j
    return [f"data: {_j.dumps(e)}" for e in events] + ["data: [DONE]"]


def test_build_codex_body_forwards_and_flattens_tools():
    body = cb.build_codex_body({
        "served_model_id": "gpt-5.5",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {
            "name": "shell", "description": "Run a shell command",
            "parameters": {"type": "object",
                           "properties": {"command": {"type": "string"}},
                           "required": ["command"]}}}],
        "tool_choice": "auto",
    })
    assert body["tools"] == [{"type": "function", "name": "shell",
        "description": "Run a shell command",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}]
    assert body["tool_choice"] == "auto"


def test_build_codex_body_no_tools_omits_key():
    body = cb.build_codex_body({"served_model_id": "m", "messages": []})
    assert "tools" not in body and "tool_choice" not in body


def test_tool_choice_named_is_flattened():
    assert cb._to_responses_tool_choice(
        {"type": "function", "function": {"name": "shell"}}
    ) == {"type": "function", "name": "shell"}


def test_aggregate_codex_sse_native_function_call():
    lines = _sse_tool(
        {"type": "response.output_item.added",
         "item": {"id": "fc_1", "type": "function_call", "call_id": "call_abc",
                  "name": "shell", "arguments": ""}, "output_index": 0},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1",
         "delta": "{\"command\": \"echo "},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1",
         "delta": "hi\"}"},
        {"type": "response.function_call_arguments.done", "item_id": "fc_1",
         "arguments": "{\"command\": \"echo hi\"}"},
        {"type": "response.completed",
         "response": {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}},
    )
    r = cb.aggregate_codex_sse(lines, 42)["response"]
    assert r["finish_reason"] == "tool_calls"
    assert r["tool_calls"] == [{"id": "call_abc", "type": "function",
        "function": {"name": "shell", "arguments": "{\"command\": \"echo hi\"}"}}]
    assert r["text"] == ""


def test_codex_call_invokes_observer_with_filtered_headers():
    """The adapter pushes status + ratelimit/usage/quota headers per call."""
    import asyncio

    seen = []

    class FakeAuth:
        def access_token(self):
            return None  # short-circuits before HTTP — observer still fires

        def account_id(self):
            return None

    call = cb.make_codex_async_call_provider(FakeAuth(), observe=lambda sig: seen.append(sig))
    r = asyncio.run(call({"served_model_id": "gpt-5.5", "messages": []}))
    assert r["error_kind"] == "auth_error"
    assert seen and seen[0]["status"] == 0          # auth short-circuit observed


class _FakeCodexCallAuth:
    def access_token(self):
        return "tok"

    def account_id(self):
        return "acct"


class _SlowCodexStreamResponse:
    status_code = 200
    headers = {}

    def __init__(self, lines, open_delay=0, first_line_delay=0):
        self._lines = lines
        self._open_delay = open_delay
        self._first_line_delay = first_line_delay

    async def __aenter__(self):
        if self._open_delay:
            import asyncio
            await asyncio.sleep(self._open_delay)
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        if self._first_line_delay:
            import asyncio
            await asyncio.sleep(self._first_line_delay)
        for line in self._lines:
            yield line

    async def aread(self):
        return b""


def _patch_codex_async_client(monkeypatch, response):
    import httpx

    class FakeAsyncClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, json=None, headers=None):
            return response

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)


def test_codex_call_first_token_timeout_opening_stream(monkeypatch):
    import asyncio

    response = _SlowCodexStreamResponse(
        ['data: {"type": "response.output_text.delta", "delta": "late"}'],
        open_delay=0.05,
    )
    _patch_codex_async_client(monkeypatch, response)

    call = cb.make_codex_async_call_provider(_FakeCodexCallAuth())
    r = asyncio.run(call({
        "served_model_id": "gpt-5.5",
        "messages": [],
        "first_token_timeout_ms": 10,
    }))
    assert r["ok"] is False
    assert r["error_kind"] == "timeout"
    assert "first token timed out" in r["error_message"]


def test_codex_call_first_token_timeout_before_delta(monkeypatch):
    import asyncio

    response = _SlowCodexStreamResponse(
        ['data: {"type": "response.output_text.delta", "delta": "late"}'],
        first_line_delay=0.05,
    )
    _patch_codex_async_client(monkeypatch, response)

    call = cb.make_codex_async_call_provider(_FakeCodexCallAuth())
    r = asyncio.run(call({
        "served_model_id": "gpt-5.5",
        "messages": [],
        "first_token_timeout_ms": 10,
    }))
    assert r["ok"] is False
    assert r["error_kind"] == "timeout"
    assert "first token timed out" in r["error_message"]


def test_messages_to_input_translates_tool_call_history():
    # the exact conversation shape that 400d live: every turn after the first
    # of a tool-using agent carries assistant tool_calls + tool results
    items = cb._messages_to_input([
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "weather in BCN?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"BCN"}'}},
            {"id": "call_2", "type": "function",
             "function": {"name": "get_time", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "22C sunny"},
        {"role": "tool", "tool_call_id": "call_2", "content": "14:00"},
        {"role": "assistant", "content": "It's 22C."},
        {"role": "user", "content": "and tomorrow?"},
    ])
    assert items == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "weather in BCN?"},
        {"type": "function_call", "call_id": "call_1",
         "name": "get_weather", "arguments": '{"city":"BCN"}'},
        {"type": "function_call", "call_id": "call_2",
         "name": "get_time", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "22C sunny"},
        {"type": "function_call_output", "call_id": "call_2", "output": "14:00"},
        {"role": "assistant", "content": "It's 22C."},
        {"role": "user", "content": "and tomorrow?"},
    ]
    # no 'tool' role and no tool_calls field may survive translation
    assert all(i.get("role") != "tool" and "tool_calls" not in i for i in items)


def test_messages_to_input_keeps_assistant_text_alongside_calls():
    items = cb._messages_to_input([
        {"role": "assistant", "content": "checking…", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]},
    ])
    assert items == [
        {"role": "assistant", "content": "checking…"},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
    ]


def test_content_to_text_coerces_part_arrays():
    assert cb._content_to_text(None) == ""
    assert cb._content_to_text("x") == "x"
    assert cb._content_to_text([{"type": "text", "text": "a"},
                                {"type": "text", "text": "b"}]) == "ab"
    assert cb._content_to_text(7) == "7"
