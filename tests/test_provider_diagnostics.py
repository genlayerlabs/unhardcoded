"""Real socket boundaries identify where a provider timeout happened."""
import asyncio
import json

import httpx
import pytest

from provider_adapters.openai_compatible import make_async_call_provider, stream_openai_compatible
from provider_adapters.diagnostics import ProviderTiming, bounded_diagnostics, upstream_metadata
from tests.test_antseed_concurrency import _req
from tests.test_streaming import FakeStreamClient, FakeStreamResponse, _openai_lines, OPENAI_REQ
from shim import _openai_usage
from host_store import routing_summary
from tests.test_reasoning_controls import host
from tests.test_compact import _PIN
from shim import ChatRequest, _request_to_contract, create_app
from fastapi.testclient import TestClient


@pytest.mark.asyncio
@pytest.mark.parametrize("send_headers", [False, True])
async def test_timeout_distinguishes_headers_from_incomplete_body(send_headers):
    tasks = set()
    closed = asyncio.Event()

    async def serve(reader, writer):
        tasks.add(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            if send_headers:
                writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
                await writer.drain()
            while True:
                if send_headers:
                    writer.write(b"1\r\n \r\n")
                    await writer.drain()
                await asyncio.sleep(.01)
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()
            closed.set()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    try:
        req = _req("diagnostic", None, timeout_ms=150)
        req.update(base_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1",
                   reasoning_effort="low", max_tokens=8000)
        async with httpx.AsyncClient(trust_env=False) as client:
            result = await make_async_call_provider(client=client)(req)
        assert result["error_kind"] == "timeout"
        d = result["diagnostics"]
        assert d["phase"] == ("response_body" if send_headers else "response_headers")
        assert d["timeout_source"] in ("attempt_deadline", "ReadTimeout")
        assert d["request_sent_ms"] < d["elapsed_ms"]
        assert d["requested_reasoning_effort"] == "low"
        assert d["requested_max_tokens"] == 8000
        assert d["requested_timeout_ms"] == 150
        if send_headers:
            assert d["http_status"] == 200
            assert d["response_headers_ms"] < d["elapsed_ms"]
        else:
            assert "response_headers_ms" not in d
        assert "body_complete_ms" not in d
    finally:
        server.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await server.wait_closed()


@pytest.mark.asyncio
async def test_buffered_preserves_metadata_without_reasoning_text_or_credentials():
    async def handler(request):
        assert json.loads(request.content)["reasoning_effort"] == "low"
        return httpx.Response(200, json={
            "id": "gen-for-support", "provider": "Example Provider",
            "choices": [{"message": {"content": "ok", "reasoning": "PRIVATE REASONING"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 9,
                      "completion_tokens_details": {"reasoning_tokens": 7}},
        })
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await make_async_call_provider(client=client)(
            {**OPENAI_REQ, "reasoning_effort": "low"})
    d = result["diagnostics"]
    assert d["upstream_id"] == "gen-for-support"
    assert d["upstream_provider"] == "Example Provider"
    assert d["tokens_reasoning"] == 7
    assert "PRIVATE REASONING" not in json.dumps(result)
    usage = _openai_usage(result["response"])
    assert usage["completion_tokens"] == 9
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 7}


@pytest.mark.asyncio
async def test_stream_observes_reasoning_and_tool_output_without_emitting_reasoning():
    lines = [
        'data: ' + json.dumps({"id": "gen-stream", "provider": "Example", "choices": [
            {"delta": {"reasoning": "PRIVATE REASONING"}}]}),
        'data: ' + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "tool-1", "function": {"name": "shell", "arguments": "{}"}}]}}]}),
        'data: ' + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                              "usage": {"completion_tokens": 10, "completion_tokens_details": {"reasoning_tokens": 0}}}),
        'data: [DONE]',
    ]
    emitted = []
    async def emit(text):
        emitted.append(text)
    result = await stream_openai_compatible(OPENAI_REQ, emit,
        client=FakeStreamClient(FakeStreamResponse(200, lines=lines)))
    assert result["ok"] and not emitted
    d = result["diagnostics"]
    assert d["first_reasoning_ms"] <= d["first_output_ms"] <= d["elapsed_ms"]
    assert d["upstream_id"] == "gen-stream"
    assert d["tokens_reasoning"] == 0
    assert "PRIVATE REASONING" not in json.dumps(result)


def test_ledger_diagnostics_are_bounded_and_allowlisted():
    secret = "do-not-store"
    diag = {"phase": "response_body", "upstream_id": "x" * 1000,
            "elapsed_ms": 12, "tokens_reasoning": 0, "connect_ms": float("nan"),
            "tls_ms": True, "messages": secret, "headers": secret, "request": secret}
    summary = routing_summary({"provider_diagnostics": [diag] * 40})
    assert len(summary["provider_diagnostics"]) == 32
    d = summary["provider_diagnostics"][0]
    assert len(d["upstream_id"]) == 160
    assert d["tokens_reasoning"] == 0
    assert "connect_ms" not in d and "tls_ms" not in d
    assert secret not in json.dumps(summary)
    assert bounded_diagnostics(None) == {}
    assert bounded_diagnostics({"elapsed_ms": 10 ** 1000}) == {}
    assert upstream_metadata({"usage": {"completion_tokens_details": "malformed"}}) == {}


@pytest.mark.asyncio
async def test_trace_does_not_retain_headers_or_exceptions():
    timing = ProviderTiming({}, 40, "buffered")
    await timing.trace("http11.receive_response_headers.complete", {
        "return_value": (b"HTTP/1.1", 200, b"OK", [(b"authorization", b"secret")])})
    await timing.trace("connection.connect_tcp.failed", {"exception": ValueError("secret")})
    assert timing.data["http_status"] == 200
    assert "secret" not in json.dumps(timing.data)


@pytest.mark.parametrize("failed", [False, True])
def test_http_trace_retains_provider_evidence_on_success_and_failure(host, monkeypatch, failed):
    monkeypatch.setattr("llm_router_host._fold_route_outcome", lambda *a, **kw: None)
    async def call(req):
        common = {"latency_ms": 20, "diagnostics": {
            "mode": "buffered", "phase": "response_body", "upstream_id": "gen-proof",
            "messages": "must-not-leak", "elapsed_ms": 20}}
        return ({**common, "ok": False, "error_kind": "timeout"} if failed else
                {**common, "ok": True, "response": {"text": "ok", "tokens_out": 9, "tokens_reasoning": 7}})
    host.set_async_call_hook(call)
    response = TestClient(create_app(host)).post("/v1/chat/completions", json={
        "policy_ir": _PIN, "messages": [{"role": "user", "content": "fixture"}]})
    assert response.status_code == (504 if failed else 200)
    data = response.json()
    d = data["x_router"]["decision_trace"]["provider_diagnostics"][0]
    assert d["upstream_id"] == "gen-proof" and d["attempt"] == 1
    assert d["provider_id"] == "comput3"
    assert "must-not-leak" not in response.text
    if failed:
        assert d["error_kind"] == "timeout"
    else:
        assert data["usage"]["completion_tokens_details"]["reasoning_tokens"] == 7
    assert routing_summary(data["x_router"]["decision_trace"])["provider_diagnostics"][0] == d


@pytest.mark.asyncio
async def test_concurrent_host_requests_keep_diagnostics_separate(host, monkeypatch):
    monkeypatch.setattr("llm_router_host._fold_route_outcome", lambda *a, **kw: None)
    async def call(req):
        ident = req["messages"][0]["content"]
        await asyncio.sleep(.01 if ident == "a" else 0)
        return {"ok": True, "response": {"text": ident}, "diagnostics": {"upstream_id": ident}}
    host.set_async_call_hook(call)
    results = await asyncio.gather(*(host.execute_async(_request_to_contract(ChatRequest(
        policy_ir=_PIN, messages=[{"role": "user", "content": name}]), "default")) for name in ("a", "b")))
    assert [r["trace"]["provider_diagnostics"][0]["upstream_id"] for r in results] == ["a", "b"]


@pytest.mark.asyncio
async def test_exhausted_connection_pool_has_no_request_sent_event():
    writers = set()
    async def hold(reader, writer):
        writers.add(writer)
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
        await writer.drain()
    server = await asyncio.start_server(hold, "127.0.0.1", 0)
    url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    try:
        async with httpx.AsyncClient(trust_env=False, limits=httpx.Limits(max_connections=1)) as client:
            async with client.stream("GET", url):
                req = {**OPENAI_REQ, "base_url": url, "timeout_ms": 100}
                result = await make_async_call_provider(client=client)(req)
        assert result["error_kind"] == "timeout"
        d = result["diagnostics"]
        assert d["phase"] == "connection_pool"
        assert d["timeout_source"] in ("attempt_deadline", "PoolTimeout")
        assert "request_sent_ms" not in d and "response_headers_ms" not in d
    finally:
        server.close()
        for writer in writers:
            writer.close()
            await writer.wait_closed()
        await server.wait_closed()
