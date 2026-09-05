from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import codex_broker  # noqa: E402
from remote_codex import RemoteCodexClient  # noqa: E402


class _Store:
    def __init__(self):
        self.reload_count = 0
        self.account = object()

    def names(self):
        return ["one"]

    def reload(self):
        self.reload_count += 1
        return ["one"]

    def select_account(self):
        return self.account


def test_broker_requires_internal_token_and_reloads(monkeypatch):
    store = _Store()
    app = codex_broker.create_app(store=store, token="internal", client=object())
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"ok": True, "accounts": 1}
        assert client.post("/v1/reload").status_code == 401
        response = client.post(
            "/v1/reload", headers={"Authorization": "Bearer internal"})
    assert response.status_code == 200
    assert response.json() == {"ok": True, "accounts": ["one"]}
    assert store.reload_count == 1


def test_broker_health_fails_when_secret_is_missing():
    app = codex_broker.create_app(store=_Store(), token="", client=object())
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 503


def test_broker_call_returns_canonical_result_and_sanitized_signals(monkeypatch):
    async def backend(request):
        assert request["served_model_id"] == "gpt-test"
        backend.observe({"status": 200, "headers": {"x-ratelimit": "9"}, "ts": 1})
        return {"ok": True, "response": {"text": "done"}}

    def factory(_store, *, observe, client):
        backend.observe = observe
        return backend

    monkeypatch.setattr(codex_broker, "make_codex_async_call_provider", factory)
    app = codex_broker.create_app(store=_Store(), token="internal", client=object())
    with TestClient(app) as client:
        response = client.post(
            "/v1/call", headers={"Authorization": "Bearer internal"},
            json={"served_model_id": "gpt-test"})
    assert response.status_code == 200
    assert response.json()["result"]["response"]["text"] == "done"
    assert response.json()["signals"] == [{
        "status": 200, "headers": {"x-ratelimit": "9"}, "ts": 1}]


def test_broker_stream_forwards_deltas_then_one_terminal_result(monkeypatch):
    async def fake_stream(request, emit, *, auth, observe, client):
        assert auth is store.account
        await emit("hel")
        await emit("lo")
        observe({"status": 200, "headers": {}, "ts": 2})
        return {"ok": True, "response": {"text": "hello"}}

    store = _Store()
    monkeypatch.setattr(codex_broker, "stream_codex", fake_stream)
    app = codex_broker.create_app(store=store, token="internal", client=object())
    with TestClient(app) as client:
        response = client.post(
            "/v1/stream", headers={"Authorization": "Bearer internal"},
            json={"served_model_id": "gpt-test"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[:2] == [
        {"event": "delta", "delta": "hel"},
        {"event": "delta", "delta": "lo"},
    ]
    assert events[-1]["event"] == "result"
    assert events[-1]["result"]["response"]["text"] == "hello"
    assert events[-1]["signals"][0]["status"] == 200


@pytest.mark.asyncio
async def test_remote_client_call_replays_signals():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer internal"
        return httpx.Response(200, json={
            "result": {"ok": True, "response": {"text": "done"}},
            "signals": [{"status": 200,
                         "headers": {"x-ratelimit-remaining": "7"}, "ts": 3}],
        })

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    remote = RemoteCodexClient("http://broker", "internal", client=http)
    signals = []
    result = await remote.call({"served_model_id": "gpt-test"},
                               observe=signals.append)
    await http.aclose()
    assert result["response"]["text"] == "done"
    assert signals[0]["headers"] == {"x-ratelimit-remaining": "7"}


@pytest.mark.asyncio
async def test_remote_client_stream_preserves_delta_order_and_result():
    wire = "\n".join([
        json.dumps({"event": "delta", "delta": "hel"}),
        json.dumps({"event": "delta", "delta": "lo"}),
        json.dumps({"event": "result", "result": {
            "ok": True, "response": {"text": "hello"}}, "signals": []}),
    ]) + "\n"

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=wire)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    remote = RemoteCodexClient("http://broker", "internal", client=http)
    deltas = []

    async def emit(delta):
        deltas.append(delta)

    result = await remote.stream({"served_model_id": "gpt-test"}, emit)
    await http.aclose()
    assert deltas == ["hel", "lo"]
    assert result["response"]["text"] == "hello"
