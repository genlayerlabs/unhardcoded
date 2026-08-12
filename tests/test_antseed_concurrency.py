"""
PR A: the async HTTP backend caps in-flight calls per marketplace seller to the
peer's advertised `max_concurrency`, so the router never trips the seller's own
"Max concurrency reached" (429). Over-cap callers wait up to the call timeout and
then yield to the next candidate as `rate_limit`.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm_router_host as H  # noqa: E402
from provider_adapters.openai_compatible import (  # noqa: E402
    make_async_call_provider,
    stream_openai_compatible,
)
from tests.test_streaming import FakeStreamClient, FakeStreamResponse, _openai_lines  # noqa: E402


class _FakeResp:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {}}


class _FakeClient:
    """Records max concurrent in-flight posts; each post sleeps `delay`."""
    def __init__(self, delay: float):
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    async def post(self, url, json=None, headers=None, timeout=None):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return _FakeResp()
        finally:
            self.in_flight -= 1


def _req(peer_id: str, cap, *, timeout_ms=5000):
    offer = {"peer_id": peer_id, "wire_model_id": "m"}
    if cap is not None:
        offer["max_concurrency"] = cap
    return {
        "api_kind": "openai_compatible",
        "base_url": "http://seller/v1",
        "served_model_id": "m",
        "provider_id": "antseed",
        "messages": [{"role": "user", "content": "hi"}],
        "offer": offer,
        "timeout_ms": timeout_ms,
    }


@pytest.fixture(autouse=True)
def _clear_gates(monkeypatch):
    monkeypatch.delenv("DISTRIBUTED_PEER_GATES", raising=False)
    monkeypatch.delenv("PEER_GATE_WAIT_S", raising=False)
    monkeypatch.delenv("PEER_LEASE_TTL_S", raising=False)
    H._PEER_GATES.clear()
    yield
    H._PEER_GATES.clear()


@pytest.mark.asyncio
async def test_inflight_capped_at_seller_max_concurrency():
    client = _FakeClient(delay=0.05)
    call = H.make_async_call_provider(client=client)
    results = await asyncio.gather(*(call(_req("peerA", 2)) for _ in range(10)))
    assert all(r["ok"] for r in results)
    assert client.max_in_flight == 2, f"cap breached: {client.max_in_flight} in flight"


@pytest.mark.asyncio
async def test_no_cap_means_no_gate():
    client = _FakeClient(delay=0.05)
    call = H.make_async_call_provider(client=client)
    # offer without max_concurrency -> ungated -> all overlap
    await asyncio.gather(*(call(_req("peerB", None)) for _ in range(8)))
    assert client.max_in_flight == 8, f"unexpected gating: {client.max_in_flight}"


@pytest.mark.asyncio
async def test_oversubscribed_yields_rate_limit():
    """cap=1, slow post, tiny timeout: the 2nd caller can't get a slot in time
    and yields to the next candidate as rate_limit instead of forcing a 429."""
    client = _FakeClient(delay=0.30)
    call = H.make_async_call_provider(client=client)
    first = asyncio.create_task(call(_req("peerC", 1, timeout_ms=5000)))
    await asyncio.sleep(0.02)  # let `first` grab the only slot
    second = await call(_req("peerC", 1, timeout_ms=50))
    assert second["ok"] is False and second["error_kind"] == "rate_limit"
    assert (await first)["ok"] is True


@pytest.mark.asyncio
async def test_first_token_timeout_uses_internal_streaming_for_json_calls():
    client = FakeStreamClient(FakeStreamResponse(200, lines=_openai_lines("ok")))
    call = make_async_call_provider(client=client)

    result = await call({
        "api_kind": "openai_compatible",
        "base_url": "http://seller/v1",
        "served_model_id": "m",
        "provider_id": "antseed",
        "messages": [{"role": "user", "content": "hi"}],
        "auth": {"kind": "none"},
        "offer": {"peer_id": "peer-stream-under-hood",
                  "wire_model_id": "m", "max_concurrency": 1},
        "first_token_timeout_ms": 1000,
    })

    assert result["ok"] is True
    assert result["response"]["text"] == "ok"
    assert client.requests[0]["json"]["stream"] is True
    # §3 response-shape parity: the stream-under-hood path returns the SAME shape
    # the non-stream path does — not just text, but finish_reason + usage — so the
    # core's aggregation and cost-accounting are unaffected by the timeout opt-in.
    assert result["response"]["finish_reason"] == "stop"
    assert result["response"]["tokens_out"] == 2
    assert result["response"]["tokens_total"] == 5


class _TrackedStreamResponse(FakeStreamResponse):
    def __init__(self, owner, delay: float):
        super().__init__(200, lines=_openai_lines("ok"))
        self.owner = owner
        self.delay = delay

    async def __aenter__(self):
        self.owner.in_flight += 1
        self.owner.max_in_flight = max(
            self.owner.max_in_flight, self.owner.in_flight)
        return self

    async def __aexit__(self, *exc):
        self.owner.in_flight -= 1
        return False

    async def aiter_lines(self):
        await asyncio.sleep(self.delay)
        for line in self._lines:
            yield line


class _TrackedStreamClient:
    def __init__(self, delay: float):
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    def stream(self, method, url, json=None, headers=None, timeout=None):
        return _TrackedStreamResponse(self, self.delay)


@pytest.mark.asyncio
async def test_streaming_calls_obey_peer_concurrency_cap():
    client = _TrackedStreamClient(delay=0.04)

    async def emit(_delta: str) -> None:
        return None

    results = await asyncio.gather(*(
        stream_openai_compatible(_req("peer-stream", 1), emit, client=client)
        for _ in range(5)))
    assert all(result["ok"] for result in results)
    assert client.max_in_flight == 1


@pytest.mark.asyncio
async def test_distributed_peer_gate_uses_shared_store(
        monkeypatch, host_store_clean):
    monkeypatch.setenv("DISTRIBUTED_PEER_GATES", "1")
    monkeypatch.setenv("PEER_GATE_WAIT_S", "1")
    client = _FakeClient(delay=0.04)
    call = make_async_call_provider(client=client)

    results = await asyncio.gather(*(
        call(_req("peer-global", 2)) for _ in range(8)))
    assert all(result["ok"] for result in results)
    assert client.max_in_flight == 2
    with host_store_clean._get_pool().connection() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM peer_concurrency_leases").fetchone()[0] == 0
