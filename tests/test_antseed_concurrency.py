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
def _clear_gates():
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
