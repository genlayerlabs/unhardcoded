"""
Proves the async core's scaling claim: many concurrent execute_async calls
share one LuaRuntime and overlap their upstream HTTP waits on a single event
loop, so wall-clock ≈ one provider latency, not the sum.

If the provider call were still made under the Lua lock (the old sync path),
N requests at `LATENCY` each would serialize to ~N * LATENCY.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router_host import LLMRouterHost  # noqa: E402

LATENCY = 0.10  # seconds each "provider call" takes
N = 25          # concurrent agents


def _make_host(hook):
    h = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "core" / "config.example.lua",
        call_provider_async=hook,
        now_ms=lambda: int(time.monotonic() * 1000),
    )
    h.init()
    return h


@pytest.mark.asyncio
async def test_concurrent_requests_overlap():
    in_flight = 0
    max_in_flight = 0

    async def hook(req):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        try:
            await asyncio.sleep(LATENCY)
            return {"ok": True, "latency_ms": int(LATENCY * 1000),
                    "response": {"text": "ok"}}
        finally:
            in_flight -= 1

    host = _make_host(hook)
    contract = {"profile": "default", "messages": [{"role": "user", "content": "hi"}]}

    t0 = time.monotonic()
    results = await asyncio.gather(*(host.execute_async(dict(contract)) for _ in range(N)))
    elapsed = time.monotonic() - t0

    assert all(r["ok"] for r in results), "every request succeeded"
    # Overlap: total time is far below the serial sum (N * LATENCY).
    assert elapsed < LATENCY * (N / 3), (
        f"expected overlap, got {elapsed:.3f}s for {N} calls of {LATENCY}s "
        f"(serial would be {N * LATENCY:.1f}s)"
    )
    # And the calls really were in flight at the same time.
    assert max_in_flight >= N // 2, f"only {max_in_flight} concurrent in flight"


@pytest.mark.asyncio
async def test_shared_state_is_coherent_under_concurrency():
    """All coroutines fold into one host-owned reliability state; the per-route
    observation count must equal the number of calls with no lost updates
    (single-thread invariant). Reliability is host-owned now (#15), so the count
    lives in route_reliability, not the engine EMA."""
    import route_reliability as rr
    rr.reset()

    async def hook(req):
        await asyncio.sleep(0.01)
        return {"ok": True, "response": {"text": "ok"}}

    host = _make_host(hook)
    contract = {"profile": "default", "messages": [{"role": "user", "content": "hi"}]}
    await asyncio.gather(*(host.execute_async(dict(contract)) for _ in range(N)))

    total_n = sum(rr.snapshot_counts().values())
    assert total_n == N, f"expected {N} recorded calls, got {total_n} (lost updates?)"
