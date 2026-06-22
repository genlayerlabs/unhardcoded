"""
Host-side per-route reliability fold: a success-rate EMA the host updates on each
call outcome and stamps onto offers. Proves the EMA math and that the async
backend folds the outcome under the route's (provider, family, peer) key.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm_router_host as H  # noqa: E402
import route_reliability as rr  # noqa: E402


def test_ema_seeds_then_decays():
    rr.reset()
    k = rr.route_key("antseed", "m", "peerX")
    assert rr.success_rate(k) is None
    rr.observe(k, False)                       # first obs seeds directly
    assert rr.success_rate(k) == 0.0
    rr.observe(k, True)                        # 0.2*1 + 0.8*0
    assert rr.success_rate(k) == pytest.approx(0.2)
    rr.observe(k, True)                        # 0.2*1 + 0.8*0.2
    assert rr.success_rate(k) == pytest.approx(0.36)


class _Resp:
    def __init__(self, content):
        self.status_code = 200
        self._c = content

    def json(self):
        return {"choices": [{"message": {"content": self._c}, "finish_reason": "stop"}],
                "usage": {}}


class _Client:
    def __init__(self, content):
        self._content = content

    async def post(self, url, json=None, headers=None, timeout=None):
        return _Resp(self._content)


def _req(peer_id, family="m"):
    return {
        "api_kind": "openai_compatible", "base_url": "http://s/v1",
        "served_model_id": "m", "provider_id": "antseed",
        "messages": [{"role": "user", "content": "hi"}],
        "offer": {"peer_id": peer_id, "model_family": family, "wire_model_id": "m"},
    }


@pytest.mark.asyncio
async def test_call_folds_route_reliability_on_success():
    rr.reset()
    H._PEER_GATES.clear()
    call = H.make_async_call_provider(client=_Client("hello"))
    r = await call(_req("peerGood"))
    assert r["ok"] is True
    assert rr.success_rate(rr.route_key("antseed", "m", "peerGood")) == 1.0


@pytest.mark.asyncio
async def test_call_folds_route_reliability_on_empty_content():
    rr.reset()
    H._PEER_GATES.clear()
    call = H.make_async_call_provider(client=_Client(""))   # empty -> bad_response
    r = await call(_req("peerBad"))
    assert r["ok"] is False and r["error_kind"] == "bad_response"
    assert rr.success_rate(rr.route_key("antseed", "m", "peerBad")) == 0.0
