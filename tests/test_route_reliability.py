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


def _req(peer_id, family="m"):
    return {
        "api_kind": "openai_compatible", "base_url": "http://s/v1",
        "served_model_id": "m", "provider_id": "antseed",
        "messages": [{"role": "user", "content": "hi"}],
        "offer": {"peer_id": peer_id, "model_family": family, "wire_model_id": "m"},
    }


# Folding moved out of the call backend into _fold_route_outcome, which runs once
# per resolved call (direct + streaming/flow paths) so all traffic feeds the same
# host-owned EMAs the algebra reads (offer.success_rate). Drive it directly.
def test_fold_route_outcome_on_success():
    rr.reset()
    H._fold_route_outcome(_req("peerGood"), {"ok": True, "latency_ms": 10})
    assert rr.success_rate(rr.route_key("antseed", "m", "peerGood")) == 1.0


def test_fold_route_outcome_on_failure():
    rr.reset()
    H._fold_route_outcome(_req("peerBad"), {"ok": False, "error_kind": "bad_response"})
    assert rr.success_rate(rr.route_key("antseed", "m", "peerBad")) == 0.0
