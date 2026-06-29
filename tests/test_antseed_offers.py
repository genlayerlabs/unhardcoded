"""
Host half of the reliability form: offers_sync surfaces the OFFERS_TOP_N cheapest
distinct peers per family (routes to rotate between) and stamps each offer with
this route's host-measured reliability (offer.success_rate), which the algebra
reads pointwise (llm-router #14).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import route_reliability as rr  # noqa: E402
import route_latency as rl  # noqa: E402
from sources.antseed import AntSeedSource  # noqa: E402
from conftest import seed_peer_offers as _seed_market  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(host_store_clean):
    # The market book now lives in the host store (peer_offers); every test seeds
    # it and needs the per-test truncation (skips if Postgres is unavailable).
    yield

CATALOG = {
    "providers": {
        "antseed": {
            "discovery": "marketplace", "discovery_id": "antseed",
            "base_url": "http://antseed:8378/v1",
            "market_price_cap": {"input": 1000, "output": 1000},
        },
    },
    "models": {"qwen3-235b-a22b": {"capabilities": {"context": 32000}}},
}
FAMILY = "qwen3-235b-a22b"


def _peer(pid, price_in, maxc=5):
    return {
        "peerId": pid, "maxConcurrency": maxc, "lastSeen": 1,
        "providerPricing": {"x": {"services": {
            FAMILY: {"inputUsdPerMillion": price_in,
                     "outputUsdPerMillion": price_in * 2}}}},
    }


def test_offers_sync_surfaces_top_n_distinct_peers(tmp_path):
    rr.reset()
    _seed_market([
        _peer("peerC", 2.0), _peer("peerA", 0.5),
        _peer("peerD", 9.0), _peer("peerB", 1.0),
    ])
    offers = AntSeedSource(CATALOG).offers_sync("antseed")
    peers = [o["peer_id"] for o in offers]
    assert len(peers) == 3
    assert peers == ["peerA", "peerB", "peerC"]  # cheapest distinct, 9.0 dropped
    assert all(o["max_concurrency"] == 5 for o in offers)
    # never-observed routes are left unstamped -> algebra default / engine fallback
    assert all(o["success_rate"] is None for o in offers)


def test_offers_sync_stamps_host_measured_reliability(tmp_path):
    rr.reset()
    _seed_market([_peer("peerA", 0.5), _peer("peerB", 1.0)])
    # peerA observed failing -> demoted; peerB never observed -> unstamped
    rr.observe(rr.route_key("antseed", FAMILY, "peerA"), False)
    offers = AntSeedSource(CATALOG).offers_sync("antseed")
    by_peer = {o["peer_id"]: o for o in offers}
    assert by_peer["peerA"]["success_rate"] == 0.0
    assert by_peer["peerB"]["success_rate"] is None


def test_offers_sync_stamps_host_measured_latency(tmp_path):
    # The latency twin of the reliability stamp: a peer observed slow carries its
    # measured latency_ms so a policy can route by speed; an unobserved peer is
    # left unstamped (None -> field default, optimistically routable).
    rr.reset()
    rl.reset()
    _seed_market([_peer("peerA", 0.5), _peer("peerB", 1.0)])
    rl.observe(rl.route_key("antseed", FAMILY, "peerA"), 12000, ok=True)
    offers = AntSeedSource(CATALOG).offers_sync("antseed")
    by_peer = {o["peer_id"]: o for o in offers}
    assert by_peer["peerA"]["latency_ms"] == 12000
    assert by_peer["peerB"]["latency_ms"] is None


def test_offers_sync_rejects_negative_priced_peer(tmp_path):
    # A peer advertising a negative price must not be admitted: a negative price
    # wins every cost-led policy ("most negative = cheapest") and bills a negative
    # cost. Free ($0) services stay routable.
    rr.reset()
    _seed_market([
        _peer("peerA", 0.5),    # normal
        _peer("free", 0.0),     # $0 is legitimate -> kept
        _peer("bogus", -1.0),   # negative in/out -> rejected
    ])
    offers = AntSeedSource(CATALOG).offers_sync("antseed")
    peers = {o["peer_id"] for o in offers}
    assert peers == {"peerA", "free"}
    assert "bogus" not in peers


def test_offers_sync_defaults_tool_capability_for_meets_req(tmp_path):
    # AntSeed market rows carry no capability data; every peer is an
    # OpenAI-compatible endpoint, so supports_tools/json default to true — else
    # the core's meets_req filters the whole peer market out of any tools request.
    rr.reset()
    _seed_market([_peer("peerA", 0.5)])
    offers = AntSeedSource(CATALOG).offers_sync("antseed")
    caps = offers[0]["capabilities"]
    assert caps.get("supports_tools") is True
    assert caps.get("supports_json_mode") is True
    assert caps.get("context") == 32000  # curated capability still present


def test_offers_sync_drops_supports_tools_for_learned_incapable_route(tmp_path):
    # the AntSeed default-true hole is closed by the learned signal: a route
    # observed to ignore tools is filtered from tool requests (no supports_tools),
    # while other caps and non-tool routing are unaffected.
    import route_tool_capability as tc
    rr.reset()
    tc.reset()
    _seed_market([_peer("peerA", 0.5)])
    rkey = rr.route_key("antseed", FAMILY, "peerA")
    for _ in range(tc._MIN_SAMPLES):       # peerA never emits tool_calls on tool reqs
        tc.observe(rkey, True, False)
    caps = AntSeedSource(CATALOG).offers_sync("antseed")[0]["capabilities"]
    assert "supports_tools" not in caps    # learned-incapable -> filtered for tools
    assert caps.get("supports_json_mode") is True  # other caps unaffected


def test_offers_sync_dedups_same_peer(tmp_path):
    rr.reset()
    p = _peer("peerA", 0.5)
    p["providerPricing"]["y"] = {"services": {
        FAMILY: {"inputUsdPerMillion": 0.6, "outputUsdPerMillion": 1.2}}}
    _seed_market([p])
    offers = AntSeedSource(CATALOG).offers_sync("antseed")
    assert [o["peer_id"] for o in offers] == ["peerA"]


def test_offers_sync_excludes_peer_outside_window(tmp_path):
    # The sliding window is now a read-time filter on observed_at: a peer last
    # seen past the window is not surfaced (degraded to "no candidate"), exactly
    # as a stale market.json used to be dropped.
    rr.reset()
    old = int(time.time() * 1000) - 20 * 60 * 1000   # 20 min ago, window is 15
    _seed_market([_peer("peerOld", 0.5)], observed_at=old)
    _seed_market([_peer("peerNew", 1.0)])
    offers = AntSeedSource(CATALOG).offers_sync("antseed")
    assert [o["peer_id"] for o in offers] == ["peerNew"]
