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
from sources.antseed import AntSeedSource  # noqa: E402
from conftest import seed_peer_offers as _seed_market, seed_route_obs  # noqa: E402


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
    "models": {"qwen3-235b-a22b": {"capabilities": {"context": 32000},
                                   "static_quality_hint": 0.90}},
}
FAMILY = "qwen3-235b-a22b"


def _peer(pid, price_in, maxc=5, service=FAMILY):
    return {
        "peerId": pid, "maxConcurrency": maxc, "lastSeen": 1,
        "providerPricing": {"x": {"services": {
            service: {"inputUsdPerMillion": price_in,
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
    _seed_market([_peer("peerA", 0.5), _peer("peerB", 1.0)])
    # peerA observed failing -> demoted; peerB never observed -> unstamped
    seed_route_obs("antseed", FAMILY, "peerA", ok=False)
    offers = AntSeedSource(CATALOG).offers_sync("antseed")
    by_peer = {o["peer_id"]: o for o in offers}
    assert by_peer["peerA"]["success_rate"] == 0.0
    assert by_peer["peerB"]["success_rate"] is None


def test_offers_sync_stamps_host_measured_latency(tmp_path):
    # The latency twin of the reliability stamp: a peer observed slow carries its
    # measured latency_ms so a policy can route by speed; an unobserved peer is
    # left unstamped (None -> field default, optimistically routable).
    _seed_market([_peer("peerA", 0.5), _peer("peerB", 1.0)])
    seed_route_obs("antseed", FAMILY, "peerA", ok=True, latency_ms=12000)
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
    _seed_market([_peer("peerA", 0.5)])
    # peerA emits no tool_calls on _MIN_SAMPLES (20) tools-requests -> incapable
    seed_route_obs("antseed", FAMILY, "peerA", ok=True, n=20,
                   tools_requested=True, tool_calls_emitted=False)
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


def test_variant_offer_inherits_the_base_family_capabilities(tmp_path):
    # A serving-mode variant inherits the base's CAPABILITIES — context is
    # architecture, no serving switch shrinks it, and withholding it would drop
    # the variant from every min_context request for no gain. It keeps its own
    # family name so no existing family_eq policy moves.
    #
    # It does NOT inherit the quality hint. "Same marker, same weights" is a
    # premise this module asserts and cannot check, and the hint is what
    # min_quality gates on (core/llm_policy/filter.lua) — see
    # test_peer_minted_variant_does_not_inherit_the_quality_hint.
    rr.reset()
    _seed_market([_peer("peerA", 0.5, service=FAMILY + ":web")])
    offer = AntSeedSource(CATALOG).offers_sync("antseed")[0]
    assert offer["model_family"] == FAMILY + "@web"
    assert offer["base_family"] == FAMILY
    assert offer["wire_model_id"] == FAMILY + ":web"   # wire name preserved
    assert offer["capabilities"]["context"] == 32000
    assert offer["quality_hint"] is None


def test_uncurated_offer_with_no_trait_source_is_left_unstamped(tmp_path):
    # Nothing authoritative is known about this name, so nothing is claimed. An
    # invented context would sneak the route past a min_context gate and land the
    # prompt on a model that silently truncates it.
    rr.reset()
    _seed_market([_peer("peerA", 0.5, service="totally-unknown-model")])
    offer = AntSeedSource(CATALOG).offers_sync("antseed")[0]
    assert offer["model_family"] == "totally-unknown-model"
    assert offer["base_family"] is None
    assert "context" not in offer["capabilities"]
    assert offer["quality_hint"] is None
    assert offer["traits"] is None


class _FakeTraitSource:
    """Stands in for sources/openrouter's live snapshot (see live_offers())."""

    def __init__(self, offers):
        self._offers = offers

    def live_offers(self):
        return self._offers


def test_uncurated_offer_joins_live_traits_for_a_real_context(tmp_path):
    rr.reset()
    _seed_market([_peer("peerA", 0.5, service="Z-AI/GLM-4.6")])
    src = AntSeedSource(CATALOG)
    src.bind_trait_source(_FakeTraitSource([{
        "model_family": "glm-4.6", "wire_model_id": "z-ai/glm-4.6",
        "capabilities": {"context": 202752}, "traits": {"bench_coding": 0.61},
    }]))
    offer = src.offers_sync("antseed")[0]
    assert offer["capabilities"]["context"] == 202752   # measured, not invented
    assert offer["traits"] == {"bench_coding": 0.61}
    # still its own family: the join stamps metadata, it never renames a route
    assert offer["model_family"] == "Z-AI/GLM-4.6"


def test_unbound_names_are_ranked_by_distinct_sellers_with_near_miss(tmp_path):
    # "161 unbound" is not actionable; a ranked queue is. `near_miss` names the
    # curated family one alias away — deliberately looser than binding (it also
    # flags a digit-run difference), because it is a question for an operator.
    rr.reset()
    _seed_market([
        _peer("peerA", 0.5, service="qwen3235b-a22b"),   # near miss (separators)
        _peer("peerB", 0.6, service="qwen3235b-a22b"),
        _peer("peerC", 0.7, service="something-else-7b"),
        _peer("peerD", 0.8, service=FAMILY),             # binds -> not listed
    ])
    src = AntSeedSource(CATALOG)
    src.offers_sync("antseed")
    top = src.snapshot_stats()["unbound_top"]
    assert [row["service"] for row in top] == ["qwen3235b-a22b", "something-else-7b"]
    assert top[0]["sellers"] == 2 and top[0]["near_miss"] == FAMILY
    assert top[1]["sellers"] == 1 and top[1]["near_miss"] is None


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


def test_peer_minted_variant_does_not_inherit_the_quality_hint(tmp_path):
    """M-1: a peer mints `<family>-fast` by spelling a suffix. `-fast` routinely
    means quantized or distilled in marketplace practice — DIFFERENT weights — so
    the variant must not arrive carrying the base's benchmark, which is exactly
    what `min_quality` gates on (core/llm_policy/filter.lua). Capabilities are
    architecture and still come across, so the variant stays routable."""
    rr.reset()
    _seed_market([_peer("peerA", 0.5, service=FAMILY),
                  _peer("peerB", 0.5, service=FAMILY + "-fast")])
    offers = {o["model_family"]: o for o in AntSeedSource(CATALOG).offers_sync("antseed")}

    base = offers[FAMILY]
    assert base["quality_hint"] == 0.90            # curated, earned, unchanged

    variant = offers[f"{FAMILY}@fast"]
    assert variant["base_family"] == FAMILY        # still anchored to the family
    assert variant["capabilities"]["context"] == 32000
    assert variant["quality_hint"] is None


def test_suppressed_tick_clears_last_ticks_curation_queue(tmp_path):
    """L-8: the funds tourniquet returns before reading the market, and used to
    leave the PREVIOUS tick's `uncurated` / `unbound_top` standing next to
    `offers = 0`. An operator reads a stale queue as "nothing left to curate" —
    and a suppressed provider is precisely when those numbers stop refreshing."""
    from conftest import seed_buyer_status
    rr.reset()
    _seed_market([_peer("peerA", 0.5, service="something-uncurated-7b")])
    src = AntSeedSource(CATALOG)

    # funded tick (no buyer_status row -> the gate fails open): a real queue
    src.offers_sync("antseed")
    seeded = src.snapshot_stats()
    assert seeded["uncurated"] == 1
    assert [r["service"] for r in seeded["unbound_top"]] == ["something-uncurated-7b"]
    assert seeded["stale"] is False

    # escrow drops below one channel reserve -> suppressed, nothing read
    seed_buyer_status("antseed", deposits_available="0.1", deposits_reserved="0")
    assert src.offers_sync("antseed") == []
    stats = src.snapshot_stats()
    assert stats["offers"] == 0 and stats["suppressed_no_funds"] == 1
    assert stats["uncurated"] == 0 and stats["unbound_top"] == []
    # UNKNOWN, not False: `stale` answers "did the last market read find fresh
    # rows", and this tick performed no read at all.
    assert stats["stale"] is None
