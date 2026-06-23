"""Essence for the per-session cache-affinity substrate (PR1).

Caching is STATE between calls: a provider discounts the reused prefix only if
the SAME peer serves the session again. That memory cannot live in the algebra
(a policy is a pure function of one call) nor in offers_sync (request-blind,
shared across sessions) — it is per-session host state, like route_latency is
per-route host state. This closes three things:

  1. route_cache folds session -> last successful route (success only, like
     route_latency); unknown session / no session -> no hot route.
  2. the central fold hook (_resolve_call_async -> _fold_route_outcome) records
     it, with the session threaded host-side from the contract.
  3. the cache_hot field is true for EXACTLY the candidate whose route is the
     session's hot route, so a policy scoring it keeps the hot peer sticky.

Zero engine change: cache_hot is a host-declared extension field (fields.lua
schema{extensions} seam) whose getter reconstructs the route key from candidate
identity exactly as _fold_route_outcome does, and compares it to the hot route
the host resolved into ctx.request.cache_hot_route per request.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import route_cache as rc           # noqa: E402
import route_reliability as rr     # noqa: E402
from llm_router_host import LLMRouterHost  # noqa: E402


# --------------------------------------------------------------------------
# 1. route_cache module — session -> hot route, success only
# --------------------------------------------------------------------------

def test_hot_route_unknown_session_is_none():
    rc.reset()
    assert rc.hot_route("s1") is None
    assert rc.hot_route(None) is None


def test_successful_call_makes_route_hot():
    rc.reset()
    k = rr.route_key("antseed", "glm-5.2", "peerX")
    rc.observe("s1", k, ok=True)
    assert rc.hot_route("s1") == k


def test_failed_call_does_not_set_affinity():
    rc.reset()
    k = rr.route_key("antseed", "glm-5.2", "peerX")
    rc.observe("s1", k, ok=False)
    # only successful calls fold, exactly like route_latency: a failure carries
    # no honest "this peer holds the prefix" signal.
    assert rc.hot_route("s1") is None


def test_latest_successful_route_wins():
    rc.reset()
    a = rr.route_key("antseed", "glm-5.2", "peerA")
    b = rr.route_key("openrouter", "minimax-m2.7", "openrouter")
    rc.observe("s1", a, ok=True)
    rc.observe("s1", b, ok=True)
    assert rc.hot_route("s1") == b   # the most recent successful peer holds the prefix


def test_sessions_are_independent():
    rc.reset()
    a = rr.route_key("antseed", "glm-5.2", "peerA")
    rc.observe("s1", a, ok=True)
    assert rc.hot_route("s2") is None


# --------------------------------------------------------------------------
# 2. central fold hook records the session's hot route
# --------------------------------------------------------------------------

@pytest.fixture
def host():
    h = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "core" / "config.example.lua",
        metrics_path=ROOT / "core" / "metrics.example.lua",
        now_ms=lambda: 1_000_000,
    )
    h.init()
    return h


def test_central_fold_records_session_hot_route(host):
    rc.reset()
    req = {"provider_id": "antseed", "model_family": "glm-5.2",
           "offer": {"model_family": "glm-5.2", "peer_id": "peerX"}}

    async def override(_r):
        return {"ok": True, "latency_ms": 5, "response": {"tool_calls": None}}

    asyncio.run(host._resolve_call_async(req, call_override=override, session="s1"))
    assert rc.hot_route("s1") == rr.route_key("antseed", "glm-5.2", "peerX")


def test_fold_without_session_is_noop(host):
    rc.reset()
    req = {"provider_id": "antseed", "model_family": "glm-5.2",
           "offer": {"model_family": "glm-5.2", "peer_id": "peerX"}}

    async def override(_r):
        return {"ok": True, "latency_ms": 5, "response": {}}

    asyncio.run(host._resolve_call_async(req, call_override=override))
    assert rc.snapshot() == {}   # no session -> nothing recorded


def test_failed_call_through_fold_sets_no_affinity(host):
    rc.reset()
    req = {"provider_id": "antseed", "model_family": "glm-5.2",
           "offer": {"model_family": "glm-5.2", "peer_id": "peerX"}}

    async def override(_r):
        return {"ok": False, "error": "boom", "response": {}}

    asyncio.run(host._resolve_call_async(req, call_override=override, session="s1"))
    assert rc.hot_route("s1") is None


# --------------------------------------------------------------------------
# 3. cache_hot field — true for exactly the session's hot route
# --------------------------------------------------------------------------

def test_cache_hot_marks_only_the_session_hot_route(host):
    # Two marketplace-style candidates, same family, different peers. The host
    # resolved this session's hot route into the contract; the cache_hot field
    # must mark exactly the matching peer, so a policy scoring it picks it.
    hot = rr.route_key("antseed", "glm-5.2", "peerHOT")
    base = {"model_family": "glm-5.2", "capabilities": {}}
    hot_cand = {**base, "provider_id": "antseed", "served_model_id": "HOT",
                "offer": {"model_family": "glm-5.2", "peer_id": "peerHOT"}}
    cold_cand = {**base, "provider_id": "antseed", "served_model_id": "COLD",
                 "offer": {"model_family": "glm-5.2", "peer_id": "peerCOLD"}}
    # a policy whose only scorer is the cache_hot affinity -> the hot peer ranks first
    policy = ["policy",
              ["and", ["meets_req"], ["not", ["is", "disabled"]]],
              ["gate", ["is", "cache_hot"], ["lit", 1]],
              ["argmax"], ["id"],
              ["always", {"action": "next_candidate"}]]
    ranked, _ = host.rank({
        "prompt": "x",
        "profile": "default",
        "policy_ir": policy,
        "cache_hot_route": hot,
        # isolate the pool to our two peers (the static catalog has no glm-5.2)
        "requirements": {"model_family": "glm-5.2"},
        "extra_candidates": [cold_cand, hot_cand],
    })
    assert ranked, "candidates should survive the filter"
    assert ranked[0]["candidate"]["served_model_id"] == "HOT"


def test_cache_hot_false_when_no_hot_route_in_contract(host):
    # No cache_hot_route in the contract -> the field is false for everyone, so
    # the affinity scorer adds nothing (no phantom stickiness for a new session).
    base = {"model_family": "glm-5.2", "capabilities": {}}
    a = {**base, "provider_id": "antseed", "served_model_id": "A",
         "offer": {"model_family": "glm-5.2", "peer_id": "peerA"}}
    b = {**base, "provider_id": "antseed", "served_model_id": "B",
         "offer": {"model_family": "glm-5.2", "peer_id": "peerB"}}
    policy = ["policy",
              ["and", ["meets_req"], ["not", ["is", "disabled"]]],
              ["gate", ["is", "cache_hot"], ["lit", 1]],
              ["argmax"], ["id"],
              ["always", {"action": "next_candidate"}]]
    ranked, _ = host.rank({
        "prompt": "x", "profile": "default", "policy_ir": policy,
        "requirements": {"model_family": "glm-5.2"},   # isolate to our two peers
        "extra_candidates": [a, b],
    })
    assert ranked
    # cache_hot is false for everyone (no hot route in the contract): both peers
    # survive, the affinity scorer adds nothing, no phantom stickiness.
    assert {r["candidate"]["served_model_id"] for r in ranked} == {"A", "B"}
