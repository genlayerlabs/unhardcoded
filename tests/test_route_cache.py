"""Essence for per-session cache affinity (the cache_hot field).

Caching is STATE between calls: a provider discounts the reused prefix only if
the SAME route serves the session again. The host remembers which route last
served a session and steers the next turn back to it. Since #4b that memory is
DERIVED on the fly from the `calls` ledger (host_store.hot_route) — the route of
the session's most recent SUCCESSFUL call — instead of an in-process fold.

The algebra never sees the route key: per request the host resolves
hot_route(session) into ctx.request.cache_hot_route, and the cache_hot field
getter reconstructs each candidate's route key the same way and compares,
exposing only a Bool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import host_store  # noqa: E402
import route_reliability as rr  # noqa: E402
from llm_router_host import LLMRouterHost  # noqa: E402
from conftest import seed_call  # noqa: E402


# --------------------------------------------------------------------------
# 1. hot_route — derived from the session's most recent SUCCESSFUL call
# --------------------------------------------------------------------------

def test_hot_route_unknown_session_is_none(host_store_clean):
    assert host_store.hot_route("s1") is None
    assert host_store.hot_route(None) is None


def test_successful_call_makes_route_hot(host_store_clean):
    seed_call(session="s1", provider="antseed", family="glm-5.2", served_by="peerX")
    assert host_store.hot_route("s1") == rr.route_key("antseed", "glm-5.2", "peerX")


def test_failed_call_does_not_set_affinity(host_store_clean):
    # only successful calls (status < 400) hold the prefix; a failure carries no
    # honest "this route holds the prefix" signal.
    seed_call(session="s1", provider="antseed", family="glm-5.2",
              served_by="peerX", status=502)
    assert host_store.hot_route("s1") is None


def test_latest_successful_route_wins(host_store_clean):
    seed_call(session="s1", provider="antseed", family="glm-5.2", served_by="peerA", ts=100)
    seed_call(session="s1", provider="openrouter", family="minimax-m2.7",
              served_by="openrouter", ts=200)
    assert host_store.hot_route("s1") == rr.route_key("openrouter", "minimax-m2.7", "openrouter")


def test_auxiliary_call_counts_but_never_replaces_chat_affinity(host_store_clean):
    seed_call(session="s1", provider="openrouter", family="chat-family",
              served_by="chat-route", route="profile:default", ts=100)
    seed_call(session="s1", provider="openai", family="summarizer",
              served_by="compact-route", route="operation:compact", ts=200)

    assert host_store.hot_route("s1") == rr.route_key(
        "openrouter", "chat-family", "chat-route")
    assert host_store.session_warm("s1") == [{
        "family": "chat-family",
        "provider": "openrouter",
        "served_by": "chat-route",
    }]
    assert host_store.session_totals("s1")["calls"] == 2


def test_sessions_are_independent(host_store_clean):
    seed_call(session="s1", provider="antseed", family="glm-5.2", served_by="peerA")
    assert host_store.hot_route("s2") is None


# --------------------------------------------------------------------------
# 2. cache_hot field — true for exactly the session's hot route
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
        "requirements": {"model_family": "glm-5.2"},
        "extra_candidates": [a, b],
    })
    assert ranked
    assert {r["candidate"]["served_model_id"] for r in ranked} == {"A", "B"}
