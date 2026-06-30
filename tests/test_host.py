"""
Integration tests for llm_router_host.py.

Run from repo root:
    pytest tests -v

These tests exercise the Python -> Lua boundary: config loads, info() reports
the catalog, rank() returns plausible candidates, marketplace discovery
threads through the host, and pin short-circuits the pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router_host import LLMRouterHost  # noqa: E402


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


def test_info_reports_catalog(host):
    info = host.info()
    assert info["initialized"] is True
    assert "comput3" in info["providers_loaded"]
    assert "antseed" in info["providers_loaded"]
    assert "hermes-3-405b" in info["models_loaded"]
    assert "default" in info["profile_names"]
    # static candidates only here (no marketplace yet — that happens per-call)
    assert info["candidates"] > 0


def test_init_logs_initialization(host):
    events = [evt for _, evt, _ in host.log_records]
    assert "router_initialized" in events


def test_rank_default_profile_returns_candidates(host):
    ranked, rejected = host.rank({"prompt": "hi", "profile": "default"})
    assert len(ranked) > 0
    first = ranked[0]
    assert "candidate" in first
    assert "score" in first
    assert "score_breakdown" in first
    # scores must be monotone non-increasing
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_default_profile_score_is_finite(host):
    # (sigma-pol/v2) the default profile scores on raw fields, not the removed
    # composite atoms. Raw fields aren't bounded to [0,1] (e.g. an unnormalized
    # `context`), so the invariant is FINITENESS: never NaN/inf — those are
    # un-JSON-able and order non-deterministically.
    import math
    ranked, _ = host.rank({"prompt": "hi", "profile": "default"})
    assert ranked
    for item in ranked:
        s = item["score"]
        assert isinstance(s, (int, float)) and math.isfinite(s), f"non-finite score {s!r}"


def test_pin_short_circuits_to_single_candidate(host):
    ranked, rejected = host.rank({
        "prompt": "x",
        "profile": "default",
        "requirements": {"pin": {"provider": "comput3", "model": "hermes-3-405b"}},
    })
    assert len(ranked) == 1
    assert ranked[0]["candidate"]["provider_id"] == "comput3"
    assert ranked[0]["candidate"]["model_family"] == "hermes-3-405b"


def test_pin_to_missing_pair_returns_empty_and_reason(host):
    ranked, rejected = host.rank({
        "prompt": "x",
        "profile": "default",
        "requirements": {"pin": {"provider": "bogus", "model": "bogus"}},
    })
    assert ranked == []
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "pin_not_found"


def test_vision_need_filters_to_vision_capable_model(host):
    ranked, _ = host.rank({
        "prompt": "describe",
        "images": [{"url": "x"}],
        "profile": "default",
    })
    assert len(ranked) > 0
    for item in ranked:
        assert item["candidate"]["model_family"] == "qwen-2.5-vl-72b"


def test_tee_profile_keeps_only_tee_providers(host):
    ranked, _ = host.rank({
        "prompt": "secret",
        "profile": "tee_only",
        "requirements": {"privacy": "tee_required"},
    })
    assert len(ranked) > 0
    for item in ranked:
        assert item["candidate"]["provider_id"] == "atoma"


def test_marketplace_discovery_merges_offers_into_pool(host):
    host.set_discover_hook(lambda did: {
        "ok": True,
        "fetched_at_ms": 1_000_000,
        "offers": [
            {
                "model_family":           "llama-3.3-70b",
                "seller_endpoint":        "https://seller.example/v1",
                "price_in_usd_per_mtok":  0.05,
                "price_out_usd_per_mtok": 0.10,
                "est_tok_s":              45,
                "capabilities": {
                    "context":            128_000,
                    "supports_tools":     True,
                    "supports_json_mode": True,
                    "supports_seed":      True,
                },
            },
        ],
    } if did == "antseed_buyer_node" else {"ok": False, "error": "unknown"})

    ranked, _ = host.rank({"prompt": "x", "profile": "default"})
    market_candidates = [
        r for r in ranked
        if r["candidate"]["discovery"] == "marketplace"
    ]
    assert len(market_candidates) >= 1, "marketplace offer should appear"
    assert market_candidates[0]["candidate"]["provider_id"] == "antseed"


# (sigma-pol/v2) test_weights_override_reranks was removed: weighted scoring
# over the composite atoms is gone, so there is no weights_override to rerank by.
# Per-call ranking is now a raw `policy_ir` scorer (see test_policy_ir.py).


def test_min_tok_s_filters_on_stamped_throughput(host):
    # Engine #15: throughput is host-measured and stamped per candidate (like
    # price); the engine reads cand.tok_s and no longer seeds it from metrics.
    # A candidate stamped above the floor passes; one below it, or one with no
    # measured throughput at all, is rejected with reason min_tok_s.
    base = {"model_family": "fam", "served_model_id": "fam", "capabilities": {},
            "tier": "fallback", "api_kind": "openai_compatible", "discovery": "static"}
    fast = {**base, "provider_id": "p_fast", "tok_s": 50.0}
    slow = {**base, "provider_id": "p_slow", "tok_s": 10.0}
    unstamped = {**base, "provider_id": "p_unstamped"}  # no tok_s -> default 0
    ranked, rejected = host.rank({
        "prompt": "x",
        "profile": "default",
        "requirements": {"min_tok_s": 39},
        "extra_candidates": [fast, slow, unstamped],
    })
    surviving = {r["candidate"]["provider_id"] for r in ranked}
    assert "p_fast" in surviving
    assert "p_slow" not in surviving
    assert "p_unstamped" not in surviving
    # static catalog candidates carry no measured tok_s either, so they too fail
    reasons = {r["reason"] for r in rejected}
    assert "min_tok_s" in reasons


def test_open_circuit_breaker_zeros_score(host):
    # Inject an open breaker on comput3 via the test backdoor. now_ms is
    # frozen at 1_000_000; we open the breaker 1s ago so it's still within
    # the rate-limit TTL (default 30s).
    runtime = host.router._test.runtime()
    runtime["circuit_breakers"]["comput3"] = host.lua.table_from({
        "open": True,
        "opened_at_ms": 999_000,
        "consecutive_failures": 3,
    })

    ranked, _ = host.rank({"prompt": "x", "profile": "default"})
    comput3_items = [
        r for r in ranked if r["candidate"]["provider_id"] == "comput3"
    ]
    assert comput3_items, "comput3 should still be in survivors (filter ignores breaker)"
    for item in comput3_items:
        assert item["score"] == 0.0, "open breaker forces final score to 0"
        # Post-IR there is no breaker_open marker in the breakdown: the scorer
        # is gate(not(is "breaker_open"), ·), which zeroes the score without
        # annotating. Provider health for dashboards now comes from the core's
        # read-only provider_status, not from score breakdowns.
        assert item["score_breakdown"].get("raw", 0.0) == 0.0


def test_classify_status_maps_payment_and_common_codes():
    # 402 (out of credits: OpenRouter, AntSeed insufficient_deposits) must be
    # distinguishable from a genuinely unknown failure.
    from llm_router_host import _classify_status

    assert _classify_status(402, "insufficient credits") == "payment_required"
    assert _classify_status(401, "") == "auth_error"
    assert _classify_status(403, "") == "auth_error"
    assert _classify_status(429, "") == "rate_limit"
    assert _classify_status(500, "") == "server_error"
    assert _classify_status(302, "") == "unknown"


def test_classify_status_400_only_real_overflow_is_context_overflow():
    # A 400 is context_overflow ONLY when the provider actually says the context
    # window was exceeded — not whenever "token"/"length"/"maximum" appears.
    from llm_router_host import _classify_status

    # Genuine overflow signatures (OpenAI / OpenRouter wording + error code).
    assert _classify_status(
        400, "This endpoint's maximum context length is 400000 tokens. However, "
             "you requested about 9000001 tokens.") == "context_overflow"
    assert _classify_status(400, "context_length_exceeded") == "context_overflow"

    # max_output_tokens too SMALL is the OPPOSITE of an overflow. The "token" in
    # "max_output_tokens" must not drag it into context_overflow (that aborts).
    assert _classify_status(
        400, "Invalid 'max_output_tokens': integer below minimum value. "
             "Expected a value >= 16, but got 4 instead.") == "bad_request"
    # Other unrelated 400s are plain bad_request too.
    assert _classify_status(400, "Unsupported parameter: 'temperature'") == "bad_request"
    assert _classify_status(
        400, "Response input messages must contain the word 'json' in some form "
             "to use 'text.format' of type 'json_object'.") == "bad_request"


def test_catalog_returns_python_config(host):
    cat = host.catalog()
    assert "openrouter" not in cat.get("providers", {})  # example config has no openrouter
    assert "antseed" in cat["providers"]
    assert "hermes-3-405b" in cat["models"]
    served = cat["models"]["hermes-3-405b"]["served_by"]
    assert isinstance(served, list) and served[0]["provider"]


def test_execute_async_call_override(host):
    import asyncio
    seen = []

    async def override(request):
        seen.append(request["provider_id"])
        return {"ok": True, "latency_ms": 1,
                "response": {"text": "via-override", "finish_reason": "stop"}}

    res = asyncio.run(host.execute_async({"prompt": "hi", "profile": "default"},
                                         call_override=override))
    assert res["ok"] and res["response"]["text"] == "via-override"
    assert seen

    # mocks still win over the override for their (provider, family) pair
    host.set_mock_response(res["chosen"]["provider_id"], res["chosen"]["model_family"],
                           {"ok": True, "latency_ms": 1,
                            "response": {"text": "via-mock", "finish_reason": "stop"}})
    res2 = asyncio.run(host.execute_async({"prompt": "hi", "profile": "default"},
                                          call_override=override))
    assert res2["ok"]
    if (res2["chosen"]["provider_id"], res2["chosen"]["model_family"]) == \
            (res["chosen"]["provider_id"], res["chosen"]["model_family"]):
        assert res2["response"]["text"] == "via-mock"


def test_execute_async_threads_first_token_timeout_to_provider_request(host):
    import asyncio
    seen = []

    async def override(request):
        seen.append(request)
        return {"ok": True, "latency_ms": 1,
                "response": {"text": "via-override", "finish_reason": "stop"}}

    res = asyncio.run(host.execute_async({
        "prompt": "hi",
        "profile": "default",
        "first_token_timeout_ms": 2500,
    }, call_override=override))

    assert res["ok"]
    assert seen
    assert seen[0]["first_token_timeout_ms"] == 2500


def test_streaming_override_path_folds_route_metrics(host, host_store_clean):
    # The fix: the override (streaming) path must record a route observation too.
    # Before, the fold lived only inside the direct hook, so reliability/latency
    # stayed empty for opencode's all-streaming traffic and flow nodes — which is
    # exactly why the flow couldn't rank by latency. Derived now (#4a) from
    # route_observations, written async, so drain the queue before reading.
    import asyncio
    import host_store

    req = {"provider_id": "antseed", "model_family": "glm-5.2",
           "offer": {"model_family": "glm-5.2", "peer_id": "peerX"}}

    async def override(r):                     # a streamed result: ok + latency_ms
        return {"ok": True, "latency_ms": 12000, "response": {"tool_calls": None}}

    asyncio.run(host._resolve_call_async(req, call_override=override))
    host_store._write_q.join()                 # drain the async route-obs write
    k = "antseed|glm-5.2|peerX"
    st = host_store.route_stats()
    assert st[k]["latency_ms"] == 12000        # latency derived from the streamed call
    assert st[k]["success_rate"] == 1.0

    # A mock records too now (#15: the host owns perf, so a mocked call is measured
    # exactly like a live one — the engine no longer folds a separate EMA).
    host_store.truncate_all_for_tests()
    host.set_mock_response("antseed", "glm-5.2",
                           {"ok": True, "latency_ms": 5, "response": {}})
    asyncio.run(host._resolve_call_async(req))
    host_store._write_q.join()
    assert host_store.route_stats()[k]["latency_ms"] == 5


def test_fold_uses_top_level_route_identity(host, host_store_clean):
    # The route identity is at the request TOP LEVEL (provider_id/model_family);
    # the per-call `offer` is None for openrouter/static/partner routes. The fold
    # read family from `offer` only, so those routes NEVER folded — latency_ms
    # stayed empty and a policy couldn't rank them by speed (the z.ai/glm-5.2 case).
    import asyncio
    import host_store
    req = {"provider_id": "openrouter", "model_family": "z-ai/glm-5.2",
           "served_model_id": "z-ai/glm-5.2"}        # no offer, no peer_id
    async def override(r):
        return {"ok": True, "latency_ms": 1500, "response": {}}
    asyncio.run(host._resolve_call_async(req, call_override=override))
    host_store._write_q.join()
    # peerless route -> served_by is the provider itself
    k = "openrouter|z-ai/glm-5.2|openrouter"
    assert host_store.route_stats()[k]["latency_ms"] == 1500   # from top-level identity
