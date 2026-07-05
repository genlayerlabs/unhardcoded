"""Essence for cross-provider cost metering + prompt-cache visibility.

Proves the dollar number is accurate across providers: the provider's own
reported cost is authoritative when present (already net of cache discounts);
otherwise it is computed from the ranked price, billing cache-READ tokens at a
fraction so a cache hit is not charged at full input price; subscription
backends are $0. And `_cached_tokens` reads the cache-read metric across the
OpenAI-compat / Codex-Responses / Anthropic usage shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shim import _executed_cost_usd, _CACHE_READ_FACTOR  # noqa: E402
from llm_router_host import _cached_tokens                # noqa: E402


def _result(resp, chosen):
    return {"response": resp, "chosen": chosen}


def test_prefers_provider_reported_cost():
    # the provider's own cost wins over any computed estimate
    out = _executed_cost_usd(_result(
        {"tokens_in": 1000, "tokens_out": 10, "cost_reported": 0.00012},
        {"provider_id": "openrouter", "price_in": 5.0, "price_out": 15.0}))
    assert out == 0.00012


def test_subscription_is_zero_even_if_priced():
    out = _executed_cost_usd(_result(
        {"tokens_in": 1000, "tokens_out": 10, "cost_reported": 0.5},
        {"provider_id": "codex"}),
        subscription_providers=frozenset(["codex"]))
    assert out == 0.0


def test_computed_fallback_discounts_cached_tokens():
    # no reported cost -> compute; the 800 cached input tokens bill at the
    # cache-read fraction, not full input price.
    out = _executed_cost_usd(_result(
        {"tokens_in": 1000, "tokens_out": 10, "tokens_cached": 800},
        {"provider_id": "x", "price_in": 5.0, "price_out": 15.0}))
    expected = round(200 / 1e6 * 5.0
                     + 800 / 1e6 * 5.0 * _CACHE_READ_FACTOR
                     + 10 / 1e6 * 15.0, 6)
    assert out == expected
    # and it is strictly cheaper than charging the cached tokens at full price
    full = round(1000 / 1e6 * 5.0 + 10 / 1e6 * 15.0, 6)
    assert out < full


def test_negative_price_never_bills_negative():
    out = _executed_cost_usd(_result(
        {"tokens_in": 1000, "tokens_out": 10},
        {"provider_id": "x", "price_in": -5.0, "price_out": -1.0}))
    assert out == 0.0


def test_cached_tokens_across_usage_shapes():
    assert _cached_tokens({"prompt_tokens_details": {"cached_tokens": 1280}}) == 1280
    assert _cached_tokens({"input_tokens_details": {"cached_tokens": 7}}) == 7
    assert _cached_tokens({"cache_read_input_tokens": 42}) == 42
    assert _cached_tokens({"prompt_tokens": 10}) is None
    assert _cached_tokens(None) is None


def test_session_owner_binds_first_writer_wins(host_store_clean):
    # Cross-consumer isolation: the FIRST consumer to use a sid owns it (#4b: the
    # owner is the caller of the session's EARLIEST call, derived from `calls`). A
    # second consumer reusing the same opaque sid must NOT steal ownership.
    from conftest import seed_call
    import host_store
    assert host_store.session_owner("sidA") is None         # unknown -> None
    seed_call(session="sidA", caller="keyA", tokens_in=10, cost_usd=0.001, ts=100)
    assert host_store.session_owner("sidA") == "keyA"
    # B reuses the same sid later: data is the same session, but the owner is the
    # first writer (earliest call).
    seed_call(session="sidA", caller="keyB", ts=200)
    assert host_store.session_owner("sidA") == "keyA"
    assert host_store.session_owner(None) is None           # no session -> None, never raises


def test_ingress_record_carries_session_to_the_ledger(host_store_clean):
    """The per-session meter DERIVES from the ingress call ledger: a request
    recorded with a session must make the sid resolvable (owner + totals).
    Regression: the proxy built its event without `session`, every row landed
    with session=NULL, and /v1/session/{sid} answered 404 for every consumer
    even though the shim had pinned the session fine."""
    import auth_proxy
    import host_store

    auth_proxy._record_request(
        caller="keyZ", method="POST", path="/v1/chat/completions", status=200,
        latency_ms=1.0, session="sid-ledger", provider="p", model_family="f",
        served_model_id="m", served_by="p", requested_model=None,
        tokens_in=5, tokens_out=2, tokens_total=7, tokens_cached=0,
        cost_usd=0.001, key_sha256="ab" * 32)

    assert host_store.session_owner("sid-ledger") == "keyZ"
    totals = host_store.session_totals("sid-ledger")
    assert totals["calls"] == 1
    assert totals["tokens_in"] == 5
    assert totals["tokens_out"] == 2
