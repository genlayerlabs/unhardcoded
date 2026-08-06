"""
End-to-end wiring against the real config.live.lua catalog: the async shim +
api_kind dispatcher route provider-specific protocols to their own backends.
Mirrors __main__'s wiring, with backends faked so no network is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router_host import LLMRouterHost  # noqa: E402
from provider_adapters.dispatcher import make_api_kind_dispatcher  # noqa: E402
from shim import create_app  # noqa: E402

LIVE_TEST_ENV = {
    "OPENAI_API_KEY": "sk-openai-test",
    "OPENROUTER_API_KEY": "sk-openrouter-test",
}


def _build_client(default_handler, codex_handler):
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=Path(__file__).resolve().parents[1] / "config.live.lua",
        metrics_path=Path(__file__).resolve().parents[1] / "metrics.live.lua",
        call_provider_async=make_api_kind_dispatcher(
            default=default_handler,
            handlers={"openai_codex": codex_handler},
        ),
        env=LIVE_TEST_ENV.copy(),
        now_ms=lambda: 1,
    )
    host.init()
    return TestClient(create_app(host, default_profile="default"))


def test_pin_routes_codex_through_dispatcher():
    async def default(req):
        return {"ok": False, "error_kind": "server_error"}

    async def codex(req):
        assert req["api_kind"] == "openai_codex"
        assert req["served_model_id"] == "gpt-5.5"
        return {"ok": True, "latency_ms": 5,
                "response": {"text": "from-codex", "finish_reason": "stop"}}

    client = _build_client(default, codex)
    r = client.post("/v1/chat/completions", json={
        "model": "pin:openai_codex/gpt-5.5",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "from-codex"
    assert body["x_router"]["provider"] == "openai_codex"


def test_pin_openai_routes_native_api_not_codex():
    async def default(req):
        assert req["api_kind"] == "openai_compatible"
        assert req["provider_id"] == "openai"
        assert req["served_model_id"] == "gpt-5.5"
        return {"ok": True, "latency_ms": 5,
                "response": {"text": "from-openai", "finish_reason": "stop"}}

    async def codex(req):
        return {"ok": False, "error_kind": "server_error"}

    client = _build_client(default, codex)
    r = client.post("/v1/chat/completions", json={
        "model": "pin:openai/gpt-5.5",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "from-openai"
    assert body["x_router"]["provider"] == "openai"


def test_vanilla_openai_request_uses_default_and_cascades():
    # A client keeps its ordinary OpenAI model string and sends no policy_ir.
    # The shim deliberately maps that to the product default; when Codex fails,
    # the default must cascade and still serve a response.
    async def default(req):
        return {"ok": True, "latency_ms": 5,
                "response": {"text": f"served-by-{req['provider_id']}", "finish_reason": "stop"}}

    async def codex(req):
        return {"ok": False, "error_kind": "server_error"}

    client = _build_client(default, codex)
    r = client.post("/v1/chat/completions", json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"].startswith("served-by-")
    assert body["x_router"]["provider"] != "openai_codex", \
        "cascaded off the failing codex provider"


def test_antseed_is_marketplace_with_no_static_rows():
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env=LIVE_TEST_ENV.copy(),
        now_ms=lambda: 1,
    )
    host.init()
    cat = host.catalog()
    # one AntSeed buyer (no tiers): a single marketplace provider
    p = cat["providers"]["antseed"]
    assert p["discovery"] == "marketplace"
    assert p["discovery_id"] == "antseed"
    assert "market_price_cap" in p and "error_map" in p
    assert [pid for pid in cat["providers"] if str(pid).startswith("antseed")] == ["antseed"]
    assert "deepseek-v3.1" not in cat["models"]     # antseed-only family removed
    for family, model in cat["models"].items():
        for served in model["served_by"]:
            assert not str(served["provider"]).startswith("antseed"), \
                f"static antseed row left on {family}"


def test_live_config_includes_provider_local_native_examples():
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env=LIVE_TEST_ENV.copy(),
        now_ms=lambda: 1,
    )
    host.init()
    cat = host.catalog()

    providers = cat["providers"]
    assert providers["openai"]["api_kind"] == "openai_compatible"
    assert providers["openai_codex"]["api_kind"] == "openai_codex"
    assert providers["anthropic"]["api_kind"] == "anthropic"
    assert providers["gemini"]["api_kind"] == "google"
    assert providers["bedrock"]["api_kind"] == "bedrock"
    assert providers["bedrock"]["aws_region"] == "us-east-1"
    assert providers["bedrock_market"]["api_kind"] == "bedrock"
    assert providers["bedrock_market"]["discovery_id"] == "bedrock_market"

    def served_by(family):
        return {
            row["provider"]: row["provider_model_id"]
            for row in cat["models"][family]["served_by"]
        }

    assert served_by("gpt-5.5")["openai"] == "gpt-5.5"
    assert served_by("gpt-5.5")["openai_codex"] == "gpt-5.5"
    assert served_by("gpt-5.4")["openai_codex"] == "gpt-5.4"
    assert served_by("gpt-5.4-mini")["openai_codex"] == "gpt-5.4-mini"
    assert served_by("claude-opus-4-8")["anthropic"] == "claude-opus-4-8"
    assert "bedrock" not in served_by("claude-opus-4-8")
    assert served_by("gemini-3.1-pro-preview")["gemini"] == \
        "gemini-3.1-pro-preview"
    assert served_by("qwen3-235b-a22b")["bedrock"] == \
        "qwen.qwen3-vl-235b-a22b"


def test_missing_provider_keys_are_filtered_before_ranking():
    async def provider_call(_req):
        return {"ok": True, "response": {"text": "ok"}}

    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env={"OPENROUTER_API_KEY": "sk-openrouter"},
        now_ms=lambda: 1,
    )
    host.init()
    host.set_async_call_hook(provider_call)
    term = ["policy",
            ["and", ["meets_req"], ["not", ["is", "disabled"]],
             ["family_eq", "gpt-5.4"]],
            ["lit", 1], ["argmax"], ["id"],
            ["always", {"action": "next_candidate"}]]

    ranked, _ = host.rank({"policy_ir": term, "requirements": {"context": 8000}})
    providers = [r["candidate"]["provider_id"] for r in ranked]

    assert "openrouter" in providers
    assert "openai" not in providers
    assert host.dump_state()["disabled_providers"]["openai"]["kind"] == \
        "auth_unconfigured"

    host.set_env("OPENAI_API_KEY", "sk-openai")
    ranked, _ = host.rank({"policy_ir": term, "requirements": {"context": 8000}})
    providers = [r["candidate"]["provider_id"] for r in ranked]

    assert "openai" in providers
    assert "openai" not in host.dump_state().get("disabled_providers", {})


def test_missing_provider_keys_preserve_existing_runtime_disable():
    async def provider_call(_req):
        return {"ok": True, "response": {"text": "ok"}}

    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env={"OPENROUTER_API_KEY": "sk-openrouter"},
        now_ms=lambda: 1,
    )
    host.init()
    host.set_async_call_hook(provider_call)
    state = host.dump_state()
    state["disabled_providers"] = {
        "openai": {"kind": "auth_error", "at_ms": 1},
    }
    host.restore_state(state)

    term = ["policy",
            ["and", ["meets_req"], ["not", ["is", "disabled"]],
             ["family_eq", "gpt-5.4"]],
            ["lit", 1], ["argmax"], ["id"],
            ["always", {"action": "next_candidate"}]]
    host.rank({"policy_ir": term, "requirements": {"context": 8000}})
    assert host.dump_state()["disabled_providers"]["openai"]["kind"] == \
        "auth_error"

    host.set_env("OPENAI_API_KEY", "sk-openai")
    host.rank({"policy_ir": term, "requirements": {"context": 8000}})
    assert host.dump_state()["disabled_providers"]["openai"]["kind"] == \
        "auth_error"


def test_context_overflow_falls_through_to_the_next_candidate():
    # A context overflow on ONE route must not abort the whole request: a
    # provider-neutral family (gpt-5.4 is served by openai AND openrouter) spans
    # candidates with different context windows, so the router must try the next
    # one. Pre-fix, retry_policies.balanced.context_overflow = "abort" stopped
    # dead on the first candidate. Here every candidate overflows, so the request
    # still exhausts — but only AFTER more than one route was tried.
    calls = []

    async def default(req):
        calls.append(req["provider_id"])
        return {"ok": False, "error_kind": "context_overflow",
                "error_message": "maximum context length is 400000 tokens",
                "latency_ms": 1}

    async def codex(req):
        calls.append(req["provider_id"])
        return {"ok": False, "error_kind": "context_overflow", "latency_ms": 1}

    client = _build_client(default, codex)
    r = client.post("/v1/chat/completions", json={
        "model": "family:gpt-5.4",   # >= 2 routes (openai + openrouter)
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert len(calls) >= 2, f"expected fall-through across candidates, got {calls}"
    assert "exhausted" in r.text, r.text   # multi-candidate terminal, not a lone abort


def test_marketplace_offers_rank_with_offer_prices():
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env={**LIVE_TEST_ENV, "ANTSEED_BASE_URL": "http://localhost:8378/v1"},
        now_ms=lambda: 1,
    )
    host.set_discover_hook(lambda did: {
        "ok": True, "fetched_at_ms": 1,
        "offers": [{
            "model_family": "claude-opus-4-8", "quality_hint": 0.93,
            "wire_model_id": "claude-opus-4-8",
            "seller_endpoint": "http://localhost:8378/v1",
            "price_in_usd_per_mtok": 1.0, "price_out_usd_per_mtok": 5.0,
            "capabilities": {"context": 200000},
        }],
    } if did == "antseed" else {"ok": False, "error": "x"})
    host.init()
    ranked, _ = host.rank({"profile": "default", "requirements": {"context": 8000}})
    pairs = [(r["candidate"]["provider_id"], r["candidate"]["model_family"]) for r in ranked]
    assert ("antseed", "claude-opus-4-8") in pairs


def test_marketplace_offers_rank_with_live_price_multiplier(monkeypatch):
    import settings

    monkeypatch.setitem(settings._overrides, "openrouter.price_multiplier", 0.8)
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env=LIVE_TEST_ENV.copy(),
        now_ms=lambda: 1,
    )
    host.set_discover_hook(lambda did: {
        "ok": True, "fetched_at_ms": 1,
        "offers": [{
            "model_family": "acme/discounted",
            "wire_model_id": "acme/discounted",
            "seller_endpoint": "https://openrouter.ai/api/v1",
            "price_in_usd_per_mtok": 10.0,
            "price_out_usd_per_mtok": 20.0,
            "capabilities": {"context": 200000},
            "traits": {"bench_intelligence": 0.5},
        }],
    } if did == "openrouter_market" else {"ok": False, "error": "x"})
    host.init()
    term = ["policy",
            ["and", ["meets_req"], ["not", ["is", "disabled"]],
             ["family_eq", "acme/discounted"]],
            ["neg", ["normalize", ["field", "price_in"]]],
            ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]

    ranked, _ = host.rank({"policy_ir": term, "requirements": {"context": 8000}})

    candidate = ranked[0]["candidate"]
    assert candidate["price_in"] == 8.0
    assert candidate["price_out"] == 16.0
    assert candidate["raw_price_in"] == 10.0
    assert candidate["raw_price_out"] == 20.0
    assert candidate["price_multiplier"] == 0.8
    assert candidate["offer"]["price_in_usd_per_mtok"] == 10.0
    assert "effective_price_in_usd_per_mtok" not in candidate["offer"]


def test_static_prices_rank_with_live_price_multiplier(monkeypatch):
    import settings

    monkeypatch.setitem(settings._overrides, "openrouter.price_multiplier", 0.5)
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env=LIVE_TEST_ENV.copy(),
        now_ms=lambda: 1,
    )
    host.init()
    host.update_metrics("openrouter", "gpt-5.5", {"price_in": 10.0, "price_out": 20.0})

    term = ["policy",
            ["and", ["meets_req"], ["provider_eq", "openrouter"], ["family_eq", "gpt-5.5"]],
            ["neg", ["normalize", ["field", "price_in"]]],
            ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]

    ranked, _ = host.rank({"policy_ir": term, "requirements": {"context": 8000}})
    candidate = ranked[0]["candidate"]
    assert candidate["raw_price_in"] == 10.0
    assert candidate["raw_price_out"] == 20.0
    assert candidate["price_in"] == 5.0
    assert candidate["price_out"] == 10.0
    assert candidate["price_multiplier"] == 0.5

    monkeypatch.setitem(settings._overrides, "openrouter.price_multiplier", 2.0)
    ranked, _ = host.rank({"policy_ir": term, "requirements": {"context": 8000}})
    candidate = ranked[0]["candidate"]
    assert candidate["raw_price_in"] == 10.0
    assert candidate["price_in"] == 20.0
    assert candidate["price_multiplier"] == 2.0


def test_discovered_family_ranks_on_inline_offer_traits():
    """A discovered family (raw model id, absent from model_meta.lua) ranks on
    the live benchmark it carries inline on the offer (c.offer.traits) — the
    OpenRouter-discovery contract. Two offers at the SAME price; only the
    inline bench_intelligence differs, so the higher-bench one must win. If the
    mfield getter ignored c.offer.traits both would default to 0 and tie."""
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env=LIVE_TEST_ENV.copy(),
        now_ms=lambda: 1,
    )
    host.set_discover_hook(lambda did: {
        "ok": True, "fetched_at_ms": 1,
        "offers": [
            {"model_family": "acme/weak-7b", "wire_model_id": "acme/weak-7b",
             "seller_endpoint": "https://openrouter.ai/api/v1",
             "price_in_usd_per_mtok": 1.0, "price_out_usd_per_mtok": 1.0,
             "capabilities": {"context": 200000}, "traits": {"bench_intelligence": 0.10}},
            {"model_family": "z-ai/glm-5.2", "wire_model_id": "z-ai/glm-5.2",
             "seller_endpoint": "https://openrouter.ai/api/v1",
             "price_in_usd_per_mtok": 1.0, "price_out_usd_per_mtok": 1.0,
             "capabilities": {"context": 200000}, "traits": {"bench_intelligence": 0.90}},
        ],
    } if did == "openrouter_market" else {"ok": False, "error": "x"})
    host.init()
    ranked, _ = host.rank({"profile": "default", "requirements": {"context": 8000}})
    discovered = [r["candidate"]["model_family"] for r in ranked
                  if r["candidate"]["provider_id"] == "openrouter_market"]
    assert discovered[:2] == ["z-ai/glm-5.2", "acme/weak-7b"], \
        "higher inline bench_intelligence must outrank the weaker one at equal price"


def test_discovered_alias_family_is_policy_addressable():
    """OpenRouter marketplace aliases let policies target canonical families
    while the provider call still uses the raw OpenRouter slug."""
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env=LIVE_TEST_ENV.copy(),
        now_ms=lambda: 1,
    )
    host.set_discover_hook(lambda did: {
        "ok": True, "fetched_at_ms": 1,
        "offers": [
            {"model_family": "gpt-5-mini", "wire_model_id": "openai/gpt-5-mini",
             "seller_endpoint": "https://openrouter.ai/api/v1",
             "price_in_usd_per_mtok": 0.25, "price_out_usd_per_mtok": 2.0,
             "capabilities": {"context": 400000, "supports_tools": True,
                              "supports_json_mode": True},
             "traits": {"bench_intelligence": 0.80}},
        ],
    } if did == "openrouter_market" else {"ok": False, "error": "x"})
    host.init()
    term = ["policy",
            ["and", ["meets_req"], ["not", ["is", "disabled"]],
             ["family_eq", "gpt-5-mini"]],
            ["neg", ["normalize", ["field", "price_in"]]],
            ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]

    ranked, _ = host.rank({"policy_ir": term, "requirements": {"context": 8000}})

    assert len(ranked) == 1
    candidate = ranked[0]["candidate"]
    assert candidate["provider_id"] == "openrouter_market"
    assert candidate["model_family"] == "gpt-5-mini"
    assert candidate["offer"]["wire_model_id"] == "openai/gpt-5-mini"


# ---- AntSeed funding admission (the host envelope's third clause) -------------

def _antseed_host(env=None):
    """The live catalog with ONE AntSeed offer available from the discover hook.
    AntSeed needs no key (`auth = none`), so nothing else gates it — the funding
    clause in policy_envelope is the only thing that can keep it out."""
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        env=(env if env is not None else LIVE_TEST_ENV.copy()),
        now_ms=lambda: 1,
    )
    host.set_discover_hook(lambda did: {
        "ok": True, "fetched_at_ms": 1,
        "offers": [
            {"model_family": "gpt-5-mini", "wire_model_id": "gpt-5-mini",
             "seller_endpoint": "http://antseed:8378/v1",
             "price_in_usd_per_mtok": 0.01, "price_out_usd_per_mtok": 0.02,
             "peer_id": "peerA",
             "capabilities": {"context": 400000, "supports_tools": True,
                              "supports_json_mode": True}},
        ],
    } if did == "antseed" else {"ok": False, "error": "x"})
    host.init()
    return host


_ANY_GPT5_MINI = ["policy",
                  ["and", ["meets_req"], ["family_eq", "gpt-5-mini"]],
                  ["neg", ["normalize", ["field", "price_in"]]],
                  ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]


def _providers_for(host):
    ranked, _ = host.rank({"policy_ir": _ANY_GPT5_MINI,
                           "requirements": {"context": 8000}})
    return [r["candidate"]["provider_id"] for r in ranked]


def test_antseed_is_rejected_until_its_escrow_can_pay():
    """The envelope keeps an unfundable AntSeed buyer out of ranking. This
    FAILS CLOSED on purpose: the engine's `credits` field defaults to 0, so a
    router that has never published a balance rejects AntSeed rather than
    routing calls into a certain 402."""
    host = _antseed_host()
    # The offer exists and is the ONLY candidate for this family — so an empty
    # ranking here is the gate rejecting it, not the family being unserved.
    assert _providers_for(host) == [], \
        "credits default 0 -> below the 1.0 floor -> not routable"

    # A funded escrow admits it (this is what sources.push_credits publishes).
    host.update_metrics("__credits", "antseed", {"free_credits_remaining_usd": 5.0})
    assert "antseed" in _providers_for(host)

    # Draining back below one channel reserve takes it out again.
    host.update_metrics("__credits", "antseed", {"free_credits_remaining_usd": 0.4})
    assert "antseed" not in _providers_for(host)


_ANY_CANDIDATE = ["policy", ["and", ["meets_req"]],
                  ["neg", ["normalize", ["field", "price_in"]]],
                  ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]


def test_the_funding_clause_is_scoped_to_antseed_only():
    """Every other provider bills against its own quota/credit mechanics and has
    no on-chain escrow, so an unscoped `credits >= 1` clause would reject the
    WHOLE catalog — the engine's `credits` field defaults to 0 for all of them.
    With no credits published anywhere, everyone except antseed must still rank."""
    host = _antseed_host()
    ranked, _ = host.rank({"policy_ir": _ANY_CANDIDATE,
                           "requirements": {"context": 8000}})
    providers = {r["candidate"]["provider_id"] for r in ranked}
    assert providers - {"antseed"}, "the clause must not empty the catalog"
    assert "antseed" not in providers


def test_operator_profiles_bypass_the_envelope_so_offers_sync_is_the_cover():
    """A deliberate property of the CORE, pinned here because the funding design
    depends on knowing it: `config.policy_envelope` is ∧-ed onto CALLER-supplied
    `policy_ir` only — an operator profile is the operator's own term and is
    ranked as written (core/llm_policy.lua, "operator's own; no envelope").

    So the envelope covers caller terms, and the offer-side tourniquet in
    sources/antseed.offers_sync — which suppresses the offers themselves, before
    any policy sees them — is what covers profile traffic. Belt and braces, but
    each one covers a different half."""
    host = _antseed_host()
    ranked, _ = host.rank({"profile": "default", "requirements": {"context": 8000}})
    providers = {r["candidate"]["provider_id"] for r in ranked}
    assert "antseed" in providers, (
        "if this ever fails the core started applying the envelope to profiles "
        "too — good news, but the comment above is then stale")


def test_cold_start_seed_makes_antseed_routable_before_the_first_refresh(
        host_store_clean, monkeypatch):
    """The cold-start hole the fail-closed envelope opens, and its fix: a fresh
    pod has published no credits yet, so AntSeed would stay rejected for up to a
    full poll interval. `sources.seed_credits` publishes the last known escrow
    from the durable buyer_status row during startup, before serving."""
    import sources as sources_mod
    from conftest import seed_buyer_status
    from sources.antseed import AntSeedSource

    catalog = {"providers": {"antseed": {
        "discovery": "marketplace", "discovery_id": "antseed",
        "base_url": "http://antseed:8378/v1"}}, "models": {}}
    seed_buyer_status("antseed", deposits_available="12.5",
                      deposits_reserved="1.0", wallet_address="0x0")

    host = _antseed_host()
    assert "antseed" not in _providers_for(host)          # cold: rejected
    assert sources_mod.seed_credits(host, [AntSeedSource(catalog)]) == 1
    assert "antseed" in _providers_for(host)              # seeded: routable


def test_cold_start_seed_leaves_the_gate_closed_when_the_escrow_is_empty(
        host_store_clean):
    import sources as sources_mod
    from conftest import seed_buyer_status
    from sources.antseed import AntSeedSource

    catalog = {"providers": {"antseed": {
        "discovery": "marketplace", "discovery_id": "antseed"}}, "models": {}}
    seed_buyer_status("antseed", deposits_available="0.23",
                      deposits_reserved="15.6", wallet_address="0x0")
    host = _antseed_host()
    assert sources_mod.seed_credits(host, [AntSeedSource(catalog)]) == 1
    assert "antseed" not in _providers_for(host), \
        "seeding publishes the truth, it does not open the gate"
