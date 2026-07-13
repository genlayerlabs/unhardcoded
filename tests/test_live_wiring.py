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


def _build_host(default_handler, codex_handler, *, config="config.live.lua"):
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=Path(__file__).resolve().parents[1] / config,
        metrics_path=Path(__file__).resolve().parents[1] / "metrics.live.lua",
        call_provider_async=make_api_kind_dispatcher(
            default=default_handler,
            handlers={"openai_codex": codex_handler},
        ),
        env=LIVE_TEST_ENV.copy(),
        now_ms=lambda: 1,
    )
    host.init()
    return host


def _build_client(default_handler, codex_handler):
    host = _build_host(default_handler, codex_handler)
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


def test_default_profile_cascades_off_a_failing_provider():
    # The codex backend fails; the router must cascade to an OpenAI-compatible
    # fallback candidate under the default policy and still serve a response.
    async def default(req):
        return {"ok": True, "latency_ms": 5,
                "response": {"text": f"served-by-{req['provider_id']}", "finish_reason": "stop"}}

    async def codex(req):
        return {"ok": False, "error_kind": "server_error"}

    client = _build_client(default, codex)
    r = client.post("/v1/chat/completions", json={
        "model": "profile:default",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"].startswith("served-by-")
    assert body["x_router"]["provider"] != "openai_codex", \
        "cascaded off the failing codex provider"


def test_compact_uses_live_server_profile_without_caller_policy():
    async def summarize(req):
        assert req["max_tokens"] == 512
        return {
            "ok": True,
            "latency_ms": 5,
            "response": {
                "text": "server-owned summary",
                "finish_reason": "stop",
                "tokens_in": 100,
                "tokens_out": 3,
            },
        }

    client = _build_client(summarize, summarize)
    messages = []
    for i in range(3):
        messages += [
            {"role": "user", "content": f"u{i} " * 200},
            {"role": "assistant", "content": f"a{i} " * 200},
        ]
    body = {
        "contract_version": 3,
        "messages": messages,
    }

    response = client.post("/v1/compact", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["compacted"] is True
    assert response.json()["summary"] == "server-owned summary"
    assert "messages" not in response.json()

    body["policy_ir"] = ["policy"]
    assert client.post("/v1/compact", json=body).status_code == 422

    body.pop("policy_ir")
    body["requirements"] = {
        "pin": {"provider": "openrouter", "model": "gpt-5.5"},
    }
    assert client.post("/v1/compact", json=body).status_code == 422


def test_live_compact_profile_enforces_envelope_and_model_family_narrowing():
    async def no_call(_req):
        return {"ok": False, "error_kind": "unexpected_call"}

    host = _build_host(no_call, no_call)

    ranked, _ = host.rank({"profile": "compact"})
    assert ranked
    assert all(row["candidate"].get("api_kind") == "openai_codex"
               or (row["candidate"].get("raw_price_in") is not None
                   and row["candidate"]["raw_price_in"] <= 1.0)
               for row in ranked)
    assert all(row["candidate"].get("api_kind") == "openai_codex"
               or (row["candidate"].get("raw_price_out") is not None
                   and row["candidate"]["raw_price_out"] <= 5.0)
               for row in ranked)

    narrowed, _ = host.rank({
        "profile": "compact",
        "requirements": {"model_family": "minimax-m2.7"},
    })
    assert narrowed
    assert {row["candidate"]["model_family"] for row in narrowed} == {
        "minimax-m2.7"}

    def extra(provider, *, price_in=0.1, price_out=0.5, reputation=100):
        return {
            "provider_id": provider,
            "model_family": f"compact-test-{provider}",
            "served_model_id": f"compact-test-{provider}",
            "served_by": provider,
            "api_kind": "openai_compatible",
            "base_url": "https://example.invalid/v1",
            "tier": "marketplace",
            "price_in": price_in,
            "price_out": price_out,
            "capabilities": {"context": 32000},
            "offer": {
                "reputation_score": reputation,
                "traits": {"bench_intelligence": 0.9},
            },
        }

    expensive, _ = host.rank({
        "profile": "compact",
        "extra_candidates": [extra("cost-test", price_out=5.01)],
        "requirements": {"model_family": "compact-test-cost-test"},
    })
    assert expensive == []

    multiplier_bypass, _ = host.rank({
        "profile": "compact",
        "extra_candidates": [{
            **extra("raw-cost-test", price_in=0.5, price_out=2.5),
            "raw_price_in": 2.0,
            "raw_price_out": 10.0,
        }],
        "requirements": {"model_family": "compact-test-raw-cost-test"},
    })
    assert multiplier_bypass == []

    negative_quote, _ = host.rank({
        "profile": "compact",
        "extra_candidates": [{
            **extra("negative-cost-test", price_in=-0.1, price_out=-0.1),
            "raw_price_in": -0.1,
            "raw_price_out": -0.1,
        }],
        "requirements": {"model_family": "compact-test-negative-cost-test"},
    })
    assert negative_quote == []

    low_trust, _ = host.rank({
        "profile": "compact",
        "extra_candidates": [extra("antseed", reputation=95)],
        "requirements": {"model_family": "compact-test-antseed"},
    })
    assert low_trust == []

    high_trust, _ = host.rank({
        "profile": "compact",
        "extra_candidates": [extra("antseed", reputation=96)],
        "requirements": {"model_family": "compact-test-antseed"},
    })
    assert [row["candidate"]["provider_id"] for row in high_trust] == ["antseed"]

    state = host.dump_state()
    state["disabled_providers"] = {
        **(state.get("disabled_providers") or {}),
        "openrouter": {"kind": "auth_error", "at_ms": 1},
    }
    host.restore_state(state)
    after_disable, _ = host.rank({"profile": "compact"})
    assert all(row["candidate"]["provider_id"] != "openrouter"
               for row in after_disable)


def test_compose_compact_profile_is_hermetic_free_local_ollama():
    async def no_call(_req):
        return {"ok": False, "error_kind": "unexpected_call"}

    host = _build_host(no_call, no_call, config="config.compose.lua")

    def candidate(provider="ollama", family="qwen2.5:0.5b", price=0.0):
        return {
            "provider_id": provider,
            "model_family": family,
            "served_model_id": family,
            "served_by": provider,
            "api_kind": "openai_compatible",
            "base_url": "http://ollama:11434/v1",
            "tier": "partner",
            "price_in": price,
            "price_out": price,
            "capabilities": {"context": 32000},
        }

    contract = {
        "profile": "compact_bdd_ollama",
        "requirements": {"model_family": "qwen2.5:0.5b"},
    }
    ranked, _ = host.rank({
        **contract,
        "extra_candidates": [candidate()],
    })
    assert [(row["candidate"]["provider_id"],
             row["candidate"]["model_family"]) for row in ranked] == [
        ("ollama", "qwen2.5:0.5b")]

    wrong_provider, _ = host.rank({
        **contract,
        "extra_candidates": [candidate(provider="openrouter")],
    })
    assert wrong_provider == []

    wrong_family, _ = host.rank({
        **contract,
        "extra_candidates": [candidate(family="other-local")],
    })
    assert wrong_family == []

    nonzero_price, _ = host.rank({
        **contract,
        "extra_candidates": [candidate(price=0.01)],
    })
    assert nonzero_price == []


def test_compose_wires_the_compaction_overlay_and_profile():
    compose = (ROOT / "compose.yml").read_text()
    router_service = compose.split("\n  ingress:", 1)[0]

    assert (ROOT / "config.compose.lua").is_file()
    assert "\n      - --config\n      - config.compose.lua\n" in router_service
    assert ("\n      - --compact-profile\n"
            "      - ${COMPACT_PROFILE:-compact}\n") in router_service


def test_compact_profile_aborts_after_one_provider_attempt():
    calls = []

    async def fail(req):
        calls.append((req["provider_id"], req["model_family"]))
        return {"ok": False, "error_kind": "server_error", "latency_ms": 1}

    host = _build_host(fail, fail)
    assert len(host.rank({"profile": "compact"})[0]) > 1
    client = TestClient(create_app(host, compact_profile="compact"))
    out = client.post("/v1/compact", json={
        "contract_version": 3,
        "messages": [
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
        ],
    }).json()

    assert out["compacted"] is False
    assert out["reason"] == "router_failed"
    assert len(calls) == 1


def test_compact_bdd_has_an_explicit_non_skipping_acceptance_mode():
    steps = (ROOT / "features" / "steps" / "steps.py").read_text()

    assert 'if _os.getenv("REQUIRE_COMPACT_BDD") == "1":' in steps
    assert "REQUIRE_COMPACT_BDD=1 forbids skipping this flow" in steps


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
