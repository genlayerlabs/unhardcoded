"""
End-to-end wiring against the real config.live.lua catalog: the async shim +
api_kind dispatcher route the openai_codex provider to the Codex backend and
everything else to the OpenAI-compatible backend. Mirrors __main__'s wiring,
with both backends faked so no network is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router_host import LLMRouterHost, make_api_kind_dispatcher  # noqa: E402
from shim import create_app  # noqa: E402


def _build_client(default_handler, codex_handler):
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=Path(__file__).resolve().parents[1] / "config.live.lua",
        metrics_path=Path(__file__).resolve().parents[1] / "metrics.live.lua",
        call_provider_async=make_api_kind_dispatcher(
            default=default_handler,
            handlers={"openai_codex": codex_handler},
        ),
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
        "model": "pin:openai/gpt-5.5",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "from-codex"
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
    assert body["x_router"]["provider"] != "openai", "cascaded off the failing codex provider"


def test_antseed_is_marketplace_with_no_static_rows():
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
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


def test_marketplace_offers_rank_with_offer_prices():
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
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
