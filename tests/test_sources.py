"""
Unit tests for the provider-sources seam: the push_prices bridge into
host.update_metrics, and the refresh scheduler's failure isolation.
No network: sources are faked.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sources as src  # noqa: E402


CATALOG = {
    "providers": {"openrouter": {"auth_env": "OPENROUTER_API_KEY"}},
    "models": {
        "gpt-5.5": {"served_by": [
            {"provider": "antseed_edge", "provider_model_id": "gpt-5.5"},
            {"provider": "openrouter", "provider_model_id": "openai/gpt-5.5"},
        ]},
    },
}

class FakeHost:
    def __init__(self):
        self.pushed = []

    def update_metrics(self, provider, family, delta):
        self.pushed.append((provider, family, delta))


def test_push_prices_only_pushes_cataloged_pairs():
    host = FakeHost()
    prices = [
        {"provider_id": "openrouter", "served_model_id": "openai/gpt-5.5",
         "model_family": "gpt-5.5",
         "price_in_usd_per_mtok": 5.0, "price_out_usd_per_mtok": 30.0},
        # unmapped: family None — must be skipped
        {"provider_id": "openrouter", "served_model_id": "nex-agi/nex-n2-pro:free",
         "model_family": None,
         "price_in_usd_per_mtok": 0.0, "price_out_usd_per_mtok": 0.0},
        # family not served by this provider in the catalog — skipped
        {"provider_id": "openrouter", "served_model_id": "x/y",
         "model_family": "no-such-family",
         "price_in_usd_per_mtok": 1.0, "price_out_usd_per_mtok": 1.0},
    ]
    pushed = src.push_prices(host, CATALOG, prices)
    assert pushed == 1
    (provider, family, delta), = host.pushed
    assert (provider, family) == ("openrouter", "gpt-5.5")
    assert delta["price_in"] == 5.0 and delta["price_out"] == 30.0
    assert isinstance(delta["price_refreshed_at"], int)


class FakeSource:
    name = "fake"
    provider_ids = ["openrouter"]
    poll_interval_s = 3600

    def __init__(self, prices=None, balances=None, boom=False):
        self._prices, self._balances, self._boom = prices or [], balances or {}, boom

    async def pricing(self):
        if self._boom:
            raise RuntimeError("upstream down")
        return self._prices

    async def balances(self):
        return self._balances


def test_refresh_once_updates_state():
    src.SOURCE_STATE.clear()
    host = FakeHost()
    s = FakeSource(
        prices=[{"provider_id": "openrouter", "served_model_id": "openai/gpt-5.5",
                 "model_family": "gpt-5.5",
                 "price_in_usd_per_mtok": 5.0, "price_out_usd_per_mtok": 30.0}],
        balances={"openrouter": {"kind": "credits_usd", "value": 276.6,
                                  "detail": {}, "fetched_at": 1}},
    )
    asyncio.run(src.refresh_once(host, CATALOG, s))
    state = src.SOURCE_STATE["fake"]
    assert state["error"] is None
    assert state["prices_pushed"] == 1
    assert state["balances"]["openrouter"]["value"] == 276.6
    assert isinstance(state["last_ok"], int)


def test_refresh_once_isolates_failures():
    src.SOURCE_STATE.clear()
    host = FakeHost()
    asyncio.run(src.refresh_once(host, CATALOG, FakeSource(boom=True)))  # must not raise
    state = src.SOURCE_STATE["fake"]
    assert "upstream down" in state["error"]
    assert host.pushed == []


def test_build_registry_includes_openrouter_only_when_configured():
    reg = src.build_registry(CATALOG)
    assert [s.name for s in reg] == ["openrouter"]
    assert src.build_registry({"providers": {"antseed": {}}}) == []


# ---- openrouter source ----------------------------------------------------

OR_MODELS_BODY = {
    "data": [
        # pricing strings are USD PER TOKEN (OpenRouter convention)
        {"id": "openai/gpt-5.5", "context_length": 400000,
         "pricing": {"prompt": "0.000005", "completion": "0.00003"}},
        {"id": "nex-agi/nex-n2-pro:free",
         "pricing": {"prompt": "0", "completion": "0"}},
    ],
}
OR_CREDITS_BODY = {"data": {"total_credits": 26660, "total_usage": 26383.4}}


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code, self._body = status_code, body

    def json(self):
        return self._body


class FakeClient:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    async def get(self, url, headers=None, timeout=None):
        self.calls.append((url, headers))
        for suffix, resp in self.routes.items():
            if url.endswith(suffix):
                return resp
        return FakeResponse(404, {})


def _or_source(routes, env=None):
    from sources.openrouter import OpenRouterSource
    env = env if env is not None else {"OPENROUTER_API_KEY": "sk-test"}
    return OpenRouterSource(CATALOG, env_get=env.get, client=FakeClient(routes))


def test_openrouter_pricing_maps_and_converts():
    s = _or_source({"/models": FakeResponse(200, OR_MODELS_BODY)})
    prices = asyncio.run(s.pricing())
    by_id = {p["served_model_id"]: p for p in prices}
    gpt = by_id["openai/gpt-5.5"]
    assert gpt["model_family"] == "gpt-5.5"            # via catalog served_by
    assert gpt["price_in_usd_per_mtok"] == 5.0          # 0.000005 * 1e6
    assert gpt["price_out_usd_per_mtok"] == 30.0
    assert by_id["nex-agi/nex-n2-pro:free"]["model_family"] is None  # unmapped


def test_openrouter_balances_from_credits():
    s = _or_source({"/credits": FakeResponse(200, OR_CREDITS_BODY)})
    balances = asyncio.run(s.balances())
    b = balances["openrouter"]
    assert b["kind"] == "credits_usd"
    assert round(b["value"], 1) == 276.6
    assert isinstance(b["fetched_at"], int)


def test_openrouter_balances_without_key_is_empty():
    s = _or_source({"/credits": FakeResponse(200, OR_CREDITS_BODY)}, env={})
    assert asyncio.run(s.balances()) == {}


def test_openrouter_pricing_http_error_raises():
    s = _or_source({"/models": FakeResponse(500, {})})
    with pytest.raises(RuntimeError):
        asyncio.run(s.pricing())


# ---- openrouter live discovery (whole catalog as marketplace offers) -------

OR_DISCOVERY_BODY = {
    "data": [
        # curated (mapped to family gpt-5.5 in CATALOG) -> served by the static
        # provider, so discovery SKIPS it (no duplicate candidate)
        {"id": "openai/gpt-5.5", "context_length": 400000,
         "pricing": {"prompt": "0.000005", "completion": "0.00003"}},
        # uncurated, full data -> a first-class discovered family with benchmarks
        {"id": "z-ai/glm-5.2", "context_length": 200000,
         "pricing": {"prompt": "0.0000004", "completion": "0.0000016"},
         "benchmarks": {"artificial_analysis": {"intelligence_index": 60,
                                                "coding_index": 50, "agentic_index": 55},
                        "design_arena": [{"win_rate": 70}]},
         "architecture": {"input_modalities": ["text", "image"],
                          "output_modalities": ["text"]},
         "supported_parameters": ["tools", "tool_choice", "structured_outputs", "reasoning"]},
        # uncurated, cheaper, weaker bench -> ranks below glm-5.2
        {"id": "acme/cheap-7b", "context_length": 32000,
         "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
         "benchmarks": {"artificial_analysis": {"intelligence_index": 30}}},
        # non-numeric pricing -> not routable, skipped
        {"id": "img/diffusion", "pricing": {"prompt": "0.01", "completion": "n/a"}},
    ],
}


def test_openrouter_provider_ids_wire_market_discovery():
    # the discover hook (serve.make_discover_hook) wires offers_sync by provider_id
    from sources.openrouter import OpenRouterSource
    assert "openrouter_market" in OpenRouterSource.provider_ids


def test_openrouter_offers_empty_before_first_refresh():
    s = _or_source({"/models": FakeResponse(200, OR_DISCOVERY_BODY)})
    assert s.offers_sync("openrouter_market") == []  # cold cache: no blocking fetch


def test_openrouter_discovery_offers_carry_full_live_traits():
    s = _or_source({"/models": FakeResponse(200, OR_DISCOVERY_BODY)})
    asyncio.run(s.pricing())  # populates the snapshot + live traits
    offers = {o["model_family"]: o for o in s.offers_sync("openrouter_market")}
    # curated family skipped (no dup); non-numeric pricing skipped
    assert set(offers) == {"glm-5.2", "cheap-7b"}
    glm = offers["glm-5.2"]
    assert glm["wire_model_id"] == "z-ai/glm-5.2"
    assert glm["seller_endpoint"].endswith("openrouter.ai/api/v1")
    assert round(glm["price_in_usd_per_mtok"], 4) == 0.4
    assert round(glm["price_out_usd_per_mtok"], 4) == 1.6
    assert glm["capabilities"]["context"] == 200000
    t = glm["traits"]
    assert t["bench_intelligence"] == 0.6 and t["bench_coding"] == 0.5
    assert t["bench_arena"] == 0.7 and t["in_image"] is True
    assert t["cap_tools"] is True and t["cap_reasoning"] is True
    # ranks are dynamic, across the live catalog (glm beats the weaker cheap-7b)
    assert t["bench_intelligence_rank"] == 1
    assert offers["cheap-7b"]["traits"]["bench_intelligence_rank"] == 2


def test_openrouter_discovery_derives_policy_families_from_raw_model_ids():
    from sources.openrouter import OpenRouterSource

    catalog = {
        "providers": {
            "openrouter": {"auth_env": "OPENROUTER_API_KEY"},
            "openrouter_market": {
                "discovery": "marketplace",
                "discovery_id": "openrouter_market",
                "service_aliases": {
                    "qwen/qwen3-235b-a22b-2507": "qwen3-235b-a22b",
                },
            },
        },
        "models": {
            "gpt-5.5": {"served_by": [
                {"provider": "openrouter", "provider_model_id": "openai/gpt-5.5"},
            ]},
        },
    }
    body = {"data": [
        {"id": "openai/gpt-5.5", "pricing": {
            "prompt": "0.000005", "completion": "0.00003"}},
        {"id": "openai/gpt-5-mini", "pricing": {
            "prompt": "0.00000025", "completion": "0.000002"}},
        {"id": "meta-llama/llama-4-maverick", "pricing": {
            "prompt": "0.0000002", "completion": "0.0000006"}},
        {"id": "unknown/raw-family", "pricing": {
            "prompt": "0.0000001", "completion": "0.0000001"}},
        {"id": "qwen/qwen3-235b-a22b-2507", "pricing": {
            "prompt": "0.0000005", "completion": "0.000001"}},
    ]}
    s = OpenRouterSource(catalog, env_get={"OPENROUTER_API_KEY": "sk-test"}.get,
                         client=FakeClient({"/models": FakeResponse(200, body)}))

    asyncio.run(s.pricing())

    offers = {o["model_family"]: o for o in s.offers_sync("openrouter_market")}
    assert set(offers) == {
        "gpt-5-mini", "llama-4-maverick", "raw-family", "qwen3-235b-a22b"}
    assert offers["gpt-5-mini"]["wire_model_id"] == "openai/gpt-5-mini"
    assert offers["llama-4-maverick"]["wire_model_id"] == "meta-llama/llama-4-maverick"
    assert offers["raw-family"]["wire_model_id"] == "unknown/raw-family"
    assert offers["qwen3-235b-a22b"]["wire_model_id"] == "qwen/qwen3-235b-a22b-2507"
    assert "openai/gpt-5-mini" not in offers
    assert "gpt-5.5" not in offers  # curated static OpenRouter route stays deduped


def test_openrouter_market_book_exposes_meta_for_dashboard():
    s = _or_source({"/models": FakeResponse(200, OR_DISCOVERY_BODY)})
    asyncio.run(s.pricing())
    book = s.market_book()
    fams = {r["model_family"]: r for r in book["rows"]}
    assert set(fams) == {"glm-5.2", "cheap-7b"}
    assert fams["glm-5.2"]["source"] == "openrouter"
    assert book["families"]["glm-5.2"]["meta"]["bench_intelligence"] == 0.6


def test_openrouter_model_meta_still_keyed_by_curated_family():
    s = _or_source({"/models": FakeResponse(200, OR_DISCOVERY_BODY)})
    meta = asyncio.run(s.model_meta())
    # only curated families land in the registered (static) meta
    assert set(meta) == {"gpt-5.5"}


def test_openrouter_discovery_rejects_negative_priced_models():
    # OpenRouter's "-1" = unpriced/variable sentinel parses to a negative price.
    # Admitting it would make the candidate win every cost-led policy (most
    # negative = "cheapest") and bill a NEGATIVE cost. Free ($0) models stay.
    body = {"data": [
        {"id": "z-ai/glm-5.2", "pricing": {"prompt": "0.0000004", "completion": "0.0000016"}},
        {"id": "free/free-model", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "junk/sentinel", "pricing": {"prompt": "-1", "completion": "-1"}},
        {"id": "junk/neg-out", "pricing": {"prompt": "0.000001", "completion": "-0.5"}},
    ]}
    s = _or_source({"/models": FakeResponse(200, body)})
    prices = asyncio.run(s.pricing())  # populates the snapshot + offers cache
    # discovery offers: negatives gone, the $0 free model kept
    assert {o["model_family"] for o in s.offers_sync("openrouter_market")} == \
        {"glm-5.2", "free-model"}
    # the curated price-enrichment list (pricing()) also drops negatives
    served = {p["served_model_id"] for p in prices}
    assert "junk/sentinel" not in served and "junk/neg-out" not in served
    assert {"z-ai/glm-5.2", "free/free-model"} <= served


def test_openrouter_discovery_stamps_capability_flags_for_meets_req():
    # the core's meets_req filters on capabilities.supports_{tools,vision,...};
    # discovery must translate OpenRouter's supported_parameters into those flags
    # or the whole live long tail is unroutable for any tools/vision/json request.
    body = {"data": [
        {"id": "z-ai/glm-5.2", "context_length": 200000,
         "pricing": {"prompt": "0.0000004", "completion": "0.0000016"},
         "architecture": {"input_modalities": ["text", "image"]},
         "supported_parameters": ["tools", "tool_choice", "response_format", "seed"]},
        {"id": "plain/text-only",
         "pricing": {"prompt": "0.000001", "completion": "0.000002"},
         "supported_parameters": ["temperature"]},
    ]}
    s = _or_source({"/models": FakeResponse(200, body)})
    asyncio.run(s.pricing())
    offers = {o["model_family"]: o for o in s.offers_sync("openrouter_market")}
    caps = offers["glm-5.2"]["capabilities"]
    assert caps.get("supports_tools") is True
    assert caps.get("supports_json_mode") is True
    assert caps.get("supports_seed") is True
    assert caps.get("supports_vision") is True
    # a model that does not declare tools must NOT claim the capability
    assert "supports_tools" not in offers["text-only"]["capabilities"]


def test_pushed_price_lands_in_dump_state():
    from llm_router_host import LLMRouterHost
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "core" / "config.example.lua",
        metrics_path=ROOT / "core" / "metrics.example.lua",
        now_ms=lambda: 1_000_000,
    )
    host.init()
    cat = host.catalog()
    provider, family = next((s["provider"], f) for f, m in cat["models"].items()
                            for s in m["served_by"])
    pushed = src.push_prices(host, cat, [{
        "provider_id": provider, "served_model_id": family,
        "model_family": family,
        "price_in_usd_per_mtok": 123.0, "price_out_usd_per_mtok": 456.0,
    }])
    assert pushed == 1
    ema = host.dump_state()["ema_metrics"][f"{provider}|{family}"]
    assert ema["price_in"] == 123.0 and ema["price_out"] == 456.0
    ranked, _ = host.rank({"prompt": "x", "profile": "default"})
    assert ranked  # ranking still functions on the updated store


def test_attach_sources_starts_and_stops_refresh_tasks(monkeypatch):
    """serve.attach_sources must wire the refresh loop into the app lifespan
    on the FastAPI/Starlette version actually installed (regression: the
    first deploy used the removed add_event_handler API)."""
    import time as _time
    from fastapi.testclient import TestClient
    from llm_router_host import LLMRouterHost
    import serve
    from shim import create_app

    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "core" / "config.example.lua",
        metrics_path=ROOT / "core" / "metrics.example.lua",
        now_ms=lambda: 1_000_000,
    )
    host.init()
    app = create_app(host, default_profile="default")

    src.SOURCE_STATE.clear()
    fake = FakeSource(prices=[], balances={"openrouter": {
        "kind": "credits_usd", "value": 1.0, "detail": {}, "fetched_at": 1}})
    monkeypatch.setattr(src, "build_registry", lambda catalog, env_get=None: [fake])

    serve.attach_sources(app, host)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200  # lifespan ran, app serves
        deadline = _time.time() + 3
        while _time.time() < deadline and "fake" not in src.SOURCE_STATE:
            _time.sleep(0.05)
        assert src.SOURCE_STATE["fake"]["last_ok"] is not None
    src.SOURCE_STATE.clear()


# ---- adapter error_map ------------------------------------------------------

def test_error_map_overrides_status_classification():
    from llm_router_host import _parse_openai_response

    class Resp:
        status_code = 502
        text = ""
        def json(self):
            return {"error": "Pinned peer 1d90f467 is outside your buyer routing policy."}

    plain = _parse_openai_response(Resp(), 10)
    assert plain["error_kind"] == "server_error"          # status fallback

    mapped = _parse_openai_response(Resp(), 10, error_map={
        "outside your buyer routing policy": "model_unavailable",
    })
    assert mapped["error_kind"] == "model_unavailable"     # substring map wins


def test_parse_surfaces_real_upstream_message_from_openrouter_envelope():
    # OpenRouter relays an upstream failure as a generic "Provider returned
    # error" envelope, stashing the real reason in metadata.raw. The adapter
    # must surface THAT reason (so the caller learns max_tokens was too small),
    # and classify on it — a max_output_tokens floor is a bad_request that falls
    # through, never a context_overflow that aborts.
    from llm_router_host import _parse_openai_response

    upstream = ('{"error": {"message": "Invalid \'max_output_tokens\': integer '
                'below minimum value. Expected a value >= 16, but got 4 '
                'instead.", "type": "invalid_request_error", "param": '
                '"max_output_tokens", "code": "integer_below_min_value"}}')

    class Resp:
        status_code = 400
        text = ""
        def json(self):
            return {"error": {"message": "Provider returned error", "code": 400,
                              "metadata": {"raw": upstream, "provider_name": "Azure"}}}

    r = _parse_openai_response(Resp(), 10)
    assert "max_output_tokens" in r["error_message"]        # real reason surfaced
    assert "Provider returned error" not in r["error_message"]  # not the envelope
    assert r["error_kind"] == "bad_request"                 # falls through, not abort


# ---- antseed source ---------------------------------------------------------

ANTSEED_CATALOG = {
    "providers": {
        "antseed_free": {
            "discovery": "marketplace", "discovery_id": "antseed_free",
            "base_url": "http://localhost:8377/v1",
            "market_price_cap": {"input": 0, "output": 0},
            "service_aliases": {"qwen3-235b-instruct": "qwen3-235b-a22b"},
        },
        "antseed_cheap": {
            "discovery": "marketplace", "discovery_id": "antseed_cheap",
            "base_url": "http://localhost:8379/v1",
            "market_price_cap": {"input": 2, "output": 10},
            "service_aliases": {"qwen3-235b-instruct": "qwen3-235b-a22b"},
        },
    },
    "models": {
        "qwen3-235b-a22b": {"static_quality_hint": 0.90,
                             "capabilities": {"context": 262000}, "served_by": []},
        "claude-sonnet-4-6": {"static_quality_hint": 0.88,
                               "capabilities": {"context": 200000}, "served_by": []},
    },
}


def _antseed_source(tmp_path, market_body=None, pins=None):
    import json as _json
    from sources.antseed import AntSeedSource
    tmp_path.mkdir(parents=True, exist_ok=True)
    market = tmp_path / "market.json"
    if market_body is None:
        market_body = (Path(__file__).parent / "fixtures" / "antseed_market.json").read_text()
    market.write_text(market_body if isinstance(market_body, str) else _json.dumps(market_body))
    # each buyer proxy is a session pinned to ONE peer; offers are restricted
    # to that peer (live finding: an unpinned buyer errors "no_peer_pinned")
    default_pins = {"antseed_free": "1d90f467689d499dc435e5744b4613c3203eb0aa",
                    "antseed_cheap": "1d90f467689d499dc435e5744b4613c3203eb0aa"}
    for pid, peer in (pins if pins is not None else default_pins).items():
        (tmp_path / f"status-{pid}.json").write_text(_json.dumps({
            "pinnedPeerId": peer, "depositsAvailable": "0.0",
            "depositsReserved": "0.0", "walletAddress": "0x0"}))
    return AntSeedSource(ANTSEED_CATALOG, market_dir=tmp_path)


def test_antseed_offers_gate_caps_aliases_and_min_price(tmp_path):
    s = _antseed_source(tmp_path)

    free = s.offers_sync("antseed_free")
    assert len(free) == 1, "only the $0 service fits the 0/0 cap"
    offer = free[0]
    assert offer["model_family"] == "qwen3-235b-a22b"      # alias-mapped
    assert offer["wire_model_id"] == "qwen3-235b-instruct" # wire name preserved
    assert offer["quality_hint"] == 0.90                   # curated, injected
    assert offer["seller_endpoint"] == "http://localhost:8377/v1"
    assert offer["price_in_usd_per_mtok"] == 0
    assert offer["peer_id"], "offer carries the peer to pin per request"

    cheap = s.offers_sync("antseed_cheap")
    fams = {o["model_family"]: o for o in cheap}
    # offers come only from the pinned peer (1d90f467...): sonnet at its
    # 0.5/2.5 price (NOT peer 0329's 0.6/3); chainscout over cap -> dropped
    assert set(fams) == {"qwen3-235b-a22b", "claude-sonnet-4-6"}
    assert fams["claude-sonnet-4-6"]["price_in_usd_per_mtok"] == 0.5
    assert fams["claude-sonnet-4-6"]["price_out_usd_per_mtok"] == 2.5


def test_antseed_offers_restricted_to_pinned_peer(tmp_path):
    s = _antseed_source(tmp_path, pins={
        "antseed_free": "0329c5d3920e301740f78d6e17b8d1a11cca9b2c",  # sells nothing free
        "antseed_cheap": "0329c5d3920e301740f78d6e17b8d1a11cca9b2c",
    })
    assert s.offers_sync("antseed_free") == []      # peer 0329 has no $0 services
    cheap = {o["model_family"]: o for o in s.offers_sync("antseed_cheap")}
    # peer 0329's sonnet price, not the cheaper one on the other peer
    assert cheap["claude-sonnet-4-6"]["price_in_usd_per_mtok"] == 0.6


def test_antseed_unpinned_buyer_uses_all_peers(tmp_path):
    # no status file yet (dump lag) -> fall back to the whole market
    s = _antseed_source(tmp_path, pins={})
    fams = {o["model_family"] for o in s.offers_sync("antseed_cheap")}
    assert "claude-sonnet-4-6" in fams


def test_antseed_pricing_rows_match_offers(tmp_path):
    s = _antseed_source(tmp_path)
    prices = asyncio.run(s.pricing())
    rows = {(p["provider_id"], p["model_family"]) for p in prices}
    assert ("antseed_free", "qwen3-235b-a22b") in rows
    assert ("antseed_cheap", "claude-sonnet-4-6") in rows


def test_antseed_drops_offer_when_cached_input_exceeds_input(tmp_path):
    # The buyer's @antseed/router-local rejects an offer whose cached-input price
    # exceeds its input price (cachedInput must be <= input) as malformed — the
    # proxy answers 502 "…outside your buyer routing policy". The router must not
    # advertise such a peer-service: it would pin a candidate the buyer refuses to
    # route to (and, for a single-seller family, make the family unavailable).
    # This is the exact prod shape that made `family:claude-fable-5` (one seller,
    # cachedInput 1.2 > input 0.54) look like "antseed is broken".
    peer = "aa11bb22cc33dd44ee55ff66aa11bb22cc33dd44"
    body = {
        "peers": [{
            "peerId": peer,
            "providers": ["openai"],
            "maxConcurrency": 4,
            "lastSeen": 1,
            "providerPricing": {"openai": {"services": {
                # valid: no cached price → admissible (under antseed_cheap cap)
                "qwen3-235b-instruct": {"inputUsdPerMillion": 1.0,
                                        "outputUsdPerMillion": 2.0},
                # malformed: cachedInput 1.2 > input 0.54 → buyer rejects it
                "claude-sonnet-4-6": {"inputUsdPerMillion": 0.54,
                                      "outputUsdPerMillion": 2.7,
                                      "cachedInputUsdPerMillion": 1.2},
            }}},
        }],
    }
    s = _antseed_source(tmp_path, market_body=body, pins={"antseed_cheap": peer})
    fams = {o["model_family"] for o in s.offers_sync("antseed_cheap")}
    assert "qwen3-235b-a22b" in fams           # valid offer kept (alias-mapped)
    assert "claude-sonnet-4-6" not in fams     # malformed cached price → dropped
    assert s.snapshot_stats()["rejected_by_buyer"] == 1


def _rep_peer(pid, rep, service="qwen3-235b-instruct"):
    d = {"peerId": pid, "providers": ["openai"], "lastSeen": 1,
         "providerPricing": {"openai": {"services": {
             service: {"inputUsdPerMillion": 1.0, "outputUsdPerMillion": 2.0}}}}}
    if rep is not None:
        d["onChainReputationScore"] = rep
    return d


def test_antseed_offer_carries_onchain_reputation(tmp_path):
    peer = "aa11bb22cc33dd44ee55ff66aa11bb22cc33dd44"
    s = _antseed_source(tmp_path, market_body={"peers": [_rep_peer(peer, 73.5)]},
                        pins={"antseed_cheap": peer})
    offer = s.offers_sync("antseed_cheap")[0]
    assert offer["model_family"] == "qwen3-235b-a22b"   # alias-mapped
    assert offer["reputation_score"] == 73.5            # carried from the dump


def test_antseed_reputation_absent_is_none_not_zero(tmp_path):
    # cold-start: an unreported reputation must stay None (never coerced to 0,
    # which a `>= floor` gate would wrongly exclude).
    peer = "aa11bb22cc33dd44ee55ff66aa11bb22cc33dd44"
    s = _antseed_source(tmp_path, market_body={"peers": [_rep_peer(peer, None)]},
                        pins={"antseed_cheap": peer})
    assert s.offers_sync("antseed_cheap")[0]["reputation_score"] is None


def test_antseed_reputation_min_drops_low_keeps_unrated(tmp_path, monkeypatch):
    import settings
    monkeypatch.setattr(settings, "_overrides", {"antseed.reputation_min": 50.0})
    lo, hi, nul = "11" * 20, "22" * 20, "33" * 20
    body = {"peers": [_rep_peer(lo, 30), _rep_peer(hi, 80), _rep_peer(nul, None)]}
    s = _antseed_source(tmp_path, market_body=body, pins={})  # unpinned -> all peers
    peers = {o["peer_id"] for o in s.offers_sync("antseed_cheap")}
    assert lo not in peers            # known-and-below-floor -> dropped
    assert hi in peers                # above floor -> kept
    assert nul in peers               # unrated -> kept (cold-start safe)
    assert s.snapshot_stats()["rejected_by_reputation"] == 1


def test_reputation_min_knob_registered_under_antseed():
    import settings
    knobs = {k["key"]: k for k in settings.current()}
    k = knobs.get("antseed.reputation_min")
    assert k is not None and k["provider"] == "antseed" and k["default"] == 0


def test_antseed_denylist_excludes_peer(tmp_path, monkeypatch):
    import settings
    a, b = "11" * 20, "22" * 20
    monkeypatch.setattr(settings, "_overrides", {"antseed.peer_denylist": [a]})
    s = _antseed_source(tmp_path, pins={},
                        market_body={"peers": [_rep_peer(a, None), _rep_peer(b, None)]})
    peers = {o["peer_id"] for o in s.offers_sync("antseed_cheap")}
    assert a not in peers and b in peers
    assert s.snapshot_stats()["denied"] == 1


def test_antseed_allowlist_restricts_to_members(tmp_path, monkeypatch):
    import settings
    a, b = "11" * 20, "22" * 20
    monkeypatch.setattr(settings, "_overrides", {"antseed.peer_allowlist": [a]})
    s = _antseed_source(tmp_path, pins={},
                        market_body={"peers": [_rep_peer(a, None), _rep_peer(b, None)]})
    peers = {o["peer_id"] for o in s.offers_sync("antseed_cheap")}
    assert a in peers and b not in peers


def test_antseed_empty_allow_deny_is_noop(tmp_path):
    a, b = "11" * 20, "22" * 20
    s = _antseed_source(tmp_path, pins={},
                        market_body={"peers": [_rep_peer(a, None), _rep_peer(b, None)]})
    assert {a, b} <= {o["peer_id"] for o in s.offers_sync("antseed_cheap")}


def test_settings_list_knob_coerces_csv_dedupes_and_validates():
    import settings
    assert settings._coerce("antseed.peer_denylist", "a, b ,a,, c") == ["a", "b", "c"]
    assert settings._coerce("antseed.peer_denylist", ["x", "x", "y"]) == ["x", "y"]
    assert settings._coerce("antseed.peer_denylist", 5) is None      # wrong shape
    knobs = {k["key"]: k for k in settings.current()}
    assert knobs["antseed.peer_allowlist"]["type"] == "list"
    assert knobs["antseed.peer_allowlist"]["default"] == []


def test_settings_validate_and_write_list_roundtrip(tmp_path, monkeypatch):
    import host_store
    import settings
    monkeypatch.setenv("ROUTER_DB_PATH", str(tmp_path / "host-store.db"))
    host_store.reset()
    settings.reload()
    new, errs = settings.validate_and_write({"antseed.peer_denylist": "p1, p2, p1"})
    assert errs == [] and new["antseed.peer_denylist"] == ["p1", "p2"]
    assert settings.get("antseed.peer_denylist") == ["p1", "p2"]
    new, errs = settings.validate_and_write({"antseed.peer_denylist": None})  # clear
    assert errs == [] and "antseed.peer_denylist" not in new


def test_antseed_stale_market_returns_no_offers(tmp_path):
    import os as _os
    s = _antseed_source(tmp_path)
    old = _os.path.getmtime(tmp_path / "market.json") - 3600
    _os.utime(tmp_path / "market.json", (old, old))
    assert s.offers_sync("antseed_free") == []
    assert s.snapshot_stats()["stale"] is True


def test_antseed_balances_from_status_files(tmp_path):
    import json as _json
    s = _antseed_source(tmp_path)
    (tmp_path / "status-antseed_free.json").write_text(_json.dumps({
        "depositsAvailable": "1.5", "depositsReserved": "0.2",
        "walletAddress": "0x7C39",
    }))
    balances = asyncio.run(s.balances())
    b = balances["antseed_free"]
    assert b["kind"] == "deposits_usdc" and b["value"] == 1.5
    assert b["detail"]["wallet"] == "0x7C39"
    s_nofiles = _antseed_source(tmp_path / "bare", pins={})
    assert asyncio.run(s_nofiles.balances()) == {}   # no status files -> absent


def test_wallet_rpc_url_default_and_disable(monkeypatch):
    from sources import antseed as a
    monkeypatch.delenv("ANTSEED_WALLET_RPC_URL", raising=False)
    assert a._wallet_rpc_url() == a._DEFAULT_BASE_RPC
    monkeypatch.setenv("ANTSEED_WALLET_RPC_URL", "")  # copied template -> still default
    assert a._wallet_rpc_url() == a._DEFAULT_BASE_RPC
    for off in ("off", "none", "disabled"):
        monkeypatch.setenv("ANTSEED_WALLET_RPC_URL", off)
        assert a._wallet_rpc_url() is None
    monkeypatch.setenv("ANTSEED_WALLET_RPC_URL", "https://my.rpc")
    assert a._wallet_rpc_url() == "https://my.rpc"


def test_fetch_chain_balances_rejects_bad_address():
    from sources import antseed as a
    # a short/non-0x address never triggers a network call (returns {} offline).
    assert asyncio.run(a._fetch_chain_balances(a._DEFAULT_BASE_RPC, "0x7C39")) == {}
    assert asyncio.run(a._fetch_chain_balances(a._DEFAULT_BASE_RPC, "")) == {}


def test_fetch_chain_balances_parses_eth_and_usdc(monkeypatch):
    from sources import antseed as a
    addr = "0x" + "ab" * 20  # valid 42-char address

    class _Resp:
        def raise_for_status(self): pass
        def json(self):
            # 1 ETH (1e18 wei) and 12.5 USDC (12_500_000 at 6 decimals).
            return [{"id": 1, "result": hex(10 ** 18)},
                    {"id": 2, "result": hex(12_500_000)}]

    class _Client:
        def __init__(self, *a_, **k_): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a_): return False
        async def post(self, url, json=None): return _Resp()

    import httpx as _httpx  # the helper does its own `import httpx`; patch the module
    monkeypatch.setattr(_httpx, "AsyncClient", _Client)
    out = asyncio.run(a._fetch_chain_balances("https://rpc", addr))
    assert out == {"wallet_eth": 1.0, "wallet_usdc": 12.5}


def test_balances_enriched_with_wallet_chain(tmp_path, monkeypatch):
    import json as _json
    from sources import antseed as a
    s = _antseed_source(tmp_path)
    (tmp_path / "status-antseed_free.json").write_text(_json.dumps({
        "depositsAvailable": "1.5", "depositsReserved": "0.0",
        "walletAddress": "0x" + "cd" * 20,
    }))

    async def _fake(rpc, address):
        return {"wallet_usdc": 7.0, "wallet_eth": 0.0}
    monkeypatch.setattr(a, "_fetch_chain_balances", _fake)
    monkeypatch.setenv("ANTSEED_WALLET_RPC_URL", a._DEFAULT_BASE_RPC)

    b = asyncio.run(s.balances())["antseed_free"]
    assert b["detail"]["wallet_usdc"] == 7.0
    assert b["detail"]["wallet_eth"] == 0.0

    # disabled RPC -> no on-chain fields, escrow still present
    monkeypatch.setenv("ANTSEED_WALLET_RPC_URL", "off")
    b2 = asyncio.run(s.balances())["antseed_free"]
    assert "wallet_usdc" not in b2["detail"] and b2["value"] == 1.5


def test_served_pairs_includes_marketplace_families():
    pairs = src._served_pairs(ANTSEED_CATALOG)
    assert ("antseed_free", "qwen3-235b-a22b") in pairs
    assert ("antseed_cheap", "claude-sonnet-4-6") in pairs


def test_discover_hook_serves_antseed_offers(tmp_path):
    s = _antseed_source(tmp_path)
    import serve
    hook = serve.make_discover_hook([s])
    r = hook("antseed_free")
    assert r["ok"] is True
    assert r["offers"][0]["model_family"] == "qwen3-235b-a22b"
    assert isinstance(r["fetched_at_ms"], int)
    assert hook("nope") == {"ok": False, "error": "unknown discovery_id"}
    # empty results must NOT be cached as ok (the core caches ok results for
    # the discovery TTL — a router that starts before the first market dump
    # would otherwise serve no antseed offers until the TTL expires)
    s2 = _antseed_source(tmp_path / "empty", market_body='{"peers": []}') if False else None
    import json as _json, os as _os
    (tmp_path / "market.json").write_text(_json.dumps({"peers": []}))
    r2 = hook("antseed_free")
    assert r2["ok"] is False and "no offers" in r2["error"]


# ---- codex passive source ---------------------------------------------------

def test_codex_source_aggregates_signals_into_quota_balance():
    from sources.codex import CodexSource
    src.SOURCE_STATE.clear()
    s = CodexSource("openai")
    now = int(time.time())
    s.ingest("openai", {"status": 200, "headers": {"x-codex-primary-used-percent": "37.5"}, "ts": now})
    s.ingest("openai", {"status": 429, "headers": {}, "ts": now})
    balances = asyncio.run(s.balances())
    b = balances["openai"]
    assert b["kind"] == "quota_window"
    assert b["value"] == 0.375                      # parsed *used-percent header / 100
    assert b["detail"]["recent_429_count"] == 1     # 429 within the recent window
    assert b["detail"]["last_429_at"] == now
    # passive sources publish synchronously — no refresh task exists for them
    assert src.SOURCE_STATE["codex"]["balances"]["openai"]["kind"] == "quota_window"


def test_codex_source_without_parseable_headers_has_none_value():
    from sources.codex import CodexSource
    src.SOURCE_STATE.clear()
    s = CodexSource("openai")
    s.ingest("openai", {"status": 200, "headers": {}, "ts": 1})
    b = asyncio.run(s.balances())["openai"]
    assert b["value"] is None                       # honest: unknown fraction
    assert b["detail"]["recent_429_count"] == 0


def test_build_registry_adds_codex_for_openai_codex_api_kind():
    cat = {"providers": {"openai": {"api_kind": "openai_codex"}}, "models": {}}
    reg = src.build_registry(cat)
    assert [s.name for s in reg] == ["codex"]
    assert reg[0].poll_interval_s == 30             # local self-refresh tick (no endpoint probe)


def test_codex_scarcity_prices_ramp():
    from sources.codex import CodexSource
    src.SOURCE_STATE.clear()
    host = FakeHost()
    s = CodexSource("openai")
    s.bind(host, ["gpt-5.5", "gpt-5.3-codex-spark"])

    def last_prices():
        out = {}
        for provider, family, delta in host.pushed:
            out[family] = (delta["price_in"], delta["price_out"])
        return out

    now = int(time.time())
    # 0% used (below demote start): price stays 0
    s.ingest("openai", {"status": 200, "headers": {"x-codex-primary-used-percent": "0"}, "ts": now})
    assert last_prices() == {"gpt-5.5": (0.0, 0.0),
                              "gpt-5.3-codex-spark": (0.0, 0.0)}

    # 75% used with start=0.5: halfway up the ramp -> 2.5 / 12.5
    host.pushed.clear()
    s.ingest("openai", {"status": 200, "headers": {"x-codex-primary-used-percent": "75"}, "ts": now})
    assert last_prices()["gpt-5.5"] == (2.5, 12.5)

    # 100%: full imputed price 5 / 25
    host.pushed.clear()
    s.ingest("openai", {"status": 200, "headers": {"x-codex-primary-used-percent": "100"}, "ts": now})
    assert last_prices()["gpt-5.5"] == (5.0, 25.0)

    # header back to 0 (and no recent 429s): price decays back DOWN to 0
    host.pushed.clear()
    s.ingest("openai", {"status": 200, "headers": {"x-codex-primary-used-percent": "0"}, "ts": now})
    assert last_prices()["gpt-5.5"] == (0.0, 0.0)


def test_quota_demotion_sheds_codex_from_cheap_bands():
    from llm_router_host import LLMRouterHost
    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        now_ms=lambda: 1,
    )
    host.init()

    # No tiers: a caller expresses its own price ceiling in its policy
    # (cmp price_out <= cap). Same scarcity mechanism, now per-call.
    def policy(cap):
        return ["policy",
                ["and", ["meets_req"], ["not", ["is", "disabled"]],
                 ["cmp", "price_out", "le", cap]],
                ["neg", ["normalize", ["field", "price_in"]]],
                ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]

    def pairs(cap):
        ranked, _ = host.rank({"policy_ir": policy(cap), "requirements": {"context": 8000}})
        return [(r["candidate"]["provider_id"], r["candidate"]["model_family"]) for r in ranked]

    assert ("openai_codex", "gpt-5.3-codex-spark") in pairs(5)   # free quota: under a $5 ceiling
    # quota exhausted: imputed full price on every codex family (gpt-5.5 is the
    # codex ROUTE of the unified family; spark is its own family)
    for fam in ("gpt-5.5", "gpt-5.3-codex-spark"):
        host.update_metrics("openai_codex", fam, {"price_in": 5.0, "price_out": 25.0})
    assert ("openai_codex", "gpt-5.3-codex-spark") not in pairs(5)   # $25 out > $5 ceiling
    assert ("openai_codex", "gpt-5.5") in pairs(25)                  # admitted under a $25 ceiling


# ---- full-market book (dashboard) -----------------------------------------

def test_antseed_market_book_full_market_with_tradability(tmp_path):
    s = _antseed_source(tmp_path)
    book = s.market_book()
    rows = {(r["model_family"], r["seller"]): r for r in book["rows"]}
    # the pinned peer's sonnet is tradable only via the buyer whose cap admits it
    pinned = rows[("claude-sonnet-4-6", "1d90f467689d499dc435e5744b4613c3203eb0aa")]
    assert set(pinned["pinned_by"]) == {"antseed_free", "antseed_cheap"}
    assert pinned["tradable_via"] == ["antseed_cheap"]  # 0.5/2.5 over free's 0/0 cap
    other = rows[("claude-sonnet-4-6", "0329c5d3920e301740f78d6e17b8d1a11cca9b2c")]
    assert other["pinned_by"] == [] and other["tradable_via"] == []
    qwen = rows[("qwen3-235b-a22b", "1d90f467689d499dc435e5744b4613c3203eb0aa")]
    assert "antseed_free" in qwen["tradable_via"]
    assert book["families"]["claude-sonnet-4-6"]["sellers_total"] == 2
    # uncurated services are exposed under their raw wire name, not hidden:
    # chainscout has no curated family, so it enters the book keyed by service.
    chainscout = rows[("chainscout", "1d90f467689d499dc435e5744b4613c3203eb0aa")]
    assert chainscout["price_in"] == 4 and chainscout["price_out"] == 24
    # it's over both buyers' caps (0/0 and 2/10), so tradable via none of them
    assert chainscout["tradable_via"] == []
    # peer announcement freshness is carried through for the dashboard view
    assert pinned["last_seen"] == 1781091331928


def test_antseed_market_book_caps_at_top3_plus_pinned(tmp_path):
    peers = [{"peerId": f"peer-{i}", "providerPricing": {"openai": {
        "services": {"claude-sonnet-4-6": {
            "inputUsdPerMillion": 0.1 * (i + 1), "outputUsdPerMillion": 1}}}}}
        for i in range(5)]
    s = _antseed_source(tmp_path, market_body={"peers": peers},
                        pins={"antseed_cheap": "peer-4"})
    book = s.market_book()
    sellers = [r["seller"] for r in book["rows"]
               if r["model_family"] == "claude-sonnet-4-6"]
    # 3 cheapest plus the pinned peer (what the router can actually call),
    # even though it is the most expensive of the five
    assert sellers == ["peer-0", "peer-1", "peer-2", "peer-4"]
    assert book["families"]["claude-sonnet-4-6"]["sellers_total"] == 5


def test_refresh_once_stores_market_book(tmp_path):
    s = _antseed_source(tmp_path)
    host = FakeHost()
    asyncio.run(src.refresh_once(host, ANTSEED_CATALOG, s))
    book = src.SOURCE_STATE["antseed"]["book"]
    assert book["rows"] and isinstance(book["fetched_at"], int)
    src.SOURCE_STATE.clear()
