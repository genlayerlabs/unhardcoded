"""BYO OpenRouter: the whole live catalog, discovered on demand without operator refreshers."""
import asyncio
from pathlib import Path

import httpx
import pytest

from llm_router_host import LLMRouterHost
from saas_routes import choices
import tenant_providers as tp

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    {"id": "vendor/curated-model", "pricing": {"prompt": "0.000002", "completion": "0.000008"},
     "context_length": 200000, "supported_parameters": ["tools"],
     "links": {"details": "/api/v1/models/vendor/curated-model/endpoints"}},
    {"id": "newlab/brand-new-model", "pricing": {"prompt": "0.0000001", "completion": "0.0000004"},
     "context_length": 131072, "supported_parameters": ["tools", "response_format"],
     "architecture": {"input_modalities": ["text", "image"]},
     "links": {"details": "/api/v1/models/newlab/brand-new-model/endpoints"}},
    {"id": "other/variable-price", "pricing": {"prompt": "-1", "completion": "-1"}, "context_length": 8192},
]


@pytest.fixture
def base(host_store_clean):
    tp._cache.clear()
    tp._openrouter_public.update(at=0.0, value=None)
    host = LLMRouterHost(ROOT/'core/router.lua', ROOT/'tests/fixtures/openrouter_byo.lua',
        discover=lambda _: {'ok': True, 'offers': [{'model_family': 'OPERATOR-ONLY'}]})
    host.init()
    return host


@pytest.fixture
def openrouter(monkeypatch):
    """Mock OpenRouter's public API; records every request path."""
    state = {"paths": [], "fail": False, "models": MODELS}
    real = httpx.AsyncClient

    def network(request):
        state["paths"].append(request.url.path + ("?" + request.url.query.decode() if request.url.query else ""))
        assert request.url.host == "openrouter.ai"
        assert "authorization" not in request.headers  # the listing needs no tenant credential
        if state["fail"]:
            return httpx.Response(503)
        if request.url.query:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"data": state["models"]})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(network)))
    return state


def tenant(base, tenant_id, key="sk-or-tenant"):
    return base.for_tenant(tenant_id, {"OPENROUTER_API_KEY": key} if key else {})


def test_byo_openrouter_lists_live_catalog_once_with_curated_prices(base, openrouter):
    child = tenant(base, 1)
    asyncio.run(tp.prepare(child))
    rows = {row["id"]: row for row in choices(child)}
    assert set(rows) == {"openrouter|curated-model", "openrouter_market|brand-new-model"}
    assert {row["label"].split(" · ")[1] for row in rows.values()} == {"OpenRouter"}
    assert (rows["openrouter|curated-model"]["price_in"], rows["openrouter|curated-model"]["price_out"]) == pytest.approx((2.0, 8.0))
    new = rows["openrouter_market|brand-new-model"]
    assert (new["price_in"], new["price_out"]) == pytest.approx((0.1, 0.4))
    assert new["tools"] is True
    assert child._connection_errors == {}
    # Only the listing (+ decision listing): no per-model endpoint detail requests.
    assert not any("/endpoints" in path for path in openrouter["paths"])


def test_new_models_appear_after_the_cache_expires(base, openrouter, monkeypatch):
    asyncio.run(tp.prepare(tenant(base, 1)))
    openrouter["models"] = MODELS + [{"id": "lab/released-today", "context_length": 32768,
                                      "pricing": {"prompt": "0.0000003", "completion": "0.0000009"}}]
    child = tenant(base, 2)
    asyncio.run(tp.prepare(child))
    assert "openrouter_market|released-today" not in {r["id"] for r in choices(child)}  # within TTL: shared snapshot
    monkeypatch.setattr(tp, "OPENROUTER_CATALOG_TTL_S", 0)
    child = tenant(base, 2)
    asyncio.run(tp.prepare(child))
    assert "openrouter_market|released-today" in {r["id"] for r in choices(child)}


def test_listing_is_shared_across_tenants_and_fetched_once_per_ttl(base, openrouter):
    async def run():
        await asyncio.gather(*(tp.prepare(tenant(base, tid)) for tid in (1, 2, 3)))
    asyncio.run(run())
    assert [p for p in openrouter["paths"] if p.endswith("/models")] == ["/api/v1/models"]


def test_without_openrouter_key_nothing_is_fetched_and_operator_discovery_is_not_used(base, openrouter):
    child = tenant(base, 1, key=None)
    asyncio.run(tp.prepare(child))
    assert openrouter["paths"] == []
    assert choices(child) == []


def test_failure_reports_error_and_never_falls_back_to_operator(base, openrouter):
    openrouter["fail"] = True
    child = tenant(base, 1)
    asyncio.run(tp.prepare(child))
    ids = {r["id"] for r in choices(child)}
    assert not any("OPERATOR-ONLY" in i for i in ids)
    assert not any(i.startswith("openrouter_market|") for i in ids)
    assert "sk-or" not in str(child._connection_errors)
    assert child._connection_errors["openrouter"]


def test_short_outage_serves_the_last_public_listing(base, openrouter, monkeypatch):
    asyncio.run(tp.prepare(tenant(base, 1)))
    monkeypatch.setattr(tp, "OPENROUTER_CATALOG_TTL_S", 0)
    openrouter["fail"] = True
    child = tenant(base, 2)
    asyncio.run(tp.prepare(child))
    assert "openrouter_market|brand-new-model" in {r["id"] for r in choices(child)}
    assert child._connection_errors == {}
