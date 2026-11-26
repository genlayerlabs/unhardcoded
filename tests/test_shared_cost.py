import json
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

import control_plane_client as cp
import shared_cost
from llm_router_host import LLMRouterHost
from provider_adapters.openai_compatible import call_with_cost_allowance
from shim import create_app

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://provider.test"
MESSAGES = [{"role": "user", "content": "Hi"}]
ATTEMPT = {"api_kind": "openai_compatible", "provider_id": "openai",
           "base_url": BASE, "served_model_id": "primary",
           "messages": MESSAGES, "max_tokens": 5}


@pytest.mark.asyncio
async def test_allowance_charges_unknown_failed_attempt_and_bounds_fallback():
    posted = []

    async def upstream(request):
        posted.append(json.loads(request.content))
        return httpx.Response(503)

    limits = {("chat", "openai", BASE, model, None): shared_cost.AttemptLimit(256, 5, Decimal(cost))
              for model, cost in (("primary", ".03"), ("backup", ".04"))}
    allowance = shared_cost.Allowance(Decimal(".05"), limits, 5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        for overrides, expected in [
            ({"messages": [{"role": "user", "content": "x" * 256}]}, "shared_cost_unbounded"),
            ({"max_tokens": 6}, "shared_cost_unbounded"),
            ({}, None),
            ({"served_model_id": "backup"}, "shared_cost_ceiling_exceeded"),
            ({}, "shared_cost_ceiling_exceeded"),
        ]:
            attempt = {**ATTEMPT, **overrides}
            prepared = allowance.prepare(attempt)
            if expected is not None:
                assert prepared is None and allowance.denial == expected
            else:
                result = await call_with_cost_allowance(attempt, allowance, client, lambda _: None, prepared)
                assert not result["ok"]

    assert allowance.charge == Decimal(".03")
    assert posted == [{"model": "primary", "messages": MESSAGES, "max_tokens": 5}]


def test_trusted_shared_chat_bounds_real_engine_dispatch(monkeypatch, tmp_path):
    config = tmp_path / "router.lua"
    config.write_text((ROOT / "tests/fixtures/saas.lua").read_text()
                      .replace("http://provider.test", BASE))
    host = LLMRouterHost(ROOT / "core/router.lua", config, env={"OPENAI_API_KEY": "test"})
    host.init()
    monkeypatch.setattr(cp, "CONTROL_PLANE_URL", "https://control.test")
    monkeypatch.setattr(cp, "CONTROL_PLANE_INTERNAL_SECRET", "test-secret")
    monkeypatch.setattr(cp, "tenant_connections", AsyncMock(return_value=({}, {}, ("openai",))))
    monkeypatch.setenv("SAAS_SHARED_PROVIDERS", "openai")
    monkeypatch.setenv("SHARED_COST_ATTEMPT_LIMITS", json.dumps([{
        "provider": "openai", "base_url": BASE, "model": "primary",
        "max_input_bytes": 256, "max_output_tokens": 5, "max_cost_usd": ".04",
    }]))
    calls = []

    def upstream(request):
        calls.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            "usage": {"cost": .025},
        })

    http = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    client = TestClient(create_app(host, shared_cost_http=http))
    headers = {"x-llm-router-tenant": "42", "x-internal-secret": "test-secret",
               shared_cost.LIMIT_HEADER: ".03", shared_cost.EXPIRY_HEADER: str(time.time() + 60)}
    payload = {"model": "family:primary", "messages": MESSAGES, "max_tokens": 5}

    rejected = client.post(shared_cost.PATH, headers=headers, json=payload)
    assert rejected.json()["error"]["code"] == "shared_cost_ceiling_exceeded"
    assert rejected.headers[shared_cost.CHARGE_HEADER] == "0"
    assert calls == []

    headers[shared_cost.LIMIT_HEADER] = ".05"
    rejected = client.post(shared_cost.PATH, headers=headers, json={**payload, "_auto_contract": {}})
    assert rejected.status_code == 400
    assert calls == []

    response = client.post(shared_cost.PATH, headers=headers, json=payload)
    assert response.status_code == 200, response.text
    assert Decimal(response.headers[shared_cost.CHARGE_HEADER]) == Decimal(".025")
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("decision_bound,peer,ceiling,expected_calls,expected_status,expected_charge", [
    (".01", "peer-one", ".06", ["decision", "market", "fallback"], 200, ".05"),
    (".01", "peer-one", ".05", ["decision", "market"], 402, ".03"),
    (None, "peer-one", ".05", ["market", "fallback"], 200, ".04"),
    (None, "other-peer", ".05", ["fallback"], 200, ".02"),
])
async def test_parallel_guarded_automatic_marketplace_admission(
        host_store_clean, monkeypatch, tmp_path, decision_bound, peer, ceiling, expected_calls,
        expected_status, expected_charge):
    import asyncio
    from collections import defaultdict
    from datetime import datetime, timezone
    from functools import partial

    import auth_proxy
    import policy_selection
    from decision_providers.jev import ENDPOINTS, MODELS

    monkeypatch.setenv("AUTOMATIC_ROUTING_ENABLED", "1")
    monkeypatch.setenv("SAAS_SHARED_PROVIDERS", "antseed,custom_cloud")
    monkeypatch.setattr(cp, "CONTROL_PLANE_URL", "https://control.test")
    monkeypatch.setattr(cp, "CONTROL_PLANE_INTERNAL_SECRET", "test-secret")
    credentials = {"OPENROUTER_API_KEY": "tenant-decision"}
    managed = ["antseed", "custom_cloud"]
    monkeypatch.setattr(cp, "tenant_connections", AsyncMock(return_value=(credentials, {}, managed)))
    monkeypatch.setattr(auth_proxy, "UPSTREAM", "https://router.test")
    config = tmp_path / "router.lua"
    config.write_text((ROOT / "tests/fixtures/managed.lua").read_text()
                      .replace("http://", "https://")
                      .replace("profiles=", "fields={bench_intelligence={sort='Num',default=0}}, profiles="))

    def discover(provider):
        return {"ok": True, "offers": [{
            "model_family": "shared-model", "wire_model_id": "peer.model",
            "peer_id": "peer-one", "seller_endpoint": "https://buyer.test/v1",
            "price_in_usd_per_mtok": 1, "price_out_usd_per_mtok": 2,
        }] if provider == "antseed" else []}

    host = LLMRouterHost(ROOT / "core/router.lua", config, discover=discover,
                         env={"CUSTOM_CLOUD_SECRET": "managed-inference"})
    host.init()
    host.update_metrics("__credits", "antseed", {"free_credits_remaining_usd": 5})
    host.update_metrics("custom_cloud", "shared-model", {"price_in": 1, "price_out": 2})
    child = host.for_tenant(42, credentials, managed)
    published = policy_selection.compile_automatic(child, {
        "kind": "automatic", "use_case": "Text assistant", "allow_decision_content": True,
        "policy_ids": ["general", "coding"],
        "constraints": {"provider_order": managed, "reliability_floor": 0},
    })
    cap = Decimal(ceiling)
    published.update(route="route:assistant", revision=1, key={"id": 1, "subject": "k:1"},
                     limits={"shared_budget": {"scope": "team", "monthly_budget_usd": cap * 2,
                                               "max_call_cost_usd": cap}})
    published["execution"]["automatic_mode"] = "active"
    monkeypatch.setattr(cp, "resolve_route", AsyncMock(return_value=published))
    monkeypatch.setattr(auth_proxy, "_caller_auth_async", AsyncMock(return_value={
        "ok": True, "tenant_id": 42, "caller": "team", "meta": {},
    }))

    limits = [
        {"provider": "antseed", "base_url": "https://buyer.test/v1", "model": "peer.model",
         "peer_id": peer, "max_input_bytes": 256, "max_output_tokens": 5, "max_cost_usd": ".02"},
        {"provider": "custom_cloud", "base_url": "https://custom.test/v1", "model": "shared-model",
         "max_input_bytes": 256, "max_output_tokens": 5, "max_cost_usd": ".03"},
    ]
    if decision_bound is not None:
        limits.append({"provider": "openrouter", "protocol": "decisions",
                       "base_url": ENDPOINTS["openrouter"], "model": MODELS["openrouter"],
                       "max_input_bytes": 4096, "max_cost_usd": decision_bound})
    monkeypatch.setenv("SHARED_COST_ATTEMPT_LIMITS", json.dumps(limits))
    calls = defaultdict(list)
    initial_calls = 0
    admitted = asyncio.Event()
    release = asyncio.Event()

    async def upstream(request):
        nonlocal initial_calls
        body = json.loads(request.content)
        if request.url.path.endswith("/decisions"):
            task, leg = body["state"]["task"], "decision"
            assert request.headers["authorization"] == "Bearer tenant-decision"
        else:
            task = body["messages"][0]["content"]
            leg = "market" if request.url.host == "buyer.test" else "fallback"
            assert body["max_tokens"] == 5
        calls[task].append(leg)
        if len(calls[task]) == 1:
            initial_calls += 1
            if initial_calls == 2:
                admitted.set()
            await asyncio.wait_for(release.wait(), timeout=10)
        if leg == "decision":
            return httpx.Response(200, json={"model": MODELS["openrouter"], "usage": {"cost": .01},
                "answers": {"selection": {"type": "choice", "choice": "coding",
                    "probabilities": {"general": 0, "coding": 1}, "confidence": 1}}})
        if leg == "market":
            assert request.headers["x-antseed-pin-peer"] == "peer-one"
            assert body["model"] == "peer.model"
            return httpx.Response(503)
        assert body["model"] == "shared-model"
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}],
                                         "usage": {"cost": .02}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as provider_http:
        monkeypatch.setattr(policy_selection, "JevDecisionProvider",
                            partial(policy_selection.JevDecisionProvider, client=provider_http))
        router_app = create_app(host, shared_cost_http=provider_http)
        async with (
            httpx.AsyncClient(transport=httpx.ASGITransport(app=router_app)) as router_http,
            httpx.AsyncClient(transport=httpx.ASGITransport(app=auth_proxy.app),
                              base_url="http://ingress.test") as ingress,
        ):
            monkeypatch.setattr(auth_proxy, "_client", router_http)

            async def post(index):
                return await ingress.post("/v1/chat/completions", headers={"Authorization": "Bearer test"},
                    json={"model": "route:assistant", "max_tokens": 5,
                          "messages": [{"role": "user", "content": f"task {index}"}]})

            attempts = [asyncio.create_task(post(i)) for i in range(2)]
            try:
                await asyncio.wait_for(admitted.wait(), timeout=10)
                refused = await post(2)
                assert refused.status_code == 402, refused.text
                assert refused.json()["error"]["code"] == "shared_budget_exhausted"
            finally:
                release.set()
                results = await asyncio.gather(*attempts)
                await asyncio.to_thread(auth_proxy.drain_settles)

    assert [r.status_code for r in results] == [expected_status] * 2, [r.text for r in results]
    assert calls == {"task 0": expected_calls, "task 1": expected_calls}
    for response in results:
        selection = response.json()["x_router"]["decision_trace"]["automatic"]
        assert selection["selected_id"] == ("general" if decision_bound is None else "coding")
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    scope = host_store_clean.subject_budgets(42, month, ["s:team"])[0]
    assert scope["spent_usd"] == pytest.approx(float(Decimal(expected_charge) * 2))
    assert scope["reserved_usd"] == 0
