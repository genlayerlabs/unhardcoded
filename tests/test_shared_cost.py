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

    limits = {("openai", BASE, model): shared_cost.AttemptLimit(256, 5, Decimal(cost))
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
