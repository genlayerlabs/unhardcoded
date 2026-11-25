from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import control_plane_client
import host_store
import internal_api
from llm_router_host import LLMRouterHost
from shim import create_app

ROOT = Path(__file__).resolve().parents[1]
NOW = 1_000_000
MIN_BOND = 10**18


def policy(bonded=False):
    predicates = [["meets_req"], ["not", ["is", "disabled"]]]
    if bonded:
        predicates.append(["is", "bonded"])
    return ["policy", ["and", *predicates], ["lit", 0], ["argmax"], ["id"],
            ["always", {"action": "next_candidate"}]]


@pytest.fixture
def router_client():
    attempts = []
    failures = set()

    async def backend(req):
        attempts.append(req["provider_id"])
        if req["provider_id"] in failures:
            return {"ok": False, "error": {"kind": "server_error", "status": 500}}
        return {"ok": True, "response": {"text": "ok", "finish_reason": "stop",
                                        "tokens_in": 1, "tokens_out": 1}}

    host = LLMRouterHost(
        ROOT / "core/router.lua", ROOT / "core/config.example.lua",
        call_provider_async=backend, now_ms=lambda: NOW, enforce_provider_auth=False,
        env={"ROUTER_MIN_PROVIDER_BOND": str(MIN_BOND),
             "ROUTER_PROVIDER_BOND_MAX_AGE_S": "60"},
    )
    host.init()
    return TestClient(create_app(host)), attempts, failures


def chat(client, bonded):
    return client.post("/v1/chat/completions", json={
        "policy_ir": policy(bonded), "messages": [{"role": "user", "content": "hi"}],
    })


def test_bonded_route_uses_only_qualifying_provider(host_store_clean, router_client):
    host_store_clean.upsert_provider_agreement("comput3", "0x1", "test", MIN_BOND, "active", NOW)
    client, attempts, _ = router_client
    assert chat(client, bonded=True).status_code == 200
    assert attempts == ["comput3"]


def test_bonded_route_never_retries_unbonded_providers(host_store_clean, router_client):
    host_store_clean.upsert_provider_agreement("comput3", "0x1", "test", MIN_BOND, "active", NOW)
    client, attempts, failures = router_client
    failures.add("comput3")
    assert chat(client, bonded=True).status_code == 502
    assert attempts and set(attempts) == {"comput3"}


@pytest.mark.parametrize("status,bond,checked_at", [
    ("exiting", MIN_BOND, NOW),
    ("defaulted", MIN_BOND, NOW),
    ("none", MIN_BOND, NOW),
    ("active", MIN_BOND - 1, NOW),
    ("active", MIN_BOND, NOW - 60_000),
    ("active", MIN_BOND, NOW + 1),
])
def test_bonded_route_fails_closed(host_store_clean, router_client, status, bond, checked_at):
    host_store_clean.upsert_provider_agreement("comput3", "0x1", "test", bond, status, checked_at)
    client, attempts, _ = router_client
    response = chat(client, bonded=True)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "no_candidates"
    assert attempts == []


def test_other_routes_skip_registry(router_client, monkeypatch):
    queries = []

    def unavailable(*args):
        queries.append(args)
        raise RuntimeError("offline")

    monkeypatch.setattr(host_store, "bonded_provider_ids", unavailable)
    client, attempts, _ = router_client
    assert chat(client, bonded=False).status_code == 200
    assert attempts and not queries
    attempts.clear()
    assert chat(client, bonded=True).status_code == 503
    assert len(queries) == 1
    assert attempts == []


def test_tenant_routes_inherit_operator_bond_thresholds(host_store_clean):
    host = LLMRouterHost(
        ROOT / "core/router.lua", ROOT / "core/config.example.lua",
        now_ms=lambda: NOW,
        env={"ROUTER_MIN_PROVIDER_BOND": str(2 * MIN_BOND),
             "ROUTER_PROVIDER_BOND_MAX_AGE_S": "60"},
    )
    host.init()
    tenant = host.for_tenant(1, {"COMPUT3_API_KEY": "test-key",
                                 "ROUTER_MIN_PROVIDER_BOND": str(MIN_BOND)})
    contract = {"policy_ir": policy(bonded=True)}
    host_store_clean.upsert_provider_agreement("comput3", "0x1", "test", MIN_BOND, "active", NOW)
    assert tenant.rank(contract)[0] == []
    host_store_clean.upsert_provider_agreement("comput3", "0x1", "test", 2 * MIN_BOND, "active", NOW)
    assert tenant.rank(contract)[0]


def test_internal_agreement_round_trip(host_store_clean, monkeypatch):
    app = FastAPI()
    app.include_router(internal_api.router)
    client = TestClient(app)
    url = "/internal/provider-agreements/comput3"
    row = {"agreement_address": "0x1", "network": "test", "status": "active",
           "bond_amount": str(2**256 - 1), "checked_at_ms": NOW}
    monkeypatch.setattr(control_plane_client, "CONTROL_PLANE_INTERNAL_SECRET", "")
    assert client.put(url, json=row).status_code == 404
    monkeypatch.setattr(control_plane_client, "CONTROL_PLANE_INTERNAL_SECRET", "test-secret")
    assert client.put(url, json=row).status_code == 403
    assert client.get(url).status_code == 403
    headers = {"x-internal-secret": "test-secret"}
    assert client.get(url, headers=headers).status_code == 404
    assert client.put(url, headers=headers, json={**row, "bond_amount": "-1"}).status_code == 400
    assert client.put(url, headers=headers, json=row).status_code == 200
    assert client.put(url, headers=headers, json={**row, "checked_at_ms": NOW - 1}).status_code == 409
    assert client.get(url, headers=headers).json() == {"provider_id": "comput3", **row}
    cleared = {"bond_amount": "0", "status": "none", "checked_at_ms": NOW + 1}
    assert client.put(url, headers=headers, json=cleared).status_code == 200
    assert client.get(url, headers=headers).json() == {
        "provider_id": "comput3", "agreement_address": None, "network": None, **cleared,
    }
