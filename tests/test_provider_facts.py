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


@pytest.fixture
def router_client():
    attempts = []
    failures = set()

    async def backend(req):
        provider_id = req["provider_id"]
        attempts.append(provider_id)
        if provider_id in failures:
            return {"ok": False, "error": {"kind": "server_error", "status": 500}}
        return {"ok": True, "response": {"text": "ok", "finish_reason": "stop",
                                        "tokens_in": 1, "tokens_out": 1}}

    host = LLMRouterHost(
        ROOT / "core/router.lua", ROOT / "core/config.example.lua",
        call_provider_async=backend, now_ms=lambda: NOW, enforce_provider_auth=False,
        env={"ROUTER_PROVIDER_FACT_MAX_AGE_S": "60"},
    )
    host.init()
    return TestClient(create_app(host)), attempts, failures


def chat(client, *fields):
    predicates = [["meets_req"], ["not", ["is", "disabled"]]]
    predicates.extend(["is", field] for field in fields)
    return client.post("/v1/chat/completions", json={
        "policy_ir": ["policy", ["and", *predicates], ["lit", 0], ["argmax"], ["id"],
                      ["always", {"action": "next_candidate"}]],
        "messages": [{"role": "user", "content": "hi"}],
    })


def test_combines_facts_and_retries_only_eligible_providers(host_store_clean, router_client):
    host_store_clean.upsert_provider_fact("comput3", "eu_hosted", True, NOW - 60_000)
    host_store_clean.upsert_provider_fact("comput3", "review.v2-ok", True, NOW)
    host_store_clean.upsert_provider_fact("morpheus", "eu_hosted", True, NOW)
    client, attempts, failures = router_client
    fields = ("fact:eu_hosted", "fact:review.v2-ok")
    assert chat(client, *fields).status_code == 200
    assert attempts == ["comput3"]

    attempts.clear()
    failures.add("comput3")
    assert chat(client, *fields).status_code == 502
    assert attempts and set(attempts) == {"comput3"}


@pytest.mark.parametrize("observation", [None, (False, NOW), (True, NOW - 60_001), (True, NOW + 1)])
def test_missing_false_stale_or_future_fact_excludes_provider(host_store_clean, router_client, observation):
    if observation is not None:
        host_store_clean.upsert_provider_fact("comput3", "eu_hosted", *observation)
    client, attempts, _ = router_client
    response = chat(client, "fact:eu_hosted")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "no_candidates"
    assert attempts == []


def test_fact_lookup_is_lazy_and_cached_per_ranking_context(router_client, monkeypatch):
    queries = []

    def unavailable(*args):
        queries.append(args)
        raise RuntimeError("offline")

    monkeypatch.setattr(host_store, "providers_with_fact", unavailable)
    client, attempts, _ = router_client
    assert chat(client).status_code == 200
    assert attempts and not queries

    attempts.clear()
    assert chat(client, "fact:eu_hosted").status_code == 503
    assert queries == [("eu_hosted", 60_000, NOW)]
    assert attempts == []


@pytest.mark.parametrize("field", ["undeclared", "fact:", "fact:.bad", "fact:" + "a" * 65])
def test_invalid_fields_are_rejected(router_client, field):
    client, attempts, _ = router_client
    response = chat(client, field)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_policy"
    assert attempts == []


def test_internal_observation_round_trip(host_store_clean, monkeypatch):
    app = FastAPI()
    app.include_router(internal_api.router)
    client = TestClient(app)
    url = "/internal/provider-facts/comput3/eu_hosted"
    row = {"value": True, "observed_at_ms": NOW, "detail": {"regions": ["eu"], "version": 1}}
    monkeypatch.setattr(control_plane_client, "CONTROL_PLANE_INTERNAL_SECRET", "")
    assert client.put(url, json=row).status_code == 404
    monkeypatch.setattr(control_plane_client, "CONTROL_PLANE_INTERNAL_SECRET", "test-secret")
    assert client.put(url, json=row).status_code == 403
    assert client.get(url).status_code == 403

    headers = {"x-internal-secret": "test-secret"}
    assert client.get(url, headers=headers).status_code == 404
    for invalid in ({"value": "true"}, {"observed_at_ms": True}, {"observed_at_ms": 2**63}):
        assert client.put(url, headers=headers, json={**row, **invalid}).status_code == 400
    assert client.put(url + "!", headers=headers, json=row).status_code == 400

    expected = {"provider_id": "comput3", "name": "eu_hosted", **row}
    assert client.put(url, headers=headers, json=row).json() == expected
    assert client.put(url, headers=headers, json={**row, "observed_at_ms": NOW - 1}).status_code == 409
    assert client.get(url, headers=headers).json() == expected

    cleared = {"value": False, "observed_at_ms": NOW + 1}
    assert client.put(url, headers=headers, json=cleared).status_code == 200
    assert client.get(url, headers=headers).json() == {**expected, **cleared, "detail": None}
