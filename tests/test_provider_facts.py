import asyncio
from pathlib import Path

import httpx

import pytest
from fastapi.testclient import TestClient

import auth_proxy
import control_plane_client
import host_store
import provider_facts
from llm_router_host import LLMRouterHost
from shim import create_app

ROOT = Path(__file__).resolve().parents[1]
NOW = 1_000_000


def make_router_client():
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


@pytest.fixture
def router_client():
    return make_router_client()


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
    client.app.state.provider_facts.refresh()
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
    client.app.state.provider_facts.refresh()
    response = chat(client, "fact:eu_hosted")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "no_candidates"
    assert attempts == []


def test_failed_refresh_retains_snapshot_until_it_expires(router_client, monkeypatch, caplog):
    client, _, _ = router_client
    facts = client.app.state.provider_facts
    clock = [100.0]
    monkeypatch.setattr(provider_facts, "monotonic", lambda: clock[0])
    monkeypatch.setattr(host_store, "load_provider_facts",
                        lambda _limit: [("eu_hosted", "comput3", NOW)])
    assert chat(client, "fact:eu_hosted").status_code == 503
    facts.refresh()
    assert chat(client, "fact:eu_hosted").status_code == 200

    def unavailable(_limit):
        raise RuntimeError("offline")

    monkeypatch.setattr(host_store, "load_provider_facts", unavailable)
    facts.refresh()
    facts.refresh()
    assert caplog.text.count("Provider fact refresh failed") == 1
    assert chat(client, "fact:eu_hosted").status_code == 200
    clock[0] += 61
    assert chat(client, "fact:eu_hosted").status_code == 503


@pytest.mark.asyncio
async def test_other_host_picks_up_observation_on_next_tick(host_store_clean, monkeypatch):
    first, _, _ = make_router_client()
    second, _, _ = make_router_client()
    first_facts = first.app.state.provider_facts
    second_facts = second.app.state.provider_facts
    first_facts.refresh()

    refreshed = asyncio.Event()
    loop = asyncio.get_running_loop()
    refresh = second_facts.refresh

    def signal_refresh():
        refresh()
        loop.call_soon_threadsafe(refreshed.set)

    monkeypatch.setattr(second_facts, "refresh", signal_refresh)
    monkeypatch.setenv("ROUTER_PROVIDER_FACT_REFRESH_S", "1")
    monkeypatch.setattr(provider_facts.random, "uniform", lambda *_: 0.8)
    task = asyncio.create_task(second_facts.run())
    try:
        await asyncio.wait_for(refreshed.wait(), 1)
        refreshed.clear()
        assert await asyncio.to_thread(first_facts.write, "comput3", "eu_hosted", True, NOW, None)
        assert first_facts.is_true("eu_hosted", "comput3", NOW, 60_000)
        assert not second_facts.is_true("eu_hosted", "comput3", NOW, 60_000)
        await asyncio.wait_for(refreshed.wait(), 1)
        assert second_facts.is_true("eu_hosted", "comput3", NOW, 60_000)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("field", ["undeclared", "fact:", "fact:.bad", "fact:" + "a" * 65])
def test_invalid_fields_are_rejected(router_client, field):
    client, attempts, _ = router_client
    response = chat(client, field)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_policy"
    assert attempts == []


def test_internal_observation_round_trip(host_store_clean, router_client, monkeypatch):
    client, _, _ = router_client
    client.app.state.provider_facts.refresh()
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
    assert chat(client, "fact:eu_hosted").status_code == 200
    assert client.put(url, headers=headers, json={**row, "observed_at_ms": NOW - 1}).status_code == 409
    assert client.get(url, headers=headers).json() == expected

    cleared = {"value": False, "observed_at_ms": NOW + 1}
    assert client.put(url, headers=headers, json=cleared).status_code == 200
    assert client.get(url, headers=headers).json() == {**expected, **cleared, "detail": None}
    assert chat(client, "fact:eu_hosted").status_code == 503


@pytest.mark.asyncio
async def test_ingress_publishes_to_its_router(host_store_clean, router_client, monkeypatch):
    client, _, _ = router_client
    facts = client.app.state.provider_facts
    facts.refresh()
    monkeypatch.setattr(control_plane_client, "CONTROL_PLANE_INTERNAL_SECRET", "test-secret")
    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app)) as upstream,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=auth_proxy.app),
                          base_url="http://ingress.test") as ingress,
    ):
        monkeypatch.setattr(auth_proxy, "_client", upstream)
        response = await ingress.put(
            "/internal/provider-facts/comput3/eu_hosted",
            headers={"x-internal-secret": "test-secret"},
            json={"value": True, "observed_at_ms": NOW},
        )
        assert response.status_code == 200
        assert facts.is_true("eu_hosted", "comput3", NOW, 60_000)


def test_cache_limit_applies_to_refresh_and_local_writes(host_store_clean, monkeypatch, caplog):
    monkeypatch.setattr(provider_facts, "_MAX_ROWS", 2)
    facts = provider_facts.ProviderFacts()
    for provider_id in ("a", "b", "c"):
        host_store_clean.upsert_provider_fact(provider_id, "ready", True, NOW)
    facts.refresh()
    assert "reached the 2-row limit" in caplog.text
    assert all(facts.is_true("ready", provider_id, NOW, 60_000) for provider_id in ("a", "b"))
    assert not facts.is_true("ready", "c", NOW, 60_000)

    assert facts.write("c", "ready", True, NOW, None)
    assert facts.is_true("ready", "c", NOW, 60_000)
    assert not facts.is_true("ready", "a", NOW, 60_000)
