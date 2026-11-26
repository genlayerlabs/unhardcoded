"""Cross-replica admission and per-pod overload behavior."""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CALLER_KEYS_JSON", "{}")
os.environ.setdefault("CALLER_KEYS_SHA256_JSON", "{}")
os.environ.setdefault("CALLER_KEYS_BOOTSTRAP_JSON", "{}")

import auth_proxy  # noqa: E402
import host_store  # noqa: E402


def test_background_ledger_preserves_cost_provenance(host_store_clean, monkeypatch):
    store = host_store_clean
    monkeypatch.setattr(store, "_SYNC_WRITES", False)
    rows = [
        (0.0, "reported", "openrouter", "reported"),
        (0.01, "computed", "openai", "estimated"),
        (0.0, "subscription", "codex", "estimated"),
        (0.02, None, "openai", "estimated"),
        (0.03, "reported", "flow", "estimated"),
        (None, None, "openai", "none"),
    ]
    now = int(time.time())
    for offset, (cost, basis, provider, _) in enumerate(rows):
        store.insert_call_async({"caller": "cost-test", "ts": now + offset,
                                 "cost_usd": cost, "cost_basis": basis, "provider": provider})
    store.insert_call_async({"caller": "cost-test", "ts": now + len(rows),
                             "cost_basis": "unknown_total", "provider": "codex",
                             "winning_inference_cost_usd": 0.0,
                             "winning_inference_cost_basis": "subscription"})
    store._write_q.join()
    usage = store.usage_rows(caller="cost-test")
    assert [(row["cost_usd"], row["cost_source"]) for row in usage] == [
        (cost, source) for cost, _, _, source in rows] + [(None, "none")]
    exported = store.usage_export("cost-test", now=now + 3600)["rows"]
    assert [(row["cost_usd"], row["cost_basis"], row["cost_source"]) for row in exported] == [
        (cost, basis, source) for cost, basis, _, source in rows] + [(None, "unknown_total", "none")]
    assert exported[-1]["winning_inference_cost_usd"] == 0.0
    assert exported[-1]["winning_inference_cost_basis"] == "subscription"


def test_background_ledger_exports_served_model_and_latency(host_store_clean, monkeypatch):
    store = host_store_clean
    monkeypatch.setattr(store, "_SYNC_WRITES", False)
    store.insert_call_async({"caller": "metadata-test", "requested_model": "route:default",
                             "served_model_id": "provider/model-v2", "latency_ms": 0.0})
    store.insert_call_async({"caller": "metadata-test", "requested_model": "route:default",
                             "status": 500})
    store._write_q.join()

    rows = store.usage_export("metadata-test", now=time.time() + 3600)["rows"]
    assert [(row["served_model_id"], row["latency_ms"]) for row in rows] == [
        ("provider/model-v2", 0.0), (None, None)]


def test_background_ledger_preserves_applied_rates(host_store_clean, monkeypatch):
    store = host_store_clean
    monkeypatch.setattr(store, "_SYNC_WRITES", False)
    scope = {"project_id": 1, "environment_id": 2}
    rates = {"input_usd_per_million": 0.0, "output_usd_per_million": 2.5}
    row = {"caller": "rate-test", "decision_trace": scope}
    store.insert_call_async(row)
    store.insert_call_async({**row, **rates})
    store._write_q.join()

    priced, unknown = store.recent_calls(caller="rate-test", **scope, strict=True)
    assert {field: priced[field] for field in rates} == rates
    assert all(unknown[field] is None for field in rates)


def test_usage_key_groups_respect_retention_before_pruning(host_store_clean, monkeypatch):
    store = host_store_clean
    now = int(time.time())
    expired_key = "a" * 64
    retained_key = "b" * 64
    store.insert_call({"caller": "retention", "ts": now - 86460, "key_sha256": expired_key})
    store.insert_call({"caller": "retention", "ts": now - 60, "key_sha256": retained_key})

    monkeypatch.setattr(store, "_RETENTION_DAYS", 1)
    groups = store.usage_groups("key", caller="retention")
    assert [(group["key"], group["last_call_ts"]) for group in groups] == [
        (retained_key, now - 60)]


def test_dashboard_issued_digest_resolves_without_local_key(host_store_clean):
    token = "only-known-by-the-shared-store"
    digest = hashlib.sha256(token.encode()).hexdigest()
    old_plaintext = dict(auth_proxy.CALLER_KEYS)
    old_hashes = dict(auth_proxy.CALLER_KEY_HASHES)
    try:
        auth_proxy.CALLER_KEYS.clear()
        auth_proxy.CALLER_KEY_HASHES.clear()
        assert host_store.set_consumer_keys(
            {"shared": {"status": "active", "keys": [{
                "sha256_prefix": digest[:12], "status": "active"}]}},
            key_digests={digest: "shared"})

        decision = auth_proxy._caller_auth(token)
        assert decision["ok"] is True
        assert decision["caller"] == "shared"
        assert decision["storage"] == "host_store"
    finally:
        auth_proxy.CALLER_KEYS.clear()
        auth_proxy.CALLER_KEYS.update(old_plaintext)
        auth_proxy.CALLER_KEY_HASHES.clear()
        auth_proxy.CALLER_KEY_HASHES.update(old_hashes)


def test_provider_relay_updates_are_per_provider_and_reads_follow_catalog(host_store_clean):
    store = host_store_clean
    store.set_provider_relay("first", "http://relay.internal/v1")
    store.set_provider_relay("second", "https://relay.example/v1")
    store.set_provider_relay("first", "https://new-relay.example/v1")
    assert store.get_provider_relays(["first"]) == {"first": "https://new-relay.example/v1"}
    store.set_provider_relay("first", None)
    store.set_provider_relay("first", None)
    assert store.get_provider_relays(["first", "second"]) == {"second": "https://relay.example/v1"}


@pytest.mark.asyncio
async def test_capacity_queue_is_bounded_and_releases(monkeypatch):
    monkeypatch.setattr(auth_proxy, "MAX_PENDING_REQUESTS", 1)
    monkeypatch.setattr(auth_proxy, "CAPACITY_QUEUE_TIMEOUT_S", 1.0)
    monkeypatch.setattr(auth_proxy, "_capacity", asyncio.Semaphore(1))
    monkeypatch.setattr(auth_proxy, "_active_requests", 0)
    monkeypatch.setattr(auth_proxy, "_pending_requests", 0)

    assert await auth_proxy._capacity_acquire() is True
    waiter = asyncio.create_task(auth_proxy._capacity_acquire())
    for _ in range(100):
        if auth_proxy._pending_requests == 1:
            break
        await asyncio.sleep(0)
    assert auth_proxy._pending_requests == 1
    assert await auth_proxy._capacity_acquire() is False

    auth_proxy._capacity_release()
    assert await waiter is True
    auth_proxy._capacity_release()
    assert auth_proxy._active_requests == 0
    assert auth_proxy._pending_requests == 0


def test_proxy_sheds_excess_load_with_retry_after(monkeypatch):
    monkeypatch.setattr(auth_proxy, "_caller_auth", lambda _token: {
        "ok": True, "caller": "load", "digest": "a" * 64,
        "meta": {"status": "active"}})
    monkeypatch.setattr(
        auth_proxy, "_route_allowed", lambda _caller, _route, _meta=None: True)
    monkeypatch.setattr(
        auth_proxy, "_rate_ok", lambda _caller, _meta=None: (True, True, 0.0))
    monkeypatch.setattr(auth_proxy, "_client", object())
    monkeypatch.setattr(auth_proxy, "MAX_PENDING_REQUESTS", 0)
    monkeypatch.setattr(auth_proxy, "CAPACITY_QUEUE_TIMEOUT_S", 2.0)
    monkeypatch.setattr(auth_proxy, "_capacity", asyncio.Semaphore(0))
    monkeypatch.setattr(auth_proxy, "_active_requests", 1)
    monkeypatch.setattr(auth_proxy, "_pending_requests", 0)

    response = TestClient(auth_proxy.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer load"},
        json={"model": "profile:default", "messages": []})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "2"
    assert response.json()["error"]["code"] == "router_overloaded"
    assert auth_proxy._active_requests == 1


def test_prometheus_surface_has_autoscaling_and_availability_metrics(monkeypatch):
    monkeypatch.setattr(auth_proxy, "_active_requests", 7)
    monkeypatch.setattr(auth_proxy, "_pending_requests", 3)
    monkeypatch.setattr(auth_proxy, "MAX_INFLIGHT_REQUESTS", 30)
    auth_proxy._metric_request(200, 250.0)
    auth_proxy._metric_reject("router_overloaded")
    auth_proxy._metric_store_error("rate_limit")

    response = TestClient(auth_proxy.app).get("/metrics")
    assert response.status_code == 200
    assert "llm_router_inflight_requests 7" in response.text
    assert "llm_router_pending_requests 3" in response.text
    assert "llm_router_capacity_requests 30" in response.text
    assert 'llm_router_requests_total{status="200"}' in response.text
    assert 'llm_router_rejections_total{reason="router_overloaded"}' in response.text
    assert 'llm_router_store_errors_total{kind="rate_limit"}' in response.text



def test_fact_policy_adds_no_database_queries(host_store_clean, monkeypatch):
    from test_provider_facts import NOW, chat, make_router_client

    client, attempts, _ = make_router_client()
    host_store_clean.upsert_provider_fact("comput3", "eu_hosted", True, NOW)
    client.app.state.provider_facts.refresh()

    checkouts = []

    def unavailable():
        checkouts.append(1)
        raise RuntimeError("database offline")

    monkeypatch.setattr(host_store, "_get_pool", unavailable)
    assert chat(client).status_code == 200
    baseline = len(checkouts)
    checkouts.clear()
    attempts.clear()
    assert chat(client, "fact:eu_hosted").status_code == 200
    assert len(checkouts) == baseline
    assert attempts == ["comput3"]


def _allowance(start, end, count):
    return {"period_start": start, "period_end": end, "request_count": count}


def test_shared_request_allowance_admits_only_available_slots(host_store_clean):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    store = host_store_clean
    now = int(time.time())
    allowance = _allowance(now - 1, now + 3600, 3)
    ready = Barrier(8)

    def reserve():
        ready.wait(timeout=5)
        return store.reserve_request_allowance(10, allowance)

    # Independent concurrent transactions contend on the shared tenant row.
    with ThreadPoolExecutor(8) as replicas:
        results = list(replicas.map(lambda _: reserve(), range(8)))
    reservations = {reservation for reservation, _ in results if reservation}
    assert len(reservations) == 3
    assert sum(reservation is not None for reservation, _ in results) == 3
    assert {end for _, end in results} == {now + 3600}
    assert store.reserve_request_allowance(11, allowance)[0] is not None


def test_request_allowance_expiry_and_rollover(host_store_clean, monkeypatch):
    from types import SimpleNamespace

    store = host_store_clean
    clock = SimpleNamespace(time=lambda: now)
    monkeypatch.setattr(store, "time", clock)
    now = 100
    allowance = _allowance(100, 8000, 1)
    abandoned, _ = store.reserve_request_allowance(10, allowance)
    assert abandoned
    now = 3699
    assert store.reserve_request_allowance(10, allowance) == (None, 8000)
    now = 3700
    replacement, _ = store.reserve_request_allowance(10, allowance)
    assert replacement and replacement != abandoned

    # A late release cannot return the replacement's slot.
    store.settle_request_allowance_async(abandoned, False)
    store.settle_request_allowance_async(replacement, True)
    assert store.reserve_request_allowance(10, allowance) == (None, 8000)

    now = 8000
    with pytest.raises(store.InvalidRequestAllowance):
        store.reserve_request_allowance(10, allowance)
    assert store.reserve_request_allowance(10, _allowance(8000, 9000, 1))[0]


@pytest.fixture
def tenant_allowance_route(host_store_clean, monkeypatch):
    import time

    import control_plane_client as cpc
    from test_auth_proxy_control_plane import _cp

    cpc.reset_for_tests()
    monkeypatch.setattr(cpc, "CONTROL_PLANE_URL", "http://cp.test")
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "local-test")
    monkeypatch.setattr(auth_proxy, "REQUEST_ALLOWANCES_ENABLED", True)
    cp = _cp(monkeypatch, [{"active": True, "consumer": "allowance-test", "tenant_id": 7}])
    now = int(time.time())
    metadata = {"request_allowance": _allowance(now - 1, now + 3600, 1)}
    get = cp.get

    async def get_with_allowance(url, **kwargs):
        response = await get(url, **kwargs)
        if "/routes/" in url:
            import httpx

            return httpx.Response(200, request=response.request, json={**response.json(), **metadata})
        return response

    monkeypatch.setattr(cp, "get", get_with_allowance)
    yield metadata
    cpc.reset_for_tests()


@pytest.mark.parametrize("failure", ["http", "chat_stream", "responses_stream"])
def test_failed_upstream_releases_tenant_slot(tenant_allowance_route, failure, monkeypatch):
    from test_auth_proxy_control_plane import _FakeUpstreamResp, _post_chat, _upstream

    upstream = _upstream(monkeypatch)
    client = TestClient(auth_proxy.app)
    if failure == "http":
        monkeypatch.setattr(_FakeUpstreamResp, "status_code", 502)
    else:
        async def stream(_self):
            if failure == "chat_stream":
                yield b'data: {"error":{"message":"failed"}}\n\n'
            else:
                yield b'data: {"type":"response.failed","response":{"status":"failed"}}\n\n'

        monkeypatch.setattr(_FakeUpstreamResp, "headers", {"content-type": "text/event-stream"})
        monkeypatch.setattr(_FakeUpstreamResp, "aiter_raw", stream, raising=False)

    assert _post_chat(client, "allowance-token").status_code == (502 if failure == "http" else 200)
    monkeypatch.setattr(_FakeUpstreamResp, "status_code", 200)
    monkeypatch.setattr(_FakeUpstreamResp, "headers", {"content-type": "application/json"})
    assert _post_chat(client, "allowance-token").status_code == 200
    refused = _post_chat(client, "allowance-token")
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "request_allowance_exhausted"
    assert 3590 <= int(refused.headers["Retry-After"]) <= 3600
    assert len(upstream.requests) == 2


@pytest.mark.parametrize("subject", [None, "k:42"])
def test_exhausted_tenant_allowance_is_reported_in_refusal_feed(
        tenant_allowance_route, subject, monkeypatch):
    from test_auth_proxy_control_plane import _post_chat, _upstream

    if subject is not None:
        tenant_allowance_route["key"] = {"id": 42, "subject": subject}
    _upstream(monkeypatch)
    client = TestClient(auth_proxy.app)
    assert _post_chat(client, "allowance-token").status_code == 200

    monkeypatch.setattr(host_store, "_SYNC_WRITES", False)
    hour = int(time.time()) // 3600 * 3600
    assert _post_chat(client, "allowance-token").status_code == 429
    host_store._refusal_q.join()

    # Read through a separate client using the shared store.
    reader = TestClient(auth_proxy.app)
    params = {"tenant_id": 7, "since_ts": hour, "until_ts": hour + 7200}
    headers = {"x-internal-secret": "local-test"}
    response = reader.get("/internal/usage/refusals", params=params, headers=headers)
    assert response.status_code == 200
    assert [(group["subject"], group["reason"], group["count"])
            for group in response.json()["groups"]] == [(subject, "request_allowance_exhausted", 1)]
    other_tenant = reader.get("/internal/usage/refusals", params={**params, "tenant_id": 8},
                              headers=headers)
    assert other_tenant.status_code == 200
    assert other_tenant.json()["groups"] == []


@pytest.mark.parametrize("enabled,include_allowance", [(False, True), (True, False), (True, True)])
def test_tenant_allowance_only_checks_store_when_configured(
        tenant_allowance_route, enabled, include_allowance, monkeypatch):
    from test_auth_proxy_control_plane import _post_chat, _upstream

    upstream = _upstream(monkeypatch)
    monkeypatch.setattr(auth_proxy, "REQUEST_ALLOWANCES_ENABLED", enabled)
    checks = []

    def store_unavailable(*args):
        checks.append(args)
        raise RuntimeError("database unavailable")

    if not include_allowance:
        tenant_allowance_route.clear()
    monkeypatch.setattr(host_store, "reserve_request_allowance", store_unavailable)
    response = _post_chat(TestClient(auth_proxy.app), "allowance-token")
    assert len(checks) == int(enabled and include_allowance)
    if enabled and include_allowance:
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "request_allowance_unavailable"
        assert not upstream.requests
    else:
        assert response.status_code == 200
        assert len(upstream.requests) == 1


@pytest.mark.asyncio
async def test_connection_check_limit_is_atomic_across_database_sessions(host_store_clean):
    reserve = host_store_clean.reserve_connection_check
    results = await asyncio.gather(*(asyncio.to_thread(reserve, 91, 12) for _ in range(4)))
    assert results.count(True) == 1
    assert await asyncio.to_thread(reserve, 92, 12)
    with host_store_clean._get_pool().connection() as conn:
        conn.execute("UPDATE connection_check_limits SET next_check = now() - interval '2 hours'")
    assert await asyncio.to_thread(reserve, 91, 12)
    with host_store_clean._get_pool().connection() as conn:
        assert conn.execute("SELECT tenant_id FROM connection_check_limits").fetchall() == [(91,)]
