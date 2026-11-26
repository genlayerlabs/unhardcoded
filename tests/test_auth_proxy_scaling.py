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


@pytest.mark.parametrize("method,path", [
    ("GET", "/v1/usage"),
    ("GET", "/v1/session/example"),
    ("POST", "/v1/chat/completions"),
])
@pytest.mark.parametrize("with_guidance", [False, True])
def test_caller_rate_limit_reuses_resolved_metadata(monkeypatch, method, path, with_guidance):
    resolver = auth_proxy.control_plane_client
    monkeypatch.setattr(resolver, "enabled", lambda: True)
    monkeypatch.setattr(auth_proxy, "_caller_auth", lambda token: {"ok": False})
    monkeypatch.setattr(auth_proxy, "_consumer_meta", lambda caller: {"status": "active"})
    monkeypatch.setattr(auth_proxy, "CALLER_KEYS", {})
    monkeypatch.setattr(auth_proxy, "CALLER_KEY_HASHES", {})
    monkeypatch.setattr(auth_proxy, "_client", object())
    monkeypatch.setattr(auth_proxy, "_record_reject", lambda **kwargs: None)
    payload = {"active": True, "consumer": "limited", "tenant_id": 7,
               "rate_per_min": 60, "burst": 10}
    expected = {"message": "caller rate limit exceeded", "type": "rate_limit_error",
                "code": "caller_rate_limit"}
    if with_guidance:
        payload.update(rate_limit_guidance="Reduce request frequency.",
                       rate_limit_help_url="https://example.com/limits")
        expected.update(guidance="Reduce request frequency.", help_url="https://example.com/limits")

    lookups = []

    async def resolve(digest):
        lookups.append(digest)
        return resolver._parse_resolved(payload)

    async def resolve_route(tenant_id, slug):
        return {"policy_ir": []}

    def consume(caller, rate, burst):
        assert (caller, rate, burst) == (f"{auth_proxy.CONTROL_PLANE_CALLER_PREFIX}limited", 60, 10)
        return False, True, 1.25

    monkeypatch.setattr(resolver, "resolve_key", resolve)
    monkeypatch.setattr(resolver, "resolve_route", resolve_route)
    monkeypatch.setattr(host_store, "consume_rate_token", consume)
    body = {"json": {"model": "route:production"}} if method == "POST" else {}
    response = TestClient(auth_proxy.app).request(
        method, path, headers={"Authorization": "Bearer limited"}, **body)

    assert response.status_code == 429
    assert response.headers["retry-after"] == "2"
    assert response.json() == {"error": expected}
    assert lookups == [hashlib.sha256(b"limited").hexdigest()]


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


def test_scoped_usage_report_against_one_million_calls(host_store_clean):
    store = host_store_clean
    start = int(time.time()) // 3600 * 3600 - 7200
    end = start + 3600
    with store._get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO calls(ts, caller, provider_id, status, cost_usd, routing_summary)
            SELECT %s + i %% 3600,
                   CASE WHEN i <= 800000 THEN 'report' ELSE 'other-' || (i %% 1000) END,
                   CASE WHEN i %% 10 <> 0 THEN 'provider-a' END,
                   CASE WHEN i %% 10 = 0 THEN 503 ELSE 200 END,
                   CASE WHEN i %% 10 <> 0 THEN 0.000125 END,
                   CASE WHEN i %% 10 = 0
                        THEN '{"project_id":"1","environment_id":"2","route":"route:default","no_candidates":true}'::jsonb
                        ELSE '{"project_id":"1","environment_id":"2"}'::jsonb END
            FROM generate_series(1, 1000000) AS rows(i)
            """, [start])
    store.insert_call({"ts": start, "caller": "report", "cost_usd": 12,
                       "decision_trace": {"project_id": 9, "environment_id": 2}})
    store.insert_call({"ts": start, "caller": "report", "cost_usd": 12,
                       "decision_trace": {"project_id": 1, "environment_id": 9}})
    store.insert_call({"ts": end, "caller": "report", "cost_usd": 12,
                       "decision_trace": {"project_id": 1, "environment_id": 2}})
    store.rollup_analytics(start, end)

    with store._get_pool().connection() as conn:
        conn.execute("ANALYZE scoped_usage_hours")
        plan = conn.execute(
            "EXPLAIN (ANALYZE, FORMAT JSON) SELECT * FROM scoped_usage_hours"
            " WHERE caller = %s AND project_id = %s AND environment_id = %s"
            " AND bucket_start >= %s AND bucket_start < %s",
            ["report", "1", "2", start, end]).fetchone()[0][0]["Plan"]
    assert "scoped_usage_hours_pkey" in str(plan)
    assert plan["Actual Rows"] == 3

    report = store.scoped_usage_report("report", 1, 2, start, end)
    assert (report["requests"], report["errors"], report["cost_usd"]) == (800000, 80000, 90.0)
    assert report["by_provider"] == [
        {"provider_id": "provider-a", "requests": 720000, "errors": 0, "cost_usd": 90.0},
        {"provider_id": "unknown", "requests": 80000, "errors": 80000, "cost_usd": 0.0},
    ]
    assert report["no_candidates_by_route"] == [{"route": "route:default", "no_candidates": 80000}]
    assert report["covered_hours"] == 1
    assert store.scoped_usage_report("report", 1, 2, start, end + 3600)["covered_hours"] == 1

    store.rollup_analytics(start, end)
    rerun = store.scoped_usage_report("report", 1, 2, start, end)
    assert {**rerun, "oldest_refresh_ts": report["oldest_refresh_ts"]} == report


def test_provider_daily_spend_serializes_and_rolls_over(host_store_clean, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from decimal import Decimal
    from threading import Barrier

    store = host_store_clean
    cap, amount = Decimal("0.30"), Decimal("0.10")
    monkeypatch.setattr(store, "_PROVIDER_SPEND_DAY", "DATE '2026-01-01'")
    ready = Barrier(8)

    def reserve():
        ready.wait(timeout=5)
        return store.reserve_provider_spend("shared", cap, amount)

    with ThreadPoolExecutor(8) as replicas:
        admitted = [day for day in replicas.map(lambda _: reserve(), range(8)) if day]
    assert len(admitted) == 3
    assert not store.provider_spend_state("shared", cap, amount)["can_admit"]
    assert store.reserve_provider_spend("other", cap, amount)
    store.settle_provider_spend_async("shared", admitted[0], -amount)
    assert store.reserve_provider_spend("shared", cap, amount)

    state = store.provider_spend_state("shared", cap, amount, override=(Decimal("0.40"),))
    assert state["can_admit"]
    assert state["resets_at"] == "2026-01-02T00:00:00Z"
    assert store.reserve_provider_spend("shared", cap, amount)
    assert not store.provider_spend_state("shared", cap, amount, override=(None,))["can_admit"]

    store.provider_spend_state("shared", cap, amount, override=(Decimal("0"),))
    monkeypatch.setattr(store, "_PROVIDER_SPEND_DAY", "DATE '2026-01-02'")
    assert store.reserve_provider_spend("shared", cap, amount)
    store.settle_provider_spend_async("shared", admitted[0], -amount)
    state = store.provider_spend_state("shared", cap, amount)
    assert state["override_usd"] is None
    assert Decimal(state["committed_usd"]) == amount


@pytest.fixture
def spend_host(monkeypatch):
    from llm_router_host import LLMRouterHost

    monkeypatch.setenv("SAAS_SHARED_PROVIDERS", "bedrock")
    host = LLMRouterHost(ROOT / "core/router.lua", ROOT / "tests/fixtures/managed.lua", env={
        "CUSTOM_CLOUD_SECRET": "fixture",
        "ROUTER_PROVIDER_DAILY_SPEND": (
            '{"bedrock":{"daily_usd":"0.10","max_attempt_usd":"0.10",'
            '"provider_ids":["bedrock","custom_cloud"]}}'),
    })
    host.init()
    return host


@pytest.mark.asyncio
async def test_provider_daily_spend_covers_concurrent_streams_and_tenant_calls(
        host_store_clean, spend_host, monkeypatch):
    from decimal import Decimal
    from saas_routes import compile_intent

    dispatches = []
    started, finish = asyncio.Event(), asyncio.Event()
    monkeypatch.setattr("llm_router_host._fold_route_outcome", lambda *args, **kwargs: None)

    async def stream(request):
        dispatches.append(request["provider_id"])
        assert request["provider_id"] == "bedrock"
        started.set()
        await finish.wait()
        return {"ok": True, "response": {"text": "streamed", "cost_reported": 0.025}}

    async def buffered(request):
        dispatches.append(request["provider_id"])
        return {"ok": True, "response": {"text": "fallback"}}

    spend_host.set_async_call_hook(buffered)
    tenants = [spend_host.for_tenant(tenant_id, {"CUSTOM_CLOUD_SECRET": "fixture"}, ["bedrock"])
               for tenant_id in (1, 2)]
    contract = {"policy_ir": compile_intent({
        "targets": ["bedrock|shared-model", "custom_cloud|shared-model"]}),
        "messages": [{"role": "user", "content": "test"}]}
    first = asyncio.create_task(tenants[0].execute_async(contract, call_override=stream))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        # Both aliases on the operator account share the open stream's charge.
        operator = await spend_host.execute_async(contract)
        assert operator["error"] == "exhausted: provider_spend_denied"
        assert dispatches == ["bedrock"]
        # The second tenant's own credential can still serve the fallback.
        second = await tenants[1].execute_async(contract)
        assert second["chosen"]["provider_id"] == "custom_cloud"
        assert any(event.get("reason") == "provider_spend_denied"
                   for event in second["trace"]["decision_path"])
        assert dispatches == ["bedrock", "custom_cloud"]
    finally:
        finish.set()
        result = await first
    assert result["ok"]
    state = host_store_clean.provider_spend_state("bedrock", *spend_host.provider_spend_limit("bedrock"))
    assert Decimal(state["committed_usd"]) == Decimal("0.025")


@pytest.mark.asyncio
async def test_provider_daily_spend_keeps_charge_on_cancel(host_store_clean, spend_host):
    from saas_routes import compile_intent

    started = asyncio.Event()

    async def stream(_request):
        started.set()
        await asyncio.Future()

    contract = {"policy_ir": compile_intent({"targets": ["bedrock|shared-model"]})}
    task = asyncio.create_task(spend_host.execute_async(contract, call_override=stream))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    denied = await spend_host.execute_async(contract, call_override=stream)
    assert denied["error"] == "exhausted: provider_spend_denied"


@pytest.mark.asyncio
async def test_provider_daily_spend_falls_back_when_store_is_unavailable(spend_host, monkeypatch):
    from saas_routes import compile_intent

    def unavailable(*args):
        raise RuntimeError("store offline")

    async def buffered(request):
        assert request["provider_id"] == "custom_cloud"
        return {"ok": True, "response": {"text": "fallback"}}

    monkeypatch.setattr(host_store, "reserve_provider_spend", unavailable)
    spend_host.set_async_call_hook(buffered)
    tenant = spend_host.for_tenant(1, {"CUSTOM_CLOUD_SECRET": "fixture"}, ["bedrock"])
    result = await tenant.execute_async({"policy_ir": compile_intent({
        "targets": ["bedrock|shared-model", "custom_cloud|shared-model"]})})
    assert result["chosen"]["provider_id"] == "custom_cloud"


def test_provider_daily_spend_operator_api(host_store_clean, spend_host, monkeypatch):
    import control_plane_client
    from shim import create_app

    monkeypatch.setattr(control_plane_client, "CONTROL_PLANE_INTERNAL_SECRET", "fixture")
    client = TestClient(create_app(spend_host))
    url = "/x/providers/bedrock/daily-spend"
    headers = {"x-internal-secret": "fixture"}
    assert client.get(url).status_code == 403
    assert client.put(url, json={"override_usd": 3}).status_code == 403
    assert client.get(url, headers=headers).json()["can_admit"]
    assert client.put(url, headers=headers, json={"override_usd": True}).status_code == 400
    paused = client.put(url, headers=headers, json={"override_usd": "0"})
    assert paused.status_code == 200
    assert not paused.json()["can_admit"]
    assert client.put(url, headers=headers, json={"override_usd": None}).json()["can_admit"]


def _run_budget_replica(tenant, replica, barrier, results):
    from concurrent.futures import ThreadPoolExecutor
    from decimal import Decimal

    def call(index):
        reservation = f"{replica}-{index}"
        rejected, ok, _ = host_store.reserve_scoped_budget(
            tenant, f"k:{replica + 1}", "2030-01", Decimal("20"),
            "common", Decimal("1.00"), Decimal("0.05"), reservation, 900)
        if not ok or rejected:
            return ok, rejected, False
        return ok, rejected, host_store.settle_subject_budget(reservation, 0.04) == (True, True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        barrier.wait(timeout=20)
        outcomes = list(pool.map(call, range(40)))
    host_store._get_pool().close()
    results.put(outcomes)


def test_shared_budget_actual_spend_under_parallel_replicas(host_store_clean):
    from multiprocessing import get_context

    ctx = get_context("spawn")
    results = ctx.Queue()
    barrier = ctx.Barrier(2)
    replicas = [ctx.Process(target=_run_budget_replica, args=(81, replica, barrier, results))
                for replica in range(2)]
    for replica in replicas:
        replica.start()
    try:
        outcomes = results.get(timeout=30) + results.get(timeout=30)
    finally:
        for replica in replicas:
            replica.join(timeout=5)
            if replica.is_alive():
                replica.terminate()
                replica.join()
    assert [replica.exitcode for replica in replicas] == [0, 0]
    assert all(ok and (settled or rejected == "s:common") for ok, rejected, settled in outcomes)
    count = sum(settled for _, _, settled in outcomes)
    keys = ["k:1", "k:2", "s:common"]
    by_replica = host_store_clean.subject_budgets(81, "2030-01", keys)
    assert 20 <= count <= 24
    assert by_replica[2]["spent_usd"] == pytest.approx(count * 0.04)
    assert by_replica[2]["spent_usd"] == pytest.approx(sum(row["spent_usd"] for row in by_replica[:2]))
    assert all(row["reserved_usd"] == 0 for row in by_replica)


def test_shared_budget_retains_ceiling_without_settlement(host_store_clean, monkeypatch):
    from decimal import Decimal

    store = host_store_clean
    now = time.time()
    monkeypatch.setattr(store.time, "time", lambda: now)
    args = (81, "g:1", "2030-01", Decimal("0.10"), "common", Decimal("0.05"), Decimal("0.05"))
    assert store.reserve_scoped_budget(*args, "lost", 1)[:2] == (None, True)
    now += 2
    assert store.reserve_scoped_budget(*args, "expired", 1)[:2] == ("s:common", True)
    now += 86400
    store._prune()
    assert store.settle_subject_budget("lost", 0) == (False, True)
    assert store.reserve_scoped_budget(*args, "pruned", 1)[:2] == ("s:common", True)
    budgets = store.subject_budgets(81, "2030-01", ["g:1", "s:common"])
    assert [row["spent_usd"] for row in budgets] == [0.05, 0.05]


def test_shared_allowance_reconciles_across_months(host_store_clean, monkeypatch):
    from decimal import Decimal

    store = host_store_clean
    ceiling = Decimal("0.040000000001")
    change = (0, ceiling * 3, Decimal("0"))
    store.shared_budget_account(81, "common", change=change)

    def reserve(period, reservation):
        return store.reserve_scoped_budget(
            81, "k:1", period, ceiling * 2, "common", Decimal("1"),
            ceiling, reservation, 900)[:2]

    assert reserve("2030-12", "known") == (None, True)
    assert reserve("2030-12", "unknown") == (None, True)
    assert reserve("2030-12", "key-full") == ("k:1", True)
    assert reserve("2031-01", "lost") == (None, True)
    assert store.settle_subject_budget("unknown", None) == (True, True)
    assert reserve("2031-01", "scope-full") == (store.SHARED_ALLOWANCE_EXHAUSTED, True)
    assert store.settle_subject_budget("known", Decimal("0.000000000001")) == (True, True)
    assert store.settle_subject_budget("known", Decimal("0")) == (False, True)

    after_grace = time.time() + 900 + 86401
    monkeypatch.setattr(store.time, "time", lambda: after_grace)
    store._prune()
    assert store.settle_subject_budget("lost", Decimal("0")) == (False, True)
    state = store.shared_budget_account(81, "common")
    assert Decimal(state["precharged_usd"]) == ceiling * 3
    assert Decimal(state["replaced_usd"]) == ceiling
    assert Decimal(state["settled_usd"]) == Decimal("0.000000000001")
    assert Decimal(state["committed_usd"]) == Decimal("0.080000000003")
    change = (state["revision"], ceiling * 3, Decimal("-0.000000000001"))
    assert Decimal(store.shared_budget_account(81, "common", change=change)["committed_usd"]) == ceiling * 2
    assert store.shared_budget_account(81, "common", change=change) is None
    assert reserve("2031-01", "after-correction") == (None, True)
    assert Decimal(store.shared_budget_account(82, "common")["committed_usd"]) == 0


def test_shared_allowance_changes_and_admission_serialize(host_store_clean):
    from concurrent.futures import ThreadPoolExecutor
    from decimal import Decimal
    from threading import Barrier

    store = host_store_clean
    store.shared_budget_account(81, "common", change=(0, Decimal("1"), Decimal("0")))
    ready = Barrier(3)

    def pause(correction):
        ready.wait(timeout=5)
        return store.shared_budget_account(81, "common",
                                           change=(1, Decimal("0"), correction))

    def reserve():
        ready.wait(timeout=5)
        return store.reserve_scoped_budget(
            81, "k:1", "2031-01", None, "common", Decimal("1"),
            Decimal("0.10"), "racing", 900)

    with ThreadPoolExecutor(3) as replicas:
        first = replicas.submit(pause, Decimal("0.01"))
        second = replicas.submit(pause, Decimal("0.02"))
        admission = replicas.submit(reserve)
        updates = [result for result in (first.result(), second.result()) if result]
        rejected, ok, _ = admission.result()
    assert len(updates) == 1
    assert ok and rejected in (None, store.SHARED_ALLOWANCE_EXHAUSTED)
    admitted = Decimal("0.10") if rejected is None else Decimal("0")
    state = store.shared_budget_account(81, "common")
    assert state["revision"] == 2
    assert Decimal(updates[0]["precharged_usd"]) == Decimal(state["precharged_usd"]) == admitted
    assert Decimal(state["committed_usd"]) == admitted + Decimal(state["correction_usd"])


def test_shared_budget_records_full_cost_if_published_ceiling_is_low(host_store_clean):
    from decimal import Decimal

    store = host_store_clean
    period = "2030-01"
    assert store.reserve_scoped_budget(
        81, "k:1", period, Decimal("1"), "common", Decimal("0.10"),
        Decimal("0.05"), "cost-overrun", 900)[:2] == (None, True)
    assert store.settle_subject_budget("cost-overrun", 0.07) == (True, True)
    assert store.settle_subject_budget("cost-overrun", 0.07) == (False, True)
    assert store.reserve_scoped_budget(
        81, "k:2", period, Decimal("1"), "common", Decimal("0.10"),
        Decimal("0.05"), "after-overrun", 900)[:2] == ("s:common", True)
    budgets = store.subject_budgets(81, period, ["k:1", "k:2", "s:common"])
    assert [row["spent_usd"] for row in budgets] == [0.07, 0.0, 0.07]


@pytest.mark.parametrize("settled", [True, False])
def test_shared_budget_handoff_blocks_until_previous_commitment_is_retired(
        host_store_clean, monkeypatch, settled):
    from decimal import Decimal

    store = host_store_clean

    def reserve(period, reservation):
        return store.reserve_scoped_budget(
            86, "k:1", period, None, "common", Decimal("0.10"),
            Decimal("0.05"), reservation, 900)[:2]

    assert reserve("2030-12", "old") == (None, True)
    assert reserve("2031-01", "new") == ("s:common", True)
    if settled:
        assert store.settle_subject_budget("old", Decimal("0.03")) == (True, True)
    else:
        expired = time.time() + 901
        monkeypatch.setattr(store.time, "time", lambda: expired)
    assert reserve("2031-01", "new") == (None, True)
    previous = store.subject_budgets(86, "2030-12", ["s:common"])[0]
    assert previous["spent_usd"] == (0.03 if settled else 0.05)
