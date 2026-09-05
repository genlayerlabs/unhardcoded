"""Cross-replica admission and per-pod overload behavior."""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
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
