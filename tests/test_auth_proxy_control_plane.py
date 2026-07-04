"""Ingress <-> external control plane integration (auth fallback + /internal/*).

The control plane's HTTP side is a fake client on control_plane_client._client;
the upstream router is a fake on auth_proxy._client. Store-backed tests use the
shared Postgres fixture (host_store_clean).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# auth_proxy reads the caller-key env at import; this module can be the first
# in the session to import it, so mirror the suite-wide fixture env here or the
# later-collected dashboard tests would see an empty CALLER_KEYS map.
os.environ.setdefault("CALLER_KEYS_JSON", '{"internal":"default"}')
os.environ.setdefault("CALLER_KEYS_SHA256_JSON", "{}")

import auth_proxy  # noqa: E402
import control_plane_client as cpc  # noqa: E402
import host_store  # noqa: E402

from conftest import require_host_store  # noqa: E402


class _FakeCPResp:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeCPClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    async def get(self, url, params=None, headers=None):
        self.calls += 1
        item = self.payloads.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeCPResp(200, item)


class _FakeUpstreamResp:
    status_code = 200
    headers = {"content-type": "application/json"}

    async def aread(self):
        return b'{"ok": true}'

    async def aclose(self):
        pass


class _FakeUpstream:
    """Captures the headers the proxy forwards to the router."""

    def __init__(self):
        self.requests: list[dict] = []

    def build_request(self, method, url, content=None, headers=None):
        self.requests.append({"method": method, "url": url, "headers": headers or {}})
        return object()

    async def send(self, req, stream=True):
        return _FakeUpstreamResp()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    cpc.reset_for_tests()
    auth_proxy._windows.clear()
    monkeypatch.setattr(cpc, "CONTROL_PLANE_URL", "http://cp.test")
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "s3cret")
    yield
    cpc.reset_for_tests()
    auth_proxy._windows.clear()


def _cp(monkeypatch, payloads) -> _FakeCPClient:
    fake = _FakeCPClient(payloads)
    monkeypatch.setattr(cpc, "_client", fake)
    return fake


def _upstream(monkeypatch) -> _FakeUpstream:
    fake = _FakeUpstream()
    monkeypatch.setattr(auth_proxy, "_client", fake)
    return fake


def _post_chat(client, token: str, extra_headers: dict | None = None):
    headers = {"Authorization": f"Bearer {token}"}
    headers.update(extra_headers or {})
    return client.post("/v1/chat/completions", headers=headers,
                       json={"model": "profile:default", "messages": []})


# ---- key resolution ----------------------------------------------------------

def test_feature_off_unknown_key_401_without_cp_call(monkeypatch):
    require_host_store()
    monkeypatch.setattr(cpc, "CONTROL_PLANE_URL", "")
    fake_cp = _cp(monkeypatch, [])
    _upstream(monkeypatch)
    r = _post_chat(TestClient(auth_proxy.app), "tok-unknown")
    assert r.status_code == 401
    assert fake_cp.calls == 0


def test_cp_resolved_key_proxies_with_caller_and_tenant_headers(monkeypatch):
    require_host_store()
    _cp(monkeypatch, [{"active": True, "consumer": "acme", "tenant_id": 7,
                       "rate_per_min": 600, "burst": 200}])
    upstream = _upstream(monkeypatch)
    r = _post_chat(TestClient(auth_proxy.app), "tok-tenant",
                   extra_headers={"x-llm-router-tenant": "999",   # smuggle attempt
                                  "x-internal-secret": "leak"})
    assert r.status_code == 200
    fwd = upstream.requests[0]["headers"]
    assert fwd["x-llm-router-caller"] == "acme"
    assert fwd["x-llm-router-tenant"] == "7"          # authed value, not the smuggled 999
    assert "x-internal-secret" not in fwd


def test_second_request_served_from_resolve_cache(monkeypatch):
    require_host_store()
    fake_cp = _cp(monkeypatch, [{"active": True, "consumer": "acme", "tenant_id": 7}])
    _upstream(monkeypatch)
    client = TestClient(auth_proxy.app)
    assert _post_chat(client, "tok-cache").status_code == 200
    assert _post_chat(client, "tok-cache").status_code == 200
    assert fake_cp.calls == 1


def test_cp_plan_rate_limits_enforced(monkeypatch):
    require_host_store()
    _cp(monkeypatch, [{"active": True, "consumer": "tiny-plan", "tenant_id": 3,
                       "rate_per_min": 1, "burst": 1}])
    _upstream(monkeypatch)
    client = TestClient(auth_proxy.app)
    assert _post_chat(client, "tok-limited").status_code == 200
    r = _post_chat(client, "tok-limited")
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "caller_rate_limit"


def test_inactive_resolve_is_401(monkeypatch):
    require_host_store()
    _cp(monkeypatch, [{"active": False}])
    _upstream(monkeypatch)
    assert _post_chat(TestClient(auth_proxy.app), "tok-revoked").status_code == 401


def test_local_plaintext_key_never_consults_cp(monkeypatch):
    require_host_store()
    fake_cp = _cp(monkeypatch, [])
    _upstream(monkeypatch)
    monkeypatch.setattr(auth_proxy, "CALLER_KEYS", {"tok-local": "operator-app"})
    r = _post_chat(TestClient(auth_proxy.app), "tok-local")
    assert r.status_code == 200
    assert fake_cp.calls == 0


def test_locally_revoked_hash_key_never_falls_through_to_cp(monkeypatch):
    require_host_store()
    fake_cp = _cp(monkeypatch, [])
    _upstream(monkeypatch)
    digest = hashlib.sha256(b"tok-revoked-local").hexdigest()
    monkeypatch.setattr(auth_proxy, "CALLER_KEY_HASHES", {digest: "operator-app"})
    host_store.set_consumer_keys({"operator-app": {
        "status": "active", "keys": [{"sha256_prefix": digest[:12], "status": "revoked"}]}})
    r = _post_chat(TestClient(auth_proxy.app), "tok-revoked-local")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "caller_key_revoked"
    assert fake_cp.calls == 0


def test_operator_kill_switch_blocks_cp_slug(monkeypatch):
    require_host_store()
    _cp(monkeypatch, [{"active": True, "consumer": "banned-tenant", "tenant_id": 4}])
    _upstream(monkeypatch)
    host_store.set_consumer_keys({"banned-tenant": {"status": "inactive"}})
    r = _post_chat(TestClient(auth_proxy.app), "tok-banned")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "caller_inactive"


def test_cp_caller_lands_in_the_ledger_under_the_tenant_slug(monkeypatch):
    require_host_store()
    _cp(monkeypatch, [{"active": True, "consumer": "acme", "tenant_id": 7}])
    _upstream(monkeypatch)
    assert _post_chat(TestClient(auth_proxy.app), "tok-ledger").status_code == 200
    host_store._write_q.join()
    rows = host_store.recent_calls(caller="acme")
    assert len(rows) == 1
    assert rows[0]["caller"] == "acme"


# ---- /internal/usage surface ---------------------------------------------------

def _seed_calls():
    now = int(time.time())
    base = {"session": "s", "key_sha256": "c" * 64, "provider": "openrouter",
            "model_family": "fam", "served_model_id": "m", "requested_model": "profile:default",
            "latency_ms": 100.0}
    host_store.insert_call({**base, "ts": now - 10, "caller": "acme", "status": 200,
                            "tokens_in": 70, "tokens_out": 30, "tokens_total": 100,
                            "tokens_cached": 25, "cost_usd": 0.01})
    host_store.insert_call({**base, "ts": now - 5, "caller": "acme", "status": 500,
                            "tokens_in": 10, "tokens_out": 0, "tokens_total": 10,
                            "cost_usd": 0.0})
    host_store.insert_call({**base, "ts": now - 5, "caller": "other", "status": 200,
                            "tokens_in": 1000, "tokens_out": 1000, "tokens_total": 2000,
                            "cost_usd": 9.99})
    return now


def test_internal_usage_hidden_without_secret(monkeypatch):
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "")
    r = TestClient(auth_proxy.app).get("/internal/usage", params={"caller": "acme"})
    assert r.status_code == 404


def test_internal_usage_wrong_secret_403():
    r = TestClient(auth_proxy.app).get("/internal/usage", params={"caller": "acme"},
                                       headers={"x-internal-secret": "wrong"})
    assert r.status_code == 403


def test_internal_usage_totals_and_daily_buckets():
    require_host_store()
    now = _seed_calls()
    client = TestClient(auth_proxy.app)
    r = client.get("/internal/usage",
                   params={"caller": "acme", "since_ts": now - 3600, "bucket": "day"},
                   headers={"x-internal-secret": "s3cret"})
    assert r.status_code == 200
    data = r.json()
    assert data["caller"] == "acme"
    assert data["runs"] == 2 and data["errors"] == 1
    assert data["tokens_in"] == 80 and data["tokens_out"] == 30
    assert data["tokens_cached"] == 25 and data["tokens_total"] == 110
    assert data["cost_usd"] == pytest.approx(0.01)
    assert data["window"]["since_ts"] == now - 3600
    assert sum(b["runs"] for b in data["buckets"]) == 2

    # missing caller -> 400, and no cross-tenant bleed
    assert client.get("/internal/usage",
                      headers={"x-internal-secret": "s3cret"}).status_code == 400


def test_internal_usage_recent_scopes_to_caller():
    require_host_store()
    _seed_calls()
    r = TestClient(auth_proxy.app).get("/internal/usage/recent",
                                       params={"caller": "acme", "limit": 10},
                                       headers={"x-internal-secret": "s3cret"})
    assert r.status_code == 200
    calls = r.json()["calls"]
    assert len(calls) == 2
    assert {c["status"] for c in calls} == {200, 500}
    assert all(c["key_sha256_prefix"] == "c" * 12 for c in calls)
    assert all("consumer_sha" not in c for c in calls)   # only the prefix leaves
    assert calls[0]["latency_ms"] == 100.0


def test_internal_usage_is_not_proxied_upstream(monkeypatch):
    """Regression: the /internal router must match BEFORE the catch-all proxy."""
    upstream = _upstream(monkeypatch)
    r = TestClient(auth_proxy.app).get("/internal/usage",
                                       headers={"x-internal-secret": "s3cret"})
    assert r.status_code == 400          # caller_required, answered locally
    assert upstream.requests == []
