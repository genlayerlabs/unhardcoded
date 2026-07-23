from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# auth_proxy parses caller maps at import time; keep tests isolated from any
# operator shell env or .env loader that may contain non-JSON placeholders.
os.environ["CALLER_KEYS_JSON"] = '{"internal":"default"}'
os.environ["CALLER_KEYS_SHA256_JSON"] = "{}"
os.environ["DASHBOARD_TRUSTED_USER_HEADER"] = ""

import auth_proxy  # noqa: E402
import host_store  # noqa: E402


@pytest.mark.asyncio
async def test_dashboard_stats_keeps_event_loop_responsive_during_slow_store_query(monkeypatch):
    """A slow synchronous stats query must not take down health/proxy traffic."""
    _install_test_state(monkeypatch)
    monkeypatch.setattr(
        auth_proxy, "_require_dashboard_context",
        lambda _request: {"viewer": "admin", "role": "admin"})

    started = threading.Event()
    release = threading.Event()

    def slow_snapshot(**_kwargs):
        started.set()
        assert release.wait(timeout=2)
        return {}

    monkeypatch.setattr(auth_proxy, "_stats_snapshot", slow_snapshot)
    request = Request({
        "type": "http", "method": "GET", "path": "/dashboard/api/stats",
        "query_string": b"timeframe=all", "headers": [],
        "client": ("test", 1), "server": ("test", 80), "scheme": "http",
    })

    # A timer releases the fake DB query even on the broken implementation,
    # preventing a deadlock while making event-loop blocking measurable.
    timer = threading.Timer(0.5, release.set)
    timer.start()
    before = time.monotonic()
    task = asyncio.create_task(auth_proxy.dashboard_stats(request))
    await asyncio.to_thread(started.wait, 1)
    await asyncio.sleep(0.05)
    elapsed = time.monotonic() - before
    release.set()
    await task
    timer.cancel()

    assert elapsed < 0.25, "synchronous stats work blocked the event loop"


def _use_db(monkeypatch, tmp_path):
    """A clean host store (truncated) for the test — isolation against the shared
    Postgres. (monkeypatch/tmp_path kept for call-site compatibility.)"""
    host_store.reset()
    host_store.truncate_all_for_tests()


def _set_issued(records):
    host_store.set_consumer_keys(records)


def _issued_data():
    return host_store.get_consumer_keys()[0]


def _issued_text():
    return json.dumps(host_store.get_consumer_keys()[0])


class _FakeUpstreamResponse:
    status_code = 200

    def json(self):
        return {"ok": True, "service": "router"}


class _FakeAsyncClient:
    async def get(self, url: str, timeout: float):
        assert url.endswith("/healthz")
        assert timeout == 5.0
        return _FakeUpstreamResponse()


def _fake_policy_catalog() -> dict:
    return {
        "providers": [{"name": "openrouter", "auth": "configured"}],
        "models": [{"name": "medium", "quality": 0.82}],
        "profiles": [{"name": "medium", "models": []}],
        "retry_policies": {"balanced": {"max_attempts": 2}},
        "source": "/tmp/config.internal.lua",
        "generated_at": 123,
    }


def _install_test_state(monkeypatch):
    monkeypatch.setattr(auth_proxy, "_client", _FakeAsyncClient())
    monkeypatch.setattr(auth_proxy, "_policy_catalog_snapshot", _fake_policy_catalog)
    stats_lock = getattr(auth_proxy, "_stats_lock")
    stats = getattr(auth_proxy, "_stats")
    with stats_lock:
        stats["total_requests"] = 3
        stats["total_errors"] = 1
        stats["total_rejects"] = 0
        stats["total_tokens_in"] = 11
        stats["total_tokens_out"] = 13
        stats["total_tokens"] = 24
        stats["synthetic_route_health"] = {
            "profile:default": {
                "route": "profile:default",
                "state": "ok",
                "source": "synthetic_probe",
                "status": 200,
                "provider": "openrouter",
                "model_family": "qwen3-235b-a22b",
                "served_model_id": "qwen/qwen3-235b-a22b-2507",
                "latency_ms": 42,
                "last_seen": 123,
                "note": None,
            }
        }


def test_route_allowed_accepts_all_alias(monkeypatch):
    monkeypatch.setattr(auth_proxy, "_consumer_meta", lambda caller: {"allowed_routes": ["all"]})
    assert auth_proxy._route_allowed("admin2", "profile:edge") is True
    assert auth_proxy._route_allowed("admin2", "openai/gpt-5.5") is True


def test_dashboard_health_summary_uses_synthetic_route_health_when_no_chat_requests():
    summary = auth_proxy._health_summary(
        [],
        [
            {"route": "profile:edge", "state": "ok", "source": "synthetic_probe", "status": 200},
            {"route": "profile:medium", "state": "ok", "source": "synthetic_probe", "status": 200},
        ],
    )
    assert summary["request_count"] == 0
    assert summary["success_rate"] is None
    assert summary["route_failures"] == 0
    assert summary["state"] == "ok"


def test_dashboard_full_requires_auth():
    client = TestClient(auth_proxy.app)
    resp = client.get("/dashboard/api/full")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "dashboard_auth"


def _dashboard_client(monkeypatch, user: str = "tester"):
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET", "test-dashboard-session-secret")
    client = TestClient(auth_proxy.app)
    client.cookies.set(auth_proxy.DASHBOARD_COOKIE_NAME, auth_proxy._make_dashboard_session(user))
    return client


def test_dashboard_full_rejects_spoofed_header_and_consumer_token():
    client = TestClient(auth_proxy.app)
    assert client.get("/dashboard/api/full", headers={"x-dashboard-user": "tester"}).status_code == 401
    assert client.get("/dashboard/api/full", headers={"Authorization": "Bearer internal"}).status_code == 401


def test_dashboard_full_returns_complete_sanitized_snapshot(monkeypatch):
    _install_test_state(monkeypatch)
    client = _dashboard_client(monkeypatch)

    resp = client.get("/dashboard/api/full?timeframe=runtime")

    assert resp.status_code == 200
    data = resp.json()
    assert data["schema_version"] == 1
    assert data["kind"] == "router_dashboard_full_snapshot"
    assert data["viewer"] == "dashboard:tester"
    assert set(data["sections"]) == {"overview", "consumers", "routing", "policies", "activity", "logins", "provider_keys"}
    assert data["sections"]["overview"]["upstream"] == {"status": 200, "health": {"ok": True, "service": "router"}}
    assert data["sections"]["overview"]["totals"]["requests"] == 3
    assert data["sections"]["policies"]["providers"][0]["auth"] == "configured"
    assert any(row["route"] == "profile:default" and row["state"] == "ok" for row in data["sections"]["routing"]["route_health"])
    assert data["raw"]["stats"]["viewer"] == "dashboard:tester"
    assert data["raw"]["policies"] == data["sections"]["policies"]
    assert data["errors"] == []
    assert data["security"] == {
        "sanitized": True,
        "raw_api_keys_exposed": False,
        "provider_credentials_exposed": False,
    }


def test_dashboard_key_creation_persists_hash_metadata_not_raw_key(monkeypatch, tmp_path):
    monkeypatch.setattr(auth_proxy, "DASHBOARD_KEY_ENV_PATH", str(tmp_path / ".env.secrets"))
    _use_db(monkeypatch, tmp_path)
    original_hashes = dict(auth_proxy.CALLER_KEY_HASHES)
    auth_proxy.CALLER_KEY_HASHES.clear()
    try:
        client = _dashboard_client(monkeypatch)
        resp = client.post("/dashboard/api/keys", json={"consumer": "crm"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["api_key"].startswith(auth_proxy.DASHBOARD_KEY_PREFIX + "_")
        env_text = (tmp_path / ".env.secrets").read_text()
        issued_text = _issued_text()
        assert body["api_key"] not in env_text
        assert body["api_key"] not in issued_text
        assert body["sha256_prefix"] in issued_text
        assert "CALLER_KEYS_SHA256_JSON" in env_text
    finally:
        auth_proxy.CALLER_KEY_HASHES.clear()
        auth_proxy.CALLER_KEY_HASHES.update(original_hashes)


def test_dashboard_key_generation_ui_has_copy_key_and_handoff_blurb(monkeypatch, tmp_path):
    client = _dashboard_client(monkeypatch)
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    html = resp.text
    assert "newKeyValue" in html
    assert "copyKey" in html
    assert "Copy key" in html
    assert "newKeyHandoffValue" in html
    assert "copyKeyHandoff" in html
    assert "Copy setup blurb" in html
    assert "buildKeyHandoff" in html
    # Two-step new-consumer-key dialog and the scoped keys drawer.
    assert "newKeyConsumer" in html
    assert "Create and generate key" in html
    assert "Shown once — copy it now" in html
    assert "keysList" in html
    assert "Generate new key" in html
    # The handoff base URL defaults to loopback and is configurable via PUBLIC_BASE_URL.
    assert "http://127.0.0.1:8080/v1" in html
    assert "router.ygr.ai" not in html
    assert "profile:default" in html
    assert "/v1/usage?window=24h&limit=50" in html
    assert "background-repeat:no-repeat" in html
    assert "background-attachment:fixed" in html
    assert "background-size:100vw 100vh" in html
    if shutil.which("node"):
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        script_path = tmp_path / "dashboard.js"
        script_path.write_text(script)
        subprocess.run(["node", "--check", str(script_path)], check=True)

    monkeypatch.setenv("PUBLIC_BASE_URL", "https://api.example.com/v1")
    overridden = client.get("/dashboard").text
    assert "https://api.example.com/v1" in overridden
    assert "http://127.0.0.1:8080/v1" not in overridden


def test_dashboard_reveal_keys_requires_admin_and_consumer_scoped(monkeypatch):
    _install_test_state(monkeypatch)
    original_plaintext = dict(auth_proxy.CALLER_KEYS)
    original_hashes = dict(auth_proxy.CALLER_KEY_HASHES)
    auth_proxy.CALLER_KEYS.clear()
    auth_proxy.CALLER_KEYS.update({"raw-crm-token": "crm", "raw-wingston-token": "wingston"})
    auth_proxy.CALLER_KEY_HASHES.clear()
    auth_proxy.CALLER_KEY_HASHES.update({"a" * 64: "crm"})
    try:
        unauth = TestClient(auth_proxy.app)
        assert unauth.get("/dashboard/api/keys/reveal?consumer=crm").status_code == 401

        # any admin works now — SSO users aren't named 'admin', and there are
        # no per-name tiers (the gate is the admin role, tested elsewhere).
        assert _dashboard_client(monkeypatch, user="tester").get(
            "/dashboard/api/keys/reveal?consumer=crm").status_code == 200

        admin = _dashboard_client(monkeypatch, user="admin")
        resp = admin.get("/dashboard/api/keys/reveal?consumer=crm")
        assert resp.status_code == 200
        body = resp.json()
        assert body["consumer"] == "crm"
        assert body["hash_only_count"] == 1
        assert [row["api_key"] for row in body["keys"]] == ["raw-crm-token"]
        assert "raw-wingston-token" not in resp.text

        full = admin.get("/dashboard/api/full")
        assert full.status_code == 200
        assert "raw-crm-token" not in full.text
        assert full.json()["security"]["raw_api_keys_exposed"] is False
    finally:
        auth_proxy.CALLER_KEYS.clear()
        auth_proxy.CALLER_KEYS.update(original_plaintext)
        auth_proxy.CALLER_KEY_HASHES.clear()
        auth_proxy.CALLER_KEY_HASHES.update(original_hashes)


def test_dashboard_reveal_keys_reports_hash_only_unrecoverable(monkeypatch):
    original_plaintext = dict(auth_proxy.CALLER_KEYS)
    original_hashes = dict(auth_proxy.CALLER_KEY_HASHES)
    auth_proxy.CALLER_KEYS.clear()
    auth_proxy.CALLER_KEY_HASHES.clear()
    auth_proxy.CALLER_KEY_HASHES.update({"b" * 64: "crm"})
    try:
        client = _dashboard_client(monkeypatch, user="admin")
        resp = client.get("/dashboard/api/keys/reveal?consumer=crm")
        assert resp.status_code == 200
        body = resp.json()
        assert body["keys"] == []
        assert body["hash_only_count"] == 1
        assert "Hash-only keys cannot be revealed" in body["message"]
    finally:
        auth_proxy.CALLER_KEYS.clear()
        auth_proxy.CALLER_KEYS.update(original_plaintext)
        auth_proxy.CALLER_KEY_HASHES.clear()
        auth_proxy.CALLER_KEY_HASHES.update(original_hashes)


def test_dashboard_list_keys_requires_admin(monkeypatch, tmp_path):
    _use_db(monkeypatch, tmp_path)
    unauth = TestClient(auth_proxy.app)
    assert unauth.get("/dashboard/api/keys/list?consumer=crm").status_code == 401


def test_dashboard_list_keys_returns_per_key_metadata_no_raw(monkeypatch, tmp_path):
    digest = hashlib.sha256("crm-token".encode()).hexdigest()
    old = hashlib.sha256("crm-old".encode()).hexdigest()
    _, _, original_plaintext, original_hashes = _with_consumer_auth(
        monkeypatch,
        tmp_path,
        issued={
            "status": "active",
            "allowed_routes": ["profile:edge"],
            "keys": [
                {"sha256_prefix": digest[:12], "status": "active", "created_at": 100},
                {"sha256_prefix": old[:12], "status": "active", "created_at": 50,
                 "expires_at": 200, "replaced_at": 90},
            ],
        },
    )
    try:
        client = _dashboard_client(monkeypatch)
        resp = client.get("/dashboard/api/keys/list?consumer=crm")
        assert resp.status_code == 200
        body = resp.json()
        assert body["consumer"] == "crm"
        assert body["status"] == "active"
        assert body["allowed_routes"] == ["profile:edge"]
        prefixes = {k["sha256_prefix"] for k in body["keys"]}
        assert digest[:12] in prefixes
        assert old[:12] in prefixes
        rotated = next(k for k in body["keys"] if k["sha256_prefix"] == old[:12])
        assert rotated["expires_at"] == 200
        assert rotated["replaced_at"] == 90
        # No raw key material of any shape is returned.
        assert "api_key" not in resp.text
        assert "crm-token" not in resp.text
        # Hash-only keys are not recoverable.
        assert all(k["recoverable"] is False for k in body["keys"])
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_dashboard_list_keys_flags_recoverable_plaintext(monkeypatch, tmp_path):
    _use_db(monkeypatch, tmp_path)
    monkeypatch.setattr(auth_proxy, "DASHBOARD_KEY_ENV_PATH", str(tmp_path / ".env.secrets"))
    token = "legacy-crm-token"
    digest = hashlib.sha256(token.encode()).hexdigest()
    original_plaintext = dict(auth_proxy.CALLER_KEYS)
    original_hashes = dict(auth_proxy.CALLER_KEY_HASHES)
    auth_proxy.CALLER_KEYS.clear()
    auth_proxy.CALLER_KEYS.update({token: "crm"})
    auth_proxy.CALLER_KEY_HASHES.clear()
    try:
        client = _dashboard_client(monkeypatch)
        resp = client.get("/dashboard/api/keys/list?consumer=crm")
        assert resp.status_code == 200
        body = resp.json()
        row = next(k for k in body["keys"] if k["sha256_prefix"] == digest[:12])
        assert row["recoverable"] is True
        assert token not in resp.text
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_dashboard_list_keys_validates_consumer(monkeypatch, tmp_path):
    _use_db(monkeypatch, tmp_path)
    client = _dashboard_client(monkeypatch)
    resp = client.get("/dashboard/api/keys/list?consumer=")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_consumer"


def test_dashboard_evaluate_alias_matches_full_shape(monkeypatch):
    _install_test_state(monkeypatch)
    client = _dashboard_client(monkeypatch)

    resp = client.get("/dashboard/api/evaluate")

    assert resp.status_code == 200
    data = resp.json()
    assert data["kind"] == "router_dashboard_full_snapshot"
    assert "stats" in data["raw"]
    assert "policies" in data["raw"]
    assert data["sections"]["policies"]["profiles"][0]["name"] == "medium"


def test_policy_catalog_uses_policy_files_and_router_rank():
    catalog = auth_proxy._policy_catalog_snapshot()
    profiles = {row["name"]: row for row in catalog["profiles"]}

    # No tiers: the catalog has only the declarative `default` fallback policy
    # (callers send their own per-call policy_ir). It is declarative, not a
    # closure file, so there are no policy_files.
    assert list(profiles) == ["default"]
    assert catalog["source"].endswith("config.live.lua")
    assert catalog["metrics_source"].endswith("metrics.live.lua")
    assert catalog["policy_files"] == []
    assert profiles["default"]["candidate_count"] > 0
    assert "router.rank" in profiles["default"]["selection_note"]


class _FakeRouteResponse:
    status_code = 200
    content = b'{"choices":[{"message":{"content":"pong"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2},"x_router":{"provider":"openai","model_family":"gpt-5.5-codex","served_model_id":"gpt-5.5"}}'
    headers = {"content-type": "application/json"}

    def json(self):
        return json.loads(self.content.decode())

    async def aread(self):
        return self.content

    async def aclose(self):
        return None


class _FakeRouteClient:
    def build_request(self, method, url, content=None, headers=None):
        return (method, url)

    async def send(self, request, stream=True):
        return _FakeRouteResponse()

    async def get(self, url: str, timeout: float = 5.0, params=None):
        return _FakeUpstreamResponse()


def _with_consumer_auth(monkeypatch, tmp_path, token="crm-token", consumer="crm", issued=None):
    monkeypatch.setattr(auth_proxy, "_client", _FakeRouteClient())
    monkeypatch.setattr(auth_proxy, "DASHBOARD_KEY_ENV_PATH", str(tmp_path / ".env.secrets"))
    _use_db(monkeypatch, tmp_path)
    digest = hashlib.sha256(token.encode()).hexdigest()
    original_plaintext = dict(auth_proxy.CALLER_KEYS)
    original_hashes = dict(auth_proxy.CALLER_KEY_HASHES)
    auth_proxy.CALLER_KEYS.clear()
    auth_proxy.CALLER_KEY_HASHES.clear()
    auth_proxy.CALLER_KEY_HASHES.update({digest: consumer})
    auth_proxy._windows.clear()
    if issued is not None:
        _set_issued({consumer: issued})
    return token, digest, original_plaintext, original_hashes


def _restore_auth_maps(original_plaintext, original_hashes):
    auth_proxy.CALLER_KEYS.clear()
    auth_proxy.CALLER_KEYS.update(original_plaintext)
    auth_proxy.CALLER_KEY_HASHES.clear()
    auth_proxy.CALLER_KEY_HASHES.update(original_hashes)
    auth_proxy._windows.clear()


class _CannedResp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload)


class _RecordingRouteClient:
    """Records the forwarded (method, url, json, params) and returns a canned
    router response — to assert the ingress proxies the consumer validation
    endpoints to the router."""
    def __init__(self, payload):
        self.calls = []
        self._payload = payload

    async def post(self, url, json=None, params=None, timeout=None):
        self.calls.append(("POST", url, json, dict(params or {})))
        return _CannedResp(self._payload)

    async def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url, None, dict(params or {})))
        return _CannedResp(self._payload)


def test_consumer_validation_endpoints_proxy_to_router(monkeypatch, tmp_path):
    # The SKILL.md dry-run surface (/x/policy/normalize, /x/rank, /x/fields, …)
    # must be reachable by a consumer key on the ingress — not just internally.
    digest = hashlib.sha256("crm-token".encode()).hexdigest()
    token, digest, op, oh = _with_consumer_auth(
        monkeypatch, tmp_path,
        issued={"status": "active", "allowed_routes": ["all"],
                "keys": [{"sha256_prefix": digest[:12], "status": "active"}]})
    rec = _RecordingRouteClient({"policy_ir": ["policy"], "fingerprint": "abc",
                                 "version": "sigma-pol/v2"})
    monkeypatch.setattr(auth_proxy, "_client", rec)
    try:
        client = TestClient(auth_proxy.app)
        h = {"Authorization": f"Bearer {token}"}
        # no key -> 401, never reaches the router
        anon = client.post("/x/policy/normalize", json={"policy_ir": ["policy"]})
        assert anon.status_code == 401 and not rec.calls
        # POST is forwarded to the router's matching path, body and all
        r = client.post("/x/policy/normalize", headers=h, json={"policy_ir": ["policy"]})
        assert r.status_code == 200 and r.json()["fingerprint"] == "abc"
        assert any(c[0] == "POST" and c[1].endswith("/x/policy/normalize")
                   and c[2] == {"policy_ir": ["policy"]} for c in rec.calls)
        # GET /x/fields is forwarded
        assert client.get("/x/fields", headers=h).status_code == 200
        assert any(c[0] == "GET" and c[1].endswith("/x/fields") for c in rec.calls)
        # GET /x/rank carries the query through
        client.get("/x/rank?profile=default", headers=h)
        assert any(c[1].endswith("/x/rank") and c[3].get("profile") == "default"
                   for c in rec.calls)
    finally:
        _restore_auth_maps(op, oh)


def test_host_enforces_consumer_inactive_allowed_routes_and_rate_limits(monkeypatch, tmp_path):
    digest = hashlib.sha256("crm-token".encode()).hexdigest()
    token, digest, original_plaintext, original_hashes = _with_consumer_auth(
        monkeypatch,
        tmp_path,
        issued={
            "status": "inactive",
            "allowed_routes": ["profile:edge"],
            "rate_per_min": 1,
            "burst": 1,
            "keys": [{"sha256_prefix": digest[:12], "status": "active"}],
        },
    )
    try:
        client = TestClient(auth_proxy.app)
        headers = {"Authorization": f"Bearer {token}"}
        inactive = client.post("/v1/chat/completions", headers=headers, json={"model": "profile:edge", "messages": []})
        assert inactive.status_code == 403
        assert inactive.json()["error"]["code"] == "caller_inactive"

        data = _issued_data()
        data["crm"]["status"] = "active"
        _set_issued(data)

        blocked = client.post("/v1/chat/completions", headers=headers, json={"model": "profile:medium", "messages": []})
        assert blocked.status_code == 403
        assert blocked.json()["error"]["code"] == "caller_route_not_allowed"

        ok = client.post("/v1/chat/completions", headers=headers, json={"model": "profile:edge", "messages": []})
        assert ok.status_code == 200
        limited = client.post("/v1/chat/completions", headers=headers, json={"model": "profile:edge", "messages": []})
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "caller_rate_limit"
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)



def test_host_rejects_revoked_and_expired_key_metadata(monkeypatch, tmp_path):
    token = "crm-token"
    digest = hashlib.sha256(token.encode()).hexdigest()
    _, _, original_plaintext, original_hashes = _with_consumer_auth(
        monkeypatch,
        tmp_path,
        token=token,
        issued={"status": "active", "keys": [{"sha256_prefix": digest[:12], "status": "revoked"}]},
    )
    try:
        client = TestClient(auth_proxy.app)
        resp = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, json={"model": "profile:edge", "messages": []})
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "caller_key_revoked"

        data = _issued_data()
        data["crm"]["keys"] = [{"sha256_prefix": digest[:12], "status": "active", "expires_at": 1}]
        _set_issued(data)
        expired = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, json={"model": "profile:edge", "messages": []})
        assert expired.status_code == 403
        assert expired.json()["error"]["code"] == "caller_key_expired"
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)

def test_dashboard_rotates_revokes_and_updates_consumer_settings(monkeypatch, tmp_path):
    digest = hashlib.sha256("crm-token".encode()).hexdigest()
    token, digest, original_plaintext, original_hashes = _with_consumer_auth(
        monkeypatch,
        tmp_path,
        issued={"status": "active", "keys": [{"sha256_prefix": digest[:12], "status": "active"}]},
    )
    try:
        client = _dashboard_client(monkeypatch)
        settings = client.post("/dashboard/api/consumers/crm", json={"status": "active", "allowed_routes": ["profile:edge", "pin:openai/*"], "rate_per_min": 9, "burst": 3})
        assert settings.status_code == 200
        assert settings.json()["settings"]["allowed_routes"] == ["profile:edge", "pin:openai/*"]
        assert settings.json()["settings"]["rate_per_min"] == 9

        created = client.post("/dashboard/api/keys", json={"consumer": "crm", "rotate": True, "grace_period_s": 60})
        assert created.status_code == 200
        body = created.json()
        assert body["api_key"].startswith(auth_proxy.DASHBOARD_KEY_PREFIX + "_")
        issued = _issued_data()["crm"]
        old = [k for k in issued["keys"] if k["sha256_prefix"] == digest[:12]][0]
        assert old["expires_at"] > old["replaced_at"]
        assert body["sha256_prefix"] in json.dumps(issued)
        assert body["api_key"] not in _issued_text()

        revoked = client.post("/dashboard/api/keys/revoke", json={"consumer": "crm", "sha256_prefix": body["sha256_prefix"]})
        assert revoked.status_code == 200
        assert body["sha256_prefix"] not in " ".join(auth_proxy.CALLER_KEY_HASHES.keys())
        issued = _issued_data()["crm"]
        assert any(k["sha256_prefix"] == body["sha256_prefix"] and k["status"] == "revoked" for k in issued["keys"])
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_restricted_consumer_fails_closed_when_route_is_missing(monkeypatch, tmp_path):
    token, _, original_plaintext, original_hashes = _with_consumer_auth(
        monkeypatch,
        tmp_path,
        issued={"status": "active", "allowed_routes": ["profile:edge"]},
    )
    try:
        client = TestClient(auth_proxy.app)
        resp = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, json={"messages": []})
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] == "caller_route_not_allowed"
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_dashboard_rotation_with_zero_grace_expires_old_key_not_new_key(monkeypatch, tmp_path):
    digest = hashlib.sha256("crm-token".encode()).hexdigest()
    token, digest, original_plaintext, original_hashes = _with_consumer_auth(
        monkeypatch,
        tmp_path,
        issued={"status": "active", "keys": [{"sha256_prefix": digest[:12], "status": "active"}]},
    )
    try:
        client = _dashboard_client(monkeypatch)
        created = client.post("/dashboard/api/keys", json={"consumer": "crm", "rotate": True, "grace_period_s": 0})
        assert created.status_code == 200
        new_token = created.json()["api_key"]
        new_prefix = created.json()["sha256_prefix"]

        issued = _issued_data()["crm"]
        new_rows = [k for k in issued["keys"] if k["sha256_prefix"] == new_prefix]
        assert len(new_rows) == 1
        assert "expires_at" not in new_rows[0]

        route_client = TestClient(auth_proxy.app)
        old = route_client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, json={"model": "profile:edge", "messages": []})
        assert old.status_code == 403
        assert old.json()["error"]["code"] == "caller_key_expired"
        new = route_client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {new_token}"}, json={"model": "profile:edge", "messages": []})
        assert new.status_code == 200
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_dashboard_per_key_rotation_only_expires_targeted_key(monkeypatch, tmp_path):
    target = hashlib.sha256("crm-token".encode()).hexdigest()
    other = hashlib.sha256("crm-other".encode()).hexdigest()
    _, _, original_plaintext, original_hashes = _with_consumer_auth(
        monkeypatch,
        tmp_path,
        issued={"status": "active", "keys": [
            {"sha256_prefix": target[:12], "status": "active"},
            {"sha256_prefix": other[:12], "status": "active"},
        ]},
    )
    try:
        client = _dashboard_client(monkeypatch)
        created = client.post("/dashboard/api/keys", json={
            "consumer": "crm", "rotate": True,
            "sha256_prefix": target[:12], "grace_period_s": 60})
        assert created.status_code == 200
        issued = _issued_data()["crm"]
        rotated = next(k for k in issued["keys"] if k["sha256_prefix"] == target[:12])
        untouched = next(k for k in issued["keys"] if k["sha256_prefix"] == other[:12])
        assert rotated["expires_at"] > 0
        assert rotated["replaced_at"] > 0
        assert "expires_at" not in untouched
        assert "replaced_at" not in untouched
        assert untouched["status"] == "active"
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_dashboard_per_key_rotation_unknown_prefix_errors(monkeypatch, tmp_path):
    digest = hashlib.sha256("crm-token".encode()).hexdigest()
    _, _, original_plaintext, original_hashes = _with_consumer_auth(
        monkeypatch,
        tmp_path,
        issued={"status": "active", "keys": [{"sha256_prefix": digest[:12], "status": "active"}]},
    )
    try:
        client = _dashboard_client(monkeypatch)
        resp = client.post("/dashboard/api/keys", json={
            "consumer": "crm", "rotate": True, "sha256_prefix": "deadbeef1234"})
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "key_not_found"
        # The failed rotation must not have minted a key or touched the record.
        issued = _issued_data()["crm"]
        assert len(issued["keys"]) == 1
        assert "expires_at" not in issued["keys"][0]

        malformed = client.post("/dashboard/api/keys", json={
            "consumer": "crm", "rotate": True, "sha256_prefix": "XYZ"})
        assert malformed.status_code == 400
        assert "sha256_prefix" in malformed.json()["error"]["message"]
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_dashboard_rotation_without_prefix_still_expires_all_active_keys(monkeypatch, tmp_path):
    first = hashlib.sha256("crm-token".encode()).hexdigest()
    second = hashlib.sha256("crm-other".encode()).hexdigest()
    _, _, original_plaintext, original_hashes = _with_consumer_auth(
        monkeypatch,
        tmp_path,
        issued={"status": "active", "keys": [
            {"sha256_prefix": first[:12], "status": "active"},
            {"sha256_prefix": second[:12], "status": "active"},
        ]},
    )
    try:
        client = _dashboard_client(monkeypatch)
        created = client.post("/dashboard/api/keys", json={"consumer": "crm", "rotate": True, "grace_period_s": 60})
        assert created.status_code == 200
        new_prefix = created.json()["sha256_prefix"]
        issued = _issued_data()["crm"]
        for prefix in (first[:12], second[:12]):
            row = next(k for k in issued["keys"] if k["sha256_prefix"] == prefix)
            assert row["expires_at"] > 0
            assert row["replaced_at"] > 0
        fresh = next(k for k in issued["keys"] if k["sha256_prefix"] == new_prefix)
        assert "expires_at" not in fresh
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_legacy_plaintext_keys_can_be_rotated_and_revoked(monkeypatch, tmp_path):
    token = "legacy-crm-token"
    digest = hashlib.sha256(token.encode()).hexdigest()
    monkeypatch.setattr(auth_proxy, "_client", _FakeRouteClient())
    monkeypatch.setattr(auth_proxy, "DASHBOARD_KEY_ENV_PATH", str(tmp_path / ".env.secrets"))
    _use_db(monkeypatch, tmp_path)
    original_plaintext = dict(auth_proxy.CALLER_KEYS)
    original_hashes = dict(auth_proxy.CALLER_KEY_HASHES)
    auth_proxy.CALLER_KEYS.clear()
    auth_proxy.CALLER_KEYS[token] = "crm"
    auth_proxy.CALLER_KEY_HASHES.clear()
    auth_proxy._windows.clear()
    try:
        client = _dashboard_client(monkeypatch)
        rotated = client.post("/dashboard/api/keys", json={"consumer": "crm", "rotate": True, "grace_period_s": 0})
        assert rotated.status_code == 200
        new_token = rotated.json()["api_key"]

        route_client = TestClient(auth_proxy.app)
        old = route_client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, json={"model": "profile:edge", "messages": []})
        assert old.status_code == 403
        assert old.json()["error"]["code"] == "caller_key_expired"
        new = route_client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {new_token}"}, json={"model": "profile:edge", "messages": []})
        assert new.status_code == 200

        auth_proxy.CALLER_KEYS[token] = "crm"
        revoked = client.post("/dashboard/api/keys/revoke", json={"consumer": "crm", "sha256_prefix": digest[:12]})
        assert revoked.status_code == 200
        assert revoked.json()["removed_plaintext"] == 1
        assert token not in auth_proxy.CALLER_KEYS
        assert token not in (tmp_path / ".env.secrets").read_text()
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_key_usage_endpoint_returns_exact_key_stats_without_exposing_raw_key(monkeypatch, tmp_path):
    token = "usage-exact-token"
    token, digest, original_plaintext, original_hashes = _with_consumer_auth(monkeypatch, tmp_path, token=token, consumer="crm")
    try:
        client = TestClient(auth_proxy.app)
        headers = {"Authorization": f"Bearer {token}"}
        routed = client.post("/v1/chat/completions", headers=headers, json={"model": "profile:edge", "messages": [{"role": "user", "content": "pong"}]})
        assert routed.status_code == 200

        usage = client.get("/v1/usage", headers=headers)
        assert usage.status_code == 200
        body = usage.json()
        assert body["kind"] == "router_key_usage"
        assert body["viewer"] == "consumer:crm"
        assert body["consumer"] == "crm"
        assert body["key_sha256_prefix"] == digest[:12]
        assert body["detail_level"] == "full"
        assert body["totals"]["requests"] == 1
        assert body["totals"]["tokens_total"] == 2
        assert body["consumer_settings"]["status"] == "active"
        assert body["consumer_settings"]["effective_per_min"] >= 1
        assert body["by_provider"]["openai"]["requests"] == 1
        assert body["by_model_family"]["gpt-5.5-codex"]["requests"] == 1
        assert body["by_route"]["profile:edge"]["requests"] == 1
        assert body["by_served_model"]["gpt-5.5"]["requests"] == 1
        assert body["by_status"] == {"200": 1}
        assert body["recent"][0]["requested_model"] == "profile:edge"
        assert body["recent"][0]["provider"] == "openai"
        assert body["recent"][0]["key_sha256_prefix"] == digest[:12]
        assert body["health_summary"]["request_count"] == 1
        assert body["health_summary"]["success_count"] == 1
        assert body["security"]["raw_api_key_exposed"] is False
        assert body["security"]["full_sha256_exposed"] is False
        assert token not in usage.text
        assert digest not in usage.text
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_dashboard_key_usage_endpoint_is_dashboard_auth_protected(monkeypatch, tmp_path):
    token = "dashboard-key-usage-token"
    token, digest, original_plaintext, original_hashes = _with_consumer_auth(monkeypatch, tmp_path, token=token, consumer="crm")
    try:
        route_client = TestClient(auth_proxy.app)
        routed = route_client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, json={"model": "profile:edge", "messages": []})
        assert routed.status_code == 200

        unauth = route_client.post("/dashboard/api/key-usage", json={"api_key": token})
        assert unauth.status_code == 401
        consumer_auth = route_client.post("/dashboard/api/key-usage", headers={"Authorization": f"Bearer {token}"}, json={"api_key": token})
        assert consumer_auth.status_code == 401

        dashboard = _dashboard_client(monkeypatch)
        resp = dashboard.post("/dashboard/api/key-usage", json={"api_key": token})
        assert resp.status_code == 200
        body = resp.json()
        assert body["viewer"] == "dashboard:tester"
        assert body["consumer"] == "crm"
        assert body["key_sha256_prefix"] == digest[:12]
        assert body["totals"]["requests"] == 1
        assert token not in resp.text
        assert digest not in resp.text
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_dashboard_admin_endpoints_reject_unauthenticated_and_consumer_bearer(monkeypatch, tmp_path):
    token, _, original_plaintext, original_hashes = _with_consumer_auth(monkeypatch, tmp_path)
    try:
        client = TestClient(auth_proxy.app)
        headers = {"Authorization": f"Bearer {token}"}
        cases = [
            ("GET", "/dashboard/api/stats", None),
            ("GET", "/dashboard/api/full", None),
            ("GET", "/dashboard/api/evaluate", None),
            ("GET", "/dashboard/api/policies", None),
            ("GET", "/dashboard/api/keys/reveal?consumer=crm", None),
            ("POST", "/dashboard/api/key-usage", {"api_key": token}),
            ("POST", "/dashboard/api/keys", {"consumer": "crm"}),
            ("POST", "/dashboard/api/keys/revoke", {"consumer": "crm", "sha256_prefix": "01234567"}),
            ("POST", "/dashboard/api/consumers/crm", {"status": "inactive"}),
        ]
        for method, path, payload in cases:
            unauth = client.request(method, path, json=payload)
            assert unauth.status_code == 401, path
            consumer_auth = client.request(method, path, headers=headers, json=payload)
            assert consumer_auth.status_code == 401, path
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)


def test_malformed_issued_key_metadata_fails_closed(monkeypatch, tmp_path):
    token, _, original_plaintext, original_hashes = _with_consumer_auth(monkeypatch, tmp_path)
    try:
        client = TestClient(auth_proxy.app)
        digest12 = hashlib.sha256(token.encode()).hexdigest()[:12]
        bad_records = [
            {"crm": ["not", "an", "object"]},
            {"crm": {"keys": "not-a-list"}},
            {"crm": {"keys": ""}},
            {"crm": {"keys": {}}},
            {"crm": {"keys": 0}},
            {"crm": {"keys": [{"sha256_prefix": digest12, "status": "garbage"}]}},
        ]
        for records in bad_records:
            _set_issued(records)
            resp = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, json={"model": "profile:edge", "messages": []})
            assert resp.status_code == 403, records
            assert resp.json()["error"]["code"] in {"caller_inactive", "caller_key_revoked"}
            auth_proxy._issued_keys_load_failed = False

        # A store LOAD FAILURE (the old unparseable file) also fails closed.
        monkeypatch.setattr(host_store, "get_consumer_keys", lambda: ({}, False))
        resp = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"}, json={"model": "profile:edge", "messages": []})
        assert resp.status_code == 403
        assert resp.json()["error"]["code"] in {"caller_inactive", "caller_key_revoked"}
        auth_proxy._issued_keys_load_failed = False
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)
        auth_proxy._issued_keys_load_failed = False


def test_usage_history_survives_stats_reset_and_supports_windows_pagination_and_costs(monkeypatch, tmp_path):
    history_path = tmp_path / "usage-history.jsonl"
    monkeypatch.setenv("ROUTER_USAGE_HISTORY_PATH", str(history_path))
    token = "usage-history-token"
    token, digest, original_plaintext, original_hashes = _with_consumer_auth(monkeypatch, tmp_path, token=token, consumer="crm")
    try:
        auth_proxy._reset_stats_for_tests()
        auth_proxy._record_request(caller="crm", method="POST", path="/v1/chat/completions", status=200, latency_ms=10, provider="openai", model_family="gpt-5.5-codex", served_model_id="gpt-5.5", requested_model="profile:edge", tokens_in=1, tokens_out=1, tokens_total=2, key_sha256=digest, ts=1_700_000_000)
        auth_proxy._record_request(caller="crm", method="POST", path="/v1/chat/completions", status=200, latency_ms=20, provider="openrouter", model_family="gpt-5.5", served_model_id="openai/gpt-5.5", requested_model="profile:edge", tokens_in=1_000_000, tokens_out=500_000, tokens_total=1_500_000, key_sha256=digest, ts=1_700_086_400)

        auth_proxy._reset_stats_for_tests()
        client = TestClient(auth_proxy.app)
        resp = client.get("/v1/usage?since=1700080000&limit=1&offset=0", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["source"]["persistent_history"] is True
        assert body["window"]["since"] == 1_700_080_000
        assert body["window"]["recent_total"] == 1
        assert body["window"]["recent_returned"] == 1
        assert body["totals"]["requests"] == 1
        assert body["totals"]["tokens_total"] == 1_500_000
        assert body["cost_estimate"]["estimated"] is True
        assert body["cost_estimate"]["usd"] == 6.25
        assert body["by_provider"]["openrouter"]["cost_usd"] == 6.25
        assert body["daily_totals"][0]["date"] == "2023-11-15"
        assert body["monthly_totals"][0]["month"] == "2023-11"
        assert body["recent"][0]["requested_model"] == "profile:edge"

        empty_page = client.get("/v1/usage?window=24h&limit=1&offset=1", headers={"Authorization": f"Bearer {token}"})
        assert empty_page.status_code == 200
        assert empty_page.json()["window"]["limit"] == 1
        assert empty_page.json()["window"]["offset"] == 1
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)
        auth_proxy._reset_stats_for_tests()


def test_dashboard_stats_defaults_to_latest_events_and_consumer_rows(monkeypatch, tmp_path):
    history_path = tmp_path / "usage-history.jsonl"
    monkeypatch.setenv("ROUTER_USAGE_HISTORY_PATH", str(history_path))
    token = "dashboard-history-token"
    token, digest, original_plaintext, original_hashes = _with_consumer_auth(monkeypatch, tmp_path, token=token, consumer="crm")
    try:
        auth_proxy._reset_stats_for_tests()
        auth_proxy._record_request(caller="crm", method="POST", path="/v1/chat/completions", status=200, latency_ms=20, provider="openrouter", model_family="gpt-5.5", served_model_id="openai/gpt-5.5", requested_model="profile:edge", tokens_in=10, tokens_out=5, tokens_total=15, key_sha256=digest, ts=1_700_086_400)
        auth_proxy._record_request(caller="crm", method="POST", path="/v1/chat/completions", status=200, latency_ms=30, provider="openrouter", model_family="gpt-5.5", served_model_id="openai/gpt-5.5", requested_model="profile:edge", tokens_in=11, tokens_out=6, tokens_total=17, key_sha256=digest, ts=1_700_086_500)
        wing_digest = hashlib.sha256("wing-token".encode()).hexdigest()
        auth_proxy.CALLER_KEY_HASHES[wing_digest] = "wingston"
        auth_proxy._record_request(caller="wingston", method="POST", path="/v1/chat/completions", status=200, latency_ms=40, provider="openai", model_family="gpt-5.5-codex", served_model_id="gpt-5.5", requested_model="profile:medium", tokens_in=20, tokens_out=10, tokens_total=30, key_sha256=wing_digest, ts=1_700_086_600)
        auth_proxy._reset_stats_for_tests()
        dashboard = _dashboard_client(monkeypatch)
        resp = dashboard.get("/dashboard/api/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["timeframe"]["selected"] == "recent"
        assert body["timeframe"]["source"] == "persistent_history"
        assert body["totals"]["requests"] == 3
        assert body["by_caller"]["crm"]["requests"] == 2
        assert body["by_caller"]["wingston"]["requests"] == 1
        crm_row = next(row for row in body["keys"] if row["consumer"] == "crm")
        wing_row = next(row for row in body["keys"] if row["consumer"] == "wingston")
        assert crm_row["stats"]["requests"] == 2
        assert crm_row["stats"]["last_seen"] == 1_700_086_500
        assert wing_row["stats"]["requests"] == 1

        selected = dashboard.get("/dashboard/api/stats?consumer=crm")
        assert selected.status_code == 200
        selected_body = selected.json()
        assert selected_body["selected_consumer"] == "crm"
        assert selected_body["totals"]["requests"] == 2
        selected_crm_row = next(row for row in selected_body["keys"] if row["consumer"] == "crm")
        assert selected_crm_row["stats"]["requests"] == 2
        selected_wing_row = next(row for row in selected_body["keys"] if row["consumer"] == "wingston")
        assert selected_wing_row["stats"]["requests"] == 0

        legacy_all = dashboard.get("/dashboard/api/stats?timeframe=all")
        assert legacy_all.status_code == 200
        assert legacy_all.json()["timeframe"]["selected"] == "recent"

        runtime = dashboard.get("/dashboard/api/stats?timeframe=runtime")
        assert runtime.status_code == 200
        assert runtime.json()["timeframe"]["selected"] == "runtime"
        assert runtime.json()["totals"]["requests"] == 0
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)
        auth_proxy._reset_stats_for_tests()


def test_dashboard_api_key_login_filters_same_dashboard_to_exact_key(monkeypatch, tmp_path):
    history_path = tmp_path / "usage-history.jsonl"
    monkeypatch.setenv("ROUTER_USAGE_HISTORY_PATH", str(history_path))
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET", "test-dashboard-session-secret")
    monkeypatch.setattr(auth_proxy, "_client", _FakeAsyncClient())
    monkeypatch.setattr(auth_proxy, "_policy_catalog_snapshot", _fake_policy_catalog)
    token_a = "consumer-dashboard-token-a"
    token_b = "consumer-dashboard-token-b"
    digest_a = hashlib.sha256(token_a.encode()).hexdigest()
    digest_b = hashlib.sha256(token_b.encode()).hexdigest()
    original_plaintext = dict(auth_proxy.CALLER_KEYS)
    original_hashes = dict(auth_proxy.CALLER_KEY_HASHES)
    auth_proxy.CALLER_KEYS.clear()
    auth_proxy.CALLER_KEY_HASHES.clear()
    auth_proxy.CALLER_KEY_HASHES.update({digest_a: "crm", digest_b: "crm"})
    try:
        auth_proxy._reset_stats_for_tests()
        auth_proxy._record_request(caller="crm", method="POST", path="/v1/chat/completions", status=200, latency_ms=10, provider="openai", model_family="gpt-5.5-codex", served_model_id="gpt-5.5", requested_model="profile:edge", tokens_in=1, tokens_out=2, tokens_total=3, key_sha256=digest_a, ts=1_700_000_000)
        auth_proxy._record_request(caller="crm", method="POST", path="/v1/chat/completions", status=200, latency_ms=20, provider="openrouter", model_family="gpt-5.5", served_model_id="openai/gpt-5.5", requested_model="profile:medium", tokens_in=10, tokens_out=20, tokens_total=30, key_sha256=digest_b, ts=1_700_000_100)
        auth_proxy._reset_stats_for_tests()
        client = TestClient(auth_proxy.app)
        login = client.post("/dashboard/api/login", json={"api_key": token_a})
        assert login.status_code == 200
        assert login.json()["role"] == "consumer"
        client.cookies.set(auth_proxy.DASHBOARD_COOKIE_NAME, auth_proxy._make_dashboard_session("crm", role="consumer", consumer="crm", key_sha256=digest_a))
        stats = client.get("/dashboard/api/stats")
        assert stats.status_code == 200
        body = stats.json()
        assert body["viewer_role"] == "consumer"
        assert body["selected_consumer"] == "crm"
        assert body["selected_key_sha256_prefix"] == digest_a[:12]
        assert body["totals"]["requests"] == 1
        assert body["totals"]["tokens_total"] == 3
        assert list(body["by_route"].keys()) == ["profile:edge"]
        assert len(body["keys"]) == 1
        assert body["keys"][0]["consumer"] == "crm"
        assert body["keys"][0]["stats"]["requests"] == 1
        assert body["recent"][0]["key_sha256_prefix"] == digest_a[:12]
        assert digest_a not in stats.text
        assert digest_b not in stats.text
        assert client.get("/dashboard/api/policies").status_code == 403
        assert client.post("/dashboard/api/keys", json={"consumer": "crm"}).status_code == 403
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)
        auth_proxy._reset_stats_for_tests()


def test_dashboard_key_usage_accepts_window_and_pagination_controls(monkeypatch, tmp_path):
    history_path = tmp_path / "usage-history.jsonl"
    monkeypatch.setenv("ROUTER_USAGE_HISTORY_PATH", str(history_path))
    token = "dashboard-usage-history-token"
    token, digest, original_plaintext, original_hashes = _with_consumer_auth(monkeypatch, tmp_path, token=token, consumer="crm")
    try:
        auth_proxy._reset_stats_for_tests()
        auth_proxy._record_request(caller="crm", method="POST", path="/v1/chat/completions", status=200, latency_ms=20, provider="openrouter", model_family="gpt-5.5", served_model_id="openai/gpt-5.5", requested_model="profile:edge", tokens_in=1_000_000, tokens_out=500_000, tokens_total=1_500_000, key_sha256=digest, ts=1_700_086_400)
        dashboard = _dashboard_client(monkeypatch)
        resp = dashboard.post("/dashboard/api/key-usage", json={"api_key": token, "since": 1700080000, "limit": 5, "offset": 0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["window"]["since"] == 1_700_080_000
        assert body["window"]["limit"] == 5
        assert body["cost_estimate"]["usd"] == 6.25
    finally:
        _restore_auth_maps(original_plaintext, original_hashes)
        auth_proxy._reset_stats_for_tests()


def test_openapi_and_dashboard_document_key_usage_controls():
    html = auth_proxy._dashboard_html()
    assert "tabKeyUsage" in html
    assert "keyUsageApiKey" in html
    assert "loadKeyUsage" in html
    assert "cost_estimate" in html
    assert "recentOffset" in html
    assert "Latest 100 events" in html
    assert "value='all' selected" not in html
    assert "dashboardLoading" in html
    assert "AbortController" in html
    spec = TestClient(auth_proxy.app).get("/openapi.json")
    assert spec.status_code == 200
    paths = spec.json()["paths"]
    assert "/v1/usage" in paths
    assert "/api/usage" in paths
    assert "/dashboard/api/key-usage" in paths
    docs = (ROOT / "docs" / "USAGE_ENDPOINTS.md").read_text()
    assert "Persistent history" in docs
    assert "?since=" in docs
    assert "?window=24h" in docs
    assert "limit" in docs and "offset" in docs
    assert "cost_estimate" in docs


def test_dashboard_html_escapes_consumer_names_for_js_context_and_logout_clears_keys():
    html = auth_proxy._dashboard_html()
    assert "function jsarg" in html
    assert "onclick=\"pickConsumer(${jsarg(r.name)})\"" in html
    assert "showLogin()" in html
    assert html.count("if(r.status===401){showLogin();return}") >= 5
    # Logout clears any raw key material still displayed (new-key dialog + the
    # in-drawer key-ready panel), so it never lingers after the session ends.
    assert "newKeyValue').value=''" in html
    assert "newKeyHandoffValue').value=''" in html
    assert "keyReady').innerHTML=''" in html
    assert "onclick=\"pickConsumer('${esc(r.name)}')\"" not in html


def test_x_paths_hidden_from_consumers(monkeypatch):
    # Even a fully authorized caller must get 404 on /x/* — the runtime
    # endpoint is dashboard-internal, not part of the consumer API.
    from fastapi.testclient import TestClient
    monkeypatch.setattr(
        auth_proxy, "_caller_auth",
        lambda token: {"ok": True, "caller": "tester", "digest": None},
    )
    with TestClient(auth_proxy.app) as client:
        r = client.get("/x/runtime", headers={"Authorization": "Bearer k"})
        assert r.status_code == 404
        r = client.post("/x/anything", headers={"Authorization": "Bearer k"})
        assert r.status_code == 404


def test_provider_health_classification():
    f = auth_proxy._provider_health_for

    # not configured: bearer auth whose env var is missing
    h = f({"auth_env": "NO_SUCH_KEY_XYZ"}, "heurist", None, {})
    assert h["state"] == "disconnected"
    assert "NO_SUCH_KEY_XYZ" in h["reason"]

    # failing: breaker open, reason carries the recent error
    runtime = {"circuit_breakers": {"openrouter": {"open": True, "consecutive_failures": 4}}}
    recent = {"openrouter": {"ok": False, "error_kind": "payment_required", "http_status": 402, "ts": 1}}
    h = f({"auth_env": "PATH"}, "openrouter", runtime, recent)  # PATH is always set
    assert h["state"] == "failing"
    assert "breaker open" in h["reason"]
    assert "payment_required(402)" in h["reason"]

    # failing: no breaker, but last recent attempt errored
    h = f({"auth": {"kind": "none"}}, "antseed_edge", {},
          {"antseed_edge": {"ok": False, "error_kind": "payment_required", "http_status": 402, "ts": 1}})
    assert h["state"] == "failing"

    # ok: recent success
    h = f({"auth": {"kind": "oauth", "provider": "codex"}}, "openai", {},
          {"openai": {"ok": True, "ts": 1}})
    assert h["state"] == "ok"

    # disabled provider wins over everything except disconnected
    h = f({"auth": {"kind": "none"}}, "p", {"disabled_providers": {"p": "auth_error"}}, {})
    assert h["state"] == "failing"
    assert "auth_error" in h["reason"]

    # idle: configured, no data at all (also the runtime-unreachable path)
    h = f({"auth": {"kind": "none"}}, "antseed_free", None, {})
    assert h["state"] == "idle"


def test_recent_provider_attempts_reads_decision_traces(monkeypatch):
    import time as _time
    from collections import deque
    now = int(_time.time())
    rows = [
        {"event": "request", "ts": now, "status": 502, "provider": None,
         "decision_trace": {"decision_path": [
             {"event": "attempted", "provider_id": "openrouter",
              "error_kind": "payment_required", "http_status": 402},
         ]}},
        {"event": "request", "ts": now - 10, "status": 200, "provider": "openai",
         "decision_trace": {"decision_path": [
             {"event": "attempted", "provider_id": "openai"},
         ]}},
    ]
    monkeypatch.setitem(auth_proxy._stats, "recent", deque(rows, maxlen=500))
    out = auth_proxy._recent_provider_attempts(window_s=900)
    assert out["openrouter"]["ok"] is False
    assert out["openrouter"]["error_kind"] == "payment_required"
    assert out["openai"]["ok"] is True


def test_dashboard_html_has_provider_health_ui():
    html = auth_proxy._dashboard_html()
    assert "connectedOnly" in html          # the toggle
    assert ".provider-chip.failing" in html  # red style
    assert "healthDot" in html              # dot renderer


def test_policy_snapshot_prefers_live_ranks():
    live = {"default": [
        {"provider": "openai", "model_family": "gpt-5.5-codex",
         "served_model_id": "gpt-5.5", "tier": "partner", "discovery": "static",
         "price_in": 0.0, "price_out": 0.0, "quality": 0.92, "score": 0.9},
        {"provider": "antseed_edge", "model_family": "claude-opus-4-8",
         "served_model_id": "claude-opus-4-8", "tier": "fallback",
         "discovery": "marketplace", "price_in": 1.0, "price_out": 5.0,
         "quality": 0.93, "score": 0.5},
    ]}
    catalog = auth_proxy._policy_catalog_snapshot(live_ranks=live)
    profiles = {row["name"]: row for row in catalog["profiles"]}
    default = profiles["default"]
    assert default["rank_source"] == "router"
    fams = {m["name"]: m for m in default["models"]}
    assert fams["claude-opus-4-8"]["served_by"][0]["provider"] == "antseed_edge"
    assert fams["claude-opus-4-8"]["served_by"][0]["price_in"] == 1.0


def test_provider_health_runway_from_balances():
    f = auth_proxy._provider_health_for

    # paid provider with empty deposits: red BEFORE any request fails
    runtime = {"balances": {"antseed_cheap": {"kind": "deposits_usdc", "value": 0.0}}}
    h = f({"auth": {"kind": "none"}, "market_price_cap": {"input": 2, "output": 10}},
          "antseed_cheap", runtime, {})
    assert h["state"] == "failing"
    assert h["runway"] == "empty"
    assert "0.0" in h["reason"]
    assert h["balance"]["kind"] == "deposits_usdc"

    # FREE buyer (cap 0/0) with 0 deposits is healthy — it pays nobody
    runtime = {"balances": {"antseed_free": {"kind": "deposits_usdc", "value": 0.0}}}
    h = f({"auth": {"kind": "none"}, "market_price_cap": {"input": 0, "output": 0}},
          "antseed_free", runtime, {"antseed_free": {"ok": True, "ts": 1}})
    assert h["state"] == "ok"
    assert h["runway"] is None

    # credits low (not empty): stays ok but runway warns
    runtime = {"balances": {"openrouter": {"kind": "credits_usd", "value": 12.0}}}
    h = f({"auth_env": "PATH"}, "openrouter", runtime, {"openrouter": {"ok": True, "ts": 1}})
    assert h["state"] == "ok"
    assert h["runway"] == "low"

    # quota window mostly used -> low
    runtime = {"balances": {"openai": {"kind": "quota_window", "value": 0.9}}}
    h = f({"auth": {"kind": "oauth"}}, "openai", runtime, {"openai": {"ok": True, "ts": 1}})
    assert h["runway"] == "low"


def test_dashboard_html_renders_runway():
    html = auth_proxy._dashboard_html()
    assert "chipBalance" in html
    assert "runway" in html


def test_proxy_passes_sse_through_unbuffered(monkeypatch):
    from fastapi.testclient import TestClient

    class FakeUpstreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        def __init__(self):
            self.aread_called = False

        async def aiter_raw(self):
            yield b"data: one\n\n"
            yield b"data: [DONE]\n\n"

        async def aread(self):
            self.aread_called = True
            return b""

        async def aclose(self):
            pass

    fake_resp = FakeUpstreamResponse()

    class FakeClient:
        def build_request(self, method, url, content=None, headers=None):
            return (method, url)

        async def send(self, req, stream=True):
            return fake_resp

    monkeypatch.setattr(auth_proxy, "_caller_auth",
                        lambda token: {"ok": True, "caller": "tester", "digest": None})
    monkeypatch.setattr(auth_proxy, "_client", FakeClient())
    client = TestClient(auth_proxy.app)  # no context: startup must not replace _client
    r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer k"},
                    json={"model": "profile:edge", "messages": [], "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert b"data: one" in r.content and b"[DONE]" in r.content
    assert fake_resp.aread_called is False   # streamed, not buffered


# ---- market tab -------------------------------------------------------------

def test_dashboard_market_requires_admin_and_proxies_router(monkeypatch):
    client = TestClient(auth_proxy.app)
    assert client.get("/dashboard/api/market").status_code == 401

    sample = {"families": [{"family": "qwen3-235b-a22b", "quality": 0.9,
                            "sellers_total": 3, "rows": []}], "ts": 1}

    async def fake_fetch():
        return sample

    monkeypatch.setattr(auth_proxy, "_fetch_live_market", fake_fetch)
    admin = _dashboard_client(monkeypatch)
    resp = admin.get("/dashboard/api/market")
    assert resp.status_code == 200
    assert resp.json() == sample


def test_dashboard_market_router_down_is_502(monkeypatch):
    async def fake_fetch():
        return None

    monkeypatch.setattr(auth_proxy, "_fetch_live_market", fake_fetch)
    client = _dashboard_client(monkeypatch)
    resp = client.get("/dashboard/api/market")
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "market"


def test_dashboard_html_has_market_tab(monkeypatch):
    client = _dashboard_client(monkeypatch)
    html = client.get("/dashboard").text
    for needle in ("tabMarket", "marketPage", "tradableOnly",
                   "loadMarket", "renderMarket", "/dashboard/api/market"):
        assert needle in html, needle


# ---- per-consumer spend ------------------------------------------------------

def test_aggregate_usage_rows_computes_per_consumer_spend(monkeypatch):
    monkeypatch.setattr(auth_proxy, "_price_table", lambda: {
        ("medium", "openrouter"): {"input": 2.0, "output": 10.0}})
    rows = [
        {"event": "request", "caller": "acme", "provider": "openrouter",
         "model_family": "medium", "status": 200,
         "tokens_in": 1_000_000, "tokens_out": 100_000},
        {"event": "request", "caller": "acme", "provider": "openai",
         "model_family": "unpriced", "status": 200,
         "tokens_in": 50, "tokens_out": 50},
        {"event": "request", "caller": "other", "provider": "openrouter",
         "model_family": "medium", "status": 200,
         "tokens_in": 500_000, "tokens_out": 0},
    ]
    agg = auth_proxy._aggregate_usage_rows(rows)
    assert agg["by_caller_all"]["acme"]["cost_usd"] == 3.0      # $2 in + $1 out
    assert agg["by_caller_all"]["other"]["cost_usd"] == 1.0
    assert agg["totals"]["cost_usd"] == 4.0                     # unpriced row adds 0
    assert agg["by_provider"]["openrouter"]["cost_usd"] == 4.0


def test_dashboard_html_has_consumer_spend_column(monkeypatch):
    client = _dashboard_client(monkeypatch)
    html = client.get("/dashboard").text
    assert "Est. spend" in html
    assert "money(stats.cost_usd)" in html


def test_record_request_accumulates_stamped_cost_in_runtime_counters():
    auth_proxy._record_request(caller="spendy-test", status=200, latency_ms=10,
                               provider="openrouter", model_family="medium",
                               tokens_in=100, tokens_out=10, tokens_total=110,
                               cost_usd=0.0123)
    auth_proxy._record_request(caller="spendy-test", status=200, latency_ms=10,
                               provider="openrouter", model_family="medium",
                               tokens_in=100, tokens_out=10, tokens_total=110,
                               cost_usd=0.0007)
    with auth_proxy._stats_lock:
        snap = auth_proxy._counter_snapshot(auth_proxy._stats["by_caller"]["spendy-test"])
    assert snap["cost_usd"] == 0.013
    # event in the recent ring carries the stamp (feeds persistent history)
    with auth_proxy._stats_lock:
        rows = [r for r in auth_proxy._stats["recent"] if r.get("caller") == "spendy-test"]
    assert rows and rows[0].get("cost_usd") in (0.0007, 0.0123)


def test_cost_for_event_prefers_stamped_over_estimate():
    prices = {("medium", "openrouter"): {"input": 2.0, "output": 10.0}}
    stamped = {"cost_usd": 0.5, "model_family": "medium", "provider": "openrouter",
               "tokens_in": 1_000_000, "tokens_out": 0}
    cost, meta = auth_proxy._cost_for_event(stamped, prices)
    assert cost == 0.5 and meta is None       # NOT the $2.00 estimate
    unstamped = {k: v for k, v in stamped.items() if k != "cost_usd"}
    cost, meta = auth_proxy._cost_for_event(unstamped, prices)
    assert cost == 2.0 and meta is not None


def test_cost_for_event_clamps_negative_spend_to_zero():
    # Old events recorded before the source clamp carry negative spend (tokens ×
    # a negative chosen price). They must never subtract from a total. Both the
    # stamped path and the read-time estimate are clamped to >= 0.
    prices = {("medium", "openrouter"): {"input": -2.0, "output": -2.0}}
    stamped_neg = {"cost_usd": -22079.0, "model_family": "medium", "provider": "openrouter",
                   "tokens_in": 1_000_000, "tokens_out": 0}
    cost, _ = auth_proxy._cost_for_event(stamped_neg, prices)
    assert cost == 0.0
    estimate_neg = {"model_family": "medium", "provider": "openrouter",
                    "tokens_in": 1_000_000, "tokens_out": 0}   # no stamp -> negative estimate
    cost, _ = auth_proxy._cost_for_event(estimate_neg, prices)
    assert cost == 0.0


def test_parse_stream_tail_extracts_usage_router_and_error():
    final = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
                        "x_router": {"provider": "openrouter", "model_family": "gpt-5.5",
                                     "served_model_id": "openai/gpt-5.5", "cost_usd": 0.000165}})
    tail = ("data: " + json.dumps({"choices": [{"delta": {"content": "hi"}}]}) + "\n\n"
            + "data: " + final + "\n\ndata: [DONE]\n\n").encode()
    meta = auth_proxy._parse_stream_tail(tail)
    assert meta["usage"]["total_tokens"] == 13
    assert meta["x_router"]["cost_usd"] == 0.000165
    assert "error" not in meta

    err_tail = b'data: {"error": {"message": "boom", "type": "router_error", "code": "x"}}\n\n'
    assert auth_proxy._parse_stream_tail(err_tail)["error"]["message"] == "boom"
    # garbage / partial lines never raise
    assert auth_proxy._parse_stream_tail(b"data: {truncated") == {}


def test_trim_sse_tail_keeps_oversized_final_event_whole():
    # The Activity-blank bug: the final SSE event carries x_router whose
    # decision_trace (the ranked catalog) pushes it WAY past 64 KiB. The old blind
    # 64 KiB byte cap sliced that event's JSON -> _parse_stream_tail failed ->
    # provider/tokens/trace recorded null. _trim_sse_tail must keep the final
    # event intact however large, so the trace still parses.
    big_trace = {"ranked": [{"provider_id": f"p{i}", "score": i} for i in range(4000)]}
    final = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
                        "x_router": {"provider": "openai", "model_family": "gpt-5.5",
                                     "cost_usd": 0.0002, "decision_trace": big_trace}})
    assert len(final) > 65536, "final event must exceed the old 64 KiB cap to bite"
    # simulate the streamed buffer: many prior content events + the giant final one
    tail = bytearray()
    for i in range(50):
        tail.extend(("data: " + json.dumps({"choices": [{"delta": {"content": "x" * 500}}]})
                     + "\n\n").encode())
        auth_proxy._trim_sse_tail(tail)
    tail.extend(("data: " + final + "\n\n").encode())
    auth_proxy._trim_sse_tail(tail)
    tail.extend(b"data: [DONE]\n\n")
    auth_proxy._trim_sse_tail(tail)

    meta = auth_proxy._parse_stream_tail(bytes(tail))
    assert meta["x_router"]["provider"] == "openai"          # was None under the old cap
    assert meta["x_router"]["model_family"] == "gpt-5.5"
    assert meta["x_router"]["decision_trace"]["ranked"][0]["provider_id"] == "p0"
    assert meta["usage"]["total_tokens"] == 13


def test_proxy_records_tokens_and_cost_from_streamed_response(monkeypatch):
    from fastapi.testclient import TestClient

    final = json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                        "x_router": {"provider": "openrouter", "model_family": "gpt-5.5",
                                     "served_model_id": "openai/gpt-5.5", "cost_usd": 0.002}})

    class FakeUpstreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_raw(self):
            yield b'data: {"choices": [{"delta": {"content": "hel"}}]}\n\n'
            yield b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n'
            yield ("data: " + final + "\n\n").encode()
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            pass

    class FakeClient:
        def build_request(self, method, url, content=None, headers=None):
            return (method, url)

        async def send(self, req, stream=True):
            return FakeUpstreamResponse()

    monkeypatch.setattr(auth_proxy, "_caller_auth",
                        lambda token: {"ok": True, "caller": "stream-spender", "digest": None})
    monkeypatch.setattr(auth_proxy, "_client", FakeClient())
    client = TestClient(auth_proxy.app)  # no context: startup must not replace _client
    r = client.post("/v1/chat/completions", headers={"Authorization": "Bearer k"},
                    json={"model": "profile:edge", "messages": [], "stream": True})
    assert r.status_code == 200 and b"[DONE]" in r.content

    with auth_proxy._stats_lock:
        rows = [dict(x) for x in auth_proxy._stats["recent"]
                if x.get("caller") == "stream-spender" and x.get("event") == "request"]
        counter = auth_proxy._counter_snapshot(auth_proxy._stats["by_caller"]["stream-spender"])
    assert rows, "streamed request must be recorded"
    assert rows[0]["tokens_total"] == 150
    assert rows[0]["cost_usd"] == 0.002
    assert rows[0]["provider"] == "openrouter"
    assert counter["cost_usd"] == 0.002


# ---- provider key reveal -----------------------------------------------------

def test_provider_key_reveal_requires_admin_and_resolves_env_and_oauth(monkeypatch, tmp_path):
    monkeypatch.setattr(auth_proxy, "_load_policy_config", lambda: {"providers": {
        "openrouter": {"auth_env": "TEST_OR_KEY"},
        "openai": {"api_kind": "openai_codex", "auth": {"kind": "oauth"}},
        "heurist": {"auth_env": "MISSING_ENV_VAR_XYZ"},
    }})
    monkeypatch.setenv("TEST_OR_KEY", "sk-or-raw-123")
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(json.dumps({"tokens": {"access_token": "codex-tok", "account_id": "a"}}))
    monkeypatch.setenv("CODEX_AUTH_PATH", str(auth_file))

    assert TestClient(auth_proxy.app).get(
        "/dashboard/api/provider-keys/reveal?provider=openrouter").status_code == 401
    # any admin works now (SSO users aren't named 'admin'); no per-name tier
    non_admin = _dashboard_client(monkeypatch, user="tester")
    assert non_admin.get(
        "/dashboard/api/provider-keys/reveal?provider=openrouter").status_code == 200

    admin = _dashboard_client(monkeypatch, user="admin")
    r = admin.get("/dashboard/api/provider-keys/reveal?provider=openrouter")
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "env" and body["value"] == "sk-or-raw-123"
    assert body["fingerprint"]

    r = admin.get("/dashboard/api/provider-keys/reveal?provider=openai")
    assert r.status_code == 200
    assert r.json()["kind"] == "oauth" and r.json()["value"] == "codex-tok"

    # configured env var that is unset, and unknown providers -> 404
    assert admin.get("/dashboard/api/provider-keys/reveal?provider=heurist").status_code == 404
    assert admin.get("/dashboard/api/provider-keys/reveal?provider=nope").status_code == 404


def test_dashboard_html_has_provider_key_reveal_buttons(monkeypatch):
    html = _dashboard_client(monkeypatch).get("/dashboard").text
    for needle in ("revealProviderKey", "copyProviderKey",
                   "/dashboard/api/provider-keys/reveal"):
        assert needle in html, needle


def test_update_provider_key_requires_admin_and_persists(monkeypatch, tmp_path):
    monkeypatch.setattr(auth_proxy, "_load_policy_config", lambda: {"providers": {
        "heurist": {"auth_env": "HEURIST_API_KEY"},
        "openai": {"api_kind": "openai_codex", "auth": {"kind": "oauth"}},
    }})
    env_file = tmp_path / ".env.secrets"
    monkeypatch.setattr(auth_proxy, "DASHBOARD_KEY_ENV_PATH", str(env_file))

    payload = {"provider": "heurist", "key": "sk-heurist-new"}
    # 401 unauthenticated; any admin allowed (no per-name tier)
    assert TestClient(auth_proxy.app).post(
        "/dashboard/api/provider-keys/update", json=payload).status_code == 401
    non_admin = _dashboard_client(monkeypatch, user="tester")
    assert non_admin.post(
        "/dashboard/api/provider-keys/update", json=payload).status_code == 200

    admin = _dashboard_client(monkeypatch, user="admin")
    r = admin.post("/dashboard/api/provider-keys/update", json=payload)
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True and r.json()["provider"] == "heurist"
    # persisted to .env.secrets under the provider's auth_env (PVC source of truth)
    assert "HEURIST_API_KEY=sk-heurist-new" in env_file.read_text()

    # provider with no auth_env (oauth/codex) -> 400, not applicable
    assert admin.post("/dashboard/api/provider-keys/update",
                       json={"provider": "openai", "key": "x"}).status_code == 400
    # unknown provider -> 404; missing key -> 400
    assert admin.post("/dashboard/api/provider-keys/update",
                       json={"provider": "nope", "key": "x"}).status_code == 404
    assert admin.post("/dashboard/api/provider-keys/update",
                       json={"provider": "heurist"}).status_code == 400


def test_dashboard_html_has_provider_key_edit_control(monkeypatch):
    html = _dashboard_client(monkeypatch).get("/dashboard").text
    for needle in ("editProviderKey", "/dashboard/api/provider-keys/update"):
        assert needle in html, needle


def test_codex_accounts_admin_add_list_delete(monkeypatch, tmp_path):
    accounts_dir = tmp_path / "codex" / "accounts"
    monkeypatch.setattr(auth_proxy, "CODEX_ACCOUNTS_DIR", str(accounts_dir))
    monkeypatch.setattr(auth_proxy, "CODEX_AUTH_PATH", None)
    auth_json = {"tokens": {"access_token": "tok-1", "refresh_token": "r", "account_id": "acct-1"}}

    # unauthenticated denied; any admin allowed (no per-name tier)
    assert TestClient(auth_proxy.app).get("/dashboard/api/codex/accounts").status_code == 401
    non_admin = _dashboard_client(monkeypatch, user="tester")
    assert non_admin.get("/dashboard/api/codex/accounts").json()["accounts"] == []

    admin = _dashboard_client(monkeypatch, user="admin")
    listed0 = admin.get("/dashboard/api/codex/accounts").json()
    assert listed0["accounts"] == [] and listed0["active"] is None and "activity" in listed0

    # add
    r = admin.post("/dashboard/api/codex/accounts", json={"name": "Team One!", "auth_json": auth_json})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True and r.json()["account"] == "team-one"
    assert (accounts_dir / "team-one.json").exists()

    # list reflects it (fingerprint present, raw token never returned)
    listed = admin.get("/dashboard/api/codex/accounts").json()["accounts"]
    assert listed and listed[0]["name"] == "team-one" and listed[0]["account_id"] == "acct-1"
    assert listed[0]["fingerprint"] and "tok-1" not in json.dumps(listed)

    # accepts a JSON string body too
    assert admin.post("/dashboard/api/codex/accounts",
                       json={"name": "two", "auth_json": json.dumps(auth_json)}).status_code == 200
    # invalid: no access_token -> 400
    assert admin.post("/dashboard/api/codex/accounts",
                       json={"name": "bad", "auth_json": {"tokens": {"refresh_token": "r"}}}).status_code == 400

    # delete
    assert admin.delete("/dashboard/api/codex/accounts/team-one").status_code == 200
    assert admin.delete("/dashboard/api/codex/accounts/team-one").status_code == 404


def test_dashboard_html_has_codex_account_ui(monkeypatch):
    html = _dashboard_client(monkeypatch).get("/dashboard").text
    for needle in ("Codex accounts", "/dashboard/api/codex/accounts",
                   "addCodexAccount", "loadCodexAccounts",
                   "/dashboard/api/codex/invites", "generateCodexInvite",
                   "revokeCodexInvite", "codexInvites", "Invite via link"):
        assert needle in html, needle


# ---- Catalog tab: SKILL.md authoring export -------------------------------

_SKILL_MARKET = {"families": [{
    "family": "gpt-5.5", "quality": 0.9, "sellers_total": 3,
    "meta": {"bench_intelligence": 0.602, "bench_intelligence_rank": 1,
             "cap_tools": True, "in_image": True},
    "rows": [{"price_in": 1.0, "price_out": 5.0},
             {"price_in": 0.6, "price_out": 3.0}]}]}

# Shape of /x/fields → host.field_schema(): the authoritative live vocabulary.
_SKILL_FIELDS = [
    {"name": "price_in", "sort": "Num", "group": "provider", "core": True},
    {"name": "quality", "sort": "Num", "group": "model", "core": True},
    {"name": "bench_coding", "sort": "Num", "group": "model", "core": False},
    {"name": "cap_tools", "sort": "Bool", "group": "model", "core": False},
]


def test_catalog_table_and_skill_render_inject_live_data():
    table = auth_proxy._catalog_table_markdown(_SKILL_MARKET)
    assert "`gpt-5.5`" in table
    assert "60 (#1)" in table          # intelligence 0.602 -> 60, rank 1
    assert "image" in table and "tools" in table
    assert "| 0.6 |" in table          # cheapest input price across sellers
    skill = auth_proxy._render_skill(_SKILL_MARKET, _SKILL_FIELDS)
    assert auth_proxy.SKILL_MARKER not in skill   # marker replaced by the table
    assert auth_proxy.FIELDS_MARKER not in skill  # vocabulary marker replaced too
    assert "policy_ir" in skill                   # the authoring guide is present
    assert "`gpt-5.5`" in skill                   # with the live table baked in
    # graceful when the router is unreachable
    assert "unavailable" in auth_proxy._catalog_table_markdown(None)


def test_field_vocabulary_derived_from_live_schema():
    vocab = auth_proxy._field_vocabulary_markdown(_SKILL_FIELDS)
    # every live field appears, tagged core vs host — nothing hardcoded
    assert "`price_in`" in vocab and "core" in vocab
    assert "`bench_coding`" in vocab and "host" in vocab
    # core fields sort before host extensions
    assert vocab.index("`price_in`") < vocab.index("`bench_coding`")
    # a field absent from the core schema cannot appear (no stale copy)
    assert "`tier_eq`" not in vocab
    # the guide bakes the live vocabulary in, not a static list
    skill = auth_proxy._render_skill(_SKILL_MARKET, _SKILL_FIELDS)
    assert "`bench_coding`" in skill and "| Sort | Scope | Group |" in skill
    # graceful when the schema is unavailable
    assert "unavailable" in auth_proxy._field_vocabulary_markdown(None)


def test_dashboard_skill_endpoint_downloads_markdown(monkeypatch):
    async def _fake_market():
        return _SKILL_MARKET

    async def _fake_fields():
        return _SKILL_FIELDS
    monkeypatch.setattr(auth_proxy, "_fetch_live_market", _fake_market)
    monkeypatch.setattr(auth_proxy, "_fetch_live_fields", _fake_fields)
    r = _dashboard_client(monkeypatch).get("/dashboard/api/skill")
    assert r.status_code == 200
    assert "text/markdown" in r.headers["content-type"]
    assert "filename=SKILL.md" in r.headers["content-disposition"]
    assert "`gpt-5.5`" in r.text and "policy_ir" in r.text
    assert "`bench_coding`" in r.text   # live field vocabulary baked in too


def test_dashboard_skill_endpoint_requires_auth():
    assert TestClient(auth_proxy.app).get("/dashboard/api/skill").status_code == 401


def test_catalog_tab_renamed_and_skill_is_its_own_tab(monkeypatch):
    html = _dashboard_client(monkeypatch).get("/dashboard").text
    assert ">Catalog<" in html                 # tab renamed from "Market"
    assert "marketSkill" not in html           # the Catalog SKILL button moved out
    assert "tabSkill" in html and "skillPage" in html  # SKILL.md is its own side-menu tab
    assert "/dashboard/api/skill" in html      # the tab loads + downloads the live SKILL.md


def test_consumer_skill_endpoint_authed_by_consumer_key(monkeypatch):
    # /skill serves the SAME SKILL.md as /dashboard/api/skill (same renderer),
    # but authenticated by a CONSUMER KEY (Bearer) instead of a dashboard session,
    # so an agent can fetch it with the credential it already calls /v1/* with.
    async def _m():
        return {}

    async def _f():
        return []

    monkeypatch.setattr(auth_proxy, "_fetch_live_market", _m)
    monkeypatch.setattr(auth_proxy, "_fetch_live_fields", _f)
    monkeypatch.setattr(auth_proxy, "_consumer_meta", lambda c: {"status": "active"})
    client = TestClient(auth_proxy.app)

    # no key / bad key -> 401 (reuses the same key auth as /v1/*)
    assert client.get("/skill").status_code == 401
    assert client.get("/skill", headers={"Authorization": "Bearer nope"}).status_code == 401

    # a valid consumer key -> 200, the markdown download
    r = client.get("/skill", headers={"Authorization": "Bearer internal"})
    assert r.status_code == 200
    assert "text/markdown" in r.headers["content-type"]
    assert "filename=SKILL.md" in r.headers["content-disposition"]


def test_cost_accuracy_rows_flags_drift_against_raw_metrics():
    """The cost-accuracy join: measured spend vs advertised list from raw metrics,
    drift-flagged per provider."""
    ema = {
        "openrouter|gpt-5.5": {"price_in": 2.0, "price_out": 10.0},   # list 2/10
        "openai|gpt-5.4": {"price_in": 2.0, "price_out": 10.0},
        "disc|fam": {"price_in": 2.0, "price_out": 10.0},
    }
    routes = [
        # openrouter REPORTS its cost (n_reported=calls): billed 15.0 vs expected
        # 12.0 -> +25% drift, real signal, warns
        {"provider": "openrouter", "family": "gpt-5.5", "calls": 100, "n_reported": 100,
         "tokens_in": 1_000_000, "tokens_out": 1_000_000, "tokens_cached": 0, "cost_usd": 15.0},
        # openai cost is COMPUTED (n_reported=0): billed exactly list -> ~1.0, derived
        {"provider": "openai", "family": "gpt-5.4", "calls": 100, "n_reported": 0,
         "tokens_in": 1_000_000, "tokens_out": 1_000_000, "tokens_cached": 0, "cost_usd": 12.0},
        # provider with any ranking multiplier: stored list price is raw, cost matches
        {"provider": "disc", "family": "fam", "calls": 50, "n_reported": 0,
         "tokens_in": 1_000_000, "tokens_out": 1_000_000, "tokens_cached": 0, "cost_usd": 12.0},
        # no ranked price -> skipped entirely
        {"provider": "bedrock", "family": "nope", "calls": 100, "n_reported": 0,
         "tokens_in": 1_000_000, "tokens_out": 0, "tokens_cached": 0, "cost_usd": 9.9},
    ]
    rows = {r["provider"]: r for r in auth_proxy._cost_accuracy_rows(
        routes, ema, lambda _p: 1.0)}
    assert "bedrock" not in rows                          # unpriced -> not compared
    assert rows["openrouter"]["signal"] == "reported" and rows["openrouter"]["warn"] is True
    assert rows["openrouter"]["deviation"] == 1.25
    assert rows["openrouter"]["measured_usd_per_mtok"] == 7.5   # 15 / 2 Mtok
    assert rows["openai"]["signal"] == "derived" and rows["openai"]["warn"] is False
    assert rows["disc"]["deviation"] == 1.0              # raw list price used directly
    # sorted by |deviation - 1| desc -> the drifting provider first
    assert auth_proxy._cost_accuracy_rows(routes, ema, lambda _p: 1.0)[0]["provider"] == "openrouter"


def test_cost_accuracy_rows_does_not_warn_a_derived_provider():
    # a compute-from-price provider (n_reported=0) with a big apparent drift is
    # reprice noise, not signal -> labelled derived and never warned (no
    # non-actionable badge that trains the operator to ignore drift).
    ema = {"anthropic|claude": {"price_in": 5.0, "price_out": 25.0}}
    routes = [{"provider": "anthropic", "family": "claude", "calls": 500, "n_reported": 0,
               "tokens_in": 1_000_000, "tokens_out": 1_000_000, "tokens_cached": 0, "cost_usd": 90.0}]
    row = auth_proxy._cost_accuracy_rows(routes, ema, lambda p: 1.0)[0]
    assert abs(row["deviation"] - 1) > 1.5 and row["signal"] == "derived" and row["warn"] is False


def test_cost_accuracy_rows_needs_minimum_calls_to_warn():
    ema = {"openrouter|f": {"price_in": 2.0, "price_out": 10.0}}
    routes = [{"provider": "openrouter", "family": "f", "calls": 5, "n_reported": 5,  # <20
               "tokens_in": 1_000_000, "tokens_out": 1_000_000, "tokens_cached": 0, "cost_usd": 20.0}]
    row = auth_proxy._cost_accuracy_rows(routes, ema, lambda p: 1.0)[0]
    assert row["signal"] == "reported" and row["deviation"] > 1.5 and row["warn"] is False
