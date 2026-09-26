"""Ingress (auth_proxy) security regressions: path traversal to router
internals, streaming budget bypass, fail-closed consumer store reads, router
admin credential, usage/session endpoint hardening, tenant/consumer collision,
and constant-time compares that must not raise on non-ASCII input."""
from __future__ import annotations

import asyncio
import hashlib
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import auth_proxy
import control_plane_client
import host_store
from conftest import require_host_store


def _authed(monkeypatch, caller="tester", meta=None):
    monkeypatch.setattr(auth_proxy, "_caller_auth",
                        lambda token: {"ok": True, "caller": caller, "digest": "d" * 64, "meta": meta or {}})
    monkeypatch.setattr(auth_proxy, "_rate_ok", lambda caller, meta=None: (True, True, 0.0))


# ---- I1: path traversal ------------------------------------------------------

_TRAVERSALS = [
    "/x/calls", "/v1/%2e%2e/x/calls", "/v1/%2E%2E/x/providers", "/%2e/x/calls",
    "/v1/..%2fx/calls", "/v1/..%2Fx/calls", "/v1/%2e%2e%2fx/calls", "/./x/calls",
    "/a/../x/calls", "/v1/..%5cx/calls", "//x/calls", "/v1//chat/completions",
    "/v1/%252e%252e/x/calls", "/%2e%2e/x/calls", "/X/calls", "/dashboard/../x/calls",
]


@pytest.mark.parametrize("raw", [
    b"/./x/calls", b"/a/../x/calls", b"/v1/../x/calls", b"/v1/./chat/completions",
    b"/v1/chat%2Fcompletions", b"/v1/..\\x/calls", b"//x/calls", b"/v1/chat/completions/..",
])
def test_proxy_path_refuses_raw_dot_and_encoded_segments(raw):
    # httpx clients normalize dot segments before sending; a raw socket does not.
    from starlette.requests import Request
    from urllib.parse import unquote
    scope = {"type": "http", "method": "POST", "headers": [], "query_string": b"",
             "raw_path": raw, "path": unquote(raw.decode())}
    assert auth_proxy._proxy_path(Request(scope)) is None


@pytest.mark.parametrize("path", _TRAVERSALS)
def test_traversal_never_reaches_router_internals(monkeypatch, path):
    _authed(monkeypatch)
    seen = []

    def handler(req: httpx.Request):
        seen.append(req.url.raw_path.decode())
        return httpx.Response(200, json={"reached": req.url.path})

    monkeypatch.setattr(auth_proxy, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    r = TestClient(auth_proxy.app).post(path, headers={"Authorization": "Bearer k"}, json={"id": "evil"})
    assert r.status_code in (400, 404), (path, r.status_code, r.text)
    assert seen == [], (path, seen)


def test_proxy_forwards_normalized_allowed_paths(monkeypatch):
    _authed(monkeypatch)
    monkeypatch.setattr(host_store, "consumer_spend_usd", lambda caller: (0.0, True))
    seen = []

    def handler(req: httpx.Request):
        seen.append(req.url.raw_path.decode())
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(auth_proxy, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    c = TestClient(auth_proxy.app)
    for p in ("/v1/chat/completions", "/v1/models", "/default/v1/chat/completions", "/api/tags"):
        assert c.post(p, headers={"Authorization": "Bearer k"}, json={}).status_code == 200, p
    assert seen == ["/v1/chat/completions", "/v1/models", "/default/v1/chat/completions", "/api/tags"]


# ---- I3/I4: consumer store read failures fail closed --------------------------

class _Up:
    async def get(self, url, timeout=None, headers=None):
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("GET", url))


def test_consumer_dashboard_fails_closed_when_consumer_store_unreadable(monkeypatch, tmp_path):
    require_host_store()
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET", "s")
    monkeypatch.setattr(auth_proxy, "_client", _Up())
    monkeypatch.setattr(auth_proxy, "DASHBOARD_KEY_ENV_PATH", str(tmp_path / ".env"))
    tok = "llmr_alice"
    dig = hashlib.sha256(tok.encode()).hexdigest()
    assert host_store.set_consumer_keys({"alice": {"status": "active", "keys": [
        {"sha256_prefix": dig[:12], "status": "active"}]}}, key_digests={dig: "alice"})
    c = TestClient(auth_proxy.app)
    c.cookies.set(auth_proxy.DASHBOARD_COOKIE_NAME, auth_proxy._make_dashboard_session(
        "alice", role="consumer", consumer="alice", key_sha256=dig))
    auth_proxy._record_request(caller="victim-co", status=200, requested_model="profile:secret",
                               error_message="victim prompt", key_sha256="b" * 64, method="POST",
                               path="/v1/chat/completions", latency_ms=1)
    d = c.get("/dashboard/api/stats?timeframe=runtime").json()
    assert d["selected_consumer"] == "alice"
    assert "victim-co" not in {x.get("caller") for x in d["recent"]}
    monkeypatch.setattr(host_store, "get_consumer_keys", lambda: ({}, False))
    for path in ("/dashboard/api/stats?timeframe=runtime", "/dashboard/api/full"):
        r = c.get(path)
        assert r.status_code == 503, (path, r.status_code, r.text[:200])
        assert "victim" not in r.text
    # Even if the pre-check races a later failure, the snapshot stays scoped.
    snap = auth_proxy._stats_snapshot(viewer="consumer:alice", upstream_status=200, upstream_health={},
                                      consumer="alice", timeframe="runtime", key_sha256=dig,
                                      viewer_role="consumer")
    assert snap["selected_consumer"] == "alice"
    assert "victim-co" not in {x.get("caller") for x in snap["recent"]}


def test_consumer_record_writes_abort_on_store_read_failure(monkeypatch, tmp_path):
    require_host_store()
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET", "s")
    monkeypatch.setattr(auth_proxy, "DASHBOARD_KEY_ENV_PATH", str(tmp_path / ".env"))
    old = "llmr_old"
    od = hashlib.sha256(old.encode()).hexdigest()
    assert host_store.set_consumer_keys({"bob": {
        "status": "inactive", "budget_usd": 1.0, "allowed_routes": ["profile:cheap"],
        "keys": [{"sha256_prefix": od[:12], "status": "active", "expires_at": int(time.time()) - 10}]}},
        key_digests={od: "bob"})
    real = host_store.get_consumer_keys
    c = TestClient(auth_proxy.app)
    c.cookies.set(auth_proxy.DASHBOARD_COOKIE_NAME, auth_proxy._make_dashboard_session("admin"))
    writes = [
        ("/dashboard/api/keys", {"consumer": "carol"}),
        ("/dashboard/api/keys/batch", {"batch": "b1", "count": 2, "budget_usd": 1}),
        ("/dashboard/api/consumers/bob", {"status": "active"}),
        ("/dashboard/api/keys/revoke", {"consumer": "bob", "sha256_prefix": od[:12]}),
    ]
    for path, body in writes:
        monkeypatch.setattr(host_store, "get_consumer_keys", lambda: ({}, False))
        r = c.post(path, json=body)
        monkeypatch.setattr(host_store, "get_consumer_keys", real)
        assert r.status_code == 503, (path, r.status_code, r.text[:200])
        records = real()[0]
        assert set(records) == {"bob"}, (path, records)
        assert records["bob"]["status"] == "inactive" and records["bob"]["budget_usd"] == 1.0
    assert auth_proxy._caller_auth(old).get("ok") is not True


# ---- M2 (client side): operator /x/* calls carry the router admin secret ------

def test_router_admin_calls_send_internal_secret_and_proxy_traffic_does_not(monkeypatch):
    monkeypatch.setattr(control_plane_client, "CONTROL_PLANE_INTERNAL_SECRET", "sek")
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET", "s")
    _authed(monkeypatch)
    monkeypatch.setattr(host_store, "consumer_spend_usd", lambda caller: (0.0, True))
    seen = {}

    def handler(req: httpx.Request):
        seen[req.url.path] = req.headers.get("x-internal-secret")
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(auth_proxy, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    admin = TestClient(auth_proxy.app)
    admin.cookies.set(auth_proxy.DASHBOARD_COOKIE_NAME, auth_proxy._make_dashboard_session("admin"))
    admin.get("/dashboard/api/wallet")
    admin.post("/dashboard/api/wallet/refresh", json={})
    TestClient(auth_proxy.app).post("/v1/chat/completions", headers={
        "Authorization": "Bearer k", "x-internal-secret": "forged"}, json={})
    assert seen["/x/wallet"] == "sek"
    assert seen["/x/wallet/refresh"] == "sek"
    assert seen["/v1/chat/completions"] is None
    assert auth_proxy._router_admin_headers({"a": "b"}) == {"a": "b", "x-internal-secret": "sek"}
    monkeypatch.setattr(control_plane_client, "CONTROL_PLANE_INTERNAL_SECRET", "")
    assert auth_proxy._router_admin_headers() == {}


# ---- S7 + S1: /v1/session and /v1/usage --------------------------------------

@pytest.mark.parametrize("sid", ["..", "%2e%2e", "a b", "x" * 129, ".hidden"])
def test_session_view_rejects_malformed_sid(monkeypatch, sid):
    _authed(monkeypatch)
    seen = []
    monkeypatch.setattr(auth_proxy, "_client", httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: seen.append(req.url.raw_path) or httpx.Response(200, json={}))))
    r = TestClient(auth_proxy.app).get(f"/v1/session/{sid}", headers={"Authorization": "Bearer k"})
    assert r.status_code in (400, 404), (sid, r.status_code)
    assert seen == []


def test_session_view_forwards_valid_sid_with_caller_and_secret(monkeypatch):
    monkeypatch.setattr(control_plane_client, "CONTROL_PLANE_INTERNAL_SECRET", "sek")
    _authed(monkeypatch, caller="alice")
    seen = []

    def handler(req):
        seen.append((req.url.raw_path.decode(), req.headers.get("x-llm-router-caller"),
                     req.headers.get("x-internal-secret")))
        return httpx.Response(200, json={"calls": 1})

    monkeypatch.setattr(auth_proxy, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    r = TestClient(auth_proxy.app).get("/v1/session/ses_01J:ab-c.d", headers={"Authorization": "Bearer k"})
    assert r.status_code == 200
    assert seen == [("/x/session/ses_01J%3Aab-c.d", "alice", "sek")]


def test_usage_and_session_endpoints_are_rate_limited(monkeypatch):
    _authed(monkeypatch)
    monkeypatch.setattr(auth_proxy, "_rate_ok", lambda caller, meta=None: (False, True, 3.0))
    monkeypatch.setattr(auth_proxy, "_client", httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={}))))
    c = TestClient(auth_proxy.app)
    for path in ("/v1/usage", "/api/usage", "/v1/session/ses_1"):
        r = c.get(path, headers={"Authorization": "Bearer k"})
        assert r.status_code == 429, (path, r.status_code)
        assert r.headers["Retry-After"] == "3"
    monkeypatch.setattr(auth_proxy, "_rate_ok", lambda caller, meta=None: (True, False, 0.0))
    assert c.get("/v1/usage", headers={"Authorization": "Bearer k"}).status_code == 503


def test_key_usage_reads_only_the_key_in_sql_off_the_event_loop(monkeypatch):
    import threading
    require_host_store()
    mine, other = "a" * 64, "b" * 64
    now = int(time.time())
    for i, sha in enumerate([mine, other, other, mine]):
        host_store.insert_call({"ts": now - 10 + i, "usage_event_id": f"e{i}", "key_sha256": sha,
                                "caller": "tester", "status": 200, "tokens_in": 1, "tokens_out": 1,
                                "provider": "p", "model_family": "f"})
    _authed(monkeypatch)
    monkeypatch.setattr(auth_proxy, "_caller_auth", lambda token: {
        "ok": True, "caller": "tester", "digest": mine, "meta": {}})

    def boom(*a, **k):
        raise AssertionError("whole-caller history read")

    monkeypatch.setattr(host_store, "usage_rows", boom)
    threads = []
    real = auth_proxy._key_usage_snapshot
    monkeypatch.setattr(auth_proxy, "_key_usage_snapshot",
                        lambda **kw: threads.append(threading.current_thread()) or real(**kw))
    r = TestClient(auth_proxy.app).get("/v1/usage", headers={"Authorization": "Bearer k"})
    assert r.status_code == 200, r.text
    assert r.json()["totals"]["requests"] == 2
    assert threads and threads[0] is not threading.main_thread()


# ---- S3 (ingress side): env-secrets writes -----------------------------------

def _admin(monkeypatch):
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET", "s")
    c = TestClient(auth_proxy.app)
    c.cookies.set(auth_proxy.DASHBOARD_COOKIE_NAME, auth_proxy._make_dashboard_session("admin"))
    return c


def test_provider_key_writes_refuse_injection_and_infra_names(monkeypatch, tmp_path):
    monkeypatch.setattr(auth_proxy, "_load_policy_config", lambda: {"providers": {
        "heurist": {"auth_env": "HEURIST_API_KEY"},
        "evil": {"auth_env": "CONTROL_PLANE_INTERNAL_SECRET"}}})
    env_file = tmp_path / ".env.secrets"
    monkeypatch.setattr(auth_proxy, "DASHBOARD_KEY_ENV_PATH", str(env_file))
    admin = _admin(monkeypatch)
    for payload in ({"provider": "heurist", "key": "sk\nCONTROL_PLANE_INTERNAL_SECRET=owned"},
                    {"provider": "heurist", "key": "sk\rX=1"}, {"provider": "heurist", "key": "a\x00b"},
                    {"provider": "evil", "key": "sk"}):
        r = admin.post("/dashboard/api/provider-keys/update", json=payload)
        assert r.status_code == 400, (payload, r.status_code, r.text)
    assert not env_file.exists()
    for auth_env, key in (("CONTROL_PLANE_INTERNAL_SECRET", "x"), ("DATABASE_URL", "x"),
                          ("NEW_API_KEY", "a\nDATABASE_URL=postgres://evil")):
        r = admin.post("/dashboard/api/provider-keys/add", json={
            "id": "newprov", "base_url": "https://api.example.com/v1", "auth_env": auth_env, "key": key,
            "served_models": [{"model_family": "m", "served_model_id": "m"}]})
        assert r.status_code == 400, (auth_env, r.status_code, r.text)
    assert not env_file.exists()
    with pytest.raises(ValueError):
        auth_proxy._upsert_env_line(env_file, "HEURIST_API_KEY", "a\nB=c")
    assert admin.post("/dashboard/api/provider-keys/update",
                      json={"provider": "heurist", "key": "sk-ok"}).status_code == 200
    assert env_file.read_text() == "HEURIST_API_KEY=sk-ok\n"


# ---- S8: constant-time compares never 500 on non-ASCII input ------------------

def test_secret_compares_reject_non_ascii_without_500(monkeypatch):
    monkeypatch.setattr(control_plane_client, "CONTROL_PLANE_INTERNAL_SECRET", "sek")
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET", "s")
    monkeypatch.setattr(auth_proxy, "DASHBOARD_TRUSTED_USER_HEADER", "x-user")
    monkeypatch.setattr(auth_proxy, "DASHBOARD_TRUSTED_USER_SECRET", "trusted")
    c = TestClient(auth_proxy.app, raise_server_exceptions=False)
    r = c.get("/internal/usage?caller=a", headers={"x-internal-secret": "s\xe9k".encode("latin-1")})
    assert r.status_code == 403
    r = c.get("/dashboard/api/stats", headers={"x-user": "op", "x-dashboard-trusted-secret": "\xe9".encode("latin-1")})
    assert r.status_code == 401
    r = c.get("/dashboard/api/stats", headers={
        "cookie": f"{auth_proxy.DASHBOARD_COOKIE_NAME}=e30.\xe9\xe9".encode("latin-1")})
    assert r.status_code == 401
    assert control_plane_client.secret_equal("é", "é") and not control_plane_client.secret_equal("é", "e")


# ---- S11: /internal/usage window validation ----------------------------------

@pytest.mark.parametrize("query", ["since_ts=200&until_ts=100", "since_ts=-1",
                                   "until_ts=99999999999999999999", "since_ts=253402300800"])
def test_internal_usage_rejects_invalid_windows(monkeypatch, query):
    monkeypatch.setattr(control_plane_client, "CONTROL_PLANE_INTERNAL_SECRET", "sek")
    called = []
    monkeypatch.setattr(host_store, "usage_totals", lambda **kw: called.append(kw) or {})
    r = TestClient(auth_proxy.app).get(f"/internal/usage?caller=a&{query}", headers={"x-internal-secret": "sek"})
    assert r.status_code == 400, (query, r.status_code, r.text)
    assert called == []


# ---- S2: tenant slug colliding with a local consumer fails closed -------------

@pytest.mark.parametrize("local", ["issued", "env_plain", "env_hash"])
def test_control_plane_caller_colliding_with_local_consumer_is_rejected(monkeypatch, local):
    require_host_store()
    monkeypatch.setattr(control_plane_client, "enabled", lambda: True)

    async def resolve(digest):
        return control_plane_client.ResolvedKey(active=True, consumer="acme", tenant_id=7,
                                                rate_per_min=None, burst=None, fetched_at=0.0)

    monkeypatch.setattr(control_plane_client, "resolve_key", resolve)
    if local == "issued":
        dig = hashlib.sha256(b"llmr_local").hexdigest()
        host_store.set_consumer_keys({"acme": {"status": "active", "keys": [
            {"sha256_prefix": dig[:12], "status": "active"}]}}, key_digests={dig: "acme"})
    elif local == "env_plain":
        monkeypatch.setattr(auth_proxy, "CALLER_KEYS", {"llmr_local": "acme"})
    else:
        monkeypatch.setattr(auth_proxy, "CALLER_KEY_HASHES", {"f" * 64: "acme"})
    auth = asyncio.run(auth_proxy._caller_auth_async("tok-tenant"))
    assert auth["ok"] is False and auth["error_code"] == "caller_name_collision"


def test_control_plane_caller_without_local_keys_still_resolves(monkeypatch):
    require_host_store()
    monkeypatch.setattr(control_plane_client, "enabled", lambda: True)

    async def resolve(digest):
        return control_plane_client.ResolvedKey(active=True, consumer="acme", tenant_id=7,
                                                rate_per_min=5, burst=5, fetched_at=0.0)

    monkeypatch.setattr(control_plane_client, "resolve_key", resolve)
    host_store.set_consumer_keys({"acme": {"status": "active", "rate_per_min": 9}})  # override, no keys
    auth = asyncio.run(auth_proxy._caller_auth_async("tok-tenant"))
    assert auth["ok"] is True and auth["caller"] == "acme" and auth["meta"]["rate_per_min"] == 9


# ---- I2: a client that disconnects mid-stream still pays ---------------------

def _serve(app):
    import socket
    import threading
    import uvicorn
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="off"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    while not srv.started:
        time.sleep(0.02)
    return srv, t, port


def _usage_sse(filler: bytes = b""):
    async def body():
        yield b'data: {"choices":[{"delta":{"content":"THE ANSWER <<END>>"}}]}\n\n'
        await asyncio.sleep(0.8)   # the provider is still finishing; usage arrives last
        if filler:
            yield filler
        yield (b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":100000,'
               b'"completion_tokens":5000,"total_tokens":105000},"x_router":{"provider":"p",'
               b'"model_family":"f","cost_usd":7.5}}\n\n')
        yield b"data: [DONE]\n\n"
    return body


def _disconnect_after_answer(port, tok):
    with httpx.Client(timeout=10) as c:
        with c.stream("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                      headers={"Authorization": f"Bearer {tok}"},
                      json={"model": "profile:default", "stream": True, "messages": []}) as r:
            for chunk in r.iter_raw():
                if b"<<END>>" in chunk:
                    break


def _budgeted_consumer(name, tok, budget=1.0):
    dig = hashlib.sha256(tok.encode()).hexdigest()
    assert host_store.set_consumer_keys({name: {"status": "active", "budget_usd": budget, "keys": [
        {"sha256_prefix": dig[:12], "status": "active"}]}}, key_digests={dig: name})


def _wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_stream_disconnect_before_usage_is_still_metered(monkeypatch):
    require_host_store()
    tok = "llmr_budgeted"
    _budgeted_consumer("hack-001", tok)
    body = _usage_sse()
    monkeypatch.setattr(auth_proxy, "_client", httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body()))))
    srv, t, port = _serve(auth_proxy.app)
    try:
        _disconnect_after_answer(port, tok)
        assert _wait_for(lambda: host_store.consumer_spend_usd("hack-001")[0] >= 7.5), \
            host_store.consumer_spend_usd("hack-001")
        with httpx.Client(timeout=10) as c:
            r = c.post(f"http://127.0.0.1:{port}/v1/chat/completions", headers={"Authorization": f"Bearer {tok}"},
                       json={"model": "profile:default", "stream": True, "messages": []})
        assert r.status_code == 402, r.text
    finally:
        srv.should_exit = True
        t.join(5)


def test_stream_cut_off_by_drain_cap_books_an_estimate_not_zero(monkeypatch):
    require_host_store()
    monkeypatch.setattr(auth_proxy, "STREAM_DRAIN_MAX_BYTES", 16)
    tok = "llmr_budgeted2"
    _budgeted_consumer("hack-002", tok)
    body = _usage_sse(filler=b": " + b"x" * 4096 + b"\n\n")
    monkeypatch.setattr(auth_proxy, "_client", httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body()))))
    srv, t, port = _serve(auth_proxy.app)
    try:
        _disconnect_after_answer(port, tok)
        assert _wait_for(lambda: host_store.recent_calls(limit=5, caller="hack-002"))
        row = host_store.recent_calls(limit=5, caller="hack-002")[0]
        assert row["cost_usd"] == pytest.approx(auth_proxy.CLOUD_BUDGET_RESERVATION_USD)
        assert host_store.consumer_spend_usd("hack-002")[0] > 0
    finally:
        srv.should_exit = True
        t.join(5)


def test_stream_completed_normally_is_forwarded_whole_and_metered(monkeypatch):
    require_host_store()
    tok = "llmr_budgeted3"
    _budgeted_consumer("hack-003", tok, budget=100.0)
    body = _usage_sse()
    monkeypatch.setattr(auth_proxy, "_client", httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body()))))
    r = TestClient(auth_proxy.app).post("/v1/chat/completions", headers={"Authorization": f"Bearer {tok}"},
                                        json={"model": "profile:default", "stream": True, "messages": []})
    assert r.status_code == 200 and r.text.endswith("data: [DONE]\n\n") and "<<END>>" in r.text
    assert _wait_for(lambda: host_store.consumer_spend_usd("hack-003")[0] == 7.5)
