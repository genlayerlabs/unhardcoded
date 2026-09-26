"""Control-plane key limits: subject rate/budget enforcement at the ingress,
the reservation ledger, and the reconciliation surface (/internal/usage
group_by + until_ts, /internal/usage/export, /internal/budgets).

The control plane is faked at its HTTP client (or at resolve_route), the
router at auth_proxy._client; budget/ledger state is the shared test Postgres.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CALLER_KEYS_JSON", '{"internal":"default"}')
os.environ.setdefault("CALLER_KEYS_SHA256_JSON", "{}")

import auth_proxy  # noqa: E402
import control_plane_client as cpc  # noqa: E402
import host_store  # noqa: E402

from conftest import require_host_store  # noqa: E402

SECRET = {"x-internal-secret": "s3cret"}
ROUTE = {"route": "route:production", "revision": 2, "policy_id": "a" * 64,
         "policy_ir": ["policy"], "execution": {}}
KEY = {"id": 42, "subject": "g:7", "labels": {"customer": "c-1"}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    cpc.reset_for_tests()
    monkeypatch.setattr(cpc, "CONTROL_PLANE_URL", "http://cp.test")
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "s3cret")
    yield
    cpc.reset_for_tests()


# ---- fakes -------------------------------------------------------------------

class _JsonResp:
    def __init__(self, cost=0.02, status_code=200):
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self._body = json.dumps({"ok": True, "x_router": {"cost_usd": cost, "provider": "p"}}).encode()

    async def aread(self):
        return self._body

    async def aclose(self):
        pass


class _StreamResp:
    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self, cost=0.03, fail_mid_stream=False):
        self.cost, self.fail = cost, fail_mid_stream
        self.closed = False

    async def aiter_raw(self):
        yield b'data: {"choices":[]}\n\n'
        if self.fail:
            raise httpx.ReadError("upstream reset")
        yield ('data: ' + json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1},
                                      "x_router": {"cost_usd": self.cost}}) + "\n\n").encode()
        yield b"data: [DONE]\n\n"

    async def aclose(self):
        self.closed = True


class _Upstream:
    def __init__(self, make=lambda: _JsonResp()):
        self.make, self.requests = make, []

    def build_request(self, method, url, content=None, headers=None):
        self.requests.append({"headers": headers or {}, "body": content})
        return object()

    async def send(self, req, stream=True):
        resp = self.make()
        if isinstance(resp, Exception):
            raise resp
        return resp


def _saas(monkeypatch, *, limits, key=KEY, upstream=None, tenant=5, caller="app-a"):
    """A resolved SaaS key whose published route carries `key`/`limits`."""
    async def auth(token):
        return {"ok": True, "caller": caller, "tenant_id": tenant,
                "digest": hashlib.sha256(token.encode()).hexdigest(), "meta": {}}

    async def route(*args, **kwargs):
        return {**ROUTE, "key": key, "limits": limits}
    monkeypatch.setattr(auth_proxy, "_caller_auth_async", auth)
    monkeypatch.setattr(cpc, "resolve_route", route)
    up = upstream or _Upstream()
    monkeypatch.setattr(auth_proxy, "_client", up)
    return up


def _post(client, token="tok-a"):
    return client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token}"},
                       json={"model": "route:production", "messages": []})


def _period():
    return time.strftime("%Y-%m", time.gmtime())


def _budget(subject="g:7", tenant=5):
    auth_proxy.drain_settles()
    return host_store.subject_budgets(tenant, _period(), [subject])[0]


def _live_reservations():
    auth_proxy.drain_settles()
    with host_store._get_pool().connection() as conn:
        return conn.execute("SELECT count(*) FROM subject_budget_reservations"
                            " WHERE NOT released").fetchone()[0]


# ---- resolve_route: key/limits parsing -----------------------------------------

def _resolve_with(monkeypatch, extra):
    def handler(request):
        return httpx.Response(200, json={**ROUTE, **extra})
    monkeypatch.setattr(cpc, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return asyncio.run(cpc.resolve_route(3, "production"))


def test_resolve_route_parses_key_limits(monkeypatch):
    data = _resolve_with(monkeypatch, {"key": KEY, "limits": {
        "monthly_budget_usd": "40.00", "rate_per_min": 120, "burst": 40}})
    assert data["key"] == KEY
    assert data["limits"] == {"monthly_budget_usd": 40.0, "rate_per_min": 120, "burst": 40}
    data = _resolve_with(monkeypatch, {"key": {"id": 9, "subject": "k:9"}, "limits": {
        "monthly_budget_usd": 2.5, "rate_per_min": None, "burst": None}})
    assert data["key"] == {"id": 9, "subject": "k:9", "labels": {}}
    assert data["limits"] == {"monthly_budget_usd": 2.5, "rate_per_min": None, "burst": None}


def test_resolve_route_without_key_limits_is_unlimited(monkeypatch):
    data = _resolve_with(monkeypatch, {})
    assert data["key"] is None and data["limits"] is None


@pytest.mark.parametrize("extra", [
    {"limits": {"rate_per_min": 5}},                                    # limits, no key
    {"key": {**KEY, "subject": "x:7"}, "limits": {}},
    {"key": {**KEY, "subject": "g:07"}, "limits": {}},
    {"key": {**KEY, "subject": "g:0"}, "limits": {}},
    {"key": {**KEY, "subject": "g:1234567890123456789"}, "limits": {}},  # 19 digits
    {"key": {**KEY, "id": "42"}, "limits": {}},
    {"key": {**KEY, "id": True}, "limits": {}},
    {"key": {**KEY, "labels": {"a": 1}}, "limits": {}},
    {"key": {**KEY, "labels": {f"k{i}": "v" for i in range(33)}}, "limits": {}},
    {"key": {**KEY, "labels": {"k": "v" * 257}}, "limits": {}},
    {"key": KEY, "limits": {"monthly_budget_usd": "0"}},
    {"key": KEY, "limits": {"monthly_budget_usd": "-1"}},
    {"key": KEY, "limits": {"monthly_budget_usd": "1e3"}},
    {"key": KEY, "limits": {"monthly_budget_usd": True}},
    {"key": KEY, "limits": {"monthly_budget_usd": 0}},
    {"key": KEY, "limits": {"rate_per_min": 0}},
    {"key": KEY, "limits": {"rate_per_min": 1.5}},
    {"key": KEY, "limits": {"burst": "10"}},
    {"key": KEY, "limits": ["bad"]},
    {"key": "g:7"},
])
def test_resolve_route_invalid_key_limits_fail_closed(monkeypatch, extra):
    with pytest.raises(cpc.RouteUnavailable):
        _resolve_with(monkeypatch, extra)


def test_invalid_limits_reject_request_before_router(monkeypatch):
    require_host_store()
    up = _Upstream()
    monkeypatch.setattr(auth_proxy, "_client", up)

    async def auth(token):
        return {"ok": True, "caller": "app-a", "tenant_id": 5, "digest": "a" * 64, "meta": {}}
    monkeypatch.setattr(auth_proxy, "_caller_auth_async", auth)

    def handler(request):
        return httpx.Response(200, json={**ROUTE, "key": {**KEY, "subject": "bogus"}})
    monkeypatch.setattr(cpc, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    r = _post(TestClient(auth_proxy.app))
    assert r.status_code == 503 and r.json()["error"]["code"] == "route_unavailable"
    assert up.requests == []


# ---- ingress enforcement -------------------------------------------------------

def test_subject_rate_limit_429_after_plan_rate(monkeypatch):
    require_host_store()
    _saas(monkeypatch, limits={"monthly_budget_usd": None, "rate_per_min": 1, "burst": None})
    client = TestClient(auth_proxy.app)
    assert _post(client).status_code == 200
    r = _post(client)
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "key_rate_limit"
    assert int(r.headers["retry-after"]) >= 1
    # The bucket is per caller+subject: another subject is unaffected.
    _saas(monkeypatch, limits={"monthly_budget_usd": None, "rate_per_min": 1, "burst": None},
          key={**KEY, "subject": "k:43"})
    assert _post(client).status_code == 200


def test_subject_rate_store_unavailable_is_503(monkeypatch):
    require_host_store()
    up = _Upstream()
    _saas(monkeypatch, limits={"monthly_budget_usd": None, "rate_per_min": 5, "burst": 5}, upstream=up)
    real = host_store.consume_rate_token
    monkeypatch.setattr(host_store, "consume_rate_token",
                        lambda consumer, *a, **k: (False, False, 0.0) if "|" in consumer else real(consumer, *a, **k))
    r = _post(TestClient(auth_proxy.app))
    assert r.status_code == 503 and r.json()["error"]["code"] == "key_rate_limit_unavailable"
    assert up.requests == []


def test_budget_settles_actual_cost_and_records_subject(monkeypatch):
    require_host_store()
    _saas(monkeypatch, limits={"monthly_budget_usd": "1.00", "rate_per_min": None, "burst": None})
    assert _post(TestClient(auth_proxy.app)).status_code == 200
    assert _budget() == {"subject": "g:7", "spent_usd": 0.02, "reserved_usd": 0.0}
    assert _live_reservations() == 0
    host_store._write_q.join()
    summary = host_store.recent_calls(caller="app-a")[0]["routing_summary"]
    assert summary["key_id"] == "42" and summary["subject"] == "g:7"


def test_budget_exhausted_is_402_without_forwarding(monkeypatch):
    require_host_store()
    up = _Upstream(lambda: _JsonResp(cost=0.04))
    _saas(monkeypatch, limits={"monthly_budget_usd": "0.10", "rate_per_min": None, "burst": None},
          upstream=up)
    client = TestClient(auth_proxy.app)
    # 0.04 + 0.04 spent; a third call needs 0.08 + 0.05 > 0.10.
    assert _post(client).status_code == 200
    assert _post(client).status_code == 200
    r = _post(client)
    assert r.status_code == 402
    err = r.json()["error"]
    assert err["code"] == "key_budget_exhausted" and err["type"] == "budget_error"
    assert err["budget_usd"] == 0.1 and err["spent_usd"] == pytest.approx(0.08)
    assert len(up.requests) == 2
    # Budgets are per tenant: the same subject in another tenant is fresh.
    _saas(monkeypatch, limits={"monthly_budget_usd": "0.10", "rate_per_min": None, "burst": None},
          upstream=up, tenant=6)
    assert _post(client).status_code == 200


def test_budget_store_unavailable_is_503(monkeypatch):
    require_host_store()
    up = _Upstream()
    _saas(monkeypatch, limits={"monthly_budget_usd": "1", "rate_per_min": None, "burst": None}, upstream=up)
    monkeypatch.setattr(host_store, "reserve_subject_budget", lambda *a, **k: (False, False, 0.0, 0.0))
    r = _post(TestClient(auth_proxy.app))
    assert r.status_code == 503 and r.json()["error"]["code"] == "key_budget_unavailable"
    assert up.requests == []


def test_reservation_released_on_upstream_error_status(monkeypatch):
    require_host_store()
    _saas(monkeypatch, limits={"monthly_budget_usd": "1", "rate_per_min": None, "burst": None},
          upstream=_Upstream(lambda: _JsonResp(cost=None, status_code=500)))
    assert _post(TestClient(auth_proxy.app)).status_code == 500
    assert _live_reservations() == 0
    assert _budget()["spent_usd"] == 0.0


def test_reservation_released_on_upstream_exception(monkeypatch):
    require_host_store()
    _saas(monkeypatch, limits={"monthly_budget_usd": "1", "rate_per_min": None, "burst": None},
          upstream=_Upstream(lambda: httpx.ConnectError("down")))
    r = _post(TestClient(auth_proxy.app))
    assert r.status_code == 502
    assert _live_reservations() == 0 and _budget()["reserved_usd"] == 0.0


def test_reservation_released_on_capacity_failure(monkeypatch):
    require_host_store()
    up = _Upstream()
    _saas(monkeypatch, limits={"monthly_budget_usd": "1", "rate_per_min": None, "burst": None}, upstream=up)

    async def full():
        return False
    monkeypatch.setattr(auth_proxy, "_capacity_acquire", full)
    r = _post(TestClient(auth_proxy.app))
    assert r.status_code == 503 and r.json()["error"]["code"] == "router_overloaded"
    assert _live_reservations() == 0 and up.requests == []


def test_reservation_released_when_request_building_raises(monkeypatch):
    require_host_store()
    up = _Upstream()
    _saas(monkeypatch, limits={"monthly_budget_usd": "1", "rate_per_min": None, "burst": None}, upstream=up)

    def boom(*a, **k):
        raise RuntimeError("bad request build")
    monkeypatch.setattr(up, "build_request", boom)
    assert _post(TestClient(auth_proxy.app)).status_code == 502
    assert _live_reservations() == 0


def test_streaming_settles_after_stream_with_streamed_cost(monkeypatch):
    require_host_store()
    _saas(monkeypatch, limits={"monthly_budget_usd": "1", "rate_per_min": None, "burst": None},
          upstream=_Upstream(lambda: _StreamResp(cost=0.03)))
    r = _post(TestClient(auth_proxy.app))
    assert r.status_code == 200 and b"[DONE]" in r.content
    assert _live_reservations() == 0
    assert _budget()["spent_usd"] == pytest.approx(0.03)


def test_streaming_upstream_failure_mid_stream_releases(monkeypatch):
    require_host_store()
    _saas(monkeypatch, limits={"monthly_budget_usd": "1", "rate_per_min": None, "burst": None},
          upstream=_Upstream(lambda: _StreamResp(fail_mid_stream=True)))
    with pytest.raises(httpx.ReadError):
        _post(TestClient(auth_proxy.app))
    assert _live_reservations() == 0


def _request(token="tok-a"):
    body = json.dumps({"model": "route:production", "messages": []}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}
    return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions",
                    "headers": [(b"authorization", f"Bearer {token}".encode()),
                                (b"content-type", b"application/json")],
                    "query_string": b"", "client": ("test", 1), "server": ("test", 80),
                    "scheme": "http", "root_path": "", "http_version": "1.1"}, receive)


@pytest.mark.parametrize("before_first_chunk", [False, True])
def test_client_disconnect_during_stream_releases(monkeypatch, before_first_chunk):
    """Starlette closes the body iterator on disconnect (or only runs the
    background task if the client left before the first chunk)."""
    require_host_store()
    stream = _StreamResp(cost=0.03)
    _saas(monkeypatch, limits={"monthly_budget_usd": "1", "rate_per_min": None, "burst": None},
          upstream=_Upstream(lambda: stream))

    async def run():
        resp = await auth_proxy.proxy("v1/chat/completions", _request())
        assert _live_reservations() == 1          # held while the stream is open
        if not before_first_chunk:
            await resp.body_iterator.__anext__()
            await resp.body_iterator.aclose()
        await resp.background()
    asyncio.run(run())
    assert stream.closed and _live_reservations() == 0
    assert _budget()["reserved_usd"] == 0.0


def test_routes_without_limits_take_no_reservation(monkeypatch):
    require_host_store()
    _saas(monkeypatch, limits=None, key=None)
    assert _post(TestClient(auth_proxy.app)).status_code == 200
    with host_store._get_pool().connection() as conn:
        assert conn.execute("SELECT count(*) FROM subject_budget_usage").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM consumer_rate_buckets"
                            " WHERE consumer LIKE '%|%'").fetchone()[0] == 0


# ---- revocation ----------------------------------------------------------------

def test_revoked_scoped_key_rejected_on_next_request_despite_cached_resolve(monkeypatch):
    """/internal/keys/resolve is cached positive for CP_RESOLVE_TTL_S, but the
    scoped route lookup carries the key digest and is never cached: once the
    control plane answers 403 for the (revoked) key, the next call is refused."""
    require_host_store()
    state = {"revoked": False, "resolves": 0, "routes": 0}

    def handler(request):
        if request.url.path == "/internal/keys/resolve":
            state["resolves"] += 1
            return httpx.Response(200, json={"active": True, "consumer": "app-r", "tenant_id": 5,
                                             "scope_version": 2, "project_id": 8, "environment_id": 9})
        state["routes"] += 1
        assert request.url.params["key_sha256"] == hashlib.sha256(b"tok-rev").hexdigest()
        if state["revoked"]:
            return httpx.Response(403, json={"error": "key_revoked"})
        return httpx.Response(200, json={**ROUTE, "scope_version": 2, "tenant_id": 5,
                                         "project_id": 8, "environment_id": 9})
    monkeypatch.setattr(cpc, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    up = _Upstream()
    monkeypatch.setattr(auth_proxy, "_client", up)
    client = TestClient(auth_proxy.app)
    assert _post(client, "tok-rev").status_code == 200
    state["revoked"] = True
    r = _post(client, "tok-rev")
    assert r.status_code == 503 and r.json()["error"]["code"] == "route_unavailable"
    assert state == {"revoked": True, "resolves": 1, "routes": 2}   # resolve still cached
    assert len(up.requests) == 1


# ---- reservation ledger --------------------------------------------------------

def test_reservation_admission_and_settle_idempotent(host_store_clean):
    hs = host_store_clean
    assert hs.reserve_subject_budget(1, "k:1", "2030-01", 0.10, 0.05, "r1") == (True, True, 0.0, 0.05)
    assert hs.reserve_subject_budget(1, "k:1", "2030-01", 0.10, 0.05, "r2")[:2] == (True, True)
    allowed, ok, spent, reserved = hs.reserve_subject_budget(1, "k:1", "2030-01", 0.10, 0.05, "r3")
    assert (allowed, ok, spent, reserved) == (False, True, 0.0, pytest.approx(0.10))
    assert hs.settle_subject_budget("r1", 0.07) == (True, True)
    assert hs.settle_subject_budget("r1", 0.07) == (False, True)       # no double count
    assert hs.settle_subject_budget("r2", None) == (True, True)        # unknown cost -> 0
    assert hs.settle_subject_budget("missing", 1.0) == (False, True)
    assert hs.subject_budgets(1, "2030-01", ["k:1", "k:2"]) == [
        {"subject": "k:1", "spent_usd": 0.07, "reserved_usd": 0.0},
        {"subject": "k:2", "spent_usd": 0.0, "reserved_usd": 0.0}]
    # spent 0.07 + 0.05 > 0.10
    assert hs.reserve_subject_budget(1, "k:1", "2030-01", 0.10, 0.05, "r4")[0] is False
    assert hs.settle_subject_budget("neg", -5) == (False, True)


def test_expired_reservations_are_swept_and_still_settle_once(host_store_clean):
    hs = host_store_clean
    t = 1_900_000_000.0
    assert hs.reserve_subject_budget(2, "g:3", "2030-02", 0.05, 0.05, "old", ttl_s=10, now=t)[0]
    assert hs.reserve_subject_budget(2, "g:3", "2030-02", 0.05, 0.05, "blocked", ttl_s=10, now=t + 5)[0] is False
    # The reads ignore an expired holder even before a sweep.
    assert hs.subject_budgets(2, "2030-02", ["g:3"], now=t + 11)[0]["reserved_usd"] == 0.0
    allowed, ok, spent, reserved = hs.reserve_subject_budget(2, "g:3", "2030-02", 0.05, 0.05, "new",
                                                              ttl_s=10, now=t + 11)
    assert (allowed, reserved) == (True, pytest.approx(0.05))
    # The dead holder's late settle books its cost once without un-reserving "new".
    assert hs.settle_subject_budget("old", 0.01) == (True, True)
    assert hs.settle_subject_budget("old", 0.01) == (False, True)
    with hs._get_pool().connection() as conn:
        spent, reserved = conn.execute("SELECT spent_usd, reserved_usd FROM subject_budget_usage"
                                       " WHERE tenant_id=2").fetchone()
    assert spent == pytest.approx(0.01) and reserved == pytest.approx(0.05)


def test_concurrent_reservations_never_exceed_limit(host_store_clean):
    hs = host_store_clean
    results, barrier = [], threading.Barrier(40)

    def worker(i):
        barrier.wait()
        results.append(hs.reserve_subject_budget(3, "g:9", "2030-03", 1.0, 0.05, f"c{i}"))
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(ok for _a, ok, _s, _r in results)
    assert sum(1 for allowed, *_ in results if allowed) == 20
    assert hs.subject_budgets(3, "2030-03", ["g:9"])[0]["reserved_usd"] == pytest.approx(1.0)


def test_routing_summary_keeps_key_identity():
    summary = host_store.routing_summary({"route": "route:x", "key_id": 42, "subject": "g:7",
                                          "prompt": "secret", "labels": {"a": "b"}})
    assert summary["key_id"] == "42" and summary["subject"] == "g:7"
    assert "prompt" not in summary and "labels" not in summary


# ---- /internal/usage group_by + until_ts ---------------------------------------

def _seed_usage(now):
    rows = [  # (caller, key, project, env, ts, tokens_in, cost, status)
        ("acme", "a" * 64, 1, 10, now - 7200, 10, 0.01, 200),
        ("acme", "a" * 64, 1, 10, now - 60, 20, 0.02, 200),
        ("acme", "b" * 64, 1, 10, now - 60, 40, 0.04, 500),
        ("acme", "b" * 64, 1, 11, now - 60, 800, 0.8, 200),     # other environment
        ("other", "a" * 64, 1, 10, now - 60, 900, 0.9, 200),   # other caller
    ]
    for caller, key, project, env, ts, tin, cost, status in rows:
        host_store.insert_call({"ts": ts, "caller": caller, "key_sha256": key, "status": status,
                                "tokens_in": tin, "tokens_out": 1, "cost_usd": cost,
                                "model_family": "fam", "requested_model": "route:production",
                                "decision_trace": {"route": "route:production", "project_id": project,
                                                   "environment_id": env, "subject": "g:7",
                                                   "key_id": 42}})


def test_usage_group_by_key_is_scoped_before_aggregation():
    require_host_store()
    now = int(time.time())
    _seed_usage(now)
    client = TestClient(auth_proxy.app)
    params = {"caller": "acme", "project_id": 1, "environment_id": 10, "group_by": "key"}
    data = client.get("/internal/usage", params=params, headers=SECRET).json()
    assert data["runs"] == 3 and data["group_by"] == "key"
    groups = {g["key"]: g for g in data["groups"]}
    assert set(groups) == {"a" * 64, "b" * 64}                 # full digests
    assert groups["a" * 64]["runs"] == 2 and groups["a" * 64]["cost_usd"] == pytest.approx(0.03)
    assert groups["b" * 64] == {"key": "b" * 64, "runs": 1, "errors": 1, "tokens_in": 40,
                                "tokens_out": 1, "tokens_cached": 0, "tokens_total": 41,
                                "cost_usd": 0.04}
    # until_ts is exclusive and applies to totals and groups alike.
    data = client.get("/internal/usage", params={**params, "until_ts": now - 60}, headers=SECRET).json()
    assert data["runs"] == 1 and [g["runs"] for g in data["groups"]] == [1]
    assert data["window"]["until_ts"] == now - 60
    # since_ts is inclusive: the boundary row at exactly since_ts counts.
    data = client.get("/internal/usage", params={**params, "since_ts": now - 60}, headers=SECRET).json()
    assert data["runs"] == 2


def test_usage_reports_ledger_watermark(monkeypatch):
    require_host_store()
    monkeypatch.setenv("LEDGER_WATERMARK_LAG_S", "120")
    client = TestClient(auth_proxy.app)
    for extra in ({}, {"group_by": "key"}, {"bucket": "day"}):
        before = int(time.time()) - 120
        data = client.get("/internal/usage", params={"caller": "acme", **extra}, headers=SECRET).json()
        assert before <= data["watermark_ts"] <= int(time.time()) - 120
    host_store._pending_add(1_000)            # a queued write holds the watermark back
    try:
        data = client.get("/internal/usage", params={"caller": "acme"}, headers=SECRET).json()
        assert data["watermark_ts"] == 1_000
    finally:
        host_store._pending_done(1_000)


@pytest.mark.parametrize("group_by,expected", [
    ("model", {"fam"}), ("route", {"route:production"})])
def test_usage_group_by_model_and_route(group_by, expected):
    require_host_store()
    _seed_usage(int(time.time()))
    data = TestClient(auth_proxy.app).get("/internal/usage", params={
        "caller": "acme", "group_by": group_by}, headers=SECRET).json()
    assert {g["key"] for g in data["groups"]} == expected
    assert sum(g["runs"] for g in data["groups"]) == 4


def test_usage_group_by_day_and_hour_are_utc_buckets():
    require_host_store()
    now = int(time.time())
    _seed_usage(now)
    client = TestClient(auth_proxy.app)
    hours = client.get("/internal/usage", params={"caller": "acme", "group_by": "hour"},
                       headers=SECRET).json()["groups"]
    expect = {time.strftime("%Y-%m-%dT%H:00:00Z", time.gmtime(ts)) for ts in (now - 7200, now - 60)}
    assert {g["key"] for g in hours} == expect
    days = client.get("/internal/usage", params={"caller": "acme", "group_by": "day"},
                      headers=SECRET).json()["groups"]
    assert sum(g["runs"] for g in days) == 4
    assert all(len(g["key"]) == 10 for g in days)
    assert client.get("/internal/usage", params={"caller": "acme", "group_by": "caller"},
                      headers=SECRET).status_code == 400


# ---- /internal/usage/export ----------------------------------------------------

def test_export_cursor_pages_in_id_order_with_watermark(monkeypatch):
    require_host_store()
    monkeypatch.setenv("LEDGER_WATERMARK_LAG_S", "0")
    now = int(time.time())
    _seed_usage(now)
    client = TestClient(auth_proxy.app)
    scope = {"caller": "acme", "project_id": 1, "environment_id": 10}
    page = client.get("/internal/usage/export", params={**scope, "limit": 2}, headers=SECRET).json()
    assert [r["tokens_in"] for r in page["rows"]] == [10, 20]
    row = page["rows"][0]
    assert set(row) == {"id", "ts", "key_sha256", "subject", "route", "model_family", "provider",
                        "status", "tokens_in", "tokens_out", "tokens_cached", "tokens_total",
                        "cost_usd", "cost_basis"}
    assert row["key_sha256"] == "a" * 64 and row["subject"] == "g:7" and row["route"] == "route:production"
    # The third scoped row (ts now-60) is not delivered yet, so the watermark
    # cannot pass it even though the lag is zero.
    assert page["watermark_ts"] <= now - 60
    page2 = client.get("/internal/usage/export", params={
        **scope, "after_id": page["next_after_id"], "limit": 2}, headers=SECRET).json()
    assert [r["tokens_in"] for r in page2["rows"]] == [40]
    assert page2["rows"][0]["id"] > page["next_after_id"]
    assert page2["watermark_ts"] >= now - 1
    page3 = client.get("/internal/usage/export", params={
        **scope, "after_id": page2["next_after_id"]}, headers=SECRET).json()
    assert page3["rows"] == [] and page3["next_after_id"] == page2["next_after_id"]
    assert page3["scope_version"] == 2


def test_export_holds_back_fresh_rows_and_caps_watermark(monkeypatch):
    require_host_store()
    monkeypatch.setenv("LEDGER_WATERMARK_LAG_S", "120")
    now = int(time.time())
    _seed_usage(now)
    page = TestClient(auth_proxy.app).get("/internal/usage/export", params={"caller": "acme"},
                                          headers=SECRET).json()
    assert page["rows"] == [] and page["next_after_id"] == 0
    assert page["watermark_ts"] <= now - 7200          # the oldest undelivered call


def test_ledger_watermark_waits_for_queued_writes(monkeypatch):
    monkeypatch.setenv("LEDGER_WATERMARK_LAG_S", "10")
    now = 2_000_000_000
    assert host_store.ledger_watermark(now) == now - 10
    host_store._pending_add(now - 500)
    try:
        assert host_store.ledger_watermark(now) == now - 500
    finally:
        host_store._pending_done(now - 500)
    assert host_store.ledger_watermark(now) == now - 10


def test_export_validation_and_auth(monkeypatch):
    client = TestClient(auth_proxy.app)
    assert client.get("/internal/usage/export", params={"caller": "a"},
                      headers={"x-internal-secret": "no"}).status_code == 403
    assert client.get("/internal/usage/export", params={"caller": "a", "after_id": -1},
                      headers=SECRET).status_code == 400
    assert client.get("/internal/usage/export", headers=SECRET).status_code == 400
    for invalid in ({"project_id": 1}, {"project_id": 1, "environment_id": 0}):
        assert client.get("/internal/usage/export", params={"caller": "a", **invalid},
                          headers=SECRET).status_code == 400
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "")
    assert client.get("/internal/usage/export", params={"caller": "a"}).status_code == 404


def test_export_limit_is_capped(monkeypatch):
    seen = {}

    def fake(caller, after_id, limit, **scope):
        seen.update(limit=limit)
        return {"rows": [], "next_after_id": after_id, "watermark_ts": 0}
    monkeypatch.setattr(host_store, "usage_export", fake)
    TestClient(auth_proxy.app).get("/internal/usage/export", params={"caller": "a", "limit": 99999},
                                   headers=SECRET)
    assert seen["limit"] == 5000


# ---- /internal/budgets ---------------------------------------------------------

def test_budgets_endpoint_auth_validation_and_tenant_scoping(monkeypatch):
    require_host_store()
    period = "2030-04"
    host_store.reserve_subject_budget(5, "g:7", period, 10, 0.05, "b1")
    host_store.settle_subject_budget("b1", 1.25)
    host_store.reserve_subject_budget(5, "g:7", period, 10, 0.05, "b2")      # in flight
    host_store.reserve_subject_budget(6, "g:7", period, 10, 0.05, "other")  # other tenant
    host_store.settle_subject_budget("other", 9.0)
    client = TestClient(auth_proxy.app)
    r = client.get("/internal/budgets", params=[("tenant_id", 5), ("period", period),
                                                ("subject", "g:7"), ("subject", "k:1")], headers=SECRET)
    assert r.status_code == 200
    assert r.json() == {"tenant_id": 5, "period": period, "subjects": [
        {"subject": "g:7", "spent_usd": 1.25, "reserved_usd": 0.05},
        {"subject": "k:1", "spent_usd": 0.0, "reserved_usd": 0.0}]}
    ok = {"tenant_id": 5, "period": period, "subject": "g:7"}
    assert client.get("/internal/budgets", params=ok, headers={"x-internal-secret": "no"}).status_code == 403
    for bad in ({"tenant_id": 0}, {"tenant_id": None}, {"period": "2030-13"}, {"period": "2030-4"},
                {"subject": "x:1"}, {"subject": None}):
        params = {k: v for k, v in {**ok, **bad}.items() if v is not None}
        assert client.get("/internal/budgets", params=params, headers=SECRET).status_code == 400, bad
    too_many = [("tenant_id", 5), ("period", period)] + [("subject", f"k:{i + 1}") for i in range(501)]
    assert client.get("/internal/budgets", params=too_many, headers=SECRET).status_code == 400

    def down(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(host_store, "subject_budgets", down)
    assert client.get("/internal/budgets", params=ok, headers=SECRET).status_code == 503
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "")
    assert client.get("/internal/budgets", params=ok).status_code == 404
