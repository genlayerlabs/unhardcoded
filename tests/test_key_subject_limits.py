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
from decimal import Decimal
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

_ORIGINAL_RESOLVE_ROUTE = cpc.resolve_route
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

    def __init__(self, cost=0.03, fail_mid_stream=False, cost_basis=None):
        self.cost, self.fail = cost, fail_mid_stream
        self.cost_basis = cost_basis
        self.closed = False

    async def aiter_raw(self):
        yield b'data: {"choices":[]}\n\n'
        if self.fail:
            raise httpx.ReadError("upstream reset")
        yield ('data: ' + json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1},
                                      "x_router": {"cost_usd": self.cost,
                                                   "cost_basis": self.cost_basis}}) + "\n\n").encode()
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


def test_shared_budget_contract_keeps_zero_and_exact_amounts(monkeypatch):
    data = _resolve_with(monkeypatch, {"key": KEY, "limits": {
        "monthly_budget_usd": 0, "shared_budget": {
            "scope": "team-1", "monthly_budget_usd": "0",
            "max_call_cost_usd": "0.000000000001", "help_url": "javascript:bad",
        }}})
    limits = data["limits"]
    assert limits["monthly_budget_usd"] == Decimal(0)
    assert limits["shared_budget"] == {
        "scope": "team-1", "monthly_budget_usd": Decimal(0),
        "max_call_cost_usd": Decimal("0.000000000001"),
        "guidance": None, "help_url": None,
    }


@pytest.mark.parametrize("overrides", [
    {"scope": "g:7"}, {"monthly_budget_usd": -1}, {"monthly_budget_usd": True},
    {"monthly_budget_usd": "0.0000000000001"}, {"max_call_cost_usd": None},
    {"max_call_cost_usd": 0},
])
def test_shared_budget_invalid_contract_fails_closed(monkeypatch, overrides):
    with pytest.raises(cpc.RouteUnavailable):
        _resolve_with(monkeypatch, {"key": KEY, "limits": {"shared_budget": {
            "scope": "team", "monthly_budget_usd": "1", "max_call_cost_usd": "0.05",
            **overrides,
        }}})


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
    {"key": KEY, "limits": {"monthly_budget_usd": "-1"}},
    {"key": KEY, "limits": {"monthly_budget_usd": "1e3"}},
    {"key": KEY, "limits": {"monthly_budget_usd": True}},
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


def test_shared_budget_composes_and_updates_on_next_resolve(monkeypatch):
    require_host_store()
    up = _saas(monkeypatch, limits={})
    shared = {"scope": "team", "monthly_budget_usd": "0.10", "max_call_cost_usd": "0.05",
              "guidance": "Contact your administrator.", "help_url": "https://example.org/limits"}
    key_limit = "0"
    route_reads = []

    def handler(request):
        route_reads.append(request.url)
        return httpx.Response(200, json={**ROUTE, "key": KEY, "limits": {
            "monthly_budget_usd": key_limit, "shared_budget": shared}})
    monkeypatch.setattr(cpc, "resolve_route", _ORIGINAL_RESOLVE_ROUTE)
    monkeypatch.setattr(cpc, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    client = TestClient(auth_proxy.app)

    assert _post(client).json()["error"]["code"] == "key_budget_exhausted"
    assert not up.requests
    assert _budget("s:team")["spent_usd"] == 0

    key_limit = "0.10"
    assert _post(client).status_code == 200
    assert _budget("s:team")["spent_usd"] == _budget()["spent_usd"] == 0.02

    shared["monthly_budget_usd"] = "0"
    response = _post(client)
    assert response.status_code == 402
    assert response.json()["error"] == {
        "message": "shared budget exhausted", "type": "budget_error",
        "code": "shared_budget_exhausted", "budget_usd": 0.0, "spent_usd": 0.02,
        "guidance": shared["guidance"], "help_url": shared["help_url"],
    }
    assert _budget()["spent_usd"] == 0.02
    assert len(route_reads) == 3 and len(up.requests) == 1


@pytest.mark.parametrize("cost,status,expected", [(None, 200, 0.05), (None, 500, 0.05), (0.0, 200, 0.0)])
def test_shared_budget_unknown_cost_keeps_ceiling(monkeypatch, cost, status, expected):
    require_host_store()
    _saas(monkeypatch, limits={"shared_budget": {
        "scope": "team", "monthly_budget_usd": Decimal("0.10"),
        "max_call_cost_usd": Decimal("0.05"),
    }}, upstream=_Upstream(lambda: _JsonResp(cost=cost, status_code=status)))
    assert _post(TestClient(auth_proxy.app)).status_code == status
    assert _budget("s:team")["spent_usd"] == expected


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
          upstream=_Upstream(lambda: _StreamResp(cost=0.03, cost_basis="reported")))
    r = _post(TestClient(auth_proxy.app))
    assert r.status_code == 200 and b"[DONE]" in r.content
    assert _live_reservations() == 0
    assert _budget()["spent_usd"] == pytest.approx(0.03)
    (row,) = host_store.usage_export("app-a", now=time.time() + 3600)["rows"]
    assert row["cost_usd"] == pytest.approx(0.03)
    assert row["cost_source"] == "reported"


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
    assert (spent, reserved) == (Decimal("0.01"), Decimal("0.05"))


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

def _seed_usage(now, **call_fields):
    rows = [  # (caller, key, project, env, ts, tokens_in, cost, status)
        ("acme", "a" * 64, 1, 10, now - 7200, 10, 0.01, 200),
        ("acme", "a" * 64, 1, 10, now - 60, 20, 0.02, 200),
        ("acme", "b" * 64, 1, 10, now - 120, 40, 0.04, 500),
        ("acme", "b" * 64, 1, 11, now - 30, 800, 0.8, 200),   # other environment
        ("other", "a" * 64, 1, 10, now - 20, 900, 0.9, 200),  # other caller
    ]
    for caller, key, project, env, ts, tin, cost, status in rows:
        host_store.insert_call({"ts": ts, "caller": caller, "key_sha256": key, "status": status,
                                "tokens_in": tin, "tokens_out": 1, "cost_usd": cost,
                                "model_family": "fam", "requested_model": "route:production",
                                "decision_trace": {"route": "route:production", "project_id": project,
                                                   "environment_id": env, "subject": "g:7",
                                                   "key_id": 42},
                                **call_fields})


def test_usage_group_by_key_is_scoped_before_aggregation():
    require_host_store()
    now = int(time.time())
    _seed_usage(now)
    host_store.insert_call({"ts": now - 10, "caller": "acme", "key_sha256": "a" * 64,
                            "decision_trace": {"project_id": 2, "environment_id": 10}})
    client = TestClient(auth_proxy.app)
    params = {"caller": "acme", "project_id": 1, "environment_id": 10, "group_by": "key"}
    data = client.get("/internal/usage", params=params, headers=SECRET).json()
    assert data["runs"] == 3 and data["group_by"] == "key"
    groups = {g["key"]: g for g in data["groups"]}
    assert set(groups) == {"a" * 64, "b" * 64}                 # full digests
    assert groups["a" * 64]["runs"] == 2 and groups["a" * 64]["cost_usd"] == pytest.approx(0.03)
    assert groups["a" * 64]["last_call_ts"] == now - 60
    assert groups["b" * 64] == {"key": "b" * 64, "runs": 1, "errors": 1, "tokens_in": 40,
                                "tokens_out": 1, "tokens_cached": 0, "tokens_total": 41,
                                "cost_usd": 0.04, "last_call_ts": now - 120}
    # until_ts is exclusive and applies to totals and groups alike.
    data = client.get("/internal/usage", params={**params, "until_ts": now - 120}, headers=SECRET).json()
    assert data["runs"] == 1 and [g["runs"] for g in data["groups"]] == [1]
    assert data["groups"][0]["last_call_ts"] == now - 7200
    assert data["window"]["until_ts"] == now - 120
    # since_ts is inclusive: the boundary row at exactly since_ts counts.
    data = client.get("/internal/usage", params={**params, "since_ts": now - 120}, headers=SECRET).json()
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
    assert all("last_call_ts" not in g for g in data["groups"])


def test_usage_group_by_day_and_hour_are_utc_buckets():
    require_host_store()
    now = int(time.time())
    _seed_usage(now)
    client = TestClient(auth_proxy.app)
    hours = client.get("/internal/usage", params={"caller": "acme", "group_by": "hour"},
                       headers=SECRET).json()["groups"]
    expect = {time.strftime("%Y-%m-%dT%H:00:00Z", time.gmtime(ts))
              for ts in (now - 7200, now - 120, now - 60, now - 30)}
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
    _seed_usage(now, served_model_id="provider/model-v2", latency_ms=12.5)
    client = TestClient(auth_proxy.app)
    scope = {"caller": "acme", "project_id": 1, "environment_id": 10}
    page = client.get("/internal/usage/export", params={**scope, "limit": 2}, headers=SECRET).json()
    assert [r["tokens_in"] for r in page["rows"]] == [10, 20]
    row = page["rows"][0]
    assert set(row) == {"id", "ts", "key_sha256", "subject", "route", "model_family", "provider",
                        "status", "tokens_in", "tokens_out", "tokens_cached", "tokens_total",
                        "cost_usd", "cost_basis", "cost_source",
                        "winning_inference_cost_usd", "winning_inference_cost_basis",
                        "served_model_id", "latency_ms"}
    assert row["key_sha256"] == "a" * 64 and row["subject"] == "g:7" and row["route"] == "route:production"
    assert row["served_model_id"] == "provider/model-v2"
    assert row["latency_ms"] == 12.5
    assert row["winning_inference_cost_usd"] is None
    assert row["winning_inference_cost_basis"] is None
    # The third scoped row (ts now-120) is not delivered yet, so the watermark
    # cannot pass it even though the lag is zero.
    assert page["watermark_ts"] <= now - 120
    page2 = client.get("/internal/usage/export", params={
        **scope, "after_id": page["next_after_id"], "limit": 2}, headers=SECRET).json()
    assert [r["tokens_in"] for r in page2["rows"]] == [40]
    assert page2["rows"][0]["served_model_id"] == "provider/model-v2"
    assert page2["rows"][0]["latency_ms"] == 12.5
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


def test_successful_call_with_unknown_cost_books_the_reservation(monkeypatch):
    """An unpriced model, or a stream closed before its usage chunk, must not
    make a successful call free against the budget."""
    require_host_store()
    _saas(monkeypatch, limits={"monthly_budget_usd": "1", "rate_per_min": None, "burst": None},
          upstream=_Upstream(lambda: _JsonResp(cost=None)))
    assert _post(TestClient(auth_proxy.app)).status_code == 200
    assert _live_reservations() == 0
    assert _budget()["spent_usd"] == pytest.approx(auth_proxy.CLOUD_BUDGET_RESERVATION_USD)


def test_failed_settle_is_retried(monkeypatch):
    require_host_store()
    monkeypatch.setattr(auth_proxy, "SETTLE_RETRY_DELAYS_S", (0.0, 0.0, 0.0))
    real, calls = host_store.settle_subject_budget, []

    def flaky(reservation_id, cost):
        calls.append(reservation_id)
        return (False, False) if len(calls) < 3 else real(reservation_id, cost)
    monkeypatch.setattr(host_store, "settle_subject_budget", flaky)
    _saas(monkeypatch, limits={"monthly_budget_usd": "1.00", "rate_per_min": None, "burst": None})
    assert _post(TestClient(auth_proxy.app)).status_code == 200
    assert _budget()["spent_usd"] == 0.02 and len(calls) == 3


def test_ingress_refusals_by_tenant_subject_and_reason(monkeypatch):
    require_host_store()
    hour = int(time.time()) // 3600 * 3600
    monkeypatch.setattr(auth_proxy.time, "time", lambda: hour + 60)
    client = TestClient(auth_proxy.app)
    up = _saas(monkeypatch, limits={"monthly_budget_usd": "0.05"},
               upstream=_Upstream(lambda: _JsonResp(cost=0.05)))
    assert _post(client).status_code == 200
    auth_proxy.drain_settles()
    for _ in range(10):
        assert _post(client).status_code == 402
    assert len(up.requests) == 1

    _saas(monkeypatch, limits={"rate_per_min": 1}, key={**KEY, "subject": "k:43"})
    assert _post(client).status_code == 200
    for _ in range(10):
        assert _post(client).status_code == 429

    # A second hour for the same subject and reason folds into the API group.
    host_store.insert_ingress_refusal_async(5, "g:7", "key_budget_exhausted", hour + 3601)
    host_store.insert_ingress_refusal_async(6, "g:7", "key_budget_exhausted", hour + 1)
    with host_store._get_pool().connection() as conn:
        assert conn.execute("SELECT count(*) FROM ingress_refusals").fetchone()[0] == 4

    path = "/internal/usage/refusals"
    params = {"tenant_id": 5, "since_ts": hour, "until_ts": hour + 7200}
    assert client.get(path, params=params).status_code == 403
    assert client.get(path, params={**params, "since_ts": hour + 1}, headers=SECRET).status_code == 400
    response = client.get(path, params=params, headers=SECRET)
    assert response.status_code == 200
    assert response.json()["groups"] == [
        {"subject": "g:7", "reason": "key_budget_exhausted", "count": 11,
         "first_ts": hour + 60, "last_ts": hour + 3601},
        {"subject": "k:43", "reason": "key_rate_limit", "count": 10,
         "first_ts": hour + 60, "last_ts": hour + 60},
    ]
    first_hour = client.get(path, params={**params, "until_ts": hour + 3600}, headers=SECRET)
    assert first_hour.json()["groups"][0]["count"] == 10
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "")
    assert client.get(path, params=params, headers=SECRET).status_code == 404


def test_ingress_refusal_does_not_wait_for_failed_write(monkeypatch):
    _saas(monkeypatch, limits={"rate_per_min": 1})
    monkeypatch.setattr(auth_proxy, "_rate_ok", lambda *args: (True, True, 0))
    monkeypatch.setattr(host_store, "consume_rate_token", lambda *args: (False, True, 1))
    monkeypatch.setattr(host_store, "_SYNC_WRITES", False)
    started = threading.Event()
    finish = threading.Event()
    timed_out = threading.Event()

    def unavailable():
        started.set()
        if not finish.wait(5):
            timed_out.set()
        raise RuntimeError("store down")

    monkeypatch.setattr(host_store, "_get_pool", unavailable)
    client = TestClient(auth_proxy.app)
    try:
        response = _post(client)
        assert response.status_code == 429
        assert response.json()["error"]["code"] == "key_rate_limit"
        assert started.wait(5)
        assert not timed_out.is_set()
        hour = int(time.time()) // 3600 * 3600
        # Let both the writer and the API read fail against the same store.
        finish.set()
        read = client.get("/internal/usage/refusals", headers=SECRET, params={
            "tenant_id": 5, "since_ts": hour, "until_ts": hour + 3600})
        assert read.status_code == 503
        assert read.json()["error"] == "refusal_store_unavailable"
    finally:
        finish.set()
        host_store._refusal_q.join()


def test_ingress_refusal_retention(monkeypatch):
    require_host_store()
    hour = int(time.time()) // 3600 * 3600
    expired = hour - 25 * 3600
    host_store.insert_ingress_refusal_async(5, "g:7", "key_rate_limit", expired)
    monkeypatch.setattr(host_store, "_RETENTION_DAYS", 1)
    monkeypatch.setattr(host_store, "_PRUNE_EVERY", 1)

    # A refusal-only workload sweeps old rows and discards stale queued jobs.
    host_store.insert_ingress_refusal_async(5, "g:7", "key_rate_limit", hour)
    host_store.insert_ingress_refusal_async(5, "g:7", "key_rate_limit", expired)
    with host_store._get_pool().connection() as conn:
        assert conn.execute("SELECT hour_ts FROM ingress_refusals").fetchall() == [(hour,)]


def test_published_route_outcomes(monkeypatch, host_store_clean):
    hour = int(time.time()) // 3600 * 3600
    monkeypatch.setattr(auth_proxy.time, "time", lambda: hour + 60)

    class StreamWithFallback(_StreamResp):
        async def aiter_raw(self):
            yield b'data: {"choices":[]}\n\n'
            meta = {"x_router": {"first_token_ms": 250, "decision_trace": {"decision_path": [
                {"event": "attempted", "attempt": 1, "error_kind": "timeout"},
                {"event": "attempted", "attempt": 1},
            ]}}}
            yield f"data: {json.dumps(meta)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    client = TestClient(auth_proxy.app)
    _saas(monkeypatch, limits={}, upstream=_Upstream(StreamWithFallback))
    assert _post(client).status_code == 200
    _saas(monkeypatch, limits={}, upstream=_Upstream(lambda: _JsonResp(status_code=504)))
    assert _post(client).status_code == 504
    _saas(monkeypatch, limits={}, upstream=_Upstream(lambda: httpx.ReadError("reset")))
    assert _post(client).status_code == 502
    # A disconnect advances the admitted-call count without adding a latency sample.
    _saas(monkeypatch, limits={}, upstream=_Upstream(lambda: _JsonResp(status_code=499)))
    assert _post(client).status_code == 499

    for _ in range(20):
        host_store.insert_route_outcome_async(
            5, 0, ROUTE["route"], hour + 80, "success", False, None, 60001)
    host_store.insert_route_outcome_async(5, 3, ROUTE["route"], hour, "success", False, None, 1)
    host_store.insert_route_outcome_async(5, 0, ROUTE["route"], hour + 3600, "other", False, None, None)
    host_store.insert_route_outcome_async(6, 0, ROUTE["route"], hour, "other", False, None, None)
    host_store.insert_route_outcome_async(5, 0, "route:other", hour, "other", False, None, None)

    path = "/internal/usage/route-outcomes"
    params = {"tenant_id": 5, "route": ROUTE["route"], "since_ts": hour, "until_ts": hour + 3600}
    assert client.get(path, params=params).status_code == 403
    assert client.get(path, params={**params, "since_ts": hour + 1}, headers=SECRET).status_code == 400
    read = client.get(path, params=params, headers=SECRET)
    assert read.status_code == 200
    data = read.json()
    assert len(data["hours"]) == 1
    row = data["hours"][0]
    assert [row[k] for k in ("hour_ts", "calls", "successes", "upstream_errors", "timeouts",
                             "fallbacks")] == [hour, 24, 21, 1, 1, 1]
    assert len(row["first_token_ms"]) == len(data["latency_upper_bounds_ms"])
    assert row["first_token_ms"][data["latency_upper_bounds_ms"].index(250)] == 1
    assert sum(row["first_token_ms"]) == 1
    assert sum(row["full_answer_ms"]) == 21
    assert row["full_answer_ms"][-1] == 20
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "")
    assert client.get(path, params=params, headers=SECRET).status_code == 404


def test_route_outcome_retention(monkeypatch, host_store_clean):
    hour = int(time.time()) // 3600 * 3600
    expired = hour - 25 * 3600
    host_store.insert_route_outcome_async(5, 0, ROUTE["route"], expired, "success", False, 1, 1)
    monkeypatch.setattr(host_store, "_RETENTION_DAYS", 1)
    monkeypatch.setattr(host_store, "_PRUNE_EVERY", 1)

    assert host_store.route_outcomes(5, 0, ROUTE["route"], expired, hour) == []
    for ts in (hour, expired):
        host_store.insert_route_outcome_async(5, 0, ROUTE["route"], ts, "success", False, 1, 1)
    with host_store._get_pool().connection() as conn:
        assert conn.execute("SELECT hour_ts FROM route_outcome_hours").fetchall() == [(hour,)]


def test_route_outcome_queue_drops_overflow(monkeypatch):
    import queue

    pending = queue.Queue(maxsize=1)
    monkeypatch.setattr(host_store, "_route_outcome_q", pending)
    monkeypatch.setattr(host_store, "_SYNC_WRITES", False)
    for _ in range(5):
        host_store.insert_route_outcome_async(
            5, 0, ROUTE["route"], int(time.time()), "success", False, 1, 1)
    assert pending.qsize() == 1


def test_route_outcomes_and_usage_for_responses_stream(monkeypatch, host_store_clean):
    class ResponsesStream(_StreamResp):
        async def aiter_raw(self):
            for event in (
                {"type": "response.created", "response": {"status": "in_progress"}},
                {"type": "response.completed", "response": {
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                    "x_router": {"cost_usd": 0.02, "cost_basis": "reported",
                                 "decision_trace": {"decision_path": [
                        {"event": "attempted", "attempt": 1, "error_kind": "rate_limit"},
                        {"event": "attempted", "attempt": 2},
                    ]}},
                }},
            ):
                yield f"data: {json.dumps(event)}\n\n".encode()

    _saas(monkeypatch, limits={}, upstream=_Upstream(ResponsesStream))
    client = TestClient(auth_proxy.app)
    response = client.post("/v1/responses", headers={"Authorization": "Bearer tok-a"},
                           json={"model": ROUTE["route"], "input": "hello", "stream": True})
    assert response.status_code == 200
    hour = int(time.time()) // 3600 * 3600
    row, = host_store.route_outcomes(5, 0, ROUTE["route"], hour, hour + 3600)
    assert (row["calls"], row["successes"], row["fallbacks"]) == (1, 1, 0)
    assert sum(row["first_token_ms"]) == 0
    assert sum(row["full_answer_ms"]) == 1
    (usage,) = host_store.usage_export("app-a", now=time.time() + 3600)["rows"]
    assert (usage["tokens_in"], usage["tokens_out"]) == (2, 3)
    assert (usage["cost_usd"], usage["cost_source"]) == (0.02, "reported")


@pytest.mark.parametrize("surface", ["json", "chat_stream", "responses_stream"])
def test_scoped_export_preserves_winning_cost_after_automatic_fallback(
        monkeypatch, host_store_clean, surface):
    from shim import _build_x_router

    scope = {"project_id": 8, "environment_id": 9}

    def resolve(request):
        scoped = {"scope_version": 2, "tenant_id": 5, **scope}
        if request.url.path == "/internal/keys/resolve":
            return httpx.Response(200, json={"active": True, "consumer": "app-a", **scoped})
        return httpx.Response(200, json={**ROUTE, "key": KEY, **scoped})

    metadata = _build_x_router({
        "chosen": {"provider_id": "openrouter"},
        "response": {"cost_reported": 0.01},
        "trace": {"automatic": {"cost_usd": 0.001}, "decision_path": [
            {"event": "attempted", "attempt": 1, "error_kind": "timeout"},
            {"event": "attempted", "attempt": 2},
        ]},
    })

    class Stream(_StreamResp):
        async def aiter_raw(self):
            payload = {"x_router": metadata}
            if surface == "responses_stream":
                payload = {"type": "response.completed", "response": payload}
            yield f"data: {json.dumps(payload)}\n\n".encode()

    upstream = _JsonResp()
    upstream._body = json.dumps({"x_router": metadata}).encode()
    if surface != "json":
        upstream = Stream()
    monkeypatch.setattr(cpc, "_client", httpx.AsyncClient(transport=httpx.MockTransport(resolve)))
    monkeypatch.setattr(auth_proxy, "_client", _Upstream(lambda: upstream))
    monkeypatch.setattr(host_store_clean, "_SYNC_WRITES", False)
    monkeypatch.setenv("LEDGER_WATERMARK_LAG_S", "0")

    endpoint = "/v1/responses" if surface == "responses_stream" else "/v1/chat/completions"
    request = {"input": "hello"} if surface == "responses_stream" else {"messages": []}
    client = TestClient(auth_proxy.app)
    response = client.post(endpoint, headers={"Authorization": "Bearer tok-a"}, json={
        **request, "model": ROUTE["route"], "stream": surface != "json",
    })
    assert response.status_code == 200
    host_store_clean._write_q.join()
    export = client.get("/internal/usage/export", headers=SECRET,
                        params={"caller": "app-a", **scope})
    assert export.status_code == 200
    row, = export.json()["rows"]
    assert row["subject"] == KEY["subject"]
    assert row["provider"] == "openrouter"
    assert row["cost_usd"] is None
    assert row["cost_basis"] == "unknown_total"
    assert row["cost_source"] == "none"
    assert row["winning_inference_cost_usd"] == 0.01
    assert row["winning_inference_cost_basis"] == "reported"
