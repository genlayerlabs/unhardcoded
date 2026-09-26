"""Regression tests for the 2026-09-26 router security audit (router side).

Each test encodes an attack that worked before the fix: caller Xforms rewriting
route keys, retry loops freezing the event loop, caller-controlled breaker
state, tenant outcomes poisoning shared observations, untimed flows, session
metering across callers, unauthenticated /x/* admin endpoints, unbounded BYO
reads, tenant-sized peer gates, NAT64 egress, tenant keys unlocking local
providers, and codex/antseed credential routing.
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm_router_host as L  # noqa: E402
from llm_router_host import LLMRouterHost  # noqa: E402
from shim import create_app  # noqa: E402

from conftest import seed_call  # noqa: E402

MSG = [{"role": "user", "content": "hi"}]


def _enveloped_config(tmp_path):
    cfg = tmp_path / "c.lua"
    cfg.write_text(f'local cfg = dofile("{ROOT}/core/config.example.lua")\n'
                   'cfg.policy_envelope = { "and", { "meets_req" }, { "not", { "is", "disabled" } } }\n'
                   'return cfg\n')
    return cfg


def _host(tmp_path, **kw):
    h = LLMRouterHost(router_path=ROOT / "core" / "router.lua", config_path=_enveloped_config(tmp_path),
                      metrics_path=ROOT / "core" / "metrics.example.lua", enforce_provider_auth=False, **kw)
    h.init()
    return h


@pytest.fixture
def no_fold(monkeypatch):
    monkeypatch.setattr(L, "_fold_route_outcome", lambda *a, **k: None)


# ---- C1: caller Xforms cannot move a call or swap its credential ------------

def test_set_param_cannot_send_internal_secret_to_attacker(tmp_path, monkeypatch, no_fold):
    from provider_adapters.openai_compatible import make_async_call_provider
    monkeypatch.setenv("CONTROL_PLANE_INTERNAL_SECRET", "TOP-SECRET-INTERNAL")
    captured = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            captured["auth"] = self.headers.get("Authorization")
            self.rfile.read(int(self.headers.get("content-length", 0)))
            body = json.dumps({"choices": [{"message": {"content": "pwned"},
                                            "finish_reason": "stop"}], "usage": {}}).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        h = _host(tmp_path, call_provider_async=make_async_call_provider(env_get=os.environ.get))
        pol = ["policy", ["top"], ["zero"], ["argmax"],
               ["seq", ["set_param", "base_url", f"http://127.0.0.1:{srv.server_port}/v1"],
                ["set_param", "api_kind", "openai_compatible"], ["set_param", "auth", False],
                ["set_param", "auth_env", "CONTROL_PLANE_INTERNAL_SECRET"]],
               ["always", {"action": "abort"}]]
        r = TestClient(create_app(h, default_profile="default")).post(
            "/v1/chat/completions", json={"messages": MSG, "policy_ir": pol})
        assert r.status_code == 400
        assert captured == {}
    finally:
        srv.shutdown()


def test_host_pins_route_keys_from_catalog(tmp_path):
    h = _host(tmp_path)
    pid, prov = next((p, v) for p, v in h.catalog()["providers"].items()
                     if v.get("discovery") == "static" and v.get("auth_env"))
    req = h._pin_route({"provider_id": pid, "base_url": "https://evil.example/v1",
                        "auth_env": "CONTROL_PLANE_INTERNAL_SECRET", "auth": False,
                        "api_kind": "openai_codex"})
    assert req["base_url"] == prov["base_url"]
    assert req["auth_env"] == prov["auth_env"]
    assert req["api_kind"] == prov["api_kind"]
    assert req["auth"] == prov.get("auth")


# ---- H2: a retry storm cannot freeze the loop past the deadline -------------

def test_huge_zero_backoff_retry_is_refused_and_request_returns(tmp_path, no_fold):
    from provider_adapters.openai_compatible import make_async_call_provider
    h = _host(tmp_path, call_provider_async=make_async_call_provider(env_get=lambda _k: None))
    app = create_app(h, default_profile="default", request_deadline_ms=2000)
    pol = ["policy", ["meets_req"], ["zero"], ["argmax"], ["id"],
           ["always", {"action": "retry_same", "attempts": 1e15, "backoff_ms": 0}]]
    out = {}

    def run():
        async def main():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://x") as c:
                r = await c.post("/v1/chat/completions", json={"messages": MSG, "policy_ir": pol})
                out["status"] = r.status_code
        asyncio.run(main())

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), "event loop frozen by a zero-backoff retry storm"
    assert out["status"] == 400


def test_execute_async_yields_between_attempts(tmp_path, no_fold):
    ticks = []

    async def failing(_req):   # returns without awaiting, like an auth_error
        return {"ok": False, "error_kind": "server_error", "http_status": 500, "latency_ms": 0}

    h = _host(tmp_path, call_provider_async=failing)
    pol = ["policy", ["top"], ["zero"], ["argmax"], ["id"],
           ["always", {"action": "retry_same", "attempts": 10, "backoff_ms": 0}]]

    async def main():
        async def ticker():
            while True:
                ticks.append(1)
                await asyncio.sleep(0)
        t = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        before = len(ticks)
        res = await h.execute_async({"messages": MSG, "policy_ir": pol})
        t.cancel()
        return res, len(ticks) - before

    res, during = asyncio.run(main())
    assert res["ok"] is False
    assert during > 5


# ---- H1: caller policies cannot hold shared breakers / disables -------------

def test_caller_open_breaker_ms_is_clamped(tmp_path, no_fold):
    async def timeout(_req):
        return {"ok": False, "error_kind": "timeout", "http_status": 0, "latency_ms": 1}

    h = _host(tmp_path, call_provider_async=timeout)
    pol = ["policy", ["top"], ["zero"], ["argmax"], ["id"],
           ["always", {"action": "next_candidate", "open_breaker_ms": 3e12}]]
    asyncio.run(h.execute_async({"messages": MSG, "policy_ir": pol}))
    now = h._now_ms()
    until = [b.get("open_until_ms") for b in (h.dump_state().get("circuit_breakers") or {}).values()
             if b.get("open")]
    assert until and max(until) <= now + 5 * 60 * 1000 + 5_000


def test_caller_disable_provider_does_not_disable_shared_provider(tmp_path, no_fold):
    async def fail(req):
        return {"ok": False, "error_kind": req.get("_kind", "server_error"), "http_status": 500,
                "latency_ms": 1}

    h = _host(tmp_path, call_provider_async=fail)
    pol = ["policy", ["top"], ["zero"], ["argmax"], ["id"],
           ["always", {"action": "disable_provider"}]]
    asyncio.run(h.execute_async({"messages": MSG, "policy_ir": pol}))
    assert not (h.dump_state().get("disabled_providers") or {})


# ---- H3: tenant BYO outcomes never land in operator route observations ------

def test_tenant_byo_outcome_not_recorded_under_operator_route(host_store_clean, monkeypatch):
    import host_store
    rows = []
    monkeypatch.setattr(host_store, "observe_route_call_async", rows.append)
    base = LLMRouterHost(ROOT / "core/router.lua", ROOT / "tests/fixtures/managed.lua")
    base.init()
    child = base.for_tenant(7, {}, connections={
        "antseed": {"gateway_url": "https://gw.attacker.example", "token": "t" * 40}})
    victim = "operator-real-peer-0xabc"
    child._tenant_offers["antseed"] = [{
        "model_family": "shared-model", "wire_model_id": "x", "peer_id": victim,
        "price_in_usd_per_mtok": 0.1, "price_out_usd_per_mtok": 0.1,
        "capabilities": {"context": 128000, "supports_tools": True}}]
    child.update_metrics("__credits", "antseed", {"free_credits_remaining_usd": 5})
    calls = []

    async def gw(request):
        calls.append(request)
        return {"ok": False, "error_kind": "server_error", "http_status": 500, "latency_ms": 1}

    child.set_async_call_hook(gw)
    asyncio.run(child.execute_async({"messages": MSG, "policy_ir": [
        "policy", ["and", ["meets_req"], ["provider_eq", "antseed"]], ["zero"], ["ordered"], ["id"],
        ["always", {"action": "next_candidate"}]]}))
    assert calls, "the BYO route was not exercised"
    assert not [r for r in rows if r["served_by"] == victim]


# ---- M4: flows honour the request deadline -----------------------------------

def test_untyped_flow_is_cut_at_request_deadline(tmp_path, no_fold):
    async def slow(_req):
        await asyncio.sleep(5)
        return {"ok": True, "response": {"text": "late"}, "latency_ms": 5000}

    h = _host(tmp_path, call_provider_async=slow)
    app = create_app(h, default_profile="default", request_deadline_ms=300)
    pol = ["policy", ["top"], ["zero"], ["argmax"], ["id"], ["always", {"action": "abort"}]]
    flow = ["flow", {"u": {"kind": "input"},
                     "a": {"kind": "llm", "system": "s", "policy": pol, "inputs": ["u"]},
                     "out": {"kind": "output", "inputs": ["a"]}}]
    t0 = time.monotonic()
    r = TestClient(app).post("/v1/chat/completions", json={"messages": MSG, "flow_ir": flow})
    assert time.monotonic() - t0 < 3
    assert r.status_code >= 400


# ---- M6: session reads are scoped to the calling owner -----------------------

def test_session_views_are_per_caller(host_store_clean, tmp_path):
    import host_store
    seed_call(session="shared-sid", provider="p", family="f", served_by="peer-a",
              caller="alice", tokens_in=10, cost_usd=1.0, ts=int(time.time()) - 5)
    seed_call(session="shared-sid", provider="q", family="g", served_by="peer-b",
              caller="bob", tokens_in=999, cost_usd=50.0)
    assert host_store.session_totals("shared-sid", "alice")["cost_usd"] == 1.0
    assert host_store.session_totals("shared-sid", "alice")["tokens_in"] == 10
    assert host_store.hot_route("shared-sid", "alice") == "p|f|peer-a"
    assert [w["served_by"] for w in host_store.session_warm("shared-sid", "alice")] == ["peer-a"]
    r = TestClient(create_app(_host(tmp_path))).get(
        "/x/session/shared-sid", headers={"x-llm-router-caller": "alice"})
    assert r.status_code == 200
    assert r.json()["calls"] == 1 and r.json()["cost_usd"] == 1.0


# ---- M2 / S9: router admin endpoints ----------------------------------------

def test_router_admin_endpoints_require_internal_secret(tmp_path, monkeypatch):
    import control_plane_client as cp
    monkeypatch.delenv("ROUTER_ADMIN_ALLOW_UNAUTHENTICATED", raising=False)
    client = TestClient(create_app(_host(tmp_path)))
    monkeypatch.setattr(cp, "CONTROL_PLANE_INTERNAL_SECRET", "")
    for path in ("/x/calls", "/x/sessions", "/x/wallet", "/x/session/abc"):
        assert client.get(path).status_code == 403, path
    for path in ("/x/providers", "/x/provider-key", "/x/config/reload", "/x/wallet/deposit"):
        assert client.post(path, json={}).status_code == 403, path
    monkeypatch.setattr(cp, "CONTROL_PLANE_INTERNAL_SECRET", "s3cret")
    assert client.get("/x/calls").status_code == 403
    assert client.get("/x/calls", headers={"x-internal-secret": "wrong"}).status_code == 403
    assert client.get("/x/calls", headers={"x-internal-secret": "s3cret"}).status_code != 403
    assert client.get("/x/runtime").status_code == 200   # Cloud reads it without a secret
    assert client.get("/openapi.json").status_code == 404


# ---- M3: BYO gateway reads are byte-capped -----------------------------------

def _byo_env(monkeypatch, handler):
    import byo_http
    monkeypatch.setattr(byo_http, "buyer_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    env = {"SAAS_TENANT_SCOPE": "7", "ANTSEED_BYO_TOKEN": "t" * 40,
           "ANTSEED_BYO_URL": "https://gw.example"}
    return env.get


BYO_REQ = {"provider_id": "antseed", "served_model_id": "m", "base_url": "https://gw.example/v1",
           "messages": MSG}


def test_byo_buffered_response_is_capped(monkeypatch):
    from provider_adapters.openai_compatible import make_async_call_provider
    monkeypatch.setenv("BYO_MAX_RESPONSE_BYTES", "1024")
    big = {"choices": [{"message": {"content": "a" * 5000}, "finish_reason": "stop"}], "usage": {}}
    env_get = _byo_env(monkeypatch, lambda _r: httpx.Response(200, json=big))
    res = asyncio.run(make_async_call_provider(env_get=env_get)(dict(BYO_REQ)))
    assert res["ok"] is False and res["error_kind"] == "bad_response"


def test_byo_stream_line_is_capped(monkeypatch):
    from provider_adapters.openai_compatible import stream_openai_compatible
    monkeypatch.setenv("BYO_MAX_SSE_LINE_BYTES", "1024")
    line = "data: " + json.dumps({"choices": [{"delta": {"content": "a" * 8000}}]})
    env_get = _byo_env(monkeypatch, lambda _r: httpx.Response(
        200, content=(line + "\n\ndata: [DONE]\n\n").encode()))

    async def emit(_d):
        pass

    res = asyncio.run(stream_openai_compatible(dict(BYO_REQ), emit, env_get=env_get))
    assert res["ok"] is False and res["error_kind"] == "bad_response"
    assert "exceeds" in res["error_message"]


# ---- L1: BYO offers cannot size the operator's shared peer gate --------------

def test_byo_peer_gate_is_tenant_scoped(monkeypatch):
    from provider_adapters import openai_compatible as oc
    monkeypatch.delenv("DISTRIBUTED_PEER_GATES", raising=False)
    monkeypatch.setenv("PEER_GATE_WAIT_S", "0.05")
    env = {"SAAS_TENANT_SCOPE": "7", "ANTSEED_BYO_TOKEN": "t" * 40, "ANTSEED_BYO_URL": "https://gw"}
    peer = "shared-peer-" + str(time.time())

    async def main():
        byo = {"provider_id": "antseed", "offer": {"peer_id": peer, "max_concurrency": 1}}
        op = {"provider_id": "antseed", "offer": {"peer_id": peer, "max_concurrency": 1}}
        tenant_slot, err1 = await oc._acquire_peer_capacity(byo, 1.0, env.get)
        operator_slot, err2 = await oc._acquire_peer_capacity(op, 1.0, lambda _k: None)
        for s in (tenant_slot, operator_slot):
            if s is not None:
                await s.release()
        return err1, err2

    assert asyncio.run(main()) == (None, None)


# ---- L2: NAT64 / IPv4-embedding addresses are not "public" -------------------

@pytest.mark.parametrize("addr", ["64:ff9b::7f00:1", "64:ff9b:1::a00:1", "::7f00:1",
                                  "::ffff:127.0.0.1", "2002:7f00:1::1", "10.0.0.1"])
def test_byo_rejects_embedded_private_addresses(addr, monkeypatch):
    import byo_http
    assert not byo_http.public_address(ipaddress.ip_address(addr))

    async def fake_getaddrinfo(*_a, **_k):
        return [(0, 0, 0, "", (addr, 443))]

    async def main():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "getaddrinfo", fake_getaddrinfo)
        async with byo_http.buyer_client() as c:
            with pytest.raises(httpx.ConnectError):
                await c.get("https://gw.example/")

    asyncio.run(main())


def test_byo_accepts_public_addresses():
    import byo_http
    assert byo_http.public_address(ipaddress.ip_address("8.8.8.8"))
    assert byo_http.public_address(ipaddress.ip_address("2606:4700::1111"))


# ---- L3: a tenant key never unlocks an operator-local provider ---------------

def test_tenant_ollama_key_does_not_unlock_local_ollama(tmp_path):
    cfg = tmp_path / "c.lua"
    cfg.write_text(f'local cfg = dofile("{ROOT}/tests/fixtures/managed.lua")\n'
                   'cfg.providers.ollama = { discovery = "marketplace", discovery_id = "ollama",'
                   ' base_url = "http://localhost:11434/v1", api_kind = "openai_compatible",'
                   ' auth_env = "OLLAMA_API_KEY", tier = "partner" }\n'
                   'cfg.providers.remote = { discovery = "static", base_url = "https://api.remote.example/v1",'
                   ' api_kind = "openai_compatible", auth_env = "REMOTE_API_KEY", tier = "partner" }\n'
                   'return cfg\n')
    base = LLMRouterHost(ROOT / "core/router.lua", cfg)
    base.init()
    child = base.for_tenant(9, {"OLLAMA_API_KEY": "tenant", "REMOTE_API_KEY": "tenant"})
    assert "ollama" not in child._tenant_allowed
    assert "remote" in child._tenant_allowed


# ---- M1: codex OAuth tokens only go to the configured upstream ---------------

class _Auth:
    def access_token(self):
        return "oauth-token"

    def account_id(self):
        return "acct"

    def select_account(self):
        return self


class _Resp:
    status_code = 200
    headers = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_lines(self):
        if False:
            yield ""

    async def aread(self):
        return b""


class _RecordingClient:
    def __init__(self, *a, **k):
        self.urls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, **_kw):
        self.urls.append(url)
        _RecordingClient.last = url
        return _Resp()

    async def aclose(self):
        pass


def test_codex_call_ignores_request_base_url(monkeypatch):
    import codex_backend as cb
    monkeypatch.setattr(httpx, "AsyncClient", _RecordingClient)
    call = cb.make_codex_async_call_provider(_Auth())
    asyncio.run(call({"served_model_id": "gpt-5.5", "messages": MSG,
                      "base_url": "https://evil.example/codex"}))
    assert _RecordingClient.last.startswith(cb.CODEX_BASE_URL)


def test_codex_stream_ignores_request_base_url():
    import codex_backend as cb
    from streaming import stream_codex
    client = _RecordingClient()

    async def emit(_d):
        pass

    asyncio.run(stream_codex({"served_model_id": "gpt-5.5", "messages": MSG,
                              "base_url": "https://evil.example/codex"},
                             emit, auth=_Auth(), client=client))
    assert client.urls and client.urls[0].startswith(cb.CODEX_BASE_URL)


# ---- L5: the codex broker bearer never rides plain http off-cluster ---------

def test_remote_codex_requires_https_off_cluster(monkeypatch):
    from remote_codex import RemoteCodexClient
    monkeypatch.delenv("CODEX_BROKER_ALLOW_HTTP", raising=False)
    with pytest.raises(ValueError):
        RemoteCodexClient("http://broker.example.com", "tok")
    with pytest.raises(ValueError):
        RemoteCodexClient("http://8.8.8.8:8090", "tok")
    for ok in ("https://broker.example.com", "http://127.0.0.1:8090",
               "http://llm-router-providers:8090", "http://broker.ns.svc.cluster.local:8090"):
        RemoteCodexClient(ok, "tok")


# ---- M5 (router side): the funded antseed proxy gets its bearer --------------

def test_antseed_proxy_token_sent_only_to_antseed(monkeypatch):
    from provider_adapters.openai_compatible import _prepare_openai_call
    monkeypatch.setenv("ANTSEED_PROXY_TOKEN", "p" * 40)
    prep, _ = _prepare_openai_call({"provider_id": "antseed", "served_model_id": "m",
                                    "base_url": "http://antseed:8378/v1", "auth": {"kind": "none"}},
                                   lambda _k: None, {}, 10)
    assert prep[2]["Authorization"] == "Bearer " + "p" * 40
    prep, _ = _prepare_openai_call({"provider_id": "other", "served_model_id": "m",
                                    "base_url": "https://x.example/v1", "auth": {"kind": "none"}},
                                   lambda _k: None, {}, 10)
    assert "Authorization" not in prep[2]


# ---- S3: provider overlay / env-secrets cannot touch infrastructure ----------

def test_overlay_rejects_infra_auth_env_private_urls_and_key_reuse(tmp_path):
    import provider_overlay as po
    h = _host(tmp_path)
    catalog = h.catalog()
    fam = next(iter(catalog["models"]))
    existing = next(p["auth_env"] for p in catalog["providers"].values()
                    if po.AUTH_ENV_RE.match(p.get("auth_env") or "") and not po.is_forbidden_auth_env(p["auth_env"]))

    def entry(**over):
        return {"base_url": "https://api.new.example/v1", "api_kind": "openai_compatible",
                "auth_env": "NEWPROV_API_KEY", "served_models": [{"family": fam}], **over}

    assert po.validate_entry("newprov", entry(), catalog) == []
    for bad in ("CONTROL_PLANE_INTERNAL_SECRET", "DATABASE_URL", "ANTSEED_CONTROL_TOKEN",
                "AWS_SESSION_TOKEN", "CODEX_BROKER_TOKEN", "EVIL\nX_API_KEY", "X_API_KEY\n"):
        assert any("auth_env" in e for e in po.validate_entry("newprov", entry(auth_env=bad), catalog)), bad
    for url in ("http://api.new.example/v1", "https://127.0.0.1/v1", "https://localhost/v1",
                "https://router:18080/v1", "https://169.254.169.254/v1"):
        assert any("base_url" in e for e in po.validate_entry("newprov", entry(base_url=url), catalog)), url
    assert any("existing provider" in e for e in po.validate_entry("newprov", entry(auth_env=existing), catalog))
    assert po.validate_entry("newprov", entry(auth_env=existing), catalog, key_supplied=True) == []


def test_env_secrets_never_override_infrastructure(tmp_path, monkeypatch):
    from env_secrets import load_env_secrets
    monkeypatch.setenv("DATABASE_URL", "postgresql://real")
    monkeypatch.setenv("CONTROL_PLANE_INTERNAL_SECRET", "real")
    monkeypatch.setenv("GROQ_API_KEY", "old")
    f = tmp_path / ".env.secrets"
    f.write_text("DATABASE_URL=postgresql://evil\nCONTROL_PLANE_INTERNAL_SECRET=evil\n"
                 "GROQ_API_KEY=new\nCALLER_KEYS_JSON={}\n")
    loaded = load_env_secrets(f)
    assert os.environ["DATABASE_URL"] == "postgresql://real"
    assert os.environ["CONTROL_PLANE_INTERNAL_SECRET"] == "real"
    assert os.environ["GROQ_API_KEY"] == "new"
    assert "CALLER_KEYS_JSON" in loaded


def test_codex_broker_non_ascii_bearer_is_401_not_500():
    import codex_broker
    app = codex_broker.create_app(store=object(), token="internal-token", client=object())
    r = TestClient(app, raise_server_exceptions=False).post(
        "/v1/call", json={}, headers={"authorization": "Bearer tést".encode()})
    assert r.status_code == 401
