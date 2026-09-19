"""Ingress <-> external control plane integration (auth fallback + /internal/*).

The control plane's HTTP side is a fake client on control_plane_client._client;
the upstream router is a fake on auth_proxy._client. Store-backed tests use the
shared Postgres fixture (host_store_clean).
"""
from __future__ import annotations

import hashlib
import json
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
        if '/routes/' in url:
            return httpx.Response(200, request=httpx.Request('GET', url), json={
                'route':'route:production', 'revision':2, 'policy_id':'a'*64,
                'policy_ir':['policy', ['top'], ['zero'], ['ordered'], ['id'], ['always', {'action':'abort'}]],
                'execution':{'timeout_ms':8000, 'first_token_timeout_ms':8000}})
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
        self.requests.append({"method": method, "url": url, "headers": headers or {}, "body": content})
        return object()

    async def send(self, req, stream=True):
        return _FakeUpstreamResp()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    cpc.reset_for_tests()
    monkeypatch.setattr(cpc, "CONTROL_PLANE_URL", "http://cp.test")
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "s3cret")
    yield
    cpc.reset_for_tests()


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
                       json={"model": "route:production", "messages": []})


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
    assert fwd["x-internal-secret"] == "s3cret"
    assert fwd["x-unhardcoded-revision"] == "2"
    assert fwd["x-unhardcoded-policy-id"] == 'a'*64
    assert json.loads(upstream.requests[0]['body'])['policy_ir'][0] == 'policy'


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


def test_saas_policy_override_and_endpoint_escape_are_blocked(monkeypatch):
    require_host_store()
    _cp(monkeypatch, [{'active':True, 'consumer':'acme', 'tenant_id':7}])
    upstream = _upstream(monkeypatch)
    client = TestClient(auth_proxy.app)
    headers = {'Authorization':'Bearer tok-route', 'x-unhardcoded-policy-id':'forged'}
    for model in ('pin:platform/forbidden', 'profile:default'):
        assert client.post('/v1/chat/completions', headers=headers, json={'model':model}).status_code == 400
    assert client.get('/v1/models', headers=headers).status_code == 400
    r = client.post('/v1/chat/completions', headers=headers, json={
        'model':'route:production', 'messages':[], 'policy_ir':['forged'],
        'flow_ir':['forged'], 'timeout_ms':999999, 'first_token_timeout_ms':999999})
    assert r.status_code == 200
    body = json.loads(upstream.requests[0]['body'])
    assert body['policy_ir'][0] == 'policy' and 'flow_ir' not in body
    assert body['timeout_ms'] == body['first_token_timeout_ms'] == 8000
    assert upstream.requests[0]['headers']['x-unhardcoded-policy-id'] == 'a'*64


@pytest.mark.parametrize('path', ['/v1/chat/completions', '/v1/responses'])
def test_only_published_preferences_reach_upstream(monkeypatch, path):
    require_host_store()
    _cp(monkeypatch, [{'active':True, 'consumer':'acme', 'tenant_id':7}])
    upstream = _upstream(monkeypatch)
    default = ['policy', ['top'], ['zero'], ['ordered'], ['id'], ['always', {'action':'abort'}]]
    variant = ['policy', ['bottom'], ['zero'], ['ordered'], ['id'], ['always', {'action':'abort'}]]
    async def resolve(*args):
        return {'revision': 1, 'policy_ir': default, 'policy_id': 'a'*64,
                'execution': {'timeout_ms': 8000, 'first_token_timeout_ms': 8000},
                'preferences': {'cost': {'policy_ir': variant, 'policy_id': 'b'*64}}}
    monkeypatch.setattr(cpc, 'resolve_route', resolve)
    client = TestClient(auth_proxy.app)
    headers = {'Authorization':'Bearer preferences', 'x-unhardcoded-preference':'forged'}
    body = {'model':'route:production', 'messages':[], 'routing_preference':'speed'}
    response = client.post(path, headers=headers, json=body)
    assert response.status_code == 400 and response.json()['error']['code'] == 'preference_not_allowed'
    assert upstream.requests == []
    response = client.post(path, headers=headers, json={**body, 'routing_preference':'cost',
        'policy_ir':['forged'], 'timeout_ms': 999999, 'flow_ir':['forged']})
    assert response.status_code == 200
    routed = json.loads(upstream.requests[0]['body'])
    assert routed['policy_ir'] == variant
    assert routed['timeout_ms'] == 8000 and 'flow_ir' not in routed and 'routing_preference' not in routed
    assert upstream.requests[0]['headers']['x-unhardcoded-policy-id'] == 'b'*64
    assert upstream.requests[0]['headers']['x-unhardcoded-preference'] == 'cost'


def test_route_resolution_failure_does_not_call_router(monkeypatch):
    require_host_store()
    _cp(monkeypatch, [{'active':True, 'consumer':'acme', 'tenant_id':7}])
    upstream = _upstream(monkeypatch)
    async def unavailable(*args):
        raise cpc.RouteUnavailable('Route unavailable')
    monkeypatch.setattr(cpc, 'resolve_route', unavailable)
    assert _post_chat(TestClient(auth_proxy.app), 'tok-route').status_code == 503
    assert upstream.requests == []


@pytest.mark.parametrize('path', ['/v1/chat/completions', '/v1/responses'])
@pytest.mark.parametrize('scoped', [False, True])
def test_full_ingress_to_engine_alias_and_failover(monkeypatch, path, scoped):
    """Actual ingress + shim + Lua, with only CP HTTP and providers faked."""
    require_host_store()
    import asyncio
    from llm_router_host import LLMRouterHost
    from saas_routes import compile_intent
    from shim import create_app
    term = compile_intent({'targets':['openai|primary', 'anthropic|backup']})
    host = LLMRouterHost(ROOT/'core/router.lua', ROOT/'tests/fixtures/saas.lua')
    host.init()
    seen = []
    async def call(request):
        seen.append((request['provider_id'], cpc.env_get('OPENAI_API_KEY')))
        if request['provider_id'] == 'openai':
            return {'ok':False, 'error_kind':'server_error'}
        return {'ok':True, 'latency_ms':10, 'response':{'text':'Fallback works', 'tokens_in':2, 'tokens_out':3}}
    host.set_async_call_hook(call)
    scope = {'scope_version': 2, 'tenant_id': 7, 'project_id': 8, 'environment_id': 9} if scoped else {}
    def cp_http(request):
        if scoped and ('/provider-env' in request.url.path or '/routes/' in request.url.path):
            assert '/projects/8/environments/9/' in request.url.path
        if scoped and '/routes/' in request.url.path:
            assert request.url.params['key_sha256'] == hashlib.sha256(b'full-chain').hexdigest()
        if request.url.path.endswith('/provider-env'):
            data = {'env':{'OPENAI_API_KEY':'sk-tenant', 'ANTHROPIC_API_KEY':'sk-backup'}}
        elif '/routes/' in request.url.path:
            data = {'policy_ir':term, 'policy_id':host.normalize_policy(term)['policy_id'],
                    'revision':3, 'execution':{'timeout_ms':8000}, 'route':'route:production'}
        else:
            data = {'active':True, 'consumer':'acme', 'tenant_id':7}
        return httpx.Response(200, json={**data, **scope})
    cp_client = httpx.AsyncClient(transport=httpx.MockTransport(cp_http))
    shim_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(host)), base_url='http://router.test')
    monkeypatch.setattr(cpc, '_client', cp_client)
    monkeypatch.setattr(auth_proxy, '_client', shim_client)
    body = {'model':'route:production', 'messages':[{'role':'user','content':'hi'}]} if 'chat' in path else {'model':'route:production','input':'hi'}
    try:
        r = TestClient(auth_proxy.app).post(path, headers={'Authorization':'Bearer full-chain',
            'x-unhardcoded-scope-version': '2', 'x-unhardcoded-project': '999',
            'x-unhardcoded-environment': '999'}, json=body)
        assert r.status_code == 200, r.text
        trace = r.json()['x_router']['decision_trace']
        assert trace['route_revision'] == '3'
        assert trace['route'] == 'route:production'
        if scoped:
            assert trace['project_id'] == 8 and trace['environment_id'] == 9
        else:
            assert 'project_id' not in trace and 'environment_id' not in trace
        assert [s['provider_id'] for s in trace['decision_path'] if s['event'] == 'attempted'] == ['openai', 'anthropic']
        assert seen == [('openai','sk-tenant'), ('anthropic','sk-tenant')]
        host_store._write_q.join()
        row = host_store.recent_calls(caller='acme')[0]
        assert row['routing_summary']['route_revision'] == '3'
        assert row['routing_summary']['attempts'][0]['error_kind'] == 'server_error'
    finally:
        asyncio.run(cp_client.aclose())
        asyncio.run(shim_client.aclose())


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


def test_scoped_usage_filters_before_aggregate_and_recent_limit():
    require_host_store()
    now = int(time.time())
    for caller, project, env, tokens in [('acme', 1, 10, 7), ('acme', 1, 11, 90),
                                        ('acme', 2, 10, 800), ('other', 1, 10, 900),
                                        ('acme', None, None, 1000)]:
        trace = {'project_id': project, 'environment_id': env} if project else None
        host_store.insert_call({'ts': now, 'caller': caller, 'status': 200, 'tokens_in': tokens,
                                'decision_trace': trace, 'cost_usd': tokens / 1000})
    client = TestClient(auth_proxy.app)
    headers = {'x-internal-secret': 's3cret'}
    scope = {'caller': 'acme', 'project_id': 1, 'environment_id': 10}
    response = client.get('/internal/usage', params={**scope, 'bucket': 'day'}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data['runs'] == 1 and data['tokens_in'] == 7
    assert data['cost_usd'] == pytest.approx(.007)
    assert sum(b['runs'] for b in data['buckets']) == 1
    assert {k: data[k] for k in scope} == scope
    assert data['scope_version'] == 2
    response = client.get('/internal/usage/recent', params={**scope, 'limit': 1}, headers=headers)
    assert response.status_code == 200
    assert response.json()['calls'][0]['tokens_in'] == 7
    for path in ('/internal/usage', '/internal/usage/recent'):
        for invalid in ({'project_id': 1}, {'environment_id': 10}, {'project_id': 0, 'environment_id': 10}):
            assert client.get(path, params={'caller': 'acme', **invalid}, headers=headers).status_code == 400


def test_scoped_metering_outage_is_unavailable_not_successful_zero(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError('database unavailable')
    monkeypatch.setattr(host_store, '_get_pool', unavailable)
    client = TestClient(auth_proxy.app)
    for path in ('/internal/usage', '/internal/usage/recent'):
        result = client.get(path, params={'caller': 'acme', 'project_id': 1, 'environment_id': 2},
                            headers={'x-internal-secret': 's3cret'})
        assert result.status_code == 503
        assert result.json() == {'error': 'ledger_unavailable'}


def test_plaintext_upstream_never_receives_trusted_scope_secret(monkeypatch):
    _upstream(monkeypatch)
    monkeypatch.setattr(cpc, 'ALLOW_INSECURE_HTTP', False)
    monkeypatch.setattr(auth_proxy, 'UPSTREAM', 'http://router.test')
    async def auth(token):
        return {'ok': True, 'caller': 'transport-fixture', 'tenant_id': 1, 'digest': 'a' * 64, 'meta': {}}
    async def route(*args, **kwargs):
        return {'route': 'route:production', 'revision': 1, 'policy_id': 'a' * 64,
                'policy_ir': ['policy'], 'execution': {}}
    monkeypatch.setattr(auth_proxy, '_caller_auth_async', auth)
    monkeypatch.setattr(cpc, 'resolve_route', route)
    monkeypatch.setattr(auth_proxy, '_rate_ok', lambda *args: True)
    result = _post_chat(TestClient(auth_proxy.app), 'fixture')
    assert result.status_code == 503
    assert result.json()['error']['code'] == 'bridge_transport_unavailable'
    assert auth_proxy._client.requests == []


@pytest.mark.parametrize('model', [None, '', 'auto'])
@pytest.mark.parametrize('path', ['/v1/chat/completions', '/v1/responses'])
def test_automatic_key_resolves_default_route_without_caller_policy(monkeypatch, model, path):
    require_host_store()
    _cp(monkeypatch, [{'active': True, 'consumer': 'auto-app', 'tenant_id': 7,
        'scope_version': 2, 'project_id': 8, 'environment_id': 9}])
    upstream = _upstream(monkeypatch)
    async def resolve(tenant_id, name, **scope):
        assert name == '__key_default__' and scope['environment_id'] == 9
        return {'route': 'route:bound', 'revision': 1, 'policy_id': 'a' * 64,
            'policy_ir': ['policy'], 'execution': {'automatic': {'trusted': True}, 'automatic_mode': 'shadow'}}
    monkeypatch.setattr(cpc, 'resolve_route', resolve)
    body = {'messages': [], '_auto_contract': {'forged': True}, 'flow_ir': ['forged']}
    if model is not None:
        body['model'] = model
    result = TestClient(auth_proxy.app).post(path, headers={'Authorization': 'Bearer automatic-key'}, json=body)
    assert result.status_code == 200, result.text
    sent = json.loads(upstream.requests[0]['body'])
    assert sent['model'] == 'route:bound' and 'flow_ir' not in sent
    assert sent['_auto_contract'] == {'automatic': {'trusted': True}, 'automatic_mode': 'shadow'}
