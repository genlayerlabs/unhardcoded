"""Managed native and marketplace providers use the existing engine/adapters."""
import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import control_plane_client as cp

from llm_router_host import LLMRouterHost
from provider_connections import connections, credential_names
from saas_routes import choices, compile_intent, preview
from shim import create_app
from test_control_plane_client import _FakeResp, _install

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def managed_host(monkeypatch):
    monkeypatch.setenv('SAAS_SHARED_PROVIDERS', 'antseed,bedrock,bedrock_market')
    def discover(pid):
        if pid == 'antseed':
            return {'ok':True,'offers':[{'model_family':'shared-model','wire_model_id':'peer.model',
                'peer_id':'peer-one','seller_endpoint':'http://buyer.test/v1',
                'price_in_usd_per_mtok':1,'price_out_usd_per_mtok':2}]}
        return {'ok':True,'offers':[]}
    base = LLMRouterHost(ROOT/'core/router.lua', ROOT/'tests/fixtures/managed.lua',
                         discover=discover, now_ms=lambda:1000)
    base.init()
    base.update_metrics('__credits', 'antseed', {'free_credits_remaining_usd':5})
    return base


def test_managed_catalog_and_actual_engine_admission(managed_host):
    child = managed_host.for_tenant(7, {}, ['antseed','bedrock','bedrock_market'])
    assert {r['provider'] for r in choices(child)} == {'antseed','bedrock'}
    info = {c['id']:c for c in connections(child)}
    assert info['antseed']['mode'] == 'buyer' and info['antseed']['connected']
    assert info['bedrock']['mode'] == 'aws'
    assert info['bedrock']['shared_providers'] == ['bedrock', 'bedrock_market'] or set(info['bedrock']['shared_providers']) == {'bedrock','bedrock_market'}
    out = preview(child, {'targets':['antseed|shared-model','bedrock|shared-model']})
    assert [r['provider'] for r in out['ranked']] == ['antseed','bedrock']


def test_no_implicit_operator_access_and_grants_must_be_exposed(managed_host, monkeypatch):
    assert choices(managed_host.for_tenant(8, {})) == []
    monkeypatch.setenv('SAAS_SHARED_PROVIDERS', '')
    assert choices(managed_host.for_tenant(7, {}, ['antseed','bedrock'])) == []


def test_antseed_funding_gate_is_preserved(managed_host):
    managed_host.update_metrics('__credits', 'antseed', {'free_credits_remaining_usd':0})
    # Core EMA updates can smooth observations; force the snapshot to zero.
    state = managed_host.dump_state()
    state['ema_metrics']['__credits|antseed']['free_credits_remaining_usd'] = 0
    managed_host.restore_state(state)
    child = managed_host.for_tenant(7, {}, ['antseed','bedrock'])
    assert {r['provider'] for r in choices(child)} == {'bedrock'}


def test_catalog_declares_new_byok_provider_without_saas_whitelist(managed_host):
    assert 'CUSTOM_CLOUD_SECRET' in credential_names(managed_host.catalog())
    child = managed_host.for_tenant(7, {'CUSTOM_CLOUD_SECRET':'tenant-secret'})
    assert {r['provider'] for r in choices(child)} == {'custom_cloud'}
    assert preview(child, {'targets':['custom_cloud|shared-model']})['ranked']


def test_antseed_to_native_bedrock_fallback_uses_real_adapters(managed_host):
    from provider_adapters.openai_compatible import make_async_call_provider
    from provider_adapters.bedrock import make_bedrock_async_call_provider
    from provider_adapters.dispatcher import make_api_kind_dispatcher
    seen = []
    def buyer(request):
        seen.append(('buyer', request.headers.get('x-antseed-pin-peer')))
        return httpx.Response(503, json={'error':{'message':'unavailable'}})
    class AWS:
        def converse(self, **request):
            seen.append(('aws', request['modelId']))
            return {'output':{'message':{'content':[{'text':'Bedrock fallback'}]}},
                    'usage':{'inputTokens':2,'outputTokens':2},'stopReason':'end_turn'}
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(buyer)) as http:
            managed_host.set_async_call_hook(make_api_kind_dispatcher(
                make_async_call_provider(client=http),
                {'bedrock':make_bedrock_async_call_provider(client=AWS())}))
            child = managed_host.for_tenant(7, {}, ['antseed','bedrock'])
            return await child.execute_async({'policy_ir':compile_intent({'targets':['antseed|shared-model','bedrock|shared-model']}),
                                               'messages':[{'role':'user','content':'hello'}]})
    result = asyncio.run(run())
    assert result['ok'], result
    assert result['response']['text'] == 'Bedrock fallback'
    assert seen == [('buyer','peer-one'),('aws','aws.profile.model')]


@pytest.fixture
def scope_client(managed_host, monkeypatch):
    monkeypatch.setattr(cp, 'CONTROL_PLANE_URL', 'http://cp.test')
    monkeypatch.setattr(cp, 'CONTROL_PLANE_INTERNAL_SECRET', 'test-secret')
    tenant_headers = {'x-llm-router-tenant': '7', 'x-internal-secret': 'test-secret'}
    headers = {**tenant_headers, 'x-unhardcoded-scope-version': '2',
               'x-unhardcoded-project': '4', 'x-unhardcoded-environment': '9'}
    payload = {'scope_version': 2, 'tenant_id': 7, 'project_id': 4, 'environment_id': 9, 'env': {}}
    return TestClient(create_app(managed_host)), headers, tenant_headers, payload


@pytest.mark.parametrize('authorization', [
    {'managed_providers': None},
    {'managed_providers': 'custom_cloud'},
    {'managed_providers': [{'id': 'custom_cloud'}]},
    {'managed_providers': ['custom_cloud', 'custom_cloud']},
    {'managed_providers': ['custom_cloud'] * 129},
    {'managed_providers': ['custom_cloud', 'unknown']},
    {'managed_providers': ['bedrock']},
    {'managed_providers': ['bedrock_market'], 'environment_id': 10},
    {'managed_providers': ['custom_cloud', 'bedrock_market'], 'env': {'CUSTOM_CLOUD_SECRET': 'tenant'}},
    {'managed_providers': ['custom_cloud'], 'connections': {'bedrock': {'region': 'test'}}},
])
def test_scope_rejects_invalid_managed_grants(scope_client, monkeypatch, authorization):
    client, headers, _, payload = scope_client
    monkeypatch.setenv('SAAS_SHARED_PROVIDERS', 'custom_cloud,bedrock_market,unknown')
    fake = _install(monkeypatch, [_FakeResp(200, {**payload, **authorization})])
    response = client.get('/x/saas/catalog', headers=headers)
    assert response.status_code == 503
    assert len(fake.calls) == 1


def test_scoped_request_executes_with_managed_credentials_until_grant_removed(
        managed_host, scope_client, monkeypatch):
    client, headers, tenant_headers, payload = scope_client
    monkeypatch.setenv('SAAS_SHARED_PROVIDERS', 'custom_cloud,bedrock')
    managed_host.set_env('CUSTOM_CLOUD_SECRET', 'operator-test-key')
    granted = {**payload, 'managed_providers': ['custom_cloud']}
    fake = _install(monkeypatch, [
        _FakeResp(200, granted), _FakeResp(200, granted), _FakeResp(200, payload), _FakeResp(200, granted),
    ])
    calls = []

    async def provider(request):
        calls.append((request['provider_id'], cp.env_get('CUSTOM_CLOUD_SECRET')))
        return {'ok': True, 'latency_ms': 1, 'response': {'text': 'done', 'tokens_in': 1, 'tokens_out': 1}}

    managed_host.set_async_call_hook(provider)
    catalog = client.get('/x/saas/catalog', headers=headers)
    assert catalog.status_code == 200
    assert {model['provider'] for model in catalog.json()['models']} == {'custom_cloud'}
    response = client.post('/x/saas/test', headers=headers, json={
        'intent': {'targets': ['custom_cloud|shared-model']}, 'prompt': 'Hello',
    })
    assert response.status_code == 200, response.text
    assert calls == [('custom_cloud', 'operator-test-key')]
    assert client.get('/x/saas/catalog', headers=headers).json()['models'] == []
    assert client.get('/x/saas/catalog', headers=tenant_headers).json()['models'] == []
    assert [call['url'].split('/internal/')[1] for call in fake.calls] == [
        'tenants/7/projects/4/environments/9/provider-env',
    ] * 3 + ['tenants/7/provider-env']
