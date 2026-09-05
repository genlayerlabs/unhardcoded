import asyncio
from pathlib import Path
import socket
import time
from unittest.mock import Mock

import httpx
import pytest

from llm_router_host import LLMRouterHost
from saas_routes import choices
import tenant_providers as tp

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def base(host_store_clean):
    tp._cache.clear()
    host = LLMRouterHost(ROOT/'core/router.lua', ROOT/'tests/fixtures/managed.lua',
        discover=lambda _: {'ok': True, 'offers': [{'model_family': 'OPERATOR-ONLY'}]})
    host.init()
    host.update_metrics('__credits', 'antseed', {'free_credits_remaining_usd': 999})
    return host


def aws(key, region='eu-west-1'):
    return {'bedrock': {'access_key_id': key, 'secret_access_key': 'secret-' + key,
                        'session_token': 'session-' + key, 'region': region}}


def buyer(token, host='buyer.example.com'):
    return {'antseed': {'gateway_url': 'https://' + host, 'token': token}}


def test_concurrent_bedrock_invocations_use_own_credentials_region_and_session(base, monkeypatch):
    import boto3
    import control_plane_client as cp
    from providers import _bedrock_adapter, _bedrock_stream_adapter
    seen = []
    def make(service, **kwargs):
        seen.append((service, kwargs))
        response = {'output': {'message': {'content': [{'text': kwargs['aws_access_key_id']}]}},
                    'usage': {'inputTokens': 1, 'outputTokens': 1}}
        return Mock(converse=lambda **_: response, converse_stream=lambda **_: {'stream': [
            {'contentBlockDelta': {'delta': {'text': kwargs['aws_access_key_id']}}}]})
    monkeypatch.setattr(boto3, 'client', make)
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'OPERATOR')
    call = _bedrock_adapter(10, cp.env_get)
    stream = _bedrock_stream_adapter(10, cp.env_get)
    async def run(tenant, key, region):
        child = base.for_tenant(tenant, {}, connections=aws(key, region))
        token = cp.activate_tenant_env(child._env)
        try:
            request = {'api_kind': 'bedrock', 'served_model_id': 'my-model',
                       'aws_region': 'us-east-1', 'messages': [{'role':'user','content':'hello'}]}
            result = await call(request)
            deltas = []
            async def emit(s): deltas.append(s)
            assert (await stream(request, emit))['ok']
            assert deltas == [key]
            return result['response']['text']
        finally:
            cp.reset_tenant_env(token)
    async def both():
        return await asyncio.gather(run(1, 'A', 'eu-west-1'), run(2, 'B', 'us-west-2'))
    assert asyncio.run(both()) == ['A', 'B']
    assert len(seen) == 4
    for _, config in seen:
        key = config['aws_access_key_id']
        assert config['aws_secret_access_key'] == 'secret-' + key
        assert config['aws_session_token'] == 'session-' + key
        assert config['region_name'] == {'A':'eu-west-1','B':'us-west-2'}[key]


def test_missing_aws_credentials_never_uses_operator_chain(monkeypatch):
    import boto3
    from provider_adapters.aws_credentials import client
    native = Mock()
    monkeypatch.setattr(boto3, 'client', native)
    with pytest.raises(ValueError):
        client('bedrock', 'us-east-1', {'SAAS_TENANT_SCOPE': '1'}.get)
    native.assert_not_called()


def test_bedrock_discovery_is_account_scoped_and_failure_never_uses_operator(base, monkeypatch):
    from sources.bedrock import BedrockSource
    seen = []
    async def pricing(self):
        key = self._env_get('AWS_ACCESS_KEY_ID')
        seen.append(key)
        if key == 'revoked':
            raise RuntimeError('secret must not leak')
        self._offers_by_provider = {pid: [{'model_family':'shared-model','wire_model_id': key + '.profile',
            'price_in_usd_per_mtok':1, 'price_out_usd_per_mtok':2}] for pid in self.provider_ids}
        return []
    monkeypatch.setattr(BedrockSource, 'pricing', pricing)
    async def run():
        for tid, key in [(1,'A'), (2,'B'), (1,'revoked')]:
            child = base.for_tenant(tid, {}, connections=aws(key))
            await tp.prepare(child)
            if key == 'revoked':
                assert choices(child) == []
                assert 'secret' not in str(child._connection_errors)
            else:
                assert child._tenant_offers['bedrock'][0]['wire_model_id'] == key + '.profile'
                assert {r['provider'] for r in choices(child)} == {'bedrock','bedrock_market'}
    asyncio.run(run())
    assert seen == ['A','B','revoked']
    assert choices(base.for_tenant(1, {})) == []


def test_antseed_buyer_offers_funding_and_wire_are_tenant_scoped(base, monkeypatch):
    import byo_http
    from provider_adapters.openai_compatible import make_async_call_provider, stream_openai_compatible
    calls = []
    def network(request):
        token = request.headers['authorization'].removeprefix('Bearer ')
        calls.append((request.url.host, token, request.url.path, request.headers.get('x-antseed-pin-peer')))
        if request.url.path == '/snapshot':
            return httpx.Response(200, json={'buyer_status': {'fetched_at': 0 if token == 'stale' else int(time.time()*1000),
                'deposits_available': 0 if token == 'empty' else 5}, 'peer_offers': [{
                    'observed_at': int(time.time()*1000), 'peer_id': 'peer-' + token,
                    'service': 'shared-model', 'price_in': 1, 'price_out': 2}]})
        import json
        if json.loads(request.content).get('stream'):
            chunk = json.dumps({'choices': [{'delta': {'content': token}}]})
            return httpx.Response(200, headers={'content-type':'text/event-stream'},
                                  content=f'data: {chunk}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={'choices': [{'message': {'content': token}}]})
    monkeypatch.setattr(byo_http, 'buyer_client', lambda: httpx.AsyncClient(transport=httpx.MockTransport(network)))
    async def run():
        for tenant, token in [(1, 'A'), (2, 'B'), (3, 'empty'), (4, 'stale')]:
            child = base.for_tenant(tenant, {}, connections=buyer(token, token.lower() + '.example.com'))
            await tp.prepare(child)
            if token in ('empty', 'stale'):
                assert choices(child) == []  # Operator's 999 credits must not help.
                continue
            assert {r['provider'] for r in choices(child)} == {'antseed'}
            offer = child._tenant_offers['antseed'][0]
            assert offer['peer_id'] == 'peer-' + token
            # Even a forged offer cannot exfiltrate a buyer's connection token.
            offer['seller_endpoint'] = 'https://evil.example.com/v1'
            result = await make_async_call_provider(env_get=child._env.get)({
                'provider_id':'antseed','served_model_id':'shared-model','offer':offer,
                'base_url': 'https://evil.example.com/v1','messages':[{'role':'user','content':'hello'}]})
            assert result['ok'] and result['response']['text'] == token
            deltas = []
            async def emit(s): deltas.append(s)
            result = await stream_openai_compatible({'provider_id':'antseed',
                'served_model_id':'shared-model','offer':offer,'base_url':'https://evil.example.com/v1',
                'messages':[{'role':'user','content':'hello'}]}, emit, env_get=child._env.get)
            assert result['ok'] and deltas == [token]
    asyncio.run(run())
    assert [(host, token, peer) for host, token, path, peer in calls if path.endswith('completions')] == [
        ('a.example.com','A','peer-A'), ('a.example.com','A','peer-A'),
        ('b.example.com','B','peer-B'), ('b.example.com','B','peer-B')]


@pytest.mark.parametrize('ip', ['127.0.0.1', '169.254.169.254', '10.0.0.1', '::1', '::ffff:127.0.0.1'])
def test_buyer_egress_rejects_private_dns_even_after_save(ip, monkeypatch):
    from byo_http import PublicHTTPS
    async def run():
        async def resolve(*a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, 443))]
        monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', resolve)
        async with httpx.AsyncClient(transport=PublicHTTPS()) as client:
            with pytest.raises(httpx.ConnectError):
                await client.get('https://buyer.example.com/snapshot')
    asyncio.run(run())


def test_public_dns_is_pinned_and_tls_hostname_preserved(monkeypatch):
    from byo_http import PublicHTTPS
    seen = []
    async def run():
        async def resolve(*a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443))]
        monkeypatch.setattr(asyncio.get_running_loop(), 'getaddrinfo', resolve)
        transport = PublicHTTPS()
        async def wire(request):
            seen.append(request)
            return httpx.Response(200, json={})
        await transport.inner.aclose()
        transport.inner = httpx.MockTransport(wire)
        async with httpx.AsyncClient(transport=transport) as client:
            await client.get('https://buyer.example.com/snapshot')
    asyncio.run(run())
    assert seen[0].url.host == '8.8.8.8'
    assert seen[0].headers['host'] == 'buyer.example.com'
    assert seen[0].extensions['sni_hostname'] == 'buyer.example.com'
