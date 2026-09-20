import asyncio
import copy
import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from decision_protocol import validate_payload, validate_response
from llm_router_host import LLMRouterHost
from provider_adapters.openai_compatible import make_async_call_provider
from shim import create_app
from sources.antseed import AntSeedSource, STALE_AFTER_S
from sources.openrouter import OpenRouterSource

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / 'policies/jev-value-v1.json').read_text())
PAYLOAD = {'state': {'ticket': 'Charged twice'}, 'questions': {
    'team': {'type': 'choice', 'instructions': 'Which department?',
             'criteria': {'billing': 'Payments', 'technical': 'Bugs'}}}}
ANSWER = {'model': 'jev-1.13', 'answers': {'team': {'type': 'choice', 'choice': 'billing',
          'probabilities': {'billing': .9, 'technical': .1}, 'confidence': .8}},
          'usage': {'input_tokens': 100, 'cost': .0000042}}


@pytest.fixture
def routed(tmp_path, monkeypatch, host_store_clean):
    config = tmp_path / 'config.lua'
    config.write_text('''return {providers={
      openrouter_market={discovery="marketplace",discovery_id="openrouter_market",api_kind="openai_compatible",auth_env="OR_KEY"},
      antseed={discovery="marketplace",discovery_id="antseed",api_kind="openai_compatible"}},
      models={},profiles={default={scorer={"zero"}}},
      policy_envelope={"and",{"meets_req"},{"not",{"is","disabled"}}}}''')
    offers = {
        'openrouter_market': [{'model_family': 'jev-1.13', 'wire_model_id': 'typesafe/jev-1.13',
            'protocol': 'decisions', 'seller_endpoint': 'https://openrouter.ai/api/v1',
            'price_in_usd_per_mtok': .042, 'price_out_usd_per_mtok': 0}],
        'antseed': [{'model_family': 'jev-1.13', 'wire_model_id': 'jev-1.13',
            'protocol': 'decisions', 'seller_endpoint': 'http://buyer.test/v1', 'peer_id': 'peer',
            'price_in_usd_per_mtok': .01, 'price_out_usd_per_mtok': 0,
            'max_concurrency': 1}],
    }
    # A zero-price chat route must never receive a decision, including fallback.
    offers['openrouter_market'].append({'model_family': 'chat', 'wire_model_id': 'chat',
        'seller_endpoint': 'https://openrouter.ai/api/v1',
        'price_in_usd_per_mtok': 0, 'price_out_usd_per_mtok': 0})
    host = LLMRouterHost(router_path=ROOT / 'core/router.lua', config_path=config,
                         env={'OR_KEY': 'provider-secret'})
    host.set_discover_hook(lambda name: {'ok': True, 'offers': offers[name]})
    host.init()
    calls = []
    behavior = {'antseed_status': 503, 'openrouter_status': 200}

    async def transport(request):
        calls.append(request)
        body = json.loads(request.content)
        assert set(body) == {'model', 'state', 'questions'}
        assert body['state'] == PAYLOAD['state']
        assert body['questions'] == PAYLOAD['questions']
        if request.url.host == 'buyer.test':
            assert request.url.path == '/v1/systemone'
            assert request.headers['x-antseed-pin-peer'] == 'peer'
            assert 'authorization' not in request.headers
            status = behavior['antseed_status']
        else:
            assert str(request.url) == 'https://openrouter.ai/api/alpha/decisions'
            assert request.headers['authorization'] == 'Bearer provider-secret'
            status = behavior['openrouter_status']
        return httpx.Response(status, json=ANSWER if status == 200 else {'error': 'unavailable'})
    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    host.set_async_call_hook(make_async_call_provider(client=client, env_get=lambda name: 'provider-secret' if name == 'OR_KEY' else None))
    return host, offers, calls, behavior


@pytest.mark.parametrize('path', ['/v1/decisions', '/v1/systemone'])
def test_real_engine_routes_decisions_and_falls_back_with_usage_and_trace(routed, path):
    host, _, calls, _ = routed
    with TestClient(create_app(host)) as client:
        response = client.post(path, json={**PAYLOAD, 'policy_ir': POLICY})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['answers'] == ANSWER['answers']
    assert result['usage']['input_tokens'] == 100
    assert result['x_router']['provider'] == 'openrouter_market'
    assert result['x_router']['cost_usd'] == .0000042
    assert len(calls) == 2
    assert [r.url.host for r in calls] == ['buyer.test', 'openrouter.ai']


def test_primary_antseed_success_does_not_call_openrouter(routed):
    host, _, calls, behavior = routed
    behavior['antseed_status'] = 200
    response = TestClient(create_app(host)).post('/v1/decisions', json={**PAYLOAD, 'policy_ir': POLICY})
    assert response.status_code == 200
    assert response.json()['x_router']['served_by'] == 'peer'
    assert len(calls) == 1


def test_failure_does_not_fall_back_to_chat(routed):
    host, _, calls, behavior = routed
    behavior['openrouter_status'] = 503
    response = TestClient(create_app(host)).post('/v1/decisions', json={**PAYLOAD, 'policy_ir': POLICY})
    assert response.status_code >= 500
    assert len(calls) == 2


@pytest.mark.parametrize('model', ['family:jev-1.13', 'pin:antseed/jev-1.13'])
def test_chat_and_streaming_cannot_reach_jev(routed, model):
    host, _, calls, _ = routed
    response = TestClient(create_app(host)).post('/v1/chat/completions', json={
        'model': model, 'stream': True, 'messages': [{'role': 'user', 'content': 'hello'}]})
    assert response.status_code == 503
    assert not calls


@pytest.mark.parametrize('extra', [{'stream': True}, {'tools': []}, {'timeout_ms': -1},
    {'state': 7}, {'questions': {}}, {'questions': {'q': {'type': 'text', 'instructions': 'hi'}}}])
def test_invalid_decision_payloads_rejected_before_spend(routed, extra):
    host, _, calls, _ = routed
    response = TestClient(create_app(host)).post('/v1/decisions', json={**PAYLOAD, **extra})
    assert response.status_code in (400, 422)
    assert not calls


def test_catalog_category_and_protocol_preview(routed, monkeypatch):
    import sources
    host, _, calls, _ = routed
    monkeypatch.setattr(sources, 'SOURCE_STATE', {'openrouter': {'book': {'rows': [
        {'model_family': 'jev-1.13', 'category': 'decision'}, {'model_family': 'chat'}]}}})
    client = TestClient(create_app(host))
    models = client.get('/v1/models?type=decisions').json()['data']
    assert models == [{'id': 'family:jev-1.13', 'object': 'model', 'type': 'decision', 'category': 'Decision models'}]
    assert client.post('/x/rank', json={'policy_ir': POLICY}).json()['ranked'] == []
    assert len(client.post('/x/rank', json={'policy_ir': POLICY, 'protocol': 'decisions'}).json()['ranked']) == 2
    assert not calls


@pytest.mark.asyncio
async def test_decision_timeout_cancels_transport_and_releases_peer_slot(monkeypatch):
    cancelled = asyncio.Event()
    async def slow(request):
        try:
            await asyncio.sleep(30)
        finally:
            cancelled.set()
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        call = make_async_call_provider(client=client, env_get=lambda key: 'key')
        result = await call({'protocol': 'decisions', 'decision': PAYLOAD,
            'provider_id': 'openrouter', 'auth_env': 'KEY', 'base_url': 'https://openrouter.ai/api/v1',
            'served_model_id': 'typesafe/jev-1.13', 'timeout_ms': 1000, 'first_token_timeout_ms': 10})
    assert result['error_kind'] == 'timeout' and cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('body', [b'{"model":"x","model":"y"}', b'x' * 131073,
    json.dumps({**ANSWER, 'answers': {'team': {'type': 'choice', 'choice': 'unknown'}}}).encode()])
async def test_invalid_upstream_responses_fail_closed(body):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=body))) as client:
        result = await make_async_call_provider(client=client, env_get=lambda key: 'key')({
            'protocol': 'decisions', 'decision': PAYLOAD, 'provider_id': 'openrouter',
            'auth_env': 'KEY', 'base_url': 'https://openrouter.ai/api/v1', 'served_model_id': 'typesafe/jev-1.13'})
    assert result['error_kind'] == 'bad_response'


def test_all_three_question_types_are_validated():
    payload = {'state': ['text'], 'questions': {
        'q': {'type': 'noul', 'instructions': 'Urgent?'},
        's': {'type': 'score', 'instructions': 'Priority?', 'criteria': ['Low', 'High']}}}
    validate_payload(payload)
    validate_response({'model': 'jev', 'answers': {'q': {'type': 'noul', 'noul': .7},
        's': {'type': 'score', 'score': .7, 'probabilities': {'0': .3, '1': .7}}}}, payload)
    with pytest.raises(ValueError):
        validate_response({'model': 'jev', 'answers': {'q': {'type': 'noul', 'noul': 2}}}, payload)


@pytest.mark.asyncio
async def test_openrouter_discovery_adds_decisions_without_losing_chat_on_failure():
    fail = False
    def handler(request):
        if request.url.params.get('output_modalities'):
            if fail:
                return httpx.Response(503)
            return httpx.Response(200, json={'data': [{'id': 'typesafe/jev-1.13',
                'architecture': {'output_modalities': ['decisions']},
                'pricing': {'prompt': '0.000000042', 'completion': '0'}}]})
        return httpx.Response(200, json={'data': [{'id': 'vendor/chat',
            'pricing': {'prompt': '0.000001', 'completion': '0.000002'}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        source = OpenRouterSource({}, client=client, route_stats=lambda: {})
        await source.pricing()
        offers = source.offers_sync('openrouter_market')
        assert {o['model_family']: o['protocol'] for o in offers} == {'chat': 'chat', 'jev-1.13': 'decisions'}
        fail = True
        await source.pricing()
        assert [o['model_family'] for o in source.live_offers()] == ['chat']


@pytest.mark.asyncio
async def test_antseed_discovery_preserves_peer_controls_and_fails_closed(host_store_clean, monkeypatch):
    import host_store
    import settings
    from conftest import seed_buyer_status, seed_peer_offers
    seed_buyer_status('antseed', deposits_available=5)
    seed_peer_offers([{'peerId': 'peer', 'maxConcurrency': 2, 'providerPricing': {
        'typesafe': {'services': {'typesafe/jev-1.13': {'inputUsdPerMillion': .01, 'outputUsdPerMillion': 0}}}}}])
    model = {'id': 'jev-1.13', 'type': 'decision', 'supported_protocols': ['typesafe-systemone'],
        'context_length': 32000, 'peers': [{'peerId': 'peer', 'serviceId': 'typesafe/jev-1.13',
            'type': 'decision', 'protocol': 'typesafe-systemone', 'inputUsdPerMillion': .01, 'outputUsdPerMillion': 0}]}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={'data': [model]}))) as client:
        source = AntSeedSource({'providers': {'antseed': {'discovery': 'marketplace',
            'discovery_id': 'antseed', 'base_url': 'http://buyer.test/v1'}}}, client=client)
        await source.pricing()
        offer, = source.offers_sync('antseed')
        assert offer['protocol'] == 'decisions' and offer['model_family'] == 'jev-1.13'
        assert offer['wire_model_id'] == 'typesafe/jev-1.13' and offer['max_concurrency'] == 2
        assert offer['capabilities'] == {'context': 32000}
        monkeypatch.setitem(settings._overrides, 'antseed.peer_denylist', ['peer'])
        assert source.offers_sync('antseed') == []
        monkeypatch.setitem(settings._overrides, 'antseed.peer_denylist', [])
        stamp, rows = source._decision_rows['antseed']
        source._decision_rows['antseed'] = (stamp - STALE_AFTER_S - 1, rows)
        assert source.offers_sync('antseed') == []


def test_persisted_protocol_metadata_quarantines_unknown_decision_models(host_store_clean):
    from conftest import seed_peer_offers
    import host_store
    seed_peer_offers([{'peerId': 'peer', 'providerPricing': {'future': {'services': {
        'future-decision': {'inputUsdPerMillion': .01, 'outputUsdPerMillion': 0}}}}}])
    with host_store._get_pool().connection() as conn:
        conn.execute("UPDATE peer_offers SET protocols=ARRAY['typesafe-systemone']")
    source = AntSeedSource({'providers': {'antseed': {'discovery': 'marketplace', 'discovery_id': 'antseed'}}})
    assert source.offers_sync('antseed') == []


@pytest.mark.asyncio
async def test_ingress_auth_restrictions_and_metering_use_existing_key_path(routed, monkeypatch):
    import auth_proxy
    import control_plane_client
    import host_store
    host, _, calls, _ = routed
    monkeypatch.setattr(control_plane_client, 'CONTROL_PLANE_URL', '')
    monkeypatch.setattr(auth_proxy, 'CALLER_KEYS', {'decision-test-key': 'decision-test'})
    monkeypatch.setattr(auth_proxy, 'CALLER_KEY_HASHES', {})
    monkeypatch.setattr(auth_proxy, 'UPSTREAM', 'http://router.test')
    auth_proxy._reset_stats_for_tests()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(host))) as upstream:
        monkeypatch.setattr(auth_proxy, '_client', upstream)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=auth_proxy.app), base_url='http://ingress.test') as client:
            denied = await client.post('/v1/decisions', json=PAYLOAD)
            assert denied.status_code == 401 and not calls
            headers = {'Authorization': 'Bearer decision-test-key'}
            ok = await client.post('/v1/decisions', headers=headers, json={**PAYLOAD, 'policy_ir': POLICY})
            assert ok.status_code == 200, ok.text
            spend, valid = host_store.consumer_spend_usd('decision-test')
            assert valid and spend == round(.0000042, 6)  # budget read's existing display precision
            rows = host_store.recent_calls(10)
            assert rows[0]['tokens_in'] == 100 and rows[0]['tokens_out'] == 0
            assert rows[0]['cost_usd'] == pytest.approx(.0000042)
            host_store.set_consumer_keys({'decision-test': {'status': 'active', 'allowed_routes': ['profile:default']}})
            denied = await client.post('/v1/decisions', headers=headers, json=PAYLOAD)
            assert denied.status_code == 403
            host_store.set_consumer_keys({'decision-test': {'status': 'inactive'}})
            denied = await client.post('/v1/decisions', headers=headers, json=PAYLOAD)
            assert denied.status_code == 403
            assert len(calls) == 2
