"""Guided intent goes through the real Lua engine; no upstream network calls."""
import asyncio
from pathlib import Path

import pytest

from llm_router_host import LLMRouterHost
from saas_routes import choices, compile_intent, preview

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tenant_host():
    base = LLMRouterHost(ROOT / 'core/router.lua', ROOT / 'tests/fixtures/saas.lua',
                         now_ms=lambda: 1000, enforce_provider_auth=False)
    base.init()
    host = base.for_tenant(7, {'OPENAI_API_KEY': 'test', 'ANTHROPIC_API_KEY': 'test'})
    for provider, model, price in [('openai', 'primary', 8), ('anthropic', 'backup', 2), ('openai', 'basic', 1)]:
        host.update_metrics(provider, model, {'ok': True, 'latency_ms': price * 100,
                                          'price_in': price, 'price_out': price * 2})
    return host


def intent(**kwargs):
    return {'targets': ['openai|primary', 'anthropic|backup'], 'goal': 'reliability',
            'workload': 'chat', 'timeout_seconds': 8, **kwargs}


def test_real_engine_preview_and_identity(tenant_host):
    out = preview(tenant_host, intent())
    assert [r['family'] for r in out['ranked']] == ['primary', 'backup']
    assert len(out['policy_id']) == 64
    assert preview(tenant_host, intent())['policy_id'] == out['policy_id']
    assert all(r['provider'] != 'platform_only' for r in choices(tenant_host))


def test_quality_ranks_all_eligible_models_and_keeps_unbenchmarked_models(tenant_host):
    out = preview(tenant_host, intent(targets='all', goal='quality'))
    assert [r['family'] for r in out['ranked']] == ['backup', 'primary', 'basic']
    assert out['excluded'] == []
    assert any('benchmark' in warning for warning in out['warnings'])

    out = preview(tenant_host, intent(targets=['openai|basic', 'openai|primary'], goal='quality'))
    assert [r['family'] for r in out['ranked']] == ['primary', 'basic']


def test_all_models_preserves_workload_and_price_gates_for_preferences(tenant_host):
    out = preview(tenant_host, intent(targets='all', workload='agent', max_price_in=3,
                                     allowed_preferences=['cost', 'speed', 'reliability', 'quality']))
    assert [r['family'] for r in out['ranked']] == ['backup']
    assert [p['eligible'] for p in out['task_previews']] == [1, 1, 1, 1]
    assert [p['preference'] for p in out['task_previews']] == ['cost', 'speed', 'reliability', 'quality']


@pytest.mark.parametrize('failure, expected', [('server_error', ['backup', 'primary']),
                                               ('refusal', ['backup'])])
def test_unrestricted_quality_only_falls_back_on_retryable_errors(tenant_host, failure, expected):
    calls = []

    async def provider(request):
        calls.append(request['model_family'])
        if request['model_family'] == 'backup':
            return {'ok': False, 'error': failure, 'error_kind': failure}
        return {'ok': True, 'latency_ms': 10, 'response': {'text': 'fallback', 'tokens_in': 2, 'tokens_out': 2}}

    tenant_host.set_async_call_hook(provider)
    out = preview(tenant_host, intent(targets='all', goal='quality'))
    result = asyncio.run(tenant_host.execute_async({'policy_ir': out['policy_ir'],
                         'messages': [{'role': 'user', 'content': 'hello'}]}))
    assert calls == expected
    assert result['ok'] == (failure == 'server_error')


def test_cost_and_hard_price_limits(tenant_host):
    assert preview(tenant_host, intent(goal='cost'))['ranked'][0]['family'] == 'backup'
    out = preview(tenant_host, intent(max_price_in=3))
    assert [r['family'] for r in out['ranked']] == ['backup']
    assert out['excluded'] and out['warnings']
    assert preview(tenant_host, intent(max_price_out=.001))['ranked'] == []


def test_capability_requirements(tenant_host):
    out = preview(tenant_host, intent(targets=['openai|basic'], workload='agent'))
    assert out['ranked'] == []
    assert out['excluded']


def test_every_authorized_preference_preserves_hard_limits(tenant_host):
    from route_contract import apply_contract
    built = preview(tenant_host, intent(max_price_in=3, workload='agent',
                                        allowed_preferences=['cost', 'speed', 'reliability']))
    published = {**built, 'preferences': built['execution']['task_policies']}
    for preference in ('cost', 'speed', 'reliability'):
        payload, selected = apply_contract({'routing_preference': preference, 'policy_ir': ['forged'],
            'flow_ir': ['forged'], 'timeout_ms': 999999}, published)
        ranked, _ = tenant_host.rank(payload)
        assert [r['candidate']['model_family'] for r in ranked] == ['backup']
        assert payload['timeout_ms'] == 8000 and 'flow_ir' not in payload
        assert payload['policy_ir'] == built['execution']['task_policies'][preference]['policy_ir']
        assert selected['policy_id'] == built['execution']['task_policies'][preference]['policy_id']


def test_preferences_cannot_authorize_a_call_when_nothing_qualifies(tenant_host):
    from route_contract import apply_contract
    built = preview(tenant_host, intent(max_price_out=.001, allowed_preferences=['cost', 'speed']))
    called = []
    async def provider(request):
        called.append(request)
        raise AssertionError('No provider should be called')
    tenant_host.set_async_call_hook(provider)
    for preference in ('cost', 'speed'):
        payload, _ = apply_contract({'routing_preference': preference, 'messages': []},
                                    {**built, 'preferences': built['execution']['task_policies']})
        result = asyncio.run(tenant_host.execute_async(payload))
        assert not result['ok']
    assert called == []


@pytest.mark.parametrize('preference', ['ignore_limits', 'speed', [], {}, 1])
def test_unpublished_preferences_are_rejected(preference):
    from route_contract import apply_contract, PreferenceNotAllowed
    with pytest.raises(PreferenceNotAllowed):
        apply_contract({'routing_preference': preference}, {'policy_ir': ['policy'], 'preferences': {}})


def test_real_failover_stays_within_approved_set(tenant_host):
    calls = []
    async def provider(request):
        calls.append(request['provider_id'])
        if request['provider_id'] == 'openai':
            return {'ok': False, 'error': 'unavailable', 'error_kind': 'server_error'}
        return {'ok': True, 'latency_ms': 10, 'response': {'text': 'backup response', 'tokens_in': 2, 'tokens_out': 2}}
    tenant_host.set_async_call_hook(provider)
    result = asyncio.run(tenant_host.execute_async({'policy_ir': compile_intent(intent()),
                         'messages': [{'role':'user','content':'hello'}]}))
    assert result['ok'], result
    assert calls == ['openai', 'anthropic']


@pytest.mark.parametrize('change', [{'targets': []}, {'targets': None}, {'targets': ['all']},
                                    {'targets': ['invalid/provider|forbidden']},
                                    {'timeout_seconds': 0}, {'max_price_in': float('nan')},
                                    {'targets': ['openai|primary'] * 2}])
def test_invalid_intents_refused(change):
    with pytest.raises(ValueError):
        compile_intent(intent(**change))


def test_activity_summary_never_includes_prompt_output_or_upstream_error():
    from host_store import routing_summary
    result = routing_summary({'route_revision':3, 'messages':['secret prompt'],
        'response':'secret output', 'decision_path':[{'event':'attempted',
        'provider_id':'openai', 'error_kind':'auth_error', 'error_message':'secret key'}]})
    assert result['attempts'] == [{'provider_id':'openai','error_kind':'auth_error'}]
    assert 'secret' not in str(result)


def test_live_catalog_accepts_guided_policy_without_platform_credentials():
    base = LLMRouterHost(ROOT/'core/router.lua', ROOT/'config.live.lua', ROOT/'metrics.live.lua')
    base.init()
    child = base.for_tenant(7, {'OPENAI_API_KEY':'sk-test', 'ANTHROPIC_API_KEY':'sk-test'})
    models = choices(child)
    assert models
    assert {m['provider'] for m in models} <= {'openai','anthropic'}
    result = preview(child, intent(targets=[models[0]['id']]))
    assert result['ranked'][0]['id'] == models[0]['id']
