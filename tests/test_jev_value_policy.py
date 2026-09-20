"""Qualify the experimental Jev-only policy; upstream transports are simulated.

These tests verify selection and failover semantics with simulated transports;
they never claim live Jev/provider availability.
"""
import json
from pathlib import Path

import pytest

from llm_router_host import LLMRouterHost

ROOT = Path(__file__).resolve().parents[1]
POLICY = json.loads((ROOT / 'policies/jev-value-v1.json').read_text())


@pytest.fixture
def jev_host(tmp_path):
    names = ('cheap', 'fast', 'slow', 'unknown_latency', 'expensive', 'unpriced', 'unreliable', 'alien')
    config = tmp_path / 'jev.lua'
    providers = ','.join(f'{p}={{discovery="marketplace",discovery_id="{p}",api_kind="openai_compatible",tier="partner"}}' for p in names)
    config.write_text('return {providers={' + providers + '},models={},profiles={default={scorer={"zero"}}},policy_envelope={"and",{"meets_req"},{"not",{"is","disabled"}}}}')
    offers = {}
    for name, price, latency, success in (
        ('cheap', .01, 450, 1), ('fast', .042, 250, 1),
        ('slow', .005, 1800, 1), ('unknown_latency', .001, None, 1),
        ('expensive', .5, 50, 1), ('unpriced', None, 50, 1),
        ('unreliable', .001, 50, .5), ('alien', 0, 1, 1),
    ):
        offers[name] = {'model_family': 'other-model' if name == 'alien' else 'jev-1.13',
                        'wire_model_id': 'typesafe/jev-1.13', 'protocol': 'decisions',
                        'price_in_usd_per_mtok': price, 'price_out_usd_per_mtok': 0,
                        'latency_ms': latency, 'success_rate': success,
                        'seller_endpoint': f'https://{name}.invalid',
                        'capabilities': {'context': 32000}}
    host = LLMRouterHost(router_path=ROOT / 'core/router.lua', config_path=config, now_ms=lambda: 1000)
    host.set_discover_hook(lambda discovery_id: {'ok': True, 'offers': [offers[discovery_id]]})
    host.init()
    return host, offers


def ranked(host):
    rows, _ = host.rank({'protocol': 'decisions', 'policy_ir': POLICY})
    return rows


def test_admission_and_measured_fast_routes_before_slow_unknown(jev_host):
    host, _ = jev_host
    assert host.normalize_policy(POLICY, admit=True)['version'] == 'sigma-pol/v2'
    rows = ranked(host)
    assert [r['candidate']['provider_id'] for r in rows] == ['cheap', 'fast', 'unknown_latency', 'slow']
    assert all(r['candidate']['model_family'] == 'jev-1.13' for r in rows)


def test_weights_reward_speed_without_being_diluted_by_unrelated_models(jev_host):
    host, offers = jev_host
    offers['fast']['price_in_usd_per_mtok'] = .011
    assert ranked(host)[0]['candidate']['provider_id'] == 'fast'
    offers['alien']['price_in_usd_per_mtok'] = 1e6
    host.invalidate_discovery('alien')
    assert ranked(host)[0]['candidate']['provider_id'] == 'fast'


def test_unpriced_expensive_unreliable_and_non_jev_never_qualify(jev_host):
    host, _ = jev_host
    names = {r['candidate']['provider_id'] for r in ranked(host)}
    assert names.isdisjoint({'unpriced', 'expensive', 'unreliable', 'alien'})


def test_open_breaker_moves_behind_healthy_routes(jev_host):
    host, _ = jev_host
    state = host.dump_state()
    state['circuit_breakers'] = {'cheap': {'open': True, 'consecutive_failures': 5,
                                  'opened_at_ms': 1000, 'open_until_ms': 60000}}
    host.restore_state(state)
    assert ranked(host)[-1]['candidate']['provider_id'] == 'cheap'


def test_missing_jev_fails_closed(jev_host):
    host, offers = jev_host
    for offer in offers.values():
        offer['model_family'] = 'other-model'
    result = host.execute({'protocol': 'decisions', 'policy_ir': POLICY, 'messages': [{'role': 'user', 'content': 'Synthetic test'}]})
    assert not result['ok']
    assert result['error'] == 'no_candidates'


def test_every_fallback_remains_jev_and_exhaustion_does_not_widen(jev_host):
    host, _ = jev_host
    for provider in ('cheap', 'fast', 'unknown_latency', 'slow'):
        host.set_mock_response(provider, 'jev-1.13', {'ok': False, 'error_kind': 'server_error', 'latency_ms': 1})
    # A successful non-Jev response must never rescue a Jev-only request.
    host.set_mock_response('alien', 'other-model', {'ok': True, 'latency_ms': 1,
                           'response': {'text': 'wrong model', 'tokens_in': 1, 'tokens_out': 1}})
    result = host.execute({'protocol': 'decisions', 'policy_ir': POLICY, 'messages': [{'role': 'user', 'content': 'Synthetic test'}]})
    assert not result['ok']
    attempted = [e for e in result['trace']['decision_path'] if e['event'] == 'attempted']
    assert [e['provider_id'] for e in attempted] == ['cheap', 'fast', 'unknown_latency', 'slow']
    assert all(e['model_family'] == 'jev-1.13' for e in attempted)


def test_next_jev_succeeds_after_primary_failure(jev_host):
    host, _ = jev_host
    host.set_mock_response('cheap', 'jev-1.13', {'ok': False, 'error_kind': 'timeout', 'latency_ms': 1})
    host.set_mock_response('fast', 'jev-1.13', {'ok': True, 'latency_ms': 1,
        'response': {'text': '{"choice":"billing"}', 'finish_reason': 'stop',
                     'tokens_in': 10, 'tokens_out': 2, 'tokens_total': 12}})
    result = host.execute({'protocol': 'decisions', 'policy_ir': POLICY, 'messages': [{'role': 'user', 'content': 'Synthetic test'}]})
    assert result['ok']
    assert result['chosen']['provider_id'] == 'fast'
    attempted = [e['provider_id'] for e in result['trace']['decision_path'] if e['event'] == 'attempted']
    assert attempted == ['cheap', 'fast']
