"""Real router admission/execution; only provider calls are mocked."""
import asyncio
import copy

import pytest
from fastapi.testclient import TestClient

from llm_router_host import LLMRouterHost
from shim import create_app
from flow_routing import decision_state
from test_flow import ROOT


def policy(family):
    return ['policy', ['and', ['meets_req'], ['family_eq', family]], ['zero'], ['argmax'],
            ['id'], ['always', {'action': 'next_candidate'}]]


def adaptive_flow():
    return ['flow', {
        'in': {'kind': 'input'},
        'generate': {'kind': 'llm', 'inputs': ['in'], 'system': 'Answer the task.',
            'policy': policy('astra'), 'routing': {
                'policy': policy('decision'), 'instructions': 'Choose economy for routine work; capable for novel reasoning or failed attempts.',
                'fallback': 'capable', 'min_confidence': .7, 'timeout_ms': 1000,
                'choices': {'economy': {'description': 'Routine generation', 'policy': policy('luna')},
                            'capable': {'description': 'Deep reasoning and repair', 'policy': policy('astra')}}}},
        'out': {'kind': 'output', 'inputs': ['generate']},
    }]


@pytest.fixture
def routing(tmp_path, host_store_clean):
    config = tmp_path / 'config.lua'
    config.write_text('''return {providers={market={discovery="marketplace",discovery_id="market",
      api_kind="openai_compatible"}}, models={},profiles={default={scorer={"zero"}}},
      policy_envelope={"and",{"meets_req"},{"not",{"is","disabled"}}}}''')
    host = LLMRouterHost(router_path=ROOT / 'core/router.lua', config_path=config)
    offers = [{'model_family': family, 'wire_model_id': family,
               'seller_endpoint': 'https://provider.invalid/v1',
               'protocol': 'decisions' if family == 'decision' else 'chat',
               'price_in_usd_per_mtok': .1, 'price_out_usd_per_mtok': .1,
               'capabilities': ['tools']} for family in ('decision', 'luna', 'astra')]
    host.set_discover_hook(lambda _: {'ok': True, 'offers': offers})
    host.init()
    seen, behavior = [], {'choice': 'economy', 'confidence': .95}

    async def provider(req):
        seen.append(copy.deepcopy(req))
        if req.get('protocol') == 'decisions':
            if behavior.get('timeout'):
                await asyncio.sleep(2)
            if behavior.get('failure'):
                return {'ok': False, 'error_kind': 'server_error'}
            decision = {'model': 'decision', 'answers': {'route': {
                'type': 'choice', 'choice': behavior['choice'],
                'confidence': behavior['confidence'],
                'probabilities': {k: (.95 if k == behavior['choice'] else .05) for k in ('economy', 'capable')}}}}
            return {'ok': True, 'latency_ms': 1, 'response': {
                'decision': decision, 'tokens_in': 100, 'tokens_out': 0, 'cost_reported': .0000042}}
        return {'ok': True, 'latency_ms': 1, 'response': {
            'text': req['served_model_id'], 'tokens_in': 20, 'tokens_out': 10,
            'cost_reported': .001, 'tool_calls': behavior.get('tool_calls')}}

    host.set_async_call_hook(provider)
    return TestClient(create_app(host)), seen, behavior


def post(client, flow=None):
    return client.post('/v1/chat/completions', json={
        'flow_ir': flow or adaptive_flow(), 'messages': [
            {'role': 'user', 'content': 'Fix the authentication bug.'},
            {'role': 'assistant', 'content': 'Previous attempt failed.'},
            {'role': 'user', 'content': 'AssertionError: session expired; inspect auth.py'}]})


@pytest.mark.parametrize('choice,expected', [('economy', 'luna'), ('capable', 'astra')])
def test_only_selected_generation_runs_with_history_and_total_cost(routing, choice, expected):
    client, seen, behavior = routing
    behavior['choice'] = choice
    response = post(client)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['choices'][0]['message']['content'] == expected
    assert [r['served_model_id'] for r in seen] == ['decision', expected]
    assert 'Previous attempt failed' in str(seen[0]['decision']['state'])
    assert 'AssertionError' in str(seen[1]['messages'])
    node = data['x_router']['decision_trace']['flow_nodes'][0]
    assert node['routing']['selected'] == choice
    assert not node['routing']['fallback_used']
    assert node['routing']['cost_usd'] == .0000042
    assert data['usage']['prompt_tokens'] == 120
    assert data['x_router']['cost_usd'] == pytest.approx(.0010042, abs=.0000005)


@pytest.mark.parametrize('override,reason', [
    ({'confidence': .1}, 'uncertain'), ({'choice': 'invented'}, 'invalid_decision'),
    ({'failure': True}, 'decision_failed'), ({'timeout': True}, 'decision_timeout'),
])
def test_declared_fallback_is_the_only_generation_on_bad_decision(routing, override, reason):
    client, seen, behavior = routing
    behavior.update(override)
    response = post(client)
    assert response.status_code == 200, response.text
    assert response.json()['choices'][0]['message']['content'] == 'astra'
    assert [r['served_model_id'] for r in seen if r.get('protocol') != 'decisions'] == ['astra']
    trace = response.json()['x_router']['decision_trace']['flow_nodes'][0]['routing']
    assert trace['fallback_reason'] == reason
    assert trace['selected'] == 'capable'


def test_invalid_branch_rejected_before_any_provider_call(routing):
    client, seen, _ = routing
    flow = adaptive_flow()
    flow[1]['generate']['routing']['choices']['economy']['policy'] = ['bad']
    assert post(client, flow).status_code == 400
    assert not seen


def test_terminal_tool_calls_survive_decision_routing(routing):
    client, _, behavior = routing
    behavior['tool_calls'] = [{'id': 'c1', 'type': 'function', 'function': {'name': 'shell', 'arguments': '{"command":"pwd"}'}}]
    response = post(client)
    assert response.status_code == 200, response.text
    assert response.json()['choices'][0]['message']['tool_calls'] == behavior['tool_calls']


def test_projection_is_bounded_and_reports_omissions():
    state = decision_state([{'role': 'tool', 'content': 'x' * 10000}] * 100, 'q' * 5000)
    assert state['omitted_messages'] > 0 and state['request_truncated']
    assert sum(len(x['message']) for x in state['history']) <= 12000
    assert all(x['truncated'] for x in state['history'])
