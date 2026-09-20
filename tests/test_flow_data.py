"""Generic typed flows: ticket triage, without any conversation-compaction code."""
import asyncio
import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flow_data import run, bounded
from flow_runner import run_flow
from llm_router_host import LLMRouterHost, _to_lua, _to_py
from shim import create_app

ROOT = Path(__file__).resolve().parents[1]


def policy(family):
    return ['policy', ['and', ['meets_req'], ['family_eq', family]], ['zero'], ['argmax'],
            ['id'], ['always', {'action': 'next_candidate'}]]


def triage():
    return ['flow', {
        'input': {'kind': 'input'},
        'decide': {'kind': 'decision', 'inputs': ['input'], 'policy': policy('decision'),
            'on_error': 'input', 'timeout_ms': 1000, 'questions': {
                key: {'type': 'choice', 'instructions': 'Which department handles this ticket?',
                      'criteria': {'support': 'Technical support', 'sales': 'Sales'}} for key in ('a', 'b')}},
        'select': {'kind': 'data', 'operation': 'select', 'inputs': ['input', 'decide'],
                   'field': 'choice', 'equals': 'support'},
        'reply': {'kind': 'llm', 'inputs': ['select'], 'policy': policy('generator'),
                  'system': 'Draft replies as a JSON map keyed by ticket ID.', 'context': 'inputs',
                  'output_format': 'json', 'skip_empty': True, 'on_error': 'input'},
        'patch': {'kind': 'data', 'operation': 'overlay', 'inputs': ['input', 'reply'], 'on_error': 'input'},
        'output': {'kind': 'output', 'inputs': ['patch']},
    }]


@pytest.fixture
def service(tmp_path, host_store_clean):
    cfg = tmp_path/'config.lua'
    cfg.write_text('''return {providers={market={discovery="marketplace",discovery_id="market",
      api_kind="openai_compatible"}}, models={},profiles={default={scorer={"zero"}}},
      policy_envelope={"and",{"meets_req"},{"not",{"is","disabled"}}}}''')
    host = LLMRouterHost(router_path=ROOT/'core/router.lua', config_path=cfg)
    host.set_discover_hook(lambda _: {'ok': True, 'offers': [
        {'model_family': family, 'wire_model_id': family, 'seller_endpoint': 'https://provider.invalid/v1',
         'protocol': 'decisions' if family == 'decision' else 'chat',
         'price_in_usd_per_mtok': .1, 'price_out_usd_per_mtok': .1,
         'capabilities': {'supports_json_mode': True}}
        for family in ('decision', 'generator')]})
    host.init()
    seen, behavior = [], {'choices': {'a': 'support', 'b': 'sales'}}
    async def provider(req):
        seen.append(copy.deepcopy(req))
        if req.get('protocol') == 'decisions':
            if behavior.get('timeout'):
                await asyncio.sleep(2)
            answers = {key: {'type': 'choice', 'choice': behavior['choices'][key],
                        'probabilities': {label: float(label == behavior['choices'][key]) for label in q['criteria']}}
                       for key, q in req['decision']['questions'].items()}
            response = {'decision': {'model': 'fixture', 'answers': answers}, 'tokens_in': 10,
                        'tokens_out': 0, 'cost_reported': .000001}
        else:
            if behavior.get('raise'):
                raise RuntimeError('provider unavailable')
            items = json.loads(req['messages'][-1]['content'])
            response = {'text': behavior.get('text', json.dumps({k: 'Reply for ' + k for k in items})),
                        'finish_reason': behavior.get('finish_reason', 'stop'),
                        'tokens_in': 20, 'tokens_out': 5, 'cost_reported': .002}
        return {'ok': True, 'latency_ms': 1, 'response': response}
    host.set_async_call_hook(provider)
    return TestClient(create_app(host)), host, seen, behavior


def post(service, flow=None, data=None):
    client, _, _, _ = service
    return client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'UNRELATED HISTORY'}],
        'flow_ir': flow or triage(), 'flow_input': data or {'a': 'Crash report', 'b': 'Price inquiry'}})


def test_generic_flow_routes_classifies_selects_and_generates_only_selected_data(service):
    response = post(service)
    assert response.status_code == 200, response.text
    body = response.json()
    assert json.loads(body['choices'][0]['message']['content']) == {'a': 'Reply for a', 'b': 'Price inquiry'}, [(n.get('kind'), n.get('error'), n.get('fallback')) for n in body['x_router']['decision_trace']['flow_nodes']]
    _, _, calls, _ = service
    assert [c['protocol'] for c in calls] == ['decisions', 'chat']
    assert calls[0]['decision']['state'] == {'a': 'Crash report', 'b': 'Price inquiry'}
    assert 'UNRELATED HISTORY' not in str(calls[1]['messages'])
    assert 'Price inquiry' not in str(calls[1]['messages'])
    assert body['x_router']['cost_usd'] == pytest.approx(.002001)
    assert body['usage']['prompt_tokens'] == 30


def test_empty_selection_makes_zero_generative_calls(service):
    service[3]['choices'] = {'a': 'sales', 'b': 'sales'}
    response = post(service)
    assert response.status_code == 200, response.text
    assert len(service[2]) == 1
    body = response.json()
    assert json.loads(body['choices'][0]['message']['content']) == {'a': 'Crash report', 'b': 'Price inquiry'}
    assert any(n.get('skipped') for n in body['x_router']['decision_trace']['flow_nodes'])
    assert body['x_router']['cost_usd'] == pytest.approx(.000001)


@pytest.mark.parametrize('override', [{'text': 'broken JSON'}, {'text': '{"invented":"x"}'},
                                    {'text': '{"a":"first","a":"second"}'},
                                    {'text': '{"a":"partial"}', 'finish_reason': 'length'}, {'raise': True}])
def test_invalid_or_failed_generation_preserves_records_and_cost(service, override):
    service[3].update(override)
    response = post(service)
    assert response.status_code == 200, response.text
    body = response.json()
    assert json.loads(body['choices'][0]['message']['content']) == {'a': 'Crash report', 'b': 'Price inquiry'}
    assert body['x_router']['cost_usd'] == (None if override.get('raise') else pytest.approx(.002001))


def test_decision_timeout_abstains_without_generation_and_cost_stays_unknown(service):
    service[3]['timeout'] = True
    response = post(service)
    assert response.status_code == 200, response.text
    assert len(service[2]) == 1
    assert response.json()['x_router']['cost_usd'] is None


@pytest.mark.parametrize('change', ['policy', 'operation', 'questions', 'path', 'data_depth'])
def test_bad_flow_is_rejected_before_any_provider_call(service, change):
    flow, data = triage(), None
    if change == 'policy':
        flow[1]['reply']['policy'] = ['bad']
    elif change == 'operation':
        flow[1]['select']['operation'] = 'eval'
    elif change == 'questions':
        flow[1]['decide']['questions']['a']['criteria'] = {}
    elif change == 'path':
        flow[1]['select']['path'] = ['unexpected']
    else:
        data = {'nested': {}}
        for _ in range(40):
            data = {'nested': data}
    response = post(service, flow, data)
    assert response.status_code == 400, response.text
    assert response.json()['error']['code'] == 'invalid_flow'
    assert not service[2]


def test_normalization_preserves_typed_flow_semantics_and_identity(service):
    _, host, _, _ = service
    original = triage()
    renamed = ['flow', {'renamed_' + k: {**n, **({'inputs': ['renamed_' + p for p in n['inputs']]} if 'inputs' in n else {})}
                        for k, n in original[1].items()}]
    a, b = host.flow_admit(original), host.flow_admit(renamed)
    assert a['encoded'] == b['encoded']
    assert host.flow_admit(a['flow_ir'])['encoded'] == a['encoded']
    renamed[1]['renamed_select']['equals'] = 'sales'
    assert host.flow_admit(renamed)['encoded'] != a['encoded']


def test_python_data_operations_match_core_reference(service):
    _, host, _, _ = service
    reference = host.lua.eval('(require("llm_policy.flow_data"))')
    cases = [
        ({'operation': 'project', 'path': ['items']}, [{'items': {'a': 'hello'}}]),
        ({'operation': 'select', 'field': 'choice', 'equals': 'yes'}, [{'a': 'hello', 'b': 'world'}, {'a': {'choice': 'yes'}}]),
        ({'operation': 'overlay'}, [{'a': True}, {'a': False}]),
        ({'operation': 'overlay', 'max_string_bytes': 3, 'only_shrink': True}, [{'a': 'hello', 'b': 'world'}, {'a': 'é', 'b': '😀'}]),
        ({'operation': 'union'}, [{'a': 'hello'}, {'b': 'world'}]),
    ]
    for node, parts in cases:
        assert run(node, parts) == _to_py(reference.run(_to_lua(host.lua, node), _to_lua(host.lua, parts)))


def test_generic_shared_deadline_cancels_inference_and_keeps_fallback():
    called, cancelled = [], []
    async def slow(nid, node, prompt):
        called.append(nid)
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(nid)
    result = asyncio.run(run_flow(triage(), '', slow, input_data={'a': 'original'}, timeout_seconds=.01))
    assert result['ok'] and result['data'] == {'a': 'original'}
    assert called == cancelled and len(called) == 1
    assert any(n.get('fallback') for n in result['trace'])


def test_record_and_depth_limits_and_overlay_unknown_keys():
    with pytest.raises(ValueError):
        run({'operation': 'union'}, [{str(i): i for i in range(129)}])
    with pytest.raises(ValueError):
        run({'operation': 'overlay'}, [{'a': 'x'}, {'unknown': 'y'}])
    assert run({'operation': 'overlay'}, [{'a': 'x'}, {'a': None}]) == {'a': None}
    with pytest.raises(ValueError):
        bounded({'a': 'x' * 1048576})


def test_documented_ticket_preset_executes_in_generic_router(service):
    flow = json.loads((ROOT/'examples/flows/ticket-triage.json').read_text())
    response = post(service, flow=flow)
    assert response.status_code == 200, response.text
    assert json.loads(response.json()['choices'][0]['message']['content']) == {
        'a': 'Reply for a', 'b': 'Price inquiry'}
    assert [c['protocol'] for c in service[2]] == ['decisions', 'chat']
