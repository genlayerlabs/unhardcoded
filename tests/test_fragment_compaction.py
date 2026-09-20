"""Whole-unit compaction under failures and strict budgets, no paid inference."""
import asyncio
import json

import pytest

from fragment_compaction import compact_fragments, size


class Models:
    def __init__(self, choices=None):
        self.calls = []
        self.choices = choices or {}
        self.summary = None
        self.fail = None

    async def execute(self, contract):
        self.calls.append(contract)
        kind = 'decision' if contract.get('protocol') == 'decisions' else 'summary'
        if self.fail == kind:
            raise TimeoutError()
        if kind == 'decision':
            answers = {}
            for key, question in contract['decision']['questions'].items():
                choice = self.choices.get(key, 'summarize')
                answers[key] = {'type': 'choice', 'choice': choice,
                    'probabilities': {label: float(label == choice) for label in question['criteria']}}
            response = {'decision': {'model': 'fixture', 'answers': answers}}
        else:
            fragments = json.loads(contract['messages'][1]['content'])
            text = json.dumps({f['id']: 'Evidence ' + f['id'] for f in fragments})
            response = {'text': self.summary if self.summary is not None else text}
        return {'ok': True, 'response': response}


def run(messages, models, **kwargs):
    return asyncio.run(compact_fragments(messages, keep_recent=kwargs.pop('keep_recent', 1),
        pinned_indices=kwargs.pop('pinned_indices', None), target_ratio=kwargs.pop('target_ratio', .1),
        decision_policy=['decision-fixture'], summary_policy=['summary-fixture'], max_tokens=512,
        execute=models.execute, costed=lambda res: {'x_router': {'cost_usd': .001},
                                                  'usage': {'prompt_tokens': 10}}, **kwargs))


def transcript():
    return [{'role': 'system', 'content': 'Keep instructions.'},
            {'role': 'user', 'content': 'Complete the task.'}] + [
        {'role': 'assistant', 'content': f'fact_{i} ' + 'x' * 1800} for i in range(10)
    ] + [{'role': 'assistant', 'content': 'Current state.'}]


def test_batches_decisions_and_summaries_with_order_and_ten_percent_target():
    messages = transcript()
    models = Models()
    out = run(messages, models)
    assert out['compacted'] and out['compaction']['target_met']
    assert out['messages'][:2] == messages[:2] and out['messages'][-1] == messages[-1]
    assert len([c for c in models.calls if c.get('protocol') == 'decisions']) == 2
    assert len([c for c in models.calls if 'messages' in c]) == 1
    for i, message in enumerate(out['messages'][2:-1], 2):
        assert f'fragment_{i}' in message['content']
    assert out['x_router']['cost_usd'] == .003
    assert out['usage']['prompt_tokens'] == 30
    assert size(out['messages']) <= size(messages) * .1


def test_tool_pair_crossing_tail_boundary_is_pinned_whole():
    messages = transcript()[:-1] + [
        {'role': 'assistant', 'tool_calls': [{'id': 'call', 'function': {'name': 'shell', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call', 'content': 'evidence'}]
    out = run(messages, Models())
    assert out['messages'][-2:] == messages[-2:]


def test_tool_pair_selected_as_one_fragment_and_replaced_as_one():
    messages = transcript()
    pair = [{'role': 'assistant', 'tool_calls': [{'id': 'call'}]},
            {'role': 'tool', 'tool_call_id': 'call', 'content': 'evidence ' * 100}]
    messages[2:3] = pair
    models = Models()
    out = run(messages, models)
    entry = next(f for f in out['compaction']['fragments'] if f['start'] == 2)
    assert entry['end'] == 4 and entry['action'] == 'summarize'
    assert not any(m.get('tool_call_id') == 'call' for m in out['messages'])
    summary_call = next(c for c in models.calls if 'messages' in c)
    fragment = json.loads(summary_call['messages'][1]['content'])[0]
    assert fragment['messages'] == pair


@pytest.mark.parametrize('bad', ['not JSON', '{}', '{"wrong_id":"invented"}', '{"fragment_2":null}'])
def test_invalid_summary_preserves_originals_and_charges_all_legs(bad):
    messages = transcript()
    models = Models()
    models.summary = bad
    out = run(messages, models)
    assert out['messages'] == messages and not out['compacted']
    assert not out['compaction']['target_met']
    assert out['x_router']['cost_usd'] == .003


@pytest.mark.parametrize('kind', ['decision', 'summary'])
def test_failed_calls_preserve_evidence_and_unknown_cost(kind):
    messages = transcript()
    models = Models()
    models.fail = kind
    out = run(messages, models)
    assert out['messages'] == messages
    assert out['x_router']['cost_usd'] is None


def test_missing_or_invented_choice_cannot_remove_evidence():
    messages = transcript()
    models = Models({f'fragment_{i}': 'invented' for i in range(2, 12)})
    out = run(messages, models)
    assert out['messages'] == messages
    assert all(c.get('protocol') == 'decisions' for c in models.calls)


def test_explicit_pins_allow_observation_user_role_without_unpinning_system():
    messages = transcript()
    messages[2]['role'] = 'user'
    out = run(messages, Models(), pinned_indices=[1])
    assert out['messages'][:2] == messages[:2]
    assert out['compaction']['fragments'][2]['action'] == 'summarize'
    default = run(messages, Models(), target_ratio=.5)
    assert messages[2] in default['messages']


def test_target_cannot_override_kept_or_protected_evidence():
    messages = transcript()
    models = Models({f'fragment_{i}': 'keep' for i in range(2, 12)})
    out = run(messages, models)
    assert out['messages'] == messages
    assert not out['compaction']['target_met']
    assert all(c.get('protocol') == 'decisions' for c in models.calls)


def test_archive_is_explicit_and_does_not_generate():
    messages = transcript()
    models = Models({f'fragment_{i}': 'archive' for i in range(2, 12)})
    out = run(messages, models)
    assert out['messages'] == messages[:2] + messages[-1:]
    assert out['compaction']['target_met']
    assert all(c.get('protocol') == 'decisions' for c in models.calls)


def test_excerpt_cannot_authorize_deleting_unseen_evidence():
    messages = transcript()
    messages[2]['content'] = 'x' * 12000 + 'critical tail'
    models = Models({'fragment_2': 'archive'})
    out = run(messages, models)
    assert out['compaction']['fragments'][2]['action'] == 'summarize'
    call = next(c for c in models.calls if 'messages' in c)
    assert 'critical tail' in call['messages'][1]['content']


def test_oversized_pinned_context_abstains_without_inference():
    messages = transcript()
    messages[0]['content'] = '😀' * 4000
    models = Models()
    out = run(messages, models)
    assert not models.calls and out['messages'] == messages
    assert 'decision_context_limit' in out['compaction']['reasons']


def test_oversized_summary_is_rejected_without_truncation():
    messages = transcript()
    models = Models()
    models.summary = json.dumps({f'fragment_{i}': 'too big' * 500 for i in range(2, 12)})
    out = run(messages, models)
    assert out['messages'] == messages
    assert 'summary_exceeds_budget' in out['compaction']['reasons']


def test_adaptive_batches_respect_ascii_wire_limit():
    messages = transcript()
    messages[0]['content'] = 'Rules ' * 2600
    for message in messages[2:-1]:
        message['content'] = '😀' * 400
    models = Models({f'fragment_{i}': 'keep' for i in range(2, 12)})
    out = run(messages, models)
    assert out['messages'] == messages
    assert len(models.calls) > 2
    assert all(len(json.dumps(c['decision']).encode()) <= 32000 for c in models.calls)
    assert sum(len(c['decision']['questions']) for c in models.calls) == 10


def test_large_recent_observation_is_kept_but_only_excerpted_for_triage():
    messages = transcript()
    messages[-1]['content'] = 'Current evidence ' * 10000
    models = Models()
    out = run(messages, models)
    assert models.calls and out['compacted']
    assert out['messages'][-1] == messages[-1]
    assert not out['compaction']['target_met']
    assert all(len(json.dumps(c['decision']).encode()) <= 32000
               for c in models.calls if 'decision' in c)


def test_wall_deadline_cancels_slow_inference_and_prevents_further_calls(monkeypatch):
    import fragment_compaction
    monkeypatch.setattr(fragment_compaction, 'MAX_SECONDS', .01)
    models = Models()
    cancelled = []
    async def slow(contract):
        models.calls.append(contract)
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
    models.execute = slow
    messages = transcript()
    out = run(messages, models)
    assert cancelled == [True] and len(models.calls) == 1
    assert out['messages'] == messages and out['x_router']['cost_usd'] is None
    assert 'compaction_deadline' in out['compaction']['reasons']
