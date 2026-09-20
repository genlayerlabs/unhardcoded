"""Pure adapter: prepare a finite flow and render its result as a conversation.

All inference, branching, data selection and replacement run in Sigma flow.
The prompts and composition are the editable selective-compaction JSON preset.
"""
from copy import deepcopy
from pathlib import Path
import json
import math

from decision_protocol import validate_payload
from flow_data import bounded, encode

PRESET = Path(__file__).resolve().parents[1] / 'examples/flows/selective-compaction.json'


def size(value):
    return len(encode(value).encode())


def units(messages):
    i = 0
    while i < len(messages):
        end = i + 1
        if messages[i].get('tool_calls'):
            while end < len(messages) and messages[end].get('role') == 'tool':
                end += 1
        yield {'id': f'fragment_{i}', 'start': i, 'end': end, 'messages': messages[i:end]}
        i = end


def prepare(messages, *, keep_recent, pinned_indices, target_ratio,
            decision_policy, summary_policy, max_tokens):
    fragments = list(units(messages))
    pins = set(pinned_indices if pinned_indices is not None else
               (i for i, m in enumerate(messages) if m.get('role') == 'user'))
    pins.update(i for i, m in enumerate(messages) if m.get('role') in ('system', 'developer'))
    instructions = [messages[i] for i in sorted(pins)]
    pins.update(range(max(0, len(messages) - keep_recent), len(messages)))
    candidates = [f for f in fragments if not any(i in pins for i in range(f['start'], f['end']))]
    recent = [encode(f['messages']) for f in fragments if f not in candidates
              and any(i >= len(messages) - keep_recent for i in range(f['start'], f['end']))]
    recent = [s if len(s) <= 1000 else s[:500] + '[excerpt]' + s[-500:] for s in recent]
    target = math.floor(size(messages) * target_ratio)
    retained_size = size([m for f in fragments if f not in candidates for m in f['messages']])
    budget = min(1048576, max(128, (target - retained_size) // max(1, len(candidates)) - 100))
    template = json.loads(PRESET.read_text())[1]
    question = template['classify']['questions']['example']
    batches, batch, reasons = [], [], []

    def materialize(items):
        state = {'instructions': instructions, 'recent_evidence': recent, 'fragments': []}
        questions, records = {}, {}
        for f in items:
            raw = encode(f['messages'])
            q = deepcopy(question)
            view = json.dumps(f['messages'])  # bound ASCII expansion before admission
            if len(view) > 2000:
                view = view[:1000] + '[excerpt; full item goes only to summary]' + view[-1000:]
                q['criteria'].pop('archive')  # unseen evidence cannot authorize deletion
            state['fragments'].append({'id': f['id'], 'evidence': view})
            questions[f['id']] = q
            records[f['id']] = raw
        validate_payload({'state': state, 'questions': questions})
        if size(records) > 50000:
            raise ValueError('summary input limit')
        return {'state': state, 'items': records}, questions

    def flush():
        nonlocal batch
        if batch:
            batches.append(batch)
            batch = []

    for f in candidates[:128]:
        try:
            if len(batch) == 8:
                flush()
            materialize(batch + [f])
        except ValueError:
            flush()
            try:
                materialize([f])
            except ValueError:
                reasons.append('fragment_or_decision_context_limit')
                continue
        batch.append(f)
    flush()
    if len(candidates) > 128:
        reasons.append('fragment_limit')
    if len(batches) > 32:
        reasons.append('batch_limit')
    nodes, inputs, included = {'input': {'kind': 'input'}}, {}, []
    patches = []
    for index, batch in enumerate(batches[:32]):
        key = f'batch_{index}'
        data, questions = materialize(batch)
        # Leave headroom under the generic flow's 1 MiB data bound.
        if size({**inputs, key: data}) > 900000:
            reasons.append('flow_input_limit')
            break
        inputs[key] = data
        included.extend(f['id'] for f in batch)
        local = deepcopy(template)
        local['classify']['questions'] = questions
        local['classify']['policy'] = decision_policy
        local['generate']['policy'] = summary_policy
        local['generate']['max_tokens'] = min(max_tokens, 4096)
        local['generate']['system'] = local['generate']['system'].replace('$BUDGET', str(budget))
        local['patch']['max_string_bytes'] = budget
        for name, node in local.items():
            if name in ('input', 'output'):
                continue
            node['inputs'] = ['input' if pre == 'input' else f'{key}_{pre}' for pre in node['inputs']]
            if node['kind'] == 'data' and node['operation'] == 'project':
                node['path'].insert(0, key)
            nodes[f'{key}_{name}'] = node
        patches.append(f'{key}_patch')
    if patches:
        nodes['merged'] = {'kind': 'data', 'operation': 'union', 'inputs': patches}
    nodes['output'] = {'kind': 'output', 'inputs': ['merged' if patches else 'input']}
    bounded(inputs)
    return {'flow_ir': ['flow', nodes], 'flow_input': inputs, 'included': included,
            'fragments': fragments, 'messages': messages, 'target_ratio': target_ratio,
            'target_bytes': target, 'reasons': reasons}


def finish(prepared, result):
    original = prepared['messages']
    values = (result.get('response') or {}).get('data') if result.get('ok') else None
    included = set(prepared['included']) if isinstance(values, dict) else set()
    output, manifest = [], []
    for f in prepared['fragments']:
        action, messages = 'keep', f['messages']
        if f['id'] in included:
            if f['id'] not in values:
                action, messages = 'archive', []
            elif values[f['id']] != encode(messages):
                text = values[f['id']]
                summary = {'role': 'assistant', 'content': f"[Summary of {f['id']}; original retained by caller]\n{text}"}
                if isinstance(text, str) and text.strip() and size(summary) < size(messages):
                    action, messages = 'summarize', [summary]
        output.extend(messages)
        manifest.append({k: f[k] for k in ('id', 'start', 'end')} | {'action': action})
    if size(output) >= size(original):
        output = original
        for entry in manifest:
            entry['action'] = 'keep'
    return {'messages': output, 'compacted': output != original,
            'compaction': {'original_bytes': size(original), 'output_bytes': size(output),
                'target_bytes': prepared['target_bytes'], 'target_ratio': prepared['target_ratio'],
                'target_met': size(output) <= prepared['target_bytes'], 'fragments': manifest,
                'reasons': prepared['reasons'],
                'flow_fingerprint': (result.get('trace') or {}).get('flow_fingerprint')}}
