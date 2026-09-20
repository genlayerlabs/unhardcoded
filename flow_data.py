"""Bounded JSON operations for Sigma flows, mirrored by core flow_data.lua.

There is no application-specific operation, expression evaluator or user code.
"""
import json

MAX_BYTES = 1048576


def is_typed(flow):
    nodes = flow[1] if isinstance(flow, list) and len(flow) == 2 and isinstance(flow[1], dict) else {}
    keys = {'output_format', 'skip_empty', 'on_error', 'context', 'max_tokens', 'timeout_ms'}
    return any(isinstance(n, dict) and (n.get('kind') in ('data', 'decision') or keys.intersection(n))
               for n in nodes.values())


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def decode(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return bounded(json.loads(text, object_pairs_hook=unique))


def bounded(value):
    def depth(v, level=0):
        if level > 32:
            raise ValueError('flow data exceeds depth limit')
        if isinstance(v, dict):
            for k, item in v.items():
                if not isinstance(k, str):
                    raise ValueError('JSON object keys must be strings')
                depth(item, level + 1)
        elif isinstance(v, list):
            for item in v:
                depth(item, level + 1)
    depth(value)
    if len(encode(value).encode()) > MAX_BYTES:
        raise ValueError('flow data exceeds 1 MiB')
    return value


def records(value):
    if not isinstance(value, dict) or len(value) > 128 or any(
            not isinstance(k, str) or not 1 <= len(k.encode()) <= 128 for k in value):
        raise ValueError('expected at most 128 keyed records')
    return value


def run(node, parts):
    if node['operation'] == 'union':
        result = {}
        for part in parts:
            part = records(part)
            if result.keys() & part.keys():
                raise ValueError('duplicate union key')
            result.update(part)
        return records(result)
    if node['operation'] == 'project':
        value = parts[0]
        for key in node['path']:
            if not isinstance(value, dict) or key not in value:
                raise ValueError('missing projection key')
            value = value[key]
        return value
    base, other = records(parts[0]), records(parts[1])
    if node['operation'] == 'select':
        return {k: v for k, v in base.items() if isinstance(other.get(k), dict)
                and other[k].get(node['field']) == node['equals']}
    if node['operation'] != 'overlay':
        raise ValueError('unknown data operation')
    removed = records(parts[2]) if len(parts) == 3 else {}
    if not other.keys() <= base.keys() or not removed.keys() <= base.keys():
        raise ValueError('unknown overlay key')
    result = {}
    for key, value in base.items():
        if key in removed:
            continue
        replacement = other.get(key)
        accept = key in other
        if accept and 'max_string_bytes' in node:
            accept = isinstance(replacement, str) and len(replacement.encode()) <= node['max_string_bytes']
        if accept and 'min_string_bytes' in node:
            accept = isinstance(replacement, str) and len(replacement.encode()) >= node['min_string_bytes']
        if accept and node.get('only_shrink'):
            accept = isinstance(replacement, str) and isinstance(value, str) and len(replacement.encode()) < len(value.encode())
        result[key] = replacement if accept else value
    return result
