"""Request-local decision routing for an admitted flow node.

The decision model selects a declared policy; it cannot supply a policy, model
name, command, or message. Only the selected generative policy is executed.
"""
from __future__ import annotations

import asyncio
import json
import math

from decision_protocol import validate_payload, validate_response


def decision_state(messages, prompt):
    """Bounded recent evidence including commands/results, with explicit truncation."""
    remaining, selected, omitted = 12000, [], 0
    for message in reversed(messages):
        text = json.dumps(message, ensure_ascii=True, separators=(',', ':'))
        if remaining < 128:
            omitted += 1
            continue
        limit = min(2400, remaining)
        selected.append({'message': text[:limit], 'truncated': len(text) > limit})
        remaining -= min(len(text), limit)
    return {'history': list(reversed(selected)), 'omitted_messages': omitted,
            'request': prompt[:2000], 'request_truncated': len(prompt) > 2000}


def result_cost(result):
    response, chosen = result.get('response') or {}, result.get('chosen') or {}
    reported = response.get('cost_reported')
    if type(reported) in (int, float) and math.isfinite(reported) and reported >= 0:
        return reported
    pin = chosen.get('raw_price_in', chosen.get('price_in'))
    pout = chosen.get('raw_price_out', chosen.get('price_out'))
    if pin is None or pout is None or response.get('tokens_in') is None:
        return None
    if 'raw_price_in' not in chosen and 'raw_price_out' not in chosen:
        mult = chosen.get('price_multiplier') or 1
        pin, pout = pin / mult, pout / mult
    cached = response.get('tokens_cached') or 0
    return (max(0, response['tokens_in'] - cached) * pin + cached * pin * .1
            + (response.get('tokens_out') or 0) * pout) / 1e6


async def select_node_policy(host, routing, messages, prompt, session=None, call_override=None):
    """Select once; declared fallback on failure/uncertainty, never run both branches."""
    choices = routing['choices']
    selected = routing['fallback']
    trace = {'selected': selected, 'fallback_used': True, 'cost_usd': None}
    payload = {'state': decision_state(messages, prompt), 'questions': {'route': {
        'type': 'choice', 'instructions': routing['instructions'],
        'criteria': {key: value['description'] for key, value in choices.items()}}}}
    contract = {'protocol': 'decisions', 'decision': payload, 'policy_ir': routing['policy'],
                'timeout_ms': routing['timeout_ms'], 'session': session}
    try:
        validate_payload(payload)
        async with asyncio.timeout(routing['timeout_ms'] / 1000):
            result = await host.execute_async(contract, call_override=call_override)
        trace['cost_usd'] = result_cost(result)
        trace['decision_trace'] = result.get('trace')
        response = result.get('response') or {}
        trace['tokens_in'] = response.get('tokens_in')
        trace['tokens_out'] = response.get('tokens_out')
        trace['model'] = (result.get('chosen') or {}).get('served_model_id')
        if not result.get('ok'):
            trace['fallback_reason'] = 'decision_failed'
        else:
            decision = validate_response(response.get('decision'), payload)
            answer = decision['answers']['route']
            confidence = answer.get('confidence')
            trace.update(proposed=answer['choice'], confidence=confidence,
                         probabilities=answer['probabilities'])
            if confidence is None or confidence < routing['min_confidence']:
                trace['fallback_reason'] = 'uncertain'
            else:
                selected = answer['choice']
                trace.update(selected=selected, fallback_used=False)
    except TimeoutError:
        trace['fallback_reason'] = 'decision_timeout'
    except (ValueError, TypeError, KeyError):
        trace['fallback_reason'] = 'invalid_decision'
    except Exception as exc:
        # Provider failures should not bypass the explicitly admitted fallback.
        trace.update(fallback_reason='decision_error', error=type(exc).__name__)
    return choices[selected]['policy'], trace
