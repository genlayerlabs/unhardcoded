"""Public System One payload validation, independent of routing and transport."""
import json
import math
import re

from decision_providers import DecisionError, probability, validate_answer

MAX_REQUEST_BYTES = 32000
MAX_RESPONSE_BYTES = 131072


def decision_family(model):
    """Normalize vendor prefixes, never collapse versions or latest aliases."""
    return str(model).rsplit('/', 1)[-1]


def known_decision_model(model):
    # Quarantine known Jev names even if a discovery snapshot lacks metadata.
    return bool(re.fullmatch(r'jev-(?:latest|\d[\w.-]*)', decision_family(model), re.I))


def validate_payload(payload):
    if not isinstance(payload, dict) or set(payload) != {'state', 'questions'}:
        raise ValueError('Provide state and questions.')
    if not isinstance(payload['state'], (str, dict, list)):
        raise ValueError('state must be text, an object or an array.')
    questions = payload['questions']
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 32:
        raise ValueError('Provide between 1 and 32 questions.')
    for name, question in questions.items():
        if not isinstance(name, str) or not 1 <= len(name) <= 64 or not isinstance(question, dict):
            raise ValueError('Invalid question.')
        if set(question) - {'type', 'instructions', 'criteria'}:
            raise ValueError('Unsupported question field.')
        instruction = question.get('instructions')
        if not isinstance(instruction, str) or not 1 <= len(instruction) <= 2000:
            raise ValueError('Each question needs instructions (1–2000 characters).')
        kind, criteria = question.get('type'), question.get('criteria')
        if kind == 'choice':
            if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 32:
                raise ValueError('Choice questions need 1–32 criteria.')
            if any(not isinstance(k, str) or not 1 <= len(k) <= 64 for k in criteria):
                raise ValueError('Invalid choice name.')
            values = criteria.values()
        elif kind == 'score':
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 32:
                raise ValueError('Score questions need 2–32 ordered criteria.')
            values = criteria
        elif kind == 'noul':
            if criteria is not None and (not isinstance(criteria, dict) or set(criteria) != {'true', 'false'}):
                raise ValueError('Noul criteria must describe true and false.')
            values = (criteria or {}).values()
        else:
            raise ValueError('Supported question types: choice, score, noul.')
        if any(v is not None and (not isinstance(v, str) or len(v) > 2000) for v in values):
            raise ValueError('Invalid criterion description.')
    if len(json.dumps(payload, allow_nan=False).encode()) > MAX_REQUEST_BYTES:
        raise ValueError('Decision request exceeds 32000 bytes.')
    return payload


def validate_response(data, payload):
    if not isinstance(data, dict) or not isinstance(data.get('model'), str) or not data['model']:
        raise ValueError('Missing decision model.')
    answers = data.get('answers')
    if not isinstance(answers, dict) or set(answers) != set(payload['questions']):
        raise ValueError('Missing or unexpected answers.')
    for key, question in payload['questions'].items():
        answer = answers[key]
        kind = question['type']
        if not isinstance(answer, dict) or answer.get('type') != kind:
            raise ValueError('Invalid answer type.')
        if answer.get('confidence') is not None and not probability(answer['confidence']):
            raise ValueError('Invalid confidence.')
        if kind == 'choice':
            try:
                validate_answer(answer, list(question['criteria']))
            except DecisionError as exc:
                raise ValueError('Invalid choice answer.') from exc
        elif kind == 'noul':
            if not probability(answer.get('noul')):
                raise ValueError('Invalid noul answer.')
        else:
            score = answer.get('score')
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= len(question['criteria']) - 1:
                raise ValueError('Invalid score.')
            probs = answer.get('probabilities')
            if probs is not None:
                if (not isinstance(probs, dict) or set(probs) != {str(i) for i in range(len(question['criteria']))}
                        or any(not probability(p) for p in probs.values()) or abs(sum(probs.values()) - 1) > 1e-6):
                    raise ValueError('Invalid score distribution.')
    usage = data.get('usage') or {}
    if not isinstance(usage, dict):
        raise ValueError('Invalid usage.')
    for key in ('input_tokens', 'output_tokens', 'total_tokens'):
        if key in usage and (type(usage[key]) is not int or usage[key] < 0):
            raise ValueError('Invalid token usage.')
    cost = usage.get('cost')
    if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
        raise ValueError('Invalid cost.')
    return {'model': data['model'], 'answers': answers, 'usage': usage}
