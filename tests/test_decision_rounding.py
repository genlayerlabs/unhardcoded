import json
import random
import time

import httpx
import pytest

from decision_protocol import validate_response
from decision_providers import (DecisionChoice, DecisionError, DecisionRequest,
                                probability_sum_valid, validate_answer)
from decision_providers.jev import JevDecisionProvider


def response(probabilities, model='typesafe/jev-1.13-20260917', kind='choice'):
    choice = max(probabilities, key=probabilities.get)
    answer = {'type': kind, 'probabilities': probabilities, 'confidence': .2}
    answer.update({'choice': choice} if kind == 'choice' else {'score': 1.0})
    return {'model': model, 'answers': {'q': answer}, 'usage': {'input_tokens': 10, 'cost': .000001}}


def payload(keys, kind='choice'):
    return {'state': 'Select an option', 'questions': {'q': {'type': kind, 'instructions': 'Select',
        'criteria': {k: k for k in keys} if kind == 'choice' else list(keys)}}}


@pytest.mark.parametrize('probs', [dict(a=.33, b=.33, c=.33), dict(a=.34, b=.34, c=.33),
                                 {str(i): .03 for i in range(32)}])
def test_rounded_jev_distribution_survives_without_modifying_values(probs):
    data = response(probs)
    accepted = validate_response(data, payload(probs))
    assert accepted['answers']['q']['probabilities'] == probs
    selected, raw, confidence, entropy = validate_answer(data['answers']['q'], list(probs), probability_decimals=2)
    assert raw == probs and confidence == .2 and selected in probs
    assert 0 <= entropy <= 1


def test_rounding_does_not_relax_other_models_or_recorded_choices():
    probs = dict(a=.33, b=.33, c=.33)
    with pytest.raises(ValueError):
        validate_response(response(probs, model='other-decision-model'), payload(probs))
    with pytest.raises(DecisionError):
        validate_answer(response(probs)['answers']['q'], list(probs))


@pytest.mark.parametrize('probs', [dict(a=.3, b=.3, c=.3), dict(a=.4, b=.4, c=.4),
                                 dict(a=.331, b=.331, c=.331), dict(a=0, b=0), dict(a=.99),
                                 dict(a=-.01, b=1.01), dict(a=True, b=0), dict(a=float('nan'), b=1)])
def test_unexplained_mass_missing_precision_and_invalid_values_still_rejected(probs):
    with pytest.raises(ValueError):
        validate_response(response(probs), payload(probs))


def test_unknown_choice_missing_option_and_non_argmax_still_rejected():
    probs = dict(a=.5, b=.25, c=.25)
    for choice in ['missing', 'b']:
        data = response(probs);data['answers']['q']['choice'] = choice
        with pytest.raises(ValueError):validate_response(data, payload(probs))
    with pytest.raises(ValueError):validate_response(response(dict(a=.5, b=.5)), payload(['a', 'b', 'c']))


def test_rounded_score_distribution_has_the_same_mass_rule():
    probs = {'0': .33, '1': .33, '2': .33}
    result = validate_response(response(probs, kind='score'), payload(probs, kind='score'))
    assert result['answers']['q']['probabilities'] == probs
    with pytest.raises(ValueError):
        validate_response(response({'0': .3, '1': .3, '2': .3}, kind='score'), payload(probs, kind='score'))


def test_real_normalized_distributions_remain_valid_after_display_rounding():
    rng = random.Random(921)
    for count in range(1, 33):
        for _ in range(40):
            weights = [rng.random() for _ in range(count)]
            total = sum(weights)
            displayed = [round(value / total, 2) for value in weights]
            assert probability_sum_valid(displayed, decimal_places=2)
    assert not probability_sum_valid([0.] * 32, decimal_places=2)
    assert not probability_sum_valid([.7] + [0.] * 31, decimal_places=2)


@pytest.mark.asyncio
async def test_direct_jev_provider_preserves_rounded_distribution_and_normalizes_only_entropy():
    req = DecisionRequest('q', 'Choose', {}, tuple(DecisionChoice(k, k) for k in ['a', 'b', 'c']))
    data = response(dict(a=.33, b=.33, c=.33))
    data['answers']['selection'] = data['answers'].pop('q')
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=data))) as client:
        result = await JevDecisionProvider('test-only', client=client).decide(req, deadline=time.monotonic() + 1)
    assert result.probabilities == dict(a=.33, b=.33, c=.33)
    assert result.normalized_entropy == pytest.approx(1.)
    assert result.confidence == .2 and result.cost_usd == .000001
