import json
from pathlib import Path

import pytest
from auto_policies import catalog
from evaluations.automatic import evaluate


def test_dataset_covers_catalog_and_declares_nontext_fixtures():
    cases = json.loads((Path(__file__).resolve().parents[1] / 'evaluations/automatic-v1.json').read_text())
    assert {c['acceptable_policies'][0] for c in cases} == {p['id'] for p in catalog()}
    assert len({c['id'] for c in cases}) == len(cases) == 24
    assert all(c['fixture_requirements'] != 'text' for c in cases if c['acceptable_policies'][0] in {'vision', 'long-document', 'coding-agent'})


def row(**kw):
    return dict(id='a', selected_id='coding', constraint_violations=0, baseline_quality=.8,
                automatic_quality=.9, baseline_latency_ms=10, automatic_latency_ms=15,
                baseline_cost_usd=.1, inference_cost_usd=.05, decision_cost_usd=.01, **kw)


def test_report_accounts_for_decision_cost_and_measured_quality():
    report = evaluate([{'id': 'a', 'acceptable_policies': ['coding']}], [row()])
    assert report['net_savings_usd'] == pytest.approx(.04)
    assert report['mean_quality_delta'] == pytest.approx(.1)
    assert report['p95_latency_delta_ms'] == 5
    assert report['selection_accuracy'] == 1


def test_unknown_cost_never_becomes_savings_and_missing_cases_fail():
    cases = [{'id': 'a', 'acceptable_policies': ['reasoning']}]
    measurement = row()
    measurement['decision_cost_usd'] = None
    report = evaluate(cases, [measurement])
    assert report['net_savings_usd'] is None and report['selection_accuracy'] == 0
    with pytest.raises(ValueError):
        evaluate(cases, [])
    measurement['decision_cost_usd'] = -1
    with pytest.raises(ValueError):
        evaluate(cases, [measurement])
