"""Offline paired evaluation. No network, credentials or production traffic."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def evaluate(cases, measurements):
    expected = {case['id']: case for case in cases}
    if len(expected) != len(cases) or not cases:
        raise ValueError('Cases require unique IDs')
    by_id = {row['id']: row for row in measurements}
    if len(by_id) != len(measurements) or set(by_id) != set(expected):
        raise ValueError('Exactly one paired measurement is required per case')
    correct, violations, unknown, quality_delta = 0, 0, 0, 0
    baseline_cost, automatic_cost = 0., 0.
    latency_deltas, confusion = [], Counter()
    for key, case in expected.items():
        row = by_id[key]
        selected = row['selected_id']
        correct += selected in case['acceptable_policies']
        confusion[(case['acceptable_policies'][0], selected)] += 1
        if type(row['constraint_violations']) is not int or row['constraint_violations'] < 0:
            raise ValueError('A measured nonnegative violation count is required')
        violations += row['constraint_violations']
        for field in ('baseline_quality', 'automatic_quality'):
            if not number(row[field]) or row[field] > 1:
                raise ValueError('Quality scores must be in [0, 1]')
        quality_delta += row['automatic_quality'] - row['baseline_quality']
        for field in ('baseline_latency_ms', 'automatic_latency_ms'):
            if not number(row[field]):
                raise ValueError('End-to-end latencies must be nonnegative')
        latency_deltas.append(row['automatic_latency_ms'] - row['baseline_latency_ms'])
        costs = [row[k] for k in ('baseline_cost_usd', 'inference_cost_usd', 'decision_cost_usd')]
        if any(v is not None and not number(v) for v in costs):
            raise ValueError('Costs must be nonnegative or null when unknown')
        if None in costs:
            unknown += 1
        else:
            baseline_cost += costs[0]
            automatic_cost += costs[1] + costs[2]
    n = len(cases)
    ordered = sorted(latency_deltas)
    return {'cases': n, 'selection_accuracy': correct / n, 'constraint_violations': violations,
        'mean_quality_delta': quality_delta / n, 'unknown_cost_pairs': unknown,
        'baseline_cost_usd': baseline_cost if not unknown else None,
        'automatic_cost_usd': automatic_cost if not unknown else None,
        'net_savings_usd': baseline_cost - automatic_cost if not unknown else None,
        'mean_latency_delta_ms': sum(ordered) / n,
        'p95_latency_delta_ms': ordered[math.ceil(n * .95) - 1],
        'confusion': [{'expected': a, 'selected': b, 'count': count}
                      for (a, b), count in sorted(confusion.items())]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path)
    parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('automatic-v1.json'))
    args = parser.parse_args()
    print(json.dumps(evaluate(json.loads(args.cases.read_text()), json.loads(args.measurements.read_text())), indent=2))
