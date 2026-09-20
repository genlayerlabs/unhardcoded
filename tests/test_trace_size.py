"""Large decision catalogs stay bounded on successful and failed responses."""
import copy
import json

import shim


def large_trace():
    return {
        "rejected": [{"reason": "protocol_mismatch" if i % 2 else "predicate",
                      "candidate": {"served_model_id": f"model-{i}", "metadata": "x" * 1024}}
                     for i in range(460)],
        "decision_path": [{"event": "attempted", "provider_id": "openrouter_market",
                           "error_kind": "timeout", "latency_ms": 300}],
        "total_latency_ms": 301, "policy_fingerprint": "fixture",
    }


def test_rejection_samples_are_bounded_without_losing_counts_or_attempts():
    trace = large_trace()
    before = copy.deepcopy(trace)
    trimmed = shim._trim_trace(trace)
    assert len(trimmed["rejected"]) == 10
    assert trimmed["rejected_total"] == 460
    assert trimmed["rejected_reasons"] == {"predicate": 230, "protocol_mismatch": 230}
    assert trimmed["decision_path"] == trace["decision_path"]
    assert trimmed["total_latency_ms"] == 301
    assert trimmed["policy_fingerprint"] == "fixture"
    assert len(json.dumps(trimmed)) < len(json.dumps(trace)) / 20
    assert trace == before
    assert shim._trim_trace(trimmed) == trimmed


def test_flow_rejections_are_bounded_at_each_nested_level():
    trace = {"flow_nodes": [{"node": "outer", "provider": "fixture", "decision_trace": {
        "flow_nodes": [{"node": "inner", "decision_trace": large_trace()}]}}]}
    trimmed = shim._trim_trace(trace)
    outer = trimmed["flow_nodes"][0]
    assert outer["provider"] == "fixture"
    inner = outer["decision_trace"]["flow_nodes"][0]["decision_trace"]
    assert len(inner["rejected"]) == 10 and inner["rejected_total"] == 460
    assert len(trace["flow_nodes"][0]["decision_trace"]["flow_nodes"][0]["decision_trace"]["rejected"]) == 460


def test_error_surface_bounds_trace_and_keeps_attempt_failure():
    response = shim._openai_error_from_router({"ok": False, "error": "exhausted", "trace": large_trace()})
    data = json.loads(response.body)
    assert response.status_code >= 400
    trace = data["x_router"]["decision_trace"]
    assert len(trace["rejected"]) == 10
    assert trace["rejected_total"] == 460
    assert trace["decision_path"][0]["error_kind"] == "timeout"
