"""Reject events (auth fail / rate limit) are a RUNTIME view, not LLM calls.

Since #5 (usage-history.jsonl retired, the persistent stats derive from the
`calls` ledger) a reject does NOT enter `calls` — it lives in the in-memory
recent feed only. The aggregation still counts reject rows separately from
requests when it is handed them (the runtime path), never inflating request
aggregates.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["CALLER_KEYS_JSON"] = '{"internal":"default"}'
os.environ["CALLER_KEYS_SHA256_JSON"] = "{}"
os.environ["DASHBOARD_TRUSTED_USER_HEADER"] = ""

import auth_proxy  # noqa: E402


def test_reject_is_recorded_in_the_runtime_feed():
    before = auth_proxy._stats["total_rejects"]
    auth_proxy._record_reject(reason="rate_limit", path="/v1/chat/completions",
                              caller="wingston", status=429, remote="10.0.0.9")
    assert auth_proxy._stats["total_rejects"] == before + 1
    row = auth_proxy._stats["recent"][0]
    assert row["event"] == "reject"
    assert row["reason"] == "rate_limit"
    assert row["caller"] == "wingston"


def test_aggregate_counts_rejects_without_inflating_requests():
    rows = [
        {"ts": 100, "event": "request", "caller": "a", "status": 200,
         "tokens_in": 5, "tokens_out": 7, "tokens_total": 12},
        {"ts": 101, "event": "reject", "caller": "b", "reason": "rate_limit", "status": 429},
        {"ts": 102, "event": "reject", "caller": "b", "reason": "route_not_allowed", "status": 403},
    ]
    agg = auth_proxy._aggregate_usage_rows(rows)
    assert agg["totals"]["requests"] == 1, "rejects must not count as requests"
    assert agg["totals"]["rejects"] == 2
    assert agg["totals"]["errors"] == 0, "reject statuses must not count as request errors"
    assert agg["totals"]["tokens_total"] == 12
    assert set(agg["by_caller"]) == {"a"}, "reject rows must not create request counters"
