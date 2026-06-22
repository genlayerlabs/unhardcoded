"""Reject events persist to the usage-history JSONL and survive into
non-runtime timeframes — without inflating request aggregates.

This is the surviving slice of PR #1's problem (stats die with the ingress
container): requests were already persisted; rejects only lived in the
in-memory `recent` deque and vanished on every recreate.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["CALLER_KEYS_JSON"] = '{"internal":"default"}'
os.environ["CALLER_KEYS_SHA256_JSON"] = "{}"
os.environ["DASHBOARD_TRUSTED_USER_HEADER"] = ""

import auth_proxy  # noqa: E402


def _with_history_path(monkeypatch, tmp_path) -> Path:
    path = tmp_path / "usage-history.jsonl"
    monkeypatch.setenv("ROUTER_USAGE_HISTORY_PATH", str(path))
    return path


def test_reject_is_persisted_without_remote(monkeypatch, tmp_path):
    path = _with_history_path(monkeypatch, tmp_path)
    auth_proxy._record_reject(reason="rate_limit", path="/v1/chat/completions",
                              caller="wingston", status=429, remote="10.0.0.9")
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(lines) == 1
    row = lines[0]
    assert row["event"] == "reject"
    assert row["reason"] == "rate_limit"
    assert row["caller"] == "wingston"
    assert "remote" not in row, "client IPs are ephemeral diagnostics, never durable"


def test_history_reader_defaults_to_requests_only(monkeypatch, tmp_path):
    path = _with_history_path(monkeypatch, tmp_path)
    rows = [
        {"ts": 100, "event": "request", "caller": "a", "status": 200},
        {"ts": 101, "event": "reject", "caller": "b", "reason": "rate_limit", "status": 429},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # default: pre-existing consumers (provider/login/key-usage snapshots)
    # keep seeing request-shaped rows only
    assert [r["caller"] for r in auth_proxy._read_usage_history()] == ["a"]
    # opt-in: the timeframe stats see both
    both = auth_proxy._read_usage_history(events=("request", "reject"))
    assert [r["event"] for r in both] == ["request", "reject"]


def test_aggregate_counts_rejects_without_inflating_requests(monkeypatch, tmp_path):
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
