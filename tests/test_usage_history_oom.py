"""Durable usage-history stays bounded so the dashboard reader can't OOM.

Regression for the auth-proxy OOM crashloop: every request persisted its full
decision_trace (policy_term AST + ranked candidates + decision_path), bloating
usage-history.jsonl to tens of MB; `_read_usage_history` loaded the whole file,
expanding it to ~1 GB of Python objects and tripping the 512Mi container limit
(OOMKilled → nginx loses its upstream → ALB 502/503). The fix strips the heavy
fields on write, bounds the read to the tail, and rotates the file at a cap.
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


def test_append_strips_heavy_decision_trace(monkeypatch, tmp_path):
    path = _with_history_path(monkeypatch, tmp_path)
    auth_proxy._append_usage_history({
        "ts": 100, "event": "request", "caller": "a", "status": 200,
        "provider": "openai", "model_family": "gpt-5.5", "tokens_total": 12,
        # the bloat that must never reach the durable file:
        "decision_trace": {"policy_term": ["policy"] * 5000, "ranked": [{"x": i} for i in range(500)],
                           "decision_path": [{"event": "attempted"}]},
        "ranked": [{"x": i} for i in range(500)],
    })
    stored = json.loads(path.read_text().splitlines()[0])
    for heavy in ("decision_trace", "ranked", "policy_term", "decision_path"):
        assert heavy not in stored, f"{heavy} must be stripped from durable rows"
    # the fields the aggregations need survive untouched
    assert stored["provider"] == "openai"
    assert stored["tokens_total"] == 12


def test_read_slims_legacy_bloated_rows(monkeypatch, tmp_path):
    path = _with_history_path(monkeypatch, tmp_path)
    # a row written by the old code path, with the trace inline
    path.write_text(json.dumps({
        "ts": 100, "event": "request", "caller": "a", "status": 200,
        "decision_trace": {"policy_term": ["policy"], "ranked": [{"x": 1}]},
    }) + "\n")
    rows = auth_proxy._read_usage_history()
    assert len(rows) == 1
    assert "decision_trace" not in rows[0], "reader must slim pre-fix bloated rows too"
    assert rows[0]["caller"] == "a"


def test_read_is_bounded_to_the_tail(monkeypatch, tmp_path):
    path = _with_history_path(monkeypatch, tmp_path)
    monkeypatch.setattr(auth_proxy, "USAGE_HISTORY_READ_TAIL_BYTES", 2000)
    lines = [json.dumps({"ts": i, "event": "request", "caller": f"c{i}", "status": 200})
             for i in range(2000)]
    path.write_text("\n".join(lines) + "\n")
    assert path.stat().st_size > 2000
    rows = auth_proxy._read_usage_history()
    # only the tail is parsed, and it must be the most recent rows (no partial
    # first line slipping through as a parse error that drops a valid row)
    assert 0 < len(rows) < 2000
    assert rows[-1]["ts"] == 1999
    tss = [r["ts"] for r in rows]
    assert tss == sorted(tss), "tail rows stay in append order"


def test_rotate_caps_file_and_keeps_recent(monkeypatch, tmp_path):
    path = _with_history_path(monkeypatch, tmp_path)
    monkeypatch.setattr(auth_proxy, "USAGE_HISTORY_MAX_BYTES", 4000)
    for i in range(2000):
        auth_proxy._append_usage_history({"ts": i, "event": "request", "caller": f"c{i}", "status": 200})
    assert path.stat().st_size <= 4000, "file must be bounded by the cap"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert rows, "rotation must keep recent rows, not empty the file"
    assert rows[-1]["ts"] == 1999, "the newest row survives rotation"
    assert all(json.loads(l) for l in path.read_text().splitlines() if l.strip()), "every kept line is valid JSON"
