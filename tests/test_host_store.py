"""The host operational store (SQLite) call ledger. Dual-written alongside
usage-history; the fact table from which route/session views are later derived.
Fail-soft writes, time-bounded retention."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import host_store as hs  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB_PATH", str(tmp_path / "host-store.db"))
    hs.reset()
    yield hs
    hs.reset()


def _row(**over):
    row = {
        "ts": 1_000_000, "usage_event_id": "ev1", "session": "sess-A",
        "key_sha256": "a" * 64, "caller": "app1", "provider": "openrouter",
        "model_family": "gpt-5.5", "served_model_id": "openai/gpt-5.5",
        "requested_model": "", "status": 200, "latency_ms": 900.0,
        "tokens_in": 700, "tokens_out": 300, "tokens_total": 1000,
        "cost_usd": 0.005,
    }
    row.update(over)
    return row


def test_insert_and_recent_roundtrip(store):
    store.insert_call(_row())
    rows = store.recent_calls()
    assert len(rows) == 1
    r = rows[0]
    assert r["provider_id"] == "openrouter"
    assert r["model_family"] == "gpt-5.5"
    assert r["route_key"] == "openrouter|gpt-5.5|openai/gpt-5.5"
    assert r["tokens_total"] == 1000
    assert r["cost_usd"] == 0.005
    assert r["session_id"] == "sess-A"


def test_count_and_ordering_newest_first(store):
    store.insert_call(_row(usage_event_id="a", ts=1))
    store.insert_call(_row(usage_event_id="b", ts=2))
    assert store.count() == 2
    assert [r["usage_event_id"] for r in store.recent_calls()] == ["b", "a"]


def test_missing_fields_become_null_not_crash(store):
    # A sparse row (e.g. an error with no tokens/cost) must insert, not raise.
    store.insert_call({"ts": 5, "status": 503, "caller": "app1"})
    r = store.recent_calls()[0]
    assert r["status"] == 503
    assert r["cost_usd"] is None and r["tokens_total"] is None
    assert r["route_key"] is None  # no provider/family -> no route key


def test_insert_is_fail_soft(store, monkeypatch):
    # A bad value must be swallowed (best-effort ledger never breaks a request).
    # Force an error by pointing the connection at an unwritable path mid-run.
    store.insert_call(_row())
    monkeypatch.setattr(hs, "_route_key", lambda *a: 1 / 0)  # raise inside insert
    store.insert_call(_row())            # must not raise
    assert store.count() == 1            # the bad insert was dropped, not fatal


def test_retention_prunes_old_rows(store, monkeypatch):
    monkeypatch.setattr(hs, "_RETENTION_DAYS", 1)
    monkeypatch.setattr(hs, "_PRUNE_EVERY", 1)   # prune on every insert
    old = int(time.time()) - 10 * 86400          # 10 days old
    new = int(time.time())
    store.insert_call(_row(usage_event_id="old", ts=old))
    store.insert_call(_row(usage_event_id="new", ts=new))   # triggers a prune
    ids = [r["usage_event_id"] for r in store.recent_calls()]
    assert "new" in ids and "old" not in ids


def test_route_key_shape_matches_provider_family_served(store):
    assert hs._route_key("antseed", "glm-5.2", "peerX") == "antseed|glm-5.2|peerX"
    assert hs._route_key(None, None, None) is None
