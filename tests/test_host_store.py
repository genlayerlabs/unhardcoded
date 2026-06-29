"""The host operational store (Postgres) call ledger + operator state. The fact
table from which route/session views are derived; durable set_* with a success
bool; time-bounded retention; one-shot legacy backfill."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import host_store as hs  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def store(host_store_clean):
    return host_store_clean


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


# ---- one-shot legacy backfill -------------------------------------------------

def _legacy_files(tmp_path, monkeypatch, *, ov=None, prov=None, keys=None):
    if ov is not None:
        (tmp_path / "ov.json").write_text(ov)
        monkeypatch.setenv("LLM_ROUTER_CONFIG_OVERRIDES", str(tmp_path / "ov.json"))
    else:
        monkeypatch.setenv("LLM_ROUTER_CONFIG_OVERRIDES", str(tmp_path / "absent-ov.json"))
    if prov is not None:
        (tmp_path / "prov.json").write_text(prov)
        monkeypatch.setenv("PROVIDERS_OVERLAY_PATH", str(tmp_path / "prov.json"))
    else:
        monkeypatch.setenv("PROVIDERS_OVERLAY_PATH", str(tmp_path / "absent-prov.json"))
    if keys is not None:
        (tmp_path / "keys.json").write_text(keys)
        monkeypatch.setenv("DASHBOARD_ISSUED_KEYS_PATH", str(tmp_path / "keys.json"))
    else:
        monkeypatch.setenv("DASHBOARD_ISSUED_KEYS_PATH", str(tmp_path / "absent-keys.json"))


def test_backfill_seeds_empty_tables(store, tmp_path, monkeypatch):
    _legacy_files(
        tmp_path, monkeypatch,
        ov=json.dumps({"compaction.at_tokens": 50000}),
        prov=json.dumps({"providers": {"groq": {
            "base_url": "https://x", "auth_env": "G", "added_at": 1}}}),
        keys=json.dumps({"crm": {"status": "active"}}))
    hs.migrate_legacy_json()
    assert hs.get_overrides() == {"compaction.at_tokens": 50000}
    assert hs.get_provider_overlays()["providers"]["groq"]["auth_env"] == "G"
    assert hs.get_consumer_keys() == ({"crm": {"status": "active"}}, True)


def test_backfill_is_noop_when_table_nonempty(store, tmp_path, monkeypatch):
    hs.set_overrides({"compaction.at_tokens": 99})        # already migrated
    _legacy_files(tmp_path, monkeypatch,
                  ov=json.dumps({"compaction.at_tokens": 50000}))
    hs.migrate_legacy_json()
    assert hs.get_overrides() == {"compaction.at_tokens": 99}  # NOT clobbered


def test_backfill_noop_when_file_absent(store, tmp_path, monkeypatch):
    _legacy_files(tmp_path, monkeypatch)                   # all paths absent
    hs.migrate_legacy_json()
    assert hs.get_overrides() == {}
    assert hs.get_provider_overlays() == {"providers": {}}


def test_backfill_failsoft_on_corrupt_file(store, tmp_path, monkeypatch):
    _legacy_files(tmp_path, monkeypatch, ov="not json{")
    hs.migrate_legacy_json()                              # must not raise
    assert hs.get_overrides() == {}                       # corrupt -> empty, not fatal


def test_set_returns_bool_contract(store):
    # Durable writes report success/failure so callers don't pretend a save that
    # didn't persist (a silent failed key revoke would be a security hole).
    assert hs.set_consumer_keys({"crm": {"status": "active"}}) is True
    assert hs.set_overrides({"compaction.at_tokens": 50000}) is True
    assert hs.set_provider_overlays({"groq": {"auth_env": "G", "added_at": 1}}) is True


def test_set_is_atomic_and_returns_false_on_failure(store, monkeypatch):
    # A set_* is ONE transaction: a failure mid-write rolls back (existing data
    # intact, no half-applied DELETE) and returns False. With the pool model each
    # write is its own connection/transaction, so a failure cannot poison another.
    import contextlib
    assert hs.set_consumer_keys({"crm": {"status": "active"}}) is True
    real_pool = hs._get_pool()

    class _FailConn:
        def __init__(self, c): self._c = c
        def execute(self, *a, **k): return self._c.execute(*a, **k)   # DELETE runs
        def cursor(self, *a, **k):
            class _C:
                def executemany(self, *a, **k): raise RuntimeError("boom")  # then dies
                def __enter__(self): return self
                def __exit__(self, *x): return False
            return _C()
        def __getattr__(self, n): return getattr(self._c, n)

    class _FailPool:
        @contextlib.contextmanager
        def connection(self):
            with real_pool.connection() as c:    # real txn -> rolls back on the raise
                yield _FailConn(c)
        def __getattr__(self, n): return getattr(real_pool, n)

    monkeypatch.setattr(hs, "_get_pool", lambda: _FailPool())
    ok = hs.set_consumer_keys({"crm2": {"status": "x"}})   # DELETE then executemany boom
    monkeypatch.setattr(hs, "_get_pool", lambda: real_pool)

    assert ok is False                                     # failure surfaced, not swallowed
    assert hs.get_consumer_keys()[0] == {"crm": {"status": "active"}}  # rolled back, intact


# ---- peer_offers (antseed market book; sidecar writes, host reads) -------------

def _insert_peer_offer(store, peer_id, service, observed_at, **over):
    row = {"price_in": 0.5, "price_out": 1.0, "price_cached_in": None,
           "max_concurrency": 5, "reputation": None, "last_seen": 1,
           "first_seen": observed_at, "fetched_at": observed_at}
    row.update(over)
    with store._get_pool().connection() as conn:
        conn.execute(
            "INSERT INTO peer_offers (peer_id, service, price_in, price_out,"
            " price_cached_in, max_concurrency, reputation, last_seen,"
            " observed_at, first_seen, fetched_at)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (peer_id, service, row["price_in"], row["price_out"],
             row["price_cached_in"], row["max_concurrency"], row["reputation"],
             row["last_seen"], observed_at, row["first_seen"], row["fetched_at"]))


def test_peer_offers_returns_rows_in_reader_shape(store):
    now = int(time.time() * 1000)
    _insert_peer_offer(store, "peerA", "gpt-5", now, reputation=80.0)
    rows = store.peer_offers()
    assert len(rows) == 1
    r = rows[0]
    assert r == {"peer_id": "peerA", "service": "gpt-5", "price_in": 0.5,
                 "price_out": 1.0, "price_cached_in": None, "max_concurrency": 5,
                 "reputation": 80.0, "last_seen": 1}


def test_peer_offers_window_filters_stale_rows(store):
    now = int(time.time() * 1000)
    _insert_peer_offer(store, "fresh", "m", now)
    _insert_peer_offer(store, "stale", "m", now - 20 * 60 * 1000)  # 20 min ago
    assert {r["peer_id"] for r in store.peer_offers(window_ms=15 * 60 * 1000)} == {"fresh"}
    # a wider window readmits the older row
    assert {r["peer_id"] for r in store.peer_offers(window_ms=30 * 60 * 1000)} == {"fresh", "stale"}


# ---- buyer_status (antseed buyer escrow/pin/wallet; sidecar writes) ------------

def test_buyer_status_roundtrip_and_absent(store):
    assert store.buyer_status("antseed") is None      # absent -> None (degraded)
    with store._get_pool().connection() as conn:
        conn.execute(
            "INSERT INTO buyer_status (pid, pinned_peer_id, deposits_available,"
            " deposits_reserved, wallet_address, connection_state, fetched_at)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s)",
            ("antseed", "peerX", "1.5", "0.2", "0xabc", "connected", 1))
    row = store.buyer_status("antseed")
    assert row == {"pid": "antseed", "pinned_peer_id": "peerX",
                   "deposits_available": "1.5", "deposits_reserved": "0.2",
                   "wallet_address": "0xabc", "connection_state": "connected"}
