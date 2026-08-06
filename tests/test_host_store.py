"""The host operational store (Postgres) call ledger + operator state. The fact
table from which route/session views are derived; durable set_* with a success
bool; time-bounded retention."""
from __future__ import annotations

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


def test_consumer_spend_usd_sums_only_that_consumer(store):
    store.insert_call(_row(usage_event_id="budget-1", caller="validator-001", cost_usd=1.25))
    store.insert_call(_row(usage_event_id="budget-2", caller="validator-001", cost_usd=0.75))
    store.insert_call(_row(usage_event_id="other", caller="validator-002", cost_usd=99))
    assert store.consumer_spend_usd("validator-001") == (2.0, True)


def test_hourly_analytics_rollup_is_idempotent_and_filterable(store):
    base = 1_800_000_000
    store.insert_call(_row(ts=base + 10, usage_event_id="a1", caller="a",
                           provider="p1", model_family="m1", status=200,
                           tokens_in=10, tokens_out=5, tokens_total=15,
                           tokens_cached=5, cost_usd=1.0))
    store.insert_call(_row(ts=base + 20, usage_event_id="a2", caller="a",
                           provider="p2", model_family="m2", status=500,
                           tokens_in=20, tokens_out=10, tokens_total=30,
                           tokens_cached=10, cost_usd=2.0))
    store.insert_call(_row(ts=base + 30, usage_event_id="b1", caller="b",
                           provider="p1", model_family="m1", status=200,
                           tokens_in=30, tokens_out=15, tokens_total=45,
                           tokens_cached=0, cost_usd=3.0))
    first = store.rollup_analytics(base, base + 3600)
    assert first["rows"] == 3
    agg, state, ok = store.analytics_aggregate(base, caller="a")
    assert ok is True
    assert agg["totals"]["requests"] == 2
    assert agg["totals"]["errors"] == 1
    assert agg["totals"]["cost_usd"] == 3.0
    assert agg["totals"]["tokens_cached"] == 15
    assert agg["totals"]["cache_hit_rate"] == 0.5
    assert agg["totals"]["cost_per_request"] == 1.5
    assert set(agg["by_provider"]) == {"p1", "p2"}
    assert state["covered_until"] >= base + 3600
    # Replacing the same buckets must not double count.
    store.rollup_analytics(base, base + 3600)
    agg2, _, _ = store.analytics_aggregate(base)
    assert agg2["totals"]["requests"] == 3
    assert agg2["totals"]["cost_usd"] == 6.0


def test_usage_rows_since_ts_filters_in_query(store):
    # The timeframe window lives in SQL now (idx_calls_ts), not Python: a windowed
    # dashboard view reads only its window, never the whole retention table to then
    # discard almost all of it. Restores the bound the retired usage-history tail
    # read gave (test_usage_history_oom).
    store.insert_call(_row(usage_event_id="old", ts=1_000_000))
    store.insert_call(_row(usage_event_id="new", ts=2_000_000))
    assert [r["usage_event_id"] for r in store.usage_rows(since_ts=1_500_000)] == ["new"]
    # since_ts=None ("all") still returns the retained rows (bounded by retention).
    assert {r["usage_event_id"] for r in store.usage_rows()} == {"old", "new"}


def test_usage_rows_caller_filters_in_query(store):
    store.insert_call(_row(usage_event_id="a", caller="app1"))
    store.insert_call(_row(usage_event_id="b", caller="app2"))
    assert {r["usage_event_id"] for r in store.usage_rows(caller="app1")} == {"a"}


def test_usage_rows_all_is_floored_at_the_retention_horizon(store, monkeypatch):
    # "all" (since_ts=None) is not a bare table scan: it is floored at
    # now - retention, so the read is always time-bounded. Rows older than the
    # horizon (not yet pruned) are excluded from the aggregation feed.
    monkeypatch.setattr(hs, "_RETENTION_DAYS", 7)
    now = int(time.time())
    store.insert_call(_row(usage_event_id="recent", ts=now - 86400))       # 1 day
    store.insert_call(_row(usage_event_id="stale", ts=now - 30 * 86400))   # 30 days
    assert {r["usage_event_id"] for r in store.usage_rows()} == {"recent"}


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


def test_served_by_and_tokens_cached_recorded(store):
    # #3: the call ledger carries the executed route identity (served_by, from
    # the engine's chosen) and the cache-token breakdown — raw per-call facts the
    # #4 route/analytics views derive from.
    store.insert_call(_row(served_by="peerGood", tokens_cached=128))
    r = store.recent_calls()[0]
    assert r["served_by"] == "peerGood"
    assert r["tokens_cached"] == 128
    # absent -> NULL, fail-soft (older rows / direct calls without the field)
    store.insert_call(_row(usage_event_id="ev2"))
    r2 = [c for c in store.recent_calls() if c["usage_event_id"] == "ev2"][0]
    assert r2["served_by"] is None and r2["tokens_cached"] is None


# ---- route_observations -> route_stats (reliability/latency derived on the fly) -

def test_route_stats_derives_reliability_and_latency(store):
    from conftest import seed_route_obs
    # peerA: 4 ok + 1 fail -> success 0.8; latency avg over OK only
    seed_route_obs("antseed", "m", "peerA", ok=True, latency_ms=100, n=3)
    seed_route_obs("antseed", "m", "peerA", ok=True, latency_ms=300)   # ok=4, lat avg=150
    seed_route_obs("antseed", "m", "peerA", ok=False, latency_ms=9999)  # failure: latency ignored
    seed_route_obs("antseed", "m", "peerB", ok=True, latency_ms=50)
    st = store.route_stats()
    assert st["antseed|m|peerA"]["success_rate"] == 0.8
    assert st["antseed|m|peerA"]["latency_ms"] == 150   # avg(100,100,100,300), failure excluded
    assert st["antseed|m|peerA"]["count"] == 5
    assert st["antseed|m|peerB"]["success_rate"] == 1.0
    assert "antseed|m|missing" not in st


def test_route_stats_window_excludes_old_observations(store):
    from conftest import seed_route_obs
    import time
    now = int(time.time() * 1000)
    seed_route_obs("p", "m", "fresh", ok=True, ts=now)
    seed_route_obs("p", "m", "stale", ok=True, ts=now - 20 * 60 * 1000)  # 20 min ago
    assert set(store.route_stats(window_ms=15 * 60 * 1000)) == {"p|m|fresh"}
    assert set(store.route_stats(window_ms=30 * 60 * 1000)) == {"p|m|fresh", "p|m|stale"}


# ---- wallet_ops: the keeper's audit trail + rate-cap ledger -------------------
# Not best-effort telemetry like `calls`: these rows gate real USDC movement, so
# the writers report failure and every read fails in the SAFE direction.

def test_wallet_op_begin_writes_the_intent_before_the_transaction(store):
    op_id = store.wallet_op_begin("antseed", "topup", amount_usdc=5.0,
                                  reason="below trigger", pre_available=0.5)
    assert op_id is not None
    (row,) = store.wallet_ops_recent("antseed")
    assert row["op"] == "topup" and row["outcome"] == "pending"
    assert row["amount_usdc"] == 5.0 and row["pre_available"] == 0.5
    assert row["reason"] == "below trigger" and row["post_available"] is None
    # `pending` means "in flight": open for reconciliation, already counted spent.
    assert [r["id"] for r in store.wallet_ops_open("antseed", "topup")] == [op_id]
    assert store.wallet_op_spend_since("antseed", "topup", 0)["spent_usdc"] == 5.0


def test_wallet_op_finish_records_the_measured_outcome(store):
    op_id = store.wallet_op_begin("antseed", "topup", amount_usdc=5.0,
                                  pre_available=0.5)
    assert store.wallet_op_finish(op_id, "effective", post_available=5.4,
                                  detail="landed") is True
    (row,) = store.wallet_ops_settled("antseed", "topup")
    assert row["outcome"] == "effective" and row["post_available"] == 5.4
    assert store.wallet_ops_open("antseed", "topup") == []


def test_failed_ops_do_not_consume_the_daily_cap(store):
    ok_id = store.wallet_op_begin("antseed", "topup", amount_usdc=5.0)
    store.wallet_op_finish(ok_id, "fired")
    bad_id = store.wallet_op_begin("antseed", "topup", amount_usdc=5.0)
    store.wallet_op_finish(bad_id, "failed")
    spend = store.wallet_op_spend_since("antseed", "topup", 0)
    assert spend["spent_usdc"] == 5.0 and spend["count"] == 1


def test_spend_ledger_is_scoped_per_provider_and_per_op(store):
    store.wallet_op_begin("antseed_a", "topup", amount_usdc=5.0)
    store.wallet_op_begin("antseed_b", "topup", amount_usdc=7.0)
    store.wallet_op_begin("antseed_a", "reclaim_withdraw")
    assert store.wallet_op_spend_since("antseed_a", "topup", 0)["spent_usdc"] == 5.0
    assert store.wallet_op_spend_since("antseed_b", "topup", 0)["spent_usdc"] == 7.0


def test_spend_ledger_window_excludes_older_ops(store):
    op_id = store.wallet_op_begin("antseed", "topup", amount_usdc=5.0)
    store.wallet_op_finish(op_id, "fired")
    old = int(time.time()) - 90_000                       # ~25h ago
    with store._get_pool().connection() as conn:
        conn.execute("UPDATE wallet_ops SET ts=%s WHERE id=%s", (old, op_id))
    since = int(time.time()) - 86_400
    assert store.wallet_op_spend_since("antseed", "topup", since)["spent_usdc"] == 0.0
    assert store.wallet_op_spend_since("antseed", "topup", 0)["spent_usdc"] == 5.0


def test_halt_is_durable_scoped_and_operator_cleared(store):
    assert store.wallet_halted("antseed", "topup") is False
    assert store.wallet_halt("antseed", "topup", "two ineffective deposits") is True
    assert store.wallet_halted("antseed", "topup") is True
    assert store.wallet_halted("antseed", "reclaim") is False   # per-class
    assert store.wallet_halted("antseed_other", "topup") is False  # per-provider
    store.wallet_halt("antseed", "topup", "again")              # idempotent
    assert len([r for r in store.wallet_ops_recent("antseed")
                if r["outcome"] == "halted"]) == 1
    assert store.wallet_clear_halt("antseed", "topup") is True
    assert store.wallet_halted("antseed", "topup") is False


def test_wallet_reads_fail_in_the_safe_direction(store, monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("db down")
    monkeypatch.setattr(hs, "_get_pool", _boom)
    # An unreadable ledger must look like "already halted, cap consumed,
    # cooldown just started" — never like "free to spend".
    assert hs.wallet_halted("antseed", "topup") is True
    spend = hs.wallet_op_spend_since("antseed", "topup", 0)
    assert spend["spent_usdc"] == float("inf") and spend["unreadable"] is True
    # ...and the intent write reports failure so the caller aborts.
    assert hs.wallet_op_begin("antseed", "topup", amount_usdc=5.0) is None


def test_provider_attempt_counts_distinguishes_wedged_from_idle(store):
    from conftest import seed_route_obs
    assert store.provider_attempt_counts("antseed") == {
        "ok": 0, "failed": 0, "total": 0}, "idle: no attempts at all"
    seed_route_obs("antseed", "m", "peerA", ok=False, n=5)
    assert store.provider_attempt_counts("antseed") == {
        "ok": 0, "failed": 5, "total": 5}, "wedged: attempts, zero successes"
    seed_route_obs("antseed", "m", "peerA", ok=True, n=1)
    assert store.provider_attempt_counts("antseed")["ok"] == 1
    # scoped per provider, and windowed
    assert store.provider_attempt_counts("openrouter")["total"] == 0
    old = int(time.time() * 1000) - 7200 * 1000
    seed_route_obs("stale_p", "m", "peerA", ok=False, n=3, ts=old)
    assert store.provider_attempt_counts("stale_p", window_ms=3_600_000)["total"] == 0


def test_provider_recent_ok_returns_newest_first(store):
    from conftest import seed_route_obs
    now = int(time.time() * 1000)
    seed_route_obs("antseed", "m", "peerA", ok=True, n=1, ts=now - 3000)
    seed_route_obs("antseed", "m", "peerA", ok=False, n=2, ts=now - 1000)
    assert store.provider_recent_ok("antseed", limit=3) == [False, False, True]
    assert store.provider_recent_ok("antseed", limit=1) == [False]
