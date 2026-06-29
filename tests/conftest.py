"""Test config for the operational store.

The host store is Postgres now, so the tests that touch it run against a real
Postgres (the compose `postgres` service, or any DATABASE_URL). For isolation
against the shared DB each store-using test TRUNCATEs first. Pure-logic tests
(translation, ranking math, validation) don't use the store fixture and need no
Postgres.

Dev convenience: if DATABASE_URL is unset we default to a local throwaway
Postgres on :55432 (e.g. `docker run -p 55432:5432 -e POSTGRES_PASSWORD=test
postgres:16`); CI/compose sets DATABASE_URL to the compose service.

Point DATABASE_URL at a DEDICATED test database (e.g. `hoststore_test`), NOT the
live operational DB a running compose stack uses: the antseed sidecar writes
peer_offers into the live DB continuously, and those writes land between a test's
TRUNCATE and its read, flaking isolation. A separate db keeps the two apart.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "DATABASE_URL", "postgresql://postgres:test@localhost:55432/hoststore")
# Writes run inline (no background writer thread) so each test's truncate fully
# isolates it — the async queue races truncate/read across the shared test DB.
os.environ.setdefault("HOST_STORE_SYNC_WRITES", "1")
# usage_rows() floors "all" reads at the retention horizon (now - retention).
# Suite fixtures seed fixed historical timestamps, so keep retention effectively
# unbounded here; tests that exercise the floor/prune set _RETENTION_DAYS locally.
os.environ.setdefault("ROUTER_DB_RETENTION_DAYS", "3650000")

import host_store  # noqa: E402


@pytest.fixture
def host_store_clean():
    """Truncate the operational store before the test (isolation against the
    shared Postgres). Skips the test if Postgres is unreachable."""
    try:
        # Drain pending async writes from a prior test FIRST so they can't land
        # after this truncate; don't close the pool (the background writer shares
        # it — closing it mid-write races and flakes isolation).
        host_store._write_q.join()
        host_store.truncate_all_for_tests()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"host store Postgres unavailable: {exc}")
    yield host_store


def seed_peer_offers(peers, observed_at=None):
    """Seed the antseed market book (peer_offers) exactly as the sidecar's
    write-market.js does: flatten each peer's providerPricing→services into one
    raw row per (peer, service), stamped observed_at (defaults to now). Shared by
    the antseed source tests so the seeding mirrors the single real writer."""
    import time
    obs = int(time.time() * 1000) if observed_at is None else observed_at
    rows = []
    for peer in peers:
        if not peer.get("peerId"):
            continue
        maxc = peer.get("maxConcurrency")
        rep = peer.get("onChainReputationScore")
        last_seen = peer.get("lastSeen")
        for pricing in (peer.get("providerPricing") or {}).values():
            for service, sp in ((pricing or {}).get("services") or {}).items():
                rows.append((
                    peer["peerId"], service,
                    sp.get("inputUsdPerMillion"), sp.get("outputUsdPerMillion"),
                    sp.get("cachedInputUsdPerMillion"),
                    maxc, rep, last_seen, obs, obs, obs))
    with host_store._get_pool().connection() as conn:
        if rows:
            # UPSERT, mirroring the real writer (antseed/write-market.js): a peer
            # listing the same service twice in one browse collapses to one row.
            conn.cursor().executemany(
                "INSERT INTO peer_offers (peer_id, service, price_in, price_out,"
                " price_cached_in, max_concurrency, reputation, last_seen,"
                " observed_at, first_seen, fetched_at)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (peer_id, service) DO UPDATE SET"
                " price_in=EXCLUDED.price_in, price_out=EXCLUDED.price_out,"
                " price_cached_in=EXCLUDED.price_cached_in,"
                " max_concurrency=EXCLUDED.max_concurrency,"
                " reputation=EXCLUDED.reputation, last_seen=EXCLUDED.last_seen,"
                " observed_at=EXCLUDED.observed_at, fetched_at=EXCLUDED.fetched_at",
                rows)


def seed_call(session=None, provider=None, family=None, served_by=None, status=200,
              caller=None, tokens_in=0, tokens_out=0, tokens_cached=0, cost_usd=0.0,
              ts=None):
    """Insert one per-request `calls` row (as the ingress would), the raw the
    per-session views (hot_route / session_*) derive from. ts in SECONDS."""
    import time
    host_store.insert_call({
        "ts": int(time.time()) if ts is None else ts,
        "session": session, "provider": provider, "model_family": family,
        "served_by": served_by, "served_model_id": served_by, "status": status,
        "caller": caller, "key_sha256": None, "tokens_in": tokens_in,
        "tokens_out": tokens_out, "tokens_total": tokens_in + tokens_out,
        "tokens_cached": tokens_cached, "cost_usd": cost_usd})


def seed_route_obs(provider, family, served_by, ok, latency_ms=None, n=1, ts=None,
                   tools_requested=False, tool_calls_emitted=False):
    """Seed n per-attempt route_observations for a route (the raw from which
    route_stats / tool_incapable_routes derive on the fly)."""
    import time
    t = int(time.time() * 1000) if ts is None else ts
    rows = [(t, provider, family, served_by, ok, latency_ms,
             tools_requested, tool_calls_emitted) for _ in range(n)]
    with host_store._get_pool().connection() as conn:
        conn.cursor().executemany(
            "INSERT INTO route_observations"
            " (ts, provider_id, model_family, served_by, ok, latency_ms,"
            " tools_requested, tool_calls_emitted)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", rows)


def seed_buyer_status(pid, pinned_peer_id=None, deposits_available=None,
                      deposits_reserved=None, wallet_address=None,
                      connection_state=None):
    """Seed the antseed buyer status (pin + escrow + wallet) as the sidecar's
    write-status.js / control.js do — one row per buyer pid. Deposits are the raw
    buyer-reported strings."""
    import time
    with host_store._get_pool().connection() as conn:
        conn.execute(
            "INSERT INTO buyer_status (pid, pinned_peer_id, deposits_available,"
            " deposits_reserved, wallet_address, connection_state, fetched_at)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s)"
            " ON CONFLICT (pid) DO UPDATE SET"
            " pinned_peer_id=EXCLUDED.pinned_peer_id,"
            " deposits_available=EXCLUDED.deposits_available,"
            " deposits_reserved=EXCLUDED.deposits_reserved,"
            " wallet_address=EXCLUDED.wallet_address,"
            " connection_state=EXCLUDED.connection_state, fetched_at=EXCLUDED.fetched_at",
            (pid, pinned_peer_id, deposits_available, deposits_reserved,
             wallet_address, connection_state, int(time.time() * 1000)))
