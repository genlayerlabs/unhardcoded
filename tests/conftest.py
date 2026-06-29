"""Test config for the operational store.

The host store is Postgres now, so the tests that touch it run against a real
Postgres (the compose `postgres` service, or any DATABASE_URL). For isolation
against the shared DB each store-using test TRUNCATEs first. Pure-logic tests
(translation, ranking math, validation) don't use the store fixture and need no
Postgres.

Dev convenience: if DATABASE_URL is unset we default to a local throwaway
Postgres on :55432 (e.g. `docker run -p 55432:5432 -e POSTGRES_PASSWORD=test
postgres:16`); CI/compose sets DATABASE_URL to the compose service.
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

import host_store  # noqa: E402


@pytest.fixture
def host_store_clean():
    """Truncate the operational store before the test (isolation against the
    shared Postgres). Skips the test if Postgres is unreachable."""
    try:
        host_store.reset()
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
