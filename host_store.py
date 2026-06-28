"""Host operational store (SQLite) — the call ledger.

The host's mutable operational state is scattered across JSON/JSONL files and
in-process dicts (usage-history.jsonl, the route_* folds, _stats). This is the
first cut of a single, transactional, queryable store for it. WAL mode, stdlib
`sqlite3` (no new dependency, no compose service), self-contained on the data
plane's volume.

`calls` is the FACT TABLE: one raw row per LLM call. It is the source of truth
from which per-route and per-session views (reliability, latency, ttft, cost,
session totals, cache-hot) are DERIVED — by query, not by host-side folding
(combining/scoring is a policy's job; the host stores raw and exposes raw).

This first cut DUAL-WRITES alongside usage-history.jsonl (it does not replace any
reader yet), so the ledger fills and can be verified before any migration. The
write is fail-soft: the ledger is best-effort and must never break a request,
exactly like the usage-history append it runs beside.

Retention is bounded by `DELETE WHERE ts < cutoff` (the discipline that fixed the
usage-history OOMKilled crashloop), pruned periodically. In-process single
connection guarded by a lock (the data plane is single-pod today; fleet-scale is
a later, separate concern — a shared store with TTL/eviction).
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger("unhardcoded.host_store")

DEFAULT_DB_PATH = "/run/llm-router/host-store.db"
# Keep the ledger bounded in time; a high-traffic router would otherwise grow it
# without limit (the usage-history lesson). Tunable; 30 days by default.
_RETENTION_DAYS = int(os.getenv("ROUTER_DB_RETENTION_DAYS", "30"))
_PRUNE_EVERY = 500  # run a retention sweep once per this many inserts

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              INTEGER NOT NULL,
    usage_event_id  TEXT,
    session_id      TEXT,
    consumer_sha    TEXT,
    caller          TEXT,
    route_key       TEXT,
    provider_id     TEXT,
    model_family    TEXT,
    served_model_id TEXT,
    requested_model TEXT,
    status          INTEGER,
    error_type      TEXT,
    latency_ms      REAL,
    tokens_in       INTEGER,
    tokens_out      INTEGER,
    tokens_total    INTEGER,
    cost_usd        REAL
);
CREATE INDEX IF NOT EXISTS idx_calls_ts       ON calls(ts);
CREATE INDEX IF NOT EXISTS idx_calls_route    ON calls(route_key, ts);
CREATE INDEX IF NOT EXISTS idx_calls_session  ON calls(session_id);
CREATE INDEX IF NOT EXISTS idx_calls_consumer ON calls(consumer_sha, ts);
"""

_lock = threading.Lock()
_conn: "sqlite3.Connection | None" = None
_inserts_since_prune = 0


def _db_path() -> str:
    return os.getenv("ROUTER_DB_PATH", DEFAULT_DB_PATH)


def _connect() -> sqlite3.Connection:
    """The process-wide connection, created (and schema-applied) on first use.
    `check_same_thread=False` + the module lock makes it safe across FastAPI's
    threadpool; WAL keeps reads concurrent with the single writer."""
    global _conn
    if _conn is None:
        path = _db_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(path, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.row_factory = sqlite3.Row
        c.executescript(_SCHEMA)
        c.commit()
        _conn = c
    return _conn


def _route_key(provider: "str | None", family: "str | None",
               served: "str | None") -> "str | None":
    """The denormalized route identity for indexing/derivation. Mirrors the
    provider|family shape of route_reliability.route_key (peer granularity is not
    available in the usage row yet — a later antseed-economics concern)."""
    if not provider and not family:
        return None
    return f"{provider or ''}|{family or ''}|{served or ''}"


def insert_call(row: dict[str, Any]) -> None:
    """Record one call into the ledger from a usage-history-shaped row. Fail-soft:
    never raises into the request path. Best-effort, like the usage-history append
    it runs beside."""
    global _inserts_since_prune
    try:
        provider = row.get("provider")
        family = row.get("model_family")
        served = row.get("served_model_id")
        sha = row.get("key_sha256")
        values = (
            int(row.get("ts") or time.time()),
            row.get("usage_event_id"),
            row.get("session"),
            sha if isinstance(sha, str) else None,
            row.get("caller"),
            _route_key(provider, family, served),
            provider, family, served,
            row.get("requested_model"),
            int(row["status"]) if str(row.get("status") or "").isdigit() else None,
            row.get("error_type"),
            float(row["latency_ms"]) if row.get("latency_ms") is not None else None,
            row.get("tokens_in"), row.get("tokens_out"), row.get("tokens_total"),
            float(row["cost_usd"]) if row.get("cost_usd") is not None else None,
        )
        with _lock:
            c = _connect()
            c.execute(
                "INSERT INTO calls (ts, usage_event_id, session_id, consumer_sha,"
                " caller, route_key, provider_id, model_family, served_model_id,"
                " requested_model, status, error_type, latency_ms, tokens_in,"
                " tokens_out, tokens_total, cost_usd)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
            c.commit()
            _inserts_since_prune += 1
            if _inserts_since_prune >= _PRUNE_EVERY:
                _inserts_since_prune = 0
                _prune_locked(c)
    except Exception as exc:  # noqa: BLE001 — the ledger must never break a request
        _log.warning("host_store insert_call failed: %s", exc)


def _prune_locked(c: sqlite3.Connection) -> None:
    cutoff = int(time.time()) - _RETENTION_DAYS * 86400
    c.execute("DELETE FROM calls WHERE ts < ?", (cutoff,))
    c.commit()


def recent_calls(limit: int = 100) -> list[dict[str, Any]]:
    """The most recent calls, newest first (operator view / verification)."""
    try:
        with _lock:
            c = _connect()
            cur = c.execute(
                "SELECT * FROM calls ORDER BY id DESC LIMIT ?", (int(limit),))
            return [dict(r) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store recent_calls failed: %s", exc)
        return []


def count() -> int:
    try:
        with _lock:
            c = _connect()
            return int(c.execute("SELECT count(*) FROM calls").fetchone()[0])
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store count failed: %s", exc)
        return 0


def reset() -> None:
    """Test hook: close + forget the connection (a fresh path/file next use)."""
    global _conn, _inserts_since_prune
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None
        _inserts_since_prune = 0
