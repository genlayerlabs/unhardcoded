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

import json
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

CREATE TABLE IF NOT EXISTS settings_overrides (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,   -- JSON-encoded knob value
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS provider_overlays (
    provider_id TEXT PRIMARY KEY,
    entry       TEXT NOT NULL,   -- JSON provider definition (never a key)
    added_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS consumer_keys (
    consumer   TEXT PRIMARY KEY,
    record     TEXT NOT NULL,   -- JSON consumer record (status/routes/limits/key hashes)
    updated_at INTEGER NOT NULL
);
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


# ---- settings_overrides (operator knob overrides; replaces overrides.json) -----

def get_overrides() -> dict[str, Any]:
    """All operator knob overrides, key -> decoded value. Empty (defaults win)
    on any error."""
    try:
        with _lock:
            c = _connect()
            cur = c.execute("SELECT key, value FROM settings_overrides")
            out: dict[str, Any] = {}
            for k, v in cur.fetchall():
                try:
                    out[k] = json.loads(v)
                except (TypeError, ValueError):
                    continue
            return out
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store get_overrides failed: %s", exc)
        return {}


def set_overrides(overrides: dict[str, Any]) -> None:
    """Replace the FULL override set (mirrors the old whole-file write). A key
    absent from `overrides` is cleared back to its default."""
    try:
        now = int(time.time())
        rows = [(k, json.dumps(v), now) for k, v in (overrides or {}).items()]
        with _lock:
            c = _connect()
            c.execute("DELETE FROM settings_overrides")
            c.executemany(
                "INSERT INTO settings_overrides(key, value, updated_at)"
                " VALUES (?,?,?)", rows)
            c.commit()
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store set_overrides failed: %s", exc)


# ---- provider_overlays (operator-added providers; replaces providers.local.json) --

def get_provider_overlays() -> dict[str, Any]:
    """The operator-added provider overlay as {'providers': {pid: entry}}. Empty
    on any error (no overlay -> only config.live.lua providers)."""
    try:
        with _lock:
            c = _connect()
            cur = c.execute("SELECT provider_id, entry FROM provider_overlays")
            providers: dict[str, Any] = {}
            for pid, entry in cur.fetchall():
                try:
                    providers[pid] = json.loads(entry)
                except (TypeError, ValueError):
                    continue
            return {"providers": providers}
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store get_provider_overlays failed: %s", exc)
        return {"providers": {}}


def set_provider_overlays(providers: dict[str, Any]) -> None:
    """Replace the FULL overlay set (mirrors the old whole-file write)."""
    try:
        now = int(time.time())
        rows = [(pid, json.dumps(entry),
                 int((entry or {}).get("added_at") or now))
                for pid, entry in (providers or {}).items()]
        with _lock:
            c = _connect()
            c.execute("DELETE FROM provider_overlays")
            c.executemany(
                "INSERT INTO provider_overlays(provider_id, entry, added_at)"
                " VALUES (?,?,?)", rows)
            c.commit()
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store set_provider_overlays failed: %s", exc)


# ---- consumer_keys (dashboard-issued consumer records; replaces the JSON) ------

def get_consumer_keys() -> "tuple[dict[str, Any], bool]":
    """(records, ok). `records` is {consumer: record}; `ok` is False ONLY on a
    real store error (so the caller can avoid clobbering good data with empty —
    an empty table is ok=True). Mirrors the old load-failed flag."""
    try:
        with _lock:
            c = _connect()
            cur = c.execute("SELECT consumer, record FROM consumer_keys")
            out: dict[str, Any] = {}
            for consumer, rec in cur.fetchall():
                try:
                    out[consumer] = json.loads(rec)
                except (TypeError, ValueError):
                    continue
            return out, True
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store get_consumer_keys failed: %s", exc)
        return {}, False


def set_consumer_keys(records: dict[str, Any]) -> None:
    """Replace the FULL set of consumer records (mirrors the old whole-file write)."""
    try:
        now = int(time.time())
        rows = [(consumer, json.dumps(rec), now)
                for consumer, rec in (records or {}).items()]
        with _lock:
            c = _connect()
            c.execute("DELETE FROM consumer_keys")
            c.executemany(
                "INSERT INTO consumer_keys(consumer, record, updated_at)"
                " VALUES (?,?,?)", rows)
            c.commit()
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store set_consumer_keys failed: %s", exc)


# ---- one-shot backfill of legacy JSON state (run once at startup) --------------

def _seed_if_empty(table: str, count_sql: str, legacy_path: str, to_rows) -> None:
    """One-time migration: if `table` is EMPTY and the legacy JSON at
    `legacy_path` exists, load it and seed via the table's setter. Idempotent —
    guarded on EMPTY (not file-exists), so it never clobbers data written after
    migration and is a no-op on every boot thereafter. Fail-soft: an absent or
    corrupt file leaves the table empty and logs (a corrupt legacy file therefore
    migrates to an empty table, not a failure)."""
    try:
        with _lock:
            c = _connect()
            if c.execute(count_sql).fetchone()[0] > 0:
                return                       # already migrated (or written since)
        p = Path(legacy_path)
        if not p.exists():
            return                           # fresh env, nothing to migrate
        data = json.loads(p.read_text())
        to_rows(data)                        # the existing set_* (its own lock)
        _log.info("host_store: seeded %s from %s", table, legacy_path)
    except Exception as exc:                 # noqa: BLE001
        _log.warning("host_store seed %s failed: %s", table, exc)


def migrate_legacy_json() -> None:
    """One-shot backfill of the legacy JSON operational state into the store,
    run once at startup BEFORE the app serves (and before settings.reload picks
    up overrides). Idempotent (guard-on-empty) so it is a safe no-op on every
    later boot. Once /x/calls and the dashboard confirm the data, delete the
    legacy files — then this function and the env paths below can be removed."""
    _seed_if_empty(
        "settings_overrides", "SELECT count(*) FROM settings_overrides",
        os.getenv("LLM_ROUTER_CONFIG_OVERRIDES",
                  "/run/llm-router/secrets/config-overrides.json"),
        lambda d: set_overrides(d if isinstance(d, dict) else {}))
    _seed_if_empty(
        "provider_overlays", "SELECT count(*) FROM provider_overlays",
        os.getenv("PROVIDERS_OVERLAY_PATH",
                  "/run/llm-router/secrets/providers.local.json"),
        lambda d: set_provider_overlays((d or {}).get("providers") or {}))
    _seed_if_empty(
        "consumer_keys", "SELECT count(*) FROM consumer_keys",
        os.getenv("DASHBOARD_ISSUED_KEYS_PATH",
                  "/run/llm-router/secrets/issued-consumer-keys.json"),
        lambda d: set_consumer_keys(d if isinstance(d, dict) else {}))


def reset() -> None:
    """Test hook: close + forget the connection (a fresh path/file next use)."""
    global _conn, _inserts_since_prune
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None
        _inserts_since_prune = 0
