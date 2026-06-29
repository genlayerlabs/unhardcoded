"""Host operational store (Postgres via psycopg3) — the call ledger + operator state.

The host's mutable operational state (operator knob overrides, the provider
overlay, dashboard-issued consumer keys, the call/usage ledger) is migrated off
the scattered JSON/JSONL files + in-process dicts into a single transactional
store.

Postgres (not SQLite) because the deployment runs the router and the ingress as
SEPARATE containers (router read-only on secrets, ingress read-write) sharing
state: a network DB both reach over TCP — no shared-file, no mount-asymmetry, no
WAL-reader-needs-RW gymnastics, and multi-pod ready. Prod = RDS; dev = a compose
`postgres` service. Connection from `DATABASE_URL`.

`calls` is the FACT TABLE: one raw row per LLM call, the source of truth from
which per-route / per-session views (reliability, latency, ttft, cost, session
totals) are DERIVED by query — not by host-side folding (combining/scoring is a
policy's job; the host stores raw and exposes raw).

Two write contracts:
- The ledger (`calls`) is best-effort TELEMETRY: fail-soft, offloaded off the
  request latency path via a bounded background queue.
- OPERATOR STATE (settings_overrides / provider_overlays / consumer_keys) is
  durable: the setters return a success bool so callers can surface a persistence
  failure — a silently-failed key revocation would be a security hole.

Schema is created idempotently under a Postgres advisory lock (race-safe across
the two containers starting at once). Retention on the ledger is bounded by
`DELETE WHERE ts < cutoff` (the discipline that fixed the usage-history OOMKilled
crashloop).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

_log = logging.getLogger("unhardcoded.host_store")

# A fixed 64-bit advisory-lock key for schema init (arbitrary constant).
_SCHEMA_LOCK_KEY = 0x686F7374_73746F72

_PRUNE_EVERY = 500          # run a retention sweep once per this many inserts
_WRITE_QUEUE_MAX = 10_000   # cap the background ledger-write backlog


def _retention_days() -> int:
    """Validated retention window — a non-integer or < 1 env value must not crash
    import or compute a future cutoff that prunes the whole ledger."""
    raw = os.getenv("ROUTER_DB_RETENTION_DAYS", "30")
    try:
        d = int(raw)
    except (TypeError, ValueError):
        _log.warning("invalid ROUTER_DB_RETENTION_DAYS=%r; using 30", raw)
        return 30
    if d < 1:
        _log.warning("ROUTER_DB_RETENTION_DAYS=%r must be >= 1; using 30", raw)
        return 30
    return d


_RETENTION_DAYS = _retention_days()

_SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS calls (
        id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        ts              BIGINT NOT NULL,
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
        latency_ms      DOUBLE PRECISION,
        tokens_in       BIGINT,
        tokens_out      BIGINT,
        tokens_total    BIGINT,
        tokens_cached   BIGINT,
        cost_usd        DOUBLE PRECISION,
        served_by       TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_calls_ts       ON calls(ts)",
    "CREATE INDEX IF NOT EXISTS idx_calls_route    ON calls(route_key, ts)",
    "CREATE INDEX IF NOT EXISTS idx_calls_session  ON calls(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_calls_consumer ON calls(consumer_sha, ts)",
    # Evolve the existing `calls` fact table in place — CREATE TABLE IF NOT EXISTS
    # never alters a table that already exists. Idempotent both ways: a no-op on a
    # fresh DB (the CREATE above already has these columns), the actual migration
    # on a DB that predates them. #3: the executed route identity + cache tokens.
    "ALTER TABLE calls ADD COLUMN IF NOT EXISTS tokens_cached BIGINT",
    "ALTER TABLE calls ADD COLUMN IF NOT EXISTS served_by TEXT",
    # Per-ATTEMPT route observations (one row per provider call the engine made,
    # including failed fallback tries — a grain `calls` does NOT have: `calls` is
    # per-REQUEST, final route only). The RAW from which reliability/latency are
    # derived on the fly (route_stats), replacing the in-process EMA dicts. Written
    # by the fold site in llm_router_host; route_key identity stays host-internal
    # (provider|family|served_by), never enters the signature.
    """CREATE TABLE IF NOT EXISTS route_observations (
        id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        ts            BIGINT NOT NULL,
        provider_id   TEXT,
        model_family  TEXT,
        served_by     TEXT,
        ok            BOOLEAN NOT NULL,
        latency_ms    DOUBLE PRECISION
    )""",
    "CREATE INDEX IF NOT EXISTS idx_route_obs_ts ON route_observations(ts)",
    "CREATE INDEX IF NOT EXISTS idx_route_obs_route"
    " ON route_observations(provider_id, model_family, served_by, ts)",
    """CREATE TABLE IF NOT EXISTS settings_overrides (
        key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at BIGINT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS provider_overlays (
        provider_id TEXT PRIMARY KEY, entry TEXT NOT NULL, added_at BIGINT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS consumer_keys (
        consumer TEXT PRIMARY KEY, record TEXT NOT NULL, updated_at BIGINT NOT NULL
    )""",
    # The antseed marketplace book. One RAW row per (peer, advertised service) —
    # the seller's announced prices/caps/reputation, stored as columns, not
    # interpreted: ranking/admission stays in sources/antseed.offers_sync (host)
    # and the Σ_pol policy (core). WRITTEN by the antseed sidecar (the only
    # producer; it runs the `antseed network browse` CLI), READ by
    # sources/antseed._load_market. `observed_at` is OUR browse stamp (epoch ms),
    # distinct from the network's `last_seen`; the 15-min sliding window that
    # merge-market.js used to union by hand is now just a read-time filter on it.
    """CREATE TABLE IF NOT EXISTS peer_offers (
        peer_id         TEXT NOT NULL,
        service         TEXT NOT NULL,
        price_in        DOUBLE PRECISION,
        price_out       DOUBLE PRECISION,
        price_cached_in DOUBLE PRECISION,
        max_concurrency INTEGER,
        reputation      DOUBLE PRECISION,
        last_seen       BIGINT,
        observed_at     BIGINT NOT NULL,
        first_seen      BIGINT,
        fetched_at      BIGINT,
        PRIMARY KEY (peer_id, service)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_peer_offers_observed ON peer_offers(observed_at)",
    # The antseed buyer's status (escrow + session pin + wallet), one row per
    # buyer pid. WRITTEN by the antseed sidecar (write-status.js on the poll loop
    # + control.js after a wallet op), READ by sources/antseed (_pinned_peer +
    # balances). Raw buyer-reported fields as columns; the deposits stay TEXT (the
    # buyer reports them as strings) and are coerced on read, as the JSON was.
    """CREATE TABLE IF NOT EXISTS buyer_status (
        pid                TEXT PRIMARY KEY,
        pinned_peer_id     TEXT,
        deposits_available TEXT,
        deposits_reserved  TEXT,
        wallet_address     TEXT,
        connection_state   TEXT,
        fetched_at         BIGINT
    )""",
]

_pool_lock = threading.Lock()
_pool: "ConnectionPool | None" = None
_prune_lock = threading.Lock()
_inserts_since_prune = 0


def _dsn() -> str:
    return os.getenv("DATABASE_URL", "postgresql://localhost/hoststore")


def _get_pool() -> ConnectionPool:
    """The process-wide connection pool, created (and schema-applied) on first
    use. The pool is thread-safe, so operations need no extra locking."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                p = ConnectionPool(_dsn(), min_size=1, max_size=8, open=True)
                _init_schema(p)
                _pool = p
    return _pool


def _init_schema(pool: ConnectionPool) -> None:
    with pool.connection() as conn:
        # Serialize concurrent schema init across containers/threads; the xact
        # lock auto-releases when this transaction commits at block exit.
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK_KEY,))
        for stmt in _SCHEMA_STATEMENTS:
            conn.execute(stmt)


def _route_key(provider: "str | None", family: "str | None",
               served: "str | None") -> "str | None":
    """The denormalized route identity for indexing/derivation. provider|family|
    served (peer granularity is not in the usage row yet — a later concern)."""
    if not provider and not family:
        return None
    return f"{provider or ''}|{family or ''}|{served or ''}"


# ---- calls ledger (best-effort telemetry) --------------------------------------

def insert_call(row: dict[str, Any]) -> None:
    """Record one call into the ledger from a usage-history-shaped row. Fail-soft:
    never raises into the request path. Best-effort telemetry."""
    global _inserts_since_prune
    try:
        provider = row.get("provider")
        family = row.get("model_family")
        served = row.get("served_model_id")
        sha = row.get("key_sha256")
        values = (
            int(row.get("ts") or time.time()),
            row.get("usage_event_id"), row.get("session"),
            sha if isinstance(sha, str) else None,
            row.get("caller"), _route_key(provider, family, served),
            provider, family, served, row.get("requested_model"),
            int(row["status"]) if str(row.get("status") or "").isdigit() else None,
            row.get("error_type"),
            float(row["latency_ms"]) if row.get("latency_ms") is not None else None,
            row.get("tokens_in"), row.get("tokens_out"), row.get("tokens_total"),
            row.get("tokens_cached"),
            float(row["cost_usd"]) if row.get("cost_usd") is not None else None,
            # which route actually served the call (marketplace peer or provider),
            # stamped by the engine on `chosen` — the per-route identity #4 derives
            # route stats from. Raw here; combining into a route key is a later step.
            row.get("served_by"),
        )
        with _get_pool().connection() as conn:   # one transaction, auto commit/rollback
            conn.execute(
                "INSERT INTO calls (ts, usage_event_id, session_id, consumer_sha,"
                " caller, route_key, provider_id, model_family, served_model_id,"
                " requested_model, status, error_type, latency_ms, tokens_in,"
                " tokens_out, tokens_total, tokens_cached, cost_usd, served_by)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", values)
        with _prune_lock:
            _inserts_since_prune += 1
            due = _inserts_since_prune >= _PRUNE_EVERY
            if due:
                _inserts_since_prune = 0
        if due:
            _prune()
    except Exception as exc:  # noqa: BLE001 — the ledger must never break a request
        _log.warning("host_store insert_call failed: %s", exc)


def _prune() -> None:
    try:
        now = time.time()
        # calls.ts is in SECONDS; route_observations.ts is in MILLISECONDS.
        with _get_pool().connection() as conn:
            conn.execute("DELETE FROM calls WHERE ts < %s",
                         (int(now) - _RETENTION_DAYS * 86400,))
            conn.execute("DELETE FROM route_observations WHERE ts < %s",
                         (int(now * 1000) - _RETENTION_DAYS * 86400 * 1000,))
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store prune failed: %s", exc)


# A bounded background queue + single worker keeps every write OFF the request's
# latency path; the payload is snapshotted so a later caller mutation can't change
# what gets written, and the queue is capped so a slow DB cannot grow an unbounded
# backlog — best-effort telemetry, so a dropped row is acceptable. The queue holds
# thunks so it serves both the call ledger and the per-attempt route observations.
_write_q: "queue.Queue" = queue.Queue(maxsize=_WRITE_QUEUE_MAX)


def _writer_loop() -> None:
    while True:
        job = _write_q.get()
        try:
            job()
        except Exception as exc:  # noqa: BLE001 — a bad row must not kill the writer
            _log.warning("host_store background write failed: %s", exc)
        finally:
            _write_q.task_done()


threading.Thread(target=_writer_loop, name="host-store-writer", daemon=True).start()


def _enqueue(job) -> None:
    try:
        _write_q.put_nowait(job)
    except queue.Full:
        _log.warning("host_store: write queue full (%d); dropping row", _WRITE_QUEUE_MAX)
    except Exception as exc:  # noqa: BLE001 — never break a request
        _log.warning("host_store enqueue failed: %s", exc)


def insert_call_async(row: dict[str, Any]) -> None:
    """Record a call WITHOUT blocking the caller — enqueue a SNAPSHOT for the
    background writer. Drops the row (best-effort) if the queue is full."""
    snap = dict(row)
    _enqueue(lambda: insert_call(snap))


def observe_route_call_async(row: dict[str, Any]) -> None:
    """Record one per-ATTEMPT route observation (provider/family/served_by + ok +
    latency) off the latency path. The raw from which route_stats() derives
    reliability/latency on the fly — replaces the in-process EMA folds."""
    snap = dict(row)
    _enqueue(lambda: _insert_route_observation(snap))


def recent_calls(limit: int = 100) -> list[dict[str, Any]]:
    """The most recent calls, newest first (operator view / verification)."""
    try:
        with _get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM calls ORDER BY id DESC LIMIT %s",
                            (int(limit),))
                return cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store recent_calls failed: %s", exc)
        return []


def count() -> int:
    try:
        with _get_pool().connection() as conn:
            return int(conn.execute("SELECT count(*) FROM calls").fetchone()[0])
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store count failed: %s", exc)
        return 0


# ---- settings_overrides (durable operator state) -------------------------------

def get_overrides() -> dict[str, Any]:
    """All operator knob overrides, key -> decoded value. Empty (defaults win) on
    any error."""
    try:
        with _get_pool().connection() as conn:
            cur = conn.execute("SELECT key, value FROM settings_overrides")
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


def set_overrides(overrides: dict[str, Any]) -> bool:
    """Replace the FULL override set. Returns True on success, False on a
    persistence failure (so the caller can report it, not pretend it saved)."""
    try:
        now = int(time.time())
        rows = [(k, json.dumps(v), now) for k, v in (overrides or {}).items()]
        with _get_pool().connection() as conn:   # atomic: commit/rollback per block
            conn.execute("DELETE FROM settings_overrides")
            if rows:
                conn.cursor().executemany(
                    "INSERT INTO settings_overrides(key, value, updated_at)"
                    " VALUES (%s,%s,%s)", rows)
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store set_overrides failed: %s", exc)
        return False


# ---- provider_overlays (durable operator state) --------------------------------

def get_provider_overlays() -> dict[str, Any]:
    """The operator-added provider overlay as {'providers': {pid: entry}}."""
    try:
        with _get_pool().connection() as conn:
            cur = conn.execute("SELECT provider_id, entry FROM provider_overlays")
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


def set_provider_overlays(providers: dict[str, Any]) -> bool:
    """Replace the FULL overlay set. Returns True on success, False on failure."""
    try:
        now = int(time.time())
        rows = [(pid, json.dumps(entry), int((entry or {}).get("added_at") or now))
                for pid, entry in (providers or {}).items()]
        with _get_pool().connection() as conn:
            conn.execute("DELETE FROM provider_overlays")
            if rows:
                conn.cursor().executemany(
                    "INSERT INTO provider_overlays(provider_id, entry, added_at)"
                    " VALUES (%s,%s,%s)", rows)
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store set_provider_overlays failed: %s", exc)
        return False


# ---- consumer_keys (durable operator state, credentials) -----------------------

def get_consumer_keys() -> "tuple[dict[str, Any], bool]":
    """(records, ok). `ok` is False on a store error OR any UNREADABLE row — a
    corrupt credentials row must fail CLOSED (the caller treats not-ok as "do not
    trust"), never silently drop a consumer back to default metadata. An empty
    table is ok=True."""
    try:
        with _get_pool().connection() as conn:
            cur = conn.execute("SELECT consumer, record FROM consumer_keys")
            out: dict[str, Any] = {}
            for consumer, rec in cur.fetchall():
                try:
                    out[consumer] = json.loads(rec)
                except (TypeError, ValueError):
                    _log.warning("host_store: undecodable consumer_keys row %r;"
                                 " failing closed", consumer)
                    return {}, False
            return out, True
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store get_consumer_keys failed: %s", exc)
        return {}, False


def set_consumer_keys(records: dict[str, Any]) -> bool:
    """Replace the FULL set of consumer records. Returns True on success, False on
    a persistence failure — a swallowed failure here would let a key revocation be
    reported as saved while still working after restart."""
    try:
        now = int(time.time())
        rows = [(consumer, json.dumps(rec), now)
                for consumer, rec in (records or {}).items()]
        with _get_pool().connection() as conn:
            conn.execute("DELETE FROM consumer_keys")
            if rows:
                conn.cursor().executemany(
                    "INSERT INTO consumer_keys(consumer, record, updated_at)"
                    " VALUES (%s,%s,%s)", rows)
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store set_consumer_keys failed: %s", exc)
        return False


# ---- route_observations (per-attempt raw; reliability/latency derived on the fly)

def _insert_route_observation(row: dict[str, Any]) -> None:
    """Append one per-attempt route observation. Fail-soft; best-effort telemetry."""
    try:
        with _get_pool().connection() as conn:
            conn.execute(
                "INSERT INTO route_observations"
                " (ts, provider_id, model_family, served_by, ok, latency_ms)"
                " VALUES (%s,%s,%s,%s,%s,%s)",
                (int(row.get("ts") or time.time() * 1000),
                 row.get("provider_id"), row.get("model_family"), row.get("served_by"),
                 bool(row.get("ok")),
                 float(row["latency_ms"]) if row.get("latency_ms") is not None else None))
    except Exception as exc:  # noqa: BLE001 — the fold must never break a request
        _log.warning("host_store route observation insert failed: %s", exc)


def route_stats(window_ms: int = 900_000) -> dict[str, dict[str, Any]]:
    """Per-route reliability + latency, DERIVED on the fly from route_observations
    over the last `window_ms` — one aggregate query, not per-candidate. Returns
    {route_key: {success_rate, latency_ms, count}} where route_key =
    provider|family|served_by (host-internal, matches route_reliability.route_key).
    Latency averages successful calls only (an error's latency is noise). Fail-soft
    -> {} so the offer is left unstamped and the algebra falls back to its default."""
    try:
        cutoff = int(time.time() * 1000) - max(0, window_ms)
        with _get_pool().connection() as conn:
            cur = conn.execute(
                "SELECT provider_id, model_family, served_by,"
                " avg(CASE WHEN ok THEN 1.0 ELSE 0.0 END) AS success_rate,"
                " avg(latency_ms) FILTER (WHERE ok AND latency_ms IS NOT NULL) AS latency_ms,"
                " count(*) AS n"
                " FROM route_observations WHERE ts >= %s"
                " GROUP BY provider_id, model_family, served_by", (cutoff,))
            out: dict[str, dict[str, Any]] = {}
            for prov, fam, sby, sr, lat, n in cur.fetchall():
                key = f"{prov}|{fam}|{sby}"
                out[key] = {
                    "success_rate": float(sr) if sr is not None else None,
                    "latency_ms": int(round(lat)) if lat is not None else None,
                    "count": int(n)}
            return out
    except Exception as exc:  # noqa: BLE001 — measurement read is best-effort
        _log.warning("host_store route_stats failed: %s", exc)
        return {}


# ---- peer_offers (antseed marketplace book; written by the sidecar) ------------

# Columns surfaced to the reader, in the row shape sources/antseed expects. The
# window/housekeeping columns (observed_at/first_seen/fetched_at) stay internal.
_PEER_OFFER_FIELDS = ("peer_id", "service", "price_in", "price_out",
                      "price_cached_in", "max_concurrency", "reputation",
                      "last_seen")


def peer_offers(window_ms: int = 900_000) -> list[dict[str, Any]]:
    """The antseed market book: one raw row per (peer, service) observed within
    the last `window_ms` of OUR browsing (the sliding window, as a read-time
    filter on observed_at). Fail-soft: returns [] on any store error, so a flaky
    DB degrades to "no antseed candidates" exactly as a missing dump did."""
    try:
        cutoff = int(time.time() * 1000) - max(0, window_ms)
        cols = ", ".join(_PEER_OFFER_FIELDS)
        with _get_pool().connection() as conn:
            cur = conn.execute(
                f"SELECT {cols} FROM peer_offers WHERE observed_at >= %s", (cutoff,))
            return [dict(zip(_PEER_OFFER_FIELDS, r)) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001 — market read is best-effort
        _log.warning("host_store peer_offers failed: %s", exc)
        return []


# ---- buyer_status (antseed buyer escrow/pin/wallet; written by the sidecar) ----

_BUYER_STATUS_FIELDS = ("pid", "pinned_peer_id", "deposits_available",
                        "deposits_reserved", "wallet_address", "connection_state")


def buyer_status(pid: str) -> "dict[str, Any] | None":
    """The antseed buyer's latest status row (session pin + escrow + wallet) for
    `pid`, or None when absent / on a store error (degraded: no pin, no balance),
    exactly as a missing status file was. Fail-soft."""
    try:
        cols = ", ".join(_BUYER_STATUS_FIELDS)
        with _get_pool().connection() as conn:
            cur = conn.execute(
                f"SELECT {cols} FROM buyer_status WHERE pid = %s", (pid,))
            row = cur.fetchone()
            return dict(zip(_BUYER_STATUS_FIELDS, row)) if row else None
    except Exception as exc:  # noqa: BLE001 — status read is best-effort
        _log.warning("host_store buyer_status failed: %s", exc)
        return None


# ---- one-shot backfill of legacy JSON state (run once at startup) --------------

def _seed_if_empty(table: str, legacy_path: str, to_rows) -> None:
    """One-time migration: if `table` is EMPTY and the legacy JSON at `legacy_path`
    exists, load it and seed via the table's setter. Idempotent — guarded on EMPTY
    (not file-exists), under an advisory lock so concurrent container starts cannot
    race. Fail-soft: an absent or corrupt file leaves the table empty and logs."""
    try:
        with _get_pool().connection() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK_KEY,))
            n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            if n > 0:
                return                       # already migrated (or written since)
            import pathlib
            p = pathlib.Path(legacy_path)
            if not p.exists():
                return                       # fresh env, nothing to migrate
            data = json.loads(p.read_text())
            to_rows(data)
            _log.info("host_store: seeded %s from %s", table, legacy_path)
    except Exception as exc:                 # noqa: BLE001
        _log.warning("host_store seed %s failed: %s", table, exc)


def migrate_legacy_json() -> None:
    """One-shot backfill of the legacy JSON operational state, run once at startup
    BEFORE the app serves. Idempotent (guard-on-empty + advisory lock) so it is a
    safe no-op on every later boot and safe to run from either container. Once the
    dashboard confirms the data, delete the legacy files; then this and the env
    paths below can be removed."""
    _seed_if_empty(
        "settings_overrides",
        os.getenv("LLM_ROUTER_CONFIG_OVERRIDES",
                  "/run/llm-router/secrets/config-overrides.json"),
        lambda d: set_overrides(d if isinstance(d, dict) else {}))
    _seed_if_empty(
        "provider_overlays",
        os.getenv("PROVIDERS_OVERLAY_PATH",
                  "/run/llm-router/secrets/providers.local.json"),
        lambda d: set_provider_overlays((d or {}).get("providers") or {}))
    _seed_if_empty(
        "consumer_keys",
        os.getenv("DASHBOARD_ISSUED_KEYS_PATH",
                  "/run/llm-router/secrets/issued-consumer-keys.json"),
        lambda d: set_consumer_keys(d if isinstance(d, dict) else {}))


def reset() -> None:
    """Test hook: close + forget the pool (recreated against DATABASE_URL next
    use). For per-test isolation against a shared DB, truncate the tables."""
    global _pool, _inserts_since_prune
    # Drain pending async writes first: closing the pool while the background
    # writer holds a connection races and flakes test isolation.
    _write_q.join()
    with _pool_lock:
        if _pool is not None:
            _pool.close()
        _pool = None
    with _prune_lock:
        _inserts_since_prune = 0


def truncate_all_for_tests() -> None:
    """Test helper: wipe every table for isolation against a shared Postgres."""
    with _get_pool().connection() as conn:
        conn.execute("TRUNCATE calls, settings_overrides, provider_overlays,"
                     " consumer_keys, peer_offers, buyer_status, route_observations")
