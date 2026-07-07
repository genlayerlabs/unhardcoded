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
    # How cost_usd was determined ('reported' = the provider's own authoritative
    # cost = an INDEPENDENT signal; 'computed' = derived from the list price =
    # tautological; 'subscription' = $0). Lets the cost-accuracy panel flag only
    # rows with real signal instead of training the operator to ignore drift.
    "ALTER TABLE calls ADD COLUMN IF NOT EXISTS cost_basis TEXT",
    # Per-ATTEMPT route observations (one row per provider call the engine made,
    # including failed fallback tries — a grain `calls` does NOT have: `calls` is
    # per-REQUEST, final route only). The RAW from which reliability/latency are
    # derived on the fly (route_stats), replacing the in-process EMA dicts. Written
    # by the fold site in llm_router_host; route_key identity stays host-internal
    # (provider|family|served_by), never enters the signature.
    """CREATE TABLE IF NOT EXISTS route_observations (
        id                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        ts                 BIGINT NOT NULL,
        provider_id        TEXT,
        model_family       TEXT,
        served_by          TEXT,
        ok                 BOOLEAN NOT NULL,
        latency_ms         DOUBLE PRECISION,
        tools_requested    BOOLEAN,
        tool_calls_emitted BOOLEAN
    )""",
    "CREATE INDEX IF NOT EXISTS idx_route_obs_ts ON route_observations(ts)",
    "CREATE INDEX IF NOT EXISTS idx_route_obs_route"
    " ON route_observations(provider_id, model_family, served_by, ts)",
    # #4c: learned tool capability is derived from these per-attempt signals.
    "ALTER TABLE route_observations ADD COLUMN IF NOT EXISTS tools_requested BOOLEAN",
    "ALTER TABLE route_observations ADD COLUMN IF NOT EXISTS tool_calls_emitted BOOLEAN",
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
    # Dashboard login audit (#5) — replaces dashboard-logins.jsonl. A small record
    # read whole by the dashboard, so it follows the consumer_keys pattern: the
    # row as a JSON record in TEXT (not analysed by column). ts in SECONDS.
    """CREATE TABLE IF NOT EXISTS login_history (
        id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        ts     BIGINT NOT NULL,
        record TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_login_history_ts ON login_history(ts)",
    # Direct-provider list prices (openai/anthropic/google) scraped from each
    # provider's OFFICIAL pricing page by sources/official_pricing, one row per
    # (provider, curated family). The DURABLE source of truth: routing coasts on
    # this table when a scrape is stale or fails, so a redesigned pricing page
    # degrades coverage instead of zeroing a price to +inf. price_cached_in is the
    # cache-READ (hit) price — kept for effective-cost work, mirrors peer_offers.
    """CREATE TABLE IF NOT EXISTS provider_prices (
        provider_id     TEXT NOT NULL,
        model_family    TEXT NOT NULL,
        price_in        DOUBLE PRECISION,
        price_out       DOUBLE PRECISION,
        price_cached_in DOUBLE PRECISION,
        updated_at      BIGINT,
        PRIMARY KEY (provider_id, model_family)
    )""",
]

_pool_lock = threading.Lock()
_pool: "ConnectionPool | None" = None
_prune_lock = threading.Lock()
_inserts_since_prune = 0


def _dsn() -> str:
    return os.getenv("DATABASE_URL", "postgresql://localhost/hoststore")


def _pool_timeout() -> float:
    raw = os.getenv("HOST_STORE_POOL_TIMEOUT", "30")
    try:
        return max(0.1, float(raw))
    except (TypeError, ValueError):
        return 30.0


def _get_pool() -> ConnectionPool:
    """The process-wide connection pool, created (and schema-applied) on first
    use. The pool is thread-safe, so operations need no extra locking."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                p = ConnectionPool(
                    _dsn(),
                    min_size=1,
                    max_size=8,
                    open=True,
                    timeout=_pool_timeout(),
                )
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
            row.get("cost_basis"),   # how cost_usd was determined (reported/computed/…)
        )
        with _get_pool().connection() as conn:   # one transaction, auto commit/rollback
            conn.execute(
                "INSERT INTO calls (ts, usage_event_id, session_id, consumer_sha,"
                " caller, route_key, provider_id, model_family, served_model_id,"
                " requested_model, status, error_type, latency_ms, tokens_in,"
                " tokens_out, tokens_total, tokens_cached, cost_usd, served_by,"
                " cost_basis)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", values)
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
            conn.execute("DELETE FROM login_history WHERE ts < %s",
                         (int(now) - _RETENTION_DAYS * 86400,))
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


# Tests set HOST_STORE_SYNC_WRITES=1 so every write runs INLINE (no background
# thread): the daemon writer otherwise interleaves its commits with the
# truncate/read of the next test on the shared DB at non-deterministic times,
# flaking isolation. Async buys nothing when the test asserts on the row right
# after writing it. Production leaves it unset and keeps the off-latency-path queue.
_SYNC_WRITES = bool(os.getenv("HOST_STORE_SYNC_WRITES"))


def _enqueue(job) -> None:
    if _SYNC_WRITES:
        try:
            job()
        except Exception as exc:  # noqa: BLE001 — mirror the writer loop's tolerance
            _log.warning("host_store sync write failed: %s", exc)
        return
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


def recent_calls(limit: int = 100, caller: "str | None" = None) -> list[dict[str, Any]]:
    """The most recent calls, newest first (operator view / verification).
    Optionally scoped to one caller (control-plane activity feed)."""
    try:
        where, params = ("", []) if caller is None else (" WHERE caller = %s", [caller])
        with _get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f"SELECT * FROM calls{where} ORDER BY id DESC LIMIT %s",
                            params + [int(limit)])
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
                " (ts, provider_id, model_family, served_by, ok, latency_ms,"
                " tools_requested, tool_calls_emitted)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (int(row.get("ts") or time.time() * 1000),
                 row.get("provider_id"), row.get("model_family"), row.get("served_by"),
                 bool(row.get("ok")),
                 float(row["latency_ms"]) if row.get("latency_ms") is not None else None,
                 bool(row.get("tools_requested")), bool(row.get("tool_calls_emitted"))))
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


def tool_incapable_routes(window_ms: int = 1_800_000, min_samples: int = 20) -> "set[str]":
    """Routes that have proven they ignore tools — derived (#4c) from
    route_observations: >= min_samples tools-requests within the last window_ms and
    ZERO tool_calls emitted. The window IS the re-test horizon (a route ages out if
    it stops being tool-tested); ANY tool_call in the window clears it. Capable is
    the default (a route not in this set), so offers_sync drops supports_tools only
    for the proven-incapable. Fail-soft -> empty set (everyone stays capable)."""
    try:
        cutoff = int(time.time() * 1000) - max(0, window_ms)
        with _get_pool().connection() as conn:
            cur = conn.execute(
                "SELECT provider_id, model_family, served_by FROM route_observations"
                " WHERE ts >= %s AND tools_requested"
                " GROUP BY provider_id, model_family, served_by"
                " HAVING count(*) >= %s AND"
                " coalesce(sum(CASE WHEN tool_calls_emitted THEN 1 ELSE 0 END), 0) = 0",
                (cutoff, min_samples))
            return {f"{p}|{f}|{s}" for p, f, s in cur.fetchall()}
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store tool_incapable_routes failed: %s", exc)
        return set()


# ---- per-session views, derived on the fly from `calls` (#4b) -------------------
# `calls` is per-request and carries session_id / status / provider_id /
# model_family / served_by / tokens / cost / caller — everything the in-process
# cache-affinity + session-meter folds held, now derived (fleet-consistent).

_SESSION_TOTALS_ZERO = {"calls": 0, "tokens_in": 0, "tokens_out": 0,
                        "tokens_cached": 0, "cost_usd": 0.0}


def hot_route(session: "str | None") -> "str | None":
    """The route (provider|family|served_by) that most recently served this
    session SUCCESSFULLY — its prompt-cache prefix is hot there. None when unknown.
    Matches route_reliability.route_key; resolved per request into the cache_hot
    field. Fail-soft."""
    if not session:
        return None
    try:
        with _get_pool().connection() as conn:
            row = conn.execute(
                "SELECT provider_id, model_family, served_by FROM calls"
                " WHERE session_id = %s AND status < 400 AND served_by IS NOT NULL"
                " ORDER BY ts DESC, id DESC LIMIT 1", (session,)).fetchone()
            return f"{row[0]}|{row[1]}|{row[2]}" if row else None
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store hot_route failed: %s", exc)
        return None


def session_totals(session: "str | None") -> dict[str, Any]:
    """The session's running totals (calls, tokens_in/out/cached, cost_usd) summed
    over its committed `calls`. The caller adds the in-flight call on top. Fail-soft
    -> zeros."""
    if not session:
        return dict(_SESSION_TOTALS_ZERO)
    try:
        with _get_pool().connection() as conn:
            r = conn.execute(
                "SELECT count(*), coalesce(sum(tokens_in),0), coalesce(sum(tokens_out),0),"
                " coalesce(sum(tokens_cached),0), coalesce(sum(cost_usd),0)"
                " FROM calls WHERE session_id = %s", (session,)).fetchone()
            return {"calls": int(r[0]), "tokens_in": int(r[1]), "tokens_out": int(r[2]),
                    "tokens_cached": int(r[3]), "cost_usd": round(float(r[4]), 6)}
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store session_totals failed: %s", exc)
        return dict(_SESSION_TOTALS_ZERO)


def session_warm(session: "str | None") -> list[dict[str, Any]]:
    """The session's warm routes for DISPLAY: per family, the most recent route
    that served it successfully ({family, provider, served_by}). Fail-soft -> []."""
    if not session:
        return []
    try:
        with _get_pool().connection() as conn:
            cur = conn.execute(
                "SELECT DISTINCT ON (model_family) model_family, provider_id, served_by"
                " FROM calls WHERE session_id = %s AND status < 400"
                " AND model_family IS NOT NULL"
                " ORDER BY model_family, ts DESC, id DESC", (session,))
            return [{"family": f, "provider": p, "served_by": s or p}
                    for f, p, s in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store session_warm failed: %s", exc)
        return []


def session_owner(session: "str | None") -> "str | None":
    """The consumer that owns the session = the caller of its EARLIEST call
    (first-writer-wins, for cross-consumer isolation). None when unknown."""
    if not session:
        return None
    try:
        with _get_pool().connection() as conn:
            row = conn.execute(
                "SELECT caller FROM calls WHERE session_id = %s"
                " ORDER BY ts ASC, id ASC LIMIT 1", (session,)).fetchone()
            return row[0] if row else None
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store session_owner failed: %s", exc)
        return None


def all_session_totals() -> dict[str, dict[str, Any]]:
    """Every session's totals (operator /x/sessions view), derived from calls."""
    try:
        with _get_pool().connection() as conn:
            cur = conn.execute(
                "SELECT session_id, count(*), coalesce(sum(tokens_in),0),"
                " coalesce(sum(tokens_out),0), coalesce(sum(tokens_cached),0),"
                " coalesce(sum(cost_usd),0) FROM calls"
                " WHERE session_id IS NOT NULL GROUP BY session_id")
            return {s: {"calls": int(c), "tokens_in": int(ti), "tokens_out": int(to),
                        "tokens_cached": int(tc), "cost_usd": round(float(cu), 6)}
                    for s, c, ti, to, tc, cu in cur.fetchall()}
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store all_session_totals failed: %s", exc)
        return {}


def cost_by_route(window_s: int = 86_400) -> list[dict[str, Any]]:
    """Per (provider, family) ledger aggregate over the last `window_s` seconds —
    the RAW from which measured effective cost is DERIVED by query (#41), for the
    dashboard's cost-accuracy panel. Fail-soft -> []."""
    try:
        floor = int(time.time()) - max(0, window_s)
        with _get_pool().connection() as conn:
            cur = conn.execute(
                "SELECT provider_id, model_family, count(*),"
                " coalesce(sum(tokens_in),0), coalesce(sum(tokens_out),0),"
                " coalesce(sum(tokens_cached),0), coalesce(sum(cost_usd),0),"
                " coalesce(sum((cost_basis = 'reported')::int),0)"
                " FROM calls WHERE ts >= %s AND cost_usd IS NOT NULL"
                " AND provider_id IS NOT NULL AND model_family IS NOT NULL"
                " GROUP BY provider_id, model_family", (floor,))
            return [{"provider": p, "family": f, "calls": int(c),
                     "tokens_in": int(ti), "tokens_out": int(to),
                     "tokens_cached": int(tc), "cost_usd": round(float(cu), 6),
                     "n_reported": int(nr)}
                    for p, f, c, ti, to, tc, cu, nr in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001 — panel is best-effort
        _log.warning("host_store cost_by_route failed: %s", exc)
        return []


# ---- usage rows + dashboard logins, from the store (#5) -------------------------

_USAGE_COLS = ("ts", "caller", "provider_id", "model_family", "served_model_id",
               "served_by", "status", "error_type", "tokens_in", "tokens_out",
               "tokens_total", "tokens_cached", "cost_usd", "requested_model",
               "consumer_sha", "usage_event_id")


def _usage_where(since_ts: "int | None" = None, caller: "str | None" = None,
                 caller_is_null: bool = False, consumer_sha: "str | None" = None,
                 provider: "str | None" = None,
                 model_family: "str | None" = None) -> "tuple[str, list[Any]]":
    """The shared WHERE for every calls-derived usage read. "all" (since_ts=None)
    is still bounded to the retention horizon so the read is ALWAYS time-bounded
    (never a bare table scan): rows older than retention are pruned anyway, so the
    floor only drops not-yet-pruned slack. A caller-supplied since_ts narrows
    further (it can't widen past what we retain). The optional filters mirror the
    dashboard's Python row filters EXACTLY (raw-column equality; `caller_is_null`
    covers the key-filter-without-known-consumer case, where the old Python
    compared `row.get("caller") == None`)."""
    floor = int(time.time()) - _RETENTION_DAYS * 86400
    if since_ts is not None:
        floor = max(int(since_ts), floor)
    clauses, params = ["ts >= %s"], [floor]
    if caller is not None:
        clauses.append("caller = %s"); params.append(caller)
    elif caller_is_null:
        clauses.append("caller IS NULL")
    if consumer_sha is not None:
        clauses.append("consumer_sha = %s"); params.append(consumer_sha)
    if provider is not None:
        clauses.append("provider_id = %s"); params.append(provider)
    if model_family is not None:
        clauses.append("model_family = %s"); params.append(model_family)
    return " WHERE " + " AND ".join(clauses), params


def _map_usage_row(r: tuple) -> dict[str, Any]:
    """Map a `calls` row to the usage-history row keys the aggregation expects."""
    d = dict(zip(_USAGE_COLS, r))
    return {
        "event": "request", "ts": d["ts"], "caller": d["caller"],
        "provider": d["provider_id"], "model_family": d["model_family"],
        "served_model_id": d["served_model_id"], "served_by": d["served_by"],
        "status": d["status"], "error_type": d["error_type"],
        "tokens_in": d["tokens_in"], "tokens_out": d["tokens_out"],
        "tokens_total": d["tokens_total"], "tokens_cached": d["tokens_cached"],
        "cost_usd": d["cost_usd"], "requested_model": d["requested_model"],
        "key_sha256": d["consumer_sha"],
        "key_sha256_prefix": (d["consumer_sha"] or "")[:12] or None,
        "usage_event_id": d.get("usage_event_id")}


def usage_rows(since_ts: "int | None" = None,
               caller: "str | None" = None) -> list[dict[str, Any]]:
    """Calls in the legacy usage-history ROW SHAPE (#5: replaces usage-history.jsonl),
    so the dashboard timeframe aggregation consumes them unchanged. Optionally
    filtered by ts >= since_ts and/or caller. ts in SECONDS. Fail-soft -> []."""
    try:
        where, params = _usage_where(since_ts=since_ts, caller=caller)
        with _get_pool().connection() as conn:
            cur = conn.execute(
                f"SELECT {', '.join(_USAGE_COLS)} FROM calls{where} ORDER BY ts",
                params)
            return [_map_usage_row(r) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store usage_rows failed: %s", exc)
        return []


def usage_rows_page(since_ts: "int | None" = None, caller: "str | None" = None,
                    caller_is_null: bool = False, consumer_sha: "str | None" = None,
                    provider: "str | None" = None, model_family: "str | None" = None,
                    limit: int = 200) -> list[dict[str, Any]]:
    """The newest `limit` usage rows in the window (dashboard Activity), newest
    first — a LIMIT'd query instead of loading the whole window to slice it.
    Tie order (ts DESC, id ASC) matches the old stable reverse sort over an
    ascending read. Fail-soft -> []."""
    try:
        where, params = _usage_where(since_ts=since_ts, caller=caller,
                                     caller_is_null=caller_is_null,
                                     consumer_sha=consumer_sha, provider=provider,
                                     model_family=model_family)
        with _get_pool().connection() as conn:
            cur = conn.execute(
                f"SELECT {', '.join(_USAGE_COLS)} FROM calls{where}"
                " ORDER BY ts DESC, id ASC LIMIT %s", params + [int(limit)])
            return [_map_usage_row(r) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store usage_rows_page failed: %s", exc)
        return []


# ---- dashboard analytics (SQL aggregation over `calls`) ------------------------
#
# The dashboard stats used to load EVERY retained row and aggregate in Python —
# O(all rows) per page load. These push the aggregation into Postgres (one
# GROUPING SETS scan over idx_calls_ts) while replicating the Python semantics of
# auth_proxy._aggregate_usage_rows / _period_totals EXACTLY:
#   * error        = COALESCE(status,0) >= 400
#   * bucket keys  = NULL or '' -> 'unknown' (caller/provider/family/route/served)
#   * by_status    = COALESCE(status,0) as text
#   * tokens_total = row tokens_total unless NULL/0, else tokens_in + tokens_out
#   * cost         = GREATEST(cost_usd, 0) when stamped, else NOT counted (NULL);
#                    `priced` counts stamped rows so callers can preserve the
#                    "cost_usd key present only if any row was priced" behavior
#   * day bucket   = UTC calendar date of ts (Python used fromtimestamp(tz=utc))
#   * rejects are never persisted to `calls`, so they don't appear here.

_AGG_INNER = (
    "SELECT id, ts, status, tokens_in, tokens_out, tokens_total, cost_usd,"
    " requested_model, model_family, provider_id,"
    " COALESCE(NULLIF(caller,''),'unknown') AS caller_k,"
    " COALESCE(NULLIF(provider_id,''),'unknown') AS provider_k,"
    " COALESCE(NULLIF(model_family,''),'unknown') AS family_k,"
    " COALESCE(NULLIF(requested_model,''),'unknown') AS route_k,"
    " COALESCE(NULLIF(served_model_id,''),'unknown') AS served_k,"
    " COALESCE(NULLIF(substr(consumer_sha,1,12),''),'unknown') AS prefix_k,"
    " COALESCE(status,0)::text AS status_k,"
    " to_char(to_timestamp(ts) AT TIME ZONE 'UTC','YYYY-MM-DD') AS day_k"
    " FROM calls"
)

_AGG_MEASURES = (
    " count(*) AS requests,"
    " count(*) FILTER (WHERE COALESCE(status,0) >= 400) AS errors,"
    " COALESCE(sum(COALESCE(tokens_in,0)),0) AS tokens_in,"
    " COALESCE(sum(COALESCE(tokens_out,0)),0) AS tokens_out,"
    " COALESCE(sum(CASE WHEN COALESCE(tokens_total,0) <> 0 THEN tokens_total"
    " ELSE COALESCE(tokens_in,0)+COALESCE(tokens_out,0) END),0) AS tokens_total,"
    " round(COALESCE(sum(GREATEST(cost_usd,0)),0)::numeric,6)::float8 AS cost_usd,"
    " count(cost_usd) AS priced,"
    " max(ts) AS last_seen"
)


def _agg_counter(row: tuple) -> dict[str, Any]:
    requests, errors, tin, tout, ttotal, cost, priced, last_seen = row
    return {"requests": int(requests), "errors": int(errors),
            "tokens_in": int(tin), "tokens_out": int(tout),
            "tokens_total": int(ttotal), "cost_usd": float(cost),
            "priced": int(priced),
            "last_seen": int(last_seen) if last_seen is not None else None}


def _empty_usage_aggregate() -> dict[str, Any]:
    return {"totals": {"requests": 0, "errors": 0, "tokens_in": 0, "tokens_out": 0,
                       "tokens_total": 0, "cost_usd": 0.0, "priced": 0,
                       "last_seen": None},
            "by_caller": {}, "by_provider": {}, "by_model_family": {},
            "by_route": {}, "by_served_model": {}, "by_status": {}, "by_day": {}}


def usage_aggregate(since_ts: "int | None" = None, caller: "str | None" = None,
                    caller_is_null: bool = False, consumer_sha: "str | None" = None,
                    provider: "str | None" = None,
                    model_family: "str | None" = None) -> dict[str, Any]:
    """Every dashboard stats aggregate in ONE window scan: overall totals plus
    the by_caller / by_provider / by_model_family / by_route / by_served_model /
    by_status / by_day breakdowns, as raw counters (see _agg_counter). Fail-soft
    -> the empty shape (same as aggregating zero rows)."""
    try:
        where, params = _usage_where(since_ts=since_ts, caller=caller,
                                     caller_is_null=caller_is_null,
                                     consumer_sha=consumer_sha, provider=provider,
                                     model_family=model_family)
        sql = ("SELECT grouping(caller_k, provider_k, family_k, route_k,"
               " served_k, status_k, day_k) AS gset,"
               " caller_k, provider_k, family_k, route_k, served_k, status_k,"
               " day_k," + _AGG_MEASURES +
               f" FROM ({_AGG_INNER}{where}) c"
               " GROUP BY GROUPING SETS ((), (caller_k), (provider_k),"
               " (family_k), (route_k), (served_k), (status_k), (day_k))")
        # grouping() bitmask -> which single-column set the row belongs to
        # (leftmost argument is the most significant bit; () = all bits set).
        sets = {63: ("by_caller", 1), 95: ("by_provider", 2),
                111: ("by_model_family", 3), 119: ("by_route", 4),
                123: ("by_served_model", 5), 126: ("by_day", 7)}
        out = _empty_usage_aggregate()
        with _get_pool().connection() as conn:
            for row in conn.execute(sql, params):
                gset = int(row[0])
                counter = _agg_counter(row[8:])
                if gset == 127:                      # () — overall totals
                    out["totals"] = counter
                elif gset == 125:                    # (status_k) — counts only
                    out["by_status"][str(row[6])] = counter["requests"]
                else:
                    bucket, idx = sets[gset]
                    out[bucket][str(row[idx])] = counter
        return out
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store usage_aggregate failed: %s", exc)
        return _empty_usage_aggregate()


def usage_totals(since_ts: "int | None" = None,
                 caller: "str | None" = None) -> dict[str, Any]:
    """One-row window totals over `calls`, including cached tokens (which the
    dashboard aggregate doesn't sum) — the control-plane metering read.
    Fail-soft -> zeros (same keys)."""
    zeros = {"requests": 0, "errors": 0, "tokens_in": 0, "tokens_out": 0,
             "tokens_cached": 0, "tokens_total": 0, "cost_usd": 0.0, "priced": 0}
    try:
        where, params = _usage_where(since_ts=since_ts, caller=caller)
        sql = (
            "SELECT count(*),"
            " count(*) FILTER (WHERE COALESCE(status,0) >= 400),"
            " COALESCE(sum(COALESCE(tokens_in,0)),0),"
            " COALESCE(sum(COALESCE(tokens_out,0)),0),"
            " COALESCE(sum(COALESCE(tokens_cached,0)),0),"
            " COALESCE(sum(CASE WHEN COALESCE(tokens_total,0) <> 0 THEN tokens_total"
            " ELSE COALESCE(tokens_in,0)+COALESCE(tokens_out,0) END),0),"
            " round(COALESCE(sum(GREATEST(cost_usd,0)),0)::numeric,6)::float8,"
            " count(cost_usd)"
            f" FROM calls{where}")
        with _get_pool().connection() as conn:
            row = conn.execute(sql, params).fetchone()
        requests, errors, tin, tout, tcached, ttotal, cost, priced = row
        return {"requests": int(requests), "errors": int(errors),
                "tokens_in": int(tin), "tokens_out": int(tout),
                "tokens_cached": int(tcached), "tokens_total": int(ttotal),
                "cost_usd": float(cost), "priced": int(priced)}
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store usage_totals failed: %s", exc)
        return zeros


def policy_backtest_groups(since_ts: "int | None" = None,
                           caller: "str | None" = None,
                           limit: int = 50) -> dict[str, Any]:
    """Top model-family groups for the Σ_pol backtester.

    The policy debugger's /x/rank surface ranks candidates by `model_family`.
    `calls.route_key` is provider|family|served and `requested_model` is the
    caller-facing alias/profile, so grouping here uses model_family: it maps 1:1
    to the `requirements.model_family` rank constraint.
    """
    try:
        limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        limit = 50
    try:
        where, params = _usage_where(since_ts=since_ts, caller=caller)
        sql = (
            "WITH base AS ("
            " -- Intentionally includes failed/error calls; unlike route_stats, this reflects actual traffic cost.\n"
            " SELECT COALESCE(NULLIF(model_family,''),'unknown') AS family_k,"
            " COALESCE(NULLIF(provider_id,''),'unknown') AS provider_k,"
            " tokens_in, tokens_out, cost_usd, latency_ms"
            f" FROM calls{where}"
            "), families AS ("
            " SELECT family_k, count(*) AS requests,"
            " COALESCE(sum(COALESCE(tokens_in,0)),0) AS tokens_in,"
            " COALESCE(sum(COALESCE(tokens_out,0)),0) AS tokens_out,"
            " round(COALESCE(sum(GREATEST(cost_usd,0)),0)::numeric,6)::float8 AS cost_usd,"
            " avg(latency_ms) AS latency_ms_avg"
            " FROM base GROUP BY family_k"
            "), ranked_families AS ("
            " SELECT family_k, requests, tokens_in, tokens_out, cost_usd,"
            " latency_ms_avg, count(*) OVER() AS groups_total,"
            " COALESCE(sum(requests) OVER(),0) AS requests_total,"
            " row_number() OVER (ORDER BY requests DESC, family_k ASC) AS rn"
            " FROM families"
            "), shown AS ("
            " SELECT * FROM ranked_families WHERE rn <= %s"
            "), providers AS ("
            " SELECT b.family_k, b.provider_k, count(*) AS requests,"
            " round(COALESCE(sum(GREATEST(b.cost_usd,0)),0)::numeric,6)::float8 AS cost_usd,"
            " avg(b.latency_ms) AS latency_ms_avg"
            " FROM base b JOIN shown s ON s.family_k = b.family_k"
            " GROUP BY b.family_k, b.provider_k"
            ")"
            " SELECT s.family_k, s.requests, s.tokens_in, s.tokens_out,"
            " s.cost_usd, s.latency_ms_avg, s.groups_total, s.requests_total,"
            " p.provider_k, p.requests, p.cost_usd, p.latency_ms_avg"
            " FROM shown s LEFT JOIN providers p ON p.family_k = s.family_k"
            " ORDER BY s.requests DESC, s.family_k ASC,"
            " p.requests DESC NULLS LAST, p.provider_k ASC"
        )
        groups: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        groups_total = 0
        requests_total = 0
        with _get_pool().connection() as conn:
            for row in conn.execute(sql, params + [limit]):
                family, requests, tin, tout, cost, lat, gtotal, rtotal, \
                    provider, prequests, pcost, plat = row
                groups_total = int(gtotal or 0)
                requests_total = int(rtotal or 0)
                family = str(family)
                if family not in groups:
                    groups[family] = {
                        "route": family,
                        "requests": int(requests or 0),
                        "tokens_in": int(tin or 0),
                        "tokens_out": int(tout or 0),
                        "actual_cost_usd": round(float(cost or 0.0), 6),
                        "actual_avg_ms": float(lat) if lat is not None else None,
                        "providers": {},
                        "provider_latency_ms": {},
                    }
                    order.append(family)
                if provider is not None:
                    pkey = str(provider)
                    groups[family]["providers"][pkey] = {
                        "requests": int(prequests or 0),
                        "cost_usd": round(float(pcost or 0.0), 6),
                    }
                    groups[family]["provider_latency_ms"][pkey] = (
                        float(plat) if plat is not None else None
                    )
        shown = [groups[name] for name in order]
        requests_covered = sum(int(g["requests"]) for g in shown)
        return {
            "groups_total": groups_total,
            "groups_shown": len(shown),
            "requests_total": requests_total,
            "requests_covered": requests_covered,
            "groups_truncated": max(0, groups_total - len(shown)),
            "requests_truncated": max(0, requests_total - requests_covered),
            "groups": shown,
        }
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store policy_backtest_groups failed: %s", exc)
        return {
            "groups_total": 0, "groups_shown": 0,
            "requests_total": 0, "requests_covered": 0,
            "groups_truncated": 0, "requests_truncated": 0,
            "groups": [],
        }


def usage_count(since_ts: "int | None" = None) -> int:
    """Row count in the window (the dashboard's history_events_all). Fail-soft -> 0."""
    try:
        where, params = _usage_where(since_ts=since_ts)
        with _get_pool().connection() as conn:
            return int(conn.execute(
                f"SELECT count(*) FROM calls{where}", params).fetchone()[0])
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store usage_count failed: %s", exc)
        return 0


def usage_provider_stats(since_ts: "int | None" = None) -> dict[str, dict[str, Any]]:
    """Per-provider usage counters + the latest event's fields in the window (the
    provider-credentials panel): {provider: counter + last_ts/last_status/
    last_route/last_model_family}. Tie order for "latest" (ts DESC, id DESC)
    matches the old ascending fold's `ts >= best` (last writer wins).
    Fail-soft -> {}."""
    try:
        where, params = _usage_where(since_ts=since_ts)
        out: dict[str, dict[str, Any]] = {}
        with _get_pool().connection() as conn:
            cur = conn.execute(
                "SELECT provider_k," + _AGG_MEASURES +
                f" FROM ({_AGG_INNER}{where}) c GROUP BY provider_k", params)
            for row in cur.fetchall():
                out[str(row[0])] = _agg_counter(row[1:])
            cur = conn.execute(
                "SELECT DISTINCT ON (provider_k) provider_k, ts, status,"
                " requested_model, model_family"
                f" FROM ({_AGG_INNER}{where}) c"
                " ORDER BY provider_k, ts DESC, id DESC", params)
            for provider_k, ts, status, route, family in cur.fetchall():
                item = out.setdefault(str(provider_k), _agg_counter((0,) * 7 + (None,)))
                item.update({"last_ts": ts, "last_status": status,
                             "last_route": route, "last_model_family": family})
        return out
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store usage_provider_stats failed: %s", exc)
        return {}


def usage_connections(since_ts: "int | None" = None,
                      caller: "str | None" = None) -> list[dict[str, Any]]:
    """Per (caller, key prefix) usage rollup in the window (the Connections
    panel): requests/errors/first_seen/last_seen plus the latest event's
    status/route/provider. Fail-soft -> []."""
    try:
        where, params = _usage_where(since_ts=since_ts, caller=caller)
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        with _get_pool().connection() as conn:
            cur = conn.execute(
                "SELECT caller_k, prefix_k, count(*),"
                " count(*) FILTER (WHERE COALESCE(status,0) >= 400),"
                " min(ts), max(ts)"
                f" FROM ({_AGG_INNER}{where}) c GROUP BY caller_k, prefix_k",
                params)
            for caller_k, prefix_k, requests, errors, first_seen, last_seen in cur.fetchall():
                grouped[(str(caller_k), str(prefix_k))] = {
                    "caller": str(caller_k), "prefix": str(prefix_k),
                    "requests": int(requests), "errors": int(errors),
                    "first_seen": int(first_seen), "last_seen": int(last_seen),
                    "last_status": None, "last_route": None, "last_provider": None}
            cur = conn.execute(
                "SELECT DISTINCT ON (caller_k, prefix_k) caller_k, prefix_k,"
                " status, requested_model, provider_id"
                f" FROM ({_AGG_INNER}{where}) c"
                " ORDER BY caller_k, prefix_k, ts DESC, id DESC", params)
            for caller_k, prefix_k, status, route, prov in cur.fetchall():
                item = grouped.get((str(caller_k), str(prefix_k)))
                if item is not None:
                    item.update({"last_status": status, "last_route": route,
                                 "last_provider": prov})
        return list(grouped.values())
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store usage_connections failed: %s", exc)
        return []


def usage_event_ids_present(ids: list[str],
                            since_ts: "int | None" = None) -> "set[str]":
    """Which of these usage_event_ids have already landed in `calls` — lets the
    dashboard fold in ONLY the not-yet-persisted runtime events instead of
    deduping the whole window in Python. `since_ts` (the oldest candidate's ts)
    keeps the probe on idx_calls_ts. Fail-soft -> set() (worst case a just-landed
    event is counted from the runtime feed too, exactly the pre-SQL behavior for
    id-less rows)."""
    if not ids:
        return set()
    try:
        where, params = _usage_where(since_ts=since_ts)
        with _get_pool().connection() as conn:
            cur = conn.execute(
                f"SELECT usage_event_id FROM calls{where}"
                " AND usage_event_id = ANY(%s)", params + [list(ids)])
            return {str(r[0]) for r in cur.fetchall()}
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store usage_event_ids_present failed: %s", exc)
        return set()


def backfill_call_costs(prices: "dict[tuple[str, str], dict[str, float]]",
                        batch_size: int = 1000) -> int:
    """One-time (idempotent) cost stamping: price every `calls` row with
    cost_usd IS NULL from the list-price table, exactly as the read-time
    fallback (auth_proxy._cost_for_event) did on every dashboard load — after
    this, SUM(cost_usd) is authoritative and the read path never re-prices.
    Rows with no price entry stay NULL (they summed as 0 before and still do).
    cost_basis follows the existing convention: 'computed' = derived from the
    list price (only set where it was NULL). Batched keyset loop; safe to run
    at every startup. Returns the number of rows stamped."""
    stamped = 0
    last_id = 0
    try:
        while True:
            with _get_pool().connection() as conn:
                rows = conn.execute(
                    "SELECT id, provider_id, model_family, tokens_in, tokens_out"
                    " FROM calls WHERE cost_usd IS NULL AND id > %s"
                    " ORDER BY id LIMIT %s", (last_id, int(batch_size))).fetchall()
                if not rows:
                    return stamped
                updates = []
                for id_, prov, fam, tin, tout in rows:
                    price = prices.get((str(fam or ""), str(prov or "")))
                    if not price:
                        continue
                    cost = round((int(tin or 0) / 1_000_000.0) * price["input"]
                                 + (int(tout or 0) / 1_000_000.0) * price["output"], 6)
                    updates.append((max(0.0, cost), id_))
                if updates:
                    conn.cursor().executemany(
                        "UPDATE calls SET cost_usd = %s,"
                        " cost_basis = COALESCE(cost_basis, 'computed')"
                        " WHERE id = %s AND cost_usd IS NULL", updates)
                stamped += len(updates)
                last_id = rows[-1][0]
    except Exception as exc:  # noqa: BLE001 — telemetry backfill must never break startup
        _log.warning("host_store backfill_call_costs failed: %s", exc)
        return stamped


def insert_login(row: dict[str, Any]) -> None:
    """Append one dashboard login audit record. Fail-soft (audit, never blocks)."""
    try:
        with _get_pool().connection() as conn:
            conn.execute("INSERT INTO login_history (ts, record) VALUES (%s,%s)",
                         (int(row.get("ts") or time.time()), json.dumps(row)))
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store insert_login failed: %s", exc)


def recent_logins(limit: int = 100) -> list[dict[str, Any]]:
    """The most recent dashboard login records, newest first. Fail-soft -> []."""
    try:
        with _get_pool().connection() as conn:
            cur = conn.execute(
                "SELECT record FROM login_history ORDER BY ts DESC, id DESC LIMIT %s",
                (int(limit),))
            out = []
            for (rec,) in cur.fetchall():
                try:
                    out.append(json.loads(rec))
                except (TypeError, ValueError):
                    continue
            return out
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store recent_logins failed: %s", exc)
        return []


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


# ---- provider_prices (direct-provider list prices; written by sources/official_pricing) ----

_PROVIDER_PRICE_FIELDS = ("provider_id", "model_family", "price_in",
                         "price_out", "price_cached_in")


def get_provider_prices(provider_id: "str | None" = None) -> list[dict[str, Any]]:
    """Last-known direct-provider list prices, one row per (provider, family).
    Fail-soft: [] on any store error, so a source coasts on nothing exactly as an
    empty scrape would. Pass provider_id to scope to one provider."""
    try:
        cols = ", ".join(_PROVIDER_PRICE_FIELDS)
        with _get_pool().connection() as conn:
            if provider_id is None:
                cur = conn.execute(f"SELECT {cols} FROM provider_prices")
            else:
                cur = conn.execute(
                    f"SELECT {cols} FROM provider_prices WHERE provider_id = %s",
                    (provider_id,))
            return [dict(zip(_PROVIDER_PRICE_FIELDS, r)) for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001 — price read is best-effort
        _log.warning("host_store get_provider_prices failed: %s", exc)
        return []


def set_provider_prices(rows: list[dict[str, Any]]) -> bool:
    """Upsert direct-provider prices (one row per (provider_id, model_family)).
    Upsert, NOT replace: providers share the table, each writes only its own rows.
    Returns False on a persistence failure so the source can log it."""
    try:
        now = int(time.time())
        values = [(r["provider_id"], r["model_family"], r.get("price_in"),
                   r.get("price_out"), r.get("price_cached_in"), now)
                  for r in (rows or [])]
        if not values:
            return True
        with _get_pool().connection() as conn:
            conn.cursor().executemany(
                "INSERT INTO provider_prices(provider_id, model_family, price_in,"
                " price_out, price_cached_in, updated_at) VALUES (%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (provider_id, model_family) DO UPDATE SET"
                " price_in=EXCLUDED.price_in, price_out=EXCLUDED.price_out,"
                " price_cached_in=EXCLUDED.price_cached_in,"
                " updated_at=EXCLUDED.updated_at", values)
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("host_store set_provider_prices failed: %s", exc)
        return False


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
                     " consumer_keys, peer_offers, buyer_status, route_observations,"
                     " login_history, provider_prices")
