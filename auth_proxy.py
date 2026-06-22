from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import settings
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from env_secrets import load_env_secrets

# Operator-managed keys/consumer-hashes live on the PVC (.env.secrets) and are
# the source of truth — load them over the container env BEFORE reading any of
# the values below, so dashboard edits survive pod restarts.
load_env_secrets()

UPSTREAM = os.getenv("ROUTER_UPSTREAM", "http://router:18080").rstrip("/")
CALLER_KEYS_JSON = os.getenv("CALLER_KEYS_JSON", "{}")
CALLER_KEYS_SHA256_JSON = os.getenv("CALLER_KEYS_SHA256_JSON", "{}")
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "600"))
BURST = int(os.getenv("BURST", "200"))
RECENT_LIMIT = int(os.getenv("DASHBOARD_RECENT_LIMIT", "200"))
DASHBOARD_TRUSTED_USER_HEADER = os.getenv("DASHBOARD_TRUSTED_USER_HEADER", "").strip()
DASHBOARD_TRUSTED_USER_SECRET = os.getenv("DASHBOARD_TRUSTED_USER_SECRET", "")
DASHBOARD_PASSWORD_SHA256 = os.getenv("DASHBOARD_PASSWORD_SHA256", "")
DASHBOARD_SESSION_SECRET = os.getenv("DASHBOARD_SESSION_SECRET", "")
# Local-dev escape hatch: when truthy, the dashboard skips auth entirely and every
# request is treated as a local admin. OFF by default — never enable in a
# deployment reachable by anyone but you.
DASHBOARD_NO_AUTH = os.getenv("DASHBOARD_NO_AUTH", "").strip().lower() in ("1", "true", "yes", "on")
DASHBOARD_COOKIE_NAME = os.getenv("DASHBOARD_COOKIE_NAME", "router_dashboard_session")
DASHBOARD_COOKIE_MAX_AGE = int(os.getenv("DASHBOARD_COOKIE_MAX_AGE", "2592000"))
DASHBOARD_COOKIE_PATH = os.getenv("DASHBOARD_COOKIE_PATH", "/dashboard")
DASHBOARD_KEY_ENV_PATH = os.getenv("DASHBOARD_KEY_ENV_PATH", "/run/llm-router/.env.secrets")
DASHBOARD_ISSUED_KEYS_PATH = os.getenv("DASHBOARD_ISSUED_KEYS_PATH", "/run/llm-router/secrets/issued-consumer-keys.json")
CODEX_ACCOUNTS_DIR = os.getenv("CODEX_ACCOUNTS_DIR", "/codex/accounts")
CODEX_AUTH_PATH = os.getenv("CODEX_AUTH_PATH") or None
DASHBOARD_LOGIN_HISTORY_PATH = os.getenv("DASHBOARD_LOGIN_HISTORY_PATH", "/run/llm-router/secrets/dashboard-logins.jsonl")
DASHBOARD_KEY_PREFIX = os.getenv("DASHBOARD_KEY_PREFIX", "llmr")
DEFAULT_ROTATION_GRACE_S = int(os.getenv("DASHBOARD_KEY_ROTATION_GRACE_S", "86400"))
DASHBOARD_POLICY_CONFIG_PATH = os.getenv("DASHBOARD_POLICY_CONFIG_PATH", "config.live.lua")
DASHBOARD_POLICY_METRICS_PATH = os.getenv("DASHBOARD_POLICY_METRICS_PATH", "metrics.live.lua")
USAGE_HISTORY_PATH_DEFAULT = os.getenv("ROUTER_USAGE_HISTORY_PATH", "/run/llm-router/usage-history.jsonl")
DASHBOARD_POLICY_DIR = os.getenv("DASHBOARD_POLICY_DIR", "policies")
ROUTER_CONTEXT_LENGTH = int(os.getenv("ROUTER_CONTEXT_LENGTH", "200000"))
ROUTE_HEALTH_ROUTES = [r.strip() for r in os.getenv("DASHBOARD_ROUTE_HEALTH_ROUTES", "profile:default").split(",") if r.strip()]
SYNTHETIC_PROBES_ENABLED = os.getenv("DASHBOARD_SYNTHETIC_PROBES_ENABLED", "1").lower() not in {"0", "false", "no", "off"}
SYNTHETIC_PROBE_INTERVAL_S = float(os.getenv("DASHBOARD_SYNTHETIC_PROBE_INTERVAL_S", "300"))
SYNTHETIC_PROBE_INITIAL_DELAY_S = float(os.getenv("DASHBOARD_SYNTHETIC_PROBE_INITIAL_DELAY_S", "45"))
SYNTHETIC_PROBE_TIMEOUT_S = float(os.getenv("DASHBOARD_SYNTHETIC_PROBE_TIMEOUT_S", "45"))
SYNTHETIC_PROBE_CALLER = os.getenv("DASHBOARD_SYNTHETIC_PROBE_CALLER", "dashboard-probe")



def _load_caller_map(raw: str, name: str) -> Dict[str, str]:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"{name} must be a JSON object")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
            raise ValueError(f"{name} keys and values must be strings")
        return data
    except Exception as exc:
        raise RuntimeError(f"invalid {name}: {exc}") from exc


CALLER_KEYS: Dict[str, str] = _load_caller_map(CALLER_KEYS_JSON, "CALLER_KEYS_JSON")
CALLER_KEY_HASHES: Dict[str, str] = _load_caller_map(CALLER_KEYS_SHA256_JSON, "CALLER_KEYS_SHA256_JSON")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
log = logging.getLogger("llm-router-auth-proxy")
app = FastAPI(title="llm-router auth proxy", docs_url=None, redoc_url=None)
_client: httpx.AsyncClient | None = None
_probe_task: asyncio.Task[None] | None = None
_windows: dict[str, deque[float]] = defaultdict(deque)
_started_wall = time.time()
_stats_lock = RLock()


def _counter() -> dict[str, Any]:
    return {
        "requests": 0,
        "errors": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_total": 0,
        "latency_ms_total": 0.0,
        "latency_ms_max": 0.0,
        "last_seen": None,
    }


def _new_stats() -> dict[str, Any]:
    return {
        "total_requests": 0,
        "total_rejects": 0,
        "total_errors": 0,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "total_tokens": 0,
        "by_caller": defaultdict(_counter),
        "by_provider": defaultdict(_counter),
        "by_model_family": defaultdict(_counter),
        "by_route": defaultdict(_counter),
        "by_served_model": defaultdict(_counter),
        "by_status": defaultdict(int),
        "by_caller_provider": defaultdict(lambda: defaultdict(_counter)),
        "by_caller_model_family": defaultdict(lambda: defaultdict(_counter)),
        "by_caller_route": defaultdict(lambda: defaultdict(_counter)),
        "by_caller_served_model": defaultdict(lambda: defaultdict(_counter)),
        "by_caller_status": defaultdict(lambda: defaultdict(int)),
        "by_key_sha256": defaultdict(_counter),
        "by_key_provider": defaultdict(lambda: defaultdict(_counter)),
        "by_key_model_family": defaultdict(lambda: defaultdict(_counter)),
        "by_key_route": defaultdict(lambda: defaultdict(_counter)),
        "by_key_served_model": defaultdict(lambda: defaultdict(_counter)),
        "by_key_status": defaultdict(lambda: defaultdict(int)),
        "key_owner": {},
        "recent": deque(maxlen=RECENT_LIMIT),
        "synthetic_route_health": {},
    }


_stats: dict[str, Any] = _new_stats()
_login_events: deque[dict[str, Any]] = deque(maxlen=RECENT_LIMIT)


def _reset_stats_for_tests() -> None:
    """Reset in-memory counters without touching persistent usage history."""
    with _stats_lock:
        _stats.clear()
        _stats.update(_new_stats())


def _consumers() -> list[str]:
    return sorted(set(CALLER_KEYS.values()) | set(CALLER_KEY_HASHES.values()) | set(_load_issued_keys().keys()))


def _safe_consumer_name(raw: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.:-]+", "-", (raw or "").strip()).strip("-._:")
    if not name or len(name) > 80:
        raise ValueError("consumer must be 1-80 chars: letters, numbers, dot, underscore, colon, or dash")
    return name


def _write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.chmod(0o600)
    tmp.replace(path)
    path.chmod(0o600)


def _upsert_env_line(path: Path, key: str, value: str) -> None:
    """Set KEY=value in an env file (used for provider API keys). Same
    semantics as _upsert_env_json but for plain string values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if path.exists() else []
    rendered = f"{key}={value}"
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith(key + "="):
            out.append(rendered)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(rendered)
    path.write_text("\n".join(out) + "\n")
    path.chmod(0o600)


def _upsert_env_json(path: Path, key: str, value: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text().splitlines() if path.exists() else []
    rendered = key + "=" + json.dumps(value, sort_keys=True, separators=(",", ":"))
    out: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith(key + "="):
            out.append(rendered)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(rendered)
    path.write_text("\n".join(out) + "\n")
    path.chmod(0o600)


_issued_keys_load_failed = False


def _load_issued_keys() -> dict[str, Any]:
    global _issued_keys_load_failed
    path = Path(DASHBOARD_ISSUED_KEYS_PATH)
    if not path.exists():
        _issued_keys_load_failed = False
        return {}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            _issued_keys_load_failed = False
            return data
    except Exception:
        pass
    _issued_keys_load_failed = True
    return {}


def _clean_route_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        items = re.split(r"[\s,]+", value)
    elif isinstance(value, list):
        items = [str(v) for v in value]
    else:
        return []
    out = []
    for item in items:
        route = item.strip()
        if route and len(route) <= 160 and route not in out:
            out.append(route)
    return out


def _optional_int(value: Any, *, min_value: int = 0, max_value: int | None = 1_000_000) -> int | None:
    if value is None or value == "":
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    if max_value is not None:
        ivalue = min(max_value, ivalue)
    return max(min_value, ivalue)


def _normalize_key_record(record: Any) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None
    prefix = str(record.get("sha256_prefix") or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{8,64}", prefix):
        return None
    status = str(record.get("status") or "active").strip().lower()
    if status not in {"active", "revoked"}:
        status = "revoked"
    out = {
        "sha256_prefix": prefix,
        "status": status,
        "created_at": _optional_int(record.get("created_at"), max_value=None),
        "viewer": str(record.get("viewer") or "")[:80] or None,
        "expires_at": _optional_int(record.get("expires_at"), max_value=None),
        "revoked_at": _optional_int(record.get("revoked_at"), max_value=None),
        "replaced_at": _optional_int(record.get("replaced_at"), max_value=None),
    }
    return {k: v for k, v in out.items() if v is not None}


def _normalize_consumer_record(consumer: str, record: Any) -> dict[str, Any]:
    now = int(time.time())
    if record is None:
        record = {}
    elif not isinstance(record, dict):
        return {"status": "inactive", "allowed_routes": [], "rate_per_min": None, "burst": None, "keys": [], "updated_at": now}
    status = str(record.get("status") or "active").strip().lower()
    if status not in {"active", "inactive"}:
        status = "inactive"
    keys = []
    if "keys" in record and not isinstance(record.get("keys"), list):
        return {"status": "inactive", "allowed_routes": [], "rate_per_min": None, "burst": None, "keys": [], "updated_at": now}
    raw_keys = record.get("keys") or []
    for item in raw_keys:
        normalized = _normalize_key_record(item)
        if not normalized:
            return {"status": "inactive", "allowed_routes": [], "rate_per_min": None, "burst": None, "keys": [], "updated_at": now}
        keys.append(normalized)
    # Backward compatibility with the older {consumer: {sha256_prefix, created_at, viewer}} shape.
    legacy = _normalize_key_record(record)
    if legacy and not any(k["sha256_prefix"] == legacy["sha256_prefix"] for k in keys):
        keys.append(legacy)
    for token, owner in CALLER_KEYS.items():
        if owner == consumer:
            digest = hashlib.sha256(token.encode()).hexdigest()
            if not any(digest.startswith(k["sha256_prefix"]) for k in keys):
                keys.append({"sha256_prefix": digest[:12], "status": "active", "storage": "CALLER_KEYS_JSON"})
    for digest, owner in CALLER_KEY_HASHES.items():
        if owner == consumer and not any(digest.startswith(k["sha256_prefix"]) for k in keys):
            keys.append({"sha256_prefix": digest[:12], "status": "active", "storage": "CALLER_KEYS_SHA256_JSON"})
    rate_per_min = _optional_int(record.get("rate_per_min"), min_value=1)
    burst = _optional_int(record.get("burst"), min_value=1)
    return {
        "status": status,
        "allowed_routes": _clean_route_list(record.get("allowed_routes")),
        "rate_per_min": rate_per_min,
        "burst": burst,
        "keys": keys,
        "updated_at": _optional_int(record.get("updated_at")) or now,
    }


def _issued_consumer_records() -> dict[str, dict[str, Any]]:
    issued = _load_issued_keys()
    consumers = sorted(set(CALLER_KEYS.values()) | set(CALLER_KEY_HASHES.values()) | set(issued.keys()))
    return {name: _normalize_consumer_record(name, issued.get(name)) for name in consumers}


def _write_issued_consumer_records(records: dict[str, dict[str, Any]]) -> None:
    compact = {}
    for consumer, record in sorted(records.items()):
        normalized = _normalize_consumer_record(consumer, record)
        if normalized["status"] == "active" and not normalized["allowed_routes"] and normalized["rate_per_min"] is None and normalized["burst"] is None and not normalized["keys"]:
            continue
        compact[consumer] = normalized
    _write_json_file(Path(DASHBOARD_ISSUED_KEYS_PATH), compact)


def _consumer_meta(consumer: str) -> dict[str, Any]:
    records = _issued_consumer_records()
    if _issued_keys_load_failed:
        meta = _normalize_consumer_record(consumer, {})
        meta["status"] = "inactive"
        return meta
    return records.get(consumer, _normalize_consumer_record(consumer, {}))


def _active_key_rows(keys: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = int(time.time())
    rows = []
    for key in keys:
        row = dict(key)
        if row.get("status") == "active" and row.get("expires_at") and int(row["expires_at"]) <= now:
            row["status"] = "expired"
        rows.append(row)
    return rows


def _consumer_key_rows(by_caller_stats: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    records = _issued_consumer_records()
    plaintext_counts: dict[str, int] = defaultdict(int)
    hash_counts: dict[str, int] = defaultdict(int)
    for consumer in CALLER_KEYS.values():
        plaintext_counts[consumer] += 1
    for consumer in CALLER_KEY_HASHES.values():
        hash_counts[consumer] += 1
    rows = []
    for name in _consumers():
        meta = records.get(name, _normalize_consumer_record(name, {}))
        rows.append({
            "consumer": name,
            "configured": name in set(CALLER_KEYS.values()) or name in set(CALLER_KEY_HASHES.values()),
            "status": meta.get("status", "active"),
            "allowed_routes": meta.get("allowed_routes") or [],
            "rate_per_min": meta.get("rate_per_min"),
            "burst": meta.get("burst"),
            "effective_rate_per_min": meta.get("rate_per_min") or RATE_PER_MIN,
            "effective_burst": meta.get("burst") or BURST,
            "issued_metadata": name in records,
            "stored_raw_key": plaintext_counts.get(name, 0) > 0,
            "plaintext_key_count": plaintext_counts.get(name, 0),
            "hash_key_count": hash_counts.get(name, 0),
            "keys": _active_key_rows(meta.get("keys") or []),
            "stats": (by_caller_stats or {}).get(name) or _counter_snapshot(_stats["by_caller"].get(name, _counter())),
        })
    return rows


def _plaintext_key_rows_for_consumer(consumer: str) -> list[dict[str, Any]]:
    rows = []
    for token, owner in sorted(CALLER_KEYS.items(), key=lambda item: hashlib.sha256(item[0].encode()).hexdigest()):
        if owner != consumer:
            continue
        digest = hashlib.sha256(token.encode()).hexdigest()
        rows.append({
            "consumer": owner,
            "api_key": token,
            "sha256_prefix": digest[:12],
            "storage": "CALLER_KEYS_JSON",
        })
    return rows


def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def _caller_from_token(token: str | None) -> str | None:
    auth = _caller_auth(token)
    return auth.get("caller") if auth.get("ok") else None


def _key_record_allows(digest: str, meta: dict[str, Any]) -> tuple[bool, str | None]:
    keys = meta.get("keys") or []
    if not keys:
        return True, None
    matching = [k for k in keys if digest.startswith(str(k.get("sha256_prefix") or ""))]
    if not matching:
        return True, None
    now = int(time.time())
    for key in matching:
        if key.get("status") == "revoked":
            return False, "caller_key_revoked"
        expires_at = int(key.get("expires_at") or 0)
        if expires_at and expires_at <= now:
            return False, "caller_key_expired"
    return True, None


def _caller_auth(token: str | None) -> dict[str, Any]:
    if not token:
        return {"ok": False, "error_code": "caller_auth"}
    digest = hashlib.sha256(token.encode()).hexdigest()
    caller = CALLER_KEYS.get(token)
    storage = "CALLER_KEYS_JSON" if caller else None
    if not caller:
        caller = CALLER_KEY_HASHES.get(digest)
        storage = "CALLER_KEYS_SHA256_JSON" if caller else None
    if not caller:
        return {"ok": False, "error_code": "caller_auth"}
    meta = _consumer_meta(caller)
    if meta.get("status") != "active":
        return {"ok": False, "caller": caller, "digest": digest, "storage": storage, "error_code": "caller_inactive"}
    allowed, key_error = _key_record_allows(digest, meta)
    if not allowed:
        return {"ok": False, "caller": caller, "digest": digest, "storage": storage, "error_code": key_error}
    return {"ok": True, "caller": caller, "digest": digest, "storage": storage, "meta": meta}


def _rate_ok(caller: str) -> bool:
    meta = _consumer_meta(caller)
    rate_per_min = int(meta.get("rate_per_min") or RATE_PER_MIN)
    burst = int(meta.get("burst") or BURST)
    now = time.monotonic()
    q = _windows[caller]
    cutoff = now - 60.0
    while q and q[0] < cutoff:
        q.popleft()
    allowed = max(rate_per_min, burst)
    if len(q) >= allowed:
        return False
    q.append(now)
    return True


def _requested_route_from(path: str, body: bytes | None) -> str | None:
    route = None
    if body:
        try:
            parsed = json.loads(body.decode("utf-8"))
            if isinstance(parsed, dict):
                route = parsed.get("model") or parsed.get("name")
        except Exception:
            route = None
    parts = [p for p in path.strip("/").split("/") if p]
    if len(parts) >= 3 and parts[1:] == ["v1", "chat", "completions"] and parts[0] != "v1":
        route = route or f"profile:{parts[0]}"
    return str(route).strip() if route else None


def _route_matches(pattern: str, route: str) -> bool:
    pattern = pattern.strip()
    if pattern in {"*", "all"} or pattern == route:
        return True
    if pattern.endswith("*") and route.startswith(pattern[:-1]):
        return True
    return False


def _route_allowed(caller: str, route: str | None) -> bool:
    allowed_routes = _consumer_meta(caller).get("allowed_routes") or []
    if not allowed_routes:
        return True
    if not route:
        return False
    return any(_route_matches(pattern, route) for pattern in allowed_routes)


def _log(event: dict) -> None:
    log.info(json.dumps(event, sort_keys=True, separators=(",", ":")))



def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(raw: str) -> bytes:
    return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))


def _make_dashboard_session(user: str, *, role: str = "admin", consumer: str | None = None, key_sha256: str | None = None) -> str:
    if not DASHBOARD_SESSION_SECRET:
        raise RuntimeError("DASHBOARD_SESSION_SECRET is not configured")
    payload = {"u": user, "r": role, "e": int(time.time()) + DASHBOARD_COOKIE_MAX_AGE}
    if consumer:
        payload["c"] = consumer
    if key_sha256:
        payload["k"] = key_sha256
    body = _b64e(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig = hmac.new(DASHBOARD_SESSION_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def _dashboard_session_context(request: Request) -> dict[str, Any] | None:
    if not DASHBOARD_SESSION_SECRET:
        return None
    raw = request.cookies.get(DASHBOARD_COOKIE_NAME, "")
    if "." not in raw:
        return None
    body, sig = raw.rsplit(".", 1)
    expected = hmac.new(DASHBOARD_SESSION_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(_b64d(body))
    except Exception:
        return None
    if int(payload.get("e", 0)) < int(time.time()):
        return None
    user = str(payload.get("u", "")).strip()
    if not user:
        return None
    role = str(payload.get("r") or "admin").strip()
    if role == "consumer":
        consumer = str(payload.get("c", "")).strip()
        key_sha256 = str(payload.get("k", "")).strip().lower()
        if not consumer or not re.fullmatch(r"[a-f0-9]{64}", key_sha256):
            return None
        return {"role": "consumer", "user": user, "viewer": f"consumer:{consumer}", "consumer": consumer, "key_sha256": key_sha256}
    return {"role": "admin", "user": user, "viewer": f"dashboard:{user}", "consumer": None, "key_sha256": None}


def _dashboard_session_user(request: Request) -> str | None:
    ctx = _dashboard_session_context(request)
    return str(ctx.get("viewer")) if ctx else None


def _dashboard_password_ok(password: str) -> bool:
    if not DASHBOARD_PASSWORD_SHA256:
        return False
    got = hashlib.sha256(password.encode()).hexdigest()
    return hmac.compare_digest(got, DASHBOARD_PASSWORD_SHA256)

def _require_dashboard_context(request: Request) -> dict[str, Any] | None:
    if DASHBOARD_NO_AUTH:
        return {"role": "admin", "user": "local-dev", "viewer": "dashboard:local-dev", "consumer": None, "key_sha256": None}
    if DASHBOARD_TRUSTED_USER_HEADER:
        trusted_user = (request.headers.get(DASHBOARD_TRUSTED_USER_HEADER) or "").strip()
        trusted_secret = (request.headers.get("x-dashboard-trusted-secret") or "").strip()
        if trusted_user and DASHBOARD_TRUSTED_USER_SECRET and hmac.compare_digest(trusted_secret, DASHBOARD_TRUSTED_USER_SECRET):
            return {"role": "admin", "user": trusted_user, "viewer": f"dashboard:{trusted_user}", "consumer": None, "key_sha256": None}
    return _dashboard_session_context(request)


def _require_dashboard_auth(request: Request) -> str | None:
    ctx = _require_dashboard_context(request)
    return str(ctx.get("viewer")) if ctx else None


def _require_admin_dashboard_auth(request: Request) -> tuple[dict[str, Any] | None, Response | None]:
    ctx = _require_dashboard_context(request)
    if not ctx:
        return None, JSONResponse(status_code=401, content={"error": {"message": "unauthorized dashboard caller", "type": "auth_error", "code": "dashboard_auth"}})
    if ctx.get("role") != "admin":
        return None, JSONResponse(status_code=403, content={"error": {"message": "admin dashboard session required", "type": "auth_error", "code": "dashboard_admin_required"}})
    return ctx, None


def _require_admin_dashboard_caller(request: Request) -> tuple[str | None, Response | None]:
    # Privileged dashboard ops (reveal keys, manage codex, add/edit providers)
    # gate on the ADMIN role — which both the SSO trusted-header path and the
    # password login produce. There are no per-user tiers, so there is no extra
    # per-name gate: access control is whoever SSO/the password admits. Returns
    # the caller (viewer) string for audit, or an error Response.
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return None, error
    return str(ctx.get("viewer")), None



def _login_history_path() -> Path | None:
    raw = (DASHBOARD_LOGIN_HISTORY_PATH or "").strip()
    return Path(raw) if raw else None


def _mask_remote(remote: str | None) -> str | None:
    if not remote:
        return None
    host = str(remote).strip()
    if not host:
        return None
    try:
        import ipaddress
        ip = ipaddress.ip_address(host)
        if ip.version == 4:
            parts = host.split(".")
            return ".".join(parts[:3] + ["0"]) + "/24"
        return str(ipaddress.ip_network(str(ip) + "/64", strict=False))
    except Exception:
        return hashlib.sha256(host.encode()).hexdigest()[:12]


def _user_agent_hash(request: Request) -> str | None:
    ua = (request.headers.get("user-agent") or "").strip()
    return hashlib.sha256(ua.encode()).hexdigest()[:12] if ua else None


def _append_login_history(row: dict[str, Any]) -> None:
    path = _login_history_path()
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        path.chmod(0o600)
    except Exception as exc:
        log.warning(json.dumps({"event": "dashboard_login_history_write_failed", "path": str(path), "error": str(exc)}))


def _read_login_history() -> list[dict[str, Any]]:
    path = _login_history_path()
    if not path or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict) and row.get("event") == "dashboard_login":
                    rows.append(row)
    except Exception as exc:
        log.warning(json.dumps({"event": "dashboard_login_history_read_failed", "path": str(path), "error": str(exc)}))
    return rows


def _record_dashboard_login(*, request: Request, role: str, user: str, consumer: str | None = None, key_sha256: str | None = None) -> None:
    key_prefix = key_sha256[:12] if key_sha256 and re.fullmatch(r"[a-f0-9]{64}", key_sha256) else None
    row = {
        "ts": int(time.time()),
        "event": "dashboard_login",
        "role": role,
        "viewer": f"consumer:{consumer}" if role == "consumer" and consumer else f"dashboard:{user}",
        "consumer": consumer,
        "key_sha256_prefix": key_prefix,
        "remote_masked": _mask_remote(request.client.host if request.client else None),
        "user_agent_hash": _user_agent_hash(request),
    }
    row = {k: v for k, v in row.items() if v is not None}
    with _stats_lock:
        _login_events.appendleft(row)
    _append_login_history(row)



def _provider_raw_credential(name: str) -> tuple[str | None, str | None]:
    """(kind, raw value) of one provider's credential — env API keys come
    from the configured env var; codex OAuth comes from auth.json via
    CodexAuth (refreshed, so the copied token is currently valid). Only the
    admin-gated reveal endpoint may call this."""
    try:
        cfg = _load_policy_config()
    except Exception:
        return None, None
    provider = (cfg.get("providers") or {}).get(name)
    if not isinstance(provider, dict):
        return None, None
    auth_env = str(provider.get("auth_env") or "").strip()
    if auth_env:
        val = os.getenv(auth_env, "").strip()
        return "env", (val or None)
    auth = provider.get("auth") if isinstance(provider.get("auth"), dict) else {}
    if auth.get("kind") == "oauth" or provider.get("api_kind") == "openai_codex":
        try:
            from codex_auth import CodexAuth
            token = CodexAuth(os.getenv("CODEX_AUTH_PATH") or None).access_token()
            return "oauth", (token or None)
        except Exception:
            return "oauth", None
    return None, None


def _provider_credentials_snapshot(*, timeframe: str = "all", viewer_role: str = "admin") -> dict[str, Any]:
    """Admin-only provider credential/status/usage view. Never returns raw env values."""
    if viewer_role != "admin":
        return {"rows": [], "privacy": {"admin_only": True, "hidden_for_consumer_sessions": True}}
    now = int(time.time())
    since = None
    if timeframe == "runtime":
        since = int(_started_wall)
    elif timeframe == "1h":
        since = now - 3600
    elif timeframe == "24h":
        since = now - 86400
    elif timeframe == "7d":
        since = now - 7 * 86400
    elif timeframe == "30d":
        since = now - 30 * 86400

    try:
        cfg = _load_policy_config()
    except Exception:
        cfg = {"providers": {}}
    providers = cfg.get("providers") or {}

    usage_rows = _read_usage_history() if timeframe != "runtime" else []
    with _stats_lock:
        usage_rows.extend([dict(r) for r in _stats["recent"] if r.get("event") == "request"])
        runtime_by_provider = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_provider"].items())}
    if since is not None:
        usage_rows = [r for r in usage_rows if int(r.get("ts") or 0) >= since]

    prices = _price_table()
    counters: dict[str, dict[str, Any]] = defaultdict(_counter)
    last_event: dict[str, dict[str, Any]] = {}
    cost_by_provider: dict[str, float] = defaultdict(float)
    for row in usage_rows:
        provider = str(row.get("provider") or "unknown")
        cost, _meta = _cost_for_event(row, prices)
        _add_counter(counters[provider], row, cost)
        if cost is not None:
            cost_by_provider[provider] = round(cost_by_provider[provider] + cost, 6)
        ts = int(row.get("ts") or 0)
        if provider not in last_event or ts >= int(last_event[provider].get("ts") or 0):
            last_event[provider] = row

    rows = []
    for name, provider in sorted(providers.items()):
        auth = provider.get("auth") if isinstance(provider.get("auth"), dict) else None
        auth_env = str(provider.get("auth_env") or "").strip() or None
        auth_kind = str(auth.get("kind")) if auth else ("env_api_key" if auth_env else "none")
        key_present = False
        key_fingerprint = None
        status = "missing"
        if auth_env:
            val = os.getenv(auth_env, "")
            key_present = bool(val.strip())
            status = "configured" if key_present else "missing"
            key_fingerprint = hashlib.sha256(val.encode()).hexdigest()[:12] if key_present else None
        elif auth_kind == "none":
            status = "no_auth_required"
        elif auth_kind == "oauth":
            status = "oauth_configured"
        elif auth_kind:
            status = "configured"
        counter = _counter_snapshot(counters.get(name, _counter()))
        if timeframe == "runtime" and not counter.get("requests"):
            counter = runtime_by_provider.get(name, counter)
        latest = last_event.get(name, {})
        rows.append({
            "provider": name,
            "tier": provider.get("tier"),
            "api_kind": provider.get("api_kind"),
            "base_url": provider.get("base_url"),
            "auth_kind": auth_kind,
            "auth_env": auth_env,
            "credential_status": status,
            "key_present": key_present,
            "key_fingerprint": key_fingerprint,
            "usage": counter,
            "estimated_cost_usd": round(cost_by_provider.get(name, 0.0), 6),
            "last_status": latest.get("status"),
            "last_route": latest.get("requested_model") or latest.get("route"),
            "last_model_family": latest.get("model_family"),
            "last_seen": latest.get("ts") or counter.get("last_seen"),
            "notes": provider.get("notes"),
        })
    known = {r["provider"] for r in rows}
    for name, counter in sorted(counters.items()):
        if name in known or name == "unknown":
            continue
        latest = last_event.get(name, {})
        rows.append({
            "provider": name,
            "tier": None,
            "api_kind": None,
            "base_url": None,
            "auth_kind": "unknown",
            "auth_env": None,
            "credential_status": "seen_in_usage_only",
            "key_present": None,
            "key_fingerprint": None,
            "usage": _counter_snapshot(counter),
            "estimated_cost_usd": round(cost_by_provider.get(name, 0.0), 6),
            "last_status": latest.get("status"),
            "last_route": latest.get("requested_model") or latest.get("route"),
            "last_model_family": latest.get("model_family"),
            "last_seen": latest.get("ts"),
            "notes": "Provider appeared in usage but is not in current config.",
        })
    rows.sort(key=lambda r: (0 if r.get("credential_status") in {"configured", "oauth_configured", "no_auth_required"} else 1, -int((r.get("usage") or {}).get("requests") or 0), str(r.get("provider"))))
    return {
        "schema_version": 1,
        "kind": "router_provider_credentials",
        "generated_at": now,
        "timeframe": timeframe,
        "rows": rows,
        "privacy": {
            "admin_only": True,
            "raw_provider_keys_exposed": False,
            "full_key_hashes_exposed": False,
            "key_fingerprint": "sha256_prefix_12_when_env_key_present",
        },
    }


def _login_connections_snapshot(*, timeframe: str = "all", consumer: str | None = None, viewer_role: str = "admin") -> dict[str, Any]:
    if viewer_role != "admin":
        return {"rows": [], "recent_dashboard_logins": [], "privacy": {"admin_only": True, "hidden_for_consumer_sessions": True}}
    now = int(time.time())
    since = None
    if timeframe == "runtime":
        since = int(_started_wall)
    elif timeframe == "1h":
        since = now - 3600
    elif timeframe == "24h":
        since = now - 86400
    elif timeframe == "7d":
        since = now - 7 * 86400
    elif timeframe == "30d":
        since = now - 30 * 86400

    dashboard_rows = list(_read_login_history())
    with _stats_lock:
        dashboard_rows.extend(list(_login_events))
        runtime_recent = list(_stats["recent"])
    dedup = {}
    for row in dashboard_rows:
        key = (row.get("ts"), row.get("viewer"), row.get("key_sha256_prefix"), row.get("remote_masked"), row.get("user_agent_hash"))
        dedup[key] = row
    dashboard_rows = list(dedup.values())
    if since is not None:
        dashboard_rows = [r for r in dashboard_rows if int(r.get("ts") or 0) >= since]
    if consumer:
        dashboard_rows = [r for r in dashboard_rows if r.get("consumer") == consumer or r.get("viewer") == f"consumer:{consumer}"]

    usage_rows = _read_usage_history() if timeframe != "runtime" else []
    usage_rows.extend([r for r in runtime_recent if r.get("event") == "request"])
    if since is not None:
        usage_rows = [r for r in usage_rows if int(r.get("ts") or 0) >= since]
    if consumer:
        usage_rows = [r for r in usage_rows if r.get("caller") == consumer]

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in usage_rows:
        caller = str(row.get("caller") or "unknown")
        prefix = str(row.get("key_sha256_prefix") or "").strip() or "unknown"
        key = (caller, prefix)
        item = grouped.setdefault(key, {"kind": "api_key", "identity": caller, "consumer": caller, "key_sha256_prefix": prefix if prefix != "unknown" else None, "first_seen": None, "last_seen": None, "requests": 0, "errors": 0, "last_status": None, "last_route": None, "last_provider": None})
        ts = int(row.get("ts") or 0)
        item["requests"] += 1
        if int(row.get("status") or 0) >= 400:
            item["errors"] += 1
        item["first_seen"] = ts if not item["first_seen"] else min(int(item["first_seen"]), ts)
        if not item["last_seen"] or ts >= int(item["last_seen"]):
            item["last_seen"] = ts
            item["last_status"] = row.get("status")
            item["last_route"] = row.get("requested_model") or row.get("route")
            item["last_provider"] = row.get("provider")
    api_rows = list(grouped.values())
    for item in api_rows:
        item["active_now"] = bool(item.get("last_seen") and now - int(item["last_seen"]) <= 3600)
        item["error_rate"] = round(item["errors"] / item["requests"], 4) if item["requests"] else 0.0

    recent_logins = sorted(dashboard_rows, key=lambda r: int(r.get("ts") or 0), reverse=True)[:100]
    rows = sorted(api_rows, key=lambda r: int(r.get("last_seen") or 0), reverse=True)
    return {
        "rows": rows,
        "recent_dashboard_logins": recent_logins,
        "generated_at": now,
        "active_window_s": 3600,
        "privacy": {
            "admin_only": True,
            "raw_api_keys_exposed": False,
            "full_key_hashes_exposed": False,
            "remote_addresses_masked": True,
            "user_agents_hashed": True,
        },
    }

@app.on_event("startup")
async def startup() -> None:
    global _client, _probe_task
    _client = httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0))
    if SYNTHETIC_PROBES_ENABLED and ROUTE_HEALTH_ROUTES:
        _probe_task = asyncio.create_task(_synthetic_probe_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    global _probe_task
    if _probe_task:
        _probe_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _probe_task
        _probe_task = None
    if _client:
        await _client.aclose()


@app.get("/healthz")
async def healthz() -> Response:
    assert _client is not None
    try:
        r = await _client.get(f"{UPSTREAM}/healthz", timeout=5.0)
        return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as exc:
        return JSONResponse(status_code=502, content={"ok": False, "error": str(exc)})


@app.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/dashboard")
@app.get("/dashboard/")
@app.get("/dashboard/provider-keys")
@app.get("/dashboard/provider-keys/")
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_dashboard_html())



@app.post("/dashboard/api/login")
@app.post("/dashboard/login")
async def dashboard_login(request: Request) -> Response:
    try:
        data = await request.json()
    except Exception:
        data = {}
    password = str(data.get("password", "")) if isinstance(data, dict) else ""
    api_key = str(data.get("api_key", "")) if isinstance(data, dict) else ""
    if api_key.strip():
        auth = _caller_auth(api_key.strip())
        if not auth.get("ok"):
            return JSONResponse(status_code=401, content={"error": {"message": "invalid or inactive API key", "type": "auth_error", "code": auth.get("error_code") or "consumer_dashboard_login"}})
        consumer = str(auth.get("caller") or "")
        digest = str(auth.get("digest") or "")
        _record_dashboard_login(request=request, role="consumer", user=consumer, consumer=consumer, key_sha256=digest)
        resp = JSONResponse(content={"ok": True, "role": "consumer", "user": consumer, "consumer": consumer, "max_age": DASHBOARD_COOKIE_MAX_AGE})
        resp.set_cookie(
            DASHBOARD_COOKIE_NAME,
            _make_dashboard_session(consumer, role="consumer", consumer=consumer, key_sha256=digest),
            max_age=DASHBOARD_COOKIE_MAX_AGE,
            httponly=True,
            secure=True,
            samesite="lax",
            path=DASHBOARD_COOKIE_PATH,
        )
        return resp
    if not _dashboard_password_ok(password):
        return JSONResponse(status_code=401, content={"error": {"message": "invalid dashboard password", "type": "auth_error", "code": "dashboard_login"}})
    _record_dashboard_login(request=request, role="admin", user="admin")
    resp = JSONResponse(content={"ok": True, "role": "admin", "user": "admin", "max_age": DASHBOARD_COOKIE_MAX_AGE})
    resp.set_cookie(
        DASHBOARD_COOKIE_NAME,
        _make_dashboard_session("admin"),
        max_age=DASHBOARD_COOKIE_MAX_AGE,
        httponly=True,
        secure=True,
        samesite="lax",
        path=DASHBOARD_COOKIE_PATH,
    )
    return resp


@app.post("/dashboard/api/logout")
@app.post("/dashboard/logout")
async def dashboard_logout() -> Response:
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie(DASHBOARD_COOKIE_NAME, path=DASHBOARD_COOKIE_PATH)
    return resp



def _load_policy_config() -> dict[str, Any]:
    """Load the Lua policy catalog for dashboard display only; never returns secrets."""
    from lupa import LuaRuntime
    from llm_router_host import _to_py

    cfg_path = Path(DASHBOARD_POLICY_CONFIG_PATH)
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    lua = LuaRuntime(unpack_returned_tuples=True)
    core_dir = (Path.cwd() / "core").resolve()
    lua.globals()["__policy_config_path"] = str(cfg_path.resolve())
    lua.globals()["__policy_core_dir"] = str(core_dir)
    lua.execute("package.path = __policy_core_dir .. '/?.lua;' .. __policy_core_dir .. '/?/init.lua;' .. package.path")
    cfg = _to_py(lua.eval("dofile(__policy_config_path)")) or {}
    return _merge_provider_overlay(cfg)


def _merge_provider_overlay(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fold operator-added providers (providers.local.json) into the parsed
    catalog so the dashboard lists them like hand-configured ones."""
    try:
        from provider_overlay import load_overlay
        overlay = load_overlay()
    except Exception:
        return cfg
    providers = cfg.setdefault("providers", {})
    models = cfg.setdefault("models", {})
    for pid, entry in (overlay.get("providers") or {}).items():
        if pid in providers or not isinstance(entry, dict):
            continue
        providers[pid] = {k: v for k, v in entry.items()
                          if k not in ("served_models", "added_at")}
        providers[pid].setdefault("api_kind", "openai_compatible")
        providers[pid].setdefault("discovery", "static")
        for sm in entry.get("served_models") or []:
            model = models.get((sm or {}).get("family"))
            if not isinstance(model, dict):
                continue
            served = model.setdefault("served_by", [])
            if not any(s.get("provider") == pid for s in served if isinstance(s, dict)):
                row = {"provider": pid}
                if sm.get("provider_model_id"):
                    row["provider_model_id"] = sm["provider_model_id"]
                served.append(row)
    return cfg


def _policy_file_profile_names() -> list[str]:
    policy_dir = Path(DASHBOARD_POLICY_DIR)
    if not policy_dir.is_absolute():
        policy_dir = Path.cwd() / policy_dir
    names = {path.stem for path in policy_dir.glob("*.lua")}
    preferred = [name for name in ("edge", "medium", "dummy") if name in names]
    return preferred + sorted(names - set(preferred))


def _resolved_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _ranked_models_for_profile(profile_name: str) -> list[dict[str, Any]]:
    """Use the router itself to derive the profile's candidate population.

    The policy files are functional DSL sentences, not static `{quality_min=...}`
    tables. Deriving diagram rows from router.rank keeps the dashboard faithful to
    config.live.lua + metrics.live.lua + policies/*.lua.
    """
    from llm_router_host import LLMRouterHost

    cfg_path = _resolved_path(DASHBOARD_POLICY_CONFIG_PATH)
    metrics_path = _resolved_path(DASHBOARD_POLICY_METRICS_PATH)
    host = LLMRouterHost(
        router_path=Path.cwd() / "core" / "router.lua",
        config_path=cfg_path,
        metrics_path=metrics_path if metrics_path.exists() else None,
    )
    host.init()
    ranked, _rejected = host.rank({"profile": profile_name, "requirements": {"context": ROUTER_CONTEXT_LENGTH}})

    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for idx, row in enumerate(ranked, start=1):
        candidate = row.get("candidate") or {}
        family = candidate.get("model_family") or row.get("model_family")
        if not family:
            continue
        if family not in grouped:
            grouped[family] = {
                "name": family,
                "quality": candidate.get("quality_hint"),
                "capabilities": candidate.get("capabilities") or {},
                "served_by": [],
            }
            order.append(family)
        grouped[family]["served_by"].append({
            "order": len(grouped[family]["served_by"]) + 1,
            "rank": idx,
            "provider": candidate.get("provider_id"),
            "provider_model_id": candidate.get("served_model_id"),
            "provider_tier": candidate.get("tier"),
            "score": row.get("score"),
        })
    return [grouped[name] for name in order]


PROVIDER_HEALTH_WINDOW_S = int(os.getenv("PROVIDER_HEALTH_WINDOW_S", "900"))


async def _fetch_router_runtime() -> dict[str, Any] | None:
    """Live breaker/disabled/EMA state from the router's /x/runtime.
    Returns None (→ health falls back to local signals) on any failure."""
    if _client is None:
        return None
    try:
        r = await _client.get(f"{UPSTREAM}/x/runtime", timeout=3.0)
        if r.status_code == 200 and isinstance(r.json(), dict):
            return r.json()
    except Exception:
        return None
    return None


async def _fetch_live_market() -> dict[str, Any] | None:
    """Full price book (per-family sellers + perf) from the router's
    /x/market. None on any failure — the Market tab shows the error."""
    if _client is None:
        return None
    try:
        r = await _client.get(f"{UPSTREAM}/x/market", timeout=3.0)
        if r.status_code == 200 and isinstance(r.json(), dict):
            return r.json()
    except Exception:
        return None
    return None


def _recent_provider_attempts(window_s: int = PROVIDER_HEALTH_WINDOW_S) -> dict[str, dict[str, Any]]:
    """Newest attempt outcome per provider from recent decision traces.
    _stats["recent"] is newest-first; within a row, decision_path is
    oldest-first, so scan it reversed and keep the first hit per provider."""
    cutoff = time.time() - window_s
    out: dict[str, dict[str, Any]] = {}
    with _stats_lock:
        rows = list(_stats["recent"])
    for row in rows:
        if not isinstance(row, dict) or row.get("event") != "request":
            continue
        ts = row.get("ts") or 0
        if ts < cutoff:
            continue
        trace = row.get("decision_trace") if isinstance(row.get("decision_trace"), dict) else None
        for e in reversed((trace or {}).get("decision_path") or []):
            if not isinstance(e, dict) or e.get("event") != "attempted":
                continue
            pid = e.get("provider_id")
            if not pid or pid in out:
                continue
            out[pid] = {
                "ok": not e.get("error_kind"),
                "error_kind": e.get("error_kind"),
                "http_status": e.get("http_status"),
                "ts": ts,
            }
        pid = row.get("provider")
        if pid and pid not in out and 200 <= int(row.get("status") or 0) < 300:
            out[pid] = {"ok": True, "error_kind": None, "http_status": None, "ts": ts}
    return out


def _runway_for(provider: dict[str, Any], balance: dict[str, Any] | None) -> str | None:
    # Runway thresholds are operator-tunable from the dashboard Config tab.
    if not balance or balance.get("value") is None:
        return None
    kind, value = balance.get("kind"), float(balance["value"])
    if kind == "deposits_usdc":
        cap = provider.get("market_price_cap") or {}
        if not (float(cap.get("input", 0) or 0) or float(cap.get("output", 0) or 0)):
            return None  # free buyer pays nobody; deposits are irrelevant
        return "empty" if value <= settings.get("antseed.runway_deposits_empty_usdc") \
            else "low" if value < settings.get("antseed.runway_deposits_low_usdc") else "ok"
    if kind == "credits_usd":
        return "empty" if value <= settings.get("openrouter.runway_credits_empty_usd") \
            else "low" if value < settings.get("openrouter.runway_credits_low_usd") else "ok"
    if kind == "quota_window":
        return "low" if value > settings.get("codex.runway_quota_low_fraction") else "ok"
    return None


async def _attach_antseed_wallet(provider_keys: dict | None) -> None:
    """Enrich the AntSeed provider-credentials row with its buyer hot-wallet
    (address / deposits / runway), read LIVE from the market so the address
    always reflects the current identity. Lets the Provider keys tab show where
    to top up instead of a useless 'no auth required' — AntSeed needs no API key,
    it needs a funded wallet."""
    if not provider_keys or not provider_keys.get("rows"):
        return
    market = await _fetch_live_market()
    w = (market or {}).get("wallet")
    if not w or not w.get("address"):
        return
    if w.get("deposits_available") is not None:
        try:
            reserved = float(w.get("deposits_reserved") or 0)
        except (TypeError, ValueError):
            reserved = 0.0
        # `reserved` is the buyer's own deposit locked in active payment channels
        # (in use, returns to available as channels settle) — NOT spent and NOT
        # lost. Run the runway off the TOTAL so a fully-reserved wallet reads as
        # funded-in-use, not "empty · top up".
        total = float(w["deposits_available"]) + reserved
        w = {**w, "runway": _runway_for(
            {"market_price_cap": {"input": 1, "output": 1}},
            {"kind": "deposits_usdc", "value": total})}
    for row in provider_keys["rows"]:
        if row.get("provider") == w.get("provider"):
            row["wallet"] = w
            break


def _provider_health_for(provider: dict[str, Any], name: str,
                         runtime: dict[str, Any] | None,
                         recent: dict[str, Any]) -> dict[str, Any]:
    """Classify one provider: disconnected | failing | ok | idle.
    Carries the typed balance and a runway verdict; a paid provider with an
    empty balance is failing BEFORE any request hits the wall."""
    rt = runtime or {}
    balance = (rt.get("balances") or {}).get(name)
    runway = _runway_for(provider, balance)

    def _result(state: str, reason: str | None) -> dict[str, Any]:
        return {"state": state, "reason": reason, "balance": balance, "runway": runway}

    auth = provider.get("auth")
    env_name = provider.get("auth_env") or (auth.get("env") if isinstance(auth, dict) else None)
    if env_name and not os.environ.get(env_name):
        return _result("disconnected", f"{env_name} not set")

    disabled = (rt.get("disabled_providers") or {}).get(name)
    if disabled:
        return _result("failing", f"disabled: {disabled}")

    last = recent.get(name)

    def _last_err() -> str | None:
        if not last or last.get("ok") or not last.get("error_kind"):
            return None
        kind = last["error_kind"]
        return f"{kind}({last['http_status']})" if last.get("http_status") else str(kind)

    breaker = (rt.get("circuit_breakers") or {}).get(name) or {}
    if breaker.get("open"):
        reason = "breaker open"
        err = _last_err()
        return _result("failing", f"{reason} · {err}" if err else reason)

    if runway == "empty":
        unit = "USDC" if (balance or {}).get("kind") == "deposits_usdc" else "USD"
        return _result("failing", f"balance empty ({balance['value']} {unit})")

    if last:
        if last.get("ok"):
            return _result("ok", None)
        return _result("failing", _last_err() or "recent error")

    return _result("idle", None)


def _group_live_rank(rows: list[dict]) -> list[dict[str, Any]]:
    """Group /x/rank rows into the family-shaped structure the policy
    diagrams render (same shape as _ranked_models_for_profile, plus prices)."""
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for idx, row in enumerate(rows, start=1):
        family = row.get("model_family")
        if not family:
            continue
        if family not in grouped:
            grouped[family] = {"name": family, "quality": row.get("quality"),
                                "capabilities": {}, "served_by": []}
            order.append(family)
        grouped[family]["served_by"].append({
            "order": len(grouped[family]["served_by"]) + 1,
            "rank": idx,
            "provider": row.get("provider"),
            "provider_model_id": row.get("served_model_id"),
            "provider_tier": row.get("tier"),
            "score": row.get("score"),
            "price_in": row.get("price_in"),
            "price_out": row.get("price_out"),
            "discovery": row.get("discovery"),
        })
    return [grouped[name] for name in order]


async def _fetch_live_ranks(profile_names: list[str]) -> dict[str, list[dict]]:
    """Rank rows per profile from the live router; missing/failed -> absent
    (the snapshot falls back to its local reconstruction, labeled)."""
    out: dict[str, list[dict]] = {}
    if _client is None:
        return out
    for name in profile_names:
        try:
            r = await _client.get(f"{UPSTREAM}/x/rank", params={"profile": name}, timeout=3.0)
            if r.status_code == 200 and isinstance(r.json(), dict):
                out[name] = r.json().get("ranked") or []
        except Exception:
            continue
    return out


def _policy_catalog_snapshot(runtime: dict[str, Any] | None = None,
                             live_ranks: dict[str, list[dict]] | None = None) -> dict[str, Any]:
    cfg = _load_policy_config()
    providers = cfg.get("providers") or {}
    models = cfg.get("models") or {}
    profiles = cfg.get("profiles") or {}
    retry_policies = cfg.get("retry_policies") or {}

    recent_attempts = _recent_provider_attempts()
    provider_rows = []
    for name, provider in sorted(providers.items()):
        auth = provider.get("auth")
        auth_kind = auth.get("kind") if isinstance(auth, dict) else ("configured" if provider.get("auth_env") else "none")
        provider_rows.append({
            "name": name,
            "tier": provider.get("tier"),
            "api_kind": provider.get("api_kind"),
            "base_url": provider.get("base_url"),
            "auth": auth_kind,
            "notes": provider.get("notes"),
            "health": _provider_health_for(provider, name, runtime, recent_attempts),
        })

    model_rows = []
    for name, model in sorted(models.items()):
        served_by = []
        for idx, candidate in enumerate(model.get("served_by") or [], start=1):
            provider_id = candidate.get("provider")
            served_by.append({
                "order": idx,
                "provider": provider_id,
                "provider_model_id": candidate.get("provider_model_id"),
                "provider_tier": (providers.get(provider_id) or {}).get("tier"),
            })
        model_rows.append({
            "name": name,
            "quality": model.get("static_quality_hint"),
            "capabilities": model.get("capabilities") or {},
            "served_by": served_by,
        })

    # All config profiles (declarative or file-based). With tiers removed this is
    # just the `default` fallback; per-call policies arrive as policy_ir.
    file_names = _policy_file_profile_names()
    policy_profile_names = [n for n in file_names if n in profiles] + \
        [n for n in profiles if n not in file_names]
    profile_rows = []
    for name in policy_profile_names:
        profile = profiles.get(name) or {}
        live = (live_ranks or {}).get(name)
        if live:
            ranked_models = _group_live_rank(live)
            rank_source = "router"
            note = "Live ranking from the router (/x/rank): live prices, breakers, marketplace offers."
        else:
            ranked_models = _ranked_models_for_profile(name)
            rank_source = "reconstructed"
            note = "Derived from router.rank using config.live.lua and metrics.live.lua."
        profile_rows.append({
            "name": name,
            "filter": "policies/%s.lua" % name,
            "weights": {},
            "retry_policy": profile.get("retry_policy"),
            "models": ranked_models,
            "candidate_count": sum(len(model.get("served_by") or []) for model in ranked_models),
            "selection_note": note,
            "rank_source": rank_source,
        })

    cfg_path = _resolved_path(DASHBOARD_POLICY_CONFIG_PATH)
    metrics_path = _resolved_path(DASHBOARD_POLICY_METRICS_PATH)
    return {
        "providers": provider_rows,
        "models": model_rows,
        "profiles": profile_rows,
        "profile_source": "policy_files",
        "policy_files": ["policies/%s.lua" % name for name in policy_profile_names if name in file_names],
        "retry_policies": retry_policies,
        "source": str(cfg_path.resolve()),
        "metrics_source": str(metrics_path.resolve()) if metrics_path.exists() else None,
        "generated_at": int(time.time()),
    }

@app.get("/dashboard/api/stats")
async def dashboard_stats(request: Request) -> Response:
    ctx = _require_dashboard_context(request)
    if not ctx:
        return JSONResponse(status_code=401, content={"error": {"message": "unauthorized dashboard caller", "type": "auth_error", "code": "dashboard_auth"}})
    caller = str(ctx.get("viewer"))
    assert _client is not None
    upstream_health: dict[str, Any]
    upstream_status = 0
    try:
        r = await _client.get(f"{UPSTREAM}/healthz", timeout=5.0)
        upstream_status = r.status_code
        upstream_health = r.json()
    except Exception as exc:
        upstream_health = {"ok": False, "error": str(exc)}
    requested_consumer = (request.query_params.get("consumer") or "").strip() or None
    consumer = str(ctx.get("consumer") or "").strip() or requested_consumer
    key_sha256 = str(ctx.get("key_sha256") or "").strip() or None
    role = str(ctx.get("role") or "admin")
    snap = _stats_snapshot(viewer=caller, upstream_status=upstream_status, upstream_health=upstream_health, consumer=consumer, timeframe=_dashboard_timeframe(request.query_params.get("timeframe") or request.query_params.get("window")), key_sha256=key_sha256, viewer_role=role, provider=request.query_params.get("provider"), model=request.query_params.get("model"))
    await _attach_antseed_wallet(snap.get("provider_keys"))
    return JSONResponse(content=snap)


async def _dashboard_full_snapshot(request: Request) -> Response:
    """Full sanitized dashboard state for quick external evaluation.

    Combines the live in-memory usage/route snapshot with the policy catalog.
    This intentionally does not expose raw API keys, env values, bearer tokens,
    or provider auth material.
    """
    ctx = _require_dashboard_context(request)
    if not ctx:
        return JSONResponse(status_code=401, content={"error": {"message": "unauthorized dashboard caller", "type": "auth_error", "code": "dashboard_auth"}})
    caller = str(ctx.get("viewer"))
    assert _client is not None
    upstream_health: dict[str, Any]
    upstream_status = 0
    try:
        r = await _client.get(f"{UPSTREAM}/healthz", timeout=5.0)
        upstream_status = r.status_code
        upstream_health = r.json()
    except Exception as exc:
        upstream_health = {"ok": False, "error": str(exc)}
    requested_consumer = (request.query_params.get("consumer") or "").strip() or None
    consumer = str(ctx.get("consumer") or "").strip() or requested_consumer
    key_sha256 = str(ctx.get("key_sha256") or "").strip() or None
    role = str(ctx.get("role") or "admin")
    stats = _stats_snapshot(viewer=caller, upstream_status=upstream_status, upstream_health=upstream_health, consumer=consumer, timeframe=_dashboard_timeframe(request.query_params.get("timeframe") or request.query_params.get("window")), key_sha256=key_sha256, viewer_role=role)
    try:
        policies = _policy_catalog_snapshot() if role == "admin" else {"providers": [], "models": [], "profiles": [], "retry_policies": {}, "consumer_visible": False, "generated_at": int(time.time())}
        policy_error = None
    except Exception as exc:
        policies = {"providers": [], "models": [], "profiles": [], "retry_policies": {}, "generated_at": int(time.time())}
        policy_error = {"message": str(exc), "type": "policy_catalog_error", "code": "policy_catalog"}
    content = {
        "schema_version": 1,
        "kind": "router_dashboard_full_snapshot",
        "generated_at": int(time.time()),
        "viewer": caller,
        "viewer_role": role,
        "selected_consumer": stats.get("selected_consumer"),
        "selected_key_sha256_prefix": stats.get("selected_key_sha256_prefix"),
        "sections": {
            "overview": {
                "uptime_s": stats.get("uptime_s"),
                "rate_limit": stats.get("rate_limit"),
                "upstream": stats.get("upstream"),
                "totals": stats.get("totals"),
                "health_summary": stats.get("health_summary"),
            },
            "consumers": {
                "consumers": stats.get("consumers"),
                "keys": stats.get("keys"),
                "by_caller": stats.get("by_caller"),
            },
            "routing": {
                "route_health": stats.get("route_health"),
                "health_summary": stats.get("health_summary"),
                "by_route": stats.get("by_route"),
                "by_provider": stats.get("by_provider"),
                "by_model_family": stats.get("by_model_family"),
                "by_served_model": stats.get("by_served_model"),
                "by_status": stats.get("by_status"),
            },
            "policies": policies,
            "activity": {
                "recent": stats.get("recent"),
            },
            "logins": stats.get("logins"),
            "provider_keys": stats.get("provider_keys"),
        },
        "raw": {
            "stats": stats,
            "policies": policies,
        },
        "errors": [policy_error] if policy_error else [],
        "security": {
            "sanitized": True,
            "raw_api_keys_exposed": False,
            "provider_credentials_exposed": False,
        },
    }
    return JSONResponse(content=content)


@app.get("/dashboard/api/full")
async def dashboard_full(request: Request) -> Response:
    return await _dashboard_full_snapshot(request)


@app.get("/dashboard/api/evaluate")
async def dashboard_evaluate(request: Request) -> Response:
    return await _dashboard_full_snapshot(request)



@app.get("/dashboard/api/policies")
async def dashboard_policies(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    caller = str(ctx.get("viewer"))
    try:
        runtime = await _fetch_router_runtime()
        live_ranks = await _fetch_live_ranks(_policy_file_profile_names())
        return JSONResponse(content=_policy_catalog_snapshot(runtime=runtime, live_ranks=live_ranks))
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": "policy_catalog_error", "code": "policy_catalog"}})

@app.get("/dashboard/api/market")
async def dashboard_market(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    data = await _fetch_live_market()
    if data is None:
        return JSONResponse(status_code=502, content={"error": {
            "message": "router /x/market unavailable",
            "type": "market_error", "code": "market"}})
    return JSONResponse(content=data)


SKILL_PATH = Path(__file__).resolve().parent / "SKILL.md"
SKILL_MARKER = "<!-- LIVE_CATALOG_TABLE -->"
FIELDS_MARKER = "<!-- FIELD_VOCABULARY -->"


def _catalog_table_markdown(market: dict | None) -> str:
    """The family-level catalog as a markdown table — the unit a Σ_pol policy
    targets (gate/score on families and fields, not individual sellers).
    Benchmarks render as 0–100 with catalog rank; prices are the cheapest
    seller per family."""
    families = (market or {}).get("families") or []
    if not families:
        return "_(live catalog unavailable — target the field vocabulary above " \
               "and confirm families with `POST /x/rank`)_"

    def bench(m: dict, key: str) -> str:
        v = m.get(key)
        if not isinstance(v, (int, float)):
            return "—"
        cell = f"{round(v * 100)}"
        rank = m.get(key + "_rank")
        if isinstance(rank, (int, float)) and rank < 1e8:
            cell += f" (#{int(rank)})"
        return cell

    def price(rows: list, field: str) -> str:
        vals = [r.get(field) for r in rows if isinstance(r.get(field), (int, float))]
        return f"{min(vals):g}" if vals else "—"

    cap_labels = [("cap_tools", "tools"), ("cap_reasoning", "reason"),
                  ("cap_structured_outputs", "struct"), ("cap_seed", "seed"),
                  ("cap_logprobs", "logprobs")]
    # (sigma-pol/v2) no "Quality" column: quality/quality_hint were removed as
    # observable fields (they meant nothing — a hand-assigned static hint), so a
    # policy can't gate/score on them. The real model-goodness signals are the
    # benchmarks below.
    out = ["| Family | Intel | Coding | Agentic | Arena | Modalities | Caps | $in/Mtok | $out/Mtok | Sellers |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for f in families:
        m = f.get("meta") or {}
        rows = f.get("rows") or []
        mods = "/".join(k for k in ("image", "audio", "file", "video")
                        if m.get("in_" + k)) or "—"
        caps = "/".join(label for key, label in cap_labels if m.get(key)) or "—"
        out.append(
            f"| `{f.get('family')}` "
            f"| {bench(m, 'bench_intelligence')} | {bench(m, 'bench_coding')} "
            f"| {bench(m, 'bench_agentic')} | {bench(m, 'bench_arena')} "
            f"| {mods} | {caps} | {price(rows, 'price_in')} "
            f"| {price(rows, 'price_out')} | {f.get('sellers_total', '—')} |")
    return "\n".join(out)


def _field_vocabulary_markdown(fields: list | None) -> str:
    """The observable field vocabulary as a markdown table, derived live from
    the core's own schema (`/x/fields` → `host.field_schema()`). This is the
    one authoritative list of fields a policy may gate/score over: keeping it
    out of the static prose means a field appended to the core (the anti-telos
    says expressiveness grows through the field vocabulary, not ad-hoc ops)
    shows up here automatically instead of silently diverging."""
    if not fields:
        return "_(field schema unavailable — the core defines the vocabulary; " \
               "fetch it from `GET /x/fields` on this host)_"
    core = sorted((f for f in fields if f.get("core")), key=lambda f: f.get("name", ""))
    ext = sorted((f for f in fields if not f.get("core")), key=lambda f: f.get("name", ""))
    out = ["| Field | Sort | Scope | Group |", "|---|---|---|---|"]
    for f in core + ext:
        scope = "core" if f.get("core") else "host"
        out.append(f"| `{f.get('name')}` | {f.get('sort', '—')} | {scope} "
                   f"| {f.get('group', '—')} |")
    return "\n".join(out)


def _render_skill(market: dict | None, fields: list | None = None) -> str:
    """The committed SKILL.md with the live catalog table AND the live field
    vocabulary injected at their markers — a self-contained doc to load into
    any assistant. The catalog and the vocabulary are both derived from the
    host/core at download time so the guide cannot drift from what the host
    actually serves."""
    try:
        text = SKILL_PATH.read_text()
    except OSError:
        text = ("# SKILL.md not found on host\n\n"
                "The authoring guide file is missing.\n\n"
                + FIELDS_MARKER + "\n\n" + SKILL_MARKER)
    vocab = ("The fields below are read live from this host's core schema "
             "(`GET /x/fields`); `core` fields exist on every conforming host, "
             "`host` fields are this host's registered extensions. Defaults "
             "when a field is absent are conservative — see *Rules* below.\n\n"
             + _field_vocabulary_markdown(fields))
    if FIELDS_MARKER in text:
        text = text.replace(FIELDS_MARKER, vocab, 1)
    catalog = ("## Live catalog (this host)\n\n"
               "Families this host serves right now — gate/score policies on "
               "these. Benchmarks are 0–100 with catalog rank in parentheses "
               "(1 = best); prices are the cheapest seller per family in "
               "USD/Mtok.\n\n" + _catalog_table_markdown(market))
    if SKILL_MARKER in text:
        return text.replace(SKILL_MARKER, catalog, 1)
    return text.rstrip() + "\n\n" + catalog + "\n"


async def _fetch_live_fields() -> list | None:
    """The observable field vocabulary from the router's /x/fields
    (core + this host's extensions). None on any failure — the guide then
    points the reader at /x/fields instead of inventing a list."""
    if _client is None:
        return None
    try:
        r = await _client.get(f"{UPSTREAM}/x/fields", timeout=3.0)
        if r.status_code == 200 and isinstance(r.json(), dict):
            return r.json().get("fields")
    except Exception:
        return None
    return None


@app.get("/dashboard/api/skill")
async def dashboard_skill(request: Request) -> Response:
    """Download a self-contained SKILL.md: the Σ_pol/Σ_flow authoring guide with
    this host's live catalog table and field vocabulary baked in. Load it into
    any assistant to generate policies that target models this host serves."""
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    text = _render_skill(await _fetch_live_market(), await _fetch_live_fields())
    return Response(content=text, media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=SKILL.md"})


async def _router_post_json(path: str, payload: dict) -> Response:
    """Admin-dashboard passthrough to the shim's policy endpoints. The shim
    (and behind it the core) is the validator; this just forwards bytes and
    status — including the 400s the builder UI needs to render verbatim."""
    if _client is None:
        return JSONResponse(status_code=503, content={"error": {
            "message": "router client not ready", "type": "router_error",
            "code": "upstream"}})
    try:
        r = await _client.post(f"{UPSTREAM}{path}", json=payload, timeout=10.0)
        return Response(content=r.content, status_code=r.status_code,
                        media_type="application/json")
    except Exception:
        return JSONResponse(status_code=502, content={"error": {
            "message": f"router {path} unavailable", "type": "router_error",
            "code": "upstream"}})


@app.post("/dashboard/api/policy/build")
async def dashboard_policy_build(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    return await _router_post_json("/x/policy/build", await request.json())


@app.post("/dashboard/api/policy/preview")
async def dashboard_policy_preview(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    return await _router_post_json("/x/rank", await request.json())


@app.post("/dashboard/api/policy/normalize")
async def dashboard_policy_normalize(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    return await _router_post_json("/x/policy/normalize", await request.json())


@app.get("/dashboard/api/fields")
async def dashboard_fields(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    try:
        r = await _client.get(f"{UPSTREAM}/x/fields", timeout=5.0)
        return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=502, content={"error": {
            "message": str(exc), "type": "router_error", "code": "fields"}})


@app.post("/dashboard/api/policy/test")
async def dashboard_policy_test(request: Request) -> Response:
    """Run the builder's policy live: execute a chat request with this term and
    a prompt against the router, and record it as an Activity event so its full
    fallback trace + cost show up under Activity."""
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    body = await request.json()
    policy_ir = body.get("policy_ir")
    prompt = str(body.get("prompt") or "").strip()
    if not isinstance(policy_ir, list) or not prompt:
        return JSONResponse(status_code=400, content={"error": {
            "message": "policy_ir (array) and a non-empty prompt are required",
            "type": "invalid_request_error", "code": "test_call"}})
    payload = {"messages": [{"role": "user", "content": prompt}],
               "policy_ir": policy_ir, "max_tokens": int(body.get("max_tokens") or 64)}
    started = time.perf_counter()
    status = 502
    provider = model_family = served_model_id = text = None
    decision_trace = None
    tokens_in = tokens_out = tokens_total = 0
    cost_usd = None
    error_type = error_code = error_message = None
    try:
        r = await _client.post(f"{UPSTREAM}/v1/chat/completions", json=payload,
                               headers={"x-llm-router-caller": "dashboard-test"}, timeout=45.0)
        status = r.status_code
        data = r.json()
        provider, model_family, served_model_id, decision_trace = _extract_router_metadata(data)
        xr = data.get("x_router") if isinstance(data, dict) else None
        if isinstance(xr, dict) and isinstance(xr.get("cost_usd"), (int, float)):
            cost_usd = float(xr["cost_usd"])
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            tokens_in = int(usage.get("prompt_tokens") or 0)
            tokens_out = int(usage.get("completion_tokens") or 0)
            tokens_total = int(usage.get("total_tokens") or (tokens_in + tokens_out))
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            text = ((choices[0] or {}).get("message") or {}).get("content")
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            error_type, error_code, error_message = err.get("type"), err.get("code"), err.get("message")
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)[:200]
        decision_trace = {"attempts": [{"error_kind": "test_error", "message": error_message}]}
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    # record as Activity (in-memory only: key_sha256=None keeps it out of
    # persistent per-consumer billing history)
    _record_request(caller="dashboard-test", method="POST", path="/x/policy/test",
                    status=status, latency_ms=latency_ms, provider=provider,
                    model_family=model_family, served_model_id=served_model_id,
                    requested_model="policy_ir", tokens_in=tokens_in, tokens_out=tokens_out,
                    tokens_total=tokens_total, cost_usd=cost_usd, decision_trace=decision_trace,
                    error_type=error_type, error_code=error_code,
                    error_message=_safe_dashboard_message(error_message, limit=300) if error_message else None,
                    key_sha256=None)
    return JSONResponse(content={
        "ok": 200 <= status < 300, "status": status, "text": text,
        "provider": provider, "model_family": model_family, "served_model_id": served_model_id,
        "cost_usd": cost_usd, "tokens_total": tokens_total, "tokens_in": tokens_in,
        "tokens_out": tokens_out, "latency_ms": latency_ms, "decision_trace": decision_trace,
        "error": error_message})


@app.post("/dashboard/api/flow/normalize")
async def dashboard_flow_normalize(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    return await _router_post_json("/x/flow/normalize", await request.json())


@app.post("/dashboard/api/flow/test")
async def dashboard_flow_test(request: Request) -> Response:
    """Run the Flow Builder's flow live: execute a chat request with this
    flow_ir and a prompt, and record it as an Activity event so its per-node
    trace + cost show up under Activity (the flow twin of policy/test)."""
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    body = await request.json()
    flow_ir = body.get("flow_ir")
    prompt = str(body.get("prompt") or "").strip()
    if not isinstance(flow_ir, list) or not prompt:
        return JSONResponse(status_code=400, content={"error": {
            "message": "flow_ir (array) and a non-empty prompt are required",
            "type": "invalid_request_error", "code": "test_call"}})
    payload = {"messages": [{"role": "user", "content": prompt}],
               "flow_ir": flow_ir, "max_tokens": int(body.get("max_tokens") or 64)}
    started = time.perf_counter()
    status = 502
    provider = model_family = served_model_id = text = None
    decision_trace = None
    tokens_in = tokens_out = tokens_total = 0
    cost_usd = None
    error_type = error_code = error_message = None
    try:
        r = await _client.post(f"{UPSTREAM}/v1/chat/completions", json=payload,
                               headers={"x-llm-router-caller": "dashboard-test"}, timeout=90.0)
        status = r.status_code
        data = r.json()
        provider, model_family, served_model_id, decision_trace = _extract_router_metadata(data)
        xr = data.get("x_router") if isinstance(data, dict) else None
        if isinstance(xr, dict) and isinstance(xr.get("cost_usd"), (int, float)):
            cost_usd = float(xr["cost_usd"])
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            tokens_in = int(usage.get("prompt_tokens") or 0)
            tokens_out = int(usage.get("completion_tokens") or 0)
            tokens_total = int(usage.get("total_tokens") or (tokens_in + tokens_out))
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            text = ((choices[0] or {}).get("message") or {}).get("content")
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            error_type, error_code, error_message = err.get("type"), err.get("code"), err.get("message")
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc)[:200]
        decision_trace = {"attempts": [{"error_kind": "test_error", "message": error_message}]}
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    _record_request(caller="dashboard-test", method="POST", path="/x/flow/test",
                    status=status, latency_ms=latency_ms, provider=provider,
                    model_family=model_family, served_model_id=served_model_id,
                    requested_model="flow_ir", tokens_in=tokens_in, tokens_out=tokens_out,
                    tokens_total=tokens_total, cost_usd=cost_usd, decision_trace=decision_trace,
                    error_type=error_type, error_code=error_code,
                    error_message=_safe_dashboard_message(error_message, limit=300) if error_message else None,
                    key_sha256=None)
    return JSONResponse(content={
        "ok": 200 <= status < 300, "status": status, "text": text,
        "provider": provider, "model_family": model_family, "served_model_id": served_model_id,
        "cost_usd": cost_usd, "tokens_total": tokens_total, "tokens_in": tokens_in,
        "tokens_out": tokens_out, "latency_ms": latency_ms, "decision_trace": decision_trace,
        "error": error_message})


@app.get("/dashboard/api/keys")
async def dashboard_keys(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    caller = str(ctx.get("viewer"))
    return JSONResponse(content={"keys": _consumer_key_rows(), "consumers": _consumers()})


@app.get("/dashboard/api/logins")
async def dashboard_logins(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    timeframe = _dashboard_timeframe(request.query_params.get("timeframe") or request.query_params.get("window"))
    consumer = (request.query_params.get("consumer") or "").strip() or None
    return JSONResponse(content=_login_connections_snapshot(timeframe=timeframe, consumer=consumer, viewer_role="admin"))


@app.get("/dashboard/api/provider-keys")
async def dashboard_provider_keys(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    timeframe = _dashboard_timeframe(request.query_params.get("timeframe") or request.query_params.get("window"))
    snap = _provider_credentials_snapshot(timeframe=timeframe, viewer_role="admin")
    await _attach_antseed_wallet(snap)
    return JSONResponse(content=snap)


@app.post("/dashboard/api/key-usage")
async def dashboard_key_usage(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    caller = str(ctx.get("viewer"))
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        api_key = str(data.get("api_key") or "").strip()
        if not api_key:
            raise ValueError("api_key is required")
        options = _usage_query_options(data)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error", "code": "invalid_key_usage_request"}})
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body", "type": "invalid_request_error", "code": "invalid_json"}})
    digest = hashlib.sha256(api_key.encode()).hexdigest()
    owner = CALLER_KEYS.get(api_key) or CALLER_KEY_HASHES.get(digest)
    with _stats_lock:
        has_usage = int(_stats["by_key_sha256"].get(digest, {}).get("requests") or 0) > 0
        owner = owner or _stats["key_owner"].get(digest)
    if not owner and not has_usage:
        rows, persistent = _usage_events_for_key(digest, owner)
        has_usage = bool(rows)
    if not owner and not has_usage:
        return JSONResponse(status_code=404, content={"error": {"message": "no configured key or recorded usage found for api_key", "type": "not_found", "code": "key_usage_not_found"}})
    return JSONResponse(content=_key_usage_snapshot(viewer=caller, key_sha256=digest, caller=owner, options=options))


@app.get("/v1/usage")
@app.get("/api/usage")
async def key_usage(request: Request) -> Response:
    token = _extract_token(request)
    auth = _caller_auth(token)
    if not auth.get("ok"):
        code = auth.get("error_code") or "caller_auth"
        status_code = 403 if code in {"caller_inactive", "caller_key_revoked", "caller_key_expired"} else 401
        messages = {
            "caller_auth": "unauthorized caller",
            "caller_inactive": "caller is inactive",
            "caller_key_revoked": "caller key is revoked",
            "caller_key_expired": "caller key is expired",
        }
        return JSONResponse(status_code=status_code, content={"error": {"message": messages.get(code, "caller not authorized"), "type": "auth_error", "code": code}})
    caller = str(auth.get("caller"))
    try:
        options = _usage_query_options(request.query_params)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error", "code": "invalid_usage_window"}})
    return JSONResponse(content=_key_usage_snapshot(viewer=f"consumer:{caller}", key_sha256=str(auth.get("digest")), caller=caller, options=options))


@app.post("/dashboard/api/consumers/{consumer}")
async def dashboard_update_consumer(consumer: str, request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    caller = str(ctx.get("viewer"))
    try:
        consumer = _safe_consumer_name(consumer)
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error", "code": "invalid_consumer"}})
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body", "type": "invalid_request_error", "code": "invalid_json"}})
    records = _issued_consumer_records()
    meta = records.get(consumer, _normalize_consumer_record(consumer, {}))
    if "status" in data:
        status = str(data.get("status") or "").strip().lower()
        if status not in {"active", "inactive"}:
            return JSONResponse(status_code=400, content={"error": {"message": "status must be active or inactive", "type": "invalid_request_error", "code": "invalid_status"}})
        meta["status"] = status
    if "allowed_routes" in data:
        meta["allowed_routes"] = _clean_route_list(data.get("allowed_routes"))
    if "rate_per_min" in data:
        meta["rate_per_min"] = _optional_int(data.get("rate_per_min"), min_value=1)
    if "burst" in data:
        meta["burst"] = _optional_int(data.get("burst"), min_value=1)
    meta["updated_at"] = int(time.time())
    records[consumer] = meta
    _write_issued_consumer_records(records)
    _log({"event": "dashboard_consumer_updated", "consumer": consumer, "viewer": caller, "status": meta.get("status")})
    return JSONResponse(content={"ok": True, "consumer": consumer, "settings": _normalize_consumer_record(consumer, meta)})


@app.post("/dashboard/api/keys/revoke")
async def dashboard_revoke_key(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    caller = str(ctx.get("viewer"))
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        consumer = _safe_consumer_name(str(data.get("consumer", "")))
        prefix = str(data.get("sha256_prefix") or "").strip().lower()
        if not re.fullmatch(r"[a-f0-9]{8,64}", prefix):
            raise ValueError("sha256_prefix must be 8-64 lowercase hex chars")
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error", "code": "invalid_key"}})
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body", "type": "invalid_request_error", "code": "invalid_json"}})
    records = _issued_consumer_records()
    meta = records.get(consumer, _normalize_consumer_record(consumer, {}))
    now = int(time.time())
    found = False
    for key in meta.get("keys") or []:
        if str(key.get("sha256_prefix") or "").startswith(prefix) or prefix.startswith(str(key.get("sha256_prefix") or "")):
            key["status"] = "revoked"
            key["revoked_at"] = now
            found = True
    new_hashes = {digest: owner for digest, owner in CALLER_KEY_HASHES.items() if not (owner == consumer and digest.startswith(prefix))}
    removed = len(CALLER_KEY_HASHES) - len(new_hashes)
    if removed:
        CALLER_KEY_HASHES.clear()
        CALLER_KEY_HASHES.update(new_hashes)
        _upsert_env_json(Path(DASHBOARD_KEY_ENV_PATH), "CALLER_KEYS_SHA256_JSON", new_hashes)
        found = True
    new_plaintext = {token: owner for token, owner in CALLER_KEYS.items() if not (owner == consumer and hashlib.sha256(token.encode()).hexdigest().startswith(prefix))}
    removed_plaintext = len(CALLER_KEYS) - len(new_plaintext)
    if removed_plaintext:
        CALLER_KEYS.clear()
        CALLER_KEYS.update(new_plaintext)
        _upsert_env_json(Path(DASHBOARD_KEY_ENV_PATH), "CALLER_KEYS_JSON", new_plaintext)
        found = True
    if not found:
        return JSONResponse(status_code=404, content={"error": {"message": "key prefix not found for consumer", "type": "not_found", "code": "key_not_found"}})
    meta["updated_at"] = now
    records[consumer] = meta
    _write_issued_consumer_records(records)
    _log({"event": "dashboard_key_revoked", "consumer": consumer, "viewer": caller, "sha256_prefix": prefix, "removed_hashes": removed, "removed_plaintext": removed_plaintext})
    return JSONResponse(content={"ok": True, "consumer": consumer, "sha256_prefix": prefix, "removed_hashes": removed, "removed_plaintext": removed_plaintext})


@app.post("/dashboard/api/provider-keys/add")
async def dashboard_add_provider(request: Request) -> Response:
    """Add a provider + its API key from the dashboard. Persists the key to
    .env.secrets (auth_env indirection) and the provider definition to
    providers.local.json, then hot-applies it via the router's /x/providers
    so it serves without a restart. Admin-only, like key reveal."""
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body", "type": "invalid_request", "code": "provider_add"}})
    pid = str(body.get("id") or "").strip().lower()
    auth_env = str(body.get("auth_env") or "").strip()
    key = str(body.get("key") or "").strip()
    entry = {
        "base_url": str(body.get("base_url") or "").strip(),
        "api_kind": "openai_compatible",
        "tier": str(body.get("tier") or "partner").strip() or "partner",
        "auth_env": auth_env,
        "served_models": [sm for sm in (body.get("served_models") or []) if isinstance(sm, dict)],
        "added_at": int(time.time()),
    }
    try:
        from provider_overlay import load_overlay, save_overlay, validate_entry
        catalog = _load_policy_config()   # includes existing overlay entries
        errors = validate_entry(pid, entry, catalog)
        if not key:
            errors.append("key is required")
        if errors:
            return JSONResponse(status_code=400, content={"error": {"message": "; ".join(errors), "type": "invalid_request", "code": "provider_add"}})
        _upsert_env_line(Path(DASHBOARD_KEY_ENV_PATH), auth_env, key)
        overlay = load_overlay()
        overlay.setdefault("providers", {})[pid] = entry
        save_overlay(overlay)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": "provider_add_error", "code": "provider_add"}})
    # hot-apply on the running router; on failure the provider still applies
    # at the next router restart (overlay + env are already persisted)
    applied_live = False
    apply_error = None
    try:
        if _client is not None:
            r = await _client.post(f"{UPSTREAM}/x/providers", json={
                "id": pid, **{k: v for k, v in entry.items() if k != "added_at"},
                "key": key}, timeout=10.0)
            applied_live = r.status_code == 200
            if not applied_live:
                apply_error = f"router /x/providers returned {r.status_code}"
    except Exception as exc:
        apply_error = str(exc)
    _log({"event": "dashboard_provider_added", "provider": pid, "viewer": caller,
          "auth_env": auth_env, "applied_live": applied_live})
    return JSONResponse(content={
        "ok": True, "provider": pid, "applied_live": applied_live,
        "note": None if applied_live else
        f"saved, but live apply failed ({apply_error}); the provider will load on the next router restart"})


@app.post("/dashboard/api/provider-keys/update")
async def dashboard_update_provider_key(request: Request) -> Response:
    """Replace the API key of an EXISTING provider (heurist/ionet/openrouter,
    or any operator-added one). Persists the new key to .env.secrets — the PVC
    source of truth that survives restarts — and hot-applies it via the
    router's /x/provider-key so it serves without a restart. Admin-only,
    like add/reveal."""
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body", "type": "invalid_request", "code": "provider_update"}})
    pid = str(body.get("provider") or body.get("id") or "").strip()
    key = str(body.get("key") or "").strip()
    if not pid:
        return JSONResponse(status_code=400, content={"error": {"message": "provider is required", "type": "invalid_request", "code": "provider_update"}})
    if not key:
        return JSONResponse(status_code=400, content={"error": {"message": "key is required", "type": "invalid_request", "code": "provider_update"}})
    try:
        cfg = _load_policy_config()
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": "provider_update_error", "code": "provider_update"}})
    provider = (cfg.get("providers") or {}).get(pid)
    if not isinstance(provider, dict):
        return JSONResponse(status_code=404, content={"error": {"message": f"provider {pid!r} not found", "type": "not_found", "code": "provider_not_found"}})
    auth_env = str(provider.get("auth_env") or "").strip()
    if not auth_env:
        return JSONResponse(status_code=400, content={"error": {"message": f"provider {pid!r} has no auth_env (e.g. oauth/codex); use the Codex account flow instead", "type": "invalid_request", "code": "provider_no_auth_env"}})
    try:
        _upsert_env_line(Path(DASHBOARD_KEY_ENV_PATH), auth_env, key)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": "provider_update_error", "code": "provider_update"}})
    # hot-apply on the running router; on failure the key still loads from
    # .env.secrets at the next router restart (now that it's persisted)
    applied_live = False
    apply_error = None
    try:
        if _client is not None:
            r = await _client.post(f"{UPSTREAM}/x/provider-key", json={
                "provider": pid, "auth_env": auth_env, "key": key}, timeout=10.0)
            applied_live = r.status_code == 200
            if not applied_live:
                apply_error = f"router /x/provider-key returned {r.status_code}"
    except Exception as exc:
        apply_error = str(exc)
    _log({"event": "dashboard_provider_key_updated", "provider": pid, "viewer": caller,
          "auth_env": auth_env, "applied_live": applied_live})
    return JSONResponse(content={
        "ok": True, "provider": pid, "applied_live": applied_live,
        "note": None if applied_live else
        f"saved, but live apply failed ({apply_error}); the key will load on the next router restart"})


async def _wallet_proxy(request: Request, op: str, *, body: dict | None = None) -> Response:
    """Admin-only proxy to the router's /x/wallet/* — which runs the AntSeed buyer
    deposit/withdraw/refresh on the sidecar and returns the refreshed wallet. Lets
    the operator fund the hot-wallet from the catalog instead of `kubectl exec`."""
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    if _client is None:
        return JSONResponse(status_code=502, content={"error": {
            "message": "router client unavailable", "type": "wallet_error", "code": "wallet"}})
    try:
        r = await _client.post(f"{UPSTREAM}/x/wallet/{op}", json=(body or {}), timeout=135.0)
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": {
            "message": f"router /x/wallet/{op} unreachable: {exc}",
            "type": "wallet_error", "code": "wallet"}})
    _log({"event": "dashboard_wallet_op", "op": op, "viewer": caller,
          "status": r.status_code, "amount": (body or {}).get("amount")})
    try:
        return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception:
        return JSONResponse(status_code=502, content={"error": {
            "message": (r.text or "")[:300], "type": "wallet_error", "code": "wallet"}})


@app.post("/dashboard/api/wallet/deposit")
async def dashboard_wallet_deposit(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _wallet_proxy(request, "deposit", body={"amount": str(body.get("amount", "")).strip()})


@app.post("/dashboard/api/wallet/withdraw")
async def dashboard_wallet_withdraw(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        body = {}
    return await _wallet_proxy(request, "withdraw", body={"amount": str(body.get("amount", "")).strip()})


@app.post("/dashboard/api/wallet/refresh")
async def dashboard_wallet_refresh(request: Request) -> Response:
    return await _wallet_proxy(request, "refresh")


@app.get("/dashboard/api/provider-keys/reveal")
async def dashboard_reveal_provider_key(request: Request) -> Response:
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    name = str(request.query_params.get("provider") or "").strip()
    kind, value = _provider_raw_credential(name)
    _log({"event": "dashboard_provider_key_revealed", "provider": name,
          "viewer": caller, "kind": kind, "found": bool(value)})
    if not value:
        return JSONResponse(status_code=404, content={"error": {
            "message": f"no recoverable credential for provider {name!r}",
            "type": "not_found", "code": "provider_key_reveal"}})
    return JSONResponse(content={
        "provider": name, "kind": kind, "value": value,
        "fingerprint": hashlib.sha256(value.encode()).hexdigest()[:12]})


def _codex_store():
    from codex_auth import CodexAuthStore
    return CodexAuthStore(CODEX_ACCOUNTS_DIR, legacy_path=CODEX_AUTH_PATH)


def _codex_provider_id() -> str | None:
    """The provider id of the codex (openai_codex) backend, from the catalog."""
    try:
        cfg = _load_policy_config()
    except Exception:
        return None
    for pid, p in (cfg.get("providers") or {}).items():
        if isinstance(p, dict) and p.get("api_kind") == "openai_codex":
            return pid
    return None


async def _codex_activity() -> dict[str, Any]:
    """Codex provider activity for the dashboard: request/error counts (from the
    ingress stats) plus the live quota + scarcity-price state (from the router's
    /x/runtime — the codex source's quota_window and the imputed ranking price).
    So the Codex panel shows what the other providers' rows show, and more."""
    pid = _codex_provider_id() or "openai"
    with _stats_lock:
        snap = _counter_snapshot(_stats["by_provider"].get(pid) or _counter())
    rt = await _fetch_router_runtime() or {}
    bal = (rt.get("balances") or {}).get(pid) or {}
    detail = bal.get("detail") or {}
    used = bal.get("value")
    price_in = None
    for k, m in (rt.get("ema_metrics") or {}).items():
        if isinstance(m, dict) and k.startswith(pid + "|") and m.get("price_in") is not None:
            price_in = m["price_in"]
            break
    return {
        "provider": pid,
        "requests": snap.get("requests", 0),
        "errors": snap.get("errors", 0),
        "error_rate": snap.get("error_rate", 0.0),
        "tokens_total": snap.get("tokens_total", 0),
        "last_seen": snap.get("last_seen"),
        "used_percent": round(used * 100, 1) if isinstance(used, (int, float)) else None,
        "recent_429": detail.get("recent_429_count"),
        "scarcity_price_in": price_in,
        "events": detail.get("events"),
    }


async def _reload_codex_router() -> tuple[bool, str | None]:
    """Ask the router to re-scan the Codex accounts dir so a dashboard change
    goes live. On failure the change still loads at the next router restart."""
    try:
        if _client is not None:
            r = await _client.post(f"{UPSTREAM}/x/codex/reload", timeout=10.0)
            if r.status_code == 200:
                return True, None
            return False, f"router /x/codex/reload returned {r.status_code}"
    except Exception as exc:
        return False, str(exc)
    return False, "router client unavailable"


@app.get("/dashboard/api/codex/accounts")
async def dashboard_list_codex_accounts(request: Request) -> Response:
    """List the Codex (ChatGPT-subscription) accounts on the PVC — name,
    account_id and a token fingerprint, never the raw token. Admin-only."""
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    try:
        accounts = _codex_store().list_accounts()
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": "codex_list_error", "code": "codex_accounts"}})
    # the store selects the first account by name today (per-call selection is a
    # follow-up); mark it so the panel shows which one actually serves traffic.
    active = sorted(a["name"] for a in accounts)[0] if accounts else None
    for a in accounts:
        a["active"] = (a["name"] == active)
    return JSONResponse(content={"accounts": accounts, "active": active,
                                 "activity": await _codex_activity()})


@app.post("/dashboard/api/codex/accounts")
async def dashboard_add_codex_account(request: Request) -> Response:
    """Add/replace a Codex account by pasting its auth.json (the output of
    `codex login`). Persisted to the PVC and hot-applied on the router.
    Admin-only, like provider keys."""
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body", "type": "invalid_request", "code": "codex_add"}})
    name = str(body.get("name") or "").strip()
    auth_json = body.get("auth_json")
    if isinstance(auth_json, str):
        try:
            auth_json = json.loads(auth_json)
        except Exception:
            return JSONResponse(status_code=400, content={"error": {"message": "auth_json is not valid JSON", "type": "invalid_request", "code": "codex_add"}})
    if not isinstance(auth_json, dict):
        return JSONResponse(status_code=400, content={"error": {"message": "auth_json (object or JSON string) is required", "type": "invalid_request", "code": "codex_add"}})
    try:
        slug = _codex_store().add_account(name, auth_json)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request", "code": "codex_add"}})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": "codex_add_error", "code": "codex_add"}})
    applied_live, apply_error = await _reload_codex_router()
    _log({"event": "dashboard_codex_account_added", "account": slug, "viewer": caller, "applied_live": applied_live})
    return JSONResponse(content={
        "ok": True, "account": slug, "applied_live": applied_live,
        "note": None if applied_live else
        f"saved, but live reload failed ({apply_error}); the account will load on the next router restart"})


@app.delete("/dashboard/api/codex/accounts/{name}")
async def dashboard_delete_codex_account(request: Request, name: str) -> Response:
    """Remove a Codex account from the PVC and hot-reload the router. Admin-only."""
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    try:
        existed = _codex_store().delete_account(name)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request", "code": "codex_delete"}})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": "codex_delete_error", "code": "codex_delete"}})
    if not existed:
        return JSONResponse(status_code=404, content={"error": {"message": f"codex account {name!r} not found", "type": "not_found", "code": "codex_account_not_found"}})
    applied_live, apply_error = await _reload_codex_router()
    _log({"event": "dashboard_codex_account_deleted", "account": name, "viewer": caller, "applied_live": applied_live})
    return JSONResponse(content={"ok": True, "account": name, "applied_live": applied_live})


@app.get("/dashboard/api/config")
async def dashboard_get_config(request: Request) -> Response:
    """Operator-tunable knobs per provider (antseed top-N, codex scarcity ramp,
    runway thresholds) with current value + default + range. Admin-only."""
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    settings.reload()
    return JSONResponse(content={"knobs": settings.current()})


@app.post("/dashboard/api/config")
async def dashboard_set_config(request: Request) -> Response:
    """Set/clear knob overrides (validated against the schema), persist to the
    PVC and hot-apply on the router. `{updates: {key: value|null}}`; null clears
    an override back to its default. Admin-only."""
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body", "type": "invalid_request", "code": "config"}})
    updates = body.get("updates") if isinstance(body, dict) else None
    if not isinstance(updates, dict):
        return JSONResponse(status_code=400, content={"error": {"message": "expected {updates: {key: value}}", "type": "invalid_request", "code": "config"}})
    applied, errors = settings.validate_and_write(updates)
    if errors:
        return JSONResponse(status_code=400, content={"error": {"message": "; ".join(errors), "type": "invalid_request", "code": "config_invalid"}})
    applied_live = False
    note = None
    try:
        if _client is not None:
            r = await _client.post(f"{UPSTREAM}/x/config/reload", timeout=10.0)
            applied_live = r.status_code == 200
            if not applied_live:
                note = f"router /x/config/reload returned {r.status_code}"
    except Exception as exc:
        note = str(exc)
    _log({"event": "dashboard_config_updated", "viewer": caller,
          "keys": sorted(updates), "applied_live": applied_live})
    return JSONResponse(content={"ok": True, "knobs": settings.current(),
                                 "applied_live": applied_live, "note": note})


@app.get("/dashboard/api/keys/reveal")
async def dashboard_reveal_keys(request: Request) -> Response:
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    try:
        consumer = _safe_consumer_name(request.query_params.get("consumer", ""))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error", "code": "invalid_consumer"}})
    plaintext_rows = _plaintext_key_rows_for_consumer(consumer)
    hash_only_count = sum(1 for owner in CALLER_KEY_HASHES.values() if owner == consumer)
    _log({"event": "dashboard_key_revealed", "consumer": consumer, "viewer": caller, "plaintext_key_count": len(plaintext_rows), "hash_only_count": hash_only_count})
    return JSONResponse(content={
        "consumer": consumer,
        "keys": plaintext_rows,
        "hash_only_count": hash_only_count,
        "message": None if plaintext_rows else "No recoverable raw key is stored for this application. Hash-only keys cannot be revealed; generate a replacement key and copy it once.",
        "warning": "Admin-only endpoint. Raw keys are returned only for legacy plaintext CALLER_KEYS_JSON entries; provider credentials and hash-only consumer keys are not exposed.",
    })


@app.post("/dashboard/api/keys")
async def dashboard_create_key(request: Request) -> Response:
    ctx, error = _require_admin_dashboard_auth(request)
    if error:
        return error
    caller = str(ctx.get("viewer"))
    try:
        data = await request.json()
        consumer = _safe_consumer_name(str(data.get("consumer", ""))) if isinstance(data, dict) else ""
        rotate = bool(data.get("rotate")) if isinstance(data, dict) else False
        grace_period_s = _optional_int(data.get("grace_period_s"), min_value=0, max_value=90 * 24 * 3600) if isinstance(data, dict) else None
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error", "code": "invalid_consumer"}})
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body", "type": "invalid_request_error", "code": "invalid_json"}})
    now = int(time.time())
    token = f"{DASHBOARD_KEY_PREFIX}_{secrets.token_urlsafe(32)}"
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    records = _issued_consumer_records()
    meta = records.get(consumer, _normalize_consumer_record(consumer, {}))
    if rotate:
        expires_at = now + (DEFAULT_ROTATION_GRACE_S if grace_period_s is None else grace_period_s)
        for key in meta.get("keys") or []:
            if key.get("sha256_prefix") == token_hash[:12]:
                continue
            if key.get("status") == "active" and not key.get("expires_at"):
                key["expires_at"] = expires_at
                key["replaced_at"] = now
    new_hashes = dict(CALLER_KEY_HASHES)
    new_hashes[token_hash] = consumer
    _upsert_env_json(Path(DASHBOARD_KEY_ENV_PATH), "CALLER_KEYS_SHA256_JSON", new_hashes)
    CALLER_KEY_HASHES.clear()
    CALLER_KEY_HASHES.update(new_hashes)
    meta.setdefault("keys", [])
    meta["keys"].append({"sha256_prefix": token_hash[:12], "status": "active", "created_at": now, "viewer": caller})
    meta["status"] = meta.get("status") or "active"
    meta["updated_at"] = now
    records[consumer] = meta
    _write_issued_consumer_records(records)
    _log({"event": "dashboard_key_created", "consumer": consumer, "viewer": caller, "rotate": rotate})
    return JSONResponse(content={"ok": True, "consumer": consumer, "api_key": token, "sha256_prefix": token_hash[:12], "rotate": rotate, "grace_period_s": None if not rotate else (DEFAULT_ROTATION_GRACE_S if grace_period_s is None else grace_period_s), "warning": "Copy now. The raw key is shown once and only hashed key metadata is persisted."})




def _ollama_show_response(model: str | None) -> dict[str, Any]:
    name = str(model or "profile:default").strip() or "profile:default"
    return {
        "license": "router compatibility metadata",
        "modelfile": f"FROM {name}\nPARAMETER num_ctx {ROUTER_CONTEXT_LENGTH}\n",
        "parameters": f"num_ctx {ROUTER_CONTEXT_LENGTH}",
        "template": "",
        "details": {"family": "llm-router", "families": ["llm-router"], "parameter_size": "router", "quantization_level": "router"},
        "model_info": {
            "router.context_length": ROUTER_CONTEXT_LENGTH,
            "llm-router.context_length": ROUTER_CONTEXT_LENGTH,
            "general.architecture": "llm-router",
        },
    }


def _record_probe(**event: Any) -> None:
    now = int(time.time())
    with _stats_lock:
        _stats["recent"].appendleft({"ts": now, "event": "probe", **event})

def _record_synthetic_probe_result(**event: Any) -> None:
    route = str(event.get("route") or event.get("requested_model") or "").strip()
    if not route:
        return
    ts = int(event.get("ts") or time.time())
    status = int(event.get("status") or 0)
    row = {
        "ts": ts,
        "event": "synthetic_probe",
        "route": route,
        "requested_model": route,
        "status": status,
        "latency_ms": float(event.get("latency_ms") or 0),
        "provider": event.get("provider"),
        "model_family": event.get("model_family"),
        "served_model_id": event.get("served_model_id"),
        "decision_trace": event.get("decision_trace") if isinstance(event.get("decision_trace"), dict) else None,
    }
    with _stats_lock:
        _stats["synthetic_route_health"][route] = row


def _extract_router_metadata(parsed: Any) -> tuple[str | None, str | None, str | None, dict[str, Any] | None]:
    if not isinstance(parsed, dict):
        return None, None, None, None
    xr = parsed.get("x_router")
    if not isinstance(xr, dict):
        return None, None, None, None
    trace = xr.get("decision_trace") if isinstance(xr.get("decision_trace"), dict) else None
    return xr.get("provider"), xr.get("model_family"), xr.get("served_model_id"), trace


async def _synthetic_probe_once(route: str) -> None:
    assert _client is not None
    started = time.perf_counter()
    status = 502
    provider = model_family = served_model_id = None
    decision_trace = None
    try:
        # Keep the probe payload as close as possible to normal Hermes traffic,
        # while bounding cost/latency. OpenAI-compatible providers require at
        # least 16 output tokens on newer GPT routes, so do not probe with tiny
        # max_tokens values that create false red route-health entries.
        payload = {
            "model": route,
            "messages": [{"role": "user", "content": "Reply exactly: pong"}],
            "max_tokens": 16,
        }
        headers = {"x-llm-router-caller": SYNTHETIC_PROBE_CALLER}
        r = await _client.post(f"{UPSTREAM}/v1/chat/completions", json=payload, headers=headers, timeout=SYNTHETIC_PROBE_TIMEOUT_S)
        status = r.status_code
        try:
            provider, model_family, served_model_id, decision_trace = _extract_router_metadata(r.json())
        except Exception:
            pass
    except Exception as exc:
        decision_trace = {"attempts": [{"error_kind": "probe_error", "message": str(exc)[:160]}]}
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        _record_synthetic_probe_result(route=route, status=status, latency_ms=latency_ms, provider=provider, model_family=model_family, served_model_id=served_model_id, decision_trace=decision_trace)
        _log({"event": "synthetic_probe", "route": route, "status": status, "latency_ms": latency_ms, "provider": provider, "model_family": model_family})


async def _synthetic_probe_loop() -> None:
    if SYNTHETIC_PROBE_INITIAL_DELAY_S > 0:
        await asyncio.sleep(SYNTHETIC_PROBE_INITIAL_DELAY_S)
    while True:
        for route in ROUTE_HEALTH_ROUTES:
            await _synthetic_probe_once(route)
        await asyncio.sleep(max(SYNTHETIC_PROBE_INTERVAL_S, 1.0))

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request) -> Response:
    if path == "x" or path.startswith("x/"):
        # Router-internal diagnostics (/x/runtime): never proxied to callers.
        return JSONResponse(status_code=404, content={"error": {"message": "not found", "type": "invalid_request_error", "code": None}})
    started = time.perf_counter()
    token = _extract_token(request)
    auth = _caller_auth(token)
    caller = auth.get("caller")
    if not auth.get("ok"):
        code = auth.get("error_code") or "caller_auth"
        status_code = 403 if code in {"caller_inactive", "caller_key_revoked", "caller_key_expired"} else 401
        _record_reject(reason=code, path="/" + path, caller=caller, status=status_code, remote=request.client.host if request.client else None)
        _log({"event": "reject", "reason": code, "path": "/" + path, "caller": caller, "remote": request.client.host if request.client else None})
        messages = {
            "caller_auth": "unauthorized caller",
            "caller_inactive": "caller is inactive",
            "caller_key_revoked": "caller key is revoked",
            "caller_key_expired": "caller key is expired",
        }
        return JSONResponse(status_code=status_code, content={"error": {"message": messages.get(code, "caller not authorized"), "type": "auth_error", "code": code}})

    caller = str(caller)
    body = await request.body()
    requested_route = _requested_route_from(path, body)
    if not _route_allowed(caller, requested_route):
        _record_reject(reason="route_not_allowed", path="/" + path, caller=caller, status=403, route=requested_route)
        _log({"event": "reject", "reason": "route_not_allowed", "caller": caller, "path": "/" + path, "route": requested_route})
        return JSONResponse(status_code=403, content={"error": {"message": "caller is not allowed to use this route", "type": "auth_error", "code": "caller_route_not_allowed"}})
    if not _rate_ok(caller):
        _record_reject(reason="rate_limit", path="/" + path, caller=caller, status=429)
        _log({"event": "reject", "reason": "rate_limit", "caller": caller, "path": "/" + path})
        return JSONResponse(status_code=429, content={"error": {"message": "caller rate limit exceeded", "type": "rate_limit_error", "code": "caller_rate_limit"}})

    if path == "api/show" and request.method.upper() == "POST":
        requested_model = None
        if body:
            try:
                parsed_body = json.loads(body.decode("utf-8"))
                if isinstance(parsed_body, dict):
                    requested_model = parsed_body.get("name") or parsed_body.get("model")
            except Exception:
                pass
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        _record_probe(caller=caller, method=request.method, path="/" + path, status=200, latency_ms=latency_ms, requested_model=requested_model, route="metadata_probe", key_sha256_prefix=str(auth.get("digest") or "")[:12] or None)
        _log({"event": "metadata_probe", "caller": caller, "method": request.method, "path": "/" + path, "status": 200, "latency_ms": latency_ms, "requested_model": requested_model})
        return JSONResponse(content=_ollama_show_response(requested_model))

    assert _client is not None
    upstream_url = f"{UPSTREAM}/{path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in {"authorization", "host", "connection", "content-length"}
    }
    headers["x-llm-router-caller"] = caller

    status = 502
    provider = None
    model_family = None
    served_model_id = None
    requested_model = None
    tokens_in = tokens_out = tokens_total = 0
    cost_usd = None
    decision_trace = None
    error_type = error_code = error_message = None
    record_in_finally = True

    def _finish():
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        _record_request(caller=caller, method=request.method, path="/" + path, status=status, latency_ms=latency_ms, provider=provider, model_family=model_family, served_model_id=served_model_id, requested_model=requested_model, tokens_in=tokens_in, tokens_out=tokens_out, tokens_total=tokens_total, cost_usd=cost_usd, decision_trace=decision_trace, error_type=error_type, error_code=error_code, error_message=error_message, key_sha256=auth.get("digest"))
        _log({"event": "request", "caller": caller, "method": request.method, "path": "/" + path, "status": status, "latency_ms": latency_ms, "provider": provider, "model_family": model_family})

    try:
        if body:
            try:
                parsed_body = json.loads(body.decode("utf-8"))
                if isinstance(parsed_body, dict):
                    requested_model = parsed_body.get("model")
            except Exception:
                pass
        upstream_req = _client.build_request(request.method, upstream_url, content=body, headers=headers)
        r = await _client.send(upstream_req, stream=True)
        status = r.status_code
        content_type = r.headers.get("content-type", "application/json")
        if content_type.startswith("text/event-stream"):
            # SSE pass-through: forward chunks unbuffered while teeing a tail
            # buffer. The shim's final chat.completion.chunk carries usage and
            # x_router (incl. the executed cost_usd), so the request is
            # recorded AFTER the stream ends — with tokens, cost, provider and
            # full-stream latency — instead of in the outer finally.
            record_in_finally = False

            async def _passthrough():
                nonlocal provider, model_family, served_model_id, \
                    tokens_in, tokens_out, tokens_total, cost_usd, \
                    decision_trace, error_type, error_code, error_message
                tail = bytearray()
                try:
                    async for chunk in r.aiter_raw():
                        tail.extend(chunk)
                        _trim_sse_tail(tail)
                        yield chunk
                finally:
                    await r.aclose()
                    try:
                        meta = _parse_stream_tail(bytes(tail))
                        xr = meta.get("x_router") or {}
                        if xr:
                            provider = xr.get("provider")
                            model_family = xr.get("model_family")
                            served_model_id = xr.get("served_model_id")
                            decision_trace = xr.get("decision_trace") if isinstance(xr.get("decision_trace"), dict) else None
                            if isinstance(xr.get("cost_usd"), (int, float)):
                                cost_usd = float(xr["cost_usd"])
                        usage = meta.get("usage")
                        if isinstance(usage, dict):
                            tokens_in = int(usage.get("prompt_tokens") or 0)
                            tokens_out = int(usage.get("completion_tokens") or 0)
                            tokens_total = int(usage.get("total_tokens") or (tokens_in + tokens_out))
                        err = meta.get("error")
                        if isinstance(err, dict):
                            error_type = err.get("type")
                            error_code = err.get("code")
                            error_message = err.get("message")
                    except Exception:
                        pass
                    _finish()
            return StreamingResponse(_passthrough(), status_code=status, media_type=content_type)
        content = await r.aread()
        await r.aclose()
        if content_type.startswith("application/json"):
            try:
                parsed = json.loads(content.decode("utf-8"))
                xr = parsed.get("x_router") if isinstance(parsed, dict) else None
                if isinstance(xr, dict):
                    provider = xr.get("provider")
                    model_family = xr.get("model_family")
                    served_model_id = xr.get("served_model_id")
                    decision_trace = xr.get("decision_trace") if isinstance(xr.get("decision_trace"), dict) else None
                    if isinstance(xr.get("cost_usd"), (int, float)):
                        cost_usd = float(xr["cost_usd"])
                usage = parsed.get("usage") if isinstance(parsed, dict) else None
                if isinstance(parsed, dict):
                    err = parsed.get("error")
                    if isinstance(err, dict):
                        error_type = err.get("type")
                        error_code = err.get("code")
                        error_message = err.get("message")
                if isinstance(usage, dict):
                    tokens_in = int(usage.get("prompt_tokens") or 0)
                    tokens_out = int(usage.get("completion_tokens") or 0)
                    tokens_total = int(usage.get("total_tokens") or (tokens_in + tokens_out))
            except Exception:
                pass
        return Response(status_code=status, content=content, media_type=content_type)
    except Exception as exc:
        error_type = "proxy_error"
        error_code = "upstream"
        error_message = str(exc)
        return JSONResponse(status_code=502, content={"error": {"message": f"upstream proxy error: {exc}", "type": "proxy_error", "code": "upstream"}})
    finally:
        if record_in_finally:
            _finish()


def _record_reject(**event: Any) -> None:
    stored = {"ts": int(time.time()), "event": "reject", **event}
    with _stats_lock:
        _stats["total_rejects"] += 1
        _stats["recent"].appendleft(stored)
    # Rejects survive ingress recreates like requests do, so non-runtime
    # timeframes keep counting them. `remote` (client IP) stays in the
    # in-memory deque only — ephemeral diagnostics, not durable data.
    _append_usage_history({k: v for k, v in stored.items() if k != "remote"})


def _usage_history_path() -> Path | None:
    raw = os.getenv("ROUTER_USAGE_HISTORY_PATH", USAGE_HISTORY_PATH_DEFAULT).strip()
    if not raw:
        return None
    return Path(raw)


# The durable usage-history is for aggregation (provider/model/status/tokens/
# cost over a timeframe); the per-request decision_trace (the full policy_term
# AST + every ranked candidate + the decision_path) is only needed for the live
# Activity view, which reads the in-memory `_stats["recent"]` deque — that keeps
# the full trace. Persisting the trace in every durable row bloated the file to
# ~23-65 KB/row, and the dashboard reader loads the whole file into memory, so a
# few thousand rows OOM-killed the auth-proxy container. Strip the heavy fields
# on write, bound the file, and bound the read.
_HISTORY_HEAVY_FIELDS = ("decision_trace", "ranked", "policy_term", "decision_path")
USAGE_HISTORY_MAX_BYTES = int(os.getenv("ROUTER_USAGE_HISTORY_MAX_BYTES", str(16 * 1024 * 1024)))
USAGE_HISTORY_READ_TAIL_BYTES = int(os.getenv("ROUTER_USAGE_HISTORY_READ_TAIL_BYTES", str(8 * 1024 * 1024)))


def _slim_history_row(row: dict[str, Any]) -> dict[str, Any]:
    """Drop the heavy debug-only fields from a durable usage row. No-op (returns
    the same object) when none are present, so the common small-row path is free."""
    if isinstance(row, dict) and any(k in row for k in _HISTORY_HEAVY_FIELDS):
        return {k: v for k, v in row.items() if k not in _HISTORY_HEAVY_FIELDS}
    return row


def _rotate_usage_history(path: Path) -> None:
    """Keep the append-only history bounded: once it grows past the cap, retain
    only the most recent half. Cheap — only rewrites when the file actually
    exceeds the cap, and reads just the tail it keeps (never the whole file)."""
    try:
        if path.stat().st_size <= USAGE_HISTORY_MAX_BYTES:
            return
        keep = max(USAGE_HISTORY_MAX_BYTES // 2, 1)
        with path.open("rb") as fh:
            fh.seek(-keep, os.SEEK_END)
            data = fh.read()
        nl = data.find(b"\n")  # drop the partial first line left by the seek
        if nl != -1:
            data = data[nl + 1:]
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except Exception as exc:
        log.warning(json.dumps({"event": "usage_history_rotate_failed", "path": str(path), "error": str(exc)}))


def _append_usage_history(row: dict[str, Any]) -> None:
    path = _usage_history_path()
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_slim_history_row(row), sort_keys=True, separators=(",", ":")) + "\n")
        _rotate_usage_history(path)
    except Exception as exc:
        log.warning(json.dumps({"event": "usage_history_write_failed", "path": str(path), "error": str(exc)}))


def _read_usage_history(*, events: tuple[str, ...] = ("request",)) -> list[dict[str, Any]]:
    # Default stays request-only: every aggregation written before reject
    # events were persisted assumes request-shaped rows. Readers that can
    # discriminate (the timeframe stats) opt in via `events`.
    path = _usage_history_path()
    if not path or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        # Read at most the last READ_TAIL_BYTES so an unbounded (or pre-rotation,
        # still-bloated) file can never balloon the reader. Bytes mode lets us
        # seek to the tail; json.loads accepts bytes directly.
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > USAGE_HISTORY_READ_TAIL_BYTES:
                fh.seek(-USAGE_HISTORY_READ_TAIL_BYTES, os.SEEK_END)
                fh.readline()  # discard the partial first line left by the seek
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict) and row.get("event") in events:
                    rows.append(_slim_history_row(row))
    except Exception as exc:
        log.warning(json.dumps({"event": "usage_history_read_failed", "path": str(path), "error": str(exc)}))
    return rows


_SSE_TAIL_HARD_CAP = 4 * 1024 * 1024  # 4 MiB backstop if no event boundary appears


def _trim_sse_tail(tail: bytearray) -> None:
    """Bound the streamed-response tail buffer WITHOUT splitting an SSE event.

    The shim's final event carries usage + x_router, and its decision_trace (the
    ranked catalog) routinely pushes that single event past 64 KiB. The old blind
    byte cap (`del tail[:len-65536]`) sliced through that event's JSON, so
    `_parse_stream_tail` could not parse it and Activity recorded empty rows
    (provider/model_family/tokens/decision_trace all null) for every real call.

    Keep only the last three COMPLETE events (SSE separates events with a blank
    line, "\\n\\n"): bounded memory, yet the x_router event — usually the
    second-to-last, just before the terminal "[DONE]" — stays whole however large
    its trace. Trim only once at least three boundaries exist, so a stream with
    fewer events is never clipped. A generous byte cap backstops a stream that
    never emits a boundary. Mutates `tail` in place.
    """
    idx = len(tail)
    found = 0
    for _ in range(3):
        j = tail.rfind(b"\n\n", 0, idx)
        if j == -1:
            break
        idx = j
        found += 1
    if found == 3:
        del tail[:idx + 2]
    if len(tail) > _SSE_TAIL_HARD_CAP:
        del tail[:len(tail) - _SSE_TAIL_HARD_CAP]


def _parse_stream_tail(tail: bytes) -> dict[str, Any]:
    """Extract usage / x_router / error from the tail of a shim SSE stream.
    The shim's final chat.completion.chunk carries usage and x_router
    together; an aborted stream emits a single error event. Scans data lines
    newest-first and stops once both usage and x_router are found."""
    out: dict[str, Any] = {}
    lines = tail.decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        if "usage" not in out and isinstance(payload.get("usage"), dict):
            out["usage"] = payload["usage"]
        if "x_router" not in out and isinstance(payload.get("x_router"), dict):
            out["x_router"] = payload["x_router"]
        if "error" not in out and isinstance(payload.get("error"), dict):
            out["error"] = payload["error"]
        if "usage" in out and "x_router" in out:
            break
    return out


def _record_request(**event: Any) -> None:
    status = int(event.get("status") or 0)
    latency_ms = float(event.get("latency_ms") or 0)
    tokens_in = int(event.get("tokens_in") or 0)
    tokens_out = int(event.get("tokens_out") or 0)
    tokens_total = int(event.get("tokens_total") or 0)
    is_error = status >= 400
    now = int(event.get("ts") or time.time())
    try:
        cost_usd = float(event["cost_usd"]) if event.get("cost_usd") is not None else None
    except (TypeError, ValueError):
        cost_usd = None
    if cost_usd is not None:
        cost_usd = max(0.0, cost_usd)  # a call's cost is >= 0; never accumulate negative spend
    key_sha256 = str(event.get("key_sha256") or "").strip().lower()
    key_sha256_prefix = key_sha256[:12] if re.fullmatch(r"[a-f0-9]{64}", key_sha256) else None
    with _stats_lock:
        _stats["total_requests"] += 1
        if is_error:
            _stats["total_errors"] += 1
        _stats["total_tokens_in"] += tokens_in
        _stats["total_tokens_out"] += tokens_out
        _stats["total_tokens"] += tokens_total
        _stats["by_status"][str(status)] += 1
        route_key = event.get("requested_model") or event.get("route") or "unknown"
        served_key = event.get("served_model_id") or "unknown"
        for group_name, key in (("by_caller", event.get("caller")), ("by_provider", event.get("provider") or "unknown"), ("by_model_family", event.get("model_family") or "unknown"), ("by_route", route_key), ("by_served_model", served_key)):
            c = _stats[group_name][str(key or "unknown")]
            c["requests"] += 1
            if is_error:
                c["errors"] += 1
            c["tokens_in"] += tokens_in
            c["tokens_out"] += tokens_out
            c["tokens_total"] += tokens_total
            c["latency_ms_total"] += latency_ms
            c["latency_ms_max"] = max(float(c["latency_ms_max"] or 0), latency_ms)
            c["last_seen"] = now
            if cost_usd is not None:
                c["cost_usd"] = round(float(c.get("cost_usd") or 0.0) + cost_usd, 6)
        caller = str(event.get("caller") or "unknown")
        for group_name, key in (("by_caller_provider", event.get("provider") or "unknown"), ("by_caller_model_family", event.get("model_family") or "unknown"), ("by_caller_route", route_key), ("by_caller_served_model", served_key)):
            c = _stats[group_name][caller][str(key or "unknown")]
            c["requests"] += 1
            if is_error:
                c["errors"] += 1
            c["tokens_in"] += tokens_in
            c["tokens_out"] += tokens_out
            c["tokens_total"] += tokens_total
            c["latency_ms_total"] += latency_ms
            c["latency_ms_max"] = max(float(c["latency_ms_max"] or 0), latency_ms)
            c["last_seen"] = now
            if cost_usd is not None:
                c["cost_usd"] = round(float(c.get("cost_usd") or 0.0) + cost_usd, 6)
        _stats["by_caller_status"][caller][str(status)] += 1
        if key_sha256_prefix:
            _stats["key_owner"][key_sha256] = caller
            c = _stats["by_key_sha256"][key_sha256]
            c["requests"] += 1
            if is_error:
                c["errors"] += 1
            c["tokens_in"] += tokens_in
            c["tokens_out"] += tokens_out
            c["tokens_total"] += tokens_total
            c["latency_ms_total"] += latency_ms
            c["latency_ms_max"] = max(float(c["latency_ms_max"] or 0), latency_ms)
            c["last_seen"] = now
            if cost_usd is not None:
                c["cost_usd"] = round(float(c.get("cost_usd") or 0.0) + cost_usd, 6)
            for group_name, key in (("by_key_provider", event.get("provider") or "unknown"), ("by_key_model_family", event.get("model_family") or "unknown"), ("by_key_route", route_key), ("by_key_served_model", served_key)):
                kc = _stats[group_name][key_sha256][str(key or "unknown")]
                kc["requests"] += 1
                if is_error:
                    kc["errors"] += 1
                kc["tokens_in"] += tokens_in
                kc["tokens_out"] += tokens_out
                kc["tokens_total"] += tokens_total
                kc["latency_ms_total"] += latency_ms
                kc["latency_ms_max"] = max(float(kc["latency_ms_max"] or 0), latency_ms)
                kc["last_seen"] = now
            _stats["by_key_status"][key_sha256][str(status)] += 1
        recent_event = {k: v for k, v in event.items() if k != "key_sha256"}
        recent_event["usage_event_id"] = str(event.get("usage_event_id") or secrets.token_hex(12))
        recent_event["key_sha256"] = key_sha256 if key_sha256_prefix else None
        if key_sha256_prefix:
            recent_event["key_sha256_prefix"] = key_sha256_prefix
        stored_event = {"ts": now, "event": "request", **recent_event}
        visible_event = {k: v for k, v in stored_event.items() if k != "key_sha256"}
        _stats["recent"].appendleft(visible_event)
    if key_sha256_prefix:
        _append_usage_history(stored_event)

def _safe_dashboard_message(value: Any, *, limit: int = 180) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    for marker in ("Bearer ", "sk-", "OPENROUTER_API_KEY", "DASHBOARD_PASSWORD"):
        if marker in text:
            text = text.split(marker, 1)[0] + marker + "[REDACTED]"
    return text[:limit]


def _event_error_kind(row: dict[str, Any]) -> str | None:
    for key in ("error_code", "error_type"):
        if row.get(key):
            return str(row.get(key))
    trace = row.get("decision_trace") if isinstance(row.get("decision_trace"), dict) else None
    attempts = []
    if isinstance(trace, dict):
        attempts = trace.get("attempts") or trace.get("decision_path") or []
    for attempt in reversed(attempts):
        if isinstance(attempt, dict) and attempt.get("error_kind"):
            return str(attempt.get("error_kind"))
    msg = str(row.get("error_message") or "")
    for needle in ("no_candidates", "network_error", "bad_response", "bad_request", "auth_error", "rate_limit", "exhausted", "unknown"):
        if needle in msg:
            return needle
    return None


def _health_summary(recent: list[dict[str, Any]], route_health: list[dict[str, Any]]) -> dict[str, Any]:
    chat_rows = [r for r in recent if r.get("event") == "request" and str(r.get("path") or "").startswith("/v1/chat/completions")]
    request_count = len(chat_rows)
    success_count = 0
    error_kinds: dict[str, int] = defaultdict(int)
    status_counts: dict[str, int] = defaultdict(int)
    failing_recent: list[dict[str, Any]] = []
    for row in chat_rows:
        status = int(row.get("status") or 0)
        status_counts[str(status)] += 1
        kind = _event_error_kind(row)
        ok = 200 <= status < 300 and not kind
        if ok:
            success_count += 1
            continue
        error_kinds[kind or "unknown"] += 1
        if len(failing_recent) < 10:
            failing_recent.append({
                "ts": row.get("ts"),
                "caller": row.get("caller"),
                "route": row.get("requested_model") or row.get("route"),
                "provider": row.get("provider"),
                "served_model_id": row.get("served_model_id"),
                "status": status,
                "error_kind": kind or "unknown",
                "error_message": _safe_dashboard_message(row.get("error_message")),
            })
    error_count = request_count - success_count
    success_rate = round(success_count / request_count, 4) if request_count else None
    route_failures = sum(1 for r in route_health if r.get("state") in {"fail", "warn"})
    route_count = len(route_health)
    route_ok = route_count > 0 and route_failures == 0 and all(r.get("state") == "ok" for r in route_health)
    if request_count and success_rate is not None and success_rate < 0.8:
        state = "down" if success_rate < 0.5 else "degraded"
    elif route_failures:
        state = "degraded"
    elif request_count == 0 and route_ok:
        state = "ok"
    elif request_count == 0:
        state = "unknown"
    else:
        state = "ok"
    return {
        "state": state,
        "request_count": request_count,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": success_rate,
        "status_counts": dict(sorted(status_counts.items())),
        "error_kinds": dict(sorted(error_kinds.items())),
        "route_failures": route_failures,
        "failing_recent": failing_recent,
    }


def _route_health_snapshot(recent: list[dict[str, Any]], synthetic: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = dict(synthetic or {})
    for row in recent:
        route = row.get("requested_model") or row.get("route")
        if route and row.get("event") == "request":
            existing = latest.get(str(route))
            if not existing or int(row.get("ts") or 0) >= int(existing.get("ts") or 0):
                latest[str(route)] = row
    out = []
    for route in ROUTE_HEALTH_ROUTES:
        row = latest.get(route)
        if not row:
            out.append({"route": route, "state": "unknown", "source": None, "status": None, "provider": None, "served_model_id": None, "latency_ms": None, "last_seen": None})
            continue
        status = int(row.get("status") or 0)
        trace = row.get("decision_trace") if isinstance(row.get("decision_trace"), dict) else None
        attempts = []
        if isinstance(trace, dict):
            attempts = trace.get("attempts") or trace.get("decision_path") or []
        empty_seen = any(a.get("error_kind") == "empty_response" for a in attempts if isinstance(a, dict))
        error_kind = _event_error_kind(row)
        state = "ok" if 200 <= status < 300 and not (empty_seen or error_kind) else "warn" if 200 <= status < 300 else "fail"
        out.append({
            "route": route,
            "state": state,
            "source": row.get("event"),
            "status": status,
            "provider": row.get("provider"),
            "model_family": row.get("model_family"),
            "served_model_id": row.get("served_model_id"),
            "latency_ms": row.get("latency_ms"),
            "last_seen": row.get("ts"),
            "error_kind": error_kind,
            "error_message": _safe_dashboard_message(row.get("error_message")),
            "note": "fallback after empty response" if empty_seen else None,
        })
    return out


def _counter_snapshot(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    req = int(out.get("requests") or 0)
    out["latency_ms_avg"] = round(float(out.pop("latency_ms_total", 0.0)) / req, 1) if req else 0.0
    out["latency_ms_max"] = round(float(out.get("latency_ms_max") or 0), 1)
    out["error_rate"] = round((int(out.get("errors") or 0) / req), 4) if req else 0.0
    return out


def _redact_usage_detail(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            skey = str(key)
            if skey in {"key_sha256", "api_key", "authorization", "Authorization"}:
                out[skey] = "[REDACTED]"
            else:
                out[skey] = _redact_usage_detail(item)
        return out
    if isinstance(value, list):
        return [_redact_usage_detail(item) for item in value]
    if isinstance(value, str):
        text = value
        text = re.sub(r"Bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
        text = re.sub(r"[A-Fa-f0-9]{64}", "[REDACTED_SHA256]", text)
        text = re.sub(r"sk-[A-Za-z0-9_\-]{12,}", "sk-[REDACTED]", text)
        return text
    return value


def _key_metadata_for_digest(digest: str, caller: str | None) -> dict[str, Any] | None:
    if not caller:
        return None
    meta = _consumer_meta(caller)
    for row in meta.get("keys") or []:
        prefix = str(row.get("sha256_prefix") or "").strip().lower()
        if prefix and (digest.startswith(prefix) or prefix.startswith(digest)):
            return {k: v for k, v in row.items() if k != "storage"}
    return {"sha256_prefix": digest[:12], "status": "active"}


def _sanitize_key_usage_event(row: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "ts", "event", "caller", "method", "path", "status", "latency_ms",
        "provider", "model_family", "served_model_id", "requested_model",
        "tokens_in", "tokens_out", "tokens_total", "error_type", "error_code",
        "error_message", "decision_trace", "key_sha256_prefix", "cost_usd",
    }
    out = {k: _redact_usage_detail(v) for k, v in row.items() if k in allowed}
    if out.get("error_message") is not None:
        out["error_message"] = _safe_dashboard_message(out.get("error_message"), limit=500)
    out["error_kind"] = _event_error_kind(out)
    return out


def _parse_usage_time(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+", text):
        return int(text)
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except Exception as exc:
        raise ValueError("since/until must be a unix timestamp or ISO-8601 datetime") from exc


def _parse_usage_window(value: Any, *, now: int | None = None) -> tuple[int | None, str | None]:
    if value is None or value == "":
        return None, None
    text = str(value).strip().lower()
    match = re.fullmatch(r"(\d+)([mhdw])", text)
    if not match:
        raise ValueError("window must look like 15m, 24h, 7d, or 4w")
    amount = int(match.group(1))
    unit = match.group(2)
    seconds = amount * {"m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    return int(now or time.time()) - seconds, text


def _usage_query_options(source: Any) -> dict[str, Any]:
    get = source.get if hasattr(source, "get") else lambda key, default=None: default
    now = int(time.time())
    since = _parse_usage_time(get("since"))
    window_since, window_label = _parse_usage_window(get("window"), now=now)
    if window_since is not None:
        since = window_since
    until = _parse_usage_time(get("until"))
    limit = _optional_int(get("limit"), min_value=1, max_value=500)
    offset = _optional_int(get("offset"), min_value=0, max_value=1_000_000) or 0
    return {"since": since, "until": until, "window": window_label, "limit": limit or 100, "offset": offset}


def _price_table() -> dict[tuple[str, str], dict[str, float]]:
    path = _resolved_path(DASHBOARD_POLICY_METRICS_PATH)
    if not path.exists():
        return {}
    try:
        text = path.read_text()
    except Exception:
        return {}
    prices: dict[tuple[str, str], dict[str, float]] = {}
    pattern = re.compile(r'\["(?P<family>[^"]+)@(?P<provider>[^"]+)"\]\s*=\s*\{(?P<body>[^}]+)\}')
    for match in pattern.finditer(text):
        body = match.group("body")
        pin = re.search(r"price_in_usd_per_mtok\s*=\s*([0-9.]+)", body)
        pout = re.search(r"price_out_usd_per_mtok\s*=\s*([0-9.]+)", body)
        if pin and pout:
            prices[(match.group("family"), match.group("provider"))] = {"input": float(pin.group(1)), "output": float(pout.group(1))}
    return prices


def _cost_for_event(row: dict[str, Any], prices: dict[tuple[str, str], dict[str, float]]) -> tuple[float | None, dict[str, Any] | None]:
    # prefer the cost stamped at execution time (the price the ranker
    # actually used); fall back to the read-time estimate for older events
    stamped = row.get("cost_usd")
    if isinstance(stamped, (int, float)):
        # Clamp: a call's cost is >= 0. Events recorded before the cost was clamped
        # at the source carry negative spend (a negative chosen price); never let
        # them subtract from a consumer's / the analytics total.
        return max(0.0, float(stamped)), None
    family = str(row.get("model_family") or "")
    provider = str(row.get("provider") or "")
    price = prices.get((family, provider))
    if not price:
        return None, None
    tokens_in = int(row.get("tokens_in") or 0)
    tokens_out = int(row.get("tokens_out") or 0)
    cost = round((tokens_in / 1_000_000.0) * price["input"] + (tokens_out / 1_000_000.0) * price["output"], 6)
    return max(0.0, cost), {"model_family": family, "provider": provider, "input_usd_per_mtok": price["input"], "output_usd_per_mtok": price["output"]}


def _add_counter(counter: dict[str, Any], row: dict[str, Any], cost: float | None = None) -> None:
    status = int(row.get("status") or 0)
    tokens_in = int(row.get("tokens_in") or 0)
    tokens_out = int(row.get("tokens_out") or 0)
    tokens_total = int(row.get("tokens_total") or (tokens_in + tokens_out))
    latency_ms = float(row.get("latency_ms") or 0)
    ts = int(row.get("ts") or 0)
    counter["requests"] += 1
    if status >= 400:
        counter["errors"] += 1
    counter["tokens_in"] += tokens_in
    counter["tokens_out"] += tokens_out
    counter["tokens_total"] += tokens_total
    counter["latency_ms_total"] += latency_ms
    counter["latency_ms_max"] = max(float(counter.get("latency_ms_max") or 0), latency_ms)
    counter["last_seen"] = max(int(counter.get("last_seen") or 0), ts) or None
    if cost is not None:
        counter["cost_usd"] = round(float(counter.get("cost_usd") or 0) + cost, 6)


def _usage_events_for_key(digest: str, caller: str | None) -> tuple[list[dict[str, Any]], bool]:
    prefix = digest[:12]
    with _stats_lock:
        memory_rows = [dict(r) for r in _stats["recent"] if r.get("key_sha256_prefix") == prefix]
    history_rows = [dict(r) for r in _read_usage_history() if r.get("key_sha256") == digest or r.get("key_sha256_prefix") == prefix]
    rows_by_id: dict[str, dict[str, Any]] = {}
    anonymous_rows: list[dict[str, Any]] = []
    for row in history_rows + memory_rows:
        if caller and row.get("caller") != caller:
            continue
        rid = str(row.get("usage_event_id") or "")
        if rid:
            rows_by_id[rid] = row
        else:
            anonymous_rows.append(row)
    rows = list(rows_by_id.values()) + anonymous_rows
    rows.sort(key=lambda r: int(r.get("ts") or 0), reverse=True)
    return rows, bool(history_rows)


def _windowed_usage_events(rows: list[dict[str, Any]], opts: dict[str, Any]) -> list[dict[str, Any]]:
    since = opts.get("since")
    until = opts.get("until")
    out = []
    for row in rows:
        ts = int(row.get("ts") or 0)
        if since is not None and ts < int(since):
            continue
        if until is not None and ts > int(until):
            continue
        out.append(row)
    return out


def _period_totals(rows: list[dict[str, Any]], *, monthly: bool) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    prices = _price_table()
    for row in rows:
        ts = int(row.get("ts") or 0)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        key = dt.strftime("%Y-%m" if monthly else "%Y-%m-%d")
        b = buckets.setdefault(key, {"month" if monthly else "date": key, **_counter()})
        cost, _ = _cost_for_event(row, prices)
        _add_counter(b, row, cost)
    return [{k: v for k, v in _counter_snapshot(b).items() if k != "last_seen"} | {"month" if monthly else "date": key} for key, b in sorted(buckets.items(), reverse=True)]


def _key_usage_snapshot(*, viewer: str, key_sha256: str, caller: str | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    digest = str(key_sha256 or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("key_sha256 must be a 64-character lowercase hex digest")
    prefix = digest[:12]
    options = options or {"since": None, "until": None, "window": None, "limit": 100, "offset": 0}
    with _stats_lock:
        inferred_caller = caller or _stats["key_owner"].get(digest) or CALLER_KEY_HASHES.get(digest)
    all_rows, persistent = _usage_events_for_key(digest, inferred_caller)
    rows = _windowed_usage_events(all_rows, options)
    prices = _price_table()
    counter = _counter()
    by_provider: dict[str, dict[str, Any]] = defaultdict(_counter)
    by_model_family: dict[str, dict[str, Any]] = defaultdict(_counter)
    by_route: dict[str, dict[str, Any]] = defaultdict(_counter)
    by_served_model: dict[str, dict[str, Any]] = defaultdict(_counter)
    by_status: dict[str, int] = defaultdict(int)
    total_cost = 0.0
    priced_events = 0
    price_sources: dict[str, dict[str, Any]] = {}
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        cost, price_meta = _cost_for_event(row, prices)
        if cost is not None:
            priced_events += 1
            total_cost = round(total_cost + cost, 6)
            row = {**row, "cost_usd": cost}
            if price_meta:
                price_sources[f"{price_meta['model_family']}@{price_meta['provider']}"] = price_meta
        _add_counter(counter, row, cost)
        _add_counter(by_provider[str(row.get("provider") or "unknown")], row, cost)
        _add_counter(by_model_family[str(row.get("model_family") or "unknown")], row, cost)
        _add_counter(by_route[str(row.get("requested_model") or row.get("route") or "unknown")], row, cost)
        _add_counter(by_served_model[str(row.get("served_model_id") or "unknown")], row, cost)
        by_status[str(int(row.get("status") or 0))] += 1
        enriched_rows.append(row)
    recent_total = len(enriched_rows)
    offset = int(options.get("offset") or 0)
    limit = int(options.get("limit") or 100)
    recent_sanitized = [_sanitize_key_usage_event(dict(r)) for r in enriched_rows[offset:offset + limit]]
    route_health = _route_health_snapshot(recent_sanitized, {})
    health_summary = _health_summary(recent_sanitized, route_health)
    meta = _consumer_meta(inferred_caller) if inferred_caller else {}
    snap_counter = _counter_snapshot(counter)
    return {
        "schema_version": 3,
        "kind": "router_key_usage",
        "detail_level": "full",
        "viewer": viewer,
        "generated_at": int(time.time()),
        "consumer": inferred_caller,
        "key_sha256_prefix": prefix,
        "key": _key_metadata_for_digest(digest, inferred_caller),
        "source": {"persistent_history": persistent, "history_path_configured": _usage_history_path() is not None},
        "window": {
            "since": options.get("since"),
            "until": options.get("until"),
            "window": options.get("window"),
            "limit": limit,
            "offset": offset,
            "recent_total": recent_total,
            "recent_returned": len(recent_sanitized),
        },
        "consumer_settings": {
            "status": meta.get("status"),
            "allowed_routes": meta.get("allowed_routes") or [],
            "rate_per_min": meta.get("rate_per_min") or RATE_PER_MIN,
            "burst": meta.get("burst") or BURST,
            "effective_per_min": max(int(meta.get("rate_per_min") or RATE_PER_MIN), int(meta.get("burst") or BURST)),
        },
        "totals": {
            "requests": snap_counter.get("requests", 0),
            "errors": snap_counter.get("errors", 0),
            "tokens_in": snap_counter.get("tokens_in", 0),
            "tokens_out": snap_counter.get("tokens_out", 0),
            "tokens_total": snap_counter.get("tokens_total", 0),
            "latency_ms_avg": snap_counter.get("latency_ms_avg", 0.0),
            "latency_ms_max": snap_counter.get("latency_ms_max", 0.0),
            "error_rate": snap_counter.get("error_rate", 0.0),
            "last_seen": snap_counter.get("last_seen"),
        },
        "cost_estimate": {
            "estimated": priced_events > 0,
            "usd": round(total_cost, 6),
            "priced_events": priced_events,
            "unpriced_events": max(0, len(enriched_rows) - priced_events),
            "source": str(_resolved_path(DASHBOARD_POLICY_METRICS_PATH)),
            "price_sources": dict(sorted(price_sources.items())),
        },
        "daily_totals": _period_totals(enriched_rows, monthly=False),
        "monthly_totals": _period_totals(enriched_rows, monthly=True),
        "by_provider": {k: _counter_snapshot(v) for k, v in sorted(by_provider.items())},
        "by_model_family": {k: _counter_snapshot(v) for k, v in sorted(by_model_family.items())},
        "by_route": {k: _counter_snapshot(v) for k, v in sorted(by_route.items())},
        "by_served_model": {k: _counter_snapshot(v) for k, v in sorted(by_served_model.items())},
        "by_status": dict(sorted(by_status.items())),
        "route_health": route_health,
        "health_summary": health_summary,
        "recent": recent_sanitized,
        "security": {
            "sanitized": True,
            "raw_api_key_exposed": False,
            "full_sha256_exposed": False,
            "provider_credentials_exposed": False,
        },
    }



DASHBOARD_TIMEFRAMES = {"runtime", "1h", "24h", "7d", "30d", "all"}


def _dashboard_timeframe(value: Any) -> str:
    text = str(value or "all").strip().lower()
    if text in {"", "history"}:
        return "all"
    if text not in DASHBOARD_TIMEFRAMES:
        return "all"
    return text


def _dashboard_timeframe_options(timeframe: str) -> dict[str, Any]:
    timeframe = _dashboard_timeframe(timeframe)
    if timeframe == "runtime" or timeframe == "all":
        return {"since": None, "until": None, "window": None, "limit": RECENT_LIMIT, "offset": 0, "timeframe": timeframe}
    since, window_label = _parse_usage_window(timeframe)
    return {"since": since, "until": None, "window": window_label, "limit": RECENT_LIMIT, "offset": 0, "timeframe": timeframe}


def _dashboard_history_rows(*, timeframe: str, consumer: str | None = None) -> list[dict[str, Any]]:
    opts = _dashboard_timeframe_options(timeframe)
    rows = _windowed_usage_events(_read_usage_history(events=("request", "reject")), opts)
    if consumer:
        rows = [r for r in rows if r.get("caller") == consumer]
    rows.sort(key=lambda r: int(r.get("ts") or 0), reverse=True)
    return rows


def _aggregate_usage_rows(rows: list[dict[str, Any]], *, selected: str | None = None) -> dict[str, Any]:
    totals = {"requests": 0, "rejects": 0, "errors": 0, "tokens_in": 0, "tokens_out": 0, "tokens_total": 0, "cost_usd": 0.0}
    prices = _price_table()
    by_caller: dict[str, dict[str, Any]] = defaultdict(_counter)
    by_provider: dict[str, dict[str, Any]] = defaultdict(_counter)
    by_model_family: dict[str, dict[str, Any]] = defaultdict(_counter)
    by_route: dict[str, dict[str, Any]] = defaultdict(_counter)
    by_served_model: dict[str, dict[str, Any]] = defaultdict(_counter)
    by_status: dict[str, int] = defaultdict(int)
    for row in rows:
        # History now carries reject events alongside requests (persisted by
        # _record_reject); they count as rejects, never as requests, and skip
        # the request-shaped aggregation below.
        if (row.get("event") or "request") != "request":
            if row.get("event") == "reject":
                totals["rejects"] += 1
            continue
        status = int(row.get("status") or 0)
        cost, _ = _cost_for_event(row, prices)
        totals["requests"] += 1
        if status >= 400:
            totals["errors"] += 1
        totals["tokens_in"] += int(row.get("tokens_in") or 0)
        totals["tokens_out"] += int(row.get("tokens_out") or 0)
        totals["tokens_total"] += int(row.get("tokens_total") or (int(row.get("tokens_in") or 0) + int(row.get("tokens_out") or 0)))
        if cost is not None:
            totals["cost_usd"] = round(totals["cost_usd"] + cost, 6)
        by_status[str(status)] += 1
        for bucket, key in (
            (by_caller, row.get("caller") or "unknown"),
            (by_provider, row.get("provider") or "unknown"),
            (by_model_family, row.get("model_family") or "unknown"),
            (by_route, row.get("requested_model") or row.get("route") or "unknown"),
            (by_served_model, row.get("served_model_id") or "unknown"),
        ):
            _add_counter(bucket[str(key or "unknown")], row, cost)
    by_caller_snap = {k: _counter_snapshot(v) for k, v in sorted(by_caller.items())}
    return {
        "totals": totals,
        "by_caller": {selected: by_caller_snap.get(selected, _counter_snapshot(_counter()))} if selected else by_caller_snap,
        "by_caller_all": by_caller_snap,
        "by_provider": {k: _counter_snapshot(v) for k, v in sorted(by_provider.items())},
        "by_model_family": {k: _counter_snapshot(v) for k, v in sorted(by_model_family.items())},
        "by_route": {k: _counter_snapshot(v) for k, v in sorted(by_route.items())},
        "by_served_model": {k: _counter_snapshot(v) for k, v in sorted(by_served_model.items())},
        "by_status": dict(sorted(by_status.items())),
        "recent": [{k: v for k, v in r.items() if k != "key_sha256"} for r in rows[:RECENT_LIMIT]],
    }


def _stats_snapshot(*, viewer: str, upstream_status: int, upstream_health: dict[str, Any], consumer: str | None = None, timeframe: str = "all", key_sha256: str | None = None, viewer_role: str = "admin", provider: str | None = None, model: str | None = None) -> dict[str, Any]:
    selected = consumer if consumer in _consumers() else None
    provider = (provider or "").strip() or None
    model = (model or "").strip() or None
    key_sha256 = str(key_sha256 or "").strip().lower()
    key_filter = key_sha256 if re.fullmatch(r"[a-f0-9]{64}", key_sha256) else None
    timeframe = _dashboard_timeframe(timeframe)
    if timeframe != "runtime":
        all_rows = _dashboard_history_rows(timeframe=timeframe)
        if key_filter:
            rows = [r for r in all_rows if r.get("caller") == selected and r.get("key_sha256") == key_filter]
            table_rows = rows
        else:
            rows = [r for r in all_rows if r.get("caller") == selected] if selected else all_rows
            table_rows = all_rows
        # Analytics filters (provider / model family) narrow the aggregated rows
        # so totals, breakdowns and the time series all reflect the selection.
        if provider:
            rows = [r for r in rows if r.get("provider") == provider]
        if model:
            rows = [r for r in rows if r.get("model_family") == model]
        agg = _aggregate_usage_rows(rows, selected=selected)
        all_agg = _aggregate_usage_rows(table_rows)
        # Surface live in-memory events (dashboard test calls, probes, just-served
        # requests) that are not persisted to billing history, so Activity always
        # shows the latest activity regardless of timeframe. Display-only: counters
        # and totals stay history-based.
        with _stats_lock:
            synthetic = dict(_stats["synthetic_route_health"])
            live = [_sanitize_key_usage_event(dict(r)) for r in _stats["recent"]
                    if (not selected or r.get("caller") == selected)
                    and (not key_filter or r.get("key_sha256_prefix") == (key_filter[:12] if key_filter else None))]
        seen_ids = {r.get("usage_event_id") for r in agg["recent"] if r.get("usage_event_id")}
        merged_recent = [r for r in live if r.get("usage_event_id") not in seen_ids] + agg["recent"]
        route_health = _route_health_snapshot(agg["recent"], synthetic)
        health_summary = _health_summary(agg["recent"], route_health)
        return {
            "viewer": viewer,
            "generated_at": int(time.time()),
            "uptime_s": round(time.time() - _started_wall, 1),
            "viewer_role": viewer_role,
            "selected_consumer": selected,
            "selected_key_sha256_prefix": key_filter[:12] if key_filter else None,
            "timeframe": {"selected": timeframe, "source": "persistent_history", "history_path_configured": _usage_history_path() is not None, "history_events": len(rows), "history_events_all": len(all_rows)},
            "rate_limit": {"rate_per_min": RATE_PER_MIN, "burst": BURST, "effective_per_min": max(RATE_PER_MIN, BURST)},
            "upstream": {"status": upstream_status, "health": upstream_health},
            "consumers": [{"name": name, "configured": True} for name in _consumers()],
            "keys": [row for row in _consumer_key_rows(all_agg["by_caller_all"]) if row.get("consumer") == selected] if key_filter and selected else _consumer_key_rows(all_agg["by_caller_all"]),
            "totals": agg["totals"],
            "by_caller": agg["by_caller"],
            "by_provider": agg["by_provider"],
            "by_model_family": agg["by_model_family"],
            "by_route": agg["by_route"],
            "by_served_model": agg["by_served_model"],
            "by_status": agg["by_status"],
            "recent": merged_recent,
            "filter_options": {"providers": sorted((all_agg.get("by_provider") or {}).keys()),
                               "models": sorted((all_agg.get("by_model_family") or {}).keys())},
            "daily_totals": _period_totals(rows, monthly=False),
            "route_health": route_health,
            "health_summary": health_summary,
            "logins": _login_connections_snapshot(timeframe=timeframe, consumer=selected, viewer_role=viewer_role),
            "provider_keys": _provider_credentials_snapshot(timeframe=timeframe, viewer_role=viewer_role),
        }
    with _stats_lock:
        by_caller_all = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_caller"].items())}
        if selected:
            if key_filter:
                selected_counter = _counter_snapshot(_stats["by_key_sha256"].get(key_filter, _counter()))
                recent = [r for r in _stats["recent"] if r.get("caller") == selected and r.get("key_sha256_prefix") == key_filter[:12]]
                by_provider = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_key_provider"].get(key_filter, {}).items())}
                by_model_family = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_key_model_family"].get(key_filter, {}).items())}
                by_route = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_key_route"].get(key_filter, {}).items())}
                by_served_model = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_key_served_model"].get(key_filter, {}).items())}
                by_status = dict(sorted(_stats["by_key_status"].get(key_filter, {}).items()))
                by_caller_all = {selected: selected_counter}
            else:
                selected_counter = by_caller_all.get(selected, _counter_snapshot(_counter()))
                recent = [r for r in _stats["recent"] if r.get("caller") == selected]
                by_provider = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_caller_provider"].get(selected, {}).items())}
                by_model_family = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_caller_model_family"].get(selected, {}).items())}
                by_route = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_caller_route"].get(selected, {}).items())}
                by_served_model = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_caller_served_model"].get(selected, {}).items())}
                by_status = dict(sorted(_stats["by_caller_status"].get(selected, {}).items()))
            totals = {
                "requests": selected_counter.get("requests", 0),
                "rejects": 0,
                "errors": selected_counter.get("errors", 0),
                "tokens_in": selected_counter.get("tokens_in", 0),
                "tokens_out": selected_counter.get("tokens_out", 0),
                "tokens_total": selected_counter.get("tokens_total", 0),
            }
            synthetic = dict(_stats["synthetic_route_health"])
            route_health = _route_health_snapshot(recent, synthetic)
            health_summary = _health_summary(recent, route_health)
            by_caller = {selected: selected_counter}
        else:
            totals = {"requests": _stats["total_requests"], "rejects": _stats["total_rejects"], "errors": _stats["total_errors"], "tokens_in": _stats["total_tokens_in"], "tokens_out": _stats["total_tokens_out"], "tokens_total": _stats["total_tokens"]}
            by_provider = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_provider"].items())}
            by_model_family = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_model_family"].items())}
            by_route = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_route"].items())}
            by_served_model = {k: _counter_snapshot(v) for k, v in sorted(_stats["by_served_model"].items())}
            by_status = dict(sorted(_stats["by_status"].items()))
            recent = list(_stats["recent"])
            synthetic = dict(_stats["synthetic_route_health"])
            route_health = _route_health_snapshot(recent, synthetic)
            health_summary = _health_summary(recent, route_health)
            by_caller = by_caller_all
        return {
            "viewer": viewer,
            "generated_at": int(time.time()),
            "uptime_s": round(time.time() - _started_wall, 1),
            "viewer_role": viewer_role,
            "selected_consumer": selected,
            "selected_key_sha256_prefix": key_filter[:12] if key_filter else None,
            "timeframe": {"selected": "runtime", "source": "memory", "history_path_configured": _usage_history_path() is not None},
            "rate_limit": {"rate_per_min": RATE_PER_MIN, "burst": BURST, "effective_per_min": max(RATE_PER_MIN, BURST)},
            "upstream": {"status": upstream_status, "health": upstream_health},
            "consumers": [{"name": name, "configured": True} for name in _consumers()],
            "keys": [row for row in _consumer_key_rows(by_caller_all) if row.get("consumer") == selected] if key_filter and selected else _consumer_key_rows(by_caller_all),
            "totals": totals,
            "by_caller": by_caller,
            "by_provider": by_provider,
            "by_model_family": by_model_family,
            "by_route": by_route,
            "by_served_model": by_served_model,
            "by_status": by_status,
            "recent": recent,
            "filter_options": {"providers": sorted((_stats["by_provider"] or {}).keys()),
                               "models": sorted((_stats["by_model_family"] or {}).keys())},
            "route_health": route_health,
            "health_summary": health_summary,
            "logins": _login_connections_snapshot(timeframe="runtime", consumer=selected, viewer_role=viewer_role),
            "provider_keys": _provider_credentials_snapshot(timeframe="runtime", viewer_role=viewer_role),
        }



def _public_base_url() -> str:
    """Public OpenAI-compatible base URL shown to users in the key-handoff blurb.
    Override with PUBLIC_BASE_URL; defaults to the loopback dev address."""
    return (os.getenv("PUBLIC_BASE_URL") or "http://127.0.0.1:8080/v1").rstrip("/")


def _dashboard_html() -> str:
    html = """<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width,initial-scale=1' />
  <meta name='robots' content='noindex,nofollow,noarchive' />
  <title>unhardcoded dashboard</title>
  <style>
    :root{color-scheme:dark;--bg:#08090a;--rail:#0b0c0f;--panel:#0f1011;--surface:#141518;--surface2:#191a1f;--line:rgba(255,255,255,.075);--line2:rgba(255,255,255,.045);--fg:#f7f8f8;--muted:#8a8f98;--muted2:#62666d;--accent:#7170ff;--accent2:#5e6ad2;--ok:#27a644;--warn:#f5b84b;--bad:#ff5c7a;--shadow:0 24px 80px rgba(0,0,0,.42)}
    *{box-sizing:border-box}html,body{min-height:100%;background:#08090a}body{margin:0;background:radial-gradient(circle at 20% -10%,rgba(113,112,255,.18),transparent 34%),linear-gradient(180deg,#090a0c,#060709 46%,#08090a);background-repeat:no-repeat;background-attachment:fixed;background-size:100vw 100vh,100vw 100vh;color:var(--fg);font:13px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;font-feature-settings:'cv01','ss03'}button,input,select,textarea{font:inherit}button{cursor:pointer}code,.mono{font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.hidden{display:none!important}.muted{color:var(--muted)}.small{font-size:12px}.right{text-align:right}.empty{color:var(--muted);padding:22px;text-align:center;border:1px dashed var(--line);border-radius:12px;background:rgba(255,255,255,.018)}
    .shell{display:grid;grid-template-columns:244px minmax(0,1fr);min-height:100vh}.sidebar{position:sticky;top:0;height:100vh;padding:18px 14px;border-right:1px solid var(--line2);background:rgba(8,9,10,.72);backdrop-filter:blur(18px);display:flex;flex-direction:column;gap:18px}.brand{display:flex;align-items:center;gap:11px;padding:4px 6px}.brandMark{width:30px;height:30px;border-radius:10px;background:linear-gradient(135deg,#5e6ad2,#7c3aed);box-shadow:0 0 0 1px rgba(255,255,255,.18) inset,0 12px 40px rgba(113,112,255,.3)}.brandTitle{font-weight:590;letter-spacing:-.01em}.sub{color:var(--muted);font-size:12px}.nav{display:flex;flex-direction:column;gap:4px}.navSection{font-size:10px;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);font-weight:600;padding:14px 10px 3px}.tab{height:36px;width:100%;display:flex;align-items:center;gap:10px;border:0;border-radius:9px;background:transparent;color:#b9c0cc;text-align:left;padding:0 10px;font-weight:510}.tab:hover{background:rgba(255,255,255,.035);color:var(--fg)}.tab.active{background:rgba(113,112,255,.13);color:var(--fg);box-shadow:0 0 0 1px rgba(113,112,255,.16) inset}.navIcon{width:18px;text-align:center;opacity:.92}.sideFoot{margin-top:auto;border:1px solid var(--line);background:rgba(255,255,255,.025);border-radius:14px;padding:12px;color:var(--muted);font-size:12px}
    .content{min-width:0;padding:24px 26px 40px}.topbar{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;margin-bottom:20px}.pageTitle h1{margin:0;font-size:26px;line-height:1.05;font-weight:590;letter-spacing:-.55px}.topActions{display:flex;gap:8px;align-items:center}.select,.input,input,select,textarea{height:34px;background:rgba(255,255,255,.025);border:1px solid var(--line);border-radius:9px;color:var(--fg);padding:0 11px;outline:none;min-width:0}textarea{height:auto;padding:10px 11px}.select{min-width:190px}select option,select optgroup{background:#141518;color:var(--fg)}select optgroup{color:var(--muted);font-weight:600}select option:checked{background:#23252b}.actRow{cursor:pointer}.actToggle{color:var(--muted);width:16px;text-align:center}.actDetailBox{padding:8px 2px 12px}.actMeta{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}.actStep{display:flex;align-items:center;gap:8px;padding:3px 0;font-size:12px}.stepN{display:inline-grid;place-items:center;width:18px;height:18px;border-radius:50%;background:rgba(255,255,255,.06);font-size:11px;color:var(--muted);flex:none}.actStep code{background:rgba(255,255,255,.05);padding:1px 6px;border-radius:6px}.btn{height:34px;border:1px solid var(--line);border-radius:9px;background:rgba(255,255,255,.045);color:var(--fg);padding:0 12px;font-weight:510;box-shadow:0 1px 0 rgba(255,255,255,.04) inset}.btn:hover{background:rgba(255,255,255,.075)}.btn.primary{border-color:rgba(113,112,255,.45);background:linear-gradient(180deg,#7170ff,#5e6ad2);color:white}.btn.danger{border-color:rgba(255,92,122,.35);color:#ffd2da}.iconBtn{width:30px;height:30px;padding:0;border-radius:8px;display:inline-grid;place-items:center}.ghost{background:transparent;border-color:transparent;color:var(--muted)}.ghost:hover{background:rgba(255,255,255,.055);color:var(--fg)}
    .grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px}.card{background:linear-gradient(180deg,rgba(255,255,255,.042),rgba(255,255,255,.022));border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);overflow:hidden}.cardPad{padding:16px}.span3{grid-column:span 3}.span4{grid-column:span 4}.span5{grid-column:span 5}.span6{grid-column:span 6}.span7{grid-column:span 7}.span8{grid-column:span 8}.span12{grid-column:span 12}.label{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);font-weight:590}.metric{font-size:28px;line-height:1.1;font-weight:590;letter-spacing:-.6px;margin-top:8px}.statSub{color:var(--muted);font-size:12px;margin-top:4px}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;background:rgba(255,255,255,.028);padding:3px 8px;color:#d0d6e0;font-weight:510;font-size:12px;white-space:nowrap}.dot{width:7px;height:7px;border-radius:50%;background:var(--muted2)}.dot.ok{background:var(--ok)}.dot.warn{background:var(--warn)}.dot.bad{background:var(--bad)}
    .healthBanner{display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;border-radius:14px;padding:14px;border:1px solid var(--line);background:rgba(255,255,255,.028)}.healthBanner.down{border-color:rgba(255,92,122,.42);background:linear-gradient(90deg,rgba(255,92,122,.16),rgba(255,255,255,.025))}.healthBanner.degraded{border-color:rgba(245,184,75,.42);background:linear-gradient(90deg,rgba(245,184,75,.13),rgba(255,255,255,.025))}.healthBanner.ok{border-color:rgba(39,166,68,.32)}.healthState{font-size:18px;font-weight:590;text-transform:capitalize}.healthMeta{display:flex;gap:8px;flex-wrap:wrap;margin-top:7px}.errorChips{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
    .toolbar{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:12px 14px;border-bottom:1px solid var(--line2);background:rgba(255,255,255,.018)}.toolbarLeft,.toolbarRight{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.search{width:260px}.seg{display:inline-flex;border:1px solid var(--line);background:rgba(255,255,255,.02);border-radius:10px;padding:2px}.seg button{height:28px;border:0;background:transparent;color:var(--muted);border-radius:8px;padding:0 10px;font-weight:510}.seg button.active{background:rgba(255,255,255,.08);color:var(--fg)}
    table{width:100%;border-collapse:separate;border-spacing:0}.dataTable th{height:34px;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em;font-weight:590;text-align:left;padding:0 14px;border-bottom:1px solid var(--line2)}.dataTable td{padding:11px 14px;border-bottom:1px solid var(--line2);vertical-align:middle}.dataTable tbody tr:hover,.dataTable tbody tr.selectedRow{background:rgba(255,255,255,.045)}.nameCell{display:flex;align-items:center;gap:10px}.avatar{width:26px;height:26px;border-radius:8px;display:grid;place-items:center;background:rgba(113,112,255,.12);color:#c9c8ff;font-weight:590;text-transform:uppercase}.rowTitle{font-weight:590}.rowMeta{font-size:12px;color:var(--muted);margin-top:1px}.actions{display:flex;gap:4px;justify-content:flex-end}.statusText{font-weight:510}.keyStack{display:flex;flex-direction:column;gap:3px}.hashText{font-size:12px;color:var(--muted);max-width:230px;overflow:hidden;text-overflow:ellipsis}.json{white-space:pre-wrap;max-height:360px;overflow:auto;background:#08090a;border:1px solid var(--line);border-radius:12px;padding:12px;color:#aab0bb}.errorbox{display:none;margin:0 0 14px;border:1px solid rgba(255,92,122,.35);background:rgba(255,92,122,.1);border-radius:12px;padding:11px;color:#ffd2da}.login{max-width:480px;margin:12vh auto;padding:22px}.login h2{margin:8px 0 6px;font-size:24px;letter-spacing:-.4px}.login .row{display:flex;gap:8px;margin-top:14px}.login input{flex:1;height:38px}.page{display:grid}.cardsOnly{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px}
    .drawerShade{position:fixed;inset:0;background:rgba(0,0,0,.55);backdrop-filter:blur(6px);z-index:50;display:none}.drawerShade.open{display:block}.drawer{position:absolute;right:0;top:0;height:100%;width:min(460px,100vw);background:rgba(15,16,17,.98);border-left:1px solid var(--line);box-shadow:-30px 0 90px rgba(0,0,0,.55);padding:18px;overflow:auto}.drawerHead{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px}.drawer h2{margin:0;font-size:20px;letter-spacing:-.35px}.drawerSection{border:1px solid var(--line);border-radius:14px;padding:14px;margin-top:10px;background:rgba(255,255,255,.025)}.formGrid{display:grid;gap:9px}.formGrid label{display:grid;gap:5px;color:var(--muted);font-size:12px}.formGrid input,.formGrid select{width:100%;height:36px}.checkRow{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px}.checkRow input{width:auto;height:auto}.resultBox{margin-top:10px}.resultBox textarea{width:100%;min-height:88px}.toast{position:fixed;right:18px;bottom:18px;z-index:80;border:1px solid var(--line);background:#15161a;border-radius:12px;padding:10px 12px;color:var(--fg);box-shadow:var(--shadow);display:none}.policy-card{padding:16px;border-bottom:1px solid var(--line2)}.policy-card:last-child{border-bottom:0}.policy-head{display:flex;justify-content:space-between;gap:12px;margin-bottom:12px}.policy-head h3{margin:0}.policy-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px}.summaryBox{border:1px solid var(--line);border-radius:12px;padding:10px;background:rgba(255,255,255,.02)}.summaryBox .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.summaryBox .v{margin-top:4px}.model-row{display:grid;grid-template-columns:180px 1fr;gap:10px;margin-top:8px}.family-card,.chain-card,.rule-card{border:1px solid var(--line);border-radius:12px;padding:10px;background:rgba(255,255,255,.02)}.provider-timeline{display:flex;gap:8px;overflow:auto;padding-top:8px}.provider-chip{min-width:170px;border:1px solid var(--line);border-radius:12px;padding:9px;background:rgba(255,255,255,.025)}.provider-chip.failing{border-color:#b3261e;background:rgba(179,38,30,.08)}.provider-chip .chipReason{font-size:11px;color:#e57373;margin-top:4px;word-break:break-word}.chipBalance.bad{color:#e57373}.chipBalance.warn{color:#e0b34c}.stepBadge{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;background:var(--accent2);font-size:12px}.modelId,.meta{font-size:12px;color:var(--muted);word-break:break-word}.providerName,.familyName,.name{font-weight:590}.flow-title{color:var(--muted);font-size:12px;margin:8px 0}.empty-policy-note{padding:12px;color:var(--muted)}
    @media(max-width:1040px){.shell{grid-template-columns:1fr}.sidebar{position:static;height:auto;flex-direction:row;align-items:center;overflow:auto}.sideFoot{display:none}.nav{flex-direction:row}.tab{white-space:nowrap}.content{padding:18px}.span3,.span4,.span5,.span6,.span7,.span8{grid-column:span 12}.topbar{flex-direction:column}.topActions{width:100%;flex-wrap:wrap}.select{flex:1}.search{width:100%}.policy-summary,.model-row{grid-template-columns:1fr}.healthBanner{grid-template-columns:1fr}.errorChips{justify-content:flex-start}}
  </style>
</head>
<body>
<div class='shell'>
  <aside class='sidebar'>
    <div class='brand'><div class='brandMark'></div><div><div class='brandTitle'>unhardcoded</div><div class='sub'>Policy router</div></div></div>
    <nav class='nav' aria-label='Dashboard'>
      <button class='tab active' id='tabOverview' data-tab='overview' type='button'><span class='navIcon'>⌁</span>Analytics</button>
      <button class='tab' id='tabBuilder' data-tab='builder' type='button'><span class='navIcon'>⚒</span>Builder</button>
      <button class='tab' id='tabActivity' data-tab='activity' type='button'><span class='navIcon'>◷</span>Activity</button>
      <button class='tab' id='tabMarket' data-tab='market' type='button'><span class='navIcon'>⚖</span>Catalog</button>
      <button class='tab' id='tabConfig' data-tab='config' type='button'><span class='navIcon'>⚙</span>Config</button>
      <div class='navSection'>Settings</div>
      <button class='tab' id='tabConsumers' data-tab='consumers' type='button'><span class='navIcon'>◉</span>Consumers</button>
      <button class='tab' id='tabProviderKeys' data-tab='providerKeys' type='button'><span class='navIcon'>◌</span>Provider keys</button>
      <button class='tab hidden' id='tabKeyUsage' data-tab='keyUsage' type='button'><span class='navIcon'>◫</span>Key usage</button>
    </nav>
    <div class='sideFoot'>Operator console for live request health, consumer keys, route probes, and policy diagrams.<br><br><code>/dashboard/api/full</code></div>
  </aside>
  <main class='content'>
    <div class='topbar'>
      <div class='pageTitle'><h1 id='pageTitle'>Analytics</h1><div class='sub' id='pageSub'>Spend, traffic and errors — filter by timeframe, consumer, provider and model.</div></div>
      <div class='topActions'><select class='select' id='consumer'><option value=''>All consumers</option></select><select class='select' id='timeframe' title='Usage timeframe'><option value='all' selected>All history</option><option value='runtime'>Since restart</option><option value='1h'>Last hour</option><option value='24h'>Last 24h</option><option value='7d'>Last 7d</option><option value='30d'>Last 30d</option></select><button class='btn' id='refresh'>Refresh</button><button class='btn' id='logout'>Log out</button></div>
    </div>
    <div id='err' class='errorbox'></div>
    <section id='login' class='card login hidden'><div class='label'>Dashboard login</div><h2>Welcome back</h2><p class='muted'>Admins can use the dashboard password. Consumers can paste their router API key to see only their own usage.</p><div class='formGrid' style='margin-top:14px'><label>Admin password<input id='password' type='password' placeholder='Dashboard password' autocomplete='current-password' /></label><button class='btn primary' id='loginBtn'>Admin log in</button><label>Consumer API key<input id='apiKeyLogin' type='password' placeholder='Router API key' autocomplete='off' /></label><button class='btn' id='apiKeyLoginBtn'>View my usage</button></div></section>

    <section class='grid hidden page' id='app'>
      <div class='card cardPad span12'><div class='toolbar'><div class='label'>Filters</div><div style='margin-left:auto;display:flex;gap:8px;align-items:center'><span class='muted small'>timeframe &amp; consumer: top right</span><select id='anProvider'><option value=''>All providers</option></select><select id='anModel'><option value=''>All models</option></select></div></div></div>
      <div class='card cardPad span3'><div class='label'>Requests</div><div id='anRequests' class='metric'>0</div><div id='anReqSub' class='statSub'>—</div></div>
      <div class='card cardPad span3'><div class='label'>Spend</div><div id='anSpend' class='metric'>$0</div><div id='anSpendSub' class='statSub'>—</div></div>
      <div class='card cardPad span3'><div class='label'>Tokens</div><div id='anTokens' class='metric'>0</div><div id='anTokSub' class='statSub'>—</div></div>
      <div class='card cardPad span3'><div class='label'>Success rate</div><div id='anSuccess' class='metric'>—</div><div id='anSuccessSub' class='statSub'>—</div></div>
      <div class='card span12'><div class='toolbar'><div class='label'>Requests &amp; spend over time</div></div><div id='anSeries' class='cardPad'></div></div>
      <div class='card span6'><div class='toolbar'><div class='label'>By provider</div></div><div id='anByProvider'></div></div>
      <div class='card span6'><div class='toolbar'><div class='label'>By model family</div></div><div id='anByModel'></div></div>
      <div class='card span6'><div class='toolbar'><div class='label'>By consumer</div></div><div id='anByConsumer'></div></div>
      <div class='card span6'><div class='toolbar'><div class='label'>By status</div></div><div id='anByStatus'></div></div>
    </section>

    <section class='grid hidden page' id='consumersPage'>
      <div class='card span12'>
        <div class='toolbar'><div class='toolbarLeft'><div class='label'>Consumers</div><input id='consumerSearch' class='input search' placeholder='Search consumers…' /><div class='seg' id='consumerStatusSeg'><button data-status='' class='active'>All</button><button data-status='active'>Active</button><button data-status='inactive'>Inactive</button></div></div><div class='toolbarRight'><button class='btn primary' id='newConsumerKey'>Generate key</button></div></div>
        <div id='keys'></div>
      </div>
      <div class='card span12'><div id='consumerDetail'></div></div>
    </section>

    <section class='grid hidden page' id='providerKeysPage'>
      <div class='card span12'><div class='toolbar'><div><div class='label'>LLM provider credentials</div><div class='muted small'>Privatized view: env names and 12-char key fingerprints only. No raw provider keys or full hashes.</div></div><div class='toolbarRight'><button class='btn primary' id='toggleAddProvider'>Add provider</button></div></div><div id='providerKeys'></div></div>
      <div class='card span12' id='addProviderCard' style='display:none'><div class='toolbar'><div><div class='label'>Add provider</div><div class='muted small'>OpenAI-compatible endpoints only. The key is stored in .env.secrets under the env var; the provider definition persists in providers.local.json and goes live immediately.</div></div></div><div class='cardPad'><div class='formGrid'><label>Provider id<input id='addProvId' placeholder='groq' /></label><label>Base URL<input id='addProvBaseUrl' placeholder='https://api.groq.com/openai/v1' /></label><label>Tier<select id='addProvTier' class='select'><option value='partner'>partner</option><option value='fallback'>fallback</option></select></label><label>Key env var<input id='addProvEnv' placeholder='GROQ_API_KEY' /></label><label>API key<input id='addProvKey' type='password' placeholder='sk-…' autocomplete='off' /></label><label>Served models (one per line: family or family=provider_model_id)<textarea id='addProvModels' rows='3' placeholder='llama-3.3-70b=llama-3.3-70b-versatile'></textarea></label></div><div class='actions' style='margin-top:10px'><button class='btn primary' id='addProvSubmit'>Add provider</button><button class='btn' id='addProvCancel'>Cancel</button></div><div id='addProvResult' class='muted small' style='margin-top:8px'></div></div></div>
      <div class='card span12'><div class='toolbar'><div><div class='label'>Codex accounts</div><div class='muted small'>ChatGPT-subscription auth.json accounts (paste the output of `codex login`). Stored on the PVC, applied live. Token fingerprints only — never the raw token.</div></div><div class='toolbarRight'><button class='btn primary' id='toggleAddCodex'>Add codex account</button></div></div><div id='codexAccounts'></div></div>
      <div class='card span12' id='addCodexCard' style='display:none'><div class='cardPad'><div class='formGrid'><label>Account name<input id='addCodexName' placeholder='team-1' /></label><label>auth.json<textarea id='addCodexJson' rows='6' placeholder='{&quot;tokens&quot;:{&quot;access_token&quot;:&quot;...&quot;,&quot;refresh_token&quot;:&quot;...&quot;,&quot;account_id&quot;:&quot;...&quot;}}'></textarea></label></div><div class='actions' style='margin-top:10px'><button class='btn primary' id='addCodexSubmit'>Save account</button><button class='btn' id='addCodexCancel'>Cancel</button></div><div id='addCodexResult' class='muted small' style='margin-top:8px'></div></div></div>
    </section>


    <section class='grid hidden page' id='keyUsagePage'>
      <div class='card span12'><div class='toolbar'><div class='toolbarLeft'><div class='label'>Per-key usage lookup</div><input id='keyUsageApiKey' class='input search' placeholder='Paste API key' /><input id='keyUsageSince' class='input' placeholder='since unix/ISO' /><input id='keyUsageWindow' class='input' placeholder='window e.g. 24h' /><input id='recentLimit' class='input' value='50' style='width:76px' /><input id='recentOffset' class='input' value='0' style='width:76px' /></div><div class='toolbarRight'><button class='btn primary' id='loadKeyUsage'>Load usage</button></div></div><div id='keyUsageResult' class='cardPad'><div class='empty'>Paste a key to view persistent totals, windows, cost_estimate, daily/monthly totals, and paginated recent requests.</div></div></div>
    </section>

    <section class='hidden page' id='builderPage'><div class='card'>
      <div class='toolbar' style='margin-bottom:4px'><div class='seg' id='builderKindSeg'><button data-kind='policy' class='active' type='button'>Policy builder</button><button data-kind='flow' type='button'>Flow builder</button></div><span class='muted small' style='margin-left:auto'>a Σ_pol policy · or a Σ_flow DAG of policies</span></div>
      <div id='policyBuilder'>
      <div class='toolbar'><div class='label'>Policy builder</div><div class='seg' id='bModeSeg' style='margin-left:auto'><button data-mode='structured' class='active' type='button'>Structured</button><button data-mode='raw' type='button'>Raw term</button></div></div>
      <div class='muted small' style='margin:2px 0 10px'>A policy is <b>one pass</b> over the candidate models: <b>Filter</b> (which qualify — conditions joined by AND) → <b>Score</b> (rank the survivors — a weighted sum of terms, higher wins) → <b>Pick</b> (the selector: argmax / sample / top&nbsp;N). Use the <b>Structured ↔ Raw term</b> toggle (top-right) to see or hand-edit the Σ_pol term as data. <b>Review ranking</b> shows how this host would order providers under it; <b>Download policy</b> saves the term to run per call.</div>
      <div class='toolbar' style='margin-bottom:12px'><span class='muted small'>Load an example →</span><button class='btn' id='bEx1' type='button'>Cheapest in the top-5 (intelligence ∩ coding)</button><button class='btn' id='bEx2' type='button'>Top 3 by combined benchmarks</button></div>
      <div id='bStructured'>
        <div class='label small' style='margin-top:4px'>Filter</div>
        <div id='bFilters'></div>
        <div class='toolbar' style='margin-top:6px'><button class='btn' id='bAddCond' type='button'>+ condition</button><button class='btn' id='bAddOr' type='button'>+ OR-group</button><span class='muted small'>base always applied: requirements met · not disabled</span></div>
        <div class='label small' style='margin-top:18px'>Score</div>
        <div id='bScores'></div>
        <div class='toolbar' style='margin-top:6px'><button class='btn' id='bAddScore' type='button'>+ term</button><label class='checkRow'><input type='checkbox' id='bGateBreaker' checked> demote breaker-open to 0 (gate · keep as last resort)</label></div>
        <div class='label small' style='margin-top:18px'>Pick</div>
        <div class='toolbar' style='margin-top:6px'><select id='b_selector'><option value='argmax'>argmax — always the best</option><option value='sample'>sample — rank-geometric draw</option></select><label id='bTempWrap' style='display:none'>temp&nbsp;<input id='b_temp' type='number' min='0' step='0.1' value='1.0' style='width:80px'></label><label>limit to top&nbsp;<input id='b_topn' type='number' min='1' step='1' placeholder='all' style='width:70px'></label><span class='muted small'>e.g. the 5 best — wraps the selector in top_k</span></div>
      </div>
      <div id='bRaw' style='display:none'>
        <div class='muted small' style='margin:4px 0'>The Σ_pol term, as data. Edits here override the structured form for Review &amp; Download — this is the full algebra (OR, gate, when, chain, failplan, evidence).</div>
        <textarea id='bRawTerm' class='mono small' spellcheck='false' style='width:100%;min-height:280px;background:rgba(255,255,255,.02);border:1px solid var(--line);border-radius:10px;padding:10px'></textarea>
      </div>
      <datalist id='familyOptions'></datalist>
      <div class='toolbar' style='margin-top:14px'><button class='btn primary' id='bReview'>Review ranking</button><button class='btn' id='bDownload'>Download policy</button><span class='muted small mono' id='bFingerprint' style='margin-left:auto'></span></div>
      <div id='bError' class='empty' style='display:none'></div>
      <pre id='bTerm' class='mono small' style='overflow:auto;max-height:200px;background:rgba(255,255,255,.02);border:1px solid var(--line);border-radius:10px;padding:10px'></pre>
      <div id='bRanked'></div>
      <div class='toolbar' style='margin-top:16px'><div class='label'>Test call</div><span class='muted small' style='margin-left:auto'>run this policy live with a prompt — appears in Activity with its full trace</span></div>
      <div style='display:flex;gap:8px;align-items:flex-start;margin-top:6px'><textarea id='bTestPrompt' rows='2' placeholder='Reply exactly: pong' style='flex:1'></textarea><button class='btn primary' id='bTestBtn' style='flex:none'>Test call</button></div>
      <div id='bTestResult'></div>
      </div>
      <div id='flowBuilder' style='display:none'>
        <div class='muted small' style='margin:2px 0 12px'>A <b>flow</b> is a DAG of model calls: the <b>input</b> <code>u</code> is the user's prompt, the <b>output</b> is the answer, and each <b>node</b> runs its own <b>policy</b> (the same Σ_pol builder) under a <b>system prompt</b> over the outputs of the nodes feeding it. A node with several inputs is a <b>fusion</b>. <b>Review flow</b> stamps its identity; <b>Test call</b> runs it and shows the per-node trace in Activity.</div>
        <div class='toolbar'><span class='muted small'>Load an example →</span><button class='btn' id='fEx1' type='button'>Mixture-of-agents (2 drafts → synthesize)</button></div>
        <div class='label small' style='margin-top:12px'>Nodes <span class='muted small'>· input <code>u</code> = the user prompt; each node may read <code>u</code> and any earlier node</span></div>
        <div id='fNodes'></div>
        <div class='toolbar' style='margin-top:8px'><button class='btn' id='fAddNode' type='button'>+ node</button><label style='margin-left:auto'>answer from&nbsp;<select id='fOutput'></select></label></div>
        <div class='toolbar' style='margin-top:14px'><button class='btn primary' id='fReview'>Review flow</button><button class='btn' id='fDownload'>Download flow</button><span class='muted small mono' id='fFingerprint' style='margin-left:auto'></span></div>
        <div id='fError' class='empty' style='display:none'></div>
        <pre id='fTerm' class='mono small' style='overflow:auto;max-height:200px;background:rgba(255,255,255,.02);border:1px solid var(--line);border-radius:10px;padding:10px'></pre>
        <div class='toolbar' style='margin-top:16px'><div class='label'>Test call</div><span class='muted small' style='margin-left:auto'>run this flow live with a prompt — appears in Activity with its per-node trace</span></div>
        <div style='display:flex;gap:8px;align-items:flex-start;margin-top:6px'><textarea id='fTestPrompt' rows='2' placeholder='What is 17 * 23?' style='flex:1'></textarea><button class='btn primary' id='fTestBtn' style='flex:none'>Test call</button></div>
        <div id='fTestResult'></div>
      </div>
    </div></section>
    <section class='hidden page' id='marketPage'><div class='card'><div class='toolbar'><div class='label'>Catalog</div><input class='search' id='marketSearch' placeholder='Filter families…' style='margin-left:auto'><label class='checkRow'><input type='checkbox' id='tradableOnly'> Tradable only</label><button class='btn' id='marketCopy' title='Copy the current catalog view as JSON'>⧉ Copy</button><button class='btn primary' id='marketSkill' title='Download a SKILL.md (Σ_pol/Σ_flow authoring guide + this live catalog) to load into any assistant'>↓ SKILL.md</button></div><div id='market'></div></div></section>
    <section class='grid hidden page' id='activityPage'><div class='card span12'><div class='toolbar'><div class='toolbarLeft'><div class='label'>Activity</div><div class='seg' id='activitySeg'><button data-kind='' class='active'>All</button><button data-kind='request'>Requests</button><button data-kind='reject'>Rejects</button><button data-kind='probe'>Probes</button></div></div></div><div id='recent'></div></div></section>
    <section class='grid hidden page' id='configPage'><div id='config'></div></section>
  </main>
</div>

<div id='drawerShade' class='drawerShade'><aside class='drawer' role='dialog' aria-modal='true' aria-labelledby='drawerTitle'><div class='drawerHead'><div><div class='label' id='drawerKicker'>Consumer</div><h2 id='drawerTitle'>Consumer controls</h2><div class='sub' id='drawerSub'>Select a row action to manage settings or keys.</div></div><button class='btn iconBtn ghost' id='closeDrawer' title='Close'>×</button></div>
  <div class='drawerSection'><div class='label'>Settings</div><div class='formGrid' style='margin-top:10px'><label>Consumer<input id='settingsConsumer' placeholder='consumer-name' /></label><label>Status<select id='settingsStatus'><option value='active'>active</option><option value='inactive'>inactive</option></select></label><label>Allowed routes<input id='settingsAllowedRoutes' placeholder='profile:default,pin:openrouter/* or all' /></label><label>Rate/min override<input id='settingsRate' placeholder='default' /></label><label>Burst override<input id='settingsBurst' placeholder='default' /></label><button class='btn primary' id='saveConsumerSettings'>Save settings</button></div></div>
  <div class='drawerSection'><div class='label'>Reveal recoverable keys</div><p class='muted small'>Only raw keys already stored in recoverable legacy storage can be revealed. Hash-only keys require replacement generation.</p><div class='formGrid'><label>Consumer<input id='revealConsumer' placeholder='consumer-name' /></label><button class='btn' id='revealKeys'>Reveal keys</button></div><div id='revealKeyBox' class='resultBox' style='display:none'><textarea id='revealKeyValue' readonly></textarea><button class='btn' id='copyRevealKey'>Copy</button></div><div id='revealMeta' class='muted small'></div></div>
  <div class='drawerSection'><div class='label'>Create / rotate key</div><p class='muted small'>Raw key is shown once. Rotation marks existing active keys to expire after the grace period.</p><div class='formGrid'><label>Consumer<input id='newKeyConsumer' placeholder='consumer-name' /></label><label class='checkRow'><input id='rotateExisting' type='checkbox' /> rotate existing keys</label><label>Grace seconds<input id='rotationGrace' placeholder='default 86400' /></label><button class='btn primary' id='createKey'>Generate key</button></div><div id='newKeyBox' class='resultBox' style='display:none'><div class='label'>Key only</div><textarea id='newKeyValue' readonly></textarea><button class='btn' id='copyKey'>Copy key only</button><div class='label' style='margin-top:12px'>User setup blurb</div><textarea id='newKeyHandoffValue' readonly style='min-height:360px'></textarea><button class='btn primary' id='copyKeyHandoff'>Copy full setup blurb</button><div class='muted small'>Paste the blurb to the user. It includes base URL, model, curl/Python examples, and usage endpoint instructions.</div></div></div>
  <div class='drawerSection'><div class='label'>Revoke key</div><div class='formGrid' style='margin-top:10px'><label>Consumer<input id='revokeConsumer' placeholder='consumer-name' /></label><label>SHA-256 prefix<input id='revokePrefix' placeholder='sha256 prefix' /></label><button class='btn danger' id='revokeKey'>Revoke key</button></div></div>
</aside></div><div id='toast' class='toast'></div>

<script>
const $=(id)=>document.getElementById(id);const fmt=(n)=>Number(n||0).toLocaleString();const ts=(s)=>s?new Date(s*1000).toLocaleString():'—';const esc=(s)=>String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
function jsarg(s){return JSON.stringify(String(s??'')).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m])).replace(/\"/g,'&quot;')}function toast(msg){$('toast').textContent=msg;$('toast').style.display='block';setTimeout(()=>$('toast').style.display='none',2200)}function showErr(msg){$('err').style.display='block';$('err').textContent=msg}function clearErr(){$('err').style.display='none';$('err').textContent=''}function pct(v){return v==null?'—':Math.round(Number(v||0)*100)+'%'}
function table(rows,cols){if(!rows||!rows.length)return '<div class="empty">No data yet.</div>';return '<table class="dataTable"><thead><tr>'+cols.map(c=>`<th class="${c.cls||''}">${c.label}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>`<td class="${c.cls||''}">${c.f(r)}</td>`).join('')+'</tr>').join('')+'</tbody></table>'}
function counterRows(obj){return Object.entries(obj||{}).map(([name,v])=>({name,...v})).sort((a,b)=>(b.requests||0)-(a.requests||0))}function cols(){return [{label:'Name',f:r=>`<span class="pill">${esc(r.name)}</span>`},{label:'Requests',cls:'right',f:r=>fmt(r.requests)},{label:'Errors',cls:'right',f:r=>fmt(r.errors)},{label:'Tokens',cls:'right',f:r=>fmt(r.tokens_total)},{label:'Avg ms',cls:'right',f:r=>fmt(Math.round(r.latency_ms_avg||0))},{label:'Last seen',f:r=>ts(r.last_seen)}]}
function syncConsumers(list,selected){const sel=$('consumer');const cur=selected||sel.value;sel.innerHTML='<option value="">All consumers</option>'+(list||[]).map(x=>`<option value="${esc(x.name)}">${esc(x.name)}</option>`).join('');sel.value=cur}
function showLogin(){document.querySelectorAll('.page').forEach(el=>el.classList.add('hidden'));$('login').classList.remove('hidden');$('newKeyValue').value='';$('newKeyHandoffValue').value='';$('revealKeyValue').value='';['newKeyBox','revealKeyBox'].forEach(id=>$(id).style.display='none')}
function renderHealthSummary(s){s=s||{};const state=s.state||'unknown';const cls=state==='ok'?'ok':state==='unknown'?'warn':state==='degraded'?'degraded':'down';const errors=Object.entries(s.error_kinds||{}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<span class="pill"><span class="dot bad"></span>${esc(k)} × ${fmt(v)}</span>`).join('');return `<div class="healthBanner ${cls}"><div><div class="label">Live chat health</div><div class="healthState ${cls==='ok'?'ok':cls==='down'?'bad':'warn'}">${esc(state)}</div></div><div><div>${fmt(s.success_count)} successful / ${fmt(s.request_count)} recent chat requests</div><div class="healthMeta"><span class="pill">success ${pct(s.success_rate)}</span><span class="pill">errors ${fmt(s.error_count)}</span><span class="pill">route failures ${fmt(s.route_failures)}</span></div></div><div class="errorChips">${errors||'<span class="pill"><span class="dot ok"></span>No dominant error</span>'}</div></div>`}
function failureRows(s){return (s?.failing_recent||[]).map(r=>({time:ts(r.ts),...r}))}function renderFailures(s,target){$(target).innerHTML=table(failureRows(s),[{label:'Time',f:r=>esc(r.time)},{label:'Caller',f:r=>esc(r.caller||'—')},{label:'Route',f:r=>esc(r.route||'—')},{label:'Status',cls:'right',f:r=>esc(r.status||'—')},{label:'Error',f:r=>`<span class="pill"><span class="dot bad"></span>${esc(r.error_kind||'unknown')}</span>`},{label:'Message',f:r=>esc(r.error_message||'—')}])}
let activeTab='overview';let lastStats={keys:[],consumers:[]};let consumerFilterStatus='';let activityKind='';function showPage(tab){activeTab=tab;const pages={overview:'app',consumers:'consumersPage',providerKeys:'providerKeysPage',keyUsage:'keyUsagePage',market:'marketPage',builder:'builderPage',activity:'activityPage',config:'configPage'};Object.values(pages).forEach(id=>$(id).classList.add('hidden'));$(pages[tab]||'app').classList.remove('hidden');const nav={overview:'tabOverview',consumers:'tabConsumers',providerKeys:'tabProviderKeys',keyUsage:'tabKeyUsage',market:'tabMarket',builder:'tabBuilder',activity:'tabActivity',config:'tabConfig'};Object.values(nav).forEach(id=>$(id).classList.remove('active'));$(nav[tab]||'tabOverview').classList.add('active');const titles={overview:['Analytics','Spend, traffic and errors — filter by timeframe, consumer, provider and model.'],consumers:['Consumers','Key handoff and consumer management without noisy row buttons.'],providerKeys:['Provider keys','Available LLM provider credentials, status, and usage.'],keyUsage:['Key usage','Lookup persistent per-key usage, costs, windows, and paginated recent calls.'],market:['Catalog','Providers data — every model family with prices, benchmarks and live performance. Download a SKILL.md to author Σ_pol/Σ_flow against this catalog.'],builder:['Policy builder','Compose your own Σ_pol policy over the live catalog — preview the ranking, download the term, run it per call.'],activity:['Activity','Recent requests, rejects, probes, and failures.'],config:['Config','Per-provider runtime knobs — applied live, persisted on the PVC.']};$('pageTitle').textContent=(titles[tab]||titles.overview)[0];$('pageSub').textContent=(titles[tab]||titles.overview)[1];$('consumer').classList.toggle('hidden',tab==='policies'||tab==='market'||tab==='builder'||tab==='keyUsage')}function tabFromLocation(){const q=new URLSearchParams(location.search).get('tab');if(q)return q;if(location.hash)return location.hash.slice(1);if(location.pathname.includes('/provider-keys'))return 'providerKeys';return 'overview'}function setTab(tab,opts={}){showPage(tab);if(tab==='providerKeys')loadCodexAccounts();if(!opts.silent){const url=new URL(location.href);if(tab==='providerKeys'){url.pathname='/dashboard/provider-keys';url.searchParams.delete('tab');url.hash=''}else{url.pathname='/dashboard';url.searchParams.set('tab',tab);url.hash=''}history.replaceState(null,'',url)}if(tab==='policies')loadPolicies();else if(tab==='market')loadMarket();else if(tab==='keyUsage'){}else if(tab==='config')loadConfig();else{if(tab==='builder'){loadBuilderFamilies();loadBuilderFields()}load()}}
/* regression marker for JS-context escaping: onclick="pickConsumer(${jsarg(r.name)})" */
function money(v){return v?('$'+Number(v).toFixed(4)):'—'}
function consumerStats(k){return k.stats||{}}
function rowForConsumer(k){const status=k.status||'inactive';const initial=(k.consumer||'?')[0];const stats=consumerStats(k);const selected=lastStats.selected_consumer===k.consumer;const admin=lastStats.viewer_role!=='consumer';const actions=admin?`<button class="btn iconBtn ghost" title="Edit consumer settings" onclick="event.stopPropagation();openDrawer(${jsarg(k.consumer)},'settings')">✎</button><button class="btn iconBtn ghost" title="Reveal recoverable keys" onclick="event.stopPropagation();openDrawer(${jsarg(k.consumer)},'reveal')">🔑</button><button class="btn iconBtn ghost" title="Generate or rotate key" onclick="event.stopPropagation();openDrawer(${jsarg(k.consumer)},'create')">＋</button>`:'';return `<tr class="${selected?'selectedRow':''}" onclick="pickConsumer(${jsarg(k.consumer)})"><td><div class="nameCell"><div class="avatar">${esc(initial)}</div><div><div class="rowTitle">${esc(k.consumer)}</div><div class="rowMeta">${esc(k.allowed_routes||'all routes')}</div></div></div></td><td><span class="pill"><span class="dot ${status==='active'?'ok':'warn'}"></span>${esc(status)}</span></td><td>${fmt(stats.requests||0)}</td><td>${fmt(stats.errors||0)}</td><td>${fmt(stats.tokens_total||0)}</td><td class="right">${money(stats.cost_usd)}</td><td>${fmt(Math.round(stats.latency_ms_avg||0))}</td><td>${ts(stats.last_seen)}</td><td>${esc(k.key_status||'—')}</td><td class="right"><div class="actions">${actions}</div></td></tr>`}
function renderConsumers(keys){const q=($('consumerSearch')?.value||'').toLowerCase();const rows=(keys||[]).filter(k=>(!q||String(k.consumer).toLowerCase().includes(q))&&(!consumerFilterStatus||k.status===consumerFilterStatus)).sort((a,b)=>((consumerStats(b).last_seen||0)-(consumerStats(a).last_seen||0))||((consumerStats(b).requests||0)-(consumerStats(a).requests||0))||String(a.consumer).localeCompare(String(b.consumer)));$('keys').innerHTML=rows.length?`<table class="dataTable"><thead><tr><th>Consumer</th><th>Status</th><th class="right">Requests</th><th class="right">Errors</th><th class="right">Tokens</th><th class="right">Est. spend</th><th class="right">Avg ms</th><th>Last seen</th><th>Key status</th><th class="right">Actions</th></tr></thead><tbody>${rows.map(rowForConsumer).join('')}</tbody></table>`:'<div class="empty">No matching consumers.</div>'}
function pickConsumer(c){scopeConsumer(c)}function scopeConsumer(c){$('consumer').value=c;activeTab='consumers';showPage('consumers');load()}
function renderConsumerDetail(d){const c=d.selected_consumer;if(!c){$('consumerDetail').innerHTML='<div class="cardPad empty">Click a consumer row to see usage, route breakdowns, and recent activity in this same view.</div>';return}const row=(d.keys||[]).find(k=>k.consumer===c)||{};const stats=row.stats||{};const recent=(d.recent||[]).slice(0,40);$('consumerDetail').innerHTML=`<div class="toolbar"><div><div class="label">${esc(c)} usage</div><div class="sub">${esc((d.timeframe||{}).selected||'all')} · ${fmt(stats.requests||0)} requests · last seen ${ts(stats.last_seen)}</div></div><div class="toolbarRight">${d.viewer_role==='consumer'?'':`<button class="btn" onclick="openDrawer(${jsarg(c)},'settings')">Settings</button><button class="btn" onclick="openDrawer(${jsarg(c)},'reveal')">Reveal key</button><button class="btn primary" onclick="openDrawer(${jsarg(c)},'create')">Generate key</button>`}</div></div><div class="cardsOnly cardPad"><div class="card cardPad span3"><div class="label">Requests</div><div class="metric">${fmt(stats.requests||0)}</div><div class="statSub">errors ${fmt(stats.errors||0)}</div></div><div class="card cardPad span3"><div class="label">Tokens</div><div class="metric">${fmt(stats.tokens_total||0)}</div><div class="statSub">in ${fmt(stats.tokens_in||0)} · out ${fmt(stats.tokens_out||0)}</div></div><div class="card cardPad span3"><div class="label">Est. spend</div><div class="metric">${money(stats.cost_usd)}</div><div class="statSub">priced from live price book</div></div><div class="card cardPad span3"><div class="label">Avg latency</div><div class="metric">${fmt(Math.round(stats.latency_ms_avg||0))}</div><div class="statSub">max ${fmt(Math.round(stats.latency_ms_max||0))} ms</div></div><div class="card cardPad span3"><div class="label">Status</div><div class="metric ${row.status==='active'?'ok':'warn'}">${esc(row.status||'—')}</div><div class="statSub">routes ${esc(row.allowed_routes||'all')}</div></div><div class="card span6"><div class="toolbar"><div class="label">Routes</div></div>${table(counterRows(d.by_route),cols())}</div><div class="card span6"><div class="toolbar"><div class="label">Providers</div></div>${table(counterRows(d.by_provider),cols())}</div><div class="card span12"><div class="toolbar"><div class="label">Recent activity</div></div>${table(recent,[{label:'Time',f:r=>ts(r.ts)},{label:'Status',cls:'right',f:r=>esc(r.status||'—')},{label:'Route',f:r=>esc(r.requested_model||r.route||'—')},{label:'Provider',f:r=>esc(r.provider||'—')},{label:'Model',f:r=>esc(r.served_model_id||'—')},{label:'Tokens',cls:'right',f:r=>fmt(r.tokens_total||0)},{label:'Error',f:r=>esc(r.error_code||r.error_type||'—')}])}</div></div>`}
function fillConsumerFields(c){['settingsConsumer','revealConsumer','newKeyConsumer','revokeConsumer'].forEach(id=>$(id).value=c||'');const row=(lastStats.keys||[]).find(k=>k.consumer===c)||{};$('settingsStatus').value=row.status||'active';$('settingsAllowedRoutes').value=row.allowed_routes==='all'?'':(row.allowed_routes||'');$('settingsRate').value=row.rate_per_min||'';$('settingsBurst').value=row.burst||'';$('drawerTitle').textContent=c||'Consumer controls';$('drawerSub').textContent=row.consumer?`${row.status||'inactive'} · ${row.key_storage||'key storage unknown'}`:'Manage settings and keys'}function openDrawer(c,mode){fillConsumerFields(c);$('drawerShade').classList.add('open');if(mode==='reveal')setTimeout(()=>$('revealKeys').focus(),0);if(mode==='create')setTimeout(()=>$('createKey').focus(),0);if(mode==='settings')setTimeout(()=>$('settingsStatus').focus(),0)}function closeDrawer(){$('drawerShade').classList.remove('open')}
async function load(){try{clearErr();const c=$('consumer').value;const tf=$('timeframe').value||'all';const qs=new URLSearchParams();if(c)qs.set('consumer',c);qs.set('timeframe',tf);const pv=$('anProvider')?$('anProvider').value:'';const md=$('anModel')?$('anModel').value:'';if(pv)qs.set('provider',pv);if(md)qs.set('model',md);const r=await fetch('/dashboard/api/stats?'+qs.toString(),{credentials:'same-origin'});if(r.status===401){showLogin();return}if(!r.ok)throw new Error(`stats ${r.status}`);const d=await r.json();lastStats=d;render(d)}catch(e){showErr(e.message)}}
function applyViewerMode(d){const consumerMode=d.viewer_role==='consumer';$('consumer').classList.toggle('hidden',consumerMode);$('newConsumerKey').classList.toggle('hidden',consumerMode);$('tabMarket').classList.toggle('hidden',consumerMode);$('tabProviderKeys').classList.toggle('hidden',consumerMode);$('tabKeyUsage').classList.add('hidden');if(consumerMode&&(activeTab==='policies'||activeTab==='market'||activeTab==='providerKeys'))activeTab='consumers';if(consumerMode)$('pageSub').textContent='Your API-key-scoped usage and activity.'}
function renderProviderKeys(d){const pk=d.provider_keys||{};const rows=(pk.rows||[]).map(r=>({...r,last_seen_text:ts(r.last_seen),requests:r.usage?.requests||0,errors:r.usage?.errors||0,tokens:r.usage?.tokens_total||0,latency:r.usage?.latency_ms_avg||0,cost:r.estimated_cost_usd||0}));$('providerKeys').innerHTML=table(rows,[{label:'Provider',f:r=>`<div class="rowTitle">${esc(r.provider)}</div><div class="rowMeta">${esc(r.api_kind||'—')} · ${esc(r.tier||'—')}</div>`},{label:'Credential / wallet',f:r=>r.wallet?walletCell(r.wallet):`<span class="pill ${r.credential_status==='missing'?'bad':'ok'}">${esc(r.credential_status)}</span><div class="rowMeta">${esc(r.auth_env||r.auth_kind||'none')}${r.key_fingerprint?' · '+esc(r.key_fingerprint):''}</div>${(r.key_present||r.credential_status==='oauth_configured'||r.auth_env)?`<div class="actions" style="margin-top:4px">${(r.key_present||r.credential_status==='oauth_configured')?`<button class="btn iconBtn ghost" title="Reveal key" onclick="revealProviderKey(${jsarg(r.provider)},this)">👁</button><button class="btn iconBtn ghost" title="Copy key to clipboard" onclick="copyProviderKey(${jsarg(r.provider)})">⧉</button>`:''}${r.auth_env?`<button class="btn iconBtn ghost" title="Set/replace key" onclick="editProviderKey(${jsarg(r.provider)})">✎</button>`:''}</div>`:''}`},{label:'Requests',cls:'right',f:r=>fmt(r.requests)},{label:'Errors',cls:'right',f:r=>fmt(r.errors)},{label:'Tokens',cls:'right',f:r=>fmt(r.tokens)},{label:'Est. cost',cls:'right',f:r=>r.cost?('$'+Number(r.cost).toFixed(4)):'—'},{label:'Last seen',f:r=>esc(r.last_seen_text)},{label:'Last route/model',f:r=>`${esc(r.last_route||'—')}<div class="rowMeta">${esc(r.last_model_family||'')}</div>`}])}
function actStep(e,n){const skip=e.event==='skipped';const ok=!skip&&!e.error_kind;const err=`${esc(e.error_kind||'')}${e.http_status?` (${esc(e.http_status)})`:''}${e.error_message?': '+esc(String(e.error_message).slice(0,200)):''}`;const dot=skip?'<span class="dot"></span>':`<span class="dot ${ok?'ok':'bad'}"></span>`;const tag=skip?'<span class="muted small">skipped</span>':ok?'<span class="ok small">ok ✓</span>':`<span class="bad small">${err}</span>`;return `<div class="actStep"><span class="stepN">${n}</span>${dot}<code>${esc(e.provider_id||e.provider||'—')}</code>${e.model_family?`<span class="muted small">${esc(e.model_family)}</span>`:''}${tag}</div>`}
function actFlowNode(n,i){const itr=n.decision_trace||{};const raw=(Array.isArray(itr.decision_path)?itr.decision_path:(Array.isArray(itr.attempts)?itr.attempts:[])).filter(e=>e&&(e.provider_id||e.provider));const steps=raw.map((e,k)=>actStep(e,k+1)).join('');const fp=n.policy_fingerprint?`<span class="muted small">policy ${esc(n.policy_fingerprint)}</span>`:'';const toks=(n.tokens_in!=null||n.tokens_out!=null)?`<span class="muted small">${fmt(n.tokens_in)}→${fmt(n.tokens_out)} tok</span>`:'';const lat=n.latency_ms!=null?`<span class="${n.latency_ms>5000?'bad':'muted'} small">${fmt(Math.round(n.latency_ms))} ms</span>`:'';return `<div class="actStep"><span class="stepN">${esc(n.node||i+1)}</span><span class="dot ok"></span><code>${esc(n.provider||'—')}</code>${n.served_model_id?`<span class="muted small">${esc(n.served_model_id)}</span>`:''}${fp}${lat}${toks}</div>${steps?`<div style="margin-left:20px;border-left:1px solid var(--line);padding-left:8px">${steps}</div>`:''}`}
function actFlowCard(n){const itr=n.decision_trace||{};const raw=(Array.isArray(itr.decision_path)?itr.decision_path:(Array.isArray(itr.attempts)?itr.attempts:[])).filter(e=>e&&(e.provider_id||e.provider));const steps=raw.map((e,k)=>actStep(e,k+1)).join('');const lat=n.latency_ms!=null?`<div class="${n.latency_ms>5000?'bad':'muted'} small">${fmt(Math.round(n.latency_ms))} ms</div>`:'';const toks=(n.tokens_in!=null||n.tokens_out!=null)?`<div class="muted small">${fmt(n.tokens_in)}→${fmt(n.tokens_out)} tok</div>`:'';return `<div style="border:1px solid var(--line);border-radius:8px;padding:6px 9px;min-width:150px;background:rgba(255,255,255,.02)"><div style="display:flex;align-items:center;gap:6px"><span class="dot ok"></span><b class="small">${esc(n.node||'?')}</b><code class="small">${esc(n.provider||'—')}</code></div>${n.served_model_id?`<div class="muted small">${esc(n.served_model_id)}</div>`:''}${lat}${toks}${steps?`<div style="margin-top:4px;border-top:1px solid var(--line);padding-top:4px">${steps}</div>`:''}</div>`}
function actFlowDag(nodes){const byId={};nodes.forEach(n=>{byId[n.node]=n});const lvl={};const level=n=>{if(lvl[n.node]!=null)return lvl[n.node];lvl[n.node]=1;let m=1;(n.inputs||[]).forEach(p=>{if(byId[p])m=Math.max(m,level(byId[p])+1)});return lvl[n.node]=m};nodes.forEach(level);const maxL=Math.max(1,...nodes.map(n=>lvl[n.node]));let rows='';for(let L=1;L<=maxL;L++){const at=nodes.filter(n=>lvl[n.node]===L);if(!at.length)continue;rows+=`<div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center">${at.map(actFlowCard).join('')}</div>`;if(L<maxL)rows+=`<div style="text-align:center;color:var(--muted);margin:1px 0">↓</div>`}return `<div style="display:flex;flex-direction:column;gap:2px">${rows}</div>`}
function actDetail(r){const tr=r.decision_trace||{};const rawPath=Array.isArray(tr.decision_path)?tr.decision_path:(Array.isArray(tr.attempts)?tr.attempts:[]);const path=rawPath.filter(e=>e&&(e.provider_id||e.provider));const steps=path.length?path.map((e,i)=>actStep(e,i+1)).join(''):'<div class="muted small">No fallback trace recorded for this event.</div>';const fnodes=Array.isArray(tr.flow_nodes)?tr.flow_nodes:null;const fp=tr.flow_fingerprint?('flow '+tr.flow_fingerprint):(tr.policy_fingerprint||r.policy_fingerprint||r.requested_model);const cost=r.cost_usd==null?'—':'$'+Number(r.cost_usd).toFixed(6);const termStr=tr.policy_term?JSON.stringify(tr.policy_term):null;const policyBlock=termStr?`<div class="label small" style="margin-top:10px;display:flex;align-items:center;gap:8px">Policy term<button class="btn" data-copyterm="${esc(termStr)}" style="height:22px;padding:0 8px;font-size:11px">Copy</button></div><pre class="mono small" style="white-space:pre-wrap;max-height:160px;overflow:auto;background:rgba(255,255,255,.02);border:1px solid var(--line);border-radius:8px;padding:8px;margin-top:4px">${esc(termStr)}</pre>`:`<div class="muted small" style="margin-top:10px">No Σ_pol term — legacy closure profile (no copyable policy object).</div>`;return `<div class="actDetailBox"><div class="actMeta"><span class="pill">policy ${fp?esc(fp):'—'}</span><span class="pill">cost ${cost}</span><span class="pill">tokens ${fmt(r.tokens_total)} · in ${fmt(r.tokens_in)} · out ${fmt(r.tokens_out)}</span><span class="pill">${fmt(Math.round(r.latency_ms||0))} ms</span>${r.served_model_id?`<span class="pill">served <code>${esc(r.served_model_id)}</code></span>`:''}</div>${fnodes?('<div class="label small" style="margin-bottom:4px">Σ_flow — node DAG ('+fnodes.length+' node'+(fnodes.length===1?'':'s')+')</div>'+(fnodes.some(n=>Array.isArray(n.inputs))?actFlowDag(fnodes):fnodes.map(actFlowNode).join(''))):('<div class="label small" style="margin-bottom:4px">Attempts — fallback order</div>'+steps)}${policyBlock}</div>`}
function renderActivity(rows){if(!rows.length){$('recent').innerHTML='<div class="empty">No activity yet.</div>';return}const head=`<tr><th></th><th>Time</th><th>Event</th><th>Caller</th><th class="right">Status</th><th>Route</th><th>Provider</th><th class="right">Cost</th><th>Error</th></tr>`;const body=rows.map((r,i)=>{const errK=r.error_kind||r.error_code||r.error_type||'';const st=Number(r.status||0);const sCls=(st>=200&&st<300)?'ok':(st>=400||errK)?'bad':'warn';const summary=`<tr class="actRow" data-i="${i}"><td class="actToggle" data-t="${i}">▸</td><td>${ts(r.ts)}</td><td><span class="pill">${esc(r.event)}</span></td><td>${esc(r.caller||'—')}</td><td class="right"><span class="${sCls}">${esc(r.status||'—')}</span></td><td>${esc(r.requested_model||r.route||'—')}</td><td>${esc(r.provider||'—')}</td><td class="right">${r.cost_usd==null?'—':'$'+Number(r.cost_usd).toFixed(6)}</td><td>${errK?`<span class="bad">${esc(errK)}</span>`:'—'}</td></tr>`;const detail=`<tr class="actDetail hidden" data-d="${i}"><td></td><td colspan="8">${actDetail(r)}</td></tr>`;return summary+detail}).join('');$('recent').innerHTML=`<table class="dataTable">${head}${body}</table>`}
function anCols(){return [{label:'Name',f:r=>`<span class="pill">${esc(r.name)}</span>`},{label:'Requests',cls:'right',f:r=>fmt(r.requests)},{label:'Errors',cls:'right',f:r=>fmt(r.errors)},{label:'Tokens',cls:'right',f:r=>fmt(r.tokens_total)},{label:'Spend',cls:'right',f:r=>r.cost_usd==null?'—':'$'+Number(r.cost_usd).toFixed(4)}]}
function fillFilter(id,opts,allLabel){const sel=$(id);if(!sel)return;const cur=sel.value;sel.innerHTML=`<option value="">${esc(allLabel)}</option>`+(opts||[]).map(o=>`<option value="${esc(o)}"${o===cur?' selected':''}>${esc(o)}</option>`).join('');if([...sel.options].some(o=>o.value===cur))sel.value=cur}
function renderSeries(days){if(!days||!days.length){$('anSeries').innerHTML='<div class="muted small">No data in this window.</div>';return}const rows=days.slice(0,30);const max=Math.max(1,...rows.map(d=>Number(d.requests||0)));$('anSeries').innerHTML=rows.map(d=>{const r=Number(d.requests||0),w=Math.round(r/max*100);return `<div style="display:flex;align-items:center;gap:10px;margin:3px 0;font-size:12px"><span class="muted" style="width:88px;flex:none">${esc(d.date||d.month||'—')}</span><div style="flex:1;background:rgba(255,255,255,.04);border-radius:4px;height:14px;overflow:hidden"><div style="width:${w}%;height:100%;background:linear-gradient(90deg,#7170ff,#5e6ad2)"></div></div><span style="width:54px;text-align:right;flex:none">${fmt(r)}</span><span class="muted" style="width:78px;text-align:right;flex:none">$${Number(d.cost_usd||0).toFixed(4)}</span></div>`}).join('')}
function renderAnalytics(d){const t=d.totals||{};const req=Number(t.requests||0),err=Number(t.errors||0);$('anRequests').textContent=fmt(req);$('anReqSub').textContent=`${fmt(err)} errors · ${fmt(t.rejects||0)} rejects`;$('anSpend').textContent='$'+Number(t.cost_usd||0).toFixed(4);$('anSpendSub').textContent=`over ${fmt(req)} requests`;$('anTokens').textContent=fmt(t.tokens_total);$('anTokSub').textContent=`in ${fmt(t.tokens_in)} · out ${fmt(t.tokens_out)}`;const sr=req?(req-err)/req:null;$('anSuccess').textContent=sr==null?'—':Math.round(sr*100)+'%';$('anSuccess').className='metric '+(sr==null?'':sr>=0.95?'ok':sr>=0.8?'warn':'bad');$('anSuccessSub').textContent=`${fmt(req-err)} ok / ${fmt(req)}`;$('anByProvider').innerHTML=table(counterRows(d.by_provider),anCols());$('anByModel').innerHTML=table(counterRows(d.by_model_family),anCols());$('anByConsumer').innerHTML=table(counterRows(d.by_caller),anCols());$('anByStatus').innerHTML=table(Object.entries(d.by_status||{}).map(([name,count])=>({name,requests:count})),[{label:'Status',f:r=>`<span class="pill">${esc(r.name)}</span>`},{label:'Count',cls:'right',f:r=>fmt(r.requests)}]);renderSeries(d.daily_totals||[]);const fo=d.filter_options||{};fillFilter('anProvider',fo.providers,'All providers');fillFilter('anModel',fo.models,'All models')}
function render(d){$('login').classList.add('hidden');applyViewerMode(d);document.querySelectorAll('.page').forEach(el=>el.classList.add('hidden'));$(({overview:'app',consumers:'consumersPage',providerKeys:'providerKeysPage',keyUsage:'keyUsagePage',market:'marketPage',builder:'builderPage',activity:'activityPage',config:'configPage'})[activeTab]||'app').classList.remove('hidden');syncConsumers(d.consumers||[],d.selected_consumer||'');renderAnalytics(d);renderConsumers(d.keys||[]);renderConsumerDetail(d);renderProviderKeys(d);let recent=(d.recent||[]);if(activityKind)recent=recent.filter(r=>r.event===activityKind);renderActivity(recent.slice(0,60))}
async function login(){try{clearErr();const r=await fetch('/dashboard/login',{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify({password:$('password').value})});if(!r.ok)throw new Error('login failed');$('password').value='';toast('Logged in');load()}catch(e){showErr(e.message)}}async function apiKeyLogin(){try{clearErr();const r=await fetch('/dashboard/login',{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify({api_key:$('apiKeyLogin').value})});if(!r.ok)throw new Error('API key login failed');$('apiKeyLogin').value='';activeTab='consumers';toast('Logged in');load()}catch(e){showErr(e.message)}}async function logout(){await fetch('/dashboard/logout',{method:'POST',credentials:'same-origin'});showLogin()}async function revealKeys(){try{const c=$('revealConsumer').value.trim();const r=await fetch('/dashboard/api/keys/reveal?consumer='+encodeURIComponent(c),{credentials:'same-origin'});if(r.status===401){showLogin();return}if(!r.ok)throw new Error(`reveal ${r.status}`);const d=await r.json();$('revealKeyBox').style.display='block';$('revealKeyValue').value=(d.keys||[]).map(k=>k.api_key).join('\\n');$('revealMeta').textContent=d.message||`${(d.keys||[]).length} recoverable · ${d.hash_only_count||0} hash-only`;toast('Reveal loaded')}catch(e){showErr(e.message)}}
function buildKeyHandoff(apiKey,consumer){
  const key=String(apiKey||'').trim();
  const name=String(consumer||'').trim();
  const auth='Authorization: '+'Bearer '+key;
  const bs=String.fromCharCode(92);
  return [
    `Your unhardcoded key is ready${name?' for '+name:''}.`,
    '',
    'API key:',
    key,
    '',
    'Use it as an OpenAI-compatible endpoint.',
    '',
    'Base URL:',
    '__PUBLIC_BASE_URL__',
    '',
    'Recommended model:',
    'profile:default',
    '',
    'Other options:',
    'family:<model-family>  ·  pin:<provider>/<family>',
    'or send your own policy per call as a Σ_pol `policy_ir` (build one in the dashboard).',
    '',
    'If your app has a “Custom OpenAI / OpenAI-compatible provider” interface, enter:',
    '',
    'Provider type: OpenAI-compatible',
    'Base URL: __PUBLIC_BASE_URL__',
    'API key: '+key,
    'Model: profile:default',
    '',
    'Quick test with curl:',
    '',
    'curl __PUBLIC_BASE_URL__/chat/completions '+bs,
    '  -H "'+auth+'" '+bs,
    '  -H "Content-Type: application/json" '+bs,
    "  -d '{",
    '    "model": "profile:default",',
    '    "messages": [',
    '      {"role": "user", "content": "Reply exactly: pong"}',
    '    ],',
    '    "max_tokens": 20',
    "  }'",
    '',
    'Python example using the OpenAI SDK:',
    '',
    'from openai import OpenAI',
    '',
    'client = OpenAI(',
    '    api_key="'+key+'",',
    '    base_url="__PUBLIC_BASE_URL__",',
    ')',
    '',
    'response = client.chat.completions.create(',
    '    model="profile:default",',
    '    messages=[',
    '        {"role": "user", "content": "Reply exactly: pong"}',
    '    ],',
    '    max_tokens=20,',
    ')',
    '',
    'print(response.choices[0].message.content)',
    '',
    'Check your usage:',
    '',
    'curl "__PUBLIC_BASE_URL__/usage?window=24h&limit=50" '+bs,
    '  -H "'+auth+'"',
    '',
    'If you get HTTP 403 caller_route_not_allowed, the key is active but route-restricted. Ask us to enable route access for the key, or regenerate it with Allowed routes = all.'
  ].join('\\n')
}
async function createKey(){try{const body={consumer:$('newKeyConsumer').value.trim(),rotate:$('rotateExisting').checked};const g=$('rotationGrace').value.trim();if(g)body.grace_period_s=Number(g);const r=await fetch('/dashboard/api/keys',{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify(body)});if(r.status===401){showLogin();return}if(!r.ok)throw new Error(`create ${r.status}`);const d=await r.json();$('newKeyBox').style.display='block';$('newKeyValue').value=d.api_key||'';$('newKeyHandoffValue').value=buildKeyHandoff(d.api_key||'',d.consumer||body.consumer);toast('Key generated — copy key or setup blurb now');load()}catch(e){showErr(e.message)}}async function saveConsumerSettings(){try{const c=$('settingsConsumer').value.trim();const routes=$('settingsAllowedRoutes').value.trim();const body={status:$('settingsStatus').value};if(routes)body.allowed_routes=routes.split(',').map(s=>s.trim()).filter(Boolean);else body.allowed_routes=[];const rate=$('settingsRate').value.trim();const burst=$('settingsBurst').value.trim();if(rate)body.rate_per_min=Number(rate);if(burst)body.burst=Number(burst);const r=await fetch('/dashboard/api/consumers/'+encodeURIComponent(c),{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify(body)});if(r.status===401){showLogin();return}if(!r.ok)throw new Error(`settings ${r.status}`);toast('Settings saved');load()}catch(e){showErr(e.message)}}async function revokeKey(){try{const r=await fetch('/dashboard/api/keys/revoke',{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify({consumer:$('revokeConsumer').value.trim(),sha256_prefix:$('revokePrefix').value.trim()})});if(r.status===401){showLogin();return}if(!r.ok)throw new Error(`revoke ${r.status}`);toast('Key revoked');load()}catch(e){showErr(e.message)}}
async function loadKeyUsage(){try{clearErr();const body={api_key:$('keyUsageApiKey').value.trim()};['Since','Window'].forEach(k=>{const v=$('keyUsage'+k).value.trim();if(v)body[k.toLowerCase()]=v});body.limit=Number($('recentLimit').value||50);body.offset=Number($('recentOffset').value||0);const r=await fetch('/dashboard/api/key-usage',{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify(body)});if(r.status===401){showLogin();return}if(!r.ok)throw new Error(`key usage ${r.status}`);const d=await r.json();const t=d.totals||{},c=d.cost_estimate||{},w=d.window||{};$('keyUsageResult').innerHTML=`<div class="grid"><div class="card cardPad span3"><div class="label">Requests</div><div class="metric">${fmt(t.requests)}</div><div class="statSub">errors ${fmt(t.errors)}</div></div><div class="card cardPad span3"><div class="label">Tokens</div><div class="metric">${fmt(t.tokens_total)}</div><div class="statSub">in ${fmt(t.tokens_in)} · out ${fmt(t.tokens_out)}</div></div><div class="card cardPad span3"><div class="label">Cost estimate</div><div class="metric">$${Number(c.usd||0).toFixed(6)}</div><div class="statSub">${fmt(c.priced_events)} priced · ${fmt(c.unpriced_events)} unpriced</div></div><div class="card cardPad span3"><div class="label">Recent page</div><div class="metric">${fmt(w.recent_returned)}</div><div class="statSub">of ${fmt(w.recent_total)} · offset ${fmt(w.offset)}</div></div><div class="card span12"><div class="toolbar"><div class="label">Recent</div></div>${table(d.recent||[],[{label:'Time',f:r=>ts(r.ts)},{label:'Status',cls:'right',f:r=>esc(r.status||'—')},{label:'Route',f:r=>esc(r.requested_model||'—')},{label:'Provider',f:r=>esc(r.provider||'—')},{label:'Model',f:r=>esc(r.model_family||'—')},{label:'Tokens',cls:'right',f:r=>fmt(r.tokens_total)},{label:'Cost',cls:'right',f:r=>r.cost_usd==null?'—':'$'+Number(r.cost_usd).toFixed(6)}])}</div><div class="card span12"><div class="toolbar"><div class="label">JSON</div></div><pre class="json">${esc(JSON.stringify({window:d.window,cost_estimate:d.cost_estimate,daily_totals:d.daily_totals,monthly_totals:d.monthly_totals},null,2))}</pre></div></div>`}catch(e){showErr(e.message)}}
function providerClass(name){name=String(name||'');if(name.includes('openrouter'))return 'openrouter';if(name.includes('codex'))return 'codex';if(name.includes('antseed'))return 'antseed';return ''}function qualityText(p){if(p.filter)return String(p.filter);let parts=[];if(p.quality_min!=null)parts.push('quality ≥ '+p.quality_min);if(p.quality_max!=null)parts.push('quality ≤ '+p.quality_max);return parts.join(' · ')||'router-ranked policy'}function modelCaps(m){const c=m.capabilities||{};return Object.entries(c).filter(([k,v])=>v).map(([k])=>k.replace(/^supports_/,'')).join(', ')||'chat'}function weightText(p){return Object.entries(p.weights||{}).map(([k,v])=>`${k} ${v}`).join(' · ')||'default'}let lastPolicies=null;function healthDot(h){const st=(h||{}).state||'idle';const cls=st==='ok'?'ok':st==='failing'?'bad':'warn';return `<span class="dot ${cls}"></span>`}function providerChip(s,hm){const h=(hm||{})[s.provider]||{};const failing=h.state==='failing';return `<div class="provider-chip ${providerClass(s.provider)}${failing?' failing':''}"><div class="stepBadge">${esc(s.order||'—')}</div><div class="modelId">priority ${esc(s.order||'—')}</div><div class="providerName">${healthDot(h)} ${esc(s.provider||'—')}</div><div class="modelId">${esc(s.provider_model_id||'—')}</div>${h.balance?`<div class="modelId chipBalance${h.runway==='empty'?' bad':h.runway==='low'?' warn':''}">${esc(h.balance.value??'?')} ${h.balance.kind==='deposits_usdc'?'USDC':h.balance.kind==='credits_usd'?'USD':'quota'}${h.runway?` · runway ${esc(h.runway)}`:''}</div>`:''}${failing&&h.reason?`<div class="chipReason">${esc(h.reason)}</div>`:''}</div>`}function policyDiagram(p,hm,connectedOnly){const models=(p.models||[]).map(m=>({...m,served_by:(m.served_by||[]).filter(s=>!connectedOnly||((hm||{})[s.provider]||{}).state!=='disconnected')})).filter(m=>(m.served_by||[]).length);const rows=models.map(m=>`<div class="model-row"><div class="family-card"><div class="name">${esc(m.name)}</div><div class="meta">quality ${esc(m.quality??'—')}<br>${esc(modelCaps(m))}</div></div><div class="chain-card"><div class="familyName">${esc(m.name)} fallback chain</div><div class="provider-timeline">${(m.served_by||[]).map(s=>providerChip(s,hm)).join('')}</div></div></div>`).join('');const empty=`<div class="empty-policy-note">${esc(p.selection_note||'No model families matched this profile filter in config.')}</div>`;return `<div class="policy-summary"><div class="summaryBox"><div class="k">Filter</div><div class="v">${esc(qualityText(p))}</div></div><div class="summaryBox"><div class="k">Weights</div><div class="v">${esc(weightText(p))}</div></div><div class="summaryBox"><div class="k">Retry policy</div><div class="v">${esc(p.retry_policy||'default')}</div></div></div><div class="flow-title">Eligible model families → provider fallback order</div>${rows||empty}`}function renderPolicies(d){lastPolicies=d;const hm={};(d.providers||[]).forEach(p=>{hm[p.name]=p.health||{}});const connectedOnly=$('connectedOnly').checked;$('policies').innerHTML=(d.profiles||[]).map(p=>`<article class="policy-card"><div class="policy-head"><div><h3><code>profile:${esc(p.name)}</code></h3><div class="muted small">${esc(qualityText(p))} · ${esc((p.models||[]).length)} model families</div></div><span class="pill">${esc(p.retry_policy||'default')}</span></div>${policyDiagram(p,hm,connectedOnly)}</article>`).join('')||'<div class="empty">No policy data.</div>'}async function loadPolicies(){try{clearErr();const r=await fetch('/dashboard/api/policies',{credentials:'same-origin'});if(r.status===401){showLogin();return}if(!r.ok)throw new Error(`policies ${r.status}`);renderPolicies(await r.json())}catch(e){showErr(e.message)}}
let lastMarket=null;const marketOpen=new Set();
function perfText(p){if(!p||!p.calls)return '<span class="muted small">no calls yet</span>';const sr=p.success_rate!=null?Math.round(p.success_rate*100)+'% ok':'—';const ms=p.latency_ms!=null?Math.round(p.latency_ms)+' ms':'—';return `${sr} · ${ms} · ${fmt(p.calls)} calls`}
function priceText(v){return v==null?'—':'$'+Number(v).toFixed(v>0&&v<0.1?3:2)}
function marketStatus(r){if(r.source!=='antseed')return r.tradable?'<span class="pill ok">live</span>':'<span class="pill bad">disabled</span>';if(r.tradable)return `<span class="pill ok">pinned ✓ via ${esc(r.via||'')}</span>`;if(r.pinned)return '<span class="pill warn">pinned · over cap</span>';return '<span class="pill">not pinned</span>'}
function metaBadges(m){if(!m)return '';const b=[];if(m.bench_intelligence!=null)b.push(`<span class="pill" title="OpenRouter intelligence index">IQ ${pct(m.bench_intelligence)}</span>`);if(m.bench_coding!=null)b.push(`<span class="pill" title="OpenRouter coding index">code ${pct(m.bench_coding)}</span>`);if(m.bench_agentic!=null)b.push(`<span class="pill" title="OpenRouter agentic index">agent ${pct(m.bench_agentic)}</span>`);if(m.bench_arena!=null)b.push(`<span class="pill" title="Design Arena win rate">arena ${pct(m.bench_arena)}</span>`);const mods=['image','audio','file','video'].filter(k=>m['in_'+k]);if(mods.length)b.push(`<span class="muted small" title="input modalities">${mods.join('/')}</span>`);return b.length?`<span style="display:inline-flex;gap:4px;align-items:center">${b.join('')}</span>`:''}
function copyAddr(a){navigator.clipboard.writeText(a).then(()=>toast('Address copied')).catch(e=>showErr(e.message))}
function walletCell(w){const rw=w.runway||'';const pill=rw==='empty'?'<span class="pill bad">empty · top up</span>':rw==='low'?'<span class="pill warn">low · top up</span>':rw==='ok'?'<span class="pill ok">funded</span>':'';const dep=w.deposits_available==null?'—':Number(w.deposits_available).toFixed(2);const res=w.deposits_reserved==null||Number(w.deposits_reserved)===0?'':' · '+Number(w.deposits_reserved).toFixed(2)+' in channels';const addr=w.address||'';return `<div style="display:flex;gap:6px;align-items:center"><b style="font-variant-numeric:tabular-nums">${dep} USDC</b><span class="muted small">${res}</span>${pill}</div>${addr?`<div class="rowMeta" style="display:flex;gap:6px;align-items:center;margin-top:3px"><code style="font-size:11px;word-break:break-all">${esc(addr)}</code><button class="btn iconBtn ghost" title="Copy wallet address" onclick="copyAddr(${jsarg(addr)})">⧉</button></div>`:'<div class="rowMeta">no address yet</div>'}${addr?`<div class="rowMeta" style="display:flex;gap:6px;margin-top:4px"><button class="btn ghost small" onclick="walletDeposit()">Deposit</button><button class="btn ghost small" onclick="walletWithdraw()">Withdraw</button><button class="btn iconBtn ghost" title="Refresh balance" onclick="walletRefresh()">↻</button></div>`:''}<div class="rowMeta">hot-wallet · top up on Base mainnet (USDC + ETH gas)${w.connection?' · '+esc(w.connection):''}</div>`}
function walletOp(op,promptMsg){return (async()=>{try{const a=prompt(promptMsg);if(a===null)return;const amount=String(a).trim();if(!/^\\d+(\\.\\d{1,6})?$/.test(amount)||Number(amount)<=0){showErr('amount must be a positive USDC value (≤6 decimals)');return}toast(op+' '+amount+' USDC…');const r=await fetch('/dashboard/api/wallet/'+op,{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify({amount})});if(r.status===401){showLogin();return}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||(op+' '+r.status));toast(op+' ok · '+amount+' USDC');loadMarket()}catch(e){showErr(e.message)}})()}
function walletDeposit(){return walletOp('deposit','Deposit how much USDC into the AntSeed deposits contract?\\n(moves wallet → escrow; needs a little ETH on Base for gas)')}
function walletWithdraw(){return walletOp('withdraw','Withdraw how much USDC from the deposits contract back to the wallet?')}
async function walletRefresh(){try{toast('Refreshing wallet…');const r=await fetch('/dashboard/api/wallet/refresh',{method:'POST',credentials:'same-origin'});if(r.status===401){showLogin();return}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||('refresh '+r.status));toast('Wallet refreshed');loadMarket()}catch(e){showErr(e.message)}}
function renderMarket(d){lastMarket=d;const q=($('marketSearch')?.value||'').toLowerCase();const tradableOnly=$('tradableOnly').checked;const fams=(d.families||[]).map(f=>({...f,rows:(f.rows||[]).filter(r=>!tradableOnly||r.tradable)})).filter(f=>f.rows.length&&(!q||f.family.toLowerCase().includes(q)));$('market').innerHTML=fams.map(f=>{const best=f.rows[0];const open=marketOpen.has(f.family);const head=`<div class="marketHead" data-fam="${esc(f.family)}" style="cursor:pointer;display:flex;gap:12px;align-items:center;padding:10px 4px;border-bottom:1px solid rgba(128,128,128,.25)"><span style="width:14px">${open?'▾':'▸'}</span><code style="min-width:200px">${esc(f.family)}</code><span class="muted small">q ${esc(f.quality??'—')}</span>${metaBadges(f.meta)}<span style="min-width:150px">from ${priceText(best?.price_in)} / ${priceText(best?.price_out)}</span><span class="muted small">${fmt(f.sellers_total)} seller${f.sellers_total===1?'':'s'}</span></div>`;const body=open?`<div style="padding:4px 0 14px 26px">${table(f.rows,[{label:'Seller',f:r=>`${esc(r.seller)}${r.source==='antseed'?`<div class="rowMeta">antseed${r.last_seen?' · seen '+ts(Math.round(r.last_seen/1000)):''}</div>`:''}`},{label:'Wire model',f:r=>esc(r.wire_model_id||'—')},{label:'$ in/Mtok',cls:'right',f:r=>priceText(r.price_in)},{label:'$ out/Mtok',cls:'right',f:r=>priceText(r.price_out)},{label:'Status',f:r=>marketStatus(r)},{label:'Performance',f:r=>perfText(r.perf)},{label:'Refreshed',f:r=>r.price_refreshed_at?ts(r.price_refreshed_at):'—'}])}</div>`:'';return head+body}).join('')||'<div class="empty">No market data yet.</div>'}
async function loadMarket(){try{clearErr();const r=await fetch('/dashboard/api/market',{credentials:'same-origin'});if(r.status===401){showLogin();return}if(!r.ok)throw new Error(`market ${r.status}`);renderMarket(await r.json())}catch(e){showErr(e.message)}}
let lastBuiltPolicy=null;
// ---- Policy builder: structured rows over RAW fields + raw-IR escape ----
// raw observable fields are data-driven from GET /x/fields (core + config
// extensions like bench_*/in_*/cap_*); seeded with the core set so the form
// renders before the fetch returns. Each entry: {name, group} (model|provider).
let bFields={num:[{name:'price_in',group:'provider'},{name:'price_out',group:'provider'},{name:'latency_ms',group:'provider'},{name:'tok_s',group:'provider'},{name:'success_rate',group:'provider'},{name:'credits',group:'provider'},{name:'context',group:'model'}],bool:[{name:'has_tee',group:'provider'},{name:'no_log',group:'provider'},{name:'breaker_open',group:'provider'},{name:'disabled',group:'provider'}]};
const bBoolNames=()=>bFields.bool.map(f=>f.name);
const bSortOf=f=>(({family_in:'cat',tier_in:'cat',min_tier:'cat',in_top_k:'cat'})[f])||(bBoolNames().includes(f)?'Bool':'Num');
const bGroupOpts=(arr,sel,provFirst)=>{const by={model:[],provider:[]};arr.forEach(f=>by[f.group==='provider'?'provider':'model'].push(f));const grp=(k,label)=>by[k].length?`<optgroup label='${label}'>`+by[k].map(f=>bOpt(f.v||f.name,f.l||f.name,sel)).join('')+`</optgroup>`:'';return provFirst?grp('provider','Provider')+grp('model','Model'):grp('model','Model')+grp('provider','Provider')};
const B_RELS=[['le','≤'],['lt','<'],['ge','≥'],['gt','>'],['eq','='],['ne','≠']];
let bFilters=[];let bScores=[{field:'field:price_in',w:'1',norm:true,inv:true}];let bRawMode=false;
const bSplit=s=>(s||'').split(',').map(x=>x.trim()).filter(Boolean);
const bOpt=(v,l,sel)=>`<option value='${esc(v)}'${v===sel?' selected':''}>${esc(l)}</option>`;
function bFail(msg){$('bError').style.display='block';$('bError').textContent=msg}
function bFieldSel(c){const cat=[{v:'in_top_k',name:'in top N by …'},{v:'family_in',name:'family ∈ set'},{v:'tier_in',name:'tier ∈ set'},{v:'min_tier',name:'tier ≥'}];return `<select class='bF-field'><optgroup label='Categorical'>`+cat.map(o=>bOpt(o.v,o.name,c.field)).join('')+`</optgroup>`+bGroupOpts(bFields.num.concat(bFields.bool),c.field,true)+`</select>`}
function bCtrls(c){const f=c.field;const sort=bSortOf(f);if(sort==='Bool')return `<select class='bF-want'>`+bOpt('1','is',c.want===false?'0':'1')+bOpt('0','is not',c.want===false?'0':'1')+`</select>`;if(f==='in_top_k'){const rk=bFields.num.filter(x=>bFields.num.some(y=>y.name===x.name+'_rank'));return `top&nbsp;<input class='bF-k' type='number' min='1' step='1' value='${esc(c.k??'5')}' style='width:60px'>&nbsp;by <select class='bF-by'>`+bGroupOpts(rk.map(x=>({v:x.name,name:x.name,group:x.group})),c.by||'',true)+`</select>`}if(f==='family_in')return `<input class='bF-vals' list='familyOptions' placeholder='gpt-5.5, claude-opus-4-8' value='${esc(c.vals||'')}'>`;if(f==='tier_in')return `<input class='bF-vals' placeholder='partner, marketplace' value='${esc(c.vals||'')}'>`;if(f==='min_tier')return `<input class='bF-vals' placeholder='partner' value='${esc(c.vals||'')}'>`;return `<select class='bF-rel'>`+B_RELS.map(([v,l])=>bOpt(v,l,c.rel||'le')).join('')+`</select><input class='bF-val' type='number' step='any' placeholder='value' value='${esc(c.val??'')}' style='width:110px'>`}
const bRowStyle="display:flex;gap:6px;align-items:center;margin-top:6px;flex-wrap:wrap";
function bCondHtml(c,i,j){const del=j==null?`data-act='del' data-i='${i}'`:`data-act='delsub' data-i='${i}' data-j='${j}'`;const dj=j==null?'':`data-j='${j}'`;return `<div class='bRow' data-i='${i}' ${dj} style='${bRowStyle}'>`+bFieldSel(c)+bCtrls(c)+`<button class='btn' ${del}>×</button></div>`}
function bRowHtml(c,i){if(c.kind==='or')return `<div class='bOr' data-i='${i}' style='border:1px dashed var(--line);border-radius:8px;padding:8px;margin-top:6px'><div class='muted small' style='margin-bottom:2px'>OR — any of:</div>`+(c.subs||[]).map((s,j)=>bCondHtml(s,i,j)).join('')+`<div class='toolbar' style='margin-top:6px'><button class='btn' data-act='addsub' data-i='${i}'>+ sub</button><button class='btn' data-act='del' data-i='${i}'>remove group</button></div></div>`;return bCondHtml(c,i,null)}
function bScoreHtml(s,i){const sel=`<select class='bS-field'>`+bGroupOpts(bFields.num.map(f=>({v:'field:'+f.name,name:f.name,group:f.group})),s.field)+`</select>`;return `<div class='bSrow' style='${bRowStyle}'><input class='bS-w' type='number' step='any' value='${esc(s.w??'')}' placeholder='weight' style='width:90px'> × ${sel}<label class='checkRow'><input type='checkbox' class='bS-norm'${s.norm?' checked':''}> normalize</label><label class='checkRow'><input type='checkbox' class='bS-inv'${s.inv?' checked':''}> invert</label><button class='btn' data-act='delscore' data-i='${i}'>×</button></div>`}
function bReadCond(el){const c={field:el.querySelector('.bF-field').value};const w=el.querySelector('.bF-want'),vs=el.querySelector('.bF-vals'),rl=el.querySelector('.bF-rel'),vl=el.querySelector('.bF-val'),kk=el.querySelector('.bF-k'),by=el.querySelector('.bF-by'),ti=el.querySelector('.bF-topinv');if(w)c.want=w.value==='1';if(vs)c.vals=vs.value;if(rl)c.rel=rl.value;if(vl)c.val=vl.value;if(kk)c.k=kk.value;if(by)c.by=by.value;if(ti)c.inv=ti.checked;return c}
function bSync(){bFilters=[...$('bFilters').querySelectorAll(':scope>.bRow,:scope>.bOr')].map(el=>el.classList.contains('bOr')?{kind:'or',subs:[...el.querySelectorAll('.bRow')].map(bReadCond)}:bReadCond(el));bScores=[...$('bScores').querySelectorAll(':scope>.bSrow')].map(el=>({field:el.querySelector('.bS-field').value,w:el.querySelector('.bS-w').value,norm:el.querySelector('.bS-norm').checked,inv:el.querySelector('.bS-inv').checked}))}
function bRender(){$('bFilters').innerHTML=bFilters.map(bRowHtml).join('')||`<div class='muted small'>No extra conditions — base filter only.</div>`;$('bScores').innerHTML=bScores.map(bScoreHtml).join('')||`<div class='muted small'>No score terms — add at least one.</div>`;$('bTempWrap').style.display=$('b_selector').value==='sample'?'':'none'}
function bSyncRender(){bSync();bRender()}
function bCondTerm(c){if(c.kind==='or'){const subs=(c.subs||[]).map(bCondTerm).filter(Boolean);return subs.length?(subs.length===1?subs[0]:['or',...subs]):null}const f=c.field;if(f==='family_in'){const v=bSplit(c.vals);return v.length?(v.length===1?['family_eq',v[0]]:['or',...v.map(x=>['family_eq',x])]):null}if(f==='tier_in'){const v=bSplit(c.vals);return v.length?(v.length===1?['tier_eq',v[0]]:['or',...v.map(x=>['tier_eq',x])]):null}if(f==='min_tier'){const v=bSplit(c.vals);return v.length?['min_tier',v[0]]:null}if(f==='in_top_k'){const k=parseInt(c.k,10);if(!(k>=1)||!c.by)return null;return ['cmp',c.by+'_rank','le',k]}if(bSortOf(f)==='Bool')return c.want===false?['not',['is',f]]:['is',f];if(c.val==null||c.val===''||isNaN(Number(c.val)))return null;return ['cmp',f,c.rel||'le',Number(c.val)]}
function bFilterPred(){const parts=[['meets_req'],['not',['is','disabled']]];bFilters.forEach(c=>{const t=bCondTerm(c);if(t)parts.push(t)});return ['and',...parts]}
function bScoreTerm(s){const p=String(s.field).split(':');let base=['field',p.length>1?p[1]:p[0]];if(s.norm)base=['normalize',base];if(s.inv)base=['neg',base];const w=Number(s.w);if(isNaN(w)||w===0)return null;return ['scale',w,base]}
function bScorer(){let t=bScores.map(bScoreTerm).filter(Boolean);let sc=t.length===0?['zero']:(t.length===1?t[0]:['add',...t]);if($('bGateBreaker').checked)sc=['gate',['not',['is','breaker_open']],sc];return sc}
function bSelTerm(){const inner=$('b_selector').value==='sample'?['sample',Number($('b_temp').value)||0]:['argmax'];const k=parseInt($('b_topn').value,10);return (k>=1)?['top_k',k,inner]:inner}
function bBuildTerm(){bSync();return ['policy',bFilterPred(),bScorer(),bSelTerm(),['id'],['always',{action:'next_candidate'}]]}
function bCurrentTerm(){if(bRawMode){let v;try{v=JSON.parse($('bRawTerm').value)}catch(e){throw new Error('Raw term is not valid JSON: '+e.message)}return v}return bBuildTerm()}
function bSetMode(m){bRawMode=(m==='raw');if(bRawMode){try{$('bRawTerm').value=JSON.stringify(bBuildTerm(),null,2)}catch(e){}}$('bStructured').style.display=bRawMode?'none':'';$('bRaw').style.display=bRawMode?'':'none';document.querySelectorAll('#bModeSeg button').forEach(x=>x.classList.toggle('active',x.dataset.mode===m))}
async function bNormalize(term){const r=await fetch('/dashboard/api/policy/normalize',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({policy_ir:term})});if(r.status===401){showLogin();throw new Error('login required')}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||`normalize ${r.status}`);lastBuiltPolicy=d;$('bTerm').textContent=JSON.stringify(d.policy_ir);$('bFingerprint').textContent='fingerprint '+d.fingerprint+' · '+d.version;return d}
async function bReview(){$('bError').style.display='none';$('bRanked').innerHTML='';try{const term=bCurrentTerm();await bNormalize(term);const r=await fetch('/dashboard/api/policy/preview',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({policy_ir:term})});if(r.status===401){showLogin();return}const pd=await r.json();if(!r.ok)throw new Error(pd.error?.message||`preview ${r.status}`);$('bRanked').innerHTML=table(pd.ranked||[],[{label:'Provider',f:x=>esc(x.provider||'—')},{label:'Model',f:x=>`<code>${esc(x.model_family||'—')}</code>`},{label:'Tier',f:x=>esc(x.tier||'—')},{label:'$/Mtok in',cls:'right',f:x=>x.price_in==null?'—':esc(x.price_in)},{label:'$/Mtok out',cls:'right',f:x=>x.price_out==null?'—':esc(x.price_out)},{label:'Score',cls:'right',f:x=>x.score==null?'—':Number(x.score).toFixed(4)}])||'<div class=empty>No survivors — your filter removed everything.</div>'}catch(e){bFail(e.message)}}
async function bDownload(){$('bError').style.display='none';try{const term=bCurrentTerm();const d=await bNormalize(term);const blob=new Blob([JSON.stringify({version:d.version,fingerprint:d.fingerprint,policy_ir:d.policy_ir},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='sigma-pol-'+(d.fingerprint||'policy')+'.json';a.click();URL.revokeObjectURL(a.href);toast('Policy downloaded — POST it as `policy_ir` in /v1/chat/completions')}catch(e){bFail(e.message)}}
async function bTest(){$('bError').style.display='none';$('bTestResult').innerHTML='<div class="muted small" style="margin-top:8px">Running…</div>';try{const term=bCurrentTerm();const prompt=$('bTestPrompt').value.trim()||'Reply exactly: pong';const r=await fetch('/dashboard/api/policy/test',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({policy_ir:term,prompt})});if(r.status===401){showLogin();return}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||(typeof d.error==='string'?d.error:'test '+r.status));const okc=d.ok?'':'bad';const out=d.text?`<pre class="mono small" style="white-space:pre-wrap;margin-top:8px;background:rgba(255,255,255,.02);border:1px solid var(--line);border-radius:8px;padding:8px">${esc(d.text)}</pre>`:`<div class="bad small" style="margin-top:8px">${esc(d.error||'no output')}</div>`;$('bTestResult').innerHTML=`<div class="actDetailBox"><div class="actMeta"><span class="pill ${okc}">status ${esc(d.status)}</span><span class="pill">${esc(d.provider||'—')}${d.served_model_id?' · '+esc(d.served_model_id):''}</span><span class="pill">cost ${d.cost_usd==null?'—':'$'+Number(d.cost_usd).toFixed(6)}</span><span class="pill">tokens ${fmt(d.tokens_total)}</span><span class="pill">${fmt(Math.round(d.latency_ms||0))} ms</span></div>${out}<div class="muted small" style="margin-top:6px">Recorded in <b>Activity</b> — open that tab for the full fallback trace.</div></div>`;toast('Test call done')}catch(e){bFail(e.message)}}
async function loadBuilderFamilies(){try{const r=await fetch('/dashboard/api/policies',{credentials:'same-origin'});if(!r.ok)return;const d=await r.json();const fams=new Set();(d.profiles||[]).forEach(p=>(p.models||[]).forEach(m=>{if(m.name)fams.add(m.name)}));$('familyOptions').innerHTML=[...fams].sort().map(f=>`<option value="${esc(f)}">`).join('')}catch(e){}}
async function loadBuilderFields(){try{const r=await fetch('/dashboard/api/fields',{credentials:'same-origin'});if(!r.ok)return;const d=await r.json();const num=[],bool=[];(d.fields||[]).forEach(f=>{const e={name:f.name,group:f.group||'model'};if(f.sort==='Bool')bool.push(e);else if(f.sort==='Num')num.push(e)});if(num.length)bFields.num=num;bFields.bool=bool;if(activeTab==='builder'&&!bRawMode)bRender()}catch(e){}}
// One-click teaching policies, authored as structured state (filters/scores/pick).
const B_EXAMPLES={ex1:{filters:[{field:'in_top_k',k:'5',by:'bench_intelligence'},{field:'in_top_k',k:'5',by:'bench_coding'}],scores:[{field:'field:price_in',w:'1',norm:true,inv:true}],gate:false,selector:'argmax',topn:''},ex2:{filters:[],scores:[{field:'field:bench_intelligence',w:'1',norm:true,inv:false},{field:'field:bench_coding',w:'1',norm:true,inv:false},{field:'field:bench_agentic',w:'1',norm:true,inv:false}],gate:false,selector:'argmax',topn:'3'}};
function bLoadExample(name){const e=B_EXAMPLES[name];if(!e)return;bSetMode('structured');bFilters=JSON.parse(JSON.stringify(e.filters));bScores=JSON.parse(JSON.stringify(e.scores));$('bGateBreaker').checked=!!e.gate;$('b_selector').value=e.selector||'argmax';$('b_topn').value=e.topn||'';$('bTempWrap').style.display=$('b_selector').value==='sample'?'':'none';bRender();bReview()}
async function fetchProviderKey(p){const r=await fetch('/dashboard/api/provider-keys/reveal?provider='+encodeURIComponent(p),{credentials:'same-origin'});if(r.status===401){showLogin();throw new Error('login required')}if(r.status===403)throw new Error('admin session required');if(!r.ok)throw new Error(`reveal ${r.status}`);return (await r.json()).value||''}
async function revealProviderKey(p,btn){try{const holder=btn.parentElement.querySelector('.provKeyValue');if(holder){holder.remove();btn.textContent='👁';return}const v=await fetchProviderKey(p);const code=document.createElement('code');code.className='provKeyValue';code.style.cssText='display:block;word-break:break-all;margin-top:4px;font-size:11px;max-width:280px';code.textContent=v;btn.parentElement.appendChild(code);btn.textContent='🙈'}catch(e){showErr(e.message)}}
async function copyProviderKey(p){try{await navigator.clipboard.writeText(await fetchProviderKey(p));toast('Key copied')}catch(e){showErr(e.message)}}
async function editProviderKey(p){try{const key=prompt('New API key for '+p+' (replaces the current one, takes effect live):');if(key===null)return;const k=key.trim();if(!k){showErr('key is empty');return}const r=await fetch('/dashboard/api/provider-keys/update',{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify({provider:p,key:k})});if(r.status===401){showLogin();return}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||`update ${r.status}`);toast(d.applied_live?'Key updated (live)':'Key saved');load()}catch(e){showErr(e.message)}}
function parseServedModels(text){return text.split('\\n').map(s=>s.trim()).filter(Boolean).map(line=>{const i=line.indexOf('=');if(i<0)return {family:line};return {family:line.slice(0,i).trim(),provider_model_id:line.slice(i+1).trim()}})}
async function addProvider(){try{const payload={id:$('addProvId').value.trim().toLowerCase(),base_url:$('addProvBaseUrl').value.trim(),tier:$('addProvTier').value,auth_env:$('addProvEnv').value.trim(),key:$('addProvKey').value.trim(),served_models:parseServedModels($('addProvModels').value)};const r=await fetch('/dashboard/api/provider-keys/add',{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify(payload)});if(r.status===401){$('addProvResult').textContent='Session expired — log in and submit again (your form values are kept).';showLogin();return}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||`add ${r.status}`);$('addProvResult').textContent=d.applied_live?`${d.provider} added and live.`:(d.note||'saved');$('addProvKey').value='';toast(d.applied_live?'Provider live':'Provider saved');load()}catch(e){$('addProvResult').textContent=e.message;showErr(e.message)}}
async function loadCodexAccounts(){try{const r=await fetch('/dashboard/api/codex/accounts',{credentials:'same-origin'});if(r.status===401){showLogin();return}if(r.status===403){$('codexAccounts').innerHTML='<div class="muted small">Admin session required to manage codex accounts.</div>';return}if(!r.ok)throw new Error('codex '+r.status);renderCodexAccounts(await r.json())}catch(e){showErr(e.message)}}
function renderCodexAccounts(d){const accts=(d&&d.accounts)||[];const a=(d&&d.activity)||{};const pr=a.scarcity_price_in;const summary=`<div class="rowMeta" style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:8px"><span>requests <b>${a.requests??0}</b></span><span>errors <b>${a.errors??0}</b>${a.error_rate?` · ${(a.error_rate*100).toFixed(1)}%`:''}</span><span>quota used <b>${a.used_percent==null?'—':a.used_percent+'%'}</b></span><span>recent 429 <b>${a.recent_429==null?'—':a.recent_429}</b></span><span>scarcity price <b>${pr==null?'—':'$'+Number(pr).toFixed(2)+'/Mtok'}</b></span></div>`;if(!accts.length){$('codexAccounts').innerHTML=summary+'<div class="muted small">No codex accounts yet — add one to enable the ChatGPT-subscription provider.</div>';return}$('codexAccounts').innerHTML=summary+table(accts,[{label:'Account',f:r=>`<div class="rowTitle">${esc(r.name)} ${r.active?'<span class="pill ok">active</span>':''}</div>`},{label:'account_id',f:r=>esc(r.account_id||'—')},{label:'Token',f:r=>r.fingerprint?esc(r.fingerprint):'<span class="pill bad">no token</span>'},{label:'',cls:'right',f:r=>`<button class="btn iconBtn ghost" title="Delete account" onclick="deleteCodexAccount(${jsarg(r.name)})">🗑</button>`}])}
async function addCodexAccount(){try{const name=$('addCodexName').value.trim();const auth_json=$('addCodexJson').value.trim();if(!name){$('addCodexResult').textContent='account name required';return}if(!auth_json){$('addCodexResult').textContent='paste the auth.json';return}const r=await fetch('/dashboard/api/codex/accounts',{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify({name,auth_json})});if(r.status===401){showLogin();return}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||`add ${r.status}`);$('addCodexResult').textContent=d.applied_live?`${d.account} added and live.`:(d.note||'saved');$('addCodexJson').value='';toast(d.applied_live?'Codex account live':'Codex account saved');loadCodexAccounts()}catch(e){$('addCodexResult').textContent=e.message;showErr(e.message)}}
async function loadConfig(){try{clearErr();const r=await fetch('/dashboard/api/config',{credentials:'same-origin'});if(r.status===401){showLogin();return}if(r.status===403){$('config').innerHTML='<div class="muted small">Admin session required to edit config.</div>';return}if(!r.ok)throw new Error('config '+r.status);renderConfig((await r.json()).knobs||[])}catch(e){showErr(e.message)}}
function renderConfig(knobs){const groups={};knobs.forEach(k=>{(groups[k.provider]=groups[k.provider]||[]).push(k)});const order=['antseed','codex','openrouter'];const provs=order.filter(p=>groups[p]).concat(Object.keys(groups).filter(p=>!order.includes(p)));$('config').innerHTML=provs.map(p=>`<div class="card span12"><div class="label">${esc(p)}</div><div style="display:flex;flex-direction:column;gap:12px;margin-top:10px">${groups[p].map(k=>`<div class="rowMeta" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap"><div style="min-width:300px"><b>${esc(k.label)}</b>${k.overridden?' <span class="pill warn">override</span>':''}<div class="muted small">${esc(k.help)}</div></div><input id="cfg_${esc(k.key)}" type="number" step="any" value="${k.value}" style="width:120px"><span class="muted small">default ${k.default} · [${k.min}, ${k.max}]</span><button class="btn ghost small" onclick="saveConfigKnob(${jsarg(k.key)})">Save</button>${k.overridden?`<button class="btn ghost small" title="Reset to default" onclick="postConfig({[${jsarg(k.key)}]:null})">Reset</button>`:''}</div>`).join('')}</div></div>`).join('')||'<div class="empty">No tunable knobs.</div>'}
async function saveConfigKnob(key){const el=$('cfg_'+key);if(!el)return;const v=el.value.trim();await postConfig({[key]:v===''?null:Number(v)})}
async function postConfig(updates){try{const r=await fetch('/dashboard/api/config',{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify({updates})});if(r.status===401){showLogin();return}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||('config '+r.status));toast(d.applied_live?'Saved · live':(d.note?'Saved · '+d.note:'Saved'));renderConfig(d.knobs||[])}catch(e){showErr(e.message)}}
async function deleteCodexAccount(name){if(!confirm('Delete codex account '+name+'?'))return;try{const r=await fetch('/dashboard/api/codex/accounts/'+encodeURIComponent(name),{method:'DELETE',credentials:'same-origin'});if(r.status===401){showLogin();return}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||`delete ${r.status}`);toast('Codex account deleted');loadCodexAccounts()}catch(e){showErr(e.message)}}
$('loginBtn').onclick=login;$('apiKeyLoginBtn').onclick=apiKeyLogin;$('password').addEventListener('keydown',e=>{if(e.key==='Enter')login()});$('apiKeyLogin').addEventListener('keydown',e=>{if(e.key==='Enter')apiKeyLogin()});$('logout').onclick=logout;$('tabOverview').onclick=()=>setTab('overview');$('tabConsumers').onclick=()=>setTab('consumers');$('tabProviderKeys').onclick=()=>setTab('providerKeys');$('tabKeyUsage').onclick=()=>setTab('keyUsage');$('tabMarket').onclick=()=>setTab('market');$('tabBuilder').onclick=()=>setTab('builder');$('tabActivity').onclick=()=>setTab('activity');$('recent').addEventListener('click',e=>{const cp=e.target.closest('[data-copyterm]');if(cp){navigator.clipboard.writeText(cp.dataset.copyterm).then(()=>toast('Policy term copied'));return}const row=e.target.closest('.actRow');if(!row)return;const det=$('recent').querySelector('.actDetail[data-d="'+row.dataset.i+'"]');if(!det)return;det.classList.toggle('hidden');const tog=row.querySelector('.actToggle');if(tog)tog.textContent=det.classList.contains('hidden')?'▸':'▾'});$('bReview').onclick=bReview;$('bDownload').onclick=bDownload;$('bTestBtn').onclick=bTest;$('bEx1').onclick=()=>bLoadExample('ex1');$('bEx2').onclick=()=>bLoadExample('ex2');$('bAddCond').onclick=()=>{bSync();bFilters.push({field:'latency_ms',rel:'le',val:''});bRender()};$('bAddOr').onclick=()=>{bSync();bFilters.push({kind:'or',subs:[{field:'latency_ms',rel:'le',val:''}]});bRender()};$('bAddScore').onclick=()=>{bSync();bScores.push({field:'field:price_in',w:'0.5',norm:true,inv:true});bRender()};$('b_selector').onchange=()=>{$('bTempWrap').style.display=$('b_selector').value==='sample'?'':'none'};document.querySelectorAll('#bModeSeg button').forEach(b=>b.onclick=()=>bSetMode(b.dataset.mode));$('bStructured').addEventListener('change',e=>{if(e.target.classList.contains('bF-field'))bSyncRender()});$('bStructured').addEventListener('click',e=>{const b=e.target.closest('[data-act]');if(!b)return;bSync();const i=+b.dataset.i,j=+b.dataset.j,act=b.dataset.act;if(act==='del')bFilters.splice(i,1);else if(act==='addsub')bFilters[i].subs.push({field:'latency_ms',rel:'le',val:''});else if(act==='delsub')bFilters[i].subs.splice(j,1);else if(act==='delscore')bScores.splice(i,1);bRender()});bRender();document.querySelector('.nav').addEventListener('click',e=>{const b=e.target.closest('[data-tab]');if(b){e.preventDefault();setTab(b.dataset.tab)}});$('refresh').onclick=()=>{if(activeTab==='policies')loadPolicies();else if(activeTab==='market')loadMarket();else if(activeTab==='keyUsage')loadKeyUsage();else load()};$('market').addEventListener('click',e=>{const h=e.target.closest('[data-fam]');if(!h)return;const fam=h.dataset.fam;if(marketOpen.has(fam))marketOpen.delete(fam);else marketOpen.add(fam);if(lastMarket)renderMarket(lastMarket)});$('marketSearch').oninput=()=>{if(lastMarket)renderMarket(lastMarket)};$('tradableOnly').checked=localStorage.getItem('tradableOnly')==='1';$('tradableOnly').onchange=()=>{localStorage.setItem('tradableOnly',$('tradableOnly').checked?'1':'0');if(lastMarket)renderMarket(lastMarket)};$('marketCopy').onclick=()=>{if(!lastMarket){showErr('No catalog data loaded yet');return}navigator.clipboard.writeText(JSON.stringify(lastMarket,null,2)).then(()=>toast('Catalog copied to clipboard')).catch(e=>showErr(e.message))};$('marketSkill').onclick=async()=>{try{const r=await fetch('/dashboard/api/skill',{credentials:'same-origin'});if(r.status===401){showLogin();return}if(!r.ok)throw new Error('skill '+r.status);const text=await r.text();const blob=new Blob([text],{type:'text/markdown'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='SKILL.md';a.click();URL.revokeObjectURL(a.href);toast('SKILL.md downloaded — load it into any assistant to author policies')}catch(e){showErr(e.message)}};$('toggleAddProvider').onclick=()=>{const c=$('addProviderCard');c.style.display=c.style.display==='none'?'':'none'};$('addProvCancel').onclick=()=>{$('addProviderCard').style.display='none'};$('addProvSubmit').onclick=addProvider;$('toggleAddCodex').onclick=()=>{const c=$('addCodexCard');c.style.display=c.style.display==='none'?'':'none'};$('addCodexCancel').onclick=()=>{$('addCodexCard').style.display='none'};$('addCodexSubmit').onclick=addCodexAccount;$('addProvId').addEventListener('blur',()=>{if(!$('addProvEnv').value.trim()&&$('addProvId').value.trim())$('addProvEnv').value=$('addProvId').value.trim().toUpperCase().replace(/[^A-Z0-9]+/g,'_')+'_API_KEY'});$('loadKeyUsage').onclick=loadKeyUsage;$('consumer').onchange=load;$('timeframe').onchange=load;$('consumerSearch').oninput=()=>renderConsumers(lastStats.keys||[]);document.querySelectorAll('#consumerStatusSeg button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#consumerStatusSeg button').forEach(x=>x.classList.remove('active'));b.classList.add('active');consumerFilterStatus=b.dataset.status;renderConsumers(lastStats.keys||[])});document.querySelectorAll('#activitySeg button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#activitySeg button').forEach(x=>x.classList.remove('active'));b.classList.add('active');activityKind=b.dataset.kind;render(lastStats)});$('newConsumerKey').onclick=()=>openDrawer('', 'create');$('closeDrawer').onclick=closeDrawer;$('drawerShade').addEventListener('click',e=>{if(e.target===$('drawerShade'))closeDrawer()});$('revealKeys').onclick=revealKeys;$('copyRevealKey').onclick=()=>navigator.clipboard.writeText($('revealKeyValue').value).then(()=>toast('Copied'));$('createKey').onclick=createKey;$('copyKey').onclick=()=>navigator.clipboard.writeText($('newKeyValue').value).then(()=>toast('Key copied'));$('copyKeyHandoff').onclick=()=>navigator.clipboard.writeText($('newKeyHandoffValue').value).then(()=>toast('Setup blurb copied'));$('saveConsumerSettings').onclick=saveConsumerSettings;$('revokeKey').onclick=revokeKey;$('anProvider').onchange=load;$('anModel').onchange=load;setTab(tabFromLocation(),{silent:true});setInterval(()=>{const ds=$('drawerShade');if(ds&&ds.classList.contains('open'))return;const ap=$('addProviderCard'),ac=$('addCodexCard');if((ap&&ap.style.display&&ap.style.display!=='none')||(ac&&ac.style.display&&ac.style.display!=='none'))return;if(document.querySelector('#recent .actDetail:not(.hidden)'))return;if(activeTab==='policies')loadPolicies();else if(activeTab==='market')loadMarket();else load()},15000);
/* ---- Flow builder: a DAG of nodes, each reusing the policy builder ---- */
let fNodes=[];let fSeq=0;let fOutput=null;
const F_DEFAULT_POLICY=()=>['policy',['and',['meets_req'],['not',['is','disabled']]],['add',['scale',0.5,['field','bench_intelligence']],['scale',0.5,['neg',['normalize',['field','price_in']]]]],['argmax'],['id'],['always',{action:'next_candidate'}]];
function fAvail(i){return ['u',...fNodes.slice(0,i).map(n=>n.id)];}
function fNodeCard(n,i){const ins=fAvail(i).map(a=>`<label class='checkRow'><input type='checkbox' class='fN-in' value='${esc(a)}'${(n.inputs||[]).includes(a)?' checked':''}> ${esc(a)}</label>`).join('');const polTag=n.custom?'custom ✓':'balanced (default)';return `<div class='fNode' data-id='${esc(n.id)}' style='border:1px solid var(--line);border-radius:10px;padding:10px;margin-top:8px'><div class='toolbar'><code>${esc(n.id)}</code><span class='muted small'>llm node</span><button class='btn' data-fact='del' data-id='${esc(n.id)}' style='margin-left:auto'>×</button></div><textarea class='fN-sys' rows='2' placeholder='System prompt for this node (e.g. Answer concisely.)' style='width:100%;margin-top:6px'>${esc(n.system||'')}</textarea><div class='muted small' style='margin-top:8px'>Inputs — its prompt is these nodes' outputs (labeled if several):</div><div class='fN-inputs' style='display:flex;gap:12px;flex-wrap:wrap;margin-top:4px'>${ins}</div><div class='toolbar' style='margin-top:8px'><span class='muted small'>policy <span class='mono'>${polTag}</span></span><button class='btn' data-fact='usepol' data-id='${esc(n.id)}'>↧ use Policy-builder term</button><button class='btn' data-fact='editpol' data-id='${esc(n.id)}'>edit raw</button></div><textarea class='fN-pol mono small' data-id='${esc(n.id)}' spellcheck='false' style='display:none;width:100%;min-height:110px;margin-top:6px;background:rgba(255,255,255,.02);border:1px solid var(--line);border-radius:8px;padding:8px'>${esc(JSON.stringify(n.policy))}</textarea></div>`;}
function fRender(){$('fNodes').innerHTML=fNodes.length?fNodes.map(fNodeCard).join(''):'<div class="muted small">No nodes yet — add one.</div>';$('fOutput').innerHTML=fNodes.map(n=>`<option value='${esc(n.id)}'${n.id===fOutput?' selected':''}>${esc(n.id)}</option>`).join('')||"<option value=''>—</option>";}
function fSync(){[...$('fNodes').querySelectorAll('.fNode')].forEach(el=>{const n=fNodes.find(x=>x.id===el.dataset.id);if(!n)return;n.system=el.querySelector('.fN-sys').value;n.inputs=[...el.querySelectorAll('.fN-in:checked')].map(c=>c.value);const pol=el.querySelector('.fN-pol');if(pol&&pol.style.display!=='none'){try{n.policy=JSON.parse(pol.value);n.custom=true}catch(e){}}});if($('fOutput').value)fOutput=$('fOutput').value;}
function fBuildIR(){fSync();const nodes={u:{kind:'input'}};fNodes.forEach(n=>{nodes[n.id]={kind:'llm',system:n.system||'',policy:n.policy,inputs:(n.inputs&&n.inputs.length)?n.inputs:['u']}});nodes.out={kind:'output',inputs:[fOutput||(fNodes.length?fNodes[fNodes.length-1].id:'u')]};return ['flow',nodes];}
function fAddNode(){fSync();const id='n'+(++fSeq);fNodes.push({id,system:'',inputs:[fNodes.length?fNodes[fNodes.length-1].id:'u'],policy:F_DEFAULT_POLICY(),custom:false});fOutput=id;fRender();}
function fFail(m){$('fError').style.display='block';$('fError').textContent=m;}
async function fNormalize(){const ir=fBuildIR();const r=await fetch('/dashboard/api/flow/normalize',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({flow_ir:ir})});if(r.status===401){showLogin();throw new Error('login required')}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||`normalize ${r.status}`);$('fTerm').textContent=JSON.stringify(d.flow_ir);$('fFingerprint').textContent='fingerprint '+d.fingerprint+' · '+d.version;return d;}
async function fReview(){$('fError').style.display='none';try{await fNormalize();toast('Flow admitted ✓')}catch(e){fFail(e.message)}}
async function fDownload(){$('fError').style.display='none';try{const d=await fNormalize();const blob=new Blob([JSON.stringify({version:d.version,fingerprint:d.fingerprint,flow_ir:d.flow_ir},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='sigma-flow-'+(d.fingerprint||'flow')+'.json';a.click();URL.revokeObjectURL(a.href);toast('Flow downloaded — POST it as `flow_ir` in /v1/chat/completions')}catch(e){fFail(e.message)}}
async function fTest(){$('fError').style.display='none';$('fTestResult').innerHTML='<div class="muted small" style="margin-top:8px">Running…</div>';try{const ir=fBuildIR();const prompt=$('fTestPrompt').value.trim()||'What is 17 * 23?';const r=await fetch('/dashboard/api/flow/test',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json'},body:JSON.stringify({flow_ir:ir,prompt})});if(r.status===401){showLogin();return}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||(typeof d.error==='string'?d.error:'test '+r.status));const okc=d.ok?'':'bad';const out=d.text?`<pre class="mono small" style="white-space:pre-wrap;margin-top:8px;background:rgba(255,255,255,.02);border:1px solid var(--line);border-radius:8px;padding:8px">${esc(d.text)}</pre>`:`<div class="bad small" style="margin-top:8px">${esc(d.error||'no output')}</div>`;const nodes=Array.isArray(d.decision_trace&&d.decision_trace.flow_nodes)?d.decision_trace.flow_nodes:[];$('fTestResult').innerHTML=`<div class="actDetailBox"><div class="actMeta"><span class="pill ${okc}">status ${esc(d.status)}</span><span class="pill">${nodes.length} node${nodes.length===1?'':'s'}</span><span class="pill">cost ${d.cost_usd==null?'—':'$'+Number(d.cost_usd).toFixed(6)}</span><span class="pill">tokens ${fmt(d.tokens_total)}</span><span class="pill">${fmt(Math.round(d.latency_ms||0))} ms</span></div>${nodes.map(actFlowNode).join('')}${out}<div class="muted small" style="margin-top:6px">Recorded in <b>Activity</b> — open that tab for the full per-node trace.</div></div>`;toast('Flow test done')}catch(e){fFail(e.message)}}
function fLoadExample(){fSeq=3;fNodes=[{id:'n1',system:'Answer the question concisely.',inputs:['u'],policy:F_DEFAULT_POLICY(),custom:false},{id:'n2',system:'Answer the question rigorously, showing your reasoning.',inputs:['u'],policy:F_DEFAULT_POLICY(),custom:false},{id:'n3',system:'You are given two draft answers. Synthesize the single best final answer.',inputs:['n1','n2'],policy:F_DEFAULT_POLICY(),custom:false}];fOutput='n3';fRender();}
function setBuilderKind(k){const flow=k==='flow';$('policyBuilder').style.display=flow?'none':'';$('flowBuilder').style.display=flow?'':'none';document.querySelectorAll('#builderKindSeg button').forEach(b=>b.classList.toggle('active',b.dataset.kind===k));if(flow&&!fNodes.length)fLoadExample();}
$('builderKindSeg').addEventListener('click',e=>{const b=e.target.closest('[data-kind]');if(b)setBuilderKind(b.dataset.kind)});
$('fAddNode').onclick=fAddNode;$('fReview').onclick=fReview;$('fDownload').onclick=fDownload;$('fTestBtn').onclick=fTest;$('fEx1').onclick=fLoadExample;$('fOutput').onchange=()=>{fOutput=$('fOutput').value};
$('flowBuilder').addEventListener('click',e=>{const b=e.target.closest('[data-fact]');if(!b)return;fSync();const id=b.dataset.id,act=b.dataset.fact,n=fNodes.find(x=>x.id===id);if(act==='del'){fNodes=fNodes.filter(x=>x.id!==id);if(fOutput===id)fOutput=fNodes.length?fNodes[fNodes.length-1].id:null;fRender()}else if(act==='usepol'){if(n){try{n.policy=bCurrentTerm();n.custom=true;toast('Captured the Policy-builder term into '+id)}catch(err){fFail(err.message)}fRender()}}else if(act==='editpol'){const ta=$('flowBuilder').querySelector('.fN-pol[data-id="'+id+'"]');if(ta)ta.style.display=ta.style.display==='none'?'':'none'}});
</script>
</body></html>"""
    return html.replace("__PUBLIC_BASE_URL__", _public_base_url())
