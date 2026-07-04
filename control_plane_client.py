"""Client for an external control plane ("bring your own control plane").

An operator can front this router with a separate control plane that owns
consumer keys and per-tenant provider credentials (any service speaking the
small HTTP contract below). The feature is OFF unless both CONTROL_PLANE_URL
and CONTROL_PLANE_INTERNAL_SECRET are set; nothing in this module assumes any
particular control-plane implementation.

Contract (all requests carry the shared secret in `x-internal-secret`):
  GET {CONTROL_PLANE_URL}/internal/keys/resolve?sha256=<64hex>
      -> {"active": bool, "consumer": str, "tenant_id": int,
          "rate_per_min": int|null, "burst": int|null}
  GET {CONTROL_PLANE_URL}/internal/tenants/<id>/provider-env
      -> {"env": {ENV_NAME: secret, ...}}

The module is a leaf (no imports from auth_proxy/shim) shared by the ingress
(key resolution) and the router (per-tenant provider env). Secrets are never
logged — events carry env NAMES and tenant ids only.
"""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

log = logging.getLogger("llm-router-control-plane")

CONTROL_PLANE_URL = os.getenv("CONTROL_PLANE_URL", "").rstrip("/")
CONTROL_PLANE_INTERNAL_SECRET = os.getenv("CONTROL_PLANE_INTERNAL_SECRET", "")
RESOLVE_TTL_S = float(os.getenv("CP_RESOLVE_TTL_S", "60"))
NEGATIVE_TTL_S = float(os.getenv("CP_NEGATIVE_TTL_S", "15"))
RESOLVE_STALE_GRACE_S = float(os.getenv("CP_RESOLVE_STALE_GRACE_S", "300"))
TENANT_ENV_TTL_S = float(os.getenv("CP_TENANT_ENV_TTL_S", "120"))
TENANT_ENV_STALE_GRACE_S = float(os.getenv("CP_TENANT_ENV_STALE_GRACE_S", "600"))
ENV_ALLOWLIST = {
    name.strip()
    for name in os.getenv(
        "CP_ENV_ALLOWLIST",
        "OPENAI_API_KEY,OPENROUTER_API_KEY,ANTHROPIC_API_KEY,GEMINI_API_KEY",
    ).split(",")
    if name.strip()
}

_RESOLVE_CACHE_MAX = 4096
_TIMEOUT = httpx.Timeout(3.0, connect=1.5)


def enabled() -> bool:
    return bool(CONTROL_PLANE_URL and CONTROL_PLANE_INTERNAL_SECRET)


def internal_secret_ok(headers: Mapping[str, str]) -> bool:
    """Validate an inbound `x-internal-secret` header. False when the shared
    secret is unconfigured — callers must treat that as 'feature hidden'."""
    if not CONTROL_PLANE_INTERNAL_SECRET:
        return False
    presented = headers.get("x-internal-secret") or ""
    return hmac.compare_digest(presented, CONTROL_PLANE_INTERNAL_SECRET)


@dataclass
class ResolvedKey:
    active: bool
    consumer: str | None
    tenant_id: int | None
    rate_per_min: int | None
    burst: int | None
    fetched_at: float  # time.monotonic()


_resolve_cache: dict[str, ResolvedKey] = {}  # sha256 hex -> entry (positive AND negative)
_resolve_inflight: dict[str, asyncio.Future] = {}
_tenant_env_cache: dict[int, tuple[dict[str, str], float]] = {}
_TENANT_ENV: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "cp_tenant_env", default=None
)
_client: httpx.AsyncClient | None = None
_collision_logged: set[str] = set()


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=_TIMEOUT)
    return _client


def sha256_hex(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _entry_ttl(entry: ResolvedKey) -> float:
    return RESOLVE_TTL_S if entry.active else NEGATIVE_TTL_S


def _evict_if_full() -> None:
    if len(_resolve_cache) < _RESOLVE_CACHE_MAX:
        return
    now = time.monotonic()
    expired = [k for k, e in _resolve_cache.items() if now - e.fetched_at > _entry_ttl(e)]
    for k in expired:
        _resolve_cache.pop(k, None)
    if len(_resolve_cache) >= _RESOLVE_CACHE_MAX:
        _resolve_cache.clear()


def _parse_resolved(data: Any) -> ResolvedKey:
    if not isinstance(data, dict):
        data = {}

    def _opt_int(value: Any) -> int | None:
        try:
            out = int(value)
        except (TypeError, ValueError):
            return None
        return out if out > 0 else None

    consumer = str(data.get("consumer") or "").strip() or None
    active = bool(data.get("active")) and consumer is not None
    return ResolvedKey(
        active=active,
        consumer=consumer if active else None,
        tenant_id=_opt_int(data.get("tenant_id")) if active else None,
        rate_per_min=_opt_int(data.get("rate_per_min")) if active else None,
        burst=_opt_int(data.get("burst")) if active else None,
        fetched_at=time.monotonic(),
    )


async def _fetch_resolve(digest: str) -> ResolvedKey | None:
    """One HTTP resolve. Returns the parsed entry (positive or negative) on a
    definitive control-plane answer, None on transport error / 5xx."""
    try:
        resp = await _get_client().get(
            f"{CONTROL_PLANE_URL}/internal/keys/resolve",
            params={"sha256": digest},
            headers={"x-internal-secret": CONTROL_PLANE_INTERNAL_SECRET},
        )
    except httpx.HTTPError as exc:
        log.warning(json.dumps({"event": "cp_resolve_error", "error": type(exc).__name__}))
        return None
    if resp.status_code >= 500:
        log.warning(json.dumps({"event": "cp_resolve_error", "status": resp.status_code}))
        return None
    if resp.status_code != 200:
        # 403 (secret mismatch) etc. — a definitive "no": cache as negative so a
        # misconfigured secret can't turn into a per-request CP hammer.
        log.warning(json.dumps({"event": "cp_resolve_rejected", "status": resp.status_code}))
        return _parse_resolved({})
    try:
        return _parse_resolved(resp.json())
    except ValueError:
        return _parse_resolved({})


async def resolve_key(digest: str) -> ResolvedKey | None:
    """Resolve a key digest against the control plane, with caching.

    Returns None when the feature is off or the CP is unreachable with no
    usable cache (the caller should 401). A stale positive entry is served for
    up to RESOLVE_STALE_GRACE_S past its TTL, but ONLY when the CP is
    unreachable — a definitive answer always replaces the cache. Negative
    entries never get grace.
    """
    if not enabled():
        return None
    now = time.monotonic()
    cached = _resolve_cache.get(digest)
    if cached is not None and now - cached.fetched_at <= _entry_ttl(cached):
        return cached

    pending = _resolve_inflight.get(digest)
    if pending is not None:
        return await asyncio.shield(pending)

    future: asyncio.Future = asyncio.get_running_loop().create_future()
    _resolve_inflight[digest] = future
    try:
        fresh = await _fetch_resolve(digest)
        if fresh is not None:
            _evict_if_full()
            _resolve_cache[digest] = fresh
            result: ResolvedKey | None = fresh
        elif (
            cached is not None
            and cached.active
            and now - cached.fetched_at <= RESOLVE_TTL_S + RESOLVE_STALE_GRACE_S
        ):
            log.warning(json.dumps({"event": "cp_resolve_stale_grace", "consumer": cached.consumer}))
            result = cached
        else:
            _resolve_cache.pop(digest, None)
            result = None
        future.set_result(result)
        return result
    except BaseException as exc:
        future.set_exception(exc)
        raise
    finally:
        _resolve_inflight.pop(digest, None)


async def tenant_env(tenant_id: int) -> dict[str, str]:
    """Cached BYO provider env for a tenant, filtered through ENV_ALLOWLIST.
    Fail-soft: refetch error -> stale within grace -> {} (platform keys)."""
    if not enabled():
        return {}
    now = time.monotonic()
    cached = _tenant_env_cache.get(tenant_id)
    if cached is not None and now - cached[1] <= TENANT_ENV_TTL_S:
        return cached[0]
    try:
        resp = await _get_client().get(
            f"{CONTROL_PLANE_URL}/internal/tenants/{int(tenant_id)}/provider-env",
            headers={"x-internal-secret": CONTROL_PLANE_INTERNAL_SECRET},
        )
        resp.raise_for_status()
        raw = resp.json().get("env")
    except (httpx.HTTPError, ValueError) as exc:
        if cached is not None and now - cached[1] <= TENANT_ENV_TTL_S + TENANT_ENV_STALE_GRACE_S:
            log.warning(json.dumps({"event": "tenant_env_stale_grace", "tenant_id": tenant_id}))
            return cached[0]
        log.warning(json.dumps({
            "event": "tenant_env_fallback", "tenant_id": tenant_id, "error": type(exc).__name__,
        }))
        return {}
    env = {
        str(k): str(v)
        for k, v in (raw or {}).items()
        if str(k) in ENV_ALLOWLIST and isinstance(v, str) and v
    }
    _tenant_env_cache[tenant_id] = (env, now)
    return env


def activate_tenant_env(env: dict[str, str] | None) -> contextvars.Token:
    return _TENANT_ENV.set(env or None)


def reset_tenant_env(token: contextvars.Token) -> None:
    _TENANT_ENV.reset(token)


def env_get(name: str) -> str | None:
    """Adapter credential lookup: per-request tenant map first, then process env."""
    override = _TENANT_ENV.get()
    if override and name in override:
        return override[name]
    return os.environ.get(name)


def log_collision_once(consumer: str) -> None:
    if consumer in _collision_logged:
        return
    _collision_logged.add(consumer)
    log.warning(json.dumps({"event": "cp_caller_collision", "caller": consumer}))


async def close() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def reset_for_tests() -> None:
    global _client
    _resolve_cache.clear()
    _resolve_inflight.clear()
    _tenant_env_cache.clear()
    _collision_logged.clear()
    _client = None
