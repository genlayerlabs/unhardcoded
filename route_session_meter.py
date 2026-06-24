"""Per-session usage meter: per-call is already on x_router; this accumulates the
RUNNING TOTAL per session (the sid the caller sends), so a conversation/agent has
both numbers — this call AND everything it has spent so far. Pairs with
route_cache (same session key): route_cache says which peer is hot, this says what
the session has cost and how much of it was served from cache.

Tracks per session: calls, tokens_in, tokens_out, tokens_cached, cost_usd. The
cache hit ratio (tokens_cached / tokens_in) and the realized cost are then exact
per session, using the provider's reported cost — no model-price guessing.

In-process (resets on restart), like route_cache / route_latency; fleet scale
needs a shared store with TTL — the same debt the sibling forms carry.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_acc: dict[str, dict] = {}   # session -> running totals
# session -> {model_family -> {family, provider, served_by}}: the routes that
# successfully served the session, i.e. which models/peers are WARM (hold the
# session's prompt-cache prefix). For DISPLAY (the warm panel); the affinity
# decision stays in route_cache. Per family, so a flow's glm AND gpt both show.
_warm: dict[str, dict[str, dict]] = {}
# session -> owning consumer key (the authed caller that FIRST wrote this sid).
# A session's economics (cost/tokens/cache + warm peers) belong to one consumer;
# this binding lets the consumer-facing view refuse to disclose another
# consumer's session (cross-consumer isolation). First-writer-wins: a different
# consumer reusing someone else's opaque sid must NOT steal or overwrite it.
_owner: dict[str, str] = {}


def observe(session: "str | None", *, tokens_in=0, tokens_out=0,
            tokens_cached=0, cost_usd=0.0, owner: "str | None" = None) -> "dict | None":
    """Fold one call's usage into the session's running total; return the new
    accumulated totals (so the caller can put per-call AND acc on the response).
    No-op (returns None) when the caller named no session. When `owner` (the
    authed consumer key) is given, bind sid->owner first-writer-wins so the
    consumer-facing view can scope reads to the owning consumer."""
    if not session:
        return None
    with _lock:
        if owner:
            _owner.setdefault(session, owner)
        a = _acc.get(session)
        if a is None:
            a = {"calls": 0, "tokens_in": 0, "tokens_out": 0,
                 "tokens_cached": 0, "cost_usd": 0.0}
            _acc[session] = a
        a["calls"] += 1
        a["tokens_in"] += int(tokens_in or 0)
        a["tokens_out"] += int(tokens_out or 0)
        a["tokens_cached"] += int(tokens_cached or 0)
        a["cost_usd"] = round(a["cost_usd"] + float(cost_usd or 0.0), 6)
        return dict(a)


def observe_route(session: "str | None", provider: "str | None",
                  family: "str | None", served_by: "str | None") -> None:
    """Record (for DISPLAY) that `family` was served warm for this session by
    `provider` via `served_by` (the peer / real backend behind a marketplace, or
    the provider itself for direct routes). Keyed per family so a multi-family
    flow shows all warm models. No-op without a session/family."""
    if not session or not family:
        return
    with _lock:
        w = _warm.get(session)
        if w is None:
            w = {}
            _warm[session] = w
        w[family] = {"family": family, "provider": provider,
                     "served_by": served_by or provider}


def warm(session: "str | None") -> list[dict]:
    """The session's warm routes: [{family, provider, served_by}], one per family."""
    if not session:
        return []
    with _lock:
        return list((_warm.get(session) or {}).values())


def owner(session: "str | None") -> "str | None":
    """The consumer key that owns this session (first writer), or None for an
    unknown / unnamed session. Used to scope the consumer-facing session view."""
    if not session:
        return None
    with _lock:
        return _owner.get(session)


def get(session: "str | None") -> "dict | None":
    if not session:
        return None
    with _lock:
        a = _acc.get(session)
        return dict(a) if a else None


def snapshot() -> dict[str, dict]:
    with _lock:
        return {s: dict(a) for s, a in _acc.items()}


def reset() -> None:
    """Test hook."""
    with _lock:
        _acc.clear()
        _warm.clear()
        _owner.clear()
