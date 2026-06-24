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


def observe(session: "str | None", *, tokens_in=0, tokens_out=0,
            tokens_cached=0, cost_usd=0.0) -> "dict | None":
    """Fold one call's usage into the session's running total; return the new
    accumulated totals (so the caller can put per-call AND acc on the response).
    No-op (returns None) when the caller named no session."""
    if not session:
        return None
    with _lock:
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
