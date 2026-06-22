"""Host-side per-route reliability fold (the host half of the reliability form).

MEASURING reliability is the host's job — llm-router #14 made the algebra read it
as a per-candidate field (`offer.success_rate`), like price, observed pointwise.
This is the host's measurement: a per-route success-rate EMA over observed call
outcomes, stamped onto AntSeed offers so the router prefers the reliable seller
and rotates off a broken one (a different route for the same family).

The measurement is deliberately the simplest honest one — a binary success EMA.
All of the richer judgment the form allows (weighting by error kind, latency-with-
error, population-relative, windowing) lives here, host-side, and can grow without
touching the algebra. Folding happens in `llm_router_host` on each call outcome;
`offers_sync` reads `success_rate(...)` to stamp the offer.

In-process (resets on restart), exactly like the engine EMA it replaces for
marketplace routes; a missing route reads None → the offer is left unstamped → the
algebra falls back to its own coarse EMA / the field default.
"""
from __future__ import annotations

import threading

# Same smoothing the engine EMA used (ema_alpha). First observation seeds the
# rate directly; subsequent ones decay geometrically toward it.
_ALPHA = 0.2

_lock = threading.Lock()
_rates: dict[str, float] = {}


def route_key(provider_id: str, model_family: str, peer_id: str) -> str:
    """Identity of a route = a specific seller peer serving a specific family.
    Stays entirely host-internal; the algebra never sees it."""
    return f"{provider_id}|{model_family}|{peer_id}"


def observe(key: str, ok: bool) -> None:
    s = 1.0 if ok else 0.0
    with _lock:
        cur = _rates.get(key)
        _rates[key] = s if cur is None else _ALPHA * s + (1.0 - _ALPHA) * cur


def success_rate(key: str) -> float | None:
    """The route's folded success rate, or None if never observed."""
    return _rates.get(key)


def snapshot() -> dict[str, float]:
    with _lock:
        return dict(_rates)


def reset() -> None:
    """Test hook."""
    with _lock:
        _rates.clear()
