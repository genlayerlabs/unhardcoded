"""Host-side per-route effective-cost fold (the measured-economics member of the
route_* family, twin of route_latency).

Price is the one ranking input the host still mostly DECLARES (sources stamp a
pulled/advertised price) instead of MEASURING. This is the measurement: a
per-route EMA of the EFFECTIVE blended cost a route actually charged, in USD per
million tokens, derived from the provider-reported cost of each successful call
(`response.cost_reported`, e.g. OpenRouter `usage.cost`) over its token count.

It exposes TRUTH, not (yet) a ranking lever: `snapshot()` lets the operator see
what each route REALLY cost versus the price the catalog stamped — the gap that
matters most for marketplace peers, which can advertise a price they do not
settle. Turning this measured cost into a ranking correction changes selection
(reserved form, not observability) and is a separate, deliberate step.

Only SUCCESSFUL calls with a reported cost (>= 0) and a positive token count fold
in: a failure carries no honest cost, and a provider that reports no cost (tokens
only — e.g. native OpenAI) leaves the route unmeasured (None), so the catalog's
stamped price stands.

The blended $/Mtok mixes input and output tokens, so it shifts with the in/out
ratio of traffic; the EMA smooths it. The ranking-grade per-direction price is a
later concern. In-process (resets on restart), like the rest of the route_*
family; reuses `route_reliability.route_key` so a route's cost, latency and
reliability share one identity.
"""
from __future__ import annotations

import threading

from route_reliability import route_key  # shared route identity  # noqa: F401

# The per-call effective rate is fairly stable per route but shifts with the
# in/out token mix; smooth like latency (0.3), not as heavily as reliability (0.2).
_ALPHA = 0.3

_lock = threading.Lock()
_ema: dict[str, float] = {}


def observe(key: str, cost_usd: "float | None", tokens: "int | float | None",
            ok: bool) -> None:
    """Fold one call's effective $/Mtok into the route's EMA. Ignored unless the
    call SUCCEEDED, the provider reported a non-negative cost, and the token count
    is positive — anything else carries no honest cost signal. A reported cost of
    0 (a genuinely free route) folds as 0; an absent cost (None) does not fold."""
    if not ok or cost_usd is None or tokens is None:
        return
    try:
        c = float(cost_usd)
        n = float(tokens)
    except (TypeError, ValueError):
        return
    if c < 0 or n <= 0:
        return
    rate = c / n * 1_000_000.0
    with _lock:
        cur = _ema.get(key)
        _ema[key] = rate if cur is None else _ALPHA * rate + (1.0 - _ALPHA) * cur


def usd_per_mtok(key: str) -> "float | None":
    """The route's folded effective cost in USD per million tokens (rounded), or
    None if never observed with a reported cost."""
    v = _ema.get(key)
    return None if v is None else round(v, 4)


def snapshot() -> dict[str, float]:
    with _lock:
        return {k: round(v, 4) for k, v in _ema.items()}


def reset() -> None:
    """Test hook."""
    with _lock:
        _ema.clear()
