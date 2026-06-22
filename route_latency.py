"""Host-side per-route latency fold (the host half of the latency form).

The twin of `route_reliability`: MEASURING speed is the host's job, and the
algebra already reads it as a per-candidate field (`offer.latency_ms`, a CORE
field, observed pointwise like price/success_rate). This is the host's
measurement: a per-route EMA of observed end-to-end call latency, stamped onto
offers in `offers_sync` so a policy can gate/score on speed — a slow $0
marketplace peer (e.g. a 12 s antseed glm-5.2) ranks below a fast partner
(a ~0.9 s gpt-5.5), instead of winning every cost-led policy and stalling the
caller.

Only SUCCESSFUL calls fold in. A fast failure (an antseed peer returning
"empty assistant content" in 1.9 s) must NOT make a broken route look fast — that
is reliability's concern (`route_reliability`), and counting its latency would
reward exactly the route we want to avoid.

The measurement is deliberately the simplest honest one — an EMA of total
response latency (TTFT + generation), which is what the non-streaming backend can
observe per call. Richer judgment the form allows (true TTFT from the first
stream chunk, p50/p95 windowing, population-relative) can grow here, host-side,
without touching the algebra. Folding happens in `llm_router_host` on each call
outcome; `offers_sync` reads `latency_ms(...)` to stamp the offer.

In-process (resets on restart), exactly like `route_reliability`; a missing route
reads None -> the offer is left unstamped -> the algebra falls back to the field
default (an unmeasured route is optimistically routable, then learns down on its
first slow observation — the same optimistic-default discipline as
`route_reliability` / `route_tool_capability`). Reuses `route_reliability.route_key`
so a route's latency and reliability share one identity.
"""
from __future__ import annotations

import threading

from route_reliability import route_key  # shared route identity  # noqa: F401  (re-exported)

# Latency shifts with load faster than reliability does, so smooth a little less
# heavily (weight a fresh sample more) than route_reliability's 0.2.
_ALPHA = 0.3

_lock = threading.Lock()
_ema: dict[str, float] = {}


def observe(key: str, latency_ms: "int | float | None", ok: bool) -> None:
    """Fold one observed call latency into the route's EMA. Ignored unless the
    call SUCCEEDED and the latency is a positive number — a failed/instant
    outcome carries no honest speed signal."""
    if not ok or latency_ms is None:
        return
    try:
        v = float(latency_ms)
    except (TypeError, ValueError):
        return
    if v <= 0:
        return
    with _lock:
        cur = _ema.get(key)
        _ema[key] = v if cur is None else _ALPHA * v + (1.0 - _ALPHA) * cur


def latency_ms(key: str) -> "int | None":
    """The route's folded latency in ms (rounded), or None if never observed."""
    v = _ema.get(key)
    return None if v is None else round(v)


def snapshot() -> dict[str, int]:
    with _lock:
        return {k: round(v) for k, v in _ema.items()}


def reset() -> None:
    """Test hook."""
    with _lock:
        _ema.clear()
