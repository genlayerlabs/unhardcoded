"""Host-side per-session cache affinity (the host half of the cache_hot field).

Prompt caching is STATE between calls: a provider discounts the reused prefix
only if the SAME peer serves the session again, so the host must remember which
route last served each session and steer the next turn back to it. That memory
cannot live in the algebra (a policy is a pure function of one call) nor in a
source's `offers_sync` (request-blind, shared across sessions) — it is
per-session host state, exactly like `route_latency` is per-route host state.

The measurement is the simplest honest one: the route that most recently served
a session SUCCESSFULLY is its hot route (it holds the KV prefix). `observe`
folds each call outcome; `hot_route` returns the session's current hot route key
(or None). Only successful calls fold, exactly like `route_latency` — a failure
carries no honest "this peer holds the prefix" signal.

The algebra never sees the route key: per request the host resolves
`hot_route(session)` into `ctx.request.cache_hot_route`, and the `cache_hot`
field getter (declared host-side in `LLMRouterHost`) reconstructs each
candidate's route key the same way `_fold_route_outcome` does and compares,
exposing only a Bool. Route identity stays 100% host-internal.

In-process (resets on restart), exactly like `route_latency` /
`route_reliability`; a new or unknown session has no hot route -> no candidate
is cache_hot -> the policy routes purely on its other terms (no phantom
affinity for a fresh session). Reuses `route_reliability.route_key` so a
session's hot route shares the one route identity. Fleet-scale (multi-process)
affinity needs a shared store with TTL + eviction — the same in-process-state
debt the sibling forms already carry, not new debt.
"""
from __future__ import annotations

import threading

from route_reliability import route_key  # shared route identity  # noqa: F401  (re-exported)

_lock = threading.Lock()
# session -> route_key of the last SUCCESSFUL call on that session.
_hot: dict[str, str] = {}


def observe(session: "str | None", key: str, ok: bool) -> None:
    """Fold one call outcome. A successful call makes its route the session's hot
    route (it now holds the prefix). Failures are ignored (no honest signal), and
    a missing session is a no-op — affinity only exists for sessions the caller
    names."""
    if not session or not ok:
        return
    with _lock:
        _hot[session] = key


def hot_route(session: "str | None") -> "str | None":
    """The route key holding this session's prefix hot, or None if unknown."""
    if not session:
        return None
    return _hot.get(session)


def snapshot() -> dict[str, str]:
    with _lock:
        return dict(_hot)


def reset() -> None:
    """Test hook."""
    with _lock:
        _hot.clear()
