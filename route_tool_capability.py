"""Host-side per-route LEARNED tool capability — the sound close to the AntSeed
optimistic `supports_tools=true` default (sources/antseed.py).

AntSeed market rows carry no capability data, so offers_sync stamps
`supports_tools=true` for every peer. KNOWN HOLE that this module closes: a peer
that ACCEPTS the `tools` field but whose model does not actually function-call
returns plain text — no tool_calls, no error — a silent tools-less answer that no
error-path retry catches.

We learn it, by the same shape as route_reliability: on each call whose REQUEST
carried tools, observe whether the RESPONSE carried tool_calls. A route
(provider|family|peer) that, over `_MIN_SAMPLES` tools-requests, emitted tool_calls
in NONE is marked tool-incapable; `offers_sync` then stops stamping supports_tools
for it, so the core's meets_req filters it out of tool requests (it still serves
non-tool requests).

Two honesty guards on the inherent ambiguity (a single response can't tell
"ignored the tools" from "the model chose not to call one"):
- the signal is statistical — `_MIN_SAMPLES` high, and ANY observed tool_call
  proves capability permanently (a capable route is never demoted);
- a tool-incapable mark EXPIRES after `_EXPIRY_MS`, so a false positive only
  sidelines a route briefly before it is re-tested.

In-process (resets on restart), like route_reliability. Route keys are built by
route_reliability.route_key — one source for the route identity.
"""
from __future__ import annotations

import threading
import time

# Mark a route tool-incapable after this many tools-requests with zero tool_calls.
_MIN_SAMPLES = 20
# A tool-incapable mark expires this long after its window started, so the route
# is re-tested (a false positive only sidelines it briefly).
_EXPIRY_MS = 30 * 60 * 1000

_lock = threading.Lock()
# route_key -> {"reqs": tools-requests seen, "calls": of which emitted tool_calls,
#               "since": now_ms when the current window started}
_state: dict[str, dict] = {}


def _now_ms() -> int:
    """Wall clock in ms. Indirected so tests can monkeypatch it."""
    return int(time.time() * 1000)


def observe(key: str, tools_requested: bool, tool_calls_emitted: bool) -> None:
    """Record one call outcome for a route. Only tools-requests carry signal."""
    if not tools_requested:
        return
    with _lock:
        st = _state.get(key)
        if st is None:
            st = {"reqs": 0, "calls": 0, "since": _now_ms()}
            _state[key] = st
        st["reqs"] += 1
        if tool_calls_emitted:
            st["calls"] += 1


def is_capable(key: str) -> bool:
    """True unless the route has proven it ignores tools (>= _MIN_SAMPLES
    tools-requests, zero tool_calls). Optimistic by default; an expired
    incapable mark is reset so the route is re-tested."""
    with _lock:
        st = _state.get(key)
        if st is None:
            return True  # optimistic default — unobserved routes are assumed capable
        if not (st["reqs"] >= _MIN_SAMPLES and st["calls"] == 0):
            return True  # capable (or not enough evidence yet)
        if _now_ms() - st["since"] >= _EXPIRY_MS:
            _state[key] = {"reqs": 0, "calls": 0, "since": _now_ms()}  # re-test
            return True
        return False


def snapshot() -> dict[str, dict]:
    with _lock:
        return {k: dict(v) for k, v in _state.items()}


def reset() -> None:
    """Test hook."""
    with _lock:
        _state.clear()
