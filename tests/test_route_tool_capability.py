"""Learned per-route tool capability (#4c): a route observed to ignore tools (a
tools-request that never returns tool_calls) is marked incapable so offers_sync
stops stamping supports_tools for it. Optimistic by default; any real tool_call
clears it; the mark is windowed, so a false positive ages out and is re-tested.
Derived on the fly from route_observations (host_store.tool_incapable_routes).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import host_store  # noqa: E402
from conftest import seed_route_obs  # noqa: E402

K = "antseed|glm-5.2|peerX"
N = 20  # _MIN_SAMPLES


def _obs(n, tool_calls_emitted=False, ts=None):
    seed_route_obs("antseed", "glm-5.2", "peerX", ok=True, n=n, ts=ts,
                   tools_requested=True, tool_calls_emitted=tool_calls_emitted)


def test_unobserved_route_is_capable(host_store_clean):
    assert K not in host_store.tool_incapable_routes()


def test_non_tool_requests_carry_no_signal(host_store_clean):
    seed_route_obs("antseed", "glm-5.2", "peerX", ok=True, n=50)  # tools_requested=False
    assert K not in host_store.tool_incapable_routes()


def test_route_that_never_emits_tool_calls_is_marked_incapable(host_store_clean):
    _obs(N - 1)
    assert K not in host_store.tool_incapable_routes()   # not enough evidence yet
    _obs(1)                                              # crosses _MIN_SAMPLES, zero tool_calls
    assert K in host_store.tool_incapable_routes()


def test_any_tool_call_clears_incapability(host_store_clean):
    _obs(1, tool_calls_emitted=True)                    # one real tool_call
    _obs(N * 2)                                         # then many without
    assert K not in host_store.tool_incapable_routes()  # capable


def test_old_observations_age_out_of_the_window(host_store_clean):
    now = int(time.time() * 1000)
    _obs(N, ts=now - 31 * 60 * 1000)                    # 31 min ago, past the 30-min window
    assert K not in host_store.tool_incapable_routes()  # aged out -> re-tested
    _obs(N, ts=now)                                     # fresh evidence -> incapable
    assert K in host_store.tool_incapable_routes()
