"""Learned per-route tool capability: a route observed to ignore tools (a
tools-request that never returns tool_calls) is marked incapable so offers_sync
stops stamping supports_tools for it. Optimistic by default; any real tool_call
proves capability; an incapable mark expires so a false positive is re-tested.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import route_tool_capability as tc  # noqa: E402

K = "antseed|glm-5.2|peerX"


def test_unobserved_route_is_capable():
    tc.reset()
    assert tc.is_capable(K) is True


def test_non_tool_requests_carry_no_signal():
    tc.reset()
    for _ in range(50):
        tc.observe(K, tools_requested=False, tool_calls_emitted=False)
    assert tc.is_capable(K) is True
    assert tc.snapshot() == {}  # nothing recorded for non-tool requests


def test_route_that_never_emits_tool_calls_is_marked_incapable():
    tc.reset()
    for _ in range(tc._MIN_SAMPLES - 1):
        tc.observe(K, True, False)
    assert tc.is_capable(K) is True  # not enough evidence yet
    tc.observe(K, True, False)       # crosses _MIN_SAMPLES with zero tool_calls
    assert tc.is_capable(K) is False


def test_any_tool_call_proves_capability_permanently():
    tc.reset()
    tc.observe(K, True, True)             # one real tool_call
    for _ in range(tc._MIN_SAMPLES * 2):
        tc.observe(K, True, False)       # then many without
    assert tc.is_capable(K) is True      # proven capable, never demoted


def test_incapable_mark_expires_and_route_is_retested(monkeypatch):
    tc.reset()
    now = [1_000_000]
    monkeypatch.setattr(tc, "_now_ms", lambda: now[0])
    for _ in range(tc._MIN_SAMPLES):
        tc.observe(K, True, False)
    assert tc.is_capable(K) is False
    now[0] += tc._EXPIRY_MS               # advance past the expiry window
    assert tc.is_capable(K) is True       # re-tested: window reset
