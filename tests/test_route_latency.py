"""Host-side per-route latency fold (twin of route_reliability): an EMA of
observed end-to-end call latency, folded only on SUCCESS, stamped onto offers as
offer.latency_ms so a policy can route by speed."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import route_latency as rl  # noqa: E402

K = rl.route_key("antseed", "glm-5.2", "peerSlow")


def test_unobserved_route_is_none():
    rl.reset()
    assert rl.latency_ms(K) is None


def test_first_observation_seeds_directly():
    rl.reset()
    rl.observe(K, 12000, ok=True)
    assert rl.latency_ms(K) == 12000


def test_ema_decays_toward_new_samples():
    rl.reset()
    rl.observe(K, 12000, ok=True)          # seed
    rl.observe(K, 2000, ok=True)           # a fast call pulls it down
    v = rl.latency_ms(K)
    assert 2000 < v < 12000                 # between, nearer the slow seed (alpha 0.3)
    assert v == round(0.3 * 2000 + 0.7 * 12000)


def test_failures_do_not_fold():
    # A fast failure (empty content in 1.9s) must NOT make a broken route look
    # fast — that is reliability's job, not latency's.
    rl.reset()
    rl.observe(K, 1900, ok=False)
    assert rl.latency_ms(K) is None
    rl.observe(K, 12000, ok=True)          # only the honest success counts
    rl.observe(K, 1900, ok=False)          # later failure ignored
    assert rl.latency_ms(K) == 12000


def test_nonpositive_or_missing_latency_ignored():
    rl.reset()
    rl.observe(K, None, ok=True)
    rl.observe(K, 0, ok=True)
    rl.observe(K, -5, ok=True)
    assert rl.latency_ms(K) is None


def test_routes_are_independent():
    rl.reset()
    fast = rl.route_key("openai", "gpt-5.5", "openai")
    rl.observe(K, 12000, ok=True)
    rl.observe(fast, 900, ok=True)
    assert rl.latency_ms(K) == 12000
    assert rl.latency_ms(fast) == 900       # a fast partner stays distinct from a slow peer


def test_snapshot_and_reset():
    rl.reset()
    rl.observe(K, 12000, ok=True)
    assert rl.snapshot() == {K: 12000}
    rl.reset()
    assert rl.snapshot() == {}
