"""Host-side per-route effective-cost fold (twin of route_latency): an EMA of the
provider-reported cost a route actually charged, in USD per million tokens,
folded only on SUCCESS with a reported cost. Measurement + observability only —
it does not (yet) change ranking."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import route_economics as re  # noqa: E402

# A peer advertising cheap but settling dear; $1.00 over 100k tokens = $10/Mtok.
K = re.route_key("antseed", "glm-5.2", "peerDear")


def test_unobserved_route_is_none():
    re.reset()
    assert re.usd_per_mtok(K) is None


def test_first_observation_seeds_directly():
    re.reset()
    re.observe(K, cost_usd=1.0, tokens=100_000, ok=True)   # $10 / Mtok
    assert re.usd_per_mtok(K) == 10.0


def test_ema_decays_toward_new_samples():
    re.reset()
    re.observe(K, cost_usd=1.0, tokens=100_000, ok=True)   # seed: $10/Mtok
    re.observe(K, cost_usd=0.2, tokens=100_000, ok=True)   # a cheaper call: $2/Mtok
    v = re.usd_per_mtok(K)
    assert 2.0 < v < 10.0
    assert v == round(0.3 * 2.0 + 0.7 * 10.0, 4)


def test_failures_do_not_fold():
    # A failed call carries no honest cost — folding it would poison the rate.
    re.reset()
    re.observe(K, cost_usd=99.0, tokens=100_000, ok=False)
    assert re.usd_per_mtok(K) is None
    re.observe(K, cost_usd=1.0, tokens=100_000, ok=True)   # only the success counts
    re.observe(K, cost_usd=99.0, tokens=100_000, ok=False)
    assert re.usd_per_mtok(K) == 10.0


def test_absent_cost_does_not_fold_but_zero_does():
    # No reported cost (None, e.g. native OpenAI) -> unmeasured, catalog stands.
    re.reset()
    re.observe(K, cost_usd=None, tokens=100_000, ok=True)
    assert re.usd_per_mtok(K) is None
    # A genuinely free route (reported 0) folds as 0.0, distinct from unmeasured.
    re.observe(K, cost_usd=0.0, tokens=100_000, ok=True)
    assert re.usd_per_mtok(K) == 0.0


def test_nonpositive_tokens_or_negative_cost_ignored():
    re.reset()
    re.observe(K, cost_usd=1.0, tokens=0, ok=True)
    re.observe(K, cost_usd=1.0, tokens=None, ok=True)
    re.observe(K, cost_usd=-1.0, tokens=100_000, ok=True)   # negative-price sentinel
    assert re.usd_per_mtok(K) is None


def test_routes_are_independent():
    re.reset()
    cheap = re.route_key("openrouter", "gpt-5.5", "openrouter")
    re.observe(K, cost_usd=1.0, tokens=100_000, ok=True)       # $10/Mtok peer
    re.observe(cheap, cost_usd=0.05, tokens=100_000, ok=True)  # $0.5/Mtok partner
    assert re.usd_per_mtok(K) == 10.0
    assert re.usd_per_mtok(cheap) == 0.5


def test_snapshot_and_reset():
    re.reset()
    re.observe(K, cost_usd=1.0, tokens=100_000, ok=True)
    assert re.snapshot() == {K: 10.0}
    re.reset()
    assert re.snapshot() == {}
