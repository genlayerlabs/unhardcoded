"""
Codex scarcity ramp: codex is imputed $0 when healthy (so it wins), but its
ranking price ramps up as the subscription is strained — from the quota header
when present, and from recently observed 429s when it is not — and decays back
as the 429s age out of the window.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sources.codex as cx  # noqa: E402
from sources.codex import CodexSource  # noqa: E402


class FakeHost:
    def __init__(self):
        self.metrics: dict = {}

    def update_metrics(self, provider, family, delta):
        self.metrics[(provider, family)] = delta


def _src(families=("gpt-5.5",)):
    h = FakeHost()
    s = CodexSource("openai")
    s.bind(h, list(families))
    return s, h


def _price_in(h):
    return h.metrics[("openai", "gpt-5.5")]["price_in"]


def test_healthy_codex_priced_zero():
    s, h = _src()
    s.ingest("openai", {"status": 200, "headers": {}, "ts": int(time.time())})
    assert _price_in(h) == 0.0  # free → wins cost-led ranking


def test_recent_429s_ramp_to_full_demote():
    s, h = _src()
    now = int(time.time())
    for _ in range(3):
        s.ingest("openai", {"status": 429, "headers": {}, "ts": now})
    assert _price_in(h) == 5.0  # SHED recent 429s → frac 1


def test_old_429s_age_out_of_window():
    s, h = _src()
    old = int(time.time()) - 120 - 10
    for _ in range(5):
        s.ingest("openai", {"status": 429, "headers": {}, "ts": old})
    assert _price_in(h) == 0.0  # outside the window → not counted → recovered


def test_quota_header_drives_ramp():
    s, h = _src()
    s.ingest("openai", {"status": 200,
                        "headers": {"x-codex-primary-used-percent": "75"},
                        "ts": int(time.time())})
    # used 0.75 with START 0.5 → frac (0.75-0.5)/0.5 = 0.5
    assert _price_in(h) == pytest.approx(5.0 * 0.5)


@pytest.mark.asyncio
async def test_pricing_returns_scarcity_for_each_family():
    s, _ = _src(("gpt-5.5", "gpt-5.3-codex-spark"))
    now = int(time.time())
    for _ in range(3):
        s.ingest("openai", {"status": 429, "headers": {}, "ts": now})
    prices = await s.pricing()
    fams = {p["model_family"]: p for p in prices}
    assert set(fams) == {"gpt-5.5", "gpt-5.3-codex-spark"}
    assert fams["gpt-5.5"]["price_in_usd_per_mtok"] == 5.0
