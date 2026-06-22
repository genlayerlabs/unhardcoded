"""
Codex observability: the dashboard's codex panel surfaces request/error counts
(ingress stats) + live quota and scarcity-price state (router /x/runtime), so
the codex provider shows what the others' rows show.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CALLER_KEYS_JSON", '{"internal":"default"}')
os.environ.setdefault("CALLER_KEYS_SHA256_JSON", "{}")
os.environ.setdefault("DASHBOARD_TRUSTED_USER_HEADER", "")

import auth_proxy  # noqa: E402


def test_codex_activity_assembles_counts_and_quota(monkeypatch):
    auth_proxy._reset_stats_for_tests()
    with auth_proxy._stats_lock:
        c = auth_proxy._stats["by_provider"]["openai"]
        c["requests"] = 10
        c["errors"] = 2
        c["tokens_total"] = 1234

    async def fake_rt():
        return {"balances": {"openai": {"kind": "quota_window", "value": 0.4,
                "detail": {"recent_429_count": 3, "events": 12}}},
                "ema_metrics": {"openai|gpt-5.5": {"price_in": 2.5}}}

    monkeypatch.setattr(auth_proxy, "_fetch_router_runtime", fake_rt)
    monkeypatch.setattr(auth_proxy, "_codex_provider_id", lambda: "openai")

    a = asyncio.run(auth_proxy._codex_activity())
    assert a["provider"] == "openai"
    assert a["requests"] == 10 and a["errors"] == 2
    assert a["error_rate"] == 0.2 and a["tokens_total"] == 1234
    assert a["used_percent"] == 40.0          # 0.4 fraction → percent
    assert a["recent_429"] == 3 and a["events"] == 12
    assert a["scarcity_price_in"] == 2.5
    auth_proxy._reset_stats_for_tests()


def test_codex_activity_survives_missing_runtime(monkeypatch):
    auth_proxy._reset_stats_for_tests()

    async def no_rt():
        return None

    monkeypatch.setattr(auth_proxy, "_fetch_router_runtime", no_rt)
    monkeypatch.setattr(auth_proxy, "_codex_provider_id", lambda: "openai")
    a = asyncio.run(auth_proxy._codex_activity())
    assert a["requests"] == 0 and a["used_percent"] is None
    assert a["scarcity_price_in"] is None
