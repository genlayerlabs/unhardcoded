from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["CALLER_KEYS_JSON"] = '{"internal":"default"}'
os.environ["CALLER_KEYS_SHA256_JSON"] = "{}"
os.environ["DASHBOARD_TRUSTED_USER_HEADER"] = ""

import auth_proxy  # noqa: E402
import host_store  # noqa: E402


POLICY = ["policy", ["meets_req"], ["zero"], ["argmax"], ["id"],
          ["always", {"action": "next_candidate"}]]


def _dashboard_client(monkeypatch, user: str = "tester"):
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET",
                        "test-dashboard-session-secret")
    client = TestClient(auth_proxy.app)
    client.cookies.set(auth_proxy.DASHBOARD_COOKIE_NAME,
                       auth_proxy._make_dashboard_session(user))
    return client


class _FakeRankResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {}
        self.text = str(self._body)

    def json(self):
        return self._body


class _RankClient:
    def __init__(self, ranked_by_family):
        self.ranked_by_family = ranked_by_family
        self.calls = []

    async def post(self, url: str, json: dict, timeout: float):
        assert url.endswith("/x/rank")
        assert timeout == 10.0
        family = (json.get("requirements") or {}).get("model_family")
        self.calls.append(json)
        result = self.ranked_by_family(family) if callable(self.ranked_by_family) \
            else self.ranked_by_family.get(family, [])
        if isinstance(result, _FakeRankResponse):
            return result
        if isinstance(result, tuple):
            status_code, body = result
            return _FakeRankResponse(status_code, body)
        return _FakeRankResponse(200, {"rank_source": "router",
                                      "ranked": result, "rejected": []})


def _seed_call(family: str, provider: str, *, tokens_in: int = 0,
               tokens_out: int = 0, cost_usd: float = 0.0,
               latency_ms: float = 0.0, caller: str = "crm",
               ts: int | None = None) -> None:
    host_store.insert_call({
        "ts": int(time.time()) - 60 if ts is None else ts,
        "usage_event_id": f"{family}-{provider}-{tokens_in}-{tokens_out}-{time.time_ns()}",
        "caller": caller,
        "provider": provider,
        "model_family": family,
        "served_model_id": f"{provider}/{family}",
        "requested_model": f"family:{family}",
        "status": 200,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "cost_usd": cost_usd,
    })


def test_policy_backtest_requires_admin_auth():
    client = TestClient(auth_proxy.app)
    resp = client.post("/dashboard/api/policy/backtest",
                       json={"policy_ir": POLICY})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "dashboard_auth"


@pytest.mark.parametrize("payload", [
    {"flow_ir": ["flow", {}]},
    {"policy_ir": ["flow", {}]},
])
def test_policy_backtest_rejects_flow_ir(monkeypatch, payload):
    resp = _dashboard_client(monkeypatch).post(
        "/dashboard/api/policy/backtest", json=payload)
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["code"] == "flow_backtest_unsupported"
    assert "out of scope" in body["error"]["message"]


def test_policy_backtest_group_math_winner_pricing_delta_and_latency(
        monkeypatch, host_store_clean):
    _seed_call("fam-a", "old-a", tokens_in=1_000_000, tokens_out=1_000_000,
               cost_usd=5.0, latency_ms=100)
    _seed_call("fam-a", "cheap", tokens_in=2_000_000, tokens_out=0,
               cost_usd=2.0, latency_ms=200)
    _seed_call("fam-b", "old-b", tokens_in=1_000_000, tokens_out=2_000_000,
               cost_usd=10.0, latency_ms=300)
    rank_client = _RankClient({
        "fam-a": [
            {"provider": "cheap", "model_family": "fam-a",
             "served_model_id": "cheap/fam-a", "price_in": 1.0,
             "price_out": 2.0, "score": 0.9},
            {"provider": "old-a", "model_family": "fam-a",
             "served_model_id": "old-a/fam-a", "price_in": 9.0,
             "price_out": 9.0, "score": 0.1},
        ],
        # No price fields here: this family exercises the _price_table fallback.
        "fam-b": [
            {"provider": "prov-y", "model_family": "fam-b",
             "served_model_id": "prov-y/fam-b", "score": 0.8},
        ],
    })
    monkeypatch.setattr(auth_proxy, "_client", rank_client)
    monkeypatch.setattr(auth_proxy, "_price_table", lambda: {
        ("fam-b", "prov-y"): {"input": 3.0, "output": 4.0},
    })

    resp = _dashboard_client(monkeypatch).post(
        "/dashboard/api/policy/backtest",
        json={"policy_ir": POLICY, "timeframe": "7d"})

    assert resp.status_code == 200
    body = resp.json()
    assert [c["requirements"]["model_family"] for c in rank_client.calls] == [
        "fam-a", "fam-b"]
    assert body["window"]["groups_total"] == 2
    assert body["window"]["groups_shown"] == 2
    assert body["window"]["requests_total"] == 3
    assert body["window"]["requests_covered"] == 3
    assert body["totals"]["actual_cost_usd"] == pytest.approx(17.0)
    assert body["totals"]["backtest_cost_usd"] == pytest.approx(16.0)
    assert body["totals"]["delta_usd"] == pytest.approx(-1.0)
    assert body["totals"]["delta_pct"] == pytest.approx(-5.8824)

    fam_a = next(g for g in body["groups"] if g["route"] == "fam-a")
    assert fam_a["requests"] == 2
    assert fam_a["tokens_in"] == 3_000_000
    assert fam_a["tokens_out"] == 1_000_000
    assert fam_a["actual"]["providers"]["old-a"] == {
        "requests": 1, "cost_usd": 5.0}
    assert fam_a["actual"]["providers"]["cheap"] == {
        "requests": 1, "cost_usd": 2.0}
    assert fam_a["backtest"]["winner_provider"] == "cheap"
    assert fam_a["backtest"]["winner_model"] == "cheap/fam-a"
    assert fam_a["backtest"]["cost_usd"] == pytest.approx(5.0)
    assert fam_a["backtest"]["admitted"] == 2
    assert fam_a["latency"]["actual_avg_ms"] == pytest.approx(150.0)
    assert fam_a["latency"]["winner_observed_avg_ms"] == pytest.approx(200.0)

    fam_b = next(g for g in body["groups"] if g["route"] == "fam-b")
    assert fam_b["backtest"]["winner_provider"] == "prov-y"
    assert fam_b["backtest"]["cost_usd"] == pytest.approx(11.0)
    assert fam_b["latency"]["winner_observed_avg_ms"] is None
    assert any("current catalog" in c for c in body["caveats"])


def test_policy_backtest_unroutable_requests_excluded_from_cost_totals(
        monkeypatch, host_store_clean):
    _seed_call("fam-ok", "actual-ok", tokens_in=1_000_000, cost_usd=2.0,
               latency_ms=50)
    _seed_call("fam-blocked", "actual-blocked", tokens_in=1_000_000,
               cost_usd=9.0, latency_ms=60)
    monkeypatch.setattr(auth_proxy, "_client", _RankClient({
        "fam-ok": [{"provider": "winner", "model_family": "fam-ok",
                    "served_model_id": "winner/fam-ok", "price_in": 1.0,
                    "price_out": 1.0}],
        "fam-blocked": [],
    }))
    monkeypatch.setattr(auth_proxy, "_price_table", lambda: {})

    resp = _dashboard_client(monkeypatch).post(
        "/dashboard/api/policy/backtest",
        json={"policy_ir": POLICY, "timeframe": "7d"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["window"]["requests_total"] == 2
    assert body["totals"]["actual_cost_usd"] == pytest.approx(2.0)
    assert body["totals"]["backtest_cost_usd"] == pytest.approx(1.0)
    assert body["totals"]["unroutable_requests"] == 1
    blocked = next(g for g in body["groups"] if g["route"] == "fam-blocked")
    assert blocked["unroutable"] is True
    assert blocked["backtest"]["admitted"] == 0
    assert blocked["backtest"]["cost_usd"] is None


def test_policy_backtest_partial_rank_failure_is_flagged_and_excluded(
        monkeypatch, host_store_clean):
    _seed_call("fam-fail", "actual-fail-a", tokens_in=1_000_000,
               cost_usd=11.0, latency_ms=70)
    _seed_call("fam-fail", "actual-fail-b", tokens_in=1_000_000,
               cost_usd=7.0, latency_ms=90)
    _seed_call("fam-ok", "actual-ok", tokens_in=1_000_000,
               cost_usd=8.0, latency_ms=50)
    rank_client = _RankClient({
        "fam-fail": (500, {"error": {"message": "router blew up",
                                     "type": "router_error",
                                     "code": "rank"}}),
        "fam-ok": [{"provider": "winner", "model_family": "fam-ok",
                    "served_model_id": "winner/fam-ok", "price_in": 2.0,
                    "price_out": 0.0}],
    })
    monkeypatch.setattr(auth_proxy, "_client", rank_client)
    monkeypatch.setattr(auth_proxy, "_price_table", lambda: {})

    resp = _dashboard_client(monkeypatch).post(
        "/dashboard/api/policy/backtest",
        json={"policy_ir": POLICY, "timeframe": "7d"})

    assert resp.status_code == 200
    body = resp.json()
    assert [c["requirements"]["model_family"] for c in rank_client.calls] == [
        "fam-fail", "fam-ok"]
    assert body["totals"]["actual_cost_usd"] == pytest.approx(8.0)
    assert body["totals"]["backtest_cost_usd"] == pytest.approx(2.0)
    assert body["totals"]["delta_usd"] == pytest.approx(-6.0)
    assert body["totals"]["unroutable_requests"] == 0
    assert body["totals"]["rank_error_requests"] == 2

    failed = next(g for g in body["groups"] if g["route"] == "fam-fail")
    assert failed["rank_error"] is True
    assert failed["rank_error_status"] == 500
    assert failed["rank_error_kind"] == "router_error"
    assert failed["requests"] == 2
    assert failed["actual"]["cost_usd"] == pytest.approx(18.0)
    assert failed["backtest"]["cost_usd"] is None
    assert failed["backtest"]["admitted"] == 0
    assert failed["latency"]["winner_observed_avg_ms"] is None

    ok = next(g for g in body["groups"] if g["route"] == "fam-ok")
    assert ok.get("rank_error") is not True
    assert ok["backtest"]["winner_provider"] == "winner"
    assert any("1 family failed to rank" in c for c in body["caveats"])


def test_policy_backtest_all_rank_failures_propagates_first_error(
        monkeypatch, host_store_clean):
    _seed_call("fam-a", "actual-a", cost_usd=1.0)
    _seed_call("fam-b", "actual-b", cost_usd=2.0)
    rank_client = _RankClient({
        "fam-a": (503, {"error": {"message": "first failure",
                                  "type": "router_error",
                                  "code": "first"}}),
        "fam-b": (500, {"error": {"message": "second failure",
                                  "type": "router_error",
                                  "code": "second"}}),
    })
    monkeypatch.setattr(auth_proxy, "_client", rank_client)
    monkeypatch.setattr(auth_proxy, "_price_table", lambda: {})

    resp = _dashboard_client(monkeypatch).post(
        "/dashboard/api/policy/backtest",
        json={"policy_ir": POLICY, "timeframe": "7d"})

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "first"
    assert [c["requirements"]["model_family"] for c in rank_client.calls] == [
        "fam-a", "fam-b"]


def test_policy_backtest_truncates_to_top_50_groups(monkeypatch,
                                                    host_store_clean):
    for i in range(51):
        _seed_call(f"fam-{i:02d}", "actual", cost_usd=0.0)
    rank_client = _RankClient(lambda family: [
        {"provider": "winner", "model_family": family,
         "served_model_id": f"winner/{family}", "price_in": 0.0,
         "price_out": 0.0}
    ])
    monkeypatch.setattr(auth_proxy, "_client", rank_client)
    monkeypatch.setattr(auth_proxy, "_price_table", lambda: {})

    resp = _dashboard_client(monkeypatch).post(
        "/dashboard/api/policy/backtest",
        json={"policy_ir": POLICY, "timeframe": "7d"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["window"]["groups_total"] == 51
    assert body["window"]["groups_shown"] == 50
    assert body["window"]["groups_truncated"] == 1
    assert body["window"]["requests_total"] == 51
    assert body["window"]["requests_covered"] == 50
    assert body["window"]["requests_truncated"] == 1
    assert len(body["groups"]) == 50
    assert len(rank_client.calls) == 50


def test_policy_backtest_empty_window_zero_shape(monkeypatch, host_store_clean):
    rank_client = _RankClient({})
    monkeypatch.setattr(auth_proxy, "_client", rank_client)
    monkeypatch.setattr(auth_proxy, "_price_table", lambda: {})

    resp = _dashboard_client(monkeypatch).post(
        "/dashboard/api/policy/backtest",
        json={"policy_ir": POLICY, "timeframe": "24h"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["window"]["timeframe"] == "24h"
    assert body["window"]["groups_total"] == 0
    assert body["window"]["groups_shown"] == 0
    assert body["window"]["requests_total"] == 0
    assert body["window"]["requests_covered"] == 0
    assert body["totals"] == {
        "actual_cost_usd": 0.0,
        "backtest_cost_usd": 0.0,
        "delta_usd": 0.0,
        "delta_pct": None,
        "unroutable_requests": 0,
        "rank_error_requests": 0,
    }
    assert body["groups"] == []
    assert rank_client.calls == []
