"""The dashboard stats SQL rewrite must be a pure performance change.

The persistent-timeframe stats used to load EVERY retained `calls` row and fold
in Python (auth_proxy._aggregate_usage_rows / _period_totals). The stats path now
aggregates in SQL (host_store.usage_aggregate, one GROUPING SETS scan) behind a
short TTL cache, with unpriced rows cost-stamped once by an idempotent backfill.
These tests pin the equivalence: the SQL path must produce EXACTLY what the
reference Python aggregation produces over the same rows — totals, every group-by,
the daily buckets, the recent page, filters, and the empty shapes.
"""
from __future__ import annotations

import hashlib
import os
import random
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["CALLER_KEYS_JSON"] = '{"internal":"default"}'
os.environ["CALLER_KEYS_SHA256_JSON"] = "{}"
os.environ["DASHBOARD_TRUSTED_USER_HEADER"] = ""

import auth_proxy  # noqa: E402
import host_store  # noqa: E402

BASE_TS = 1_700_000_000  # 2023-11-14T22:13:20Z — rows span several UTC days

PRICES_LUA = """
return {
  ["fam-a@prov-x"] = { price_in_usd_per_mtok = 2.0, price_out_usd_per_mtok = 10.0 },
  ["fam-b@prov-y"] = { price_in_usd_per_mtok = 0.5, price_out_usd_per_mtok = 1.5 },
}
"""

SHA = {c: hashlib.sha256(f"key-{c}".encode()).hexdigest() for c in ("acme", "crm", "wing")}


def _use_prices(monkeypatch, tmp_path) -> None:
    path = tmp_path / "metrics.live.lua"
    path.write_text(PRICES_LUA)
    monkeypatch.setattr(auth_proxy, "DASHBOARD_POLICY_METRICS_PATH", str(path))


def _seed(n: int = 300) -> None:
    """Synthetic calls spanning multiple days/providers/consumers/statuses, with
    a mix of stamped, negative-stamped, priceable-NULL and unpriceable-NULL
    costs, plus NULL/0 tokens_total and missing group keys."""
    rng = random.Random(42)
    for i in range(n):
        prov, fam = rng.choice([("prov-x", "fam-a"), ("prov-y", "fam-b"),
                                ("prov-z", "fam-c"), (None, None), ("", "")])
        caller = rng.choice(["acme", "crm", "wing", None])
        kind = rng.choice(["stamped", "stamped", "neg", "none", "none", "none"])
        cost = {"stamped": round(rng.uniform(0.0, 0.01), 6),
                "neg": -1.0, "none": None}[kind]
        tin, tout = rng.randrange(0, 5000), rng.randrange(0, 3000)
        host_store.insert_call({
            "ts": BASE_TS + i * 3517,  # ~12 days of spread
            "usage_event_id": f"ev-{i}",
            "caller": caller,
            "provider": prov,
            "model_family": fam,
            "served_model_id": f"{fam}-served" if fam else None,
            "requested_model": rng.choice(["profile:edge", "profile:medium", None]),
            "status": rng.choice([200, 200, 200, 200, 400, 500, 429]),
            "latency_ms": rng.uniform(1, 250),
            "tokens_in": tin, "tokens_out": tout,
            "tokens_total": rng.choice([tin + tout, 0, None]),
            "key_sha256": SHA[caller] if caller else None,
            "cost_usd": cost,
            "cost_basis": "reported" if kind == "stamped" else None,
        })


def _expected(all_rows, *, selected=None, key_filter=None, provider=None, model=None):
    """The OLD _stats_snapshot row pipeline, verbatim: caller/key filters, then
    provider/model on the aggregated set only; table_rows never narrowed by
    provider/model."""
    rows = sorted(all_rows, key=lambda r: int(r.get("ts") or 0), reverse=True)
    if key_filter:
        frows = [r for r in rows if r.get("caller") == selected and r.get("key_sha256") == key_filter]
        table_rows = frows
    else:
        frows = [r for r in rows if r.get("caller") == selected] if selected else rows
        table_rows = rows
    if provider:
        frows = [r for r in frows if r.get("provider") == provider]
    if model:
        frows = [r for r in frows if r.get("model_family") == model]
    agg = auth_proxy._aggregate_usage_rows(frows, selected=selected)
    table_agg = auth_proxy._aggregate_usage_rows(table_rows)
    return {
        "agg": agg,
        "keys_by_caller": table_agg["by_caller_all"],
        "filter_options": {"providers": sorted(table_agg["by_provider"].keys()),
                           "models": sorted(table_agg["by_model_family"].keys())},
        "daily_totals": auth_proxy._period_totals(frows, monthly=False),
        "history_events": len(frows),
        "history_events_all": len(rows),
    }


def _assert_counters_equal(got, exp, label=""):
    assert set(got) == set(exp), (label, set(got) ^ set(exp))
    for k, v in exp.items():
        if k in ("cost_usd", "latency_ms_avg", "latency_ms_max", "error_rate"):
            assert got[k] == pytest.approx(v, abs=1e-9), (label, k, got[k], v)
        else:
            assert got[k] == v, (label, k, got[k], v)


def _assert_bundle_equal(got, exp):
    _assert_counters_equal(got["agg"]["totals"], exp["agg"]["totals"], "totals")
    for bucket in ("by_caller", "by_caller_all", "by_provider", "by_model_family",
                   "by_route", "by_served_model"):
        g, e = got["agg"][bucket], exp["agg"][bucket]
        assert list(g.keys()) == list(e.keys()), (bucket, list(g), list(e))
        for key in e:
            _assert_counters_equal(g[key], e[key], f"{bucket}[{key}]")
    assert got["agg"]["by_status"] == exp["agg"]["by_status"]
    # recent: same events, same order, same fields — except cost_usd, which the
    # backfill intentionally stamps onto previously-unpriced rows.
    g_recent, e_recent = got["agg"]["recent"], exp["agg"]["recent"]
    assert [r.get("usage_event_id") for r in g_recent] == [r.get("usage_event_id") for r in e_recent]
    for g, e in zip(g_recent, e_recent):
        assert set(g) == set(e)
        for k in e:
            if k != "cost_usd":
                assert g[k] == e[k], (k, g[k], e[k])
        assert "key_sha256" not in g
    assert [d["date"] for d in got["daily_totals"]] == [d["date"] for d in exp["daily_totals"]]
    for g, e in zip(got["daily_totals"], exp["daily_totals"]):
        _assert_counters_equal(g, e, f"daily[{e['date']}]")
    assert list(got["keys_by_caller"].keys()) == list(exp["keys_by_caller"].keys())
    for key in exp["keys_by_caller"]:
        _assert_counters_equal(got["keys_by_caller"][key], exp["keys_by_caller"][key], f"keys[{key}]")
    assert got["filter_options"] == exp["filter_options"]
    assert got["history_events"] == exp["history_events"]
    assert got["history_events_all"] == exp["history_events_all"]


def test_sql_stats_bundle_matches_python_aggregation(host_store_clean, monkeypatch, tmp_path):
    _use_prices(monkeypatch, tmp_path)
    _seed()
    # Reference: the retired Python pipeline over the PRE-backfill rows, pricing
    # unstamped rows at read time from the price table.
    rows = auth_proxy._read_usage_history()
    assert len(rows) == 300
    stamped = host_store.backfill_call_costs(auth_proxy._price_table())
    assert stamped > 0
    mid_ts = BASE_TS + 150 * 3517
    cases = [
        {},                                                          # all history
        {"since": mid_ts},                                           # timeframe floor
        {"selected": "acme"},                                        # consumer filter
        {"selected": "acme", "provider": "prov-x", "model": "fam-a"},  # analytics filters
        {"provider": "prov-x"},                                      # provider only, all consumers
        {"selected": "crm", "key_filter": SHA["crm"]},               # per-key dashboard login
    ]
    for case in cases:
        since = case.pop("since", None)
        window = [r for r in rows if since is None or int(r["ts"]) >= since]
        exp = _expected(window, selected=case.get("selected"),
                        key_filter=case.get("key_filter"),
                        provider=case.get("provider"), model=case.get("model"))
        got = auth_proxy._stats_history_bundle(
            since=since, selected=case.get("selected"),
            key_filter=case.get("key_filter"),
            provider=case.get("provider"), model=case.get("model"))
        _assert_bundle_equal(got, exp)


def test_sql_stats_bundle_empty_db_matches_python_empty_shapes(host_store_clean):
    exp = _expected([])
    got = auth_proxy._stats_history_bundle(since=None, selected=None,
                                           key_filter=None, provider=None, model=None)
    _assert_bundle_equal(got, exp)
    assert got["agg"]["totals"] == {"requests": 0, "rejects": 0, "errors": 0,
                                    "tokens_in": 0, "tokens_out": 0,
                                    "tokens_total": 0, "cost_usd": 0.0}
    assert got["daily_totals"] == []
    assert got["agg"]["recent"] == []


def test_provider_and_connection_rollups_match_python_fold(host_store_clean, monkeypatch, tmp_path):
    _use_prices(monkeypatch, tmp_path)
    _seed(120)
    rows = auth_proxy._read_usage_history()
    host_store.backfill_call_costs(auth_proxy._price_table())
    # provider rollup — replicate the old _provider_credentials_snapshot fold
    prices = auth_proxy._price_table()
    exp_counters: dict = {}
    exp_last: dict = {}
    for row in rows:  # ascending ts, ties resolved by later row — as before
        provider = str(row.get("provider") or "unknown")
        c = exp_counters.setdefault(provider, auth_proxy._counter())
        cost, _ = auth_proxy._cost_for_event(row, prices)
        auth_proxy._add_counter(c, row, cost)
        if provider not in exp_last or int(row.get("ts") or 0) >= int(exp_last[provider].get("ts") or 0):
            exp_last[provider] = row
    got = host_store.usage_provider_stats()
    assert set(got) == set(exp_counters)
    for name, c in exp_counters.items():
        _assert_counters_equal(auth_proxy._counter_snapshot(auth_proxy._counter_from_sql(got[name])),
                               auth_proxy._counter_snapshot(c), f"provider[{name}]")
        assert got[name]["last_ts"] == exp_last[name]["ts"]
        assert got[name]["last_status"] == exp_last[name]["status"]
        assert got[name]["last_route"] == exp_last[name]["requested_model"]
        assert got[name]["last_model_family"] == exp_last[name]["model_family"]
    # connections rollup — replicate the old _login_connections_snapshot fold
    exp_grouped: dict = {}
    for row in rows:
        caller = str(row.get("caller") or "unknown")
        prefix = str(row.get("key_sha256_prefix") or "").strip() or "unknown"
        item = exp_grouped.setdefault((caller, prefix), {
            "requests": 0, "errors": 0, "first_seen": None, "last_seen": None,
            "last_status": None, "last_route": None, "last_provider": None})
        ts = int(row.get("ts") or 0)
        item["requests"] += 1
        if int(row.get("status") or 0) >= 400:
            item["errors"] += 1
        item["first_seen"] = ts if not item["first_seen"] else min(item["first_seen"], ts)
        if not item["last_seen"] or ts >= item["last_seen"]:
            item.update({"last_seen": ts, "last_status": row.get("status"),
                         "last_route": row.get("requested_model"),
                         "last_provider": row.get("provider")})
    got_conns = {(c["caller"], c["prefix"]): c for c in host_store.usage_connections()}
    assert set(got_conns) == set(exp_grouped)
    for key, e in exp_grouped.items():
        g = got_conns[key]
        for field in e:
            assert g[field] == e[field], (key, field, g[field], e[field])


def test_cost_backfill_is_idempotent_and_scoped_to_null_rows(host_store_clean):
    prices = {("fam-a", "prov-x"): {"input": 2.0, "output": 10.0}}
    mk = lambda i, **kw: host_store.insert_call({  # noqa: E731
        "ts": BASE_TS + i, "usage_event_id": f"bf-{i}", "caller": "acme",
        "provider": kw.get("provider", "prov-x"), "model_family": kw.get("family", "fam-a"),
        "status": 200, "tokens_in": 1_000_000, "tokens_out": 100_000,
        "tokens_total": 1_100_000, "key_sha256": SHA["acme"],
        "cost_usd": kw.get("cost"), "cost_basis": kw.get("basis")})
    mk(0)                                        # NULL + priceable
    mk(1)                                        # NULL + priceable
    mk(2, provider="prov-z", family="fam-c")     # NULL + no price entry
    mk(3, cost=0.5, basis="reported")            # already stamped — untouched
    assert host_store.backfill_call_costs(prices) == 2
    assert host_store.backfill_call_costs(prices) == 0  # idempotent
    with host_store._get_pool().connection() as conn:
        rows = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT usage_event_id, cost_usd, cost_basis FROM calls").fetchall()}
    assert rows["bf-0"] == (3.0, "computed")     # $2/M in + $10/M out
    assert rows["bf-1"] == (3.0, "computed")
    assert rows["bf-2"] == (None, None)          # unpriceable stays NULL (sums as 0)
    assert rows["bf-3"] == (0.5, "reported")     # provider-reported stamp preserved


def test_stats_snapshot_ttl_cache_reuses_window_aggregation(host_store_clean, monkeypatch, tmp_path):
    _use_prices(monkeypatch, tmp_path)
    _seed(20)
    calls = {"n": 0}
    for name in ("usage_aggregate", "usage_rows_page", "usage_count",
                 "usage_provider_stats", "usage_connections"):
        real = getattr(host_store, name)

        def counting(*a, __real=real, **kw):
            calls["n"] += 1
            return __real(*a, **kw)

        monkeypatch.setattr(host_store, name, counting)
    monkeypatch.setattr(auth_proxy, "_SNAPSHOT_TTL_S", 60.0)
    auth_proxy._reset_stats_for_tests()  # clears the snapshot cache
    snap1 = auth_proxy._stats_snapshot(viewer="t", upstream_status=200,
                                       upstream_health={"ok": True}, timeframe="24h")
    first = calls["n"]
    assert first >= 1
    snap2 = auth_proxy._stats_snapshot(viewer="t", upstream_status=200,
                                       upstream_health={"ok": True}, timeframe="24h")
    assert calls["n"] == first, "second call within the TTL must not re-aggregate"
    assert snap2["totals"] == snap1["totals"]
    assert snap2["daily_totals"] == snap1["daily_totals"]
    # a different query tuple is a different cache entry — it must hit the store
    auth_proxy._stats_snapshot(viewer="t", upstream_status=200,
                               upstream_health={"ok": True}, timeframe="7d")
    assert calls["n"] > first
    auth_proxy._reset_stats_for_tests()


def test_recent_dashboard_is_bounded_and_single_flight(host_store_clean, monkeypatch, tmp_path):
    _use_prices(monkeypatch, tmp_path)
    _seed(150)
    real = host_store.usage_rows_page
    calls = {"n": 0}

    def slow_page(*args, **kwargs):
        calls["n"] += 1
        time.sleep(0.05)
        return real(*args, **kwargs)

    monkeypatch.setattr(host_store, "usage_rows_page", slow_page)
    monkeypatch.setattr(auth_proxy, "_provider_credentials_snapshot", lambda **kwargs: {"rows": []})
    monkeypatch.setattr(auth_proxy, "_login_connections_snapshot", lambda **kwargs: {"rows": []})
    monkeypatch.setattr(auth_proxy, "_SNAPSHOT_TTL_S", 60.0)
    auth_proxy._reset_stats_for_tests()
    results = []

    def load():
        results.append(auth_proxy._stats_snapshot(
            viewer="t", upstream_status=200, upstream_health={"ok": True}))

    threads = [threading.Thread(target=load) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls["n"] == 1
    assert len(results) == 4
    assert all(result["timeframe"]["selected"] == "recent" for result in results)
    assert all(len(result["recent"]) <= 100 for result in results)
    assert all(result["totals"]["requests"] == 100 for result in results)
    auth_proxy._reset_stats_for_tests()
