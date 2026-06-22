"""
Integration tests for the FastAPI shim.

These boot a real LLMRouterHost backed by mock provider responses (set via
set_mock_response). The router runs end-to-end inside lupa; only the
outbound HTTP to the upstream provider is mocked.

Run from repo root:
    pytest tests -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router_host import LLMRouterHost  # noqa: E402

from shim import create_app  # noqa: E402


def _ok_response(text: str = "hi back") -> dict:
    return {
        "ok": True,
        "latency_ms": 10,
        "response": {
            "text":          text,
            "tool_calls":    None,
            "finish_reason": "stop",
            "tokens_in":     7,
            "tokens_out":    3,
            "tokens_total":  10,
            "raw_model":     "mock-model-id",
        },
    }


def _err_response(kind: str = "server_error", status: int = 500) -> dict:
    return {
        "ok":            False,
        "error_kind":    kind,
        "http_status":   status,
        "latency_ms":    5,
        "error_message": f"mock {kind}",
    }


@pytest.fixture
def host():
    h = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "core" / "config.example.lua",
        metrics_path=ROOT / "core" / "metrics.example.lua",
        now_ms=lambda: 1_000_000,
    )
    h.init()
    return h


@pytest.fixture
def client(host):
    app = create_app(host, default_profile="default")
    return TestClient(app)


# ---- liveness / introspection ------------------------------------------

def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "initialized": True}


# ---- wallet control endpoints ------------------------------------------

def test_wallet_deposit_503_without_control(client, monkeypatch):
    monkeypatch.delenv("ANTSEED_CONTROL_URL", raising=False)
    monkeypatch.delenv("ANTSEED_CONTROL_TOKEN", raising=False)
    r = client.post("/x/wallet/deposit", json={"amount": "5"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "wallet_control_unavailable"


def test_wallet_deposit_rejects_bad_amount(client, monkeypatch):
    # control configured so we reach (and fail at) amount validation, no network
    monkeypatch.setenv("ANTSEED_CONTROL_URL", "http://antseed:8379")
    monkeypatch.setenv("ANTSEED_CONTROL_TOKEN", "t")
    for bad in ["0", "-1", "abc", "1.1234567", "", "1e3"]:
        r = client.post("/x/wallet/deposit", json={"amount": bad})
        assert r.status_code == 400, f"{bad!r} should be rejected"
        assert r.json()["error"]["code"] == "wallet_amount"


def test_list_models_exposes_profiles_and_families(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert "profile:default" in ids
    assert "profile:cheap_explore" in ids
    assert any(i.startswith("family:") for i in ids)


# ---- model field convention --------------------------------------------

def test_empty_model_uses_default_profile(client, host):
    # Set a mock for every (provider, family) in the catalog so SOME candidate
    # succeeds regardless of which one default picks.
    for prov, fam in _all_pairs(host):
        host.set_mock_response(prov, fam, _ok_response("ok"))
    r = client.post("/v1/chat/completions", json={"model": "", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "ok"
    assert body["x_router"]["provider"] is not None


def test_profile_prefix_routes_to_that_profile(client, host):
    for prov, fam in _all_pairs(host):
        host.set_mock_response(prov, fam, _ok_response("cheap"))
    r = client.post("/v1/chat/completions", json={
        "model": "profile:cheap_explore",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "cheap"


def test_family_prefix_filters_to_family(client, host):
    # Only mock the deepseek family — if shim correctly constrains to that
    # family, requests succeed; if it ignores the filter and picks something
    # else, they fail.
    for prov, fam in _all_pairs(host):
        if fam == "deepseek-v3":
            host.set_mock_response(prov, fam, _ok_response("deepseek"))
    r = client.post("/v1/chat/completions", json={
        "model": "family:deepseek-v3",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert r.json()["x_router"]["model_family"] == "deepseek-v3"


def test_pin_prefix_short_circuits_to_single_pair(client, host):
    host.set_mock_response("comput3", "hermes-3-405b", _ok_response("pinned"))
    r = client.post("/v1/chat/completions", json={
        "model": "pin:comput3/hermes-3-405b",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["x_router"]["provider"] == "comput3"
    assert body["x_router"]["model_family"] == "hermes-3-405b"


def test_unknown_model_string_falls_back_to_default(client, host):
    for prov, fam in _all_pairs(host):
        host.set_mock_response(prov, fam, _ok_response("fallback"))
    r = client.post("/v1/chat/completions", json={
        "model": "totally-made-up-model-name",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "fallback"


# ---- response shape ----------------------------------------------------

def test_response_shape_is_openai_compatible(client, host):
    for prov, fam in _all_pairs(host):
        host.set_mock_response(prov, fam, _ok_response("hi"))
    r = client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}],
    })
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert isinstance(body["created"], int)
    assert body["choices"][0]["index"] == 0
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    # x_router metadata is non-standard but useful
    assert body["x_router"]["served_model_id"] is not None


# ---- failure / fallback paths ------------------------------------------

def test_stream_true_pseudo_streams_mocked_result(client, host):
    # Mocked backends complete without a commit point; the shim must still
    # deliver a valid SSE stream (role chunk, text, final, DONE).
    for prov, fam in _all_pairs(host):
        host.set_mock_response(prov, fam, _ok_response("streamed!"))
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}], "stream": True,
    }) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(chunk for chunk in r.iter_text())
    assert '"role": "assistant"' in body
    assert "streamed!" in body
    assert '"finish_reason"' in body
    assert body.rstrip().endswith("data: [DONE]")


def test_stream_true_all_fail_returns_json_error(client, host):
    # Every candidate fails pre-delta: client gets the normal JSON error
    # (with traces), NOT a 200 SSE stream.
    for prov, fam in _all_pairs(host):
        host.set_mock_response(prov, fam, _err_response("server_error", 500))
    r = client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}], "stream": True,
    })
    assert r.status_code >= 500
    err = r.json()["error"]
    assert "exhausted" in err["code"]
    assert r.json()["x_router"]["decision_trace"]


def test_all_candidates_fail_returns_5xx(client, host):
    # No mocks set => _default_mock_call returns no_mock_set for every call.
    # Router exhausts candidates → exhausted: <last_error_kind>.
    r = client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code >= 500
    err = r.json()["error"]
    assert err["type"] == "router_error"
    assert "exhausted" in err["code"] or "no_candidates" in err["code"]


def test_bad_request_abort_maps_to_400(client, host):
    # bad_request aborts in the router and returns the bare kind (not
    # "exhausted: ..."). The shim must still map it to 400, not 502.
    host.set_mock_response("comput3", "hermes-3-405b", _err_response("bad_request", 400))
    r = client.post("/v1/chat/completions", json={
        "model": "pin:comput3/hermes-3-405b",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"


def test_pin_to_missing_pair_returns_5xx(client):
    r = client.post("/v1/chat/completions", json={
        "model": "pin:nope/nada",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code >= 500
    assert r.json()["error"]["type"] == "router_error"


def test_fallback_on_first_candidate_failure(client, host):
    # Make every candidate fail except a known-second-tier one to force the
    # router to walk past failures into a working candidate.
    pairs = _all_pairs(host)
    assert len(pairs) >= 2
    target_prov, target_fam = pairs[-1]
    for prov, fam in pairs:
        if (prov, fam) == (target_prov, target_fam):
            host.set_mock_response(prov, fam, _ok_response("fallback worked"))
        else:
            host.set_mock_response(prov, fam, _err_response("server_error", 500))
    r = client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "fallback worked"
    assert body["x_router"]["provider"] == target_prov


# ---- error observability -------------------------------------------------

def test_error_response_carries_decision_trace(client, host):
    # When the router exhausts candidates, the client/proxy must be able to
    # see WHICH providers were tried and WHY each failed — not just
    # "exhausted: <last_kind>".
    for prov, fam in _all_pairs(host):
        host.set_mock_response(prov, fam, _err_response("server_error", 500))
    r = client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code >= 500
    xr = r.json()["x_router"]
    attempts = [e for e in xr["decision_trace"]["decision_path"]
                if e.get("event") == "attempted"]
    assert attempts, "decision_path records the attempts"
    for a in attempts:
        assert a["provider_id"]
        assert a["error_kind"] == "server_error"
        assert a["http_status"] == 500
        assert "mock server_error" in a["error_message"]


def test_error_message_summarizes_attempts_per_provider(client, host):
    pairs = _all_pairs(host)
    for prov, fam in pairs:
        host.set_mock_response(prov, fam, _err_response("server_error", 500))
    r = client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}],
    })
    msg = r.json()["error"]["message"]
    assert "exhausted" in msg
    # every attempted provider shows up as provider=kind(status)
    assert "server_error(500)" in msg
    assert any(prov in msg for prov, _ in pairs)


def test_success_response_carries_decision_trace(client, host):
    # A success that needed fallback hops should expose those hops too.
    pairs = _all_pairs(host)
    target = pairs[-1]
    for pair in pairs:
        if pair == target:
            host.set_mock_response(*pair, _ok_response("ok"))
        else:
            host.set_mock_response(*pair, _err_response("server_error", 500))
    r = client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    trace = r.json()["x_router"]["decision_trace"]
    events = {e.get("event") for e in trace["decision_path"]}
    assert "attempted" in events


# ---- runtime state endpoint ----------------------------------------------

def test_x_runtime_exposes_breaker_state_after_failures(client, host):
    # Drive failures into one provider, then read the live runtime state.
    pairs = _all_pairs(host)
    for prov, fam in pairs:
        host.set_mock_response(prov, fam, _err_response("server_error", 500))
    client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}],
    })

    r = client.get("/x/runtime")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["ts"], int)
    breakers = body["circuit_breakers"]
    assert breakers, "at least one breaker entry after an all-fail request"
    first = next(iter(breakers.values()))
    assert first["consecutive_failures"] >= 1
    assert "disabled_providers" in body
    assert "ema_metrics" in body


def test_x_runtime_includes_source_state_and_balances(client):
    import sources as src
    src.SOURCE_STATE.clear()
    src.SOURCE_STATE["openrouter"] = {
        "last_ok": 1781090000, "error": None, "prices_pushed": 9,
        "balances": {"openrouter": {"kind": "credits_usd", "value": 276.6,
                                     "detail": {}, "fetched_at": 1781090000}},
    }
    body = client.get("/x/runtime").json()
    assert body["balances"]["openrouter"]["kind"] == "credits_usd"
    assert body["sources"]["openrouter"]["prices_pushed"] == 9
    assert "balances" not in body["sources"]["openrouter"]  # not duplicated
    src.SOURCE_STATE.clear()


def test_x_rank_returns_live_ranking(client, host):
    r = client.get("/x/rank", params={"profile": "default"})
    assert r.status_code == 200
    body = r.json()
    assert body["profile"] == "default"
    assert body["rank_source"] == "router"
    row = body["ranked"][0]
    for key in ("provider", "model_family", "served_model_id", "tier",
                "discovery", "price_in", "price_out", "quality", "score"):
        assert key in row
    assert isinstance(body["rejected"], list)


def test_x_rank_unknown_profile_is_400(client):
    r = client.get("/x/rank", params={"profile": "nope"})
    assert r.status_code == 400


# ---- helpers -----------------------------------------------------------

def _all_pairs(host) -> list[tuple[str, str]]:
    """Every (provider_id, model_family) pair in the loaded catalog."""
    info = host.info()
    pairs: list[tuple[str, str]] = []
    # info doesn't expose pairs directly; use rank() on default to enumerate.
    ranked, _ = host.rank({"prompt": "x", "profile": "default"})
    seen = set()
    for r in ranked:
        c = r["candidate"]
        key = (c["provider_id"], c["model_family"])
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


# ---- market price book endpoint -------------------------------------------

def test_x_market_merges_book_rows_with_seller_totals(client, host):
    import sources as src
    fam = next(iter((host.catalog().get("models") or {}).keys()))
    src.SOURCE_STATE.clear()
    src.SOURCE_STATE["antseed"] = {"book": {
        "rows": [{"model_family": fam,
                  "seller": "1d90f467689d499dc435e5744b4613c3203eb0aa",
                  "wire_model_id": "x-wire", "price_in": 0.0, "price_out": 0.0,
                  "pinned_by": ["antseed_free"], "tradable_via": ["antseed_free"]}],
        "families": {fam: {"sellers_total": 7}},
        "fetched_at": 1781090000,
    }}
    r = client.get("/x/market")
    assert r.status_code == 200
    entry = next(f for f in r.json()["families"] if f["family"] == fam)
    ant = [row for row in entry["rows"] if row["source"] == "antseed"]
    assert len(ant) == 1
    assert ant[0]["seller"] == "peer 1d90f467"        # short peer id
    assert ant[0]["wire_model_id"] == "x-wire"
    assert ant[0]["tradable"] is True and ant[0]["pinned"] is True
    assert ant[0]["price_refreshed_at"] == 1781090000
    direct = len([row for row in entry["rows"] if row["source"] != "antseed"])
    assert entry["sellers_total"] == direct + 7       # book count, not shown rows
    src.SOURCE_STATE.clear()


def test_x_market_attaches_model_meta_and_book_last_seen(client, host):
    import sources as src
    fam = next(iter((host.catalog().get("models") or {}).keys()))
    src.SOURCE_STATE.clear()
    src.SOURCE_STATE["antseed"] = {"book": {
        "rows": [{"model_family": fam,
                  "seller": "1d90f467689d499dc435e5744b4613c3203eb0aa",
                  "wire_model_id": "x-wire", "price_in": 0.0, "price_out": 0.0,
                  "last_seen": 1781091331928,
                  "pinned_by": ["antseed_free"], "tradable_via": ["antseed_free"]}],
        "families": {fam: {"sellers_total": 1}}, "fetched_at": 1781090000,
    }}
    body = client.get("/x/market").json()
    # every family carries a meta object (registered model-level traits)
    assert all("meta" in f for f in body["families"])
    entry = next(f for f in body["families"] if f["family"] == fam)
    ant = next(row for row in entry["rows"] if row["source"] == "antseed")
    assert ant["last_seen"] == 1781091331928
    src.SOURCE_STATE.clear()


def test_x_market_attaches_live_perf_after_calls(client, host):
    import sources as src
    src.SOURCE_STATE.clear()
    for prov, fam in _all_pairs(host):
        host.set_mock_response(prov, fam, _ok_response("ok"))
    resp = client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}]})
    provider = resp.json()["x_router"]["provider"]
    body = client.get("/x/market").json()
    perfs = [row["perf"] for f in body["families"] for row in f["rows"]
             if row["seller"] == provider and row["perf"]]
    assert perfs, "the called provider should show live perf"
    assert perfs[0]["calls"] >= 1
    assert 0 <= perfs[0]["success_rate"] <= 1


def test_x_market_orders_families_by_quality_desc(client):
    body = client.get("/x/market").json()
    qualities = [f["quality"] if f["quality"] is not None else -1
                 for f in body["families"]]
    assert qualities == sorted(qualities, reverse=True)


# ---- executed cost stamping -------------------------------------------------

def test_executed_cost_usd_rules():
    from shim import _executed_cost_usd
    priced = {"chosen": {"provider_id": "openrouter", "price_in": 2.0, "price_out": 10.0},
              "response": {"tokens_in": 1_000_000, "tokens_out": 100_000}}
    assert _executed_cost_usd(priced) == 3.0
    # subscription backends bill $0 even when ranked with a shadow price
    assert _executed_cost_usd(dict(priced, chosen={**priced["chosen"], "provider_id": "openai"}),
                              frozenset({"openai"})) == 0.0
    # no price on the candidate -> None (read-time estimator is the fallback)
    assert _executed_cost_usd({"chosen": {"provider_id": "p"}, "response": {}}) is None
    # a NEGATIVE price never bills a negative cost — clamped to 0 at the source
    # (the -$22k consumer row was tokens × a negative chosen price)
    neg = {"chosen": {"provider_id": "openrouter", "price_in": -0.5, "price_out": -0.5},
           "response": {"tokens_in": 42_700_705, "tokens_out": 42_700_705}}
    assert _executed_cost_usd(neg) == 0.0


def test_x_router_carries_executed_cost(client, host):
    for prov, fam in _all_pairs(host):
        host.set_mock_response(prov, fam, _ok_response("ok"))
    r = client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}]})
    xr = r.json()["x_router"]
    assert "price_in" in xr and "price_out" in xr
    assert xr["cost_usd"] is not None       # example metrics price every pair
