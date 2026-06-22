"""
Per-call Σ_pol policies through the shim: POST /v1/chat/completions with a
`policy_ir` term, and POST /x/rank (the builder's dry-run preview).

The boundary under test is the form-delta's security invariant: a caller's
policy is UNTRUSTED DATA — admitted by the core (sorts/arity/depth/nodes),
∧-narrowed by the host's `policy_envelope`, never interpreted host-side,
and refused with a 400 (never a crash, never code execution).

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


# A minimal complete Policy term: accept everything the envelope allows, rank by
# cheapest (neg-normalized price), pick the best, no transform, plain fallthrough.
def _policy(pred=None) -> list:
    return [
        "policy",
        pred if pred is not None else ["top"],
        ["neg", ["normalize", ["field", "price_in"]]],
        ["argmax"],
        ["id"],
        ["always", {"action": "next_candidate"}],
    ]


@pytest.fixture
def host(tmp_path):
    # The core's example config, wrapped to add the host envelope — the same
    # ∧-composition config.live.lua ships (requirements hold, disabled stay out).
    cfg = tmp_path / "config_envelope.lua"
    cfg.write_text(
        f'local cfg = dofile("{ROOT}/core/config.example.lua")\n'
        'cfg.policy_envelope = { "and", { "meets_req" },'
        ' { "not", { "is", "disabled" } } }\n'
        "return cfg\n"
    )
    h = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=cfg,
        metrics_path=ROOT / "core" / "metrics.example.lua",
        now_ms=lambda: 1_000_000,
    )
    h.init()
    return h


@pytest.fixture
def client(host):
    return TestClient(create_app(host, default_profile="default"))


def _mock_everything(host, text="ok"):
    ranked, _ = host.rank({"profile": "default"})
    for r in ranked:
        c = r["candidate"]
        host.set_mock_response(c["provider_id"], c["model_family"],
                               _ok_response(text))


# ---- happy path ----------------------------------------------------------

def test_policy_ir_executes_and_reports_fingerprint(client, host):
    _mock_everything(host, "via term")
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "policy_ir": _policy(),
    })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "via term"
    # policy identity surfaces: the same fingerprint the trace carries
    assert body["x_router"]["policy_fingerprint"]
    assert (body["x_router"]["decision_trace"]["policy_fingerprint"]
            == body["x_router"]["policy_fingerprint"])


def test_rank_post_previews_a_policy_term(client):
    r = client.post("/x/rank", json={"policy_ir": _policy()})
    assert r.status_code == 200
    body = r.json()
    assert body["ranked"], "a top-accepting policy ranks the whole catalog"
    # rank is a dry run: rows carry scores, nothing was called
    assert all("score" in row for row in body["ranked"])


# ---- admission: untrusted input is refused, never executed ---------------

def test_unknown_op_is_a_400_not_a_crash(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "policy_ir": ["policy", ["frobnicate"], ["field", "context"],
                      ["argmax"], ["id"],
                      ["always", {"action": "next_candidate"}]],
    })
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "invalid_policy"
    assert "frobnicate" in err["message"]


def test_node_bomb_is_rejected_at_admission(client):
    # > max_nodes (4096) leaves in one AC op: admission must refuse before
    # any recursion blows up — O(|term|) cost is the spec's promise.
    bomb = ["and"] + [["top"] for _ in range(5000)]
    r = client.post("/x/rank", json={"policy_ir": _policy(pred=bomb)})
    assert r.status_code == 400
    assert "max size" in r.json()["error"]["message"]


def test_non_term_policy_is_a_400(client):
    r = client.post("/x/rank", json={"policy_ir": ["zero"]})
    assert r.status_code == 400  # a Scorer is not a Policy


# ---- the builder: compose -> preview -> download -> execute ---------------

def test_policy_build_composes_a_runnable_term(client):
    spec = {"scorer": ["add",
                       ["scale", 2, ["field", "context"]],
                       ["scale", 1, ["neg", ["normalize", ["field", "price_in"]]]]],
            "filter": ["requirements", "not_disabled", {"price_max": {"output": 50}}],
            "selector": "argmax"}
    r = client.post("/x/policy/build", json=spec)
    assert r.status_code == 200
    built = r.json()
    assert built["policy_ir"][0] == "policy"
    assert built["fingerprint"]
    assert built["version"].startswith("sigma-pol/")
    # the composed term must pass the real admission point and rank
    r2 = client.post("/x/rank", json={"policy_ir": built["policy_ir"]})
    assert r2.status_code == 200
    assert r2.json()["ranked"], "a sane spec ranks something"


def test_policy_build_family_in_filters_to_the_set(client):
    # The family-set filter the dashboard builder now exposes: "cheapest among
    # {hermes-3-405b, deepseek-v3}". family_in lowers to or(family_eq...) in the
    # core's elaborate (the family_eq predicate added to the IR signature).
    spec = {"scorer": ["neg", ["normalize", ["field", "price_in"]]],
            "filter": ["requirements", "not_disabled",
                       {"family_in": ["hermes-3-405b", "deepseek-v3"]}],
            "selector": "argmax"}
    built = client.post("/x/policy/build", json=spec).json()
    assert built["policy_ir"][0] == "policy"
    ranked = client.post("/x/rank",
                         json={"policy_ir": built["policy_ir"]}).json()["ranked"]
    assert ranked, "the family set is non-empty in the example catalog"
    assert {r["model_family"] for r in ranked} <= {"hermes-3-405b", "deepseek-v3"}, \
        "only candidates in the requested family set survive"


def test_policy_normalize_fingerprints_a_raw_term(client):
    # The builder composes the IR directly now; /x/policy/normalize canonicalizes
    # a raw term and stamps its identity for Download. A raw-field scorer
    # (normalize/neg over an observed field) round-trips and still ranks.
    term = ["policy",
            ["and", ["meets_req"], ["not", ["is", "disabled"]],
             ["cmp", "price_out", "le", 20]],
            ["scale", 1, ["neg", ["normalize", ["field", "price_in"]]]],
            ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]
    r = client.post("/x/policy/normalize", json={"policy_ir": term})
    assert r.status_code == 200
    body = r.json()
    assert body["policy_ir"][0] == "policy"
    assert body["fingerprint"]
    assert body["version"].startswith("sigma-pol/")
    # the normalized term must pass the real admission point and rank
    assert client.post("/x/rank",
                       json={"policy_ir": body["policy_ir"]}).status_code == 200


def test_policy_build_rejects_unknown_directive(client):
    r = client.post("/x/policy/build",
                    json={"filter": ["frobnicate"]})
    assert r.status_code == 400


# ---- routing by speed: offer.latency_ms is a real, routable field ----------

def test_policy_can_route_by_latency_field(client):
    # The host now MEASURES per-route latency (route_latency) and stamps it as
    # offer.latency_ms. Prove the algebra actually routes on it: a policy that
    # gates on latency_ms and scores by -latency_ms (fastest wins) is admitted and
    # ranks through the real core — so a slow $0 peer can be filtered/outranked by
    # a fast one instead of stalling the caller for 12s.
    term = ["policy",
            ["and", ["meets_req"], ["not", ["is", "disabled"]],
             ["cmp", "latency_ms", "le", 5000]],
            ["neg", ["field", "latency_ms"]],
            ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]
    r = client.post("/x/rank", json={"policy_ir": term})
    assert r.status_code == 200, r.text
    assert all("score" in row for row in r.json()["ranked"])


def test_built_term_executes_and_reports_identity(client, host):
    built = client.post("/x/policy/build",
                        json={"scorer": ["neg", ["normalize", ["field", "price_in"]]]}).json()
    _mock_everything(host, "via built policy")
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "policy_ir": built["policy_ir"],
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "via built policy"
    # NB: the EXECUTED fingerprint differs from the composed one by design —
    # the host envelope is ∧-ed on at execution. The downloaded policy's
    # identity is the composed fingerprint; the run's identity includes the
    # envelope. Both must exist.
    assert r.json()["x_router"]["policy_fingerprint"]
    assert built["fingerprint"]


# (sigma-pol/v2) weighted scoring was removed; build_policy no longer mirrors
# the core's renormalize_weights, so the host-side parity test is gone with it.

# ---- the envelope: callers narrow, never widen ---------------------------

def test_envelope_keeps_disabled_providers_out(client, host):
    ranked, _ = host.rank({"policy_ir": _policy()})
    assert ranked, "baseline: the catalog ranks"
    victim = ranked[0]["candidate"]["provider_id"]

    # Disable the top provider engine-side (post-IR shape: { kind, at_ms }).
    host.router._test.runtime()["disabled_providers"][victim] = (
        host.lua.table_from({"kind": "auth_error", "at_ms": 999_000})
    )

    # The caller's term accepts EVERYTHING (pred = top) — but the envelope
    # is ∧-ed on by the core, so the disabled provider must not come back.
    r = client.post("/x/rank", json={"policy_ir": _policy(pred=["top"])})
    assert r.status_code == 200
    providers = {row["provider"] for row in r.json()["ranked"]}
    assert victim not in providers, "envelope must override a permissive term"


def test_emitted_trace_caps_ranked_to_top_n(client, host):
    # The decision_trace sent to the client must not carry the whole ranked
    # catalog (it bloated every response to ~95 KB). Keep top-N + ranked_total.
    import shim
    big = {"ranked": [{"provider_id": f"p{i}"} for i in range(300)],
           "decision_path": [{"event": "attempted", "provider_id": "p0"}],
           "policy_fingerprint": "fp"}
    trimmed = shim._trim_trace(big)
    assert len(trimmed["ranked"]) == shim._TRACE_RANKED_TOP_N == 10
    assert trimmed["ranked_total"] == 300
    assert trimmed["decision_path"] == big["decision_path"]   # attempts untouched
    # a short ranked is passed through unchanged (no ranked_total noise)
    small = {"ranked": [{"provider_id": "p0"}], "policy_fingerprint": "fp"}
    assert shim._trim_trace(small) == small
    assert shim._trim_trace(None) is None


def test_emitted_trace_caps_ranked_inside_flow_nodes():
    # A Σ_flow nests a full decision_trace per node; each carries its own ranked
    # catalog. Without trimming those, an N-node flow emits N× the catalog (the
    # ensemble's trace overflowed the proxy tail -> Activity showed nothing). The
    # per-node provider/peer/decision_path/latency — what makes the flow visible —
    # must survive intact.
    import shim
    flow = {"flow_fingerprint": "ff", "flow_nodes": [
        {"node": "glm", "provider": "antseed", "latency_ms": 12000,
         "decision_trace": {"ranked": [{"provider_id": f"p{i}"} for i in range(200)],
                            "decision_path": [{"event": "attempted", "provider_id": "peerX",
                                               "latency_ms": 12000}]}},
        {"node": "merge", "provider": "openai", "latency_ms": 900,
         "decision_trace": {"ranked": [{"provider_id": f"q{i}"} for i in range(200)],
                            "decision_path": [{"event": "attempted", "provider_id": "openai"}]}},
    ]}
    out = shim._trim_trace(flow)
    glm, merge = out["flow_nodes"]
    assert len(glm["decision_trace"]["ranked"]) == 10           # nested ranked capped
    assert glm["decision_trace"]["ranked_total"] == 200
    assert glm["provider"] == "antseed" and glm["latency_ms"] == 12000   # node detail intact
    assert glm["decision_trace"]["decision_path"][0]["provider_id"] == "peerX"  # the peer it called
    assert merge["provider"] == "openai"
