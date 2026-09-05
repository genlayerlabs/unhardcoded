"""Essence for /v1/compact — append-only context sealing (stateless).

A coding/agent loop grows its context until it must be compacted. Done naively
(re-summarize everything) the seal rewrites the prefix and destroys the prompt
cache. /v1/compact instead seals only the AGED middle and splices it back so the
frozen prefix and the recent tail are byte-identical — everything upstream stays
cache-hot. This closes:
  1. the append-only splice (prefix preserved, aged sealed once, recent verbatim);
  2. the no-op when there is nothing worth sealing;
  3. the anti-orphan guard (a leading tool message whose assistant got dropped
     would 400 the next provider call).

The summarizer is a cheaply-routed call; here it is mocked, so the test is
hermetic and deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router_host import LLMRouterHost  # noqa: E402
from shim import create_app               # noqa: E402


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
    return TestClient(create_app(host, default_profile="default"))


# Pin the summarizer to a known config.example route so we can mock it.
_PIN = ["policy",
        ["and", ["meets_req"], ["not", ["is", "disabled"]],
                ["family_eq", "hermes-3-405b"]],
        ["neg", ["normalize", ["field", "price_in"]]],
        ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]


def _ok(text):
    return {"ok": True, "latency_ms": 5,
            "response": {"text": text, "tokens_in": 5, "tokens_out": 3}}


def test_compact_splices_append_only(client, host):
    host.set_mock_response("comput3", "hermes-3-405b", _ok("SEALED."))
    sys0 = {"role": "system", "content": "You are a coding agent. <skill>"}
    convo = []
    for i in range(8):
        convo.append({"role": "user", "content": f"u{i}"})
        convo.append({"role": "assistant", "content": f"a{i}"})
    msgs = [sys0] + convo

    r = client.post("/v1/compact",
                    json={"messages": msgs, "keep_recent": 4, "policy_ir": _PIN})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["compacted"] is True
    m = out["messages"]
    assert m[0] == sys0                       # frozen prefix byte-identical
    assert m[1]["role"] == "system" and "SEALED." in m[1]["content"]
    assert m[-4:] == msgs[-4:]                 # recent tail verbatim
    assert len(m) < len(msgs)                  # actually compacted


def test_compact_noop_when_short(client):
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "hi"}]
    r = client.post("/v1/compact", json={"messages": msgs, "keep_recent": 6})
    assert r.status_code == 200
    assert r.json() == {"messages": msgs, "compacted": False}


def test_compact_drops_orphan_tool_at_seam(client, host):
    # The assistant that issued call_1 is in the aged span (dropped); its tool
    # answer is the first "recent" message -> it must not survive as an orphan.
    host.set_mock_response("comput3", "hermes-3-405b", _ok("S"))
    sys0 = {"role": "system", "content": "sys"}
    aged = [{"role": "user", "content": "u0"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "x", "arguments": "{}"}}]}]
    recent = [{"role": "tool", "tool_call_id": "call_1", "content": "result"},
              {"role": "user", "content": "u1"},
              {"role": "assistant", "content": "a1"},
              {"role": "user", "content": "u2"},
              {"role": "assistant", "content": "a2"}]
    msgs = [sys0] + aged + recent

    r = client.post("/v1/compact",
                    json={"messages": msgs, "keep_recent": 5, "policy_ir": _PIN})
    out = r.json()["messages"]
    roles = [x["role"] for x in out]
    seal_idx = roles.index("system", 1)        # the sealed block
    assert out[seal_idx + 1]["role"] != "tool", f"orphan tool after seal: {roles}"


# ---- cost accounting on the seal leg (usage + x_router) -------------------
#
# The seal is a real billable LLM call. The wire contract (additive, optional):
# every 2xx /v1/compact response that FOLLOWS host.execute_async also carries
# "usage" (OpenAI token shape, the summarizer leg's tokens) and "x_router"
# (same block _build_x_router emits on chat) — including the compacted:false
# partial-failure return. Early returns that precede execution carry neither.
# "messages"/"compacted" stay byte-identical (SZC's compact_splice reads only
# "messages").

def _long_convo(n_pairs=8):
    sys0 = {"role": "system", "content": "You are a coding agent. <skill>"}
    convo = []
    for i in range(n_pairs):
        convo.append({"role": "user", "content": f"u{i}"})
        convo.append({"role": "assistant", "content": f"a{i}"})
    return [sys0] + convo


def _summarizer_result(text, tokens_in=41, tokens_out=7, cost=0.000123):
    # cost_reported is the provider's OWN cost for THIS call — authoritative in
    # _executed_cost_usd, so the asserted dollar number is exactly the seal leg's.
    return {"ok": True, "latency_ms": 5,
            "response": {"text": text, "tokens_in": tokens_in,
                         "tokens_out": tokens_out, "cost_reported": cost}}


def test_compact_success_carries_summarizer_cost(client, host):
    host.set_mock_response("comput3", "hermes-3-405b",
                           _summarizer_result("SEALED."))
    msgs = _long_convo()
    r = client.post("/v1/compact",
                    json={"messages": msgs, "keep_recent": 4, "policy_ir": _PIN})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["compacted"] is True
    # usage = the summarizer call's tokens, OpenAI shape
    assert out["usage"]["prompt_tokens"] == 41
    assert out["usage"]["completion_tokens"] == 7
    # x_router = the same block the chat path emits, costed
    xr = out["x_router"]
    assert xr["cost_usd"] == 0.000123
    assert xr["cost_basis"] == "reported"
    assert xr["provider"] == "comput3"
    assert xr["model_family"] == "hermes-3-405b"


def test_compact_partial_failure_still_carries_cost(client, host):
    # An empty summary is still a billed call: compacted:false AFTER execution
    # must carry the accounting keys (the proxy records the spend either way).
    host.set_mock_response("comput3", "hermes-3-405b",
                           _summarizer_result("", tokens_in=13, tokens_out=0,
                                              cost=0.00005))
    msgs = _long_convo()
    r = client.post("/v1/compact",
                    json={"messages": msgs, "keep_recent": 4, "policy_ir": _PIN})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["compacted"] is False
    assert out["messages"] == msgs           # never lose content
    assert out["usage"]["prompt_tokens"] == 13
    assert out["x_router"]["cost_usd"] == 0.00005


def test_compact_tokenless_provider_omits_usage_but_keeps_x_router(client, host):
    # A provider that reports no token counts: "usage" must be OMITTED (never
    # an empty {}), while x_router still carries the billed cost — the exact
    # shape the metering proxy's legacy/costed distinction keys on.
    host.set_mock_response("comput3", "hermes-3-405b",
                           {"ok": True, "latency_ms": 5,
                            "response": {"text": "SEALED.",
                                         "cost_reported": 0.000123}})
    msgs = _long_convo()
    r = client.post("/v1/compact",
                    json={"messages": msgs, "keep_recent": 4, "policy_ir": _PIN})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["compacted"] is True
    assert "usage" not in out
    assert out["x_router"]["cost_usd"] == 0.000123


def test_openai_usage_preserves_explicit_zero_cached_tokens():
    # tokens_cached: 0 = caching evaluated, no hits — must survive as an
    # explicit cached_tokens: 0 (x_router passes the 0 through; the OpenAI
    # block has to agree). Absent = provider never reported it: no details key.
    from shim import _openai_usage
    zero = _openai_usage({"tokens_in": 5, "tokens_out": 3, "tokens_cached": 0})
    assert zero["prompt_tokens_details"] == {"cached_tokens": 0}
    absent = _openai_usage({"tokens_in": 5, "tokens_out": 3})
    assert "prompt_tokens_details" not in absent


def test_compact_early_return_has_no_cost_keys(client):
    # Nothing worth sealing -> no LLM call was made -> no usage, no x_router.
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "hi"}]
    r = client.post("/v1/compact", json={"messages": msgs, "keep_recent": 6})
    assert r.status_code == 200
    out = r.json()
    assert set(out.keys()) == {"messages", "compacted"}
    assert out == {"messages": msgs, "compacted": False}


def test_compact_splice_untouched_by_accounting(client, host):
    # SZC compat: "messages"/"compacted" are byte-identical to the pre-contract
    # splice — the new keys are purely additive.
    host.set_mock_response("comput3", "hermes-3-405b",
                           _summarizer_result("SEALED."))
    msgs = _long_convo()
    r = client.post("/v1/compact",
                    json={"messages": msgs, "keep_recent": 4, "policy_ir": _PIN})
    out = r.json()
    m = out["messages"]
    assert m[0] == msgs[0]                                   # frozen prefix
    assert m[1] == {"role": "system",
                    "content": "[Earlier conversation, sealed summary]\nSEALED."}
    assert m[2:] == msgs[-4:]                                # recent tail verbatim
    assert out["compacted"] is True
    assert set(out.keys()) == {"messages", "compacted", "usage", "x_router"}


def test_compact_cost_is_the_seal_leg_not_the_conversation(client, host):
    # The dollars/tokens belong to the SUMMARIZER call, independent of how big
    # the original conversation is: double the conversation, same accounting.
    host.set_mock_response("comput3", "hermes-3-405b",
                           _summarizer_result("S", tokens_in=41, tokens_out=7,
                                              cost=0.000123))
    small = client.post("/v1/compact", json={
        "messages": _long_convo(8), "keep_recent": 4, "policy_ir": _PIN}).json()
    big = client.post("/v1/compact", json={
        "messages": _long_convo(80), "keep_recent": 4, "policy_ir": _PIN}).json()
    assert small["usage"] == big["usage"] == {
        "prompt_tokens": 41, "completion_tokens": 7}
    assert small["x_router"]["cost_usd"] == big["x_router"]["cost_usd"] == 0.000123


# ---- the compaction trigger flag (x_router.compact) -----------------------

def _ok_tokens(n_in):
    return {"ok": True, "latency_ms": 5,
            "response": {"text": "ok", "tokens_in": n_in, "tokens_out": 2}}


def test_compact_flag_set_when_input_large(client, host):
    # tokens_in over the default threshold (24000) -> the response tells the
    # agent to seal.
    host.set_mock_response("comput3", "hermes-3-405b", _ok_tokens(30000))
    r = client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}],
        "policy_ir": _PIN})
    assert r.status_code == 200, r.text
    assert r.json()["x_router"]["compact"] is True


def test_compact_flag_unset_when_input_small(client, host):
    host.set_mock_response("comput3", "hermes-3-405b", _ok_tokens(100))
    r = client.post("/v1/chat/completions", json={
        "model": "", "messages": [{"role": "user", "content": "hi"}],
        "policy_ir": _PIN})
    assert r.status_code == 200, r.text
    assert r.json()["x_router"]["compact"] is False
