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
