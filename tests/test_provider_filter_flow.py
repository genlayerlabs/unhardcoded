"""
Provider filtering in a flow (engine #18 `provider_eq`).

A flow node's policy can restrict WHICH provider serves it — route by who serves,
not just by model family. Here a flow pins its nodes to {openrouter, openai} and
every routed node lands on one of those, never another catalog provider; pinning
to a single provider routes every node there. This is the host-side proof that
`provider_eq` (and its `or`-composition) flows through flow-node admission.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router_host import LLMRouterHost, make_api_kind_dispatcher  # noqa: E402
from shim import create_app  # noqa: E402


def _provider_policy(*providers: str) -> list:
    """A policy whose filter admits ONLY the given providers (an `or` of
    `provider_eq`, collapsing to a single `provider_eq` for one), scored by
    context so the pick is deterministic."""
    eqs = [["provider_eq", p] for p in providers]
    allow = ["or", *eqs] if len(eqs) > 1 else eqs[0]
    return ["policy", ["and", ["meets_req"], allow],
            ["field", "context"], ["argmax"], ["id"],
            ["always", {"action": "next_candidate"}]]


def _flow(policy: list) -> list:
    """u -> {a, b} -> f -> out, every llm node sharing one policy."""
    return ["flow", {
        "u":   {"kind": "input"},
        "a":   {"kind": "llm", "system": "Answer concisely.", "policy": policy, "inputs": ["u"]},
        "b":   {"kind": "llm", "system": "Answer rigorously.", "policy": policy, "inputs": ["u"]},
        "f":   {"kind": "llm", "system": "Synthesize.", "policy": policy, "inputs": ["a", "b"],
                "template": "A:\n$1\n\nB:\n$2"},
        "out": {"kind": "output", "inputs": ["f"]},
    }]


def _client() -> TestClient:
    async def backend(req):
        # echo the routed provider so the per-node trace shows where it landed;
        # the dispatcher falls back to this for openai_codex too, so `openai`
        # (the ChatGPT-subscription provider) succeeds like any other here.
        return {"ok": True, "latency_ms": 1,
                "response": {"text": f"[{req['provider_id']}] ok", "finish_reason": "stop",
                             "tokens_in": 3, "tokens_out": 5}}

    host = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "config.live.lua",
        metrics_path=ROOT / "metrics.live.lua",
        call_provider_async=make_api_kind_dispatcher(default=backend),
        env={
            "OPENAI_API_KEY": "sk-openai-test",
            "OPENROUTER_API_KEY": "sk-openrouter-test",
        },
        now_ms=lambda: 1,
    )
    host.init()
    return TestClient(create_app(host, default_profile="default"),
                      raise_server_exceptions=False)


def _node_providers(body: dict) -> list[str]:
    return [n["provider"] for n in body["x_router"]["decision_trace"]["flow_nodes"]]


def test_flow_filters_to_openrouter_and_openai():
    """The flow admits only openrouter + openai; every node routes within that
    set and never to another provider in the catalog (heurist, antseed, …)."""
    r = _client().post("/v1/chat/completions", json={
        "model": "", "flow_ir": _flow(_provider_policy("openrouter", "openai")),
        "messages": [{"role": "user", "content": "What is 17*23?"}]})
    assert r.status_code == 200, r.text
    provs = _node_providers(r.json())
    assert len(provs) == 3, provs                       # a, b, f all routed
    assert set(provs) <= {"openrouter", "openai"}, provs


def test_flow_pinned_to_a_single_provider_routes_only_there():
    """A singleton provider_eq filter pins every node to exactly that provider —
    proving the predicate excludes everything else, not just narrows."""
    for only in ("openrouter", "openai"):
        r = _client().post("/v1/chat/completions", json={
            "model": "", "flow_ir": _flow(_provider_policy(only)),
            "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200, r.text
        provs = _node_providers(r.json())
        assert provs and set(provs) == {only}, (only, provs)


def test_flow_excluding_a_provider_routes_around_it():
    """`not(provider_eq)` is the disable-a-provider case: forbid openrouter and no
    node may land on it, even though openrouter serves the most families."""
    policy = ["policy", ["and", ["meets_req"], ["not", ["provider_eq", "openrouter"]]],
              ["field", "context"], ["argmax"], ["id"],
              ["always", {"action": "next_candidate"}]]
    r = _client().post("/v1/chat/completions", json={
        "model": "", "flow_ir": _flow(policy),
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert "openrouter" not in _node_providers(r.json())
