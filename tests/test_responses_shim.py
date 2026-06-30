"""Integration tests for the /v1/responses shim. Mirrors test_shim.py: a real
LLMRouterHost backed by mock provider responses; only outbound HTTP is mocked."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_router_host import LLMRouterHost  # noqa: E402
from shim import create_app  # noqa: E402


def _ok(text="hi back", tool_calls=None):
    return {"ok": True, "latency_ms": 10, "response": {
        "text": text, "tool_calls": tool_calls, "finish_reason":
        "tool_calls" if tool_calls else "stop", "tokens_in": 7,
        "tokens_out": 3, "tokens_total": 10, "raw_model": "mock-model-id"}}


def _all_pairs(host) -> list[tuple[str, str]]:
    """Every (provider_id, model_family) pair in the loaded catalog (via rank)."""
    ranked, _ = host.rank({"prompt": "x", "profile": "default"})
    pairs: list[tuple[str, str]] = []
    seen = set()
    for r in ranked:
        c = r["candidate"]
        key = (c["provider_id"], c["model_family"])
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


@pytest.fixture
def host():
    h = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "core" / "config.example.lua",
        metrics_path=ROOT / "core" / "metrics.example.lua",
        now_ms=lambda: 1_000_000)
    h.init()
    return h


@pytest.fixture
def client(host):
    return TestClient(create_app(host, default_profile="default"))


def _seed(host, resp):
    for prov, fam in _all_pairs(host):
        host.set_mock_response(prov, fam, resp)


def test_responses_nonstream_returns_response_object(client, host):
    _seed(host, _ok("hello from router"))
    r = client.post("/v1/responses", json={
        "model": "", "input": "hi", "stream": False})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    msg = [o for o in body["output"] if o["type"] == "message"][0]
    assert msg["content"][0]["text"] == "hello from router"
    assert body["usage"]["total_tokens"] == 10
    assert body["x_router"]["provider"] is not None


def test_responses_input_items_and_instructions(client, host):
    _seed(host, _ok("ok"))
    r = client.post("/v1/responses", json={
        "model": "", "instructions": "be terse",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        "stream": False})
    assert r.status_code == 200
    assert r.json()["output"][0]["content"][0]["text"] == "ok"


def test_responses_stream_emits_completed(client, host):
    _seed(host, _ok("streamed hi"))
    with client.stream("POST", "/v1/responses", json={
            "model": "", "input": "hi", "stream": True}) as r:
        assert r.status_code == 200
        body = "".join(chunk for chunk in r.iter_text())
    events = [ln[len("event:"):].strip()
              for ln in body.split("\n") if ln.startswith("event:")]
    assert events[0] == "response.created"
    assert "response.output_text.delta" in events
    assert events[-1] == "response.completed"
    # the completed event carries the assistant text
    datas = [json.loads(ln[len("data:"):].strip())
             for ln in body.split("\n") if ln.startswith("data:")]
    completed = [d for d in datas if d.get("type") == "response.completed"][0]
    msg = [o for o in completed["response"]["output"] if o["type"] == "message"][0]
    assert msg["content"][0]["text"] == "streamed hi"


def test_responses_tool_call_surfaces_function_call_item(client, host):
    tcs = [{"id": "call_1", "type": "function",
            "function": {"name": "shell", "arguments": '{"command":"ls"}'}}]
    _seed(host, _ok(text="", tool_calls=tcs))
    r = client.post("/v1/responses", json={
        "model": "", "input": "run ls",
        "tools": [{"type": "function", "name": "shell",
                   "parameters": {"type": "object",
                                  "properties": {"command": {"type": "string"}}}}],
        "stream": False})
    assert r.status_code == 200
    fc = [o for o in r.json()["output"] if o["type"] == "function_call"][0]
    assert fc["name"] == "shell" and fc["call_id"] == "call_1"
    assert fc["arguments"] == '{"command":"ls"}'


def test_responses_profiled_route(client, host):
    _seed(host, _ok("cheap"))
    r = client.post("/cheap_explore/v1/responses", json={
        "model": "ignored", "input": "hi", "stream": False})
    assert r.status_code == 200
    assert r.json()["output"][0]["content"][0]["text"] == "cheap"


def test_responses_router_error_maps_to_status(client, host):
    # no mocks set -> candidates fail -> router returns an error
    r = client.post("/v1/responses", json={
        "model": "", "input": "hi", "stream": False})
    assert r.status_code >= 400
    assert r.json()["error"]["type"] == "router_error"


def test_responses_stream_failure_sequence_numbers_strictly_increase(
        client, host, monkeypatch):
    # The streaming failure path (gen_running) only runs when the task is STILL
    # in flight at the early-fail check, then fails. Force it deterministically:
    # collapse the early-fail window AND make execute_async suspend before
    # returning a router error (so it is not already done at the check). Every
    # numbered SSE frame must then have a STRICTLY INCREASING sequence_number per
    # the Responses contract. Regression guard: response.created and
    # response.failed both defaulted to sequence_number=0, colliding on this path
    # and tripping strict clients (e.g. the Codex CLI).
    import asyncio
    import shim as _shim

    async def _slow_error(contract):
        await asyncio.sleep(0.02)  # suspend so the task is "still running"
        return {"ok": False, "error": "router error"}

    monkeypatch.setattr(_shim, "_EARLY_FAIL_S", 0.0)
    monkeypatch.setattr(host, "execute_async", _slow_error)

    with client.stream("POST", "/v1/responses", json={
            "model": "", "input": "hi", "stream": True}) as r:
        assert r.status_code == 200
        body = "".join(chunk for chunk in r.iter_text())
    datas = [json.loads(ln[len("data:"):].strip())
             for ln in body.split("\n") if ln.startswith("data:")]
    types = [d.get("type") for d in datas]
    assert "response.created" in types and "response.failed" in types
    seqs = [d["sequence_number"] for d in datas if "sequence_number" in d]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), \
        f"sequence_numbers must be strictly increasing, got {seqs}"
