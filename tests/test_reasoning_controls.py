"""Explicit generation controls must reach the OpenAI-compatible wire unchanged."""
import copy
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from provider_adapters.openai_compatible import make_async_call_provider, stream_openai_compatible
from shim import ChatRequest, _request_to_contract, create_app
from llm_router_host import LLMRouterHost
from tests.test_compact import _PIN
from tests.test_streaming import OPENAI_REQ, FakeStreamClient, FakeStreamResponse, _openai_lines


CONTROLS = [{"reasoning": {"enabled": False}}, {"reasoning": {"effort": "low"}},
            {"reasoning_effort": "low"}, {}]


@pytest.fixture
def host():
    root = Path(__file__).resolve().parents[1]
    host = LLMRouterHost(router_path=root/'core/router.lua',
        config_path=root/'core/config.example.lua', metrics_path=root/'core/metrics.example.lua',
        env={"COMPUT3_API_KEY": "fixture-credential"})
    host.init()
    return host


@pytest.mark.parametrize("controls", CONTROLS)
def test_chat_contract_preserves_explicit_controls(controls):
    contract = _request_to_contract(ChatRequest(**controls), "default")
    assert {k: contract[k] for k in ("reasoning", "reasoning_effort") if k in contract} == controls


@pytest.mark.parametrize("controls", CONTROLS)
@pytest.mark.asyncio
async def test_buffered_adapter_sends_controls(controls):
    seen = []
    async def respond(request):
        import json
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await make_async_call_provider(client=client)({**OPENAI_REQ, **controls})
    assert result["ok"]
    assert {k: seen[0][k] for k in ("reasoning", "reasoning_effort") if k in seen[0]} == controls


@pytest.mark.parametrize("controls", CONTROLS)
@pytest.mark.asyncio
async def test_streaming_adapter_sends_controls(controls):
    client = FakeStreamClient(FakeStreamResponse(200, _openai_lines("ok")))
    await stream_openai_compatible({**OPENAI_REQ, **controls}, lambda _: None, client=client)
    body = client.requests[0]["json"]
    assert {k: body[k] for k in ("reasoning", "reasoning_effort") if k in body} == controls


@pytest.mark.parametrize("endpoint", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("controls", CONTROLS[:3])
def test_http_to_core_provider_preserves_controls(host, endpoint, stream, controls):
    seen = []
    async def call(req):
        seen.append(copy.deepcopy(req))
        return {"ok": True, "response": {"text": "ok", "tokens_in": 1, "tokens_out": 1}}
    host.set_async_call_hook(call)
    payload = {"policy_ir": _PIN, "stream": stream, **controls}
    payload.update({"input": "hello"} if endpoint.endswith("responses") else
                   {"messages": [{"role": "user", "content": "hello"}]})
    response = TestClient(create_app(host)).post(endpoint, json=payload)
    assert response.status_code == 200, response.text
    assert len(seen) == 1
    assert {k: seen[0][k] for k in ("reasoning", "reasoning_effort") if k in seen[0]} == controls


def test_policy_can_set_reasoning_effort_before_provider_call(host):
    seen = []
    async def call(req):
        seen.append(req)
        return {"ok": True, "response": {"text": "ok"}}
    host.set_async_call_hook(call)
    policy = copy.deepcopy(_PIN)
    policy[4] = ["set_param", "reasoning_effort", "low"]
    response = TestClient(create_app(host)).post(
        "/v1/chat/completions", json={"messages": [], "policy_ir": policy})
    assert response.status_code == 200, response.text
    assert seen[0]["reasoning_effort"] == "low"


@pytest.mark.parametrize("controls", CONTROLS[:3])
def test_flow_controls_reach_generation_but_not_native_decisions(host, monkeypatch, controls):
    from tests.test_flow_data import triage
    seen = []
    async def execute(contract, **kwargs):
        seen.append(copy.deepcopy(contract))
        if contract.get("protocol") == "decisions":
            response = {"decision": {"model": "fixture", "answers": {
                key: {"type": "choice", "choice": "support",
                      "probabilities": {"support": 1.0, "sales": 0.0}} for key in ("a", "b")}}}
        else:
            response = {"text": '{"a":"Reply A","b":"Reply B"}', "finish_reason": "stop"}
        return {"ok": True, "response": response}
    monkeypatch.setattr(host, "execute_async", execute)
    response = TestClient(create_app(host)).post("/v1/chat/completions", json={
        "flow_ir": triage(), "flow_input": {"a": "Crash", "b": "Question"}, **controls})
    assert response.status_code == 200, response.text
    assert len(seen) == 2
    assert seen[0]["protocol"] == "decisions"
    assert not any(k in seen[0] for k in controls)
    assert {k: seen[1][k] for k in controls} == controls
