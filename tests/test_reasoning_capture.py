"""No paid calls: provider-exposed reasoning survives the buffered wire boundary."""
import httpx
import pytest
from provider_adapters.openai_compatible import make_async_call_provider
from shim import _router_response_to_openai
from tests.test_streaming import OPENAI_REQ

@pytest.mark.asyncio
@pytest.mark.parametrize("content,finish", [('{}', 'stop'), ('', 'length')])
async def test_reasoning_and_usage_survive_adapter_and_public_chat(content, finish):
    message = {"role": "assistant", "content": content, "reasoning": "Exposed explanation",
               "reasoning_details": [{"type": "reasoning.summary", "summary": "summary"},
                                     {"type": "reasoning.encrypted", "data": "opaque"}]}
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30,
             "completion_tokens_details": {"reasoning_tokens": 15}, "cost": 0.001}
    async def respond(request):
        return httpx.Response(200, json={"model": "fixture", "choices": [
            {"message": message, "finish_reason": finish}], "usage": usage})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await make_async_call_provider(client=client)(OPENAI_REQ)
    assert result['ok']
    public = _router_response_to_openai(result, 'fixture')
    assert public['choices'][0]['message'] == message
    assert public['choices'][0]['finish_reason'] == finish
    assert public['usage']['completion_tokens_details']['reasoning_tokens'] == 15
    assert public['usage']['cost'] == .001

from tests.test_reasoning_controls import host
from tests.test_compact import _PIN
from fastapi.testclient import TestClient
from shim import create_app

def test_reasoning_survives_real_engine_and_chat_endpoint(host):
    async def provider(req):
        return {"ok": True, "response": {"text": "{}", "provider_message": {
            "reasoning": "explanation", "reasoning_details": [{"type": "reasoning.summary", "summary": "summary"}]},
            "provider_usage": {"completion_tokens_details": {"reasoning_tokens": 8}}, "tokens_out": 10}}
    host.set_async_call_hook(provider)
    response = TestClient(create_app(host)).post('/v1/chat/completions', json={"messages": [], "policy_ir": _PIN})
    assert response.status_code == 200
    assert response.json()['choices'][0]['message']['reasoning'] == 'explanation'

@pytest.mark.asyncio
async def test_buffered_sse_keeps_reasoning():
    import json
    from tests.test_streaming import FakeStreamClient, FakeStreamResponse
    chunks = [{"choices": [{"delta": {"reasoning": "Think "}}]},
              {"choices": [{"delta": {"reasoning": "again", "content": "{}"}, "finish_reason": "stop"}]}]
    client = FakeStreamClient(FakeStreamResponse(200, ['data: '+json.dumps(c) for c in chunks]+['data: [DONE]']))
    result = await make_async_call_provider(client=client)({**OPENAI_REQ, "first_token_timeout_ms": 1000})
    assert result['ok']
    assert result['response']['provider_message']['reasoning'] == 'Think again'

def test_codex_oauth_summary_survives_without_exposing_internal_thought():
    import json
    from codex_backend import build_codex_body, aggregate_codex_sse
    req = build_codex_body({"served_model_id": "fixture", "reasoning_effort": "low"})
    assert req['reasoning'] == {'effort': 'low', 'summary': 'auto'}
    item = {"id": "r1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "Exposed summary"}], "encrypted_content": "opaque"}
    events = [{"type": "response.reasoning_summary_text.delta", "item_id": "r1", "summary_index": 0, "delta": "Exposed summary"},
              {"type": "response.output_item.done", "item": item},
              {"type": "response.output_text.delta", "delta": "{}"},
              {"type": "response.completed", "response": {"output": [item], "usage": {"output_tokens": 25, "output_tokens_details": {"reasoning_tokens": 20}}}}]
    result = aggregate_codex_sse(['data: '+json.dumps(e) for e in events], 1)
    public = _router_response_to_openai(result, 'fixture')
    assert public['x_reasoning_items'] == [item]
    assert public['choices'][0]['message']['reasoning_details'][0]['summary'] == 'Exposed summary'
    assert public['usage']['completion_tokens_details']['reasoning_tokens'] == 20
