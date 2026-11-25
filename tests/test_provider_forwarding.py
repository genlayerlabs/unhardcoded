"""Exercise a catalog-configured HTTP relay through the router shim."""
import asyncio
import json

import httpx
import pytest

from provider_adapters.openai_compatible import (
    make_async_call_provider,
    stream_openai_compatible,
)
from shim import create_app
from tests.test_compact import _PIN
from tests.test_reasoning_controls import host
from tests.test_streaming import _openai_lines

RELAY_BASE = "http://127.0.0.1:8765/relay/v1"
RELAY_URL = RELAY_BASE + "/chat/completions"


def check_wire(request):
    assert str(request.url) == RELAY_URL
    assert request.headers["authorization"] == "Bearer fixture-credential"
    assert json.loads(request.content)["messages"] == [{"role": "user", "content": "hi"}]


def payload(stream=False):
    return {"policy_ir": _PIN, "stream": stream,
            "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.asyncio
async def test_relay_connection_failure_retries_same_endpoint(host):
    host.config.providers.comput3.base_url = RELAY_BASE
    attempts = []

    async def refuse(request):
        check_wire(request)
        attempts.append(request.url)
        raise httpx.ConnectError("relay unavailable", request=request)

    request_body = payload()
    request_body["policy_ir"] = [
        *_PIN[:-1], ["always", {"action": "retry_same", "attempts": 2,
                                "backoff_ms": 0, "then_action": "abort"}],
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(refuse)) as upstream:
        host.set_async_call_hook(make_async_call_provider(client=upstream, env_get=host._env.get))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(host)),
                                     base_url="http://router") as client:
            response = await client.post("/v1/chat/completions", json=request_body)
    assert response.is_error
    assert len(attempts) >= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["deadline", "disconnect", "stream_before_delta", "stream_after_delta"])
async def test_cancel_in_flight_relay(host, stop):
    started = asyncio.Event()
    closed = asyncio.Event()
    streamed = asyncio.Event()
    disconnected = asyncio.Event()
    host.config.providers.comput3.base_url = RELAY_BASE
    stream = stop.startswith("stream_")

    class RelayBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            if stop == "stream_after_delta":
                yield (_openai_lines("first")[0] + "\n\n").encode()
            await asyncio.Future()

        async def aclose(self):
            closed.set()

    async def relay(request):
        check_wire(request)
        assert bool(json.loads(request.content).get("stream")) == stream
        return httpx.Response(200, stream=RelayBody())

    body = json.dumps(payload(stream)).encode()
    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        if messages:
            return messages.pop()
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if b'"content":"first"' in message.get("body", b"").replace(b" ", b""):
            streamed.set()

    async with httpx.AsyncClient(transport=httpx.MockTransport(relay)) as upstream:
        async def streaming_call(request, emit):
            return await stream_openai_compatible(request, emit, client=upstream,
                                                   env_get=host._env.get)

        host.set_async_call_hook(make_async_call_provider(client=upstream, env_get=host._env.get))
        app = create_app(host, streaming_call=streaming_call,
                         request_deadline_ms=250 if stop == "deadline" else 5000)
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": "POST", "scheme": "http",
                 "path": "/v1/chat/completions", "query_string": b"",
                 "headers": [(b"content-type", b"application/json")]}
        async with asyncio.timeout(3):
            task = asyncio.create_task(app(scope, receive, send))
            try:
                await started.wait()
                if stop == "stream_after_delta":
                    await streamed.wait()
                if stop != "deadline":
                    disconnected.set()
                await task
                assert closed.is_set()
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    if stop == "deadline":
        assert sent[0]["status"] == 504
