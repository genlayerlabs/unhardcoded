"""Buffered generation must finish within its budget even while bytes arrive."""
import asyncio

import httpx
import pytest

from provider_adapters.openai_compatible import make_async_call_provider
from tests.test_antseed_concurrency import _FakeClient, _req
from tests.test_streaming import FakeStreamClient, FakeStreamResponse, _openai_lines


@pytest.mark.asyncio
async def test_trickling_http_body_cannot_extend_generation_deadline():
    closed = asyncio.Event()
    handlers = set()

    async def trickle(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         b"Transfer-Encoding: chunked\r\n\r\n")
            # Every chunk arrives well inside HTTPX's inactivity timeout.
            # Without a wall deadline this never returns a JSON response.
            while True:
                writer.write(b"1\r\n \r\n")
                await writer.drain()
                await asyncio.sleep(0.01)
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            writer.close()
            closed.set()

    server = await asyncio.start_server(trickle, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        request = _req("deadline-trickle", None, timeout_ms=100)
        request["base_url"] = f"http://127.0.0.1:{port}/v1"
        async with httpx.AsyncClient(trust_env=False) as client:
            result = await asyncio.wait_for(make_async_call_provider(client=client)(request), 1)
        assert not result["ok"] and result["error_kind"] == "timeout"
        assert result["latency_ms"] >= 80
        await asyncio.wait_for(closed.wait(), 1)
    finally:
        server.close()
        await server.wait_closed()
        for task in handlers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)


@pytest.mark.asyncio
async def test_deadline_cancels_request_and_releases_peer_slot():
    client = _FakeClient(delay=1)
    call = make_async_call_provider(client=client)
    result = await call(_req("deadline-cleanup", 1, timeout_ms=20))
    assert not result["ok"] and result["error_kind"] == "timeout"
    assert client.in_flight == 0
    client.delay = 0
    assert (await call(_req("deadline-cleanup", 1, timeout_ms=200)))["ok"]


@pytest.mark.asyncio
async def test_buffered_stream_is_bounded_after_its_first_token():
    class SlowTail(FakeStreamResponse):
        closed = False

        async def aiter_lines(self):
            yield _openai_lines("first")[0]
            await asyncio.sleep(1)
            for line in _openai_lines("last"):
                yield line

        async def __aexit__(self, *exc):
            self.closed = True
            return False

    response = SlowTail(200)
    request = _req("deadline-buffered-stream", 1, timeout_ms=20)
    request["first_token_timeout_ms"] = 500
    result = await make_async_call_provider(client=FakeStreamClient(response))(request)
    assert not result["ok"] and result["error_kind"] == "timeout"
    assert response.closed


@pytest.mark.asyncio
async def test_external_cancellation_is_not_reported_as_provider_timeout():
    client = _FakeClient(delay=1)
    call = make_async_call_provider(client=client)
    task = asyncio.create_task(call(_req("deadline-cancel", 1, timeout_ms=1000)))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.in_flight == 0

