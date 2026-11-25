"""Run the chain-free router demo with ESLA_BIN pointing to the SDK executable."""
import asyncio
import json
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from esla_routing import InvalidUpstreamPath, load_routes, upstream_url
from llm_router_host import LLMRouterHost, make_async_call_provider
from provider_adapters.openai_compatible import stream_openai_compatible
from shim import create_app

ROOT = Path(__file__).resolve().parents[1]
AGREEMENT = "0x00000000000000000000000000000000000e51a1"


def route_config(caller):
    return json.dumps({"comput3": {"agreement": AGREEMENT, "sidecar_url": f"{caller}/bonded"}})


def test_esla_route_configuration():
    url = "https://provider.example/v1beta/models/gemini:streamGenerateContent?alt=sse"
    caller = "http://127.0.0.1:8081"
    routes = load_routes(route_config(caller))
    assert upstream_url({}, url) == url
    assert upstream_url({"_esla_sidecar_url": routes["comput3"]}, url) == (
        f"{caller}/bonded/v1beta/models/gemini:streamGenerateContent?alt=sse"
    )
    with pytest.raises(InvalidUpstreamPath):
        upstream_url({"_esla_sidecar_url": routes["comput3"]}, "https://provider.example/%2e%2e/other")
    with pytest.raises(ValueError, match="ESLA_ROUTES"):
        load_routes(route_config("http://secret@localhost"))


def local_address():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{sock.getsockname()[1]}"


@contextmanager
def service(binary, address, *args):
    with subprocess.Popen([binary, *args, "-listen", address],
                          stderr=subprocess.PIPE, text=True) as process:
        try:
            deadline = time.monotonic() + 5
            while True:
                if process.poll() is not None:
                    pytest.fail(process.stderr.read())
                try:
                    with socket.create_connection(("127.0.0.1", int(address.split(":")[1])), .1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        pytest.fail(f"{args[0]} did not start on {address}")
                    time.sleep(.02)
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


@pytest.fixture
def esla_demo(tmp_path):
    binary = os.getenv("ESLA_BIN")
    if not binary:
        pytest.skip("Set ESLA_BIN to run the chain-free sidecar demo")
    binary = str(Path(binary).resolve())

    def cli(*args):
        return subprocess.check_output([binary, *args], text=True, timeout=15).strip()

    client_key = str(tmp_path / "client.key")
    provider_key = str(tmp_path / "provider.key")
    agreement = tmp_path / "agreement.json"
    agreement.write_text(json.dumps({
        "address": AGREEMENT, "client_key": cli("keygen", "-out", client_key),
        "provider_key": cli("keygen", "-out", provider_key),
        "start": int(time.time()) - 5, "window_seconds": 3600,
        "commit_grace_seconds": 600, "slow_ms": 10000,
    }))
    upstream, proxy, caller = (local_address() for _ in range(3))
    routes = tmp_path / "routes.json"
    routes.write_text(json.dumps([{
        "name": "bonded", "agreement": str(agreement), "provider": f"http://{proxy}",
    }]))
    caller_db = str(tmp_path / "caller.db")
    provider_db = str(tmp_path / "provider.db")

    def commitment(role, key, db):
        return dict(line.split(None, 1) for line in cli(
            "commit", "-role", role, "-agreement", str(agreement), "-key", key,
            "-db", db, "-window", "0", "-dry-run",
        ).splitlines())

    with service(binary, proxy, "provider", "-agreement", str(agreement), "-key",
                 provider_key, "-db", provider_db, "-upstream", f"http://{upstream}"):
        with service(binary, caller, "caller", "-routes", str(routes), "-key",
                     client_key, "-db", caller_db):
            yield (
                f"http://{caller}",
                lambda errors: service(binary, upstream, "fake-upstream", "-error-rate", str(errors)),
                lambda: (commitment("caller", client_key, caller_db),
                         commitment("provider", provider_key, provider_db)),
            )


@pytest.mark.asyncio
async def test_esla_window_matches_router_attempts(esla_demo, host_store_clean):
    caller, fake_upstream, commitments = esla_demo
    env = {"ESLA_ROUTES": route_config(caller), "COMPUT3_API_KEY": "test"}
    host = LLMRouterHost(ROOT / "core/router.lua", ROOT / "core/config.example.lua",
                         env=env, call_provider_async=make_async_call_provider(env.get))
    host.init()

    async def stream(request, emit):
        return await stream_openai_compatible(request, emit, env_get=env.get)

    app = create_app(host, streaming_call=stream)
    body = {"model": "pin:comput3/hermes-3-405b",
            "messages": [{"role": "user", "content": "say hello"}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://router") as client:
        with fake_upstream(0):
            response = await client.post("/v1/chat/completions", json=body)
            assert response.status_code == 200, response.text
            assert "Bonded providers" in response.json()["choices"][0]["message"]["content"]
            response = await client.post("/v1/chat/completions", json={**body, "stream": True})
            assert response.status_code == 200, response.text
            assert "Bonded " in response.text and "data: [DONE]" in response.text
        with fake_upstream(1):
            response = await client.post("/v1/chat/completions", json=body)
            assert response.status_code >= 500, response.text

    with host_store_clean._get_pool().connection() as conn:
        calls, errors, slow = conn.execute(
            "SELECT count(*), count(*) FILTER (WHERE NOT ok), "
            "count(*) FILTER (WHERE latency_ms > 10000) FROM route_observations "
            "WHERE provider_id = 'comput3'"
        ).fetchone()
    expected = {"calls": calls, "errors": errors, "slow": slow}
    assert calls >= 3 and calls - errors == 2

    # The caller seals and acknowledges asynchronously after forwarding the response.
    deadline = time.monotonic() + 5
    while True:
        caller_window, provider_window = commitments()
        if (caller_window["pending"] == "0"
                and caller_window["root"] == provider_window["root"]):
            break
        if time.monotonic() >= deadline:
            pytest.fail(f"window did not settle: {caller_window}, {provider_window}")
        await asyncio.sleep(.05)
    assert json.loads(caller_window["summary"]) == expected
    assert json.loads(provider_window["summary"]) == expected


def configured_host(http, env):
    host = LLMRouterHost(ROOT / "core/router.lua", ROOT / "core/config.example.lua",
                         env=env, call_provider_async=make_async_call_provider(env.get, client=http))
    host.init()
    return host


@pytest.mark.asyncio
async def test_sidecar_outage_falls_back_without_calling_mapped_provider(monkeypatch):
    seen = []

    async def send(request):
        seen.append(request.url)
        if request.url.host == "caller":
            raise httpx.ConnectError("caller unavailable", request=request)
        assert request.url.host == "api.mor.org"
        return httpx.Response(200, json={"choices": [{"message": {"content": "fallback"}}]})

    env = {"COMPUT3_API_KEY": "test", "MORPHEUS_API_KEY": "test",
           "ESLA_ROUTES": route_config("http://caller")}
    monkeypatch.setattr("llm_router_host._fold_route_outcome", lambda *args, **kwargs: None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as http:
        host = configured_host(http, env)
        result = await host.execute_async({
            "prompt": "hello",
            "requirements": {"model_family": "deepseek-v3"},
            "policy_ir": ["policy", ["meets_req"], ["field", "context"],
                          ["prefer", ["provider_eq", "comput3"], ["argmax"]],
                          ["id"], ["always", {"action": "next_candidate"}]],
        })
    assert result["ok"] and result["response"]["text"] == "fallback"
    assert seen[0] == httpx.URL("http://caller/bonded/v1/chat/completions")
    assert seen[-1].host == "api.mor.org"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stop", ["disconnect", "deadline"])
async def test_cancel_mapped_attempt_before_response(stream, stop, monkeypatch):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def send(request):
        assert request.url.host == "caller"
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    env = {"COMPUT3_API_KEY": "test", "ESLA_ROUTES": route_config("http://caller")}
    monkeypatch.setattr("llm_router_host._fold_route_outcome", lambda *args, **kwargs: None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as http:
        host = configured_host(http, env)

        async def stream_call(request, emit):
            return await stream_openai_compatible(request, emit, env_get=env.get, client=http)

        app = create_app(host, streaming_call=stream_call,
                         request_deadline_ms=200 if stop == "deadline" else 5000)
        if stop == "disconnect":
            # Unlike ASGITransport, this client can send http.disconnect while
            # the handler waits for its first upstream byte.
            received = False

            async def receive():
                nonlocal received
                if not received:
                    received = True
                    return {"type": "http.request", "body": request_body}
                await entered.wait()
                return {"type": "http.disconnect"}

            async def ignore_response(message):
                pass

            request_body = json.dumps({
                "model": "pin:comput3/hermes-3-405b",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            }).encode()
            await asyncio.wait_for(app({
                "type": "http", "method": "POST", "path": "/v1/chat/completions",
                "headers": [(b"content-type", b"application/json")], "query_string": b"",
            }, receive, ignore_response), timeout=2)
        else:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                         base_url="http://router") as client:
                response = await client.post("/v1/chat/completions", json={
                    "model": "pin:comput3/hermes-3-405b",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": stream,
                })
                assert response.status_code == 504
        assert cancelled.is_set()
