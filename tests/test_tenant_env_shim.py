"""Per-tenant BYO provider credentials through the router (shim + adapters).

Covers the full chain: the trusted x-llm-router-tenant header -> _activate_tenant
-> ContextVar -> control_plane_client.env_get inside the provider call, including
task-context propagation (streaming/timeout paths create tasks) and isolation
between concurrent tenants. The control plane is faked at the HTTP boundary.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import control_plane_client as cpc  # noqa: E402
from llm_router_host import LLMRouterHost  # noqa: E402
from shim import create_app  # noqa: E402


class _FakeCPResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class _FakeCPClient:
    def __init__(self, env_by_tenant):
        self.env_by_tenant = env_by_tenant
        self.calls: list[str] = []

    async def get(self, url, params=None, headers=None):
        self.calls.append(url)
        tenant_id = int(url.rstrip("/").split("/")[-2])
        return _FakeCPResp({"env": self.env_by_tenant.get(tenant_id, {})})


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    cpc.reset_for_tests()
    monkeypatch.setattr(cpc, "CONTROL_PLANE_URL", "http://cp.test")
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "s3cret")
    yield
    cpc.reset_for_tests()


@pytest.fixture
def host():
    h = LLMRouterHost(
        router_path=ROOT / "core" / "router.lua",
        config_path=ROOT / "core" / "config.example.lua",
        metrics_path=ROOT / "core" / "metrics.example.lua",
        now_ms=lambda: 1_000_000,
        # The custom call hook below would otherwise turn on auth enforcement
        # and pre-disable every example provider (their env keys are unset here).
        enforce_provider_auth=False,
    )
    h.init()
    return h


def _ok_result():
    return {"ok": True, "latency_ms": 10,
            "response": {"text": "hi", "tool_calls": None, "finish_reason": "stop",
                         "tokens_in": 7, "tokens_out": 3, "tokens_total": 10,
                         "raw_model": "mock-model-id"}}


def _capture_hook(seen: list):
    """An async provider hook that records what env_get resolves AT CALL TIME —
    i.e. inside host.execute_async, past any create_task boundaries."""

    async def hook(request: dict) -> dict:
        await asyncio.sleep(0.005)   # let concurrent requests interleave
        seen.append(cpc.env_get("OPENAI_API_KEY"))
        return _ok_result()

    return hook


def _chat(client, tenant: int | None):
    headers = {"x-llm-router-tenant": str(tenant)} if tenant is not None else {}
    return client.post("/v1/chat/completions", headers=headers,
                       json={"model": "profile:default",
                             "messages": [{"role": "user", "content": "hi"}]})


def test_tenant_header_activates_byo_key(host, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    monkeypatch.setattr(cpc, "_client", _FakeCPClient({7: {"OPENAI_API_KEY": "sk-tenant"}}))
    seen: list = []
    host.set_async_call_hook(_capture_hook(seen))
    client = TestClient(create_app(host, default_profile="default"))
    assert _chat(client, tenant=7).status_code == 200
    assert seen == ["sk-tenant"]


def test_no_header_uses_platform_key(host, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    fake = _FakeCPClient({})
    monkeypatch.setattr(cpc, "_client", fake)
    seen: list = []
    host.set_async_call_hook(_capture_hook(seen))
    client = TestClient(create_app(host, default_profile="default"))
    assert _chat(client, tenant=None).status_code == 200
    assert seen == ["sk-platform"]
    assert fake.calls == []


def test_feature_off_ignores_header(host, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    monkeypatch.setattr(cpc, "CONTROL_PLANE_URL", "")
    fake = _FakeCPClient({7: {"OPENAI_API_KEY": "sk-tenant"}})
    monkeypatch.setattr(cpc, "_client", fake)
    seen: list = []
    host.set_async_call_hook(_capture_hook(seen))
    client = TestClient(create_app(host, default_profile="default"))
    assert _chat(client, tenant=7).status_code == 200
    assert seen == ["sk-platform"]
    assert fake.calls == []


def test_concurrent_tenants_are_isolated(host, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    monkeypatch.setattr(cpc, "_client", _FakeCPClient({
        1: {"OPENAI_API_KEY": "sk-one"}, 2: {"OPENAI_API_KEY": "sk-two"}}))
    seen: list = []
    host.set_async_call_hook(_capture_hook(seen))
    app = create_app(host, default_profile="default")

    # Drive the ASGI app directly so both requests share one event loop and
    # genuinely interleave inside the provider hook.
    import httpx

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            body = {"model": "profile:default",
                    "messages": [{"role": "user", "content": "hi"}]}
            r1, r2, r3 = await asyncio.gather(
                c.post("/v1/chat/completions", json=body,
                       headers={"x-llm-router-tenant": "1"}),
                c.post("/v1/chat/completions", json=body,
                       headers={"x-llm-router-tenant": "2"}),
                c.post("/v1/chat/completions", json=body),
            )
            assert r1.status_code == r2.status_code == r3.status_code == 200

    asyncio.run(run())
    assert sorted(seen, key=str) == ["sk-one", "sk-platform", "sk-two"]


def test_streaming_request_carries_tenant_env(host, monkeypatch):
    """stream:true goes through asyncio.create_task in the shim — the context
    must propagate into the task."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    monkeypatch.setattr(cpc, "_client", _FakeCPClient({7: {"OPENAI_API_KEY": "sk-tenant"}}))
    seen: list = []

    async def streaming_call(request, emit):
        seen.append(cpc.env_get("OPENAI_API_KEY"))
        emit({"delta": "hi"})
        return _ok_result()

    host.set_async_call_hook(_capture_hook(seen))
    client = TestClient(create_app(host, default_profile="default",
                                   streaming_call=streaming_call))
    r = client.post("/v1/chat/completions",
                    headers={"x-llm-router-tenant": "7"},
                    json={"model": "profile:default", "stream": True,
                          "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert "sk-tenant" in seen


def test_adapter_authorization_header_uses_tenant_key(monkeypatch):
    """End of the chain: the OpenAI-compatible adapter builds Authorization from
    control_plane_client.env_get, so an active tenant map changes the wire key."""
    from provider_adapters.openai_compatible import make_async_call_provider

    captured: dict = {}

    class _FakeHTTPResp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "ok"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2}}

    class _FakeHTTPClient:
        async def post(self, url, json=None, headers=None, timeout=None):
            captured["headers"] = headers
            return _FakeHTTPResp()

    call = make_async_call_provider(env_get=cpc.env_get, client=_FakeHTTPClient())
    request = {"provider_id": "openai", "served_model_id": "gpt-x",
               "base_url": "https://api.test/v1",
               "auth": {"kind": "bearer", "env": "OPENAI_API_KEY"},
               "messages": [{"role": "user", "content": "hi"}]}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-platform")
    token = cpc.activate_tenant_env({"OPENAI_API_KEY": "sk-tenant"})
    try:
        result = asyncio.run(call(dict(request)))
    finally:
        cpc.reset_tenant_env(token)
    assert result["ok"] is True
    assert captured["headers"]["Authorization"] == "Bearer sk-tenant"

    result = asyncio.run(call(dict(request)))
    assert captured["headers"]["Authorization"] == "Bearer sk-platform"
