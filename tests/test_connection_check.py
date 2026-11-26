import asyncio
import json
from pathlib import Path
import time
from unittest.mock import AsyncMock, MagicMock

from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
import httpx
import pytest

import auth_proxy
import connection_check
import control_plane_client as cp
import host_store
from llm_router_host import LLMRouterHost
from shim import create_app

ROOT = Path(__file__).resolve().parents[1]
SECRET = {"x-internal-secret": "internal-test"}
TOKEN = "sample-provider-token"
PATH = "/internal/tenants/7/provider-connections/{}/check"


@pytest.fixture
def router(monkeypatch):
    monkeypatch.setattr(cp, "CONTROL_PLANE_INTERNAL_SECRET", SECRET["x-internal-secret"])
    monkeypatch.setattr(cp, "CONTROL_PLANE_URL", "https://control.invalid")
    credentials = AsyncMock(return_value=({
        "OPENAI_API_KEY": TOKEN, "ANTHROPIC_API_KEY": TOKEN,
        "GEMINI_API_KEY": TOKEN, "OPENROUTER_API_KEY": TOKEN,
    }, {
        "bedrock": {"access_key_id": "tenant-access", "secret_access_key": TOKEN,
                    "session_token": "tenant-session", "region": "eu-west-1"},
        "antseed": {"gateway_url": "https://buyer.invalid", "token": TOKEN},
    }, ()))
    monkeypatch.setattr(cp, "tenant_connections", credentials)
    host = LLMRouterHost(ROOT / "core/router.lua", ROOT / "config.live.lua", env={})
    host.init()
    monkeypatch.setattr(host_store, "reserve_connection_check", MagicMock(return_value=True))
    return TestClient(create_app(host)), host, credentials


def fake_http(monkeypatch, handler):
    original = httpx.AsyncClient
    monkeypatch.setattr(connection_check.httpx, "AsyncClient", lambda **kwargs: original(
        transport=httpx.MockTransport(handler), **kwargs,
    ))


@pytest.mark.parametrize(("provider", "header", "auth_value", "list_key"), [
    ("openai", "authorization", "Bearer " + TOKEN, "data"),
    ("anthropic", "x-api-key", TOKEN, "data"),
    ("gemini", "x-goog-api-key", TOKEN, "models"),
])
def test_checks_runtime_destination_with_tenant_auth(router, monkeypatch, provider, header, auth_value, list_key):
    client, host, credentials = router
    host.set_provider_relays({provider: "https://relay.invalid/v1"})
    before = host.dump_state()

    def upstream(request):
        assert request.method == "GET"
        assert str(request.url) == "https://relay.invalid/v1/models"
        assert request.headers[header] == auth_value
        return httpx.Response(200, json={list_key: [{"id": "test"}]})

    fake_http(monkeypatch, upstream)
    response = client.post(PATH.format(provider), headers=SECRET,
                           params={"project_id": 8, "environment_id": 9})
    assert response.status_code == 200
    result = response.json()
    assert result == {"working": True, "reason": "ok", "message": "Connection works.",
                      "model_count": 1, "elapsed_ms": result["elapsed_ms"]}
    assert result["elapsed_ms"] >= 0
    assert host.dump_state() == before
    assert credentials.await_args.kwargs == {
        "project_id": 8, "environment_id": 9, "require_fresh": True,
    }


@pytest.mark.parametrize(("status", "error", "reason"), [
    (401, {"message": " Invalid API key  "}, "bad_credential"),
    (403, {"code": "token_expired", "message": "token expired"}, "expired"),
    (503, {"message": "service unavailable"}, "unreachable"),
    (429, {"type": "insufficient_quota"}, "unfunded"),
    (429, {"message": "too many requests"}, "rate_limited"),
    (400, {"message": "key prefix sample-provider"}, "unknown"),
])
def test_provider_failures_use_stable_reasons_without_echoing_credentials(
    router, monkeypatch, caplog, status, error, reason,
):
    client, _, _ = router
    fake_http(monkeypatch, lambda _: httpx.Response(status, json={"error": error}))
    response = client.post(PATH.format("openai"), headers=SECRET)
    assert response.status_code == 200
    data = response.json()
    assert not data["working"]
    assert data["reason"] == reason
    assert data["model_count"] is None
    if status == 401:
        assert data["message"] == "Invalid API key"
    assert "sample-provider" not in response.text + caplog.text


def test_google_classifies_structured_key_errors(router, monkeypatch):
    client, _, _ = router
    fake_http(monkeypatch, lambda _: httpx.Response(400, json={"error": {
        "status": "INVALID_ARGUMENT", "details": [{"reason": "API_KEY_INVALID"}],
    }}))
    result = client.post(PATH.format("gemini"), headers=SECRET).json()
    assert result["reason"] == "bad_credential"


def test_falls_back_to_one_token_when_listing_unavailable(router, monkeypatch):
    client, _, _ = router
    methods = []

    def upstream(request):
        methods.append(request.method)
        if request.method == "GET":
            return httpx.Response(405)
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["max_tokens"] == 1
        assert body["model"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hi"}}]})

    fake_http(monkeypatch, upstream)
    result = client.post(PATH.format("openai"), headers=SECRET).json()
    assert result["working"] and result["model_count"] is None
    assert methods == ["GET", "POST"]


def test_public_model_list_does_not_validate_openrouter_credential(router, monkeypatch):
    client, host, _ = router
    host.config.providers["openrouter_market"].base_url = "https://unused.invalid/v1"

    def upstream(request):
        assert str(request.url) == "https://openrouter.ai/api/v1/key"
        assert request.headers["authorization"] == "Bearer " + TOKEN
        return httpx.Response(401)

    fake_http(monkeypatch, upstream)
    result = client.post(PATH.format("openrouter_market"), headers=SECRET).json()
    assert result["reason"] == "bad_credential"


@pytest.mark.parametrize("expired", [False, True])
def test_bedrock_uses_tenant_region_and_credentials(router, monkeypatch, expired):
    client, _, _ = router
    aws = MagicMock()
    listing = aws.return_value.__enter__.return_value.list_foundation_models
    if expired:
        listing.side_effect = ClientError({"Error": {"Code": "ExpiredTokenException"},
                                          "ResponseMetadata": {"HTTPStatusCode": 403}}, "ListFoundationModels")
    else:
        listing.return_value = {"modelSummaries": [{"modelId": "model"}]}
    monkeypatch.setattr("boto3.client", aws)
    result = client.post(PATH.format("bedrock_market"), headers=SECRET).json()
    assert result["reason"] == ("expired" if expired else "ok")
    assert result["model_count"] == (None if expired else 1)
    assert aws.call_args.args == ("bedrock",)
    assert aws.call_args.kwargs["region_name"] == "eu-west-1"
    assert aws.call_args.kwargs["aws_access_key_id"] == "tenant-access"
    assert aws.call_args.kwargs["aws_secret_access_key"] == TOKEN
    assert aws.call_args.kwargs["aws_session_token"] == "tenant-session"


@pytest.mark.parametrize(("funds", "age", "reason"), [
    (3, 0, "ok"), (0, 0, "unfunded"), (3, 900, "unreachable"),
])
def test_buyer_checks_own_snapshot(router, monkeypatch, funds, age, reason):
    client, _, _ = router
    now_ms = time.time() * 1000

    def upstream(request):
        assert str(request.url) == "https://buyer.invalid/snapshot"
        assert request.headers["authorization"] == "Bearer " + TOKEN
        return httpx.Response(200, json={
            "buyer_status": {"deposits_available": funds, "fetched_at": now_ms - age * 1000},
            "peer_offers": [{"service": "model", "observed_at": now_ms}] * 2,
        })

    monkeypatch.setattr("byo_http.buyer_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(upstream)))
    result = client.post(PATH.format("antseed"), headers=SECRET).json()
    assert result["reason"] == reason
    assert result["model_count"] == (1 if reason == "ok" else None)


def test_timeout_and_unavailable_tenant_configuration(router, monkeypatch):
    client, _, credentials = router

    async def upstream(_):
        await asyncio.sleep(10)

    fake_http(monkeypatch, upstream)
    monkeypatch.setattr(connection_check, "TIMEOUT", 0.05)
    response = client.post(PATH.format("openai"), headers=SECRET)
    assert response.json()["reason"] == "unreachable"
    assert response.json()["elapsed_ms"] < 1000
    credentials.side_effect = cp.RouteUnavailable("sensitive upstream information")
    assert client.post(PATH.format("openai"), headers=SECRET).json()["reason"] == "unreachable"


@pytest.mark.asyncio
async def test_endpoint_auth_validation_and_rate_limit(router, monkeypatch):
    client, _, credentials = router
    path = PATH.format("openai")
    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app)) as shim_client,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=auth_proxy.app),
                          base_url="http://ingress.test") as ingress,
    ):
        monkeypatch.setattr(auth_proxy, "_connection_check_client", shim_client)
        assert client.post(path).status_code == 403
        assert (await ingress.post(path)).status_code == 403
        assert client.post(path, headers=SECRET, params={"project_id": 8}).status_code == 400
        assert (await ingress.post(path, headers=SECRET, params={"project_id": 8})).status_code == 400
        host_store.reserve_connection_check.return_value = False
        limited = await ingress.post(path, headers=SECRET, params={"connection_ref": "saved"})
        assert limited.status_code == 429
        assert limited.headers["retry-after"] == str(connection_check.INTERVAL)
        host_store.reserve_connection_check.assert_called_once_with(7, connection_check.INTERVAL)
        credentials.assert_not_awaited()
        monkeypatch.setattr(cp, "CONTROL_PLANE_INTERNAL_SECRET", "")
        assert (await ingress.post(path, headers=SECRET)).status_code == 404


def test_store_failure_prevents_probe(router):
    client, _, credentials = router
    host_store.reserve_connection_check.side_effect = RuntimeError()
    assert client.post(PATH.format("openai"), headers=SECRET).status_code == 503
    credentials.assert_not_awaited()


@pytest.mark.parametrize("grants, reason", [
    (("openai",), "ok"),
    (("openai", "anthropic"), "unreachable"),
    ((), "bad_credential"),
])
def test_scoped_check_obeys_managed_grants(router, monkeypatch, grants, reason):
    client, host, credentials = router
    monkeypatch.setenv("SAAS_SHARED_PROVIDERS", "openai")
    host.set_env("OPENAI_API_KEY", TOKEN)
    host.set_provider_relays({"openai": "https://relay.invalid/v1"})
    credentials.return_value = ({}, {}, grants)
    seen = []

    def upstream(request):
        seen.append(str(request.url))
        assert request.headers["authorization"] == "Bearer " + TOKEN
        return httpx.Response(200, json={"data": [{"id": "test"}]})

    fake_http(monkeypatch, upstream)
    result = client.post(PATH.format("openai"), headers=SECRET,
                         params={"project_id": 8, "environment_id": 9}).json()

    assert result["reason"] == reason
    assert result["working"] is (reason == "ok")
    assert seen == (["https://relay.invalid/v1/models"] if reason == "ok" else [])
    credentials.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(("identity_override", "status", "reason"), [
    ({}, 200, "ok"),
    ({"tenant_id": 8}, 200, "unreachable"),
    ({"provider_id": "anthropic"}, 200, "unreachable"),
    ({"connection_ref": "different"}, 200, "unreachable"),
    ({}, 404, "unreachable"),
    ({"env": {}, "managed_providers": ["openai"]}, 200, "bad_credential"),
])
async def test_saved_connection_uses_only_exact_credentials(
    router, monkeypatch, identity_override, status, reason,
):
    client, host, legacy_credentials = router
    monkeypatch.setenv("SAAS_SHARED_PROVIDERS", "openai")
    host.set_env("OPENAI_API_KEY", TOKEN)
    host.set_provider_relays({"openai": "https://relay.invalid/v1"})
    identity = {"provider_id": "openai", "connection_ref": "saved_ABC-42"}
    control_requests = []
    provider_requests = []

    def control(request):
        control_requests.append(request)
        return httpx.Response(status, json={
            "tenant_id": 7, **identity,
            "env": {"OPENAI_API_KEY": "selected-token"},
            "connections": {"antseed": {"gateway_url": "https://ignored.invalid", "token": TOKEN}},
            "base_url": "https://ignored.invalid",
            **identity_override,
        })

    def upstream(request):
        provider_requests.append(request)
        return httpx.Response(200, json={"data": []})

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(control)) as control_client,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app)) as shim_client,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=auth_proxy.app),
                          base_url="http://ingress.test") as ingress,
    ):
        monkeypatch.setattr(cp, "_client", control_client)
        monkeypatch.setattr(auth_proxy, "_connection_check_client", shim_client)
        fake_http(monkeypatch, upstream)
        response = await ingress.post(PATH.format("openai"), headers=SECRET,
                                      params={"connection_ref": identity["connection_ref"]})

    assert response.status_code == 200
    assert response.json()["reason"] == reason
    assert "selected-token" not in response.text
    legacy_credentials.assert_not_awaited()
    request, = control_requests
    assert request.method == "POST"
    assert request.url.path == "/internal/tenants/7/provider-connections/probe-credentials"
    assert request.headers["x-internal-secret"] == SECRET["x-internal-secret"]
    assert json.loads(request.content) == identity
    if reason == "ok":
        provider_request, = provider_requests
        assert str(provider_request.url) == "https://relay.invalid/v1/models"
        assert provider_request.headers["authorization"] == "Bearer selected-token"
    else:
        assert provider_requests == []


@pytest.mark.parametrize("params", [
    [("connection_ref", "")],
    [("connection_ref", "a/b")],
    [("connection_ref", "a" * 129)],
    [("connection_ref", "one"), ("connection_ref", "two")],
    [("connection_ref", "one"), ("project_id", "8"), ("environment_id", "9")],
])
def test_invalid_saved_connection_scope_rejected_before_reservation(router, params):
    client, _, _ = router
    assert client.post(PATH.format("openai"), headers=SECRET, params=params).status_code == 400
    host_store.reserve_connection_check.assert_not_called()
