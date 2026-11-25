import asyncio
from unittest.mock import Mock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import auth_proxy
import billing_quota
import control_plane_client as cpc
from test_auth_proxy_control_plane import _cp, _upstream, _post_chat, _FakeUpstreamResp
from test_cloud_billing_quota import contract


@pytest.fixture(autouse=True)
def cloud(monkeypatch, host_store_clean):
    cpc.reset_for_tests()
    monkeypatch.setattr(cpc, "CONTROL_PLANE_URL", "http://cp.test")
    monkeypatch.setattr(cpc, "CONTROL_PLANE_INTERNAL_SECRET", "s3cret")
    monkeypatch.setenv("CLOUD_MONTHLY_QUOTAS_REQUIRED", "1")
    _cp(monkeypatch, [{"active": True, "consumer": "billing-test", "tenant_id": 7}])
    yield
    cpc.reset_for_tests()


@pytest.fixture
def route_quota(monkeypatch):
    quota = contract()
    resolve_route = cpc.resolve_route

    async def route_with_quota(*args, **kwargs):
        return {**await resolve_route(*args, **kwargs), "billing_quota": quota}

    monkeypatch.setattr(cpc, "resolve_route", route_with_quota)
    return quota


@pytest.mark.parametrize("status,success", [(200, True), (500, False)])
def test_ingress_settles_only_successful_responses(monkeypatch, route_quota, status, success):
    _upstream(monkeypatch)
    monkeypatch.setattr(_FakeUpstreamResp, "status_code", status)
    reserve = Mock(return_value="ticket")
    settle = Mock()
    monkeypatch.setattr(billing_quota, "reserve", reserve)
    monkeypatch.setattr(billing_quota, "settle", settle)
    response = _post_chat(TestClient(auth_proxy.app), "tok-billing")
    assert response.status_code == status
    reserve.assert_called_once_with(7, route_quota)
    settle.assert_called_once_with("ticket", success)


@pytest.mark.parametrize("chunk,success", [
    (b"data: [DONE]\n\n", True),
    (b'data: {"error":{"message":"upstream failed"}}\n\n', False),
    (b'data: {"type":"response.failed","response":{"status":"failed"}}\n\n', False),
])
def test_stream_settles_after_upstream_finishes(monkeypatch, chunk, success):
    _upstream(monkeypatch)
    settle = Mock()

    async def stream(self):
        settle.assert_not_called()
        yield chunk

    monkeypatch.setattr(_FakeUpstreamResp, "headers", {"content-type": "text/event-stream"})
    monkeypatch.setattr(_FakeUpstreamResp, "aiter_raw", stream, raising=False)
    monkeypatch.setattr(billing_quota, "reserve", Mock(return_value="ticket"))
    monkeypatch.setattr(billing_quota, "settle", settle)
    response = _post_chat(TestClient(auth_proxy.app), "tok-billing")
    assert response.content == chunk
    settle.assert_called_once_with("ticket", success)


@pytest.mark.asyncio
@pytest.mark.parametrize("drain_outcome,success", [("bytes", True), ("timeout", True), ("error", False)])
async def test_disconnected_stream_settles_after_bounded_drain(monkeypatch, drain_outcome, success):
    _upstream(monkeypatch)
    client_left = asyncio.Event()
    settle = Mock()
    chunk = b'data: {"choices":[]}\n\n'

    async def stream(self):
        yield chunk
        await client_left.wait()
        if drain_outcome == "timeout":
            await asyncio.Future()
        if drain_outcome == "error":
            yield b'data: {"error":{"message":"upstream failed"}}\n\n'
        else:
            yield chunk
        pytest.fail("The drain byte cap should stop the pump")

    async def receive():
        return {"type": "http.request", "body": b'{"model":"route:production","messages":[]}'}

    monkeypatch.setattr(auth_proxy, "STREAM_DRAIN_TIMEOUT_S", 0.15)
    monkeypatch.setattr(auth_proxy, "STREAM_DRAIN_MAX_BYTES", 1)
    monkeypatch.setattr(_FakeUpstreamResp, "headers", {"content-type": "text/event-stream"})
    monkeypatch.setattr(_FakeUpstreamResp, "aiter_raw", stream, raising=False)
    monkeypatch.setattr(billing_quota, "reserve", Mock(return_value="ticket"))
    monkeypatch.setattr(billing_quota, "settle", settle)
    path = "/v1/chat/completions"
    request = Request({"type": "http", "path": path, "method": "POST", "query_string": b"",
                       "headers": [(b"authorization", b"Bearer tok-billing")]}, receive)
    response = await auth_proxy.proxy(path.lstrip("/"), request)
    assert await anext(response.body_iterator) == chunk
    settle.assert_not_called()
    await response.body_iterator.aclose()
    client_left.set()
    await asyncio.wait_for(asyncio.gather(*auth_proxy._stream_pumps), timeout=2)
    settle.assert_called_once_with("ticket", success)


def test_free_exhaustion_never_forwards_request(monkeypatch, route_quota):
    upstream = _upstream(monkeypatch)
    route_quota["limit"] = 0
    response = _post_chat(TestClient(auth_proxy.app), "tok-billing")
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "monthly_quota_exceeded"
    assert response.json()["error"]["type"] == "rate_limit_error"
    assert 0 < int(response.headers["Retry-After"]) <= 1000
    assert not upstream.requests


def test_billing_database_outage_never_forwards_request(monkeypatch):
    upstream = _upstream(monkeypatch)
    metric = Mock()
    monkeypatch.setattr(auth_proxy, "_metric_store_error", metric)
    monkeypatch.setattr(billing_quota, "reserve", Mock(side_effect=RuntimeError("unavailable")))
    response = _post_chat(TestClient(auth_proxy.app), "tok-billing")
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"
    metric.assert_called_once_with("billing_quota")
    assert not upstream.requests


def test_flag_off_allows_tenant_without_quota(monkeypatch):
    monkeypatch.setenv("CLOUD_MONTHLY_QUOTAS_REQUIRED", "0")
    upstream = _upstream(monkeypatch)
    reserve = Mock()
    monkeypatch.setattr(billing_quota, "reserve", reserve)
    response = _post_chat(TestClient(auth_proxy.app), "tok-billing")
    assert response.status_code == 200
    assert len(upstream.requests) == 1
    reserve.assert_not_called()


def test_flag_on_requires_quota(monkeypatch):
    upstream = _upstream(monkeypatch)
    metric = Mock()
    monkeypatch.setattr(auth_proxy, "_metric_store_error", metric)
    response = _post_chat(TestClient(auth_proxy.app), "tok-billing")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "quota_unavailable"
    metric.assert_not_called()
    assert not upstream.requests


def test_flag_on_allows_operator_key_without_tenant(monkeypatch):
    upstream = _upstream(monkeypatch)
    reserve = Mock()
    monkeypatch.setattr(auth_proxy, "CALLER_KEYS", {"tok-local": "operator-app"})
    monkeypatch.setattr(billing_quota, "reserve", reserve)
    response = _post_chat(TestClient(auth_proxy.app), "tok-local")
    assert response.status_code == 200
    assert len(upstream.requests) == 1
    reserve.assert_not_called()
