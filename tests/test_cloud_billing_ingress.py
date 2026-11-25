from unittest.mock import Mock

import pytest
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


@pytest.mark.parametrize("status,success", [(200, True), (500, False)])
def test_ingress_settles_only_successful_responses(monkeypatch, status, success):
    _upstream(monkeypatch)
    quota = contract()
    resolve_route = cpc.resolve_route

    async def route_with_quota(*args, **kwargs):
        return {**await resolve_route(*args, **kwargs), "billing_quota": quota}

    monkeypatch.setattr(cpc, "resolve_route", route_with_quota)
    monkeypatch.setattr(_FakeUpstreamResp, "status_code", status)
    reserve = Mock(return_value="ticket")
    settle = Mock()
    monkeypatch.setattr(billing_quota, "reserve", reserve)
    monkeypatch.setattr(billing_quota, "settle", settle)
    response = _post_chat(TestClient(auth_proxy.app), "tok-billing")
    assert response.status_code == status
    reserve.assert_called_once_with(7, quota)
    settle.assert_called_once_with("ticket", success)


@pytest.mark.parametrize("chunk,success", [
    (b"data: [DONE]\n\n", True),
    (b'data: {"error":{"message":"upstream failed"}}\n\n', False),
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


def test_free_exhaustion_never_forwards_request(monkeypatch):
    upstream = _upstream(monkeypatch)
    monkeypatch.setattr(billing_quota, "reserve", Mock(return_value=None))
    response = _post_chat(TestClient(auth_proxy.app), "tok-billing")
    assert response.status_code == 429
    assert response.json()["error"]["code"] == "monthly_quota_exceeded"
    assert not upstream.requests


def test_billing_database_outage_never_forwards_request(monkeypatch):
    upstream = _upstream(monkeypatch)
    monkeypatch.setattr(billing_quota, "reserve", Mock(side_effect=RuntimeError("unavailable")))
    response = _post_chat(TestClient(auth_proxy.app), "tok-billing")
    assert response.status_code == 503
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
    response = _post_chat(TestClient(auth_proxy.app), "tok-billing")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "quota_unavailable"
    assert not upstream.requests
