from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
import auth_proxy
import billing_quota
import control_plane_client as cpc
from test_auth_proxy_control_plane import _cp, _upstream, _post_chat, _FakeUpstreamResp


@pytest.fixture(autouse=True)
def cloud(monkeypatch):
    cpc.reset_for_tests()
    monkeypatch.setattr(cpc, 'CONTROL_PLANE_URL', 'http://cp.test')
    monkeypatch.setattr(cpc, 'CONTROL_PLANE_INTERNAL_SECRET', 's3cret')
    monkeypatch.setenv('CLOUD_MONTHLY_QUOTAS_REQUIRED', '1')
    _cp(monkeypatch, [{'active':True,'consumer':'billing-test','tenant_id':7}])
    yield
    cpc.reset_for_tests()


@pytest.mark.parametrize('status,success', [(200,True),(500,False)])
def test_ingress_settles_only_successful_responses(monkeypatch,status,success):
    _upstream(monkeypatch)
    monkeypatch.setattr(_FakeUpstreamResp,'status_code',status)
    reserve=Mock(return_value='ticket');settle=Mock()
    monkeypatch.setattr(billing_quota,'reserve',reserve)
    monkeypatch.setattr(billing_quota,'settle',settle)
    response=_post_chat(TestClient(auth_proxy.app),'tok-billing')
    assert response.status_code==status
    reserve.assert_called_once()
    settle.assert_called_once_with('ticket',success)


def test_free_exhaustion_never_forwards_request(monkeypatch):
    upstream=_upstream(monkeypatch)
    monkeypatch.setattr(billing_quota,'reserve',Mock(return_value=None))
    response=_post_chat(TestClient(auth_proxy.app),'tok-billing')
    assert response.status_code==429
    assert response.json()['error']['code']=='monthly_quota_exceeded'
    assert not upstream.requests


def test_billing_database_outage_never_forwards_request(monkeypatch):
    upstream=_upstream(monkeypatch)
    monkeypatch.setattr(billing_quota,'reserve',Mock(side_effect=RuntimeError('unavailable')))
    response=_post_chat(TestClient(auth_proxy.app),'tok-billing')
    assert response.status_code==503
    assert not upstream.requests
