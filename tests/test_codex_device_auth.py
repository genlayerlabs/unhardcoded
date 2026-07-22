from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codex_auth import (  # noqa: E402
    DEVICE_VERIFY_URL,
    DeviceAuthError,
    device_code_exchange,
    device_token_poll,
    device_usercode_request,
)


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _fake_id_token(account_id="acct-42"):
    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJub25lIn0.{b64}.sig"


def test_usercode_request_parses_string_interval():
    calls = []

    def post(url, json=None):
        calls.append((url, json))
        return _Resp(200, {"device_auth_id": "da-1", "user_code": "AB-12", "interval": "5"})

    out = device_usercode_request(http_post=post)
    assert out == {"device_auth_id": "da-1", "user_code": "AB-12", "interval": 5}
    url, body = calls[0]
    assert url.endswith("/api/accounts/deviceauth/usercode")
    assert body == {"client_id": "app_EMoamEEZ73f0CkXaXp7hrann"}
    assert DEVICE_VERIFY_URL == "https://auth.openai.com/codex/device"


def test_usercode_request_failure_raises():
    with pytest.raises(DeviceAuthError):
        device_usercode_request(http_post=lambda url, json=None: _Resp(500))


def test_token_poll_pending_then_success():
    assert device_token_poll("da-1", "AB-12", http_post=lambda u, json=None: _Resp(403)) is None
    assert device_token_poll("da-1", "AB-12", http_post=lambda u, json=None: _Resp(404)) is None
    ok = device_token_poll(
        "da-1", "AB-12",
        http_post=lambda u, json=None: _Resp(200, {
            "authorization_code": "code-1", "code_challenge": "ch", "code_verifier": "ver"}))
    assert ok == {"authorization_code": "code-1", "code_verifier": "ver"}
    with pytest.raises(DeviceAuthError):
        device_token_poll("da-1", "AB-12", http_post=lambda u, json=None: _Resp(500))


def test_exchange_is_form_encoded_and_builds_auth_json():
    seen = {}

    def post_form(url, data=None):
        seen["url"], seen["data"] = url, data
        return _Resp(200, {"access_token": "at-1", "refresh_token": "rt-1",
                           "id_token": _fake_id_token("acct-42")})

    out = device_code_exchange("code-1", "ver", http_post_form=post_form)
    assert seen["url"] == "https://auth.openai.com/oauth/token"
    assert seen["data"] == {
        "grant_type": "authorization_code", "code": "code-1",
        "redirect_uri": "https://auth.openai.com/deviceauth/callback",
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann", "code_verifier": "ver"}
    t = out["tokens"]
    assert t["access_token"] == "at-1" and t["refresh_token"] == "rt-1"
    assert t["account_id"] == "acct-42" and out["last_refresh"]


def test_exchange_failure_and_missing_token_raise():
    with pytest.raises(DeviceAuthError):
        device_code_exchange("c", "v", http_post_form=lambda u, data=None: _Resp(400))
    with pytest.raises(DeviceAuthError):
        device_code_exchange("c", "v",
                             http_post_form=lambda u, data=None: _Resp(200, {"id_token": "x"}))
    # access_token present but id_token missing the account claim -> fail fast
    with pytest.raises(DeviceAuthError):
        device_code_exchange("c", "v",
                             http_post_form=lambda u, data=None: _Resp(200, {"access_token": "at"}))


def test_network_errors_surface_as_device_auth_error():
    def down(url, json=None, data=None):
        raise ConnectionError("boom")

    with pytest.raises(DeviceAuthError):
        device_usercode_request(http_post=down)
    with pytest.raises(DeviceAuthError):
        device_token_poll("da", "UC", http_post=down)
    with pytest.raises(DeviceAuthError):
        device_code_exchange("c", "v", http_post_form=down)
