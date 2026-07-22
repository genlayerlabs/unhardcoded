from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["CALLER_KEYS_JSON"] = '{"internal":"default"}'
os.environ["CALLER_KEYS_SHA256_JSON"] = "{}"
os.environ["DASHBOARD_TRUSTED_USER_HEADER"] = ""

import auth_proxy  # noqa: E402
import codex_auth  # noqa: E402


def _admin(monkeypatch):
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET", "test-dashboard-session-secret")
    client = TestClient(auth_proxy.app)
    client.cookies.set(auth_proxy.DASHBOARD_COOKIE_NAME, auth_proxy._make_dashboard_session("admin"))
    return client


def _wire(monkeypatch, tmp_path):
    accounts_dir = tmp_path / "codex" / "accounts"
    monkeypatch.setattr(auth_proxy, "CODEX_ACCOUNTS_DIR", str(accounts_dir))
    monkeypatch.setattr(auth_proxy, "CODEX_AUTH_PATH", None)

    async def fake_reload():
        return True, None

    monkeypatch.setattr(auth_proxy, "_reload_codex_router", fake_reload)
    return accounts_dir


def test_invites_admin_create_list_revoke(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    assert TestClient(auth_proxy.app).post(
        "/dashboard/api/codex/invites", json={"name": "x"}).status_code == 401

    admin = _admin(monkeypatch)
    r = admin.post("/dashboard/api/codex/invites", json={"name": "Team One!"},
                   headers={"x-forwarded-proto": "https", "x-forwarded-host": "router.example"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["name"] == "team-one"
    assert d["url"].startswith("https://router.example/codex/onboard/")
    token = d["url"].rsplit("/", 1)[1]

    listed = admin.get("/dashboard/api/codex/invites").json()["invites"]
    assert listed[0]["name"] == "team-one" and listed[0]["status"] == "pending"
    assert admin.post("/dashboard/api/codex/invites", json={"name": ""}).status_code == 400

    assert admin.delete(f"/dashboard/api/codex/invites/{token}").status_code == 200
    assert admin.delete(f"/dashboard/api/codex/invites/{token}").status_code == 404


def test_onboard_page_and_dead_link(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    admin = _admin(monkeypatch)
    url = admin.post("/dashboard/api/codex/invites", json={"name": "t1"}).json()["url"]
    token = url.rsplit("/", 1)[1]

    anon = TestClient(auth_proxy.app)
    page = anon.get(f"/codex/onboard/{token}")
    assert page.status_code == 200
    assert "Sign in with ChatGPT" in page.text and "noindex" in page.text
    dead = anon.get("/codex/onboard/not-a-token")
    assert dead.status_code == 404 and "expired" in dead.text.lower()


def test_onboard_full_flow(monkeypatch, tmp_path):
    accounts_dir = _wire(monkeypatch, tmp_path)
    admin = _admin(monkeypatch)
    url = admin.post("/dashboard/api/codex/invites", json={"name": "t1"}).json()["url"]
    token = url.rsplit("/", 1)[1]
    anon = TestClient(auth_proxy.app)

    monkeypatch.setattr(codex_auth, "device_usercode_request", lambda **kw: {
        "device_auth_id": "da-1", "user_code": "AB-12", "interval": 0})
    started = anon.post(f"/codex/onboard/{token}/start")
    assert started.status_code == 200
    assert started.json() == {"user_code": "AB-12",
                              "verification_url": codex_auth.DEVICE_VERIFY_URL,
                              "interval": 0}

    # pending first, then approved
    polls = iter([None, {"authorization_code": "code-1", "code_verifier": "ver"}])
    monkeypatch.setattr(codex_auth, "device_token_poll", lambda *a, **kw: next(polls))
    monkeypatch.setattr(codex_auth, "device_code_exchange", lambda *a, **kw: {
        "tokens": {"access_token": "at", "refresh_token": "rt",
                   "id_token": "id", "account_id": "acct-9"},
        "last_refresh": "2026-07-22T00:00:00+00:00"})

    assert anon.get(f"/codex/onboard/{token}/status").json()["status"] == "awaiting"
    done = anon.get(f"/codex/onboard/{token}/status").json()
    assert done["status"] == "connected" and done["name"] == "t1"

    stored = json.loads((accounts_dir / "t1.json").read_text())
    assert stored["tokens"]["account_id"] == "acct-9"
    # single use: status keeps reporting connected, start refuses
    assert anon.get(f"/codex/onboard/{token}/status").json()["status"] == "connected"
    assert anon.post(f"/codex/onboard/{token}/start").status_code == 409
    # invite list shows used
    assert admin.get("/dashboard/api/codex/invites").json()["invites"][0]["status"] == "used"


def test_onboard_poll_error_resets_device_state(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    admin = _admin(monkeypatch)
    url = admin.post("/dashboard/api/codex/invites", json={"name": "t1"}).json()["url"]
    token = url.rsplit("/", 1)[1]
    anon = TestClient(auth_proxy.app)
    monkeypatch.setattr(codex_auth, "device_usercode_request", lambda **kw: {
        "device_auth_id": "da-1", "user_code": "AB-12", "interval": 0})
    anon.post(f"/codex/onboard/{token}/start")

    def boom(*a, **kw):
        raise codex_auth.DeviceAuthError("device auth poll failed with status 500")

    monkeypatch.setattr(codex_auth, "device_token_poll", boom)
    out = anon.get(f"/codex/onboard/{token}/status").json()
    assert out["status"] == "error" and "500" in out["message"]
    # device state was cleared -> back to pending, teammate can retry
    assert anon.get(f"/codex/onboard/{token}/status").json()["status"] == "pending"
