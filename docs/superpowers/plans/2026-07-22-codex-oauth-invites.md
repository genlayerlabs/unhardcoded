# Codex OAuth Invite Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dashboard-generated single-use invite links that let a teammate OAuth their ChatGPT account into the router via OpenAI's device-code flow — no CLI, no pasting auth.json.

**Architecture:** A sync device-flow client in `codex_auth.py` (usercode → poll → form-encoded code exchange → auth.json dict); a file-backed `CodexInviteStore` in a new `codex_invites.py` (`{CODEX_ACCOUNTS_DIR}/_invites.json`); admin invite endpoints + public token-gated onboarding page/endpoints in `auth_proxy.py`, reusing the existing `_codex_store().add_account` + `/x/codex/reload` path; small JS/HTML additions to the dashboard codex panel.

**Tech Stack:** Python 3 / FastAPI / httpx (sync calls via `asyncio.to_thread`), pytest + fastapi TestClient, vanilla-JS inline dashboard.

**Spec:** `docs/superpowers/specs/2026-07-22-codex-oauth-invite-design.md`

## Global Constraints

- Branch: `feat/codex-oauth-invites`; every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Invite file MUST be `_invites.json` (underscore prefix — `CodexAuthStore.reload()` treats non-underscore `*.json` as accounts).
- OAuth code exchange MUST be form-encoded (`data=`, not `json=`); refresh stays JSON as today.
- `redirect_uri` for the exchange: `https://auth.openai.com/deviceauth/callback`.
- Device endpoints: `https://auth.openai.com/api/accounts/deviceauth/{usercode,token}`; user verification page `https://auth.openai.com/codex/device`; `interval` arrives as a **string**; device codes expire after 15 minutes.
- Never log or return raw tokens; `_invites.json` never contains tokens.
- Onboard page: `noindex,nofollow` robots meta + transparency copy about OpenAI's "a website gave you this code" warning.
- Existing tests must keep passing: `python -m pytest tests/ -x -q`.

---

### Task 1: Device-flow client in `codex_auth.py`

**Files:**
- Modify: `codex_auth.py` (add constants + 4 functions + `_jwt_payload` refactor at the bottom near `_jwt_exp`)
- Test: `tests/test_codex_device_auth.py` (new)

**Interfaces:**
- Produces: `device_usercode_request(*, http_post=None, client_id=CODEX_CLIENT_ID) -> dict` returning `{"device_auth_id": str, "user_code": str, "interval": int}`; `device_token_poll(device_auth_id, user_code, *, http_post=None) -> dict | None` (None = pending, dict has `authorization_code`/`code_verifier`, raises `DeviceAuthError` on fatal); `device_code_exchange(authorization_code, code_verifier, *, http_post_form=None, client_id=CODEX_CLIENT_ID) -> dict` returning an auth.json-shaped dict; `DEVICE_VERIFY_URL`; `DeviceAuthError`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_codex_device_auth.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_codex_device_auth.py -q`
Expected: ImportError (`DeviceAuthError` etc. not defined).

- [ ] **Step 3: Implement in `codex_auth.py`**

Refactor `_jwt_exp` to share a `_jwt_payload` helper, and add below the existing constants:

```python
DEVICE_AUTH_BASE_URL = "https://auth.openai.com/api/accounts/deviceauth"
DEVICE_VERIFY_URL = "https://auth.openai.com/codex/device"
DEVICE_CALLBACK_URL = "https://auth.openai.com/deviceauth/callback"


class DeviceAuthError(RuntimeError):
    """Fatal failure in the device-code flow (not the pending state)."""
```

New functions (module level, near the bottom with the other helpers):

```python
def device_usercode_request(*, http_post=None, client_id: str = CODEX_CLIENT_ID) -> dict:
    """Step 1 of the device flow: mint a one-time user code. Mirrors
    `codex login --device-auth` (POST /api/accounts/deviceauth/usercode)."""
    post = http_post or _default_http_post
    resp = post(f"{DEVICE_AUTH_BASE_URL}/usercode", json={"client_id": client_id})
    status = getattr(resp, "status_code", 0)
    if status != 200:
        raise DeviceAuthError(f"device usercode request failed with status {status}")
    data = resp.json()
    device_auth_id = data.get("device_auth_id")
    user_code = data.get("user_code") or data.get("usercode")
    if not device_auth_id or not user_code:
        raise DeviceAuthError("device usercode response missing device_auth_id/user_code")
    try:
        interval = int(str(data.get("interval") or "5").strip())
    except ValueError:
        interval = 5
    return {"device_auth_id": str(device_auth_id), "user_code": str(user_code),
            "interval": max(1, interval)}


def device_token_poll(device_auth_id: str, user_code: str, *, http_post=None) -> dict | None:
    """One poll of /deviceauth/token. None while the user hasn't approved
    (403/404 per the CLI); the authorization code + server-generated PKCE
    verifier once they have; DeviceAuthError on anything else."""
    post = http_post or _default_http_post
    resp = post(f"{DEVICE_AUTH_BASE_URL}/token",
                json={"device_auth_id": device_auth_id, "user_code": user_code})
    status = getattr(resp, "status_code", 0)
    if status in (403, 404):
        return None
    if status != 200:
        raise DeviceAuthError(f"device auth poll failed with status {status}")
    data = resp.json()
    if not data.get("authorization_code") or not data.get("code_verifier"):
        raise DeviceAuthError("device auth token response missing authorization_code/code_verifier")
    return {"authorization_code": data["authorization_code"],
            "code_verifier": data["code_verifier"]}


def device_code_exchange(authorization_code: str, code_verifier: str, *,
                         http_post_form=None, client_id: str = CODEX_CLIENT_ID) -> dict:
    """Exchange the device-flow authorization code for tokens and return an
    auth.json-shaped dict ready for CodexAuthStore.add_account(). The token
    endpoint requires form encoding here (unlike the JSON refresh call)."""
    post = http_post_form or _default_http_post_form
    resp = post(OAUTH_TOKEN_URL, data={
        "grant_type":    "authorization_code",
        "code":          authorization_code,
        "redirect_uri":  DEVICE_CALLBACK_URL,
        "client_id":     client_id,
        "code_verifier": code_verifier,
    })
    status = getattr(resp, "status_code", 0)
    if status != 200:
        raise DeviceAuthError(f"token exchange failed with status {status}")
    data = resp.json()
    if not data.get("access_token"):
        raise DeviceAuthError("token exchange response has no access_token")
    from datetime import datetime, timezone
    return {
        "tokens": {
            "access_token":  data["access_token"],
            "refresh_token": data.get("refresh_token"),
            "id_token":      data.get("id_token"),
            "account_id":    id_token_account_id(data.get("id_token")),
        },
        "last_refresh": datetime.now(timezone.utc).isoformat(),
    }


def id_token_account_id(id_token: str | None) -> str | None:
    """chatgpt_account_id from the id_token's https://api.openai.com/auth claim."""
    payload = _jwt_payload(id_token) or {}
    auth = payload.get("https://api.openai.com/auth")
    if isinstance(auth, dict) and auth.get("chatgpt_account_id"):
        return str(auth["chatgpt_account_id"])
    return None


def _default_http_post_form(url: str, data: dict):  # pragma: no cover - needs network
    import httpx
    return httpx.post(url, data=data, timeout=30.0)
```

`_jwt_payload` refactor:

```python
def _jwt_payload(token: str | None) -> dict | None:
    """Best-effort decode of a JWT's payload segment (no signature check)."""
    if not token or token.count(".") != 2:
        return None
    import base64
    payload_b64 = token.split(".")[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _jwt_exp(token: str | None) -> float | None:
    """Best-effort extraction of the `exp` claim from a JWT access token."""
    exp = (_jwt_payload(token) or {}).get("exp")
    return float(exp) if isinstance(exp, (int, float)) else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_codex_device_auth.py tests/test_codex_auth_store.py tests/test_codex.py -q`
Expected: all PASS (the store/backend suites prove the refactor broke nothing).

- [ ] **Step 5: Commit**

```bash
git add codex_auth.py tests/test_codex_device_auth.py
git commit -m "feat: device-code OAuth client for Codex (usercode/poll/exchange)"
```

---

### Task 2: `CodexInviteStore` in new `codex_invites.py`

**Files:**
- Create: `codex_invites.py`
- Test: `tests/test_codex_invites.py` (new)

**Interfaces:**
- Consumes: `codex_auth._safe_account_name`.
- Produces: `CodexInviteStore(accounts_dir, *, now=None)` with `create(name) -> dict` (incl. `token`), `get(token) -> dict | None`, `list() -> list[dict]` (each with `status`), `revoke(token) -> bool`, `set_device(token, device: dict) -> None`, `clear_device(token) -> None`, `mark_used(token) -> None`, `due_for_poll(token) -> bool`, `status_of(invite) -> str` in `{"pending","awaiting","used","expired"}`; constants `INVITE_TTL_S = 86400`, `DEVICE_CODE_TTL_S = 900`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_codex_invites.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codex_invites import DEVICE_CODE_TTL_S, INVITE_TTL_S, CodexInviteStore  # noqa: E402


def _store(tmp_path, t=[1000.0]):
    return CodexInviteStore(tmp_path / "accounts", now=lambda: t[0]), t


def test_create_get_list_revoke(tmp_path):
    store, _ = _store(tmp_path)
    inv = store.create("Team One!")
    assert inv["name"] == "team-one" and inv["token"] and inv["expires_at"] == 1000 + INVITE_TTL_S
    got = store.get(inv["token"])
    assert got["name"] == "team-one"
    assert [i["name"] for i in store.list()] == ["team-one"]
    assert store.list()[0]["status"] == "pending"
    assert store.revoke(inv["token"]) is True
    assert store.get(inv["token"]) is None
    assert store.revoke(inv["token"]) is False


def test_invite_file_is_underscore_reserved(tmp_path):
    store, _ = _store(tmp_path)
    store.create("a")
    files = [f.name for f in (tmp_path / "accounts").glob("*.json")]
    assert files == ["_invites.json"]  # CodexAuthStore skips _-prefixed files


def test_same_name_replaces_pending_but_not_used(tmp_path):
    store, _ = _store(tmp_path)
    first = store.create("team-1")
    second = store.create("team-1")
    assert store.get(first["token"]) is None and store.get(second["token"])
    store.mark_used(second["token"])
    third = store.create("team-1")
    assert store.get(second["token"])["used_at"]  # used invites are kept for audit
    assert store.get(third["token"])


def test_status_transitions_and_expiry(tmp_path):
    store, t = _store(tmp_path)
    inv = store.create("x")
    tok = inv["token"]
    assert store.status_of(store.get(tok)) == "pending"
    store.set_device(tok, {"device_auth_id": "da", "user_code": "UC", "interval": 5})
    assert store.status_of(store.get(tok)) == "awaiting"
    t[0] += DEVICE_CODE_TTL_S + 1          # device code expired -> back to pending
    assert store.status_of(store.get(tok)) == "pending"
    store.clear_device(tok)
    store.mark_used(tok)
    assert store.status_of(store.get(tok)) == "used"
    other = store.create("y")
    t[0] += INVITE_TTL_S + 1
    assert store.status_of(store.get(other["token"])) == "expired"


def test_due_for_poll_respects_interval(tmp_path):
    store, t = _store(tmp_path)
    tok = store.create("x")["token"]
    store.set_device(tok, {"device_auth_id": "da", "user_code": "UC", "interval": 5})
    assert store.due_for_poll(tok) is True     # first poll immediately
    assert store.due_for_poll(tok) is False    # too soon
    t[0] += 5
    assert store.due_for_poll(tok) is True
    assert store.due_for_poll("nonexistent") is False


def test_no_tokens_ever_stored(tmp_path):
    store, _ = _store(tmp_path)
    tok = store.create("x")["token"]
    store.set_device(tok, {"device_auth_id": "da", "user_code": "UC", "interval": 5})
    text = (tmp_path / "accounts" / "_invites.json").read_text()
    assert "access_token" not in text and "refresh_token" not in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_codex_invites.py -q`
Expected: ModuleNotFoundError `codex_invites`.

- [ ] **Step 3: Implement `codex_invites.py`**

```python
"""
codex_invites.py — single-use invite links for the server-side Codex OAuth
onboarding flow (see docs/superpowers/specs/2026-07-22-codex-oauth-invite-design.md).

An invite binds a secret URL token to a target account name. The teammate who
opens the link signs in via OpenAI's device-code flow; the auth_proxy endpoints
drive that flow and store the resulting account through CodexAuthStore.

Persisted to {accounts_dir}/_invites.json — the underscore prefix is required:
CodexAuthStore.reload() treats non-underscore *.json files in that dir as
accounts. The file never contains OAuth tokens; only the short-lived
device_auth_id/user_code pair (useless without the user's in-browser approval).
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from pathlib import Path

INVITE_TTL_S = 24 * 3600        # invite link lifetime
DEVICE_CODE_TTL_S = 15 * 60     # OpenAI device codes expire after 15 minutes


class CodexInviteStore:
    def __init__(self, accounts_dir: str | Path, *, now=None):
        self._path = Path(accounts_dir) / "_invites.json"
        self._now = now or time.time
        self._lock = threading.Lock()

    # ---- persistence ---------------------------------------------------

    def _load(self) -> dict:
        try:
            raw = json.loads(self._path.read_text())
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, invites: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(invites, indent=2))
        try:
            self._path.chmod(0o600)
        except OSError:
            pass

    def _mutate(self, token: str, fn) -> bool:
        """Apply fn(invite) to one invite under the lock; False if unknown."""
        with self._lock:
            invites = self._load()
            inv = invites.get(token)
            if not isinstance(inv, dict):
                return False
            fn(inv)
            self._save(invites)
            return True

    # ---- lifecycle -----------------------------------------------------

    def create(self, name: str, *, ttl_s: int = INVITE_TTL_S) -> dict:
        from codex_auth import _safe_account_name
        slug = _safe_account_name(name)
        now = self._now()
        token = secrets.token_urlsafe(32)
        with self._lock:
            invites = self._load()
            # One pending link per name; used invites are kept for audit.
            invites = {t: inv for t, inv in invites.items()
                       if not (inv.get("name") == slug and not inv.get("used_at"))}
            invites[token] = {"name": slug, "created_at": now, "expires_at": now + ttl_s}
            self._save(invites)
            return {"token": token, **invites[token]}

    def get(self, token: str) -> dict | None:
        inv = self._load().get(token)
        return {"token": token, **inv} if isinstance(inv, dict) else None

    def list(self) -> list[dict]:
        out = []
        for token, inv in sorted(self._load().items(),
                                 key=lambda kv: kv[1].get("created_at", 0), reverse=True):
            out.append({"token": token, **inv, "status": self.status_of(inv)})
        return out

    def revoke(self, token: str) -> bool:
        with self._lock:
            invites = self._load()
            if token not in invites:
                return False
            del invites[token]
            self._save(invites)
            return True

    # ---- device-flow state ---------------------------------------------

    def set_device(self, token: str, device: dict) -> None:
        now = self._now()
        self._mutate(token, lambda inv: inv.update(
            device_auth_id=device["device_auth_id"], user_code=device["user_code"],
            interval=device.get("interval", 5), device_started_at=now, last_poll_at=None))

    def clear_device(self, token: str) -> None:
        self._mutate(token, lambda inv: [
            inv.pop(k, None)
            for k in ("device_auth_id", "user_code", "interval",
                      "device_started_at", "last_poll_at")])

    def mark_used(self, token: str) -> None:
        self._mutate(token, lambda inv: inv.update(used_at=self._now()))

    def due_for_poll(self, token: str) -> bool:
        """Interval guard: True at most once per OpenAI-mandated interval.
        Stamps last_poll_at when it grants a poll."""
        now = self._now()
        granted = []

        def check(inv):
            last = inv.get("last_poll_at")
            if last is None or now - last >= inv.get("interval", 5):
                inv["last_poll_at"] = now
                granted.append(True)

        return self._mutate(token, check) and bool(granted)

    # ---- status ---------------------------------------------------------

    def status_of(self, invite: dict) -> str:
        now = self._now()
        if invite.get("used_at"):
            return "used"
        if now > invite.get("expires_at", 0):
            return "expired"
        started = invite.get("device_started_at")
        if invite.get("device_auth_id") and started and now - started <= DEVICE_CODE_TTL_S:
            return "awaiting"
        return "pending"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_codex_invites.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add codex_invites.py tests/test_codex_invites.py
git commit -m "feat: file-backed single-use invite store for codex onboarding"
```

---

### Task 3: Invite + onboarding endpoints in `auth_proxy.py`

**Files:**
- Modify: `auth_proxy.py` (new section right after `dashboard_delete_codex_account`, ~line 2903)
- Test: `tests/test_auth_proxy_codex_invites.py` (new)

**Interfaces:**
- Consumes: Task 1 functions, Task 2 store, existing `_require_admin_dashboard_caller`, `_codex_store`, `_reload_codex_router`, `_log`, `CODEX_ACCOUNTS_DIR`.
- Produces: `POST/GET /dashboard/api/codex/invites`, `DELETE /dashboard/api/codex/invites/{token}`, `GET /codex/onboard/{token}` (HTML), `POST /codex/onboard/{token}/start`, `GET /codex/onboard/{token}/status`; helpers `_invite_store()`, `_request_origin(request)`, `_onboard_html(...)` (Task 4 fills the HTML body).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_auth_proxy_codex_invites.py
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
    # single use: page + status now report used, start refuses
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_auth_proxy_codex_invites.py -q`
Expected: 404s / AttributeErrors (endpoints missing).

- [ ] **Step 3: Implement the endpoints in `auth_proxy.py`**

Insert after `dashboard_delete_codex_account` (~line 2903):

```python
# ---- Codex onboarding invites (server-side OAuth device flow) --------------
# Spec: docs/superpowers/specs/2026-07-22-codex-oauth-invite-design.md

def _invite_store():
    from codex_invites import CodexInviteStore
    return CodexInviteStore(CODEX_ACCOUNTS_DIR)


def _request_origin(request: Request) -> str:
    """Public origin for links we hand out, honoring the ingress's
    X-Forwarded-* headers; falls back to the request's own scheme/host."""
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "http")
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host")
            or request.url.netloc)
    return f"{proto.split(',')[0].strip()}://{host.split(',')[0].strip()}"


def _invite_view(inv: dict, origin: str) -> dict:
    return {"name": inv["name"], "status": inv.get("status"),
            "url": f"{origin}/codex/onboard/{inv['token']}",
            "created_at": inv.get("created_at"), "expires_at": inv.get("expires_at"),
            "used_at": inv.get("used_at")}


@app.post("/dashboard/api/codex/invites")
async def dashboard_create_codex_invite(request: Request) -> Response:
    """Mint a single-use onboarding link that lets a teammate OAuth their
    ChatGPT account into the named Codex slot. Admin-only."""
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "invalid JSON body", "type": "invalid_request", "code": "codex_invite"}})
    try:
        inv = _invite_store().create(str(body.get("name") or ""))
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request", "code": "codex_invite"}})
    _log({"event": "dashboard_codex_invite_created", "account": inv["name"], "viewer": caller})
    return JSONResponse(content={"ok": True, **_invite_view({**inv, "status": "pending"}, _request_origin(request))})


@app.get("/dashboard/api/codex/invites")
async def dashboard_list_codex_invites(request: Request) -> Response:
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    origin = _request_origin(request)
    return JSONResponse(content={"invites": [_invite_view(i, origin) for i in _invite_store().list()]})


@app.delete("/dashboard/api/codex/invites/{token}")
async def dashboard_revoke_codex_invite(request: Request, token: str) -> Response:
    caller, error = _require_admin_dashboard_caller(request)
    if error:
        return error
    if not _invite_store().revoke(token):
        return JSONResponse(status_code=404, content={"error": {"message": "invite not found", "type": "not_found", "code": "codex_invite_not_found"}})
    _log({"event": "dashboard_codex_invite_revoked", "viewer": caller})
    return JSONResponse(content={"ok": True})


def _live_invite(token: str):
    """(invite, error_response). Invalid, expired and used links all present
    as a dead link to the visitor (no distinguishing oracle)."""
    store = _invite_store()
    inv = store.get(token)
    if inv is None:
        return None, None, "missing"
    return store, inv, store.status_of(inv)


@app.get("/codex/onboard/{token}")
async def codex_onboard_page(token: str) -> Response:
    store, inv, status = _live_invite(token)
    if inv is None or status == "expired":
        return HTMLResponse(_onboard_dead_html(), status_code=404)
    return HTMLResponse(_onboard_html(inv["name"], connected=(status == "used")))


@app.post("/codex/onboard/{token}/start")
async def codex_onboard_start(token: str) -> Response:
    import codex_auth
    store, inv, status = _live_invite(token)
    if inv is None or status == "expired":
        return JSONResponse(status_code=404, content={"error": {"message": "invite link expired", "type": "not_found", "code": "codex_onboard"}})
    if status == "used":
        return JSONResponse(status_code=409, content={"error": {"message": "invite already used", "type": "invalid_request", "code": "codex_onboard_used"}})
    try:
        device = await asyncio.to_thread(codex_auth.device_usercode_request)
    except codex_auth.DeviceAuthError as exc:
        return JSONResponse(status_code=502, content={"error": {"message": str(exc), "type": "device_auth_error", "code": "codex_onboard_start"}})
    store.set_device(token, device)
    _log({"event": "codex_invite_signin_started", "account": inv["name"]})
    return JSONResponse(content={"user_code": device["user_code"],
                                 "verification_url": codex_auth.DEVICE_VERIFY_URL,
                                 "interval": device["interval"]})


@app.get("/codex/onboard/{token}/status")
async def codex_onboard_status(token: str) -> Response:
    import codex_auth
    store, inv, status = _live_invite(token)
    if inv is None:
        return JSONResponse(status_code=404, content={"error": {"message": "invite link expired", "type": "not_found", "code": "codex_onboard"}})
    if status == "used":
        return JSONResponse(content={"status": "connected", "name": inv["name"]})
    if status == "expired":
        return JSONResponse(content={"status": "expired"})
    if status == "pending":
        if inv.get("device_auth_id"):
            store.clear_device(token)   # device code timed out; allow a fresh start
        return JSONResponse(content={"status": "pending"})
    # status == "awaiting": at most one upstream poll per interval
    if not store.due_for_poll(token):
        return JSONResponse(content={"status": "awaiting"})
    try:
        result = await asyncio.to_thread(
            codex_auth.device_token_poll, inv["device_auth_id"], inv["user_code"])
        if result is None:
            return JSONResponse(content={"status": "awaiting"})
        auth_json = await asyncio.to_thread(
            codex_auth.device_code_exchange,
            result["authorization_code"], result["code_verifier"])
    except codex_auth.DeviceAuthError as exc:
        store.clear_device(token)
        _log({"event": "codex_invite_device_error", "account": inv["name"], "error": str(exc)})
        return JSONResponse(content={"status": "error", "message": str(exc)})
    slug = _codex_store().add_account(inv["name"], auth_json)
    applied_live, _ = await _reload_codex_router()
    store.mark_used(token)
    _log({"event": "codex_invite_connected", "account": slug, "applied_live": applied_live})
    return JSONResponse(content={"status": "connected", "name": slug, "applied_live": applied_live})
```

Note: `import asyncio` — check the file header imports; add if missing. `_onboard_html(name, connected=False)` / `_onboard_dead_html()` are defined in Task 4 — for this task's test run, add minimal stubs returning `"<html>..."` containing the needles `Sign in with ChatGPT` / `noindex` / `expired` (the real page replaces them in Task 4; keep the stubs' needles).

- [ ] **Step 4: Run tests, expect pass**

Run: `python -m pytest tests/test_auth_proxy_codex_invites.py tests/test_auth_proxy_dashboard_full.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add auth_proxy.py tests/test_auth_proxy_codex_invites.py
git commit -m "feat: codex onboarding invite endpoints (admin mint/list/revoke + public device-flow driver)"
```

---

### Task 4: Onboarding page HTML

**Files:**
- Modify: `auth_proxy.py` (replace the Task-3 stubs, near `_dashboard_html`)
- Test: extend `tests/test_auth_proxy_codex_invites.py`

**Interfaces:**
- Consumes: routes from Task 3.
- Produces: `_onboard_html(name: str, connected: bool = False) -> str`, `_onboard_dead_html() -> str`.

- [ ] **Step 1: Extend the page test**

```python
def test_onboard_page_content(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path)
    admin = _admin(monkeypatch)
    url = admin.post("/dashboard/api/codex/invites", json={"name": "t1"}).json()["url"]
    token = url.rsplit("/", 1)[1]
    html = TestClient(auth_proxy.app).get(f"/codex/onboard/{token}").text
    for needle in ("Sign in with ChatGPT", "noindex", "t1",
                   "/start", "/status", "cancel", "one-time code"):
        assert needle in html, needle
```

- [ ] **Step 2: Run it, expect failure on missing needles**

Run: `python -m pytest tests/test_auth_proxy_codex_invites.py::test_onboard_page_content -q`

- [ ] **Step 3: Implement the real page**

Replace the stubs with a self-contained dark page consistent with the dashboard palette (same `--bg`/`--accent` values, inline CSS, no external assets):

```python
def _onboard_dead_html() -> str:
    return """<!doctype html><html lang='en'><head><meta charset='utf-8'/>
<meta name='viewport' content='width=device-width,initial-scale=1'/>
<meta name='robots' content='noindex,nofollow,noarchive'/><title>Link expired</title>
<style>body{margin:0;background:#08090a;color:#f7f8f8;font:15px/1.5 Inter,ui-sans-serif,system-ui,sans-serif;display:grid;place-items:center;min-height:100vh}main{max-width:420px;padding:32px;text-align:center}h1{font-size:22px}p{color:#8a8f98}</style>
</head><body><main><h1>This link is no longer valid</h1>
<p>The invite has expired or was already used. Ask the person who sent it for a new link.</p>
</main></body></html>"""


def _onboard_html(name: str, connected: bool = False) -> str:
    import html as _html
    safe = _html.escape(name)
    state = "connected" if connected else "idle"
    idle_cls = "hidden" if connected else ""
    done_cls = "" if connected else "hidden"
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'/>
<meta name='viewport' content='width=device-width,initial-scale=1'/>
<meta name='robots' content='noindex,nofollow,noarchive'/><title>Connect ChatGPT — {safe}</title>
<style>
body{{margin:0;background:radial-gradient(circle at 20% -10%,rgba(113,112,255,.18),transparent 34%),#08090a;color:#f7f8f8;font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif;display:grid;place-items:center;min-height:100vh}}
main{{max-width:460px;padding:34px;border:1px solid rgba(255,255,255,.075);border-radius:18px;background:rgba(15,16,17,.92);box-shadow:0 24px 80px rgba(0,0,0,.42)}}
h1{{font-size:21px;letter-spacing:-.3px;margin:0 0 6px}}p{{color:#8a8f98;margin:10px 0}}
.btn{{display:inline-block;border:0;border-radius:10px;background:linear-gradient(180deg,#7170ff,#5e6ad2);color:#fff;font:inherit;font-weight:590;padding:11px 18px;cursor:pointer;text-decoration:none}}
.code{{font:600 30px/1 'JetBrains Mono',ui-monospace,monospace;letter-spacing:.14em;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:12px;padding:14px 18px;text-align:center;margin:14px 0;user-select:all}}
.muted{{font-size:13px;color:#62666d}}.ok{{color:#27a644}}.err{{color:#ff5c7a}}
.hidden{{display:none}}.copy{{margin-left:10px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);color:#f7f8f8;border-radius:8px;padding:6px 10px;cursor:pointer}}
</style></head><body><main data-state='{state}'>
<h1>Connect your ChatGPT account</h1>
<p>This links your ChatGPT subscription to the <b>unhardcoded</b> router as account <b>{safe}</b>. Only continue if a person you trust sent you this link.</p>
<div id='idle' class='{idle_cls}'>
  <button class='btn' id='startBtn'>Sign in with ChatGPT</button>
  <p class='muted'>You'll sign in on openai.com and enter a one-time code. OpenAI's page warns about codes given to you by websites — that warning refers to this flow; continue only because your operator sent you this link, otherwise cancel.</p>
</div>
<div id='steps' class='hidden'>
  <p>1 · Open <a id='verifyLink' class='btn' target='_blank' rel='noopener'>openai.com sign-in</a></p>
  <p>2 · Enter this one-time code <button class='copy' id='copyBtn'>copy</button></p>
  <div class='code' id='userCode'></div>
  <p class='muted' id='waitMsg'>Waiting for you to finish signing in… this page updates automatically.</p>
</div>
<div id='done' class='{done_cls}'>
  <p class='ok'>✓ Connected as <b>{safe}</b>. You can close this page.</p>
</div>
<p class='err hidden' id='errMsg'></p>
</main><script>
const S={{start:location.pathname+'/start',status:location.pathname+'/status'}};
const $=id=>document.getElementById(id);let timer=null;
function show(err){{$('errMsg').textContent=err||'';$('errMsg').classList.toggle('hidden',!err)}}
async function start(){{show('');try{{const r=await fetch(S.start,{{method:'POST'}});const d=await r.json();
if(!r.ok)throw new Error(d.error&&d.error.message||('start failed ('+r.status+')'));
$('verifyLink').href=d.verification_url;$('userCode').textContent=d.user_code;
$('idle').classList.add('hidden');$('steps').classList.remove('hidden');poll()}}
catch(e){{show(e.message)}}}}
async function poll(){{clearTimeout(timer);try{{const r=await fetch(S.status);const d=await r.json();
if(d.status==='connected'){{$('steps').classList.add('hidden');$('done').classList.remove('hidden');return}}
if(d.status==='error'){{$('steps').classList.add('hidden');$('idle').classList.remove('hidden');show(d.message||'sign-in failed — try again');return}}
if(d.status==='pending'&&!$('steps').classList.contains('hidden')){{$('steps').classList.add('hidden');$('idle').classList.remove('hidden');show('The sign-in expired — start again.');return}}
}}catch(e){{}}timer=setTimeout(poll,5000)}}
$('startBtn').onclick=start;
$('copyBtn').onclick=()=>navigator.clipboard.writeText($('userCode').textContent);
</script></body></html>"""
```

- [ ] **Step 4: Run the invite test file, expect pass**

Run: `python -m pytest tests/test_auth_proxy_codex_invites.py -q`

- [ ] **Step 5: Commit**

```bash
git add auth_proxy.py tests/test_auth_proxy_codex_invites.py
git commit -m "feat: codex onboarding page (device-code sign-in UI)"
```

---

### Task 5: Dashboard codex panel — invite UI

**Files:**
- Modify: `auth_proxy.py` `_dashboard_html()` (codex cards ~line 4487-4488, JS ~4763-4780)
- Test: extend `tests/test_auth_proxy_dashboard_full.py::test_dashboard_html_has_codex_account_ui`

**Interfaces:**
- Consumes: Task 3 endpoints.
- Produces: dashboard UI only.

- [ ] **Step 1: Extend the HTML needle test**

In `test_dashboard_html_has_codex_account_ui`, extend the needles tuple to:

```python
    for needle in ("Codex accounts", "/dashboard/api/codex/accounts",
                   "addCodexAccount", "loadCodexAccounts",
                   "/dashboard/api/codex/invites", "generateCodexInvite",
                   "revokeCodexInvite", "codexInvites", "Invite via link"):
        assert needle in html, needle
```

- [ ] **Step 2: Run it, expect failure**

Run: `python -m pytest tests/test_auth_proxy_dashboard_full.py::test_dashboard_html_has_codex_account_ui -q`

- [ ] **Step 3: Implement the UI**

a) In the codex accounts card toolbar (line ~4487) add a second button before the paste toggle:

```html
<button class='btn' id='toggleInviteCodex'>Invite via link</button>
```

and after `<div id='codexAccounts'></div>` add `<div id='codexInvites'></div>`.

b) After the `addCodexCard` card (line ~4488) add:

```html
<div class='card span12' id='inviteCodexCard' style='display:none'><div class='cardPad'><div class='formGrid'><label>Account name<input id='inviteCodexName' placeholder='team-1' /></label></div><div class='actions' style='margin-top:10px'><button class='btn primary' id='inviteCodexSubmit'>Generate invite link</button><button class='btn' id='inviteCodexCancel'>Cancel</button></div><div id='inviteCodexResult' class='muted small' style='margin-top:8px'></div></div></div>
```

c) In the JS block (near `loadCodexAccounts`, line ~4763) add:

```javascript
async function loadCodexInvites(){try{const r=await fetch('/dashboard/api/codex/invites',{credentials:'same-origin'});if(!r.ok){$('codexInvites').innerHTML='';return}renderCodexInvites((await r.json()).invites||[])}catch(e){}}
function renderCodexInvites(list){if(!list.length){$('codexInvites').innerHTML='';return}const pill=s=>s==='used'?'<span class="pill ok">used</span>':s==='expired'?'<span class="pill bad">expired</span>':s==='awaiting'?'<span class="pill warn">awaiting sign-in</span>':'<span class="pill">pending</span>';$('codexInvites').innerHTML='<div class="label" style="padding:12px 14px 4px">Onboarding invites</div>'+table(list,[{label:'Account',f:r=>`<div class="rowTitle">${esc(r.name)}</div>`},{label:'Status',f:r=>pill(r.status)},{label:'Expires',f:r=>r.expires_at?new Date(r.expires_at*1000).toLocaleString():'—'},{label:'',cls:'right',f:r=>`<button class="btn ghost small" onclick="navigator.clipboard.writeText(${jsarg(r.url)}).then(()=>toast('Invite link copied'))">Copy link</button> <button class="btn iconBtn ghost" title="Revoke invite" onclick="revokeCodexInvite(${jsarg(r.url.split('/').pop())})">🗑</button>`}])}
async function generateCodexInvite(){try{const name=$('inviteCodexName').value.trim();if(!name){$('inviteCodexResult').textContent='account name required';return}const r=await fetch('/dashboard/api/codex/invites',{method:'POST',headers:{'content-type':'application/json'},credentials:'same-origin',body:JSON.stringify({name})});if(r.status===401){showLogin();return}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||`invite ${r.status}`);await navigator.clipboard.writeText(d.url).catch(()=>{});$('inviteCodexResult').innerHTML='Invite for <b>'+esc(d.name)+'</b> copied to clipboard — send it to the teammate. Valid 24h, single use.<br><code>'+esc(d.url)+'</code>';toast('Invite link copied');loadCodexInvites()}catch(e){$('inviteCodexResult').textContent=e.message;showErr(e.message)}}
async function revokeCodexInvite(token){if(!confirm('Revoke this invite link?'))return;try{const r=await fetch('/dashboard/api/codex/invites/'+encodeURIComponent(token),{method:'DELETE',credentials:'same-origin'});if(r.status===401){showLogin();return}const d=await r.json();if(!r.ok)throw new Error(d.error?.message||`revoke ${r.status}`);toast('Invite revoked');loadCodexInvites()}catch(e){showErr(e.message)}}
```

d) Call `loadCodexInvites()` from inside `loadCodexAccounts()` (append `loadCodexInvites();` after `renderCodexAccounts(await r.json())`), and wire the toggles next to the existing `toggleAddCodex` wiring (line ~4780):

```javascript
$('toggleInviteCodex').onclick=()=>{const c=$('inviteCodexCard');c.style.display=c.style.display==='none'?'':'none'};$('inviteCodexCancel').onclick=()=>{$('inviteCodexCard').style.display='none'};$('inviteCodexSubmit').onclick=generateCodexInvite;
```

e) In the 15-second auto-refresh guard (`setInterval` at the end), add `inviteCodexCard` to the open-cards check alongside `addCodexCard`:

```javascript
const ap=$('addProviderCard'),ac=$('addCodexCard'),ic=$('inviteCodexCard');if((ap&&ap.style.display&&ap.style.display!=='none')||(ac&&ac.style.display&&ac.style.display!=='none')||(ic&&ic.style.display&&ic.style.display!=='none'))return;
```

- [ ] **Step 4: Run dashboard tests, expect pass**

Run: `python -m pytest tests/test_auth_proxy_dashboard_full.py -q`

- [ ] **Step 5: Commit**

```bash
git add auth_proxy.py tests/test_auth_proxy_dashboard_full.py
git commit -m "feat: dashboard UI to mint/list/revoke codex onboarding invites"
```

---

### Task 6: Docs

**Files:**
- Modify: `docs/OPENAI-CODEX.md` (Setup section)

- [ ] **Step 1: Rewrite the Setup section**

Replace the `## Setup` section with:

```markdown
## Setup

**Invite flow (recommended — no CLI needed):** in the dashboard's *Provider
keys* tab → *Codex accounts* → **Invite via link**, name the account (e.g.
`team-1`) and send the generated single-use link (valid 24 h) to the account
holder. They click **Sign in with ChatGPT**, sign in on openai.com, and enter
the one-time code shown; the router captures the tokens via OpenAI's device
authorization flow (`POST auth.openai.com/api/accounts/deviceauth/usercode` →
poll `/deviceauth/token` → form-encoded code exchange at `/oauth/token`),
stores the account under `$CODEX_ACCOUNTS_DIR/<name>.json`, and hot-reloads.
Note: OpenAI's device page warns about codes supplied by websites — this flow
is that case by design; it is for trusted operators only.

**Manual fallback (CLI):**

    # 1. Authenticate (opens a browser; "Sign in with ChatGPT")
    codex login                       # writes ~/.codex/auth.json

    # 2. Paste ~/.codex/auth.json into the dashboard's "Add codex account"
    #    form, or start the shim directly against the file:
    python -m hosts.python_shim --config hosts/python_shim/config.live.lua \
        --codex-auth ~/.codex/auth.json

`--codex-auth` defaults to `~/.codex/auth.json`. Treat that file — and invite
links — like passwords.
```

- [ ] **Step 2: Commit**

```bash
git add docs/OPENAI-CODEX.md
git commit -m "docs: document the codex invite onboarding flow"
```

---

### Task 7: Full verification + PR

- [ ] **Step 1: Full test suite**

Run: `python -m pytest tests/ -q`
Expected: everything passes (note: some suites need the dev Postgres; if unavailable, run at minimum `tests/test_codex_device_auth.py tests/test_codex_invites.py tests/test_auth_proxy_codex_invites.py tests/test_auth_proxy_dashboard_full.py tests/test_codex_auth_store.py tests/test_codex.py` and say so in the PR).

- [ ] **Step 2: Self code-review (superpowers:requesting-code-review), fix findings, commit fixes**

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin feat/codex-oauth-invites
gh pr create --title "feat: server-side Codex OAuth onboarding via single-use invite links" --body "..."
```

PR body: summary of the flow, spec/plan pointers, test evidence, the ToS/transparency caveat, and the note that the live OpenAI flow is untested in CI.
