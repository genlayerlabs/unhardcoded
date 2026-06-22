"""
codex_auth.py — read and refresh the ChatGPT-subscription token that the Codex
CLI stores at ~/.codex/auth.json, for use by the llm-router "openai_codex"
provider.

UNOFFICIAL / ToS-RISKY. The OpenAI Apps SDK OAuth does NOT grant inference on a
ChatGPT subscription; only the Codex login + local-proxy pattern does. This
mimics what `codex login` produces. OpenAI may close this without notice
(Anthropic and Google closed equivalents in 2026). See docs/OPENAI-CODEX.md.

auth.json shape (as written by `codex login`):
    {
      "tokens": {
        "access_token":  "...",
        "refresh_token": "...",
        "id_token":      "...",
        "account_id":    "..."        # may also live at top level
      },
      "last_refresh": "<iso8601>"
    }
Some versions store the token fields at the top level; we accept both.

Refresh: POST https://auth.openai.com/oauth/token with the public Codex client
id, grant_type=refresh_token. Refreshed tokens are written back to auth.json.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

# Refresh proactively once the token is within this margin of expiry. The Codex
# access token is a JWT; we read its `exp` when present, else fall back to a
# fixed TTL after the last refresh.
_REFRESH_MARGIN_S = 300
_FALLBACK_TTL_S = 25 * 60


class CodexAuth:
    """Loads, caches and refreshes the Codex subscription token. Thread-safe;
    callable token getters are exposed for the credential resolver."""

    def __init__(
        self,
        auth_path: str | Path | None = None,
        *,
        http_post: "Any" = None,
        now: "Any" = None,
        client_id: str = CODEX_CLIENT_ID,
    ):
        self._path = Path(auth_path or (Path.home() / ".codex" / "auth.json"))
        self._client_id = client_id
        self._now = now or time.time
        # http_post(url, json=...) -> object with .status_code and .json();
        # injectable for tests. Defaults to httpx.post.
        self._http_post = http_post
        self._lock = threading.Lock()
        self._tokens: dict = {}
        self._exp: float | None = None
        self._loaded = False

    # ---- public API ----------------------------------------------------

    def access_token(self) -> str | None:
        """Return a currently-valid access token, refreshing if near expiry."""
        with self._lock:
            if not self._loaded:
                self._load()
            if self._needs_refresh():
                self._refresh()
            return self._tokens.get("access_token")

    def account_id(self) -> str | None:
        with self._lock:
            if not self._loaded:
                self._load()
            return self._tokens.get("account_id")

    # ---- internals -----------------------------------------------------

    def _load(self) -> None:
        self._loaded = True
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, ValueError):
            self._tokens = {}
            self._exp = None
            return
        self._tokens = _extract_tokens(raw)
        self._exp = _jwt_exp(self._tokens.get("access_token"))

    def _needs_refresh(self) -> bool:
        if not self._tokens.get("access_token"):
            return bool(self._tokens.get("refresh_token"))
        if self._exp is not None:
            return self._now() >= (self._exp - _REFRESH_MARGIN_S)
        # No exp claim: refresh if we have a refresh token and no recency info.
        return False

    def _refresh(self) -> None:
        refresh_token = self._tokens.get("refresh_token")
        if not refresh_token:
            return
        post = self._http_post or _default_http_post
        try:
            resp = post(OAUTH_TOKEN_URL, json={
                "grant_type":    "refresh_token",
                "refresh_token": refresh_token,
                "client_id":     self._client_id,
            })
        except Exception:
            return  # keep the old token; the call may still 401 and surface it
        if getattr(resp, "status_code", 0) != 200:
            return
        try:
            data = resp.json()
        except Exception:
            return
        if data.get("access_token"):
            self._tokens["access_token"] = data["access_token"]
        if data.get("refresh_token"):
            self._tokens["refresh_token"] = data["refresh_token"]
        if data.get("id_token"):
            self._tokens["id_token"] = data["id_token"]
        self._exp = _jwt_exp(self._tokens.get("access_token"))
        self._write_back()

    def _write_back(self) -> None:
        try:
            existing = json.loads(self._path.read_text())
        except (OSError, ValueError):
            existing = {}
        if isinstance(existing.get("tokens"), dict):
            existing["tokens"].update({
                k: self._tokens[k]
                for k in ("access_token", "refresh_token", "id_token")
                if self._tokens.get(k)
            })
        else:
            existing.update({
                k: self._tokens[k]
                for k in ("access_token", "refresh_token", "id_token")
                if self._tokens.get(k)
            })
        try:
            self._path.write_text(json.dumps(existing, indent=2))
        except OSError:
            pass


_ACCOUNT_NAME_RE = None  # lazily compiled in _safe_account_name


def _safe_account_name(name: str) -> str:
    """A filesystem-safe account slug (lowercase alnum + dash/underscore)."""
    import re
    slug = re.sub(r"[^a-z0-9._-]+", "-", str(name or "").strip().lower()).strip("-._")
    if not slug:
        raise ValueError("account name must contain at least one alphanumeric character")
    if len(slug) > 64:
        slug = slug[:64].strip("-._")
    return slug


def _fingerprint(token: str | None) -> str | None:
    if not token:
        return None
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()[:12]


class CodexAuthStore:
    """Manages N Codex accounts, each a `CodexAuth` over its own auth.json on
    the PVC (default dir /codex/accounts/<name>.json). The dashboard adds /
    lists / deletes accounts; the router self-discovers the dir at startup and
    on reload. Exposes `access_token()` / `account_id()` so it is a drop-in for
    a single `CodexAuth` — until the policy drives per-call account selection
    (a follow-up), it serves the first available account.

    A legacy single `/codex/auth.json` is picked up as the `default` account so
    existing single-account deployments keep working with no migration."""

    def __init__(
        self,
        accounts_dir: str | Path | None = None,
        *,
        legacy_path: str | Path | None = None,
        http_post: "Any" = None,
        now: "Any" = None,
    ):
        import os
        self._dir = Path(accounts_dir or os.getenv("CODEX_ACCOUNTS_DIR", "/codex/accounts"))
        self._legacy_path = Path(legacy_path) if legacy_path else None
        self._http_post = http_post
        self._now = now
        self._lock = threading.Lock()
        self._accounts: "dict[str, CodexAuth]" = {}
        self.reload()

    # ---- discovery -----------------------------------------------------

    def reload(self) -> list[str]:
        """Re-scan the accounts dir (+ legacy file). Returns the account names."""
        found: "dict[str, CodexAuth]" = {}
        try:
            if self._dir.is_dir():
                for f in sorted(self._dir.glob("*.json")):
                    found[f.stem] = CodexAuth(f, http_post=self._http_post, now=self._now)
        except OSError:
            pass
        if self._legacy_path and self._legacy_path.exists() and "default" not in found:
            found["default"] = CodexAuth(self._legacy_path, http_post=self._http_post, now=self._now)
        with self._lock:
            self._accounts = found
        return sorted(found)

    # ---- selection (drop-in CodexAuth surface) -------------------------

    def _select(self) -> "CodexAuth | None":
        with self._lock:
            for _name in sorted(self._accounts):
                return self._accounts[_name]
        return None

    def access_token(self) -> str | None:
        acct = self._select()
        return acct.access_token() if acct else None

    def account_id(self) -> str | None:
        acct = self._select()
        return acct.account_id() if acct else None

    # ---- management ----------------------------------------------------

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._accounts)

    def get(self, name: str) -> "CodexAuth | None":
        with self._lock:
            return self._accounts.get(_safe_account_name(name))

    def _account_path(self, name: str) -> Path:
        if name == "default" and self._legacy_path is not None and \
                not (self._dir / "default.json").exists():
            return self._legacy_path
        return self._dir / f"{name}.json"

    def list_accounts(self) -> list[dict]:
        """Per-account metadata for the dashboard (never the raw token). Reads
        the files directly — no network refresh, so it is cheap to poll."""
        out = []
        for name in self.names():
            account_id = None
            fingerprint = None
            try:
                raw = json.loads(self._account_path(name).read_text())
                tokens = _extract_tokens(raw if isinstance(raw, dict) else {})
                account_id = tokens.get("account_id")
                fingerprint = _fingerprint(tokens.get("access_token"))
            except (OSError, ValueError):
                pass
            out.append({"name": name, "account_id": account_id, "fingerprint": fingerprint})
        return out

    def add_account(self, name: str, data: dict) -> str:
        """Validate a pasted auth.json and persist it as a new account file.
        Returns the stored account slug. Caller re-applies live via reload()."""
        slug = _safe_account_name(name)
        tokens = _extract_tokens(data if isinstance(data, dict) else {})
        if not tokens.get("access_token"):
            raise ValueError("auth.json has no access_token (expected `codex login` output)")
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{slug}.json"
        path.write_text(json.dumps(data, indent=2))
        try:
            path.chmod(0o600)
        except OSError:
            pass
        self.reload()
        return slug

    def delete_account(self, name: str) -> bool:
        slug = _safe_account_name(name)
        path = self._dir / f"{slug}.json"
        existed = path.exists()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        self.reload()
        return existed


def _extract_tokens(raw: dict) -> dict:
    """Accept both top-level and nested `tokens` layouts."""
    nested = raw.get("tokens") if isinstance(raw.get("tokens"), dict) else {}
    out = {}
    for k in ("access_token", "refresh_token", "id_token", "account_id"):
        out[k] = nested.get(k) or raw.get(k)
    return out


def _jwt_exp(token: str | None) -> float | None:
    """Best-effort extraction of the `exp` claim from a JWT access token."""
    if not token or token.count(".") != 2:
        return None
    import base64
    payload_b64 = token.split(".")[1]
    payload_b64 += "=" * (-len(payload_b64) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None
    exp = payload.get("exp")
    return float(exp) if isinstance(exp, (int, float)) else None


def _default_http_post(url: str, json: dict):  # pragma: no cover - needs network
    import httpx
    return httpx.post(url, json=json, timeout=30.0)
