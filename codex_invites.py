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
import logging
import os
import secrets
import threading
import time
from pathlib import Path

INVITE_TTL_S = 24 * 3600        # invite link lifetime
DEVICE_CODE_TTL_S = 15 * 60     # OpenAI device codes expire after 15 minutes

log = logging.getLogger(__name__)

# Module-level lock: callers construct short-lived store instances per request,
# so a per-instance lock would give zero cross-request exclusion over the
# shared read-modify-write of _invites.json (e.g. a /status poll could write
# back an invite the dashboard just revoked).
_lock = threading.Lock()


class CodexInviteStore:
    def __init__(self, accounts_dir: str | Path, *, now=None):
        self._path = Path(accounts_dir) / "_invites.json"
        self._now = now or time.time
        self._lock = _lock

    # ---- persistence ---------------------------------------------------

    def _load(self) -> dict:
        try:
            raw = json.loads(self._path.read_text())
            return raw if isinstance(raw, dict) else {}
        except OSError:
            return {}
        except ValueError:
            log.warning("codex invites file %s is corrupt; treating as empty", self._path)
            return {}

    def _save(self, invites: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(invites, indent=2))
        try:
            tmp.chmod(0o600)
        except OSError:
            pass
        os.replace(tmp, self._path)

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
