from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codex_invites import DEVICE_CODE_TTL_S, INVITE_TTL_S, CodexInviteStore  # noqa: E402


def _store(tmp_path):
    t = [1000.0]
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
