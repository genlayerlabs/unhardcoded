from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from codex_auth import CodexAuthStore  # noqa: E402


def _auth(account_id: str, token: str = "tok") -> dict:
    # No `exp` claim -> CodexAuth never tries to refresh, so tests stay offline.
    return {"tokens": {"access_token": token, "refresh_token": "r", "account_id": account_id}}


def test_store_discovers_dir_accounts_and_selects_first(tmp_path):
    d = tmp_path / "accounts"
    d.mkdir()
    (d / "team-b.json").write_text(json.dumps(_auth("acct-b", "tok-b")))
    (d / "team-a.json").write_text(json.dumps(_auth("acct-a", "tok-a")))

    store = CodexAuthStore(d)
    assert store.names() == ["team-a", "team-b"]
    # first by sorted name -> team-a
    assert store.account_id() == "acct-a"
    assert store.access_token() == "tok-a"

    listed = {a["name"]: a for a in store.list_accounts()}
    assert listed["team-a"]["account_id"] == "acct-a"
    assert listed["team-b"]["fingerprint"] and listed["team-b"]["fingerprint"] != listed["team-a"]["fingerprint"]


def test_store_legacy_single_file_is_default_account(tmp_path):
    legacy = tmp_path / "auth.json"
    legacy.write_text(json.dumps(_auth("legacy-acct")))
    store = CodexAuthStore(tmp_path / "missing-accounts", legacy_path=legacy)
    assert store.names() == ["default"]
    assert store.account_id() == "legacy-acct"
    assert store.list_accounts()[0]["account_id"] == "legacy-acct"


def test_store_empty_is_inactive_not_an_error(tmp_path):
    store = CodexAuthStore(tmp_path / "none")
    assert store.names() == []
    assert store.access_token() is None
    assert store.account_id() is None
    assert store.list_accounts() == []


def test_add_account_validates_and_persists(tmp_path):
    d = tmp_path / "accounts"
    store = CodexAuthStore(d)

    with pytest.raises(ValueError):
        store.add_account("bad", {"tokens": {"refresh_token": "r"}})  # no access_token
    with pytest.raises(ValueError):
        store.add_account("!!!", _auth("x"))  # name has no usable characters

    slug = store.add_account("Team One!", _auth("acct-1"))
    assert slug == "team-one"
    assert (d / "team-one.json").exists()
    assert "team-one" in store.names()
    assert store.get("team-one") is not None


def test_delete_account(tmp_path):
    d = tmp_path / "accounts"
    store = CodexAuthStore(d)
    store.add_account("gone", _auth("g"))
    assert store.delete_account("gone") is True
    assert store.names() == []
    assert store.delete_account("gone") is False  # already absent


def test_reload_picks_up_external_writes(tmp_path):
    d = tmp_path / "accounts"
    d.mkdir()
    store = CodexAuthStore(d)
    assert store.names() == []
    (d / "new.json").write_text(json.dumps(_auth("n")))
    assert store.reload() == ["new"]
    assert store.account_id() == "n"
