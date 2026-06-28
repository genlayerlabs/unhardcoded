"""
Operator settings store: PVC-backed overrides for tunable knobs, validated
against the schema (unknown keys and out-of-range values are rejected, so a bad
override can never break ranking). null clears an override back to its default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import host_store  # noqa: E402
import settings  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("ROUTER_DB_PATH", str(tmp_path / "host-store.db"))
    host_store.reset()
    settings.reload()
    yield
    host_store.reset()


def test_override_roundtrip_and_validation(store):
    assert settings.get("antseed.offers_top_n") == 3            # schema default
    assert settings.get("codex.imputed_price_in") == 5.0

    new, errs = settings.validate_and_write({"antseed.offers_top_n": 5})
    assert not errs and new["antseed.offers_top_n"] == 5
    assert settings.get("antseed.offers_top_n") == 5            # applied live
    row = next(k for k in settings.current() if k["key"] == "antseed.offers_top_n")
    assert row["overridden"] and row["value"] == 5 and row["default"] == 3

    # out of range → rejected, previous value kept
    _, errs = settings.validate_and_write({"antseed.offers_top_n": 99})
    assert errs and settings.get("antseed.offers_top_n") == 5

    # unknown key → rejected
    _, errs = settings.validate_and_write({"nope.x": 1})
    assert errs

    # null clears the override back to the default
    _, errs = settings.validate_and_write({"antseed.offers_top_n": None})
    assert not errs and settings.get("antseed.offers_top_n") == 3


def test_bad_override_value_falls_back_to_default(store):
    # A malformed stored value is skipped (get_overrides is fail-soft per key),
    # so the schema default wins — a bad override can never break ranking.
    with host_store._lock:
        c = host_store._connect()
        c.execute("INSERT INTO settings_overrides(key, value, updated_at)"
                  " VALUES (?,?,?)", ("codex.quota_429_shed", "not json{", 0))
        c.commit()
    settings.reload()
    assert settings.get("codex.quota_429_shed") == 3.0
