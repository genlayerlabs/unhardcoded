"""Test config for the operational store.

The host store is Postgres now, so the tests that touch it run against a real
Postgres (the compose `postgres` service, or any DATABASE_URL). For isolation
against the shared DB each store-using test TRUNCATEs first. Pure-logic tests
(translation, ranking math, validation) don't use the store fixture and need no
Postgres.

Dev convenience: if DATABASE_URL is unset we default to a local throwaway
Postgres on :55432 (e.g. `docker run -p 55432:5432 -e POSTGRES_PASSWORD=test
postgres:16`); CI/compose sets DATABASE_URL to the compose service.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "DATABASE_URL", "postgresql://postgres:test@localhost:55432/hoststore")

import host_store  # noqa: E402


@pytest.fixture
def host_store_clean():
    """Truncate the operational store before the test (isolation against the
    shared Postgres). Skips the test if Postgres is unreachable."""
    try:
        host_store.reset()
        host_store.truncate_all_for_tests()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"host store Postgres unavailable: {exc}")
    yield host_store
