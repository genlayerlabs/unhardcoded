"""Bridge the antseed sidecar's Node unit tests into the Python suite.

The sidecar (write-market.js / write-status.js / control.js) is Node, and its
DATABASE_URL handling (antseed/db.js) is where the prod outage lived: compose
feeds a `postgres://` URL, prod feeds a libpq kv conninfo, and only the kv path
broke node-postgres (`getaddrinfo ENOTFOUND base`). compose-only testing could
never catch it because the failing format never appears in dev. db.test.js
exercises BOTH formats; running it from pytest means it executes wherever the
suite runs, not just by hand.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not on PATH (antseed sidecar tests)")


def test_antseed_node_unit_tests():
    """`node --test antseed/` — the sidecar's DATABASE_URL parser (db.js) handles
    both the compose `postgres://` URL and the prod libpq kv conninfo."""
    proc = subprocess.run(
        ["node", "--test", "antseed/"],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"antseed node tests failed:\n{proc.stdout}\n{proc.stderr}")
