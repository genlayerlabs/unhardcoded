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


def _run_node_test(rel_path: str) -> None:
    proc = subprocess.run(
        # Pin the actual test file. Node 26 stopped resolving a directory passed
        # to `--test` even though older releases discovered db.test.js there.
        ["node", "--test", rel_path],
        cwd=_REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"antseed node tests failed ({rel_path}):\n{proc.stdout}\n{proc.stderr}")


def test_antseed_node_unit_tests():
    """The sidecar's DATABASE_URL parser (db.js) handles
    both the compose `postgres://` URL and the prod libpq kv conninfo."""
    _run_node_test("antseed/db.test.js")


def test_antseed_control_amount_cap():
    """The control server's deposit-amount guard (antseed/amount.js). /deposit is
    now called autonomously by the router's wallet keeper, so the per-deposit
    ceiling has to hold server-side, not only in the caller."""
    _run_node_test("antseed/amount.test.js")
