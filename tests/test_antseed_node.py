"""Bridge the antseed sidecar's Node unit tests into the Python suite.

The sidecar (write-market.js / write-status.js / control.js) is Node, and its
DATABASE_URL handling (antseed/db.js) is where the prod outage lived: compose
feeds a `postgres://` URL, prod feeds a libpq kv conninfo, and only the kv path
broke node-postgres (`getaddrinfo ENOTFOUND base`). compose-only testing could
never catch it because the failing format never appears in dev. db.test.js
exercises BOTH formats; running it from pytest means it executes wherever the
suite runs, not just by hand.
"""
import re
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


def test_antseed_control_queue_admission():
    """The control server's mutation queue (antseed/queue.js). Its time bound is
    what makes the sidecar's worst case FINITE — the precondition for the wallet
    keeper picking a client timeout that strictly exceeds it. Unbounded, a
    deposit queued behind a long reclaim phase timed out on the caller while
    still executing here: real USDC moved and the ledger recorded nothing."""
    _run_node_test("antseed/queue.test.js")


def test_antseed_reclaim_channel_selection():
    """The reclaim channel-id selector (antseed/ids.js) — what makes the keeper's
    per-cycle transaction cap and its dust filter binding rather than decorative.
    reclaim.mjs itself deep-imports @antseed/cli internals that exist only in the
    sidecar image, so the selection logic lives here where it can be tested."""
    _run_node_test("antseed/ids.test.js")


def test_antseed_pre_broadcast_classification():
    """The pre-broadcast classifier (antseed/broadcast.js) — the thing that
    decides whether a failed `antseed buyer deposit` is recorded as `failed`
    (costs the router nothing) or `unknown` (costs a slot of the daily cap).

    It lives in the sidecar rather than the keeper because only the sidecar has
    the evidence: the keeper is handed `(stderr || stdout)[:600]`, one stream and
    truncated, and the buyer CLI prints its transaction hash on the other one.
    The fixtures are the real prod output byte for byte."""
    _run_node_test("antseed/broadcast.test.js")


_LOCAL_IMPORT = re.compile(
    r"""(?:require\(\s*|from\s+)['"](\./[^'"]+)['"]""")


def test_every_local_import_is_shipped_in_the_sidecar_image():
    """Every `require('./x.js')` in a shipped sidecar module resolves to another
    shipped file.

    The node tests above run against the REPO, so they pass whether or not a file
    reaches the image. That gap shipped a control.js requiring ./ids.js into an
    image built from an explicit COPY list that named neither ids.js nor queue.js:
    the control server died at import (`Cannot find module './ids.js'`), :8379
    never bound, and every wallet endpoint 502'd — with the only symptom a
    Cloudflare error page on the deposit button. Nothing else noticed, because the
    market/status writers do not import it and the buyer proxy is a separate
    process.
    """
    antseed = _REPO_ROOT / "antseed"
    shipped = {p.name for p in antseed.iterdir()
               if p.suffix in (".js", ".mjs") and not p.name.endswith(".test.js")}
    assert "control.js" in shipped and "ids.js" in shipped, shipped

    missing = []
    for name in sorted(shipped):
        for spec in _LOCAL_IMPORT.findall((antseed / name).read_text()):
            target = spec[2:]  # drop the leading "./"
            if target not in shipped:
                missing.append(f"{name} imports {spec!r}, which is not shipped")
    assert not missing, "\n".join(missing)

    # The COPY must be pattern-based; an explicit list is what drifted.
    copy_lines = [ln for ln in (_REPO_ROOT / "Dockerfile.antseed").read_text().splitlines()
                  if ln.startswith("COPY ") and "/usr/local/lib/antseed/" in ln]
    assert copy_lines, "Dockerfile.antseed has no COPY into /usr/local/lib/antseed/"
    assert any("antseed/*.js" in ln for ln in copy_lines), (
        "COPY must glob antseed/*.js — naming files individually is how ids.js "
        f"and queue.js were left out of the image: {copy_lines}")
