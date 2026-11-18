"""The surfaces around model lines: the admin-dashboard passthrough, the replica
refresh loop, the shipped example suite and the qualifier CLI."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["CALLER_KEYS_JSON"] = '{"internal":"default"}'
os.environ["CALLER_KEYS_SHA256_JSON"] = "{}"
os.environ["CALLER_KEYS_BOOTSTRAP_JSON"] = "{}"
os.environ["DASHBOARD_TRUSTED_USER_HEADER"] = ""

import auth_proxy  # noqa: E402
import model_line_qualifier as qualifier  # noqa: E402
import model_lines  # noqa: E402
import serve  # noqa: E402


# ---- dashboard passthrough -----------------------------------------------------

class _RecordingRouter:
    """Stands in for the ingress's upstream client and records what it was sent."""

    def __init__(self, status=200, body=None, raises=None):
        self.calls: list[dict] = []
        self.status = status
        self.body = body if body is not None else {"ok": True}
        self.raises = raises

    async def request(self, method, url, json=None, headers=None, timeout=None):
        self.calls.append({"method": method, "url": url, "json": json,
                           "headers": dict(headers or {})})
        if self.raises is not None:
            raise self.raises
        return httpx.Response(self.status, json=self.body)


def _admin(monkeypatch, user="alice"):
    monkeypatch.setattr(auth_proxy, "DASHBOARD_SESSION_SECRET", "test-dashboard-session-secret")
    client = TestClient(auth_proxy.app)
    client.cookies.set(auth_proxy.DASHBOARD_COOKIE_NAME,
                       auth_proxy._make_dashboard_session(user))
    return client


def test_dashboard_lines_require_an_admin_session():
    client = TestClient(auth_proxy.app)
    assert client.get("/dashboard/api/lines").status_code == 401
    for action in ("promote", "rollback"):
        r = client.post(f"/dashboard/api/lines/alpha/{action}", json={"family": "m-2"})
        assert r.status_code == 401, action


def test_dashboard_promote_forwards_with_secret_and_session_actor(monkeypatch):
    import control_plane_client as cp
    router = _RecordingRouter(body={"ok": True, "family": "m-2", "previous_family": "m-1"})
    monkeypatch.setattr(auth_proxy, "_client", router)
    monkeypatch.setattr(cp, "CONTROL_PLANE_INTERNAL_SECRET", "s3cret")
    client = _admin(monkeypatch)

    r = client.post("/dashboard/api/lines/alpha/promote",
                    json={"family": "m-2", "expected_current": "m-1",
                          "actor": "someone-else"})
    assert r.status_code == 200 and r.json()["family"] == "m-2"
    call = router.calls[0]
    assert call["method"] == "POST" and call["url"].endswith("/x/lines/alpha/promote")
    assert call["headers"]["x-internal-secret"] == "s3cret"
    # the audit actor is the authenticated viewer, not what the body claimed
    assert call["json"]["actor"] == "dashboard:alice"
    assert call["json"]["expected_current"] == "m-1"

    assert client.get("/dashboard/api/lines").status_code == 200
    assert router.calls[1]["method"] == "GET"
    assert router.calls[1]["url"].endswith("/x/lines")


def test_dashboard_relays_router_conflicts_and_rejects_bad_input(monkeypatch):
    router = _RecordingRouter(status=409, body={"error": {"code": "model_line_conflict"}})
    monkeypatch.setattr(auth_proxy, "_client", router)
    client = _admin(monkeypatch)

    r = client.post("/dashboard/api/lines/alpha/rollback", json={})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "model_line_conflict"

    assert client.post("/dashboard/api/lines/Bad-Name/promote",
                       json={"family": "m-2"}).status_code == 400
    assert client.post("/dashboard/api/lines/alpha/promote",
                       content=b"[1]", headers={"content-type": "application/json"}
                       ).status_code == 400
    assert len(router.calls) == 1      # nothing malformed reached the router


def test_dashboard_says_when_a_move_may_have_been_applied(monkeypatch):
    client = _admin(monkeypatch)
    # sent, no answer: the router may have committed the move
    monkeypatch.setattr(auth_proxy, "_client",
                        _RecordingRouter(raises=httpx.ReadTimeout("slow")))
    r = client.post("/dashboard/api/lines/alpha/rollback", json={})
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "model_line_outcome_unknown"
    # never connected: nothing happened, a plain retry is safe
    monkeypatch.setattr(auth_proxy, "_client",
                        _RecordingRouter(raises=httpx.ConnectError("refused")))
    r = client.post("/dashboard/api/lines/alpha/rollback", json={})
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream"
    # a read has no outcome to be unsure about
    monkeypatch.setattr(auth_proxy, "_client",
                        _RecordingRouter(raises=httpx.ReadTimeout("slow")))
    assert client.get("/dashboard/api/lines").status_code == 502


# ---- replica refresh -----------------------------------------------------------

def test_refresh_task_rereads_heads_and_survives_failures(monkeypatch):
    monkeypatch.setenv("MODEL_LINES_REFRESH_S", "0.01")
    monkeypatch.setattr(model_lines, "definitions", lambda: {"alpha": {}})
    seen = []

    def reload():
        seen.append(1)
        if len(seen) == 1:
            raise RuntimeError("store hiccup")
        return True

    monkeypatch.setattr(model_lines, "reload", reload)

    async def scenario():
        task = serve.start_model_lines_refresh()
        assert task is not None
        for _ in range(200):
            if len(seen) >= 3:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert len(seen) >= 3              # kept going after the failed read


@pytest.mark.parametrize("interval, lines", [("0", {"alpha": {}}), ("30", {}),
                                             ("-5", {"alpha": {}})])
def test_refresh_task_is_off_without_lines_or_interval(monkeypatch, interval, lines):
    monkeypatch.setenv("MODEL_LINES_REFRESH_S", interval)
    monkeypatch.setattr(model_lines, "definitions", lambda: lines)

    async def scenario():
        return serve.start_model_lines_refresh()

    assert asyncio.run(scenario()) is None


# ---- shipped suite and CLI -----------------------------------------------------

def test_shipped_example_suite_loads():
    suites = qualifier.load_suites([ROOT / "evaluations" / "line_suites"])
    assert [s["name"] for s in suites] == ["example-ticket-triage"]
    assert suites[0]["request"]["timeout_ms"] == 22000
    good = '{"label": "bug", "confidence": 90, "summary": "export is broken"}'
    assert qualifier.answer_is_valid(good, suites[0]["expect"])


def _cli(*args, env=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "qualify_model_lines.py"), *args],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "LINE_QUALIFIER_API_KEY": "", **(env or {})})


def test_cli_requires_router_url_and_suites():
    assert _cli("--suites", str(ROOT / "evaluations" / "line_suites"),
                env={"LINE_QUALIFIER_ROUTER_URL": ""}).returncode == 2
    assert _cli("--router-url", "http://127.0.0.1:9",
                env={"LINE_QUALIFIER_SUITES": ""}).returncode == 2


def test_cli_reports_an_unreachable_router_as_one_json_line():
    # port 9 (discard) refuses the connection: the run cannot start at all
    done = _cli("--router-url", "http://127.0.0.1:9", "--mode", "report",
                "--suites", str(ROOT / "evaluations" / "line_suites"))
    assert done.returncode == 1
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["ok"] is False and out["error"]


def test_cli_reports_a_bad_suite_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"name": "x", "prompts": []}', encoding="utf-8")
    done = _cli("--router-url", "http://127.0.0.1:9", "--suites", str(bad))
    assert done.returncode == 1
    assert json.loads(done.stdout.strip())["error"].startswith("suites:")
