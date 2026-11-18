"""Model lines: definitions, the `line_<name>` observation fields, head moves and
the operator endpoints.

The engine-backed tests boot a real host so they exercise the actual Python/Lua
seam: a policy naming a line must rank the head family, follow a head move
without rebuilding the engine, and be refused when the line is not declared.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_lines  # noqa: E402
from llm_router_host import LLMRouterHost  # noqa: E402
from shim import create_app  # noqa: E402


def _doc(**lines):
    return {"version": 1, "lines": lines}


def _line(head, match=None, **kw):
    return {"head": head, "match": match, **kw}


@pytest.fixture
def lines_file(tmp_path, monkeypatch):
    """Install a small definitions file for the test and restore the shipped one
    afterwards, so no other test sees these lines or their moved heads."""
    def install(doc):
        path = tmp_path / "model_lines.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        monkeypatch.setenv("MODEL_LINES_FILE", str(path))
        return model_lines.configure()
    yield install
    monkeypatch.delenv("MODEL_LINES_FILE", raising=False)
    model_lines.configure()


def _host():
    h = LLMRouterHost(router_path=ROOT / "core" / "router.lua",
                      config_path=ROOT / "core" / "config.example.lua",
                      metrics_path=ROOT / "core" / "metrics.example.lua",
                      now_ms=lambda: 1_000_000)
    h.init()
    return h


def _policy(*lines):
    preds = [["is", model_lines.field_name(n)] for n in lines]
    pred = preds[0] if len(preds) == 1 else ["or", *preds]
    return ["policy", ["and", ["meets_req"], pred], ["field", "success_rate"],
            ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]


def _replica_sees(monkeypatch, **heads):
    """What a replica's refresh timer does after another process moved a head:
    the next store read returns the new rows."""
    import host_store
    rows = {line: {"family": family, "previous_family": None,
                   "updated_at": 1, "updated_by": "t"} for line, family in heads.items()}
    monkeypatch.setattr(host_store, "get_model_line_heads", lambda: rows)
    assert model_lines.reload() is True


def _ranked_families(host, policy):
    ranked, _ = host.rank({"prompt": "hi", "policy_ir": policy})
    return {r["candidate"]["model_family"] for r in ranked}


# ---- definitions ---------------------------------------------------------------

def test_shipped_definitions_are_valid_and_self_consistent():
    defs = model_lines.load_definitions(model_lines.DEFAULT_PATH)
    assert defs, "the shipped model_lines.json declares lines"
    for name, d in defs.items():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name)
        if d["pattern"] is not None:
            assert model_lines.version_key(d["pattern"], d["head"]) is not None, name
        assert d["max_cost_ratio"] >= 1
        assert 0 < d["min_pass_rate"] <= 1


@pytest.mark.parametrize("doc, message", [
    ({"version": 2, "lines": {}}, "version 1"),
    (_doc(**{"Bad-Name": _line("m-1")}), "invalid line name"),
    (_doc(a=_line("")), "`head` must be a family name"),
    (_doc(a=_line("m-1", "m-(")), "bad `match`"),
    (_doc(a=_line("m-1", r"^m-\d+$")), "capture group"),
    (_doc(a=_line("other-1", r"^m-(\d+)$")), "does not match its own"),
    (_doc(a=_line("m-1", None, max_cost_ratio=0)), "positive number"),
    (_doc(a=_line("m-1", None, min_pass_rate=1.5)), "must be <= 1"),
    (_doc(a=_line("m-1", None, surprise=1)), "unknown key"),
    ({"version": 1, "defaults": {"nope": 1}, "lines": {}}, "unknown default knob"),
])
def test_invalid_definitions_are_refused(doc, message):
    with pytest.raises(model_lines.ModelLineError, match=re.escape(message)):
        model_lines.parse_definitions(doc)


def test_knobs_resolve_line_over_file_defaults_over_builtins():
    defs = model_lines.parse_definitions({
        "version": 1, "defaults": {"max_latency_ms": 9000},
        "lines": {"a": _line("m-1"), "b": _line("m-2", None, max_latency_ms=4000)}})
    assert defs["a"]["max_latency_ms"] == 9000
    assert defs["b"]["max_latency_ms"] == 4000
    assert defs["a"]["max_cost_ratio"] == model_lines.QUALIFY_DEFAULTS["max_cost_ratio"]


def test_invalid_file_declares_no_lines(lines_file, tmp_path, monkeypatch):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("MODEL_LINES_FILE", str(bad))
    assert model_lines.configure() == {}
    assert model_lines.heads() == {}


# ---- successor discovery -------------------------------------------------------

CATALOG = [
    "gpt-6-sol", "gpt-6.1-sol", "gpt-6.2-sol", "gpt-6.2-sol-pro", "gpt-6.2-sol:batch",
    "gpt-6.2-astra", "openai/gpt-6.2-sol", "gpt-oss-120b",
    "deepseek-v4.1-flash-1017", "deepseek-v4.1-flash", "deepseek-v4.1-flash-1201",
    "deepseek-v4.2-flash-0105", "deepseek-v4-flash-0731",
    "grok-4.3", "grok-4.7", "grok-4.20", "grok-4.20-multi-agent", "grok-4.7-build-fast",
    "mistral-large-2512", "mistral-large-4", "mistral-large-4-0", "mistral-large-2606",
]


def _def(match, head="x"):
    return {"pattern": re.compile(match) if match else None, "head": head}


@pytest.mark.parametrize("match, head, expected", [
    # tier siblings (-pro, :batch, vendor-prefixed aliases) are not successors
    (r"^gpt-(\d+(?:\.\d+)?)-sol$", "gpt-6.1-sol", ["gpt-6.2-sol"]),
    (r"^gpt-(\d+(?:\.\d+)?)-sol$", "gpt-6-sol", ["gpt-6.2-sol", "gpt-6.1-sol"]),
    (r"^gpt-(\d+(?:\.\d+)?)-sol$", "gpt-6.2-sol", []),
    # version first, snapshot date second; the undated alias is not in the line
    (r"^deepseek-v(\d+(?:\.\d+)?)-flash-(\d{4})$", "deepseek-v4.1-flash-1017",
     ["deepseek-v4.2-flash-0105", "deepseek-v4.1-flash-1201"]),
    (r"^grok-(\d+\.\d+)$", "grok-4.3", ["grok-4.20", "grok-4.7"]),
    # a date-coded line ignores the differently-versioned `-4` naming
    (r"^mistral-large-(\d{4})$", "mistral-large-2512", ["mistral-large-2606"]),
])
def test_successors_are_strictly_newer_members_of_the_line(match, head, expected):
    assert model_lines.successors(_def(match), head, CATALOG) == expected


@pytest.mark.parametrize("line, catalog, expected", [
    # marketplace look-alikes without the dot must not outrank the real series
    ("glm", ["glm-5.3", "glm-52", "glm-5.4", "glm-6", "glm-5.3-flash"], ["glm-6", "glm-5.4"]),
    ("minimax", ["minimax-m3", "minimax-m27", "minimax-m25", "minimax-m3.5",
                 "minimax-m3.1-flash"], ["minimax-m3.5"]),
    ("openai_sol", ["gpt-6.1-sol", "gpt-62-sol", "gpt-6.2-sol", "gpt-6.2-sol-pro",
                    "gpt-6.2-sol:batch"], ["gpt-6.2-sol"]),
    ("anthropic_sonnet", ["claude-sonnet-5.5", "claude-sonnet-5-5", "claude-sonnet-6",
                          "claude-sonnet-4-6"], ["claude-sonnet-6"]),
    ("deepseek_flash", ["deepseek-v4.1-flash-1017", "deepseek-v4.1-flash",
                        "deepseek-v41-flash", "deepseek-v4.1-flash-1201"],
     ["deepseek-v4.1-flash-1201"]),
    ("gpt_oss", ["gpt-oss-120b", "gpt-oss-20b", "gpt-oss-240b"], []),
])
def test_shipped_patterns_follow_the_real_series_only(line, catalog, expected):
    d = model_lines.load_definitions(model_lines.DEFAULT_PATH)[line]
    assert model_lines.successors(d, d["head"], catalog) == expected


def test_successors_respect_limit_and_patternless_lines():
    d = _def(r"^gpt-(\d+(?:\.\d+)?)-sol$")
    assert model_lines.successors(d, "gpt-6-sol", CATALOG, limit=1) == ["gpt-6.2-sol"]
    assert model_lines.successors(_def(None), "gpt-oss-120b", CATALOG) == []
    # a forced head outside the pattern has no automatic successors
    assert model_lines.successors(d, "gpt-oss-120b", CATALOG) == []


# ---- the observation field -----------------------------------------------------

def test_policy_naming_a_line_ranks_only_its_head(lines_file):
    lines_file(_doc(alpha=_line("deepseek-v3"), beta=_line("llama-3.3-70b")))
    host = _host()
    names = {f["name"]: f for f in host.field_schema()}
    assert names["line_alpha"] == {"name": "line_alpha", "sort": "Bool",
                                   "group": "model", "core": False}
    assert _ranked_families(host, _policy("alpha")) == {"deepseek-v3"}
    assert _ranked_families(host, _policy("alpha", "beta")) == {
        "deepseek-v3", "llama-3.3-70b"}


def test_head_move_is_visible_to_a_running_engine(lines_file, monkeypatch):
    lines_file(_doc(alpha=_line("deepseek-v3")))
    host = _host()
    policy = _policy("alpha")
    before = host.normalize_policy(policy, admit=True)["policy_id"]
    assert _ranked_families(host, policy) == {"deepseek-v3"}

    _replica_sees(monkeypatch, alpha="hermes-3-405b")
    assert _ranked_families(host, policy) == {"hermes-3-405b"}
    # the policy term, and so its identity, did not change
    assert host.normalize_policy(policy, admit=True)["policy_id"] == before


def test_every_live_engine_follows_a_move_and_dead_ones_are_dropped(lines_file, monkeypatch):
    import gc
    lines_file(_doc(alpha=_line("deepseek-v3")))
    first, second = _host(), _host()
    policy = _policy("alpha")
    _replica_sees(monkeypatch, alpha="hermes-3-405b")
    assert _ranked_families(first, policy) == {"hermes-3-405b"}
    assert _ranked_families(second, policy) == {"hermes-3-405b"}

    # an engine built after the move starts on the moved head, not the seed
    assert _ranked_families(_host(), policy) == {"hermes-3-405b"}

    del second
    gc.collect()
    _replica_sees(monkeypatch, alpha="llama-3.3-70b")        # must not touch a dead engine
    assert _ranked_families(first, policy) == {"llama-3.3-70b"}


def test_engine_that_fails_to_sync_does_not_block_the_others(lines_file, monkeypatch):
    lines_file(_doc(alpha=_line("deepseek-v3")))
    broken, healthy = _host(), _host()
    monkeypatch.setattr(broken, "sync_model_line_heads",
                        lambda heads: (_ for _ in ()).throw(RuntimeError("lua busy")))
    _replica_sees(monkeypatch, alpha="hermes-3-405b")
    policy = _policy("alpha")
    assert _ranked_families(healthy, policy) == {"hermes-3-405b"}
    assert _ranked_families(broken, policy) == {"deepseek-v3"}   # kept its last table


def test_head_absent_from_catalog_ranks_nothing(lines_file):
    lines_file(_doc(alpha=_line("not-a-real-family")))
    assert _ranked_families(_host(), _policy("alpha")) == set()


def test_undeclared_line_fails_admission(lines_file):
    lines_file(_doc(alpha=_line("deepseek-v3")))
    with pytest.raises(Exception, match="undeclared field 'line_missing'"):
        _host().normalize_policy(_policy("missing"), admit=True)


def test_model_meta_exposes_line_membership(lines_file):
    lines_file(_doc(alpha=_line("deepseek-v3")))
    meta = _host().model_meta()
    assert meta["deepseek-v3"]["line_alpha"] is True
    # only the head carries the mark; other families are not padded with false
    assert "line_alpha" not in meta["hermes-3-405b"]


def test_no_lines_declared_adds_no_fields(lines_file):
    lines_file(_doc())
    assert not [f for f in _host().field_schema() if f["name"].startswith("line_")]


# ---- head moves (store-backed) -------------------------------------------------

def test_promote_rollback_and_audit_trail(host_store_clean, lines_file):
    lines_file(_doc(alpha=_line("m-1", r"^m-(\d+)$")))
    assert model_lines.reload() is True
    assert model_lines.head_of("alpha") == "m-1"

    moved = model_lines.promote("alpha", "m-2", actor="tester", detail={"why": "t"})
    assert moved == {"ok": True, "family": "m-2", "previous_family": "m-1"}
    assert model_lines.head_of("alpha") == "m-2"

    snap = model_lines.snapshot()
    row = snap["lines"][0]
    assert (row["head"], row["seed"], row["source"], row["previous"],
            row["updated_by"]) == ("m-2", "m-1", "store", "m-1", "tester")
    assert snap["store_ok"] is True
    assert snap["events"][0]["kind"] == "promoted"
    assert snap["events"][0]["detail"] == {"why": "t"}

    back = model_lines.rollback("alpha", actor="tester")
    assert back == {"ok": True, "family": "m-1", "previous_family": "m-2"}
    assert model_lines.head_of("alpha") == "m-1"
    assert [e["kind"] for e in host_store_clean.model_line_events("alpha")] == [
        "rolled_back", "promoted"]


def test_promote_is_a_compare_and_set(host_store_clean, lines_file):
    lines_file(_doc(alpha=_line("m-1", r"^m-(\d+)$")))
    assert model_lines.promote("alpha", "m-2", actor="a", expected_current="m-1")["ok"]
    stale = model_lines.promote("alpha", "m-3", actor="b", expected_current="m-1")
    assert stale == {"ok": False, "reason": "conflict", "family": "m-2"}
    same = model_lines.promote("alpha", "m-2", actor="b")
    assert same == {"ok": False, "reason": "unchanged", "family": "m-2"}
    assert model_lines.head_of("alpha") == "m-2"


def test_promote_refuses_families_outside_the_line_unless_forced(
        host_store_clean, lines_file):
    lines_file(_doc(alpha=_line("m-1", r"^m-(\d+)$")))
    with pytest.raises(model_lines.ModelLineError, match="does not match"):
        model_lines.promote("alpha", "other-9", actor="t")
    with pytest.raises(model_lines.ModelLineError, match="unknown line"):
        model_lines.promote("nope", "m-2", actor="t")
    assert model_lines.promote("alpha", "other-9", actor="t", force=True)["ok"]


def test_rollback_without_a_previous_head(host_store_clean, lines_file):
    lines_file(_doc(alpha=_line("m-1")))
    assert model_lines.rollback("alpha", actor="t") == {
        "ok": False, "reason": "no_previous", "family": "m-1"}


def test_store_outage_keeps_last_known_heads(host_store_clean, lines_file, monkeypatch):
    lines_file(_doc(alpha=_line("m-1", r"^m-(\d+)$")))
    model_lines.promote("alpha", "m-2", actor="t")
    monkeypatch.setattr(host_store_clean, "get_model_line_heads", lambda: None)
    assert model_lines.reload() is False
    assert model_lines.head_of("alpha") == "m-2"     # not snapped back to the seed
    assert model_lines.snapshot(events=0)["store_ok"] is False


def test_stored_heads_for_undeclared_or_malformed_lines_are_ignored(
        host_store_clean, lines_file):
    lines_file(_doc(alpha=_line("m-1"), beta=_line("n-1")))
    host_store_clean.set_model_line_head("ghost", "m-9", seed_family=None, actor="t")
    host_store_clean.set_model_line_head("beta", "bad family!", seed_family="n-1", actor="t")
    assert model_lines.reload() is True
    assert model_lines.heads() == {"alpha": "m-1", "beta": "n-1"}


# ---- operator endpoints --------------------------------------------------------

def test_lines_endpoints_move_heads_and_report_conflicts(host_store_clean, lines_file):
    lines_file(_doc(alpha=_line("deepseek-v3", r"^deepseek-v(\d+)$")))
    client = TestClient(create_app(_host()))

    state = client.get("/x/lines").json()
    assert state["lines"][0]["head"] == "deepseek-v3"

    r = client.post("/x/lines/alpha/promote",
                    json={"family": "deepseek-v4", "actor": "op",
                          "expected_current": "deepseek-v3"})
    assert r.status_code == 200 and r.json()["previous_family"] == "deepseek-v3"
    assert client.get("/x/lines").json()["lines"][0]["head"] == "deepseek-v4"

    r = client.post("/x/lines/alpha/promote",
                    json={"family": "deepseek-v5", "expected_current": "deepseek-v3"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "model_line_conflict"

    r = client.post("/x/lines/alpha/promote", json={"family": "llama-3.3-70b"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_line"
    assert client.post("/x/lines/nope/rollback", json={}).status_code == 400

    r = client.post("/x/lines/alpha/rollback", json={"actor": "op"})
    assert r.status_code == 200 and r.json()["family"] == "deepseek-v3"


def test_lines_endpoints_are_admin_gated(tmp_path, monkeypatch, lines_file):
    import control_plane_client as cp
    lines_file(_doc(alpha=_line("deepseek-v3")))
    monkeypatch.delenv("ROUTER_ADMIN_ALLOW_UNAUTHENTICATED", raising=False)
    monkeypatch.setattr(cp, "CONTROL_PLANE_INTERNAL_SECRET", "s3cret")
    client = TestClient(create_app(_host()))
    assert client.get("/x/lines").status_code == 403
    assert client.post("/x/lines/alpha/promote",
                       json={"family": "x"}).status_code == 403
    assert client.post("/x/lines/alpha/rollback", json={}).status_code == 403
    ok = client.get("/x/lines", headers={"x-internal-secret": "s3cret"})
    assert ok.status_code == 200
