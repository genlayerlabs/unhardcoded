"""The model-line qualifier: answer validation, the three gates, the consecutive
pass requirement, and what each mode is allowed to write.

Transport is faked at the `post`/`get` boundary (the router's OpenAI-compatible
surface); the store is the real one, so promotions, proposals and rollbacks are
asserted on what an operator would actually read back.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import model_line_qualifier as q  # noqa: E402
import model_lines  # noqa: E402

GOOD = '{"decision": "authentic", "confidence": 80, "summary": "consistent"}'
SUITE = q.parse_suite({
    "name": "memories",
    "request": {"temperature": 0.7, "max_tokens": 4000,
                "first_token_timeout_ms": 7000, "timeout_ms": 22000},
    "expect": {"type": "json", "required": {
        "decision": {"enum": ["authentic", "suspicious", "inconclusive"]},
        "confidence": {"type": "integer", "min": 0, "max": 100},
        "summary": {"type": "string"}}},
    "prompts": ["p1", "p2", "p3"],
})


@pytest.fixture
def lines(tmp_path, monkeypatch, host_store_clean):
    def install(**lines_):
        path = tmp_path / "model_lines.json"
        path.write_text(json.dumps({"version": 1, "lines": lines_}), encoding="utf-8")
        monkeypatch.setenv("MODEL_LINES_FILE", str(path))
        model_lines.configure()
        return host_store_clean
    yield install
    monkeypatch.delenv("MODEL_LINES_FILE", raising=False)
    model_lines.configure()


class FakeRouter:
    """`behaviour[family]` is a dict: text, cost, status. Records every body."""

    def __init__(self, families, behaviour):
        self.families = families
        self.behaviour = behaviour
        self.bodies: list[dict] = []

    def get(self, path):
        assert path == "/v1/models"
        return 200, {"data": [{"id": "profile:default"}]
                     + [{"id": f"family:{f}"} for f in self.families]}

    def post(self, body):
        self.bodies.append(body)
        family = body["policy_ir"][1][-1][1]
        b = self.behaviour.get(family, {})
        status = b.get("status", 200)
        if body["messages"][0]["content"] in b.get("fail_on", ()):
            status = 504
        if status != 200:
            return status, {"error": {"code": b.get("code", "timeout")}}
        return 200, {
            "choices": [{"message": {"content": b.get("text", GOOD)},
                         "finish_reason": b.get("finish", "stop")}],
            "x_router": {"model_family": b.get("served", family),
                         "provider": "openrouter", "cost_usd": b.get("cost", 0.001)}}

    def run(self, mode, **kw):
        return q.run(post=self.post, get=self.get, suites=[SUITE], mode=mode,
                     workers=2, **kw)


def _line(head, **kw):
    return {"head": head, "match": r"^m-(\d+)$", **kw}


# ---- answers -------------------------------------------------------------------

@pytest.mark.parametrize("text, ok", [
    (GOOD, True),
    ("```json\n" + GOOD + "\n```", True),            # fenced, as lenient callers accept
    ("Sure! " + GOOD + " Hope that helps.", True),
    ('{"decision": "maybe", "confidence": 80, "summary": "x"}', False),
    ('{"decision": "authentic", "confidence": "80", "summary": "x"}', False),
    ('{"decision": "authentic", "confidence": 180, "summary": "x"}', False),
    ('{"decision": "authentic", "confidence": true, "summary": "x"}', False),
    ('{"decision": "authentic", "confidence": 80}', False),
    ('{"decision": "authentic", "confidence": 80, "summary": "  "}', False),
    ("I cannot help with that.", False),
    ("[1, 2]", False),
    ("", False),
])
def test_json_answers_are_checked_against_the_suite(text, ok):
    assert q.answer_is_valid(text, SUITE["expect"]) is ok


def test_text_and_regex_expectations():
    assert q.answer_is_valid("anything", {"type": "text"})
    assert not q.answer_is_valid("   ", {"type": "text"})
    assert q.answer_is_valid("result: true", {"type": "regex", "pattern": r"\btrue\b"})
    assert not q.answer_is_valid("result: maybe", {"type": "regex", "pattern": r"\btrue\b"})


@pytest.mark.parametrize("doc, message", [
    ({"name": "Bad Name", "prompts": ["p"]}, "short slug"),
    ({"name": "s", "prompts": []}, "non-empty list"),
    ({"name": "s", "prompts": ["p"], "request": {"model": "x"}}, "may only set"),
    ({"name": "s", "prompts": ["p"], "request": {"policy_ir": []}}, "may only set"),
    ({"name": "s", "prompts": ["p"], "expect": {"type": "xml"}}, "expect.type"),
    ({"name": "s", "prompts": ["p"], "expect": {"type": "regex", "pattern": "("}}, "bad expect.pattern"),
    # no pattern would compile to "match anything", a refusal included
    ({"name": "s", "prompts": ["p"], "expect": {"type": "regex"}}, "non-empty string"),
    ({"name": "s", "prompts": ["p"], "expect": {"type": "regex", "pattern": ""}}, "non-empty string"),
    ({"name": "s", "prompts": ["p"], "repeats": 9}, "repeats"),
])
def test_invalid_suites_are_refused(doc, message):
    with pytest.raises(q.SuiteError, match=message):
        q.parse_suite(doc)


def test_load_suites_reads_directories_and_rejects_duplicates(tmp_path):
    for i in (1, 2):
        (tmp_path / f"s{i}.json").write_text(
            json.dumps({"name": f"s{i}", "prompts": ["p"]}), encoding="utf-8")
    assert [s["name"] for s in q.load_suites([tmp_path])] == ["s1", "s2"]
    (tmp_path / "dup.json").write_text(
        json.dumps({"name": "s1", "prompts": ["p"]}), encoding="utf-8")
    with pytest.raises(q.SuiteError, match="duplicate"):
        q.load_suites([tmp_path])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(q.SuiteError, match="no prompt suites"):
        q.load_suites([empty])


# ---- probes --------------------------------------------------------------------

def test_probe_isolates_one_family_and_carries_the_suite_request():
    router = FakeRouter(["m-1"], {})
    summary = q.probe_family(router.post, "m-1", [SUITE], "qualify-m-1")
    assert summary["probes"] == 3 and summary["pass_rate"] == 1.0
    body = router.bodies[0]
    assert body["policy_ir"][1] == ["and", ["meets_req"], ["not", ["is", "disabled"]],
                                    ["family_eq", "m-1"]]
    assert body["policy_ir"][-1] == ["always", {"action": "next_candidate"}]
    assert (body["first_token_timeout_ms"], body["timeout_ms"]) == (7000, 22000)
    assert body["messages"] == [{"role": "user", "content": "p1"}]
    assert body["model"] == "policy:qualify-m-1"


def test_probe_failures_are_counted_not_raised():
    def boom(body):
        raise ConnectionError("down")
    summary = q.probe_family(boom, "m-1", [SUITE], "x")
    assert summary["passed"] == 0 and summary["pass_rate"] == 0.0
    assert summary["errors"] == {"memories:ConnectionError": 3}
    assert summary["mean_cost_usd"] is None and summary["latency_p95_ms"] is None
    # the network failed, not the model: nothing may be concluded from it
    assert summary["inconclusive"] == 3 and not q.conclusive(summary)


@pytest.mark.parametrize("status, code, inconclusive", [
    (429, "caller_rate_limit", True),        # our own key, before any provider
    (402, "key_budget_exhausted", True),
    (503, "router_overloaded", True),
    (401, "consumer_auth", True),
    (502, "upstream", True),
    (429, "exhausted", False),               # the provider's limit: the route's problem
    (504, "timeout", False),
    (400, "invalid_request_error", False),
])
def test_ingress_rejections_are_inconclusive_model_failures_are_not(status, code, inconclusive):
    router = FakeRouter(["m-1"], {"m-1": {"status": status, "code": code}})
    summary = q.probe_family(router.post, "m-1", [SUITE], "x")
    assert summary["passed"] == 0
    assert (summary["inconclusive"] == 3) is inconclusive
    assert q.conclusive(summary) is (not inconclusive)


def test_exhausted_output_budget_is_reported_as_truncated():
    router = FakeRouter(["m-1"], {"m-1": {"text": "", "finish": "length"}})
    summary = q.probe_family(router.post, "m-1", [SUITE], "x")
    assert summary["passed"] == 0
    assert summary["errors"] == {"memories:truncated": 3}


def test_answer_from_another_family_does_not_credit_the_candidate():
    router = FakeRouter(["m-2"], {"m-2": {"served": "m-1"}})
    summary = q.probe_family(router.post, "m-2", [SUITE], "x")
    assert summary["passed"] == 0
    assert summary["errors"] == {"memories:served_other_family": 3}


def test_cost_is_unknown_unless_every_answered_probe_reported_one():
    results = [{"suite": "s", "status": 200, "latency_ms": 10, "valid": True, "cost_usd": 0.002},
               {"suite": "s", "status": 200, "latency_ms": 30, "valid": True, "cost_usd": None}]
    assert q.summarize("m", results)["mean_cost_usd"] is None
    results[1]["cost_usd"] = 0.004
    s = q.summarize("m", results)
    assert s["mean_cost_usd"] == 0.003
    assert (s["latency_p50_ms"], s["latency_p90_ms"], s["latency_p95_ms"]) == (10, 30, 30)


def test_p90_ignores_the_single_slowest_probe_of_a_run():
    results = [{"suite": "s", "status": 200, "latency_ms": ms, "valid": True, "cost_usd": 0.001}
               for ms in [*range(1000, 16000, 1000), 40000]]        # 16 probes
    s = q.summarize("m", results)
    assert (s["latency_p90_ms"], s["latency_p95_ms"]) == (15000, 40000)


# ---- gates ---------------------------------------------------------------------

DEF = {"min_pass_rate": 0.9, "min_suite_pass_rate": 0.5, "max_latency_ms": 15000,
       "max_latency_ratio": 1.25, "max_cost_ratio": 1.1}


def _summary(pass_rate=1.0, p90=5000, cost=0.001, probes=16, suites=None):
    passed = round(pass_rate * probes)
    return {"probes": probes, "pass_rate": pass_rate, "latency_p90_ms": p90,
            "mean_cost_usd": cost, "costs": [cost] * probes, "inconclusive": 0,
            "suites": suites or {"s": {"probes": probes, "passed": passed}}}


@pytest.mark.parametrize("candidate, head, qualified, failing", [
    (_summary(), _summary(), True, None),
    (_summary(cost=0.0011), _summary(cost=0.001), True, None),       # at the ratio
    (_summary(cost=0.0012), _summary(cost=0.001), False, "cost"),
    (_summary(pass_rate=0.9375), _summary(), True, None),            # 15/16: noise
    (_summary(pass_rate=0.875), _summary(), False, "answers"),       # 14/16
    # over the absolute bound, and slower than the head by more than the ratio
    (_summary(p90=15001), _summary(p90=5000), False, "latency"),
    # over the absolute bound but no slower than a head that already is
    (_summary(p90=24000), _summary(p90=20000), True, None),
    (_summary(p90=25001), _summary(p90=20000), False, "latency"),
    # the relative allowance needs a head latency to be relative to
    (_summary(p90=24000), _summary(p90=None, pass_rate=0.0, cost=None), False, "latency"),
    (_summary(p90=None, pass_rate=0.0), _summary(), False, "answers"),
    (_summary(cost=None), _summary(), False, "cost"),
    (_summary(), _summary(cost=None), False, "cost"),                 # no baseline
    # a degraded head is still a baseline: it is the one worth replacing
    (_summary(), _summary(pass_rate=0.5), True, None),
    (_summary(cost=0.0012), _summary(pass_rate=0.5, cost=0.001), False, "cost"),
    (_summary(cost=0.0), _summary(cost=0.0), True, None),
    (_summary(cost=0.001), _summary(cost=0.0), False, "cost"),
])
def test_gates_fail_closed(candidate, head, qualified, failing):
    verdict = q.judge(candidate, head, DEF)
    assert verdict["qualified"] is qualified
    if failing:
        assert verdict["gates"][failing]["ok"] is False


def test_whole_suite_failure_cannot_hide_behind_the_aggregate():
    # 9 general prompts pass, the single specialist prompt fails: 90% overall
    candidate = _summary(pass_rate=0.9, probes=10, suites={
        "general": {"probes": 9, "passed": 9}, "specialist": {"probes": 1, "passed": 0}})
    verdict = q.judge(candidate, _summary(probes=10), DEF)
    assert verdict["qualified"] is False
    assert verdict["gates"]["answers"]["suites"] == {"general": 1.0, "specialist": 0.0}
    # the same overall rate spread across suites is just noise
    spread = _summary(pass_rate=0.9, probes=10, suites={
        "general": {"probes": 6, "passed": 5}, "specialist": {"probes": 4, "passed": 4}})
    assert q.judge(spread, _summary(probes=10), DEF)["qualified"] is True


def test_cost_is_compared_over_the_same_probes_and_counts_unusable_answers():
    # Nine prompts cost both sides $0.001. On the tenth the head answers for
    # $0.01 and the candidate charges $0.02 for an answer that is not valid.
    head = _summary(probes=10)
    head["costs"] = [0.001] * 9 + [0.01]
    candidate = _summary(pass_rate=0.9, probes=10)
    candidate["costs"] = [0.001] * 9 + [0.02]
    cost = q.judge(candidate, head, DEF)["gates"]["cost"]
    assert cost["ok"] is False and cost["paired"] == 10
    assert cost["ratio"] == round(0.029 / 0.019, 4)


def test_cost_pairs_only_probes_both_sides_reported():
    head = _summary(probes=4)
    head["costs"] = [0.004, None, 0.004, 0.004]        # one head probe had no response
    candidate = _summary(probes=4)
    candidate["costs"] = [0.004, 0.5, 0.004, 0.004]    # its cost there has no counterpart
    cost = q.judge(candidate, head, DEF)["gates"]["cost"]
    assert cost["ok"] is True and cost["paired"] == 3 and cost["ratio"] == 1.0
    # too few pairs to say anything: fail closed
    head["costs"] = [0.004, None, None, None]
    assert q.judge(candidate, head, DEF)["gates"]["cost"]["reason"] == "no_head_baseline"


def test_consecutive_passes_stop_at_the_first_failure():
    def ev(family, kind, head="m-1"):
        return {"family": family, "kind": kind, "previous_family": head}
    events = [ev("m-2", "candidate_passed"),
              ev("m-3", "candidate_failed"),          # another family
              ev("m-2", "candidate_passed"),
              ev("m-2", "candidate_failed"),
              ev("m-2", "candidate_passed")]
    assert q.consecutive_passes(events, "m-2", "m-1") == 2
    assert q.consecutive_passes(events, "m-3", "m-1") == 0
    assert q.consecutive_passes(events, "m-9", "m-1") == 0


def test_passes_against_an_earlier_head_do_not_count():
    def ev(head):
        return {"family": "m-3", "kind": "candidate_passed", "previous_family": head}
    events = [ev("m-2"), ev("m-1"), ev("m-1")]        # newest first
    assert q.consecutive_passes(events, "m-3", "m-2") == 1
    assert q.consecutive_passes(events, "m-3", "m-1") == 0   # newest is vs another head


# ---- runs ----------------------------------------------------------------------

def test_auto_promotes_only_after_consecutive_qualifying_runs(lines):
    store = lines(alpha=_line("m-1"))
    router = FakeRouter(["m-1", "m-2"], {})

    first = router.run("auto")
    assert first["moves"] == [] and first["proposals"] == []
    assert first["lines"]["alpha"]["candidates"][0]["streak"] == 1
    assert model_lines.head_of("alpha") == "m-1"

    second = router.run("auto")
    assert second["moves"] == [{"line": "alpha", "from": "m-1", "to": "m-2"}]
    assert model_lines.head_of("alpha") == "m-2"
    promoted = store.model_line_events("alpha", kinds=("promoted",))[0]
    assert (promoted["actor"], promoted["family"], promoted["previous_family"]) == (
        "qualifier", "m-2", "m-1")
    assert promoted["detail"]["gates"]["cost"]["ok"] is True
    assert promoted["detail"]["streak"] == 2

    third = router.run("auto")                # nothing newer than the new head
    assert third["moves"] == [] and third["lines"]["alpha"]["candidates"] == []


def test_a_failed_run_resets_the_streak(lines):
    lines(alpha=_line("m-1"))
    router = FakeRouter(["m-1", "m-2"], {})
    router.run("auto")
    router.behaviour["m-2"] = {"text": "I cannot help with that."}
    assert router.run("auto")["moves"] == []
    router.behaviour["m-2"] = {}
    assert router.run("auto")["moves"] == []          # streak restarted at 1
    assert router.run("auto")["moves"] == [{"line": "alpha", "from": "m-1", "to": "m-2"}]


def test_propose_records_one_standing_proposal_and_moves_nothing(lines):
    store = lines(alpha=_line("m-1"))
    router = FakeRouter(["m-1", "m-2"], {})
    router.run("propose")
    for _ in range(3):
        report = router.run("propose")
        assert report["proposals"] == [{"line": "alpha", "from": "m-1", "to": "m-2"}]
        assert report["moves"] == []
    assert model_lines.head_of("alpha") == "m-1"
    assert len(store.model_line_events("alpha", kinds=("proposed",))) == 1

    # an operator approves it; the next run has nothing left to propose
    assert model_lines.promote("alpha", "m-2", actor="op", expected_current="m-1")["ok"]
    assert router.run("propose")["proposals"] == []


def test_report_mode_writes_nothing(lines):
    store = lines(alpha=_line("m-1", consecutive_passes=1))
    router = FakeRouter(["m-1", "m-2"], {})
    report = router.run("report")
    assert report["proposals"] == [{"line": "alpha", "from": "m-1", "to": "m-2"}]
    assert store.model_line_events() == []
    assert model_lines.head_of("alpha") == "m-1"


@pytest.mark.parametrize("behaviour, gate", [
    ({"cost": 0.005}, "cost"),                       # 5x the head's measured cost
    ({"status": 504, "code": "timeout"}, "answers"),
    ({"text": '{"decision": "authentic"}'}, "answers"),
])
def test_candidate_failing_a_gate_is_recorded_and_never_promoted(lines, behaviour, gate):
    store = lines(alpha=_line("m-1", consecutive_passes=1))
    router = FakeRouter(["m-1", "m-2"], {"m-2": behaviour})
    report = router.run("auto")
    assert report["moves"] == []
    candidate = report["lines"]["alpha"]["candidates"][0]
    assert candidate["qualified"] is False and candidate["gates"][gate]["ok"] is False
    assert store.model_line_events("alpha")[0]["kind"] == "candidate_failed"
    assert model_lines.head_of("alpha") == "m-1"


def test_newest_qualifying_candidate_wins_and_older_ones_are_capped(lines):
    lines(alpha=_line("m-1", consecutive_passes=1, max_candidates=2))
    router = FakeRouter(["m-1", "m-2", "m-3", "m-4"], {"m-4": {"cost": 0.01}})
    report = router.run("auto")
    probed = [c["family"] for c in report["lines"]["alpha"]["candidates"]]
    assert probed == ["m-4", "m-3"]                   # newest two only
    assert report["moves"] == [{"line": "alpha", "from": "m-1", "to": "m-3"}]


def test_unhealthy_head_blocks_promotion_and_is_flagged(lines):
    store = lines(alpha=_line("m-1", consecutive_passes=1))
    router = FakeRouter(["m-1", "m-2"], {"m-1": {"status": 504}})
    report = router.run("auto")
    assert report["unhealthy_heads"] == ["alpha"]
    assert report["moves"] == []                      # no baseline, no promotion
    assert report["lines"]["alpha"]["candidates"][0]["gates"]["cost"]["reason"] == "no_head_baseline"
    assert store.model_line_events("alpha", kinds=("head_unhealthy",))[0]["family"] == "m-1"


def test_degraded_head_is_flagged_and_can_still_be_replaced(lines):
    store = lines(alpha=_line("m-1", consecutive_passes=1))
    router = FakeRouter(["m-1", "m-2"], {"m-1": {"fail_on": ["p3"]}})         # 2/3 pass
    report = router.run("auto")
    assert report["unhealthy_heads"] == ["alpha"]
    assert report["lines"]["alpha"]["head_summary"]["pass_rate"] < 0.9
    assert report["moves"] == [{"line": "alpha", "from": "m-1", "to": "m-2"}]
    # replaced by a qualified successor: no separate unhealthy alarm for it
    assert store.model_line_events("alpha", kinds=("head_unhealthy",)) == []


def test_auto_rolls_back_its_own_promotion_when_the_predecessor_still_passes(lines):
    store = lines(alpha=_line("m-1", consecutive_passes=1))
    router = FakeRouter(["m-1", "m-2"], {})
    assert router.run("auto")["moves"] == [{"line": "alpha", "from": "m-1", "to": "m-2"}]

    router.behaviour["m-2"] = {"text": "refused"}
    report = router.run("auto")
    assert report["unhealthy_heads"] == ["alpha"]
    assert report["moves"] == [{"line": "alpha", "from": "m-2", "to": "m-1"}]
    assert model_lines.head_of("alpha") == "m-1"
    assert store.model_line_events("alpha")[0]["kind"] == "rolled_back"


def test_auto_never_rolls_back_a_head_an_operator_chose(lines):
    store = lines(alpha=_line("m-1"))
    assert model_lines.promote("alpha", "m-2", actor="dashboard:alice")["ok"]
    router = FakeRouter(["m-1", "m-2"], {"m-2": {"status": 504}})
    report = router.run("auto")
    assert report["moves"] == [] and model_lines.head_of("alpha") == "m-2"
    assert store.model_line_events("alpha")[0]["kind"] == "head_unhealthy"


def test_propose_mode_never_rolls_back(lines):
    lines(alpha=_line("m-1", consecutive_passes=1))
    router = FakeRouter(["m-1", "m-2"], {})
    router.run("auto")
    router.behaviour["m-2"] = {"status": 504}
    assert router.run("propose")["moves"] == []
    assert model_lines.head_of("alpha") == "m-2"


def test_streak_restarts_when_the_head_changes_under_the_candidate(lines):
    lines(alpha=_line("m-1"))
    router = FakeRouter(["m-1", "m-2", "m-3"], {"m-2": {"cost": 0.01}})   # m-2 never qualifies
    assert router.run("auto")["moves"] == []          # m-3: one pass, against m-1
    # an operator moves the line to m-2: m-3's pass was measured against m-1
    assert model_lines.promote("alpha", "m-2", actor="op")["ok"]
    router.behaviour["m-2"] = {}
    report = router.run("auto")
    assert report["moves"] == []
    assert report["lines"]["alpha"]["candidates"][0]["streak"] == 1
    assert router.run("auto")["moves"] == [{"line": "alpha", "from": "m-2", "to": "m-3"}]


def test_inconclusive_head_freezes_the_line(lines):
    store = lines(alpha=_line("m-1", consecutive_passes=1))
    router = FakeRouter(["m-1", "m-2"], {"m-1": {"status": 429, "code": "caller_rate_limit"}})
    report = router.run("auto")
    assert report["inconclusive_lines"] == ["alpha"]
    assert report["unhealthy_heads"] == [] and report["moves"] == []
    assert report["lines"]["alpha"]["head_healthy"] is None
    assert store.model_line_events("alpha") == []     # nothing judged, nothing recorded


def test_inconclusive_head_is_never_rolled_back(lines):
    lines(alpha=_line("m-1", consecutive_passes=1))
    router = FakeRouter(["m-1", "m-2"], {})
    assert router.run("auto")["moves"] == [{"line": "alpha", "from": "m-1", "to": "m-2"}]
    # our own key is rate limited while probing the new head; the old one answers
    router.behaviour["m-2"] = {"status": 429, "code": "key_rate_limit"}
    assert router.run("auto")["moves"] == []
    assert model_lines.head_of("alpha") == "m-2"


def test_inconclusive_candidate_neither_extends_nor_breaks_its_streak(lines):
    store = lines(alpha=_line("m-1"))
    router = FakeRouter(["m-1", "m-2"], {})
    router.run("auto")                                               # streak 1
    router.behaviour["m-2"] = {"status": 503, "code": "router_overloaded"}
    report = router.run("auto")
    assert report["moves"] == []
    assert report["lines"]["alpha"]["candidates"][0]["inconclusive"] is True
    assert len(store.model_line_events("alpha")) == 1                # no event for it
    router.behaviour["m-2"] = {}
    assert router.run("auto")["moves"] == [{"line": "alpha", "from": "m-1", "to": "m-2"}]


def test_run_stops_when_a_result_cannot_be_recorded(lines, monkeypatch):
    store = lines(alpha=_line("m-1"))
    router = FakeRouter(["m-1", "m-2"], {})
    router.run("auto")                                               # streak 1
    router.behaviour["m-2"] = {"text": "refused"}                    # this run fails...
    monkeypatch.setattr(store, "add_model_line_event", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="audit write failed"):    # ...and cannot be recorded
        router.run("auto")
    assert model_lines.head_of("alpha") == "m-1"


def test_auto_does_not_undo_its_own_rollback(lines):
    store = lines(alpha=_line("m-1", consecutive_passes=1))
    router = FakeRouter(["m-1", "m-2"], {})
    router.run("auto")                                               # m-1 -> m-2
    router.behaviour["m-2"] = {"text": "refused"}
    assert router.run("auto")["moves"] == [{"line": "alpha", "from": "m-2", "to": "m-1"}]
    # now m-1 fails too, and m-2 answers again but at 5x the cost: it must not
    # come back on answer rate alone just because it is the "previous" head
    router.behaviour = {"m-1": {"status": 504}, "m-2": {"cost": 0.005}}
    report = router.run("auto")
    assert report["moves"] == [] and model_lines.head_of("alpha") == "m-1"
    assert store.model_line_events("alpha")[0]["kind"] == "head_unhealthy"


def test_rollback_is_skipped_when_the_line_moved_during_the_probes(lines, monkeypatch):
    store = lines(alpha=_line("m-1", consecutive_passes=1))
    router = FakeRouter(["m-1", "m-2", "m-3"], {"m-3": {"cost": 0.01}})
    router.run("auto")                                               # m-1 -> m-2
    router.behaviour["m-2"] = {"text": "refused"}
    real_post = router.post
    moved = []

    def post(body):
        # an operator promotes m-3 while the qualifier is still probing
        if not moved:
            moved.append(model_lines.promote("alpha", "m-3", actor="op")["ok"])
        return real_post(body)

    report = q.run(post=post, get=router.get, suites=[SUITE], mode="auto", workers=1)
    assert moved == [True]
    # m-2 did fail its probes, but it is no longer the move to undo: the latest
    # move is the operator's, so the qualifier only records what it saw
    assert report["moves"] == []
    assert "rolled_back" not in report["lines"]["alpha"]["head_action"]
    assert store.get_model_line_heads()["alpha"]["family"] == "m-3"   # the operator's move stands
    assert store.model_line_events("alpha", kinds=("rolled_back",)) == []


def test_run_refuses_to_decide_without_the_store(lines, monkeypatch):
    store = lines(alpha=_line("m-1"))
    monkeypatch.setattr(store, "get_model_line_heads", lambda: None)
    router = FakeRouter(["m-1", "m-2"], {})
    with pytest.raises(RuntimeError, match="host store unavailable"):
        router.run("auto")
    assert router.run("report")["lines"]["alpha"]["head"] == "m-1"   # read-only is fine


def test_run_fails_loudly_when_the_router_lists_no_families(lines):
    lines(alpha=_line("m-1"))
    router = FakeRouter([], {})
    with pytest.raises(RuntimeError, match="no families"):
        router.run("report")
    with pytest.raises(ValueError, match="unknown line"):
        FakeRouter(["m-1"], {}).run("report", only_lines=["ghost"])


def test_only_lines_restricts_probing(lines):
    lines(alpha=_line("m-1"), beta={"head": "n-1", "match": r"^n-(\d+)$"})
    router = FakeRouter(["m-1", "m-2", "n-1", "n-2"], {})
    report = router.run("report", only_lines=["beta"])
    assert list(report["lines"]) == ["beta"]
    probed = {b["policy_ir"][1][-1][1] for b in router.bodies}
    assert probed == {"n-1", "n-2"}
