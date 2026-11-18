"""Qualify successors for model lines by probing them with real workloads.

"Newer" is not "better for this workload": a successor can refuse a prompt the
head answers, blow the caller's deadline, or cost several times more through the
routes that actually serve it. So a line never moves on a version number alone.
Each run:

1. Probes every line's current head with the prompt suites — the baseline, and
   a health check of what is live.
2. Finds successors by the line's name pattern (model_lines.successors) and
   probes the newest few exactly the same way: one family, no fallback, the
   caller's own deadlines.
3. A candidate QUALIFIES when it answers validly on enough probes, is no slower
   than an absolute bound or than the head by more than a ratio, and its MEASURED
   cost per probe is within the allowed ratio of the head's measured cost on the
   same prompts. List prices are not
   consulted: marketplace routes make them a poor predictor of what a call costs.
4. A candidate that qualified on enough consecutive runs is promoted (`auto`) or
   recorded as a proposal for an operator (`propose`). `report` writes nothing.
5. A head that stops passing is flagged; in `auto`, a head this qualifier itself
   promoted is rolled back when its predecessor still passes.
6. A probe the ingress or the network stopped (our own rate limit or budget, an
   overloaded router) is inconclusive, not failed: a family with any such probe
   is not judged on that run.

Probes go through the public ingress with a consumer key, so their spend lands
in the call ledger like any other caller's.

The decision logic is pure and takes its transport (`post`, `get`) and store as
arguments; `scripts/qualify_model_lines.py` is the thin CLI around it.
"""
from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import model_lines

ACTOR = "qualifier"
MODES = ("report", "propose", "auto")

# Body keys a suite may set on its probe requests. Everything else in the body
# (model, messages, the single-family policy) is owned by the qualifier.
_REQUEST_KEYS = {"temperature", "max_tokens", "response_format",
                 "first_token_timeout_ms", "timeout_ms", "top_p", "seed"}


class SuiteError(ValueError):
    """A prompt suite file is not usable."""


# ---- suites ---------------------------------------------------------------------

def parse_suite(data: Any, source: str = "suite") -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SuiteError(f"{source}: suite must be an object")
    name = data.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.\-]{0,63}", name):
        raise SuiteError(f"{source}: `name` must be a short slug")
    prompts = data.get("prompts")
    if (not isinstance(prompts, list) or not prompts
            or not all(isinstance(p, str) and p.strip() for p in prompts)):
        raise SuiteError(f"{source}: `prompts` must be a non-empty list of strings")
    request = data.get("request") or {}
    if not isinstance(request, dict) or set(request) - _REQUEST_KEYS:
        raise SuiteError(
            f"{source}: `request` may only set {sorted(_REQUEST_KEYS)}")
    expect = data.get("expect") or {"type": "text"}
    if not isinstance(expect, dict) or expect.get("type") not in {"text", "json", "regex"}:
        raise SuiteError(f"{source}: `expect.type` must be text, json or regex")
    if expect["type"] == "regex":
        pattern = expect.get("pattern")
        # An absent pattern would compile to the empty expression, which matches
        # every answer — a refusal included.
        if not isinstance(pattern, str) or not pattern:
            raise SuiteError(f"{source}: expect.pattern must be a non-empty string")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise SuiteError(f"{source}: bad expect.pattern: {exc}") from exc
    repeats = data.get("repeats", 1)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 5:
        raise SuiteError(f"{source}: `repeats` must be an integer in [1, 5]")
    return {"name": name, "prompts": prompts, "request": request,
            "expect": expect, "repeats": repeats}


def load_suites(paths: "list[str | Path]") -> list[dict[str, Any]]:
    """Load suites from files and/or directories (every *.json inside)."""
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    suites, seen = [], set()
    for f in files:
        suite = parse_suite(json.loads(f.read_text(encoding="utf-8")), str(f))
        if suite["name"] in seen:
            raise SuiteError(f"duplicate suite name {suite['name']!r}")
        seen.add(suite["name"])
        suites.append(suite)
    if not suites:
        raise SuiteError("no prompt suites found")
    return suites


def _json_object(text: str) -> Any:
    """The JSON object in a model answer, tolerating a code fence or prose
    around it the way lenient callers do: first `{` to last `}`."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    return json.loads(text[start:end + 1])


def _check_value(value: Any, spec: dict[str, Any]) -> bool:
    kind = spec.get("type")
    if "enum" in spec:
        return value in spec["enum"]
    if kind == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif kind == "string":
        ok = isinstance(value, str) and (bool(value.strip()) or spec.get("allow_empty", False))
    elif kind == "boolean":
        ok = isinstance(value, bool)
    else:
        ok = value is not None
    if ok and kind in {"integer", "number"}:
        if "min" in spec and value < spec["min"]:
            return False
        if "max" in spec and value > spec["max"]:
            return False
    return ok


def answer_is_valid(text: str, expect: dict[str, Any]) -> bool:
    """Whether a model answer satisfies a suite's `expect` block."""
    if not isinstance(text, str) or not text.strip():
        return False
    kind = expect.get("type", "text")
    if kind == "text":
        return True
    if kind == "regex":
        return re.search(expect.get("pattern") or "", text) is not None
    try:
        obj = _json_object(text)
    except (ValueError, TypeError):
        return False
    if not isinstance(obj, dict):
        return False
    for key, spec in (expect.get("required") or {}).items():
        if key not in obj or not _check_value(obj[key], spec if isinstance(spec, dict) else {}):
            return False
    return True


# ---- probing --------------------------------------------------------------------

# Errors the INGRESS returns before any provider sees the request, or that mean
# the probe never reached the router: our own key's rate limit or budget, the
# router being overloaded or unreachable. They say nothing about the model, so a
# probe that ends this way is inconclusive, not failed.
_INGRESS_CODES = frozenset({
    "consumer_auth", "caller_rate_limit", "caller_rate_limit_unavailable",
    "key_rate_limit", "key_rate_limit_unavailable", "key_budget_exhausted",
    "key_budget_unavailable", "consumer_budget_exhausted",
    "consumer_budget_unavailable", "consumer_store_unavailable",
    "router_overloaded", "route_unavailable", "upstream", "upstream_error",
})


def probe_policy(family: str) -> list:
    """One family, no fallback: a failure must surface as a failure, not as
    another model's answer. Ranks that family's routes the way the validator
    policies do (price, reliability, latency)."""
    return ["policy",
            ["and", ["meets_req"], ["not", ["is", "disabled"]], ["family_eq", family]],
            ["add",
             ["scale", 0.25, ["neg", ["normalize", ["field", "price_in"]]]],
             ["scale", 0.45, ["neg", ["normalize", ["field", "price_out"]]]],
             ["scale", 0.2, ["field", "success_rate"]],
             ["scale", 0.1, ["neg", ["normalize", ["field", "latency_ms"]]]]],
            ["argmax"], ["id"], ["always", {"action": "next_candidate"}]]


def probe_once(post: Callable[[dict], "tuple[int, Any]"], family: str,
               suite: dict[str, Any], prompt: str, label: str) -> dict[str, Any]:
    """One probe. `post(body) -> (http_status, parsed_json_or_None)`. Never
    raises: a transport error is an inconclusive probe."""
    body = {**suite["request"],
            "model": f"policy:{label}",
            "messages": [{"role": "user", "content": prompt}],
            "policy_ir": probe_policy(family)}
    started = time.monotonic()
    try:
        status, data = post(body)
    except Exception as exc:  # noqa: BLE001 — a probe never takes the run down
        status, data = 0, {"error": {"code": type(exc).__name__, "message": str(exc)}}
    out: dict[str, Any] = {"suite": suite["name"], "status": status,
                           "latency_ms": int((time.monotonic() - started) * 1000),
                           "valid": False, "inconclusive": False, "cost_usd": None}
    data = data if isinstance(data, dict) else {}
    if status != 200:
        err = data.get("error") if isinstance(data.get("error"), dict) else {}
        code = str(err.get("code") or err.get("type") or "error")[:60]
        out["error"] = code
        out["inconclusive"] = status in (0, 401, 403) or code in _INGRESS_CODES
        return out
    x_router = data.get("x_router") if isinstance(data.get("x_router"), dict) else {}
    served = x_router.get("model_family")
    out["provider"] = x_router.get("provider")
    cost = x_router.get("cost_usd")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
        out["cost_usd"] = float(cost)
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    if served is not None and served != family:
        # The single-family policy makes this impossible; treat a mismatch as a
        # failed probe rather than crediting the candidate with another model.
        out["error"] = "served_other_family"
        return out
    out["valid"] = answer_is_valid(text, suite["expect"])
    if not out["valid"]:
        # A reasoning model that spends the whole output budget thinking returns
        # an empty or cut-off answer. Name it: the fix is a budget, not a model.
        out["error"] = ("truncated" if choice.get("finish_reason") == "length"
                        else "invalid_answer")
    return out


def probe_family(post: Callable[[dict], "tuple[int, Any]"], family: str,
                 suites: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Run every suite prompt against one family, sequentially — a burst of our
    own probes must not be what trips the provider's rate limit. The order is
    fixed, so probe i of one family is the same prompt as probe i of another."""
    results = []
    for suite in suites:
        for _ in range(suite["repeats"]):
            for prompt in suite["prompts"]:
                results.append(probe_once(post, family, suite, prompt, label))
    return summarize(family, results)


def summarize(family: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = [r for r in results if r["valid"]]
    latencies = sorted(r["latency_ms"] for r in passed)
    errors: dict[str, int] = {}
    by_suite: dict[str, dict[str, int]] = {}
    for r in results:
        stats = by_suite.setdefault(r["suite"], {"probes": 0, "passed": 0})
        stats["probes"] += 1
        stats["passed"] += 1 if r["valid"] else 0
        if not r["valid"]:
            key = f"{r['suite']}:{r.get('error') or r['status']}"
            errors[key] = errors.get(key, 0) + 1
    providers: dict[str, int] = {}
    for r in passed:
        providers[str(r.get("provider"))] = providers.get(str(r.get("provider")), 0) + 1
    # What each probe cost, valid answer or not: a model that charges for an
    # unusable answer has still charged. None where the router reported no cost
    # (a probe that never produced a response has none to report).
    costs = [r["cost_usd"] for r in results]
    answered = [r for r in results if r["status"] == 200]
    known = [c for c in costs if c is not None]

    def pct(q: float) -> "int | None":
        if not latencies:
            return None
        return latencies[min(len(latencies) - 1, math.ceil(q * len(latencies)) - 1)]

    return {
        "family": family, "probes": total, "passed": len(passed),
        "inconclusive": sum(1 for r in results if r.get("inconclusive")),
        "pass_rate": round(len(passed) / total, 4) if total else 0.0,
        "suites": by_suite,
        "latency_p50_ms": pct(0.5), "latency_p90_ms": pct(0.9),
        "latency_p95_ms": pct(0.95),
        # Known only when every response that came back carried a cost.
        "mean_cost_usd": (round(sum(known) / len(known), 8)
                          if known and len(known) == len(answered) else None),
        "costs": costs,
        "providers": providers, "errors": errors,
    }


# ---- decisions (pure) -----------------------------------------------------------

def conclusive(summary: dict[str, Any]) -> bool:
    """False when any probe was stopped by the ingress or the transport rather
    than answered or failed by the model: such a run decides nothing about it."""
    return summary["probes"] > 0 and not summary.get("inconclusive")


def healthy(summary: dict[str, Any], definition: dict[str, Any]) -> bool:
    """Enough valid answers overall AND in every suite. The overall bar leaves
    room for one transient error; the per-suite bar stops a model that fails a
    whole workload from hiding behind the others."""
    if summary["probes"] <= 0 or summary["pass_rate"] < definition["min_pass_rate"]:
        return False
    for stats in (summary.get("suites") or {}).values():
        if stats["probes"] and stats["passed"] / stats["probes"] < definition["min_suite_pass_rate"]:
            return False
    return True


def _paired_cost_ratio(candidate: dict[str, Any], head: dict[str, Any]) -> dict[str, Any]:
    """Candidate vs head cost over the probes where BOTH reported one, so the
    two sides are summed over the same prompts. Fails closed when the candidate
    has an answered probe without a cost, or when too few probes pair up to say
    anything."""
    c_costs, h_costs = candidate.get("costs") or [], head.get("costs") or []
    if candidate["mean_cost_usd"] is None:
        return {"ok": False, "reason": "candidate_cost_unknown"}
    pairs = [(c, h) for c, h in zip(c_costs, h_costs) if c is not None and h is not None]
    if len(pairs) * 2 < max(len(c_costs), 1):
        return {"ok": False, "reason": "no_head_baseline", "paired": len(pairs)}
    c_sum, h_sum = sum(c for c, _ in pairs), sum(h for _, h in pairs)
    out: dict[str, Any] = {"paired": len(pairs),
                           "candidate_usd": round(c_sum / len(pairs), 8),
                           "head_usd": round(h_sum / len(pairs), 8)}
    if h_sum == 0:
        return {**out, "ok": c_sum == 0, "reason": "head_is_free"}
    return {**out, "ok": None, "ratio": round(c_sum / h_sum, 4)}


def judge(candidate: dict[str, Any], head: dict[str, Any],
          definition: dict[str, Any]) -> dict[str, Any]:
    """Gate a candidate's probe summary against the head's. Every gate fails
    closed: an unknown cost, or a head with no successful probe to compare
    against, does not qualify anyone.

    The head does NOT have to be healthy to serve as the baseline. A head that
    still answers some probes is exactly the one worth replacing, and its
    successful probes still say what it costs and how fast it is."""
    gates: dict[str, dict[str, Any]] = {}
    gates["answers"] = {
        "ok": healthy(candidate, definition),
        "pass_rate": candidate["pass_rate"], "min": definition["min_pass_rate"],
        "suites": {name: round(s["passed"] / s["probes"], 4) if s["probes"] else 0.0
                   for name, s in (candidate.get("suites") or {}).items()},
        "min_suite": definition["min_suite_pass_rate"]}
    # p90, not the maximum: with a few dozen probes p95 IS the slowest call, and
    # one slow provider hop would decide the gate.
    p90, head_p90 = candidate["latency_p90_ms"], head["latency_p90_ms"]
    relative = (round(head_p90 * definition["max_latency_ratio"])
                if head_p90 is not None else None)
    gates["latency"] = {
        "ok": p90 is not None and (p90 <= definition["max_latency_ms"]
                                   or (relative is not None and p90 <= relative)),
        "p90_ms": p90, "head_p90_ms": head_p90,
        "max_ms": definition["max_latency_ms"], "max_vs_head_ms": relative}
    cost = _paired_cost_ratio(candidate, head)
    if cost["ok"] is None:
        cost["ok"] = cost["ratio"] <= definition["max_cost_ratio"]
        cost["max_ratio"] = definition["max_cost_ratio"]
    gates["cost"] = cost
    return {"qualified": all(g["ok"] for g in gates.values()), "gates": gates}


def consecutive_passes(events: list[dict[str, Any]], family: str, head: str) -> int:
    """How many of this family's most recent qualification results (newest
    first) were passes AGAINST THE CURRENT HEAD, stopping at the first failure
    or at the first result measured against a different head: a pass against a
    baseline that is no longer live says nothing about replacing this one."""
    n = 0
    for event in events:
        if event.get("family") != family:
            continue
        if event.get("previous_family") != head:
            break
        if event.get("kind") == "candidate_passed":
            n += 1
        elif event.get("kind") == "candidate_failed":
            break
    return n


# ---- the run --------------------------------------------------------------------

def list_families(get: Callable[[str], "tuple[int, Any]"]) -> list[str]:
    """Family names the router can currently route, from GET /v1/models."""
    status, data = get("/v1/models")
    if status != 200 or not isinstance(data, dict):
        raise RuntimeError(f"/v1/models returned {status}")
    out = []
    for row in data.get("data") or []:
        ident = row.get("id") if isinstance(row, dict) else None
        if isinstance(ident, str) and ident.startswith("family:"):
            out.append(ident[len("family:"):])
    if not out:
        raise RuntimeError("/v1/models listed no families")
    return out


def _stored(summary: dict[str, Any]) -> dict[str, Any]:
    """A summary as kept in the audit trail: without the per-probe cost list."""
    return {k: v for k, v in summary.items() if k != "costs"}


def run(*, post: Callable[[dict], "tuple[int, Any]"],
        get: Callable[[str], "tuple[int, Any]"],
        suites: list[dict[str, Any]], mode: str = "propose",
        only_lines: "list[str] | None" = None, workers: int = 4,
        store: Any = None) -> dict[str, Any]:
    """One qualification run over every (or the selected) line. Returns the
    report; in `propose`/`auto` it also writes audit events and, in `auto`,
    moves heads. `store` defaults to host_store (injected in tests).

    Raises when the run cannot be trusted to continue: the store is unreachable,
    or an audit event could not be written. The streak a promotion rests on is
    read from those events, so a run that cannot record a result must stop
    before it decides anything on top of it."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if store is None:
        import host_store as store  # noqa: PLC0415
    writes = mode != "report"
    definitions = model_lines.definitions()
    store_ok = model_lines.reload()
    if writes and not store_ok:
        raise RuntimeError("host store unavailable: refusing to decide on stale heads")
    names = [n for n in definitions if not only_lines or n in only_lines]
    unknown = sorted(set(only_lines or []) - set(definitions))
    if unknown:
        raise ValueError(f"unknown line(s): {unknown}")
    families = list_families(get)
    available = set(families)
    snapshot = {row["line"]: row for row in model_lines.snapshot(events=0)["lines"]}

    def record(line: str, kind: str, **kw: Any) -> None:
        if writes and not store.add_model_line_event(line, kind, actor=ACTOR, **kw):
            raise RuntimeError(f"audit write failed ({line}: {kind}): stopping "
                               "before any decision rests on a missing result")

    plan: dict[str, dict[str, Any]] = {}
    to_probe: dict[str, None] = {}
    for name in names:
        d = definitions[name]
        head = snapshot[name]["head"]
        cands = model_lines.successors(d, head, families, int(d["max_candidates"]))
        plan[name] = {"head": head, "candidates": cands,
                      "previous": snapshot[name].get("previous")}
        for family in [head, *cands]:
            to_probe[family] = None

    def probe(family: str) -> dict[str, Any]:
        return probe_family(post, family, suites, f"qualify-{family}"[:60])

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        summaries = dict(zip(to_probe, pool.map(probe, list(to_probe))))

    report: dict[str, Any] = {"mode": mode, "ts": int(time.time()),
                              "suites": [s["name"] for s in suites],
                              "lines": {}, "moves": [], "proposals": [],
                              "unhealthy_heads": [], "inconclusive_lines": []}
    for name in names:
        d = definitions[name]
        head = plan[name]["head"]
        head_summary = summaries[head]
        line_report: dict[str, Any] = {
            "head": head, "head_available": head in available,
            "head_summary": _stored(head_summary), "candidates": [], "action": None}
        report["lines"][name] = line_report
        if not conclusive(head_summary):
            # The ingress or the network stopped some of the head's probes. With
            # no trustworthy baseline nothing on this line is judged this run:
            # not its candidates, and not the head's own health.
            line_report["head_healthy"] = None
            report["inconclusive_lines"].append(name)
            continue
        head_ok = healthy(head_summary, d)
        line_report["head_healthy"] = head_ok
        events = (store.model_line_events(
            name, limit=200, kinds=("candidate_passed", "candidate_failed"))
            if writes else [])

        eligible: "tuple[str, dict[str, Any]] | None" = None
        needed = int(d["consecutive_passes"])
        for family in plan[name]["candidates"]:
            summary = summaries[family]
            if not conclusive(summary):
                line_report["candidates"].append({
                    "family": family, "summary": _stored(summary),
                    "qualified": None, "inconclusive": True})
                continue      # neither a pass nor a failure: the streak stands
            verdict = judge(summary, head_summary, d)
            detail = {"summary": _stored(summary), "head": head,
                      "head_summary": _stored(head_summary), **verdict,
                      "suites": report["suites"]}
            streak = (consecutive_passes(events, family, head) + 1
                      if verdict["qualified"] else 0)
            line_report["candidates"].append({
                "family": family, "summary": _stored(summary), **verdict,
                "streak": streak, "needed": needed})
            record(name, "candidate_passed" if verdict["qualified"] else "candidate_failed",
                   family=family, previous_family=head, detail=detail)
            # Candidates are newest-first: the first eligible one is the pick.
            if eligible is None and verdict["qualified"] and streak >= needed:
                eligible = (family, {**detail, "streak": streak})

        promoted = False
        if eligible is not None:
            family, detail = eligible
            if mode == "auto":
                result = model_lines.promote(
                    name, family, actor=ACTOR, detail=detail, expected_current=head)
                line_report["action"] = {"promote": family, **result}
                if result.get("ok"):
                    promoted = True
                    report["moves"].append({"line": name, "from": head, "to": family})
            else:
                line_report["action"] = {"propose": family}
                report["proposals"].append({"line": name, "from": head, "to": family})
                if writes and not _already_proposed(store, name, family, head):
                    record(name, "proposed", family=family, previous_family=head,
                           detail=detail)

        if not head_ok:
            report["unhealthy_heads"].append(name)
            if promoted:
                continue      # the line just moved to a qualified successor
            action = _handle_unhealthy_head(
                name, d, plan[name], head_summary, summaries, post, suites,
                mode, store, record)
            line_report["head_action"] = action
            if action.get("rolled_back"):
                report["moves"].append(
                    {"line": name, "from": head, "to": action["rolled_back"]})
    return report


def _already_proposed(store: Any, line: str, family: str, head: str) -> bool:
    """A standing proposal is recorded once, not once per run."""
    latest = store.model_line_events(
        line, limit=1, kinds=("proposed", "promoted", "rolled_back"))
    return bool(latest and latest[0].get("kind") == "proposed"
                and latest[0].get("family") == family
                and latest[0].get("previous_family") == head)


def _qualifier_promoted(store: Any, line: str, head: str, previous: str) -> bool:
    """True when the move that made `head` current is this qualifier's own
    PROMOTION from `previous`. Not an operator's choice, and not one of the
    qualifier's rollbacks: undoing a rollback would restore a head that was
    removed for failing, on nothing more than the old head failing too."""
    latest = store.model_line_events(line, limit=1, kinds=("promoted", "rolled_back"))
    return bool(latest and latest[0].get("kind") == "promoted"
                and latest[0].get("actor") == ACTOR
                and latest[0].get("family") == head
                and latest[0].get("previous_family") == previous)


def _handle_unhealthy_head(name: str, d: dict[str, Any], plan: dict[str, Any],
                           head_summary: dict[str, Any],
                           summaries: dict[str, dict[str, Any]],
                           post: Callable, suites: list[dict[str, Any]],
                           mode: str, store: Any,
                           record: Callable[..., None]) -> dict[str, Any]:
    """A head that no longer passes its probes. Always recorded. Rolled back only
    in `auto`, only when the current move is this qualifier's own promotion, and
    only when the head it replaced still passes — otherwise there is nothing
    better to fall back to and an operator has to look."""
    head, previous = plan["head"], plan["previous"]
    detail: dict[str, Any] = {"head": head, "head_summary": _stored(head_summary)}
    action: dict[str, Any] = {"unhealthy": head}
    if (mode == "auto" and previous and previous != head
            and _qualifier_promoted(store, name, head, previous)):
        prev_summary = summaries.get(previous) or probe_family(
            post, previous, suites, f"qualify-{previous}"[:60])
        detail["previous"] = previous
        detail["previous_summary"] = _stored(prev_summary)
        if conclusive(prev_summary) and healthy(prev_summary, d):
            # Compare-and-set on the exact move that was probed: if the line
            # moved while the probes ran, this is no longer the move to undo.
            result = model_lines.rollback(
                name, actor=ACTOR, detail=detail,
                expected_current=head, expected_previous=previous)
            if result.get("ok"):
                action["rolled_back"] = previous
                return action
            action["rollback_failed"] = result.get("reason")
    record(name, "head_unhealthy", family=head, detail=detail)
    return action
