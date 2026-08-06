"""The AntSeed funding autonomy loop and every guardrail standing between it and
real USDC on Base mainnet.

Nothing here touches a wallet: the sidecar control plane is replaced by a
recording double (`_FakeControl`), so the only thing under test is which calls
the keeper *decides* to make. That is the interesting part — the failure mode
this code has to be trusted against is not a broken HTTP call, it is a correct
HTTP call that should never have been made.
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import host_store  # noqa: E402
import settings  # noqa: E402
import wallet_keeper as wk  # noqa: E402
from conftest import seed_buyer_status, seed_route_obs  # noqa: E402

PID = "antseed"


@pytest.fixture(autouse=True)
def _clean(host_store_clean, monkeypatch):
    # The keeper reads its state from the store and its knobs from settings; both
    # need per-test isolation. The control endpoint is faked, but set the env so
    # `cycle()` gets past its "configured?" check.
    settings.reload()
    monkeypatch.setenv("ANTSEED_CONTROL_URL", "http://127.0.0.1:8379")
    monkeypatch.setenv("ANTSEED_CONTROL_TOKEN", "t0ken")
    wk.KEEPER_STATE.update({"enabled": False, "last_cycle": None,
                            "providers": {}, "error": None})
    # A readable hot wallet is a PRECONDITION for depositing, not a bonus (the
    # floor cannot be checked without one), so it is part of the healthy baseline
    # here exactly as a readable buyer_status is. Tests that care about the
    # unreadable case clear it explicitly.
    _chain()
    yield
    import sources as _sources
    _sources.SOURCE_STATE.clear()


def _chain(usdc=500.0, eth=0.05, pid=PID, age_s=0.0):
    """Seed the untrusted public-RPC reading the source publishes. `age_s` ages
    the reading — the keeper bounds it, so a stale one is no reading at all."""
    import sources as _sources
    detail = {}
    if usdc is not None:
        detail["wallet_usdc"] = usdc
    if eth is not None:
        detail["wallet_eth"] = eth
    _sources.SOURCE_STATE.setdefault("antseed", {}).setdefault("balances", {})[pid] = {
        "kind": "deposits_usdc", "value": 0.5, "detail": detail,
        "fetched_at": int(time.time() - age_s)}


def _no_chain(pid=PID):
    """No usable hot-wallet reading at all (RPC down, cold start, or disabled)."""
    import sources as _sources
    _sources.SOURCE_STATE.clear()


def _age_status(pid, age_s):
    """Backdate the sidecar's write stamp on a buyer_status row."""
    with host_store._get_pool().connection() as conn:
        conn.execute("UPDATE buyer_status SET fetched_at=%s WHERE pid=%s",
                     (int((time.time() - age_s) * 1000), pid))


def _knobs(**over):
    """The default knob set with overrides — bypasses settings so a test can pin
    exactly one relation without persisting an override."""
    base = dict(enabled=True, min_available_usdc=1.1, topup_trigger_usdc=2.0,
                topup_amount_usdc=5.0, topup_wallet_floor_usdc=1.0,
                topup_daily_cap_usdc=10.0, topup_cooldown_s=900,
                reclaim_min_usdc=3.0)
    base.update(over)
    return wk.Knobs(**base)


class _FakeControl(wk.WalletKeeper):
    """A keeper whose control plane records instead of transacting. The allowlist
    lives in `control()`, ABOVE this seam, so the double cannot widen it."""

    def __init__(self, pids=(PID,), responses=None):
        super().__init__(list(pids))
        self.calls: list[tuple[str, dict | None]] = []
        self.responses = responses or {}

    async def _control_post(self, op, body, timeout):
        self.calls.append((op, body))
        return self.responses.get(op, {"ok": True})

    @property
    def ops(self):
        return [op for op, _ in self.calls]


def _run(coro):
    return asyncio.run(coro)


# ---- kill switch -------------------------------------------------------------

def test_keeper_ships_dark_and_the_kill_switch_is_off_by_default():
    # The whole loop is gated on one knob whose default is 0. A funding robot
    # that arms itself on deploy is not a funding robot anyone should merge.
    assert settings.SCHEMA["antseed.keeper_enabled"]["default"] == 0
    assert settings.get("antseed.keeper_enabled") == 0
    assert wk.load_knobs().enabled is False

    seed_buyer_status(PID, deposits_available="0.01", deposits_reserved="20.0")
    k = _FakeControl()
    _run(k.cycle())
    assert k.calls == [], "a disabled keeper must not touch the control plane"
    assert host_store.wallet_ops_recent(PID) == []


def test_kill_switch_is_re_read_every_cycle(monkeypatch):
    # Flipping the knob must take effect without a restart: the loop reads it
    # per cycle, not once at construction.
    seed_buyer_status(PID, deposits_available="50.0", deposits_reserved="0.0")
    settings.validate_and_write({"antseed.keeper_enabled": 1})
    k = _FakeControl()
    assert _run(k.cycle())["enabled"] is True
    settings.validate_and_write({"antseed.keeper_enabled": 0})
    assert _run(k.cycle())["enabled"] is False


# ---- knob cross-validation ---------------------------------------------------

def test_knobs_cross_validated_trigger_must_exceed_the_tourniquet():
    # A trigger at or below the offer tourniquet means funding never reacts
    # before routing goes dark — an automation that is a silent no-op.
    settings.validate_and_write({"antseed.keeper_enabled": 1,
                                 "antseed.topup_trigger_usdc": 1.0,
                                 "antseed.min_available_usdc": 1.1})
    with pytest.raises(wk.KnobError, match="must be ABOVE"):
        wk.load_knobs()
    # ...and the loop stands down instead of acting on it.
    seed_buyer_status(PID, deposits_available="0.1", deposits_reserved="20.0")
    k = _FakeControl()
    state = _run(k.cycle())
    assert k.calls == [] and "must be ABOVE" in state["error"]


def test_knobs_cross_validated_amount_must_fit_the_daily_cap():
    settings.validate_and_write({"antseed.topup_amount_usdc": 20,
                                 "antseed.topup_daily_cap_usdc": 10})
    with pytest.raises(wk.KnobError, match="daily_cap"):
        wk.load_knobs()


def test_a_non_knoberror_from_the_knobs_still_updates_the_reported_state(monkeypatch):
    """M-6: only `KnobError` was caught, so an unreadable settings store (or a
    knob that would not coerce) propagated straight out of `cycle()` — leaving
    KEEPER_STATE holding the LAST SUCCESSFUL cycle. /x/runtime then went on
    describing a healthy keeper that had not run since."""
    monkeypatch.setattr(wk, "load_knobs",
                        lambda: (_ for _ in ()).throw(RuntimeError("settings down")))
    wk.KEEPER_STATE.update({"enabled": True, "last_cycle": 1,
                            "providers": {PID: {"decision": "acted"}}, "error": None})
    state = _run(_FakeControl().cycle())
    assert state["enabled"] is False
    assert "settings down" in state["error"]
    assert state["providers"] == {}, "the stale per-provider view must not persist"


def test_topup_amount_knob_cannot_exceed_the_hard_per_deposit_limit():
    # settings.SCHEMA caps the knob at 50, and the keeper re-checks the same
    # ceiling — the schema is operator-facing, the constant is the invariant.
    assert settings.SCHEMA["antseed.topup_amount_usdc"]["max"] == wk.MAX_TOPUP_USDC
    assert settings._coerce("antseed.topup_amount_usdc", 500) is None


# ---- fail-closed on unreadable state -----------------------------------------

def test_keeper_does_nothing_when_buyer_status_is_missing():
    # No status row at all. The keeper FAILS CLOSED (the offer gate fails open):
    # money must never move on a state we cannot read.
    k = _FakeControl()
    out = _run(k.cycle_provider(PID, _knobs()))
    assert out == {"decision": "status_unreadable"}
    assert k.calls == []


def test_keeper_does_nothing_when_deposits_are_unparseable():
    seed_buyer_status(PID, deposits_available="n/a", deposits_reserved="0")
    k = _FakeControl()
    assert _run(k.cycle_provider(PID, _knobs()))["decision"] == "status_unreadable"
    assert k.calls == []


def test_a_STALE_buyer_status_stands_the_keeper_down(caplog):
    """H-3: `buyer_status` had no freshness bound anywhere. `fetched_at` was in
    the table but not in `_BUYER_STATUS_FIELDS`, so NO consumer could check it —
    the keeper, the offer tourniquet and the envelope's credits clause all keyed
    off one unbounded-age signal and failed open together.

    The realistic pairing is a dead sidecar plus an escrow drained by
    settlements: the row keeps reporting a healthy balance while traffic spends
    against an escrow nobody is measuring. Acting on it is worst precisely then.
    The pattern already existed one function up — `peer_offers(STALE_AFTER_S)`."""
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="20.0")
    _age_status(PID, wk.STATUS_MAX_AGE_S + 60)
    _wedge()
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    with caplog.at_level("WARNING"):
        out = _run(k.cycle_provider(PID, _knobs()))
    assert out["decision"] == "status_stale"
    assert k.calls == [], "no money moves on a row nobody is writing"
    assert "is the sidecar alive?" in caplog.text


def test_a_status_row_with_no_write_stamp_is_treated_as_stale():
    # Undated is not the same as fresh. Fail closed on both.
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="20.0")
    with host_store._get_pool().connection() as conn:
        conn.execute("UPDATE buyer_status SET fetched_at=NULL WHERE pid=%s", (PID,))
    k = _FakeControl()
    assert _run(k.cycle_provider(PID, _knobs()))["decision"] == "status_stale"
    assert k.calls == []


def test_a_fresh_status_still_acts():
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    k = _FakeControl()
    assert _run(k.cycle_provider(PID, _knobs()))["decision"] == "acted"


def test_buyer_status_exposes_its_write_stamp():
    # The bound above is unenforceable if the column never reaches the reader.
    seed_buyer_status(PID, deposits_available="1.0", deposits_reserved="0.0")
    assert "fetched_at" in host_store._BUYER_STATUS_FIELDS
    assert isinstance(host_store.buyer_status(PID)["fetched_at"], int)


# ---- the withdraw prohibition ------------------------------------------------

def test_escrow_withdraw_is_not_automatable():
    # `buyer withdraw` moves funds OUT of the system — the exfiltration path if
    # the control token leaks. It must be unreachable from the keeper by
    # construction, not merely unused.
    assert "withdraw" not in wk.ALLOWED_CONTROL_OPS
    assert "withdraw" in wk.FORBIDDEN_CONTROL_OPS
    k = _FakeControl()
    for verb in ("withdraw", "buyer/withdraw", "/withdraw"):
        with pytest.raises(ValueError, match="human-initiated"):
            _run(k.control(verb, {"amount": "5"}))
    with pytest.raises(ValueError, match="not an allowlisted verb"):
        _run(k.control("status"))
    assert k.calls == [], "no forbidden verb reached the control plane"


def test_reclaim_withdraw_is_allowed_and_is_a_different_verb():
    # channel -> escrow. The funds never leave the escrow, so this one is safe.
    assert "reclaim/withdraw" in wk.ALLOWED_CONTROL_OPS
    k = _FakeControl()
    assert _run(k.control("reclaim/withdraw"))["ok"] is True


# ---- top-up: cooldown, daily cap, wallet floor -------------------------------

def _fire_one_topup(k, available="0.5", reserved="0.0"):
    seed_buyer_status(PID, deposits_available=available, deposits_reserved=reserved)
    return _run(k._maybe_topup(PID, _knobs(), float(available)))


def test_topup_fires_below_the_trigger_and_writes_the_intent_first():
    k = _FakeControl()
    assert _fire_one_topup(k) == "topup_fired"
    assert k.calls == [("deposit", {"amount": "5.000000"})]
    (row,) = host_store.wallet_ops_recent(PID)
    assert row["op"] == "topup" and row["amount_usdc"] == 5.0
    assert row["pre_available"] == 0.5 and row["outcome"] == "fired"
    assert "trigger" in row["reason"]


def test_topup_skipped_when_already_funded():
    seed_buyer_status(PID, deposits_available="9.0", deposits_reserved="0.0")
    k = _FakeControl()
    assert _run(k._maybe_topup(PID, _knobs(), 9.0)) == "funded"
    assert k.calls == []


def test_topup_cooldown_blocks_a_second_deposit():
    k = _FakeControl()
    assert _fire_one_topup(k) == "topup_fired"
    assert _fire_one_topup(k) == "cooldown"
    assert len(k.calls) == 1, "the cooldown must stop the second deposit"


def test_topup_cooldown_is_derived_from_the_durable_ledger_not_memory():
    # A pod restart (a fresh keeper object) must not reset the cooldown.
    assert _fire_one_topup(_FakeControl()) == "topup_fired"
    assert _fire_one_topup(_FakeControl()) == "cooldown"


def test_topup_daily_cap_blocks_further_deposits():
    # cap 10, amount 5, cooldown 0 -> exactly two deposits per rolling 24h.
    knobs = _knobs(topup_cooldown_s=0)
    k = _FakeControl()
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    assert _run(k._maybe_topup(PID, knobs, 0.5)) == "topup_fired"
    assert _run(k._maybe_topup(PID, knobs, 0.5)) == "topup_fired"
    assert _run(k._maybe_topup(PID, knobs, 0.5)) == "daily_cap"
    assert len(k.calls) == 2


def test_daily_cap_counts_a_deposit_of_unknown_outcome():
    # An intent row the keeper never got an answer for still counts against the
    # cap — the conservative reading, since the transaction may well have landed.
    knobs = _knobs(topup_cooldown_s=0)
    op_id = host_store.wallet_op_begin(PID, "topup", amount_usdc=9.0,
                                       pre_available=0.5)
    host_store.wallet_op_finish(op_id, "unknown")
    # Age it past the error backoff, so the CAP is what is under test here and
    # not the backoff an `unknown` outcome now also earns.
    _age_all_topups(-1)
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    k = _FakeControl()
    assert _run(k._maybe_topup(PID, knobs, 0.5)) == "daily_cap"
    assert k.calls == []


def test_topup_blocked_when_it_would_breach_the_hot_wallet_floor(monkeypatch):
    # The hot-wallet balance is an UNTRUSTED public-RPC read: veto-only.
    _chain(usdc=5.2, eth=0.01)
    k = _FakeControl()
    # 5.2 - 5.0 = 0.2, below the 1.0 floor -> refuse.
    assert _fire_one_topup(k) == "wallet_floor"
    assert k.calls == []


def test_an_unreadable_hot_wallet_VETOES_the_deposit(monkeypatch):
    """INVERTED. This used to assert that no RPC reading at all still fires the
    deposit ("unknown does not authorize or block"), which quietly made the floor
    optional: `_fetch_chain_balances` returns {} on ANY failure, so the key is
    simply absent and the check is SKIPPED. An RPC outage — or whoever controls
    the default public endpoint, https://mainnet.base.org — removed the guardrail
    by not answering, and the module's claim that an attacker "can never induce a
    spend" was false as written. A veto that only works when the attacker
    cooperates is not a veto."""
    _no_chain()
    k = _FakeControl()
    assert _fire_one_topup(k) == "wallet_unreadable"
    assert k.calls == [], "no reading, no deposit"

    # ...and the cost of failing closed is a DELAY, not a loss: the reading
    # refreshes every poll tick and the deposit goes through on the next cycle.
    _chain()
    assert _fire_one_topup(k) == "topup_fired"


def test_a_stale_hot_wallet_reading_is_no_reading_at_all():
    # The balance is present but was fetched six hours ago. Treating it as
    # current is how a long-since-spent wallet authorizes a deposit the floor
    # would have blocked — the reading is not weaker with age, it is a different
    # wallet state.
    _chain(usdc=500.0, age_s=wk.CHAIN_READ_MAX_AGE_S + 60)
    k = _FakeControl()
    assert _fire_one_topup(k) == "wallet_unreadable"
    assert k.calls == []


def test_topup_never_exceeds_the_hard_per_deposit_limit():
    # Even if a knob somehow carried a larger value past validation, the amount
    # handed to the control plane is clamped.
    k = _FakeControl()
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    _run(k._maybe_topup(PID, _knobs(topup_amount_usdc=500.0,
                                    topup_daily_cap_usdc=1000.0), 0.5))
    (_op, body) = k.calls[0]
    assert float(body["amount"]) == wk.MAX_TOPUP_USDC


def test_topup_refuses_to_fire_when_the_intent_row_cannot_be_persisted(monkeypatch):
    # Audit-before-action: an on-chain spend with no ledger row is worse than a
    # missed top-up (no cap accounting, no reconciliation, no audit trail).
    monkeypatch.setattr(host_store, "wallet_op_begin", lambda *a, **kw: None)
    k = _FakeControl()
    assert _fire_one_topup(k) == "ledger_unavailable"
    assert k.calls == []


def test_a_deposit_that_never_reached_the_cli_is_failed_and_costs_nothing():
    """INVERTED, and narrowed. This test used to accept ANY unsuccessful
    response as `failed` — an outcome that consumes neither the daily cap nor the
    cooldown — which is only sound when the buyer CLI provably never ran. Now the
    sidecar says so explicitly (`attempted`), and only that answer keeps the
    cheap outcome. Here: a 400 from the amount validator, rejected before the CLI
    is invoked."""
    k = _FakeControl(responses={"deposit": {
        "ok": False, "attempted": False, "error": "amount must be positive"}})
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    assert _run(k._maybe_topup(PID, _knobs(topup_cooldown_s=0), 0.5)) == "topup_failed"
    (row,) = host_store.wallet_ops_recent(PID)
    assert row["outcome"] == "failed" and "positive" in row["detail"]
    spend = host_store.wallet_op_spend_since(PID, "topup", 0)
    assert spend["spent_usdc"] == 0.0, "a deposit that never ran spent nothing"


def test_a_TIMED_OUT_deposit_is_unknown_and_COUNTS_AS_SPENT(caplog):
    """THE INVERSION THAT MATTERS. A timed-out deposit used to be recorded
    `failed`, and `failed` consumes neither the cap nor the cooldown — so the
    keeper would re-fire on the very next cycle, on top of a transaction the
    sidecar was still executing. Real USDC moved and the ledger recorded nothing.

    The keeper's client timeout could not save it either: 130s against a sidecar
    whose worst case was ~150s behind an unbounded queue, so a dashboard deposit
    or a 240s reclaim phase queued ahead GUARANTEED the timeout on a request that
    would still execute.

    `host_store.wallet_op_spend_since` already reasoned this way for the crash
    path ("a keeper that died mid-deposit must ASSUME the transaction landed");
    the timeout path now agrees."""
    k = _FakeControl(responses={"deposit": {
        "ok": False, "attempted": True, "error": "ReadTimeout: timed out"}})
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    with caplog.at_level("ERROR"):
        assert _run(k._maybe_topup(PID, _knobs(topup_cooldown_s=0), 0.5)) == "topup_unknown"
    assert "UNRESOLVED" in caplog.text and "may have landed" in caplog.text

    (row,) = host_store.wallet_ops_recent(PID)
    assert row["outcome"] == "unknown"
    spend = host_store.wallet_op_spend_since(PID, "topup", 0)
    assert spend["spent_usdc"] == 5.0, \
        "a deposit that may have landed must consume the cap"
    assert spend["last_ts"] is not None, "...and start the cooldown"


def test_every_inconclusive_transport_failure_counts_as_spent():
    # The full set the reviews named: a read timeout, a reset connection, a 502
    # from a CLI that exited non-zero, a 504 from a CLI killed mid-broadcast.
    # None of them can rule out a broadcast transaction.
    for error in ("ReadTimeout", "ConnectError: reset by peer",
                  "cli failed", "timed out after 120000ms and was killed"):
        host_store.wallet_clear_halt(PID, "topup")
        with host_store._get_pool().connection() as conn:
            conn.execute("DELETE FROM wallet_ops WHERE pid=%s", (PID,))
        k = _FakeControl(responses={"deposit": {
            "ok": False, "attempted": True, "error": error}})
        seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
        assert _run(k._maybe_topup(PID, _knobs(topup_cooldown_s=0), 0.5)) == "topup_unknown"
        assert host_store.wallet_op_spend_since(PID, "topup", 0)["spent_usdc"] == 5.0, \
            f"{error!r} must count as spent"


def test_an_unrecognised_failure_defaults_to_unknown_not_failed():
    # A response with no `attempted` field at all (an older sidecar, a proxy in
    # between). Being wrong in the conservative direction burns a cap slot; being
    # wrong the other way moves USDC with nothing in the ledger.
    k = _FakeControl(responses={"deposit": {"ok": False, "error": "???"}})
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    assert _run(k._maybe_topup(PID, _knobs(topup_cooldown_s=0), 0.5)) == "topup_unknown"
    assert host_store.wallet_op_spend_since(PID, "topup", 0)["spent_usdc"] == 5.0


def test_the_client_timeout_strictly_exceeds_the_sidecars_worst_case():
    """A client timeout below the server's worst case is not a timeout, it is a
    lie: the sidecar goes on executing a request the keeper has written off. The
    numbers here mirror antseed/control.js, which publishes its own budgets on
    /budgets; if that file's constants move, this fails."""
    control_js = (ROOT / "antseed" / "control.js").read_text()
    for name, expected in (("DEPOSIT_TIMEOUT_MS", wk.CONTROL_DEPOSIT_S),
                           ("STATUS_TIMEOUT_MS", wk.CONTROL_STATUS_S),
                           ("DB_TIMEOUT_MS", wk.CONTROL_DB_S),
                           ("QUEUE_WAIT_BUDGET_MS", wk.CONTROL_QUEUE_WAIT_S),
                           ("RECLAIM_SCAN_TIMEOUT_MS", wk.CONTROL_RECLAIM_SCAN_S),
                           ("RECLAIM_TX_TIMEOUT_MS", wk.CONTROL_RECLAIM_TX_S)):
        m = re.search(rf"^const {name} = (\d+);", control_js, re.M)
        assert m, f"{name} not found in antseed/control.js"
        assert int(m.group(1)) == expected * 1000, \
            f"{name} moved in control.js; wallet_keeper's budget is now a lie"

    # deposit: queue wait + CLI + post-op status write + DB, and then some.
    assert wk.DEPOSIT_TIMEOUT_S > (wk.CONTROL_QUEUE_WAIT_S + wk.CONTROL_DEPOSIT_S
                                   + wk.CONTROL_STATUS_S + wk.CONTROL_DB_S)
    assert wk.RECLAIM_TX_TIMEOUT_S > (wk.CONTROL_QUEUE_WAIT_S + wk.CONTROL_RECLAIM_TX_S
                                      + wk.CONTROL_STATUS_S + wk.CONTROL_DB_S)
    assert wk.RECLAIM_SCAN_TIMEOUT_S > wk.CONTROL_RECLAIM_SCAN_S


def test_the_deposit_call_actually_uses_that_budget():
    # The constant is worthless if the call site still passes its own number.
    seen = {}

    class _Timed(_FakeControl):
        async def _control_post(self, op, body, timeout):
            seen[op] = timeout
            return {"ok": True}

    k = _Timed()
    _fire_one_topup(k)
    assert seen["deposit"] == wk.DEPOSIT_TIMEOUT_S


# ---- the real transport: what `attempted` is derived FROM --------------------
# Everything above rides `_FakeControl`, which replaces `_control_post` wholesale
# — so the mapping that decides whether real money may have moved is not covered
# by any of it. These drive the actual method.

class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code, self._payload, self.text = status_code, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _post_returning(monkeypatch, result):
    """Point `_control_post`'s httpx call at `result` — a response, or an
    exception instance to raise."""
    import httpx

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **kw):
            if isinstance(result, BaseException):
                raise result
            return result

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _Client())


def test_control_post_marks_a_transport_failure_as_ATTEMPTED(monkeypatch):
    # The C1 core. A read timeout means the request reached the wire and the
    # sidecar may be executing it right now.
    import httpx
    k = wk.WalletKeeper([PID])
    for exc in (httpx.ReadTimeout("timed out"), httpx.ConnectError("reset"),
                httpx.RemoteProtocolError("boom"), RuntimeError("something else")):
        _post_returning(monkeypatch, exc)
        resp = _run(k._control_post("deposit", {"amount": "1.0"}, 1.0))
        assert resp["ok"] is False
        assert resp["attempted"] is True, f"{type(exc).__name__} must count as spent"


def test_control_post_marks_a_pre_flight_failure_as_NOT_attempted(monkeypatch):
    # Raised while BUILDING the request — and NOT subclasses of HTTPError, so
    # they used to escape `_control_post` entirely and leave the caller's intent
    # row dangling as `pending` (M-9).
    import httpx
    k = wk.WalletKeeper([PID])
    for exc in (httpx.InvalidURL("bad"), httpx.UnsupportedProtocol("nope")):
        _post_returning(monkeypatch, exc)
        resp = _run(k._control_post("deposit", {"amount": "1.0"}, 1.0))
        assert resp["ok"] is False and resp["attempted"] is False


def test_control_post_with_no_endpoint_is_not_attempted(monkeypatch):
    monkeypatch.delenv("ANTSEED_CONTROL_URL", raising=False)
    resp = _run(wk.WalletKeeper([PID])._control_post("deposit", None, 1.0))
    assert resp["ok"] is False and resp["attempted"] is False


def test_control_post_trusts_the_sidecars_own_attempted_flag(monkeypatch):
    # control.js states it per-branch, and it knows things the status code does
    # not — e.g. a 504 from a CLI it killed mid-broadcast.
    k = wk.WalletKeeper([PID])
    _post_returning(monkeypatch, _Resp(504, {"error": "killed", "attempted": True}))
    assert _run(k._control_post("deposit", None, 1.0))["attempted"] is True

    _post_returning(monkeypatch, _Resp(400, {"error": "bad amount",
                                             "attempted": False}))
    assert _run(k._control_post("deposit", None, 1.0))["attempted"] is False


def test_control_post_falls_back_to_the_status_code(monkeypatch):
    # An older sidecar, or something in between, that says nothing about it.
    k = wk.WalletKeeper([PID])
    for status, expected in ((400, False), (401, False), (404, False), (429, False),
                             (502, True), (500, True), (504, True), (503, True)):
        _post_returning(monkeypatch, _Resp(status, {"error": "x"}))
        got = _run(k._control_post("deposit", None, 1.0))["attempted"]
        assert got is expected, f"HTTP {status} should be attempted={expected}"


def test_control_post_treats_a_non_json_error_body_as_attempted(monkeypatch):
    # A proxy's HTML error page. Unparseable is not evidence of anything.
    k = wk.WalletKeeper([PID])
    _post_returning(monkeypatch, _Resp(502, None, text="<html>bad gateway</html>"))
    resp = _run(k._control_post("deposit", None, 1.0))
    assert resp["attempted"] is True and "bad gateway" in resp["error"]


def test_a_success_is_attempted_even_if_the_sidecar_omits_the_field(monkeypatch):
    k = wk.WalletKeeper([PID])
    _post_returning(monkeypatch, _Resp(200, {"ok": True, "stdout": "done"}))
    resp = _run(k._control_post("deposit", None, 1.0))
    assert resp["ok"] is True and resp["attempted"] is True


def test_the_intent_row_never_dangles_when_the_deposit_call_raises(monkeypatch):
    """M-9: `import httpx` sat outside the try and the except caught only
    `httpx.HTTPError`, so an ImportError/InvalidURL propagated AFTER the intent
    row was written — leaving it `pending`, later reconciled to `unknown`, i.e.
    burning a cap slot for a request that never left the process.
    `_fire_reclaim_phase` already had this guard; `_maybe_topup` did not."""
    class _Boom(_FakeControl):
        async def _control_post(self, op, body, timeout):
            self.calls.append((op, body))
            raise RuntimeError("something unforeseen")

    k = _Boom()
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    with pytest.raises(RuntimeError):
        _run(k._maybe_topup(PID, _knobs(topup_cooldown_s=0), 0.5))
    (row,) = host_store.wallet_ops_recent(PID)
    assert row["outcome"] == "unknown", "the row must never be left pending"
    assert host_store.wallet_ops_open(PID) == []


# ---- the error breaker (C2): failed/unknown must not re-fire forever ---------

def _failing_keeper(error="ReadTimeout", attempted=True):
    return _FakeControl(responses={"deposit": {
        "ok": False, "attempted": attempted, "error": error}})


def test_repeated_unresolvable_deposits_back_off_instead_of_re_firing():
    """Neither cap, cooldown nor breaker used to see a `failed` deposit, so a
    permanently failing top-up re-fired every 60s forever with no backoff and no
    strike count anywhere. The backoff floor applies even at cooldown 0, because
    an operator may legitimately set 0 and 0 doubled is still 0."""
    knobs = _knobs(topup_cooldown_s=0)
    k = _failing_keeper()
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    assert _run(k._maybe_topup(PID, knobs, 0.5)) == "topup_unknown"
    # One strike -> the backoff floor now stands between us and the next attempt,
    # despite the operator's cooldown being 0.
    assert wk.TOPUP_BACKOFF_BASE_S > wk.CYCLE_S
    assert k._topup_cooldown_s(PID, knobs) == wk.TOPUP_BACKOFF_BASE_S
    assert _run(k._maybe_topup(PID, knobs, 0.5)) == "cooldown"
    assert len(k.calls) == 1, "the backoff must stop the immediate retry"


def test_the_backoff_doubles_per_strike_and_is_capped():
    knobs = _knobs(topup_cooldown_s=0)
    k = _failing_keeper()
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    for strike in range(1, wk.TOPUP_ERROR_STRIKES_TO_HALT):
        _age_all_topups(-1)          # let the previous backoff elapse
        assert _run(k._maybe_topup(PID, knobs, 0.5)) == "topup_unknown"
        assert k._topup_cooldown_s(PID, knobs) == min(
            wk.TOPUP_BACKOFF_BASE_S * 2 ** (strike - 1), wk.TOPUP_BACKOFF_CAP_S)


def test_consecutive_unresolvable_deposits_hard_halt_the_topups(caplog):
    """The pump breaker only ever counted MEASURED misses (`ineffective`), so a
    deposit that could never be measured at all escaped it entirely. Three in a
    row — each of which may have moved USDC — is a stop-and-get-a-human."""
    # A cap wide enough that the BREAKER is what stops this, not the cap. The
    # cap is in fact the first layer — since C1 an `unknown` deposit consumes it
    # — but the breaker has to work independently of how it is set.
    knobs = _knobs(topup_cooldown_s=0, topup_daily_cap_usdc=1000.0)
    k = _failing_keeper()
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    for _ in range(wk.TOPUP_ERROR_STRIKES_TO_HALT):
        _age_all_topups(-1)
        assert _run(k._maybe_topup(PID, knobs, 0.5)) == "topup_unknown"
    assert host_store.wallet_halted(PID, "topup") is False, "not until settle runs"

    with caplog.at_level("ERROR"):
        assert k._settle_topups(PID, 0.5, 0.0) == "error_halt"
    assert "HARD HALT" in caplog.text
    assert host_store.wallet_halted(PID, "topup") is True
    assert _run(k._maybe_topup(PID, knobs, 0.5)) == "halted"


def test_one_success_breaks_the_error_run():
    # A consecutive-failure breaker that counts non-consecutive failures is just
    # a lifetime counter, and it would eventually halt a healthy system.
    knobs = _knobs(topup_cooldown_s=0, topup_daily_cap_usdc=1000.0)
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    bad = _failing_keeper()
    for _ in range(wk.TOPUP_ERROR_STRIKES_TO_HALT - 1):
        _age_all_topups(-1)
        _run(bad._maybe_topup(PID, knobs, 0.5))
    _age_all_topups(-1)
    assert _run(_FakeControl()._maybe_topup(PID, knobs, 0.5)) == "topup_fired"
    _age_all_topups(-1)
    assert _run(bad._maybe_topup(PID, knobs, 0.5)) == "topup_unknown"
    assert bad._settle_topups(PID, 0.5, 0.0) != "error_halt"
    assert host_store.wallet_halted(PID, "topup") is False


def _age_all_topups(ago_s):
    """Backdate every topup row so cooldowns/backoffs have elapsed."""
    with host_store._get_pool().connection() as conn:
        conn.execute("UPDATE wallet_ops SET ts=%s, updated_at=%s WHERE pid=%s",
                     (int(time.time()) - max(ago_s, wk.TOPUP_BACKOFF_CAP_S + 1),
                      int(time.time()) - max(ago_s, wk.TOPUP_BACKOFF_CAP_S + 1), PID))


# ---- the money-pump breaker --------------------------------------------------

def _settled_topup(pre, amount, post, ago_s=120, pre_reserved=0.0):
    """A previously fired deposit, old enough to be judged."""
    op_id = host_store.wallet_op_begin(PID, "topup", amount_usdc=amount,
                                       pre_available=pre,
                                       pre_reserved=pre_reserved)
    host_store.wallet_op_finish(op_id, "fired")
    with host_store._get_pool().connection() as conn:
        conn.execute("UPDATE wallet_ops SET ts=%s, updated_at=%s WHERE id=%s",
                     (int(time.time()) - ago_s, int(time.time()) - ago_s, op_id))
    return op_id


def test_effective_deposit_is_recorded_as_effective():
    _settled_topup(pre=0.5, amount=5.0, post=5.5)
    k = _FakeControl()
    assert k._settle_topups(PID, 5.5, 0.0) is None  # 0.5 -> 5.5 = +5.0 of 5.0
    (row,) = host_store.wallet_ops_settled(PID, "topup", limit=5)
    assert row["outcome"] == "effective" and row["post_available"] == 5.5


def test_deposit_that_moves_less_than_80_percent_is_ineffective():
    _settled_topup(pre=0.5, amount=5.0, post=3.0)
    k = _FakeControl()
    k._settle_topups(PID, 3.0, 0.0)                 # +2.5 of 5.0 = 50%
    (row,) = host_store.wallet_ops_settled(PID, "topup", limit=5)
    assert row["outcome"] == "ineffective"


def test_a_deposit_reserved_by_an_opening_channel_is_still_effective():
    """The breaker scored the SPENDABLE half of the escrow, so a deposit that
    landed and was immediately reserved by an opening channel looked like a
    deposit that never arrived — and a channel opens in the ~90s settle window
    whenever the provider is busy. Two busy cycles would hard-halt a perfectly
    healthy system. The escrow is `available + reserved`; measuring one column of
    a two-column ledger is what produced the false positive."""
    _settled_topup(pre=0.5, amount=5.0, post=1.5, pre_reserved=0.0)
    k = _FakeControl()
    # +1.0 spendable, +4.0 reserved by channels that opened: the money arrived.
    assert k._settle_topups(PID, 1.5, 4.0) is None
    (row,) = host_store.wallet_ops_settled(PID, "topup", limit=5)
    assert row["outcome"] == "effective" and row["post_reserved"] == 4.0


def test_a_reclaim_landing_in_the_window_no_longer_masks_a_missing_deposit():
    # The symmetric error: reclaim returns 5.0 from channels to `available` while
    # the deposit itself never arrived. Scoring `available` alone read the
    # reclaim's gain as the deposit's, and a genuinely ineffective deposit passed.
    _settled_topup(pre=0.5, amount=5.0, post=5.5, pre_reserved=8.0)
    k = _FakeControl()
    # available 0.5 -> 5.5, but reserved fell 8.0 -> 3.0: the escrow is flat.
    k._settle_topups(PID, 5.5, 3.0)
    (row,) = host_store.wallet_ops_settled(PID, "topup", limit=5)
    assert row["outcome"] == "ineffective"


def test_two_consecutive_ineffective_deposits_hard_halt_topups():
    # The money pump: deposits keep landing somewhere the escrow never sees.
    # One strike is noise; two in a row means STOP and get a human.
    _settled_topup(pre=0.5, amount=5.0, post=0.6, ago_s=300)
    k = _FakeControl()
    k._settle_topups(PID, 0.6, 0.0)
    assert host_store.wallet_halted(PID, "topup") is False, "one strike is not a halt"
    _settled_topup(pre=0.6, amount=5.0, post=0.7, ago_s=120)
    assert k._settle_topups(PID, 0.7, 0.0) == "pump_halt"
    assert host_store.wallet_halted(PID, "topup") is True
    # ...and the halt actually stops spending.
    seed_buyer_status(PID, deposits_available="0.7", deposits_reserved="0.0")
    assert _run(k._maybe_topup(PID, _knobs(topup_cooldown_s=0), 0.7)) == "halted"
    assert k.calls == []


def test_the_halt_survives_a_restart_and_only_an_operator_clears_it():
    host_store.wallet_halt(PID, "topup", "test")
    seed_buyer_status(PID, deposits_available="0.1", deposits_reserved="0.0")
    fresh = _FakeControl()                          # a brand-new process
    assert _run(fresh._maybe_topup(PID, _knobs(topup_cooldown_s=0), 0.1)) == "halted"
    host_store.wallet_clear_halt(PID, "topup")
    assert host_store.wallet_halted(PID, "topup") is False
    assert _run(fresh._maybe_topup(PID, _knobs(topup_cooldown_s=0), 0.1)) == "topup_fired"


def test_an_unreadable_ledger_reads_as_halted_not_as_free_to_spend(monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("db down")
    monkeypatch.setattr(host_store, "_get_pool", _boom)
    assert host_store.wallet_halted(PID, "topup") is True
    spend = host_store.wallet_op_spend_since(PID, "topup", 0)
    assert spend["spent_usdc"] == float("inf"), "an unreadable cap is a full cap"


def test_a_halt_that_cannot_be_persisted_is_not_reported_as_a_halt(monkeypatch,
                                                                   caplog):
    """`wallet_halt` short-circuited on `wallet_halted`, which fails SOFT TO TRUE
    — so on a store outage it returned True without writing anything and the
    keeper logged "HARD HALT" for a breaker that would evaporate on the next
    restart. The de-duplication is now one statement, and the caller is told."""
    def _boom(*a, **kw):
        raise RuntimeError("db down")
    monkeypatch.setattr(host_store, "_get_pool", _boom)
    assert host_store.wallet_halt(PID, "topup", "because") is False

    monkeypatch.undo()
    assert host_store.wallet_halted(PID, "topup") is False, "nothing was written"
    with caplog.at_level("ERROR"):
        wk.WalletKeeper._halt_topups(PID, "pump_halt", "test")
    assert "HARD HALT" in caplog.text and host_store.wallet_halted(PID, "topup")


def test_wallet_halt_is_idempotent_and_writes_exactly_one_row():
    assert host_store.wallet_halt(PID, "topup", "first") is True
    assert host_store.wallet_halt(PID, "topup", "second") is True
    halts = [r for r in host_store.wallet_ops_recent(PID) if r["outcome"] == "halted"]
    assert len(halts) == 1 and "first" in halts[0]["reason"]


def test_pending_intent_is_reconciled_as_unknown_on_restart():
    # The keeper died between writing the intent and hearing back. Whether the
    # transaction landed is unknowable: count it as spent.
    op_id = host_store.wallet_op_begin(PID, "topup", amount_usdc=5.0,
                                       pre_available=0.5)
    assert host_store.wallet_ops_open(PID, "topup")[0]["outcome"] == "pending"
    k = _FakeControl()
    assert k._settle_topups(PID, 0.5, 0.0) is None  # NOT a pump strike
    (row,) = host_store.wallet_ops_recent(PID)
    assert row["id"] == op_id and row["outcome"] == "unknown"
    assert host_store.wallet_halted(PID, "topup") is False


def test_orphaned_reclaim_intents_are_reconciled_too():
    """`_settle_topups` scans only `op="topup"`, so a reclaim intent whose control
    call died left a row nothing would ever revisit — permanently `pending`, no
    reader able to interpret it and the reclaim breaker unable to count it."""
    op_id = host_store.wallet_op_begin(PID, "reclaim_withdraw", pre_available=0.5)
    assert host_store.wallet_ops_open(PID)[0]["outcome"] == "pending"
    _FakeControl()._reconcile_orphans(PID)
    (row,) = host_store.wallet_ops_recent(PID)
    assert row["id"] == op_id and row["outcome"] == "unknown"
    assert host_store.wallet_ops_open(PID) == []


def test_a_deposit_is_not_judged_before_a_status_cycle_has_passed():
    _settled_topup(pre=0.5, amount=5.0, post=0.5, ago_s=5)   # fired 5s ago
    k = _FakeControl()
    k._settle_topups(PID, 0.5, 0.0)
    assert host_store.wallet_ops_settled(PID, "topup", limit=5) == []
    assert host_store.wallet_ops_open(PID, "topup")[0]["outcome"] == "fired"


# ---- reclaim -----------------------------------------------------------------

_SCAN_OK = {
    "ok": True, "operatorIsSelf": True, "operatorSet": True,
    "channels": [
        {"id": "c1", "reclaimable": "4.000000", "closeRequested": False},
        {"id": "c2", "reclaimable": "0.010000", "closeRequested": False},
    ],
}


def _wedge(pid=PID, family="qwen3", peer="peerA", n=12):
    """n failed attempts and no successes in the last hour — the "provider is
    fully wedged" evidence the reclaim trigger requires. n must clear
    WEDGE_MIN_ATTEMPTS: route_observations is written through a lossy queue, so a
    handful of rows is not evidence for force-closing anything."""
    seed_route_obs(pid, family, peer, ok=False, n=n)


def test_reclaim_requires_short_funds_reserved_escrow_and_a_wedged_provider():
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    knobs = _knobs()
    # funded, nothing ratcheted, traffic unknown -> no reclaim
    assert _run(k._maybe_reclaim(PID, knobs, available=9.0, reserved=3.5)) == "not_short"
    # short but little reserved -> nothing to unwind
    assert _run(k._maybe_reclaim(PID, knobs, 1.5, 1.0)) == "nothing_reserved"
    # short + reserved, but no traffic at all AND the escrow is above the
    # tourniquet -> IDLE, not wedged
    assert _run(k._maybe_reclaim(PID, knobs, 1.5, 20.0)) == "not_wedged"
    assert k.calls == []


def test_reclaim_never_churns_a_provider_that_is_still_succeeding():
    seed_route_obs(PID, "qwen3", "peerA", ok=False, n=9)
    seed_route_obs(PID, "qwen3", "peerA", ok=True, n=1)      # one success
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    assert _run(k._maybe_reclaim(PID, _knobs(), 1.5, 20.0)) == "not_wedged"
    assert k.calls == []


def test_a_tourniquet_armed_escrow_IS_evidence_of_a_wedge():
    """H-1: reclaim was structurally unreachable in exactly the prod state that
    motivated this module. Escrow $0.23, $15.63 stranded, tourniquet armed at
    1.1 -> the offer gate suppresses every antseed offer -> no attempts are ever
    made -> after an hour the attempt-based wedge test sees `total == 0`, reads
    it as IDLE, and returns `not_wedged` forever.

    "Offers are suppressed right now" is a STRONGER proof that the channels are
    not working capital than any number of failed attempts: there cannot be
    attempts, by construction."""
    knobs = _knobs()                                   # min_available 1.1
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    assert host_store.provider_attempt_counts(PID)["total"] == 0, "no traffic"
    assert k._wedged(PID, knobs, available=0.23) is True
    assert _run(k._maybe_reclaim(PID, knobs, 0.23, 15.63)) == "reclaim_request_close"
    assert k.ops == ["reclaim/scan", "reclaim/request-close"]


def test_reclaim_still_fires_once_a_topup_has_lifted_the_escrow():
    """H-1's second half. Top-up has no wedged requirement, so it deposits $5 and
    `available` rises above the trigger — after which a shortness-only test
    returns `not_short` forever and the stranded USDC is unrecoverable by
    automation. Escrow only comes back down by spending, so the lockout was
    permanent: the keeper was a money pump IN only, contradicting its own
    docstring. A reserve that dwarfs the spendable escrow is a ratchet worth
    unwinding whatever the balance says."""
    _wedge()
    knobs = _knobs()                                   # trigger 2.0
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    # Topped up to 5.0 — comfortably "funded" — with 15.63 stuck in channels.
    assert 15.63 >= wk.RATCHET_RESERVED_RATIO * max(5.0, knobs.topup_trigger_usdc)
    assert _run(k._maybe_reclaim(PID, knobs, 5.0, 15.63)) == "reclaim_request_close"


def test_a_handful_of_observations_is_not_evidence_for_a_force_close():
    """M-11: `route_observations` is written through a lossy queue that DROPS
    rows when full, so a spend decision derived from a tiny sample is derived
    from noise. Below the minimum the wedge test abstains."""
    knobs = _knobs()
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    seed_route_obs(PID, "qwen3", "peerA", ok=False, n=wk.WEDGE_MIN_ATTEMPTS - 1)
    assert k._wedged(PID, knobs, available=1.5) is False
    assert _run(k._maybe_reclaim(PID, knobs, 1.5, 20.0)) == "not_wedged"
    seed_route_obs(PID, "qwen3", "peerA", ok=False, n=1)      # now at the minimum
    assert k._wedged(PID, knobs, available=1.5) is True


def test_reclaim_requests_close_on_wedged_provider_and_skips_dust_channels():
    """STRENGTHENED. This used to assert only that the keeper *logged* one
    channel — against a mocked control plane, so it passed while the real sidecar
    ignored the filter entirely: reclaim.mjs acted on EVERY eligible channel and
    the only thing crossing the wire was a phase name. 100 channels with 1 worth
    reclaiming meant 100 transactions. The keeper now NAMES its channels, so the
    assertion is on the wire payload."""
    _wedge()
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "reclaim_request_close"
    assert k.ops == ["reclaim/scan", "reclaim/request-close"]
    # only c1 (4.0 USDC) counts; c2 holds 0.01, under MIN_CHANNEL_RECLAIMABLE
    assert k.calls[1] == ("reclaim/request-close", {"ids": ["c1"]}), \
        "the dust filter must reach the sidecar, not just the log line"
    (row,) = host_store.wallet_ops_recent(PID)
    assert row["op"] == "reclaim_request_close" and "1 of 2 channels" in row["reason"]


def test_reclaim_bootstraps_the_operator_once_then_stops_for_the_cycle():
    _wedge()
    scan = {**_SCAN_OK, "operatorIsSelf": False, "operatorSet": False}
    k = _FakeControl(responses={"reclaim/scan": scan})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "reclaim_set_operator"
    # set-operator only: the next cycle re-scans against the new chain state.
    assert k.ops == ["reclaim/scan", "reclaim/set-operator"]
    (row,) = host_store.wallet_ops_recent(PID)
    assert row["op"] == "reclaim_set_operator" and row["outcome"] == "ok"


def test_set_operator_does_not_re_fire_while_its_tx_is_unconfirmed():
    """M-13: reclaim.mjs's getOperator reads CONFIRMED state, so until the
    assignment lands every cycle saw `operatorIsSelf: false` and sent another
    setOperator — one redundant transaction per minute."""
    _wedge()
    scan = {**_SCAN_OK, "operatorIsSelf": False, "operatorSet": False}
    k = _FakeControl(responses={"reclaim/scan": scan})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "reclaim_set_operator"
    # The chain has not caught up yet; the scan still says "not self".
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "cooldown"
    assert k.ops.count("reclaim/set-operator") == 1


def test_reclaim_has_a_cooldown_of_its_own():
    """M-12: reclaim fires REAL transactions and had no cap, cooldown or breaker
    — at CYCLE_S it re-ran request-close every 60s. One challenge window is the
    natural spacing: nothing it starts can finish sooner."""
    _wedge()
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "reclaim_request_close"
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "cooldown"
    assert k.ops.count("reclaim/request-close") == 1
    # The read-only scan still runs every cycle, so /x/runtime stays current.
    assert k.ops.count("reclaim/scan") == 2

    # ...and the cooldown is durable, not process memory.
    with host_store._get_pool().connection() as conn:
        conn.execute("UPDATE wallet_ops SET ts=%s WHERE pid=%s",
                     (int(time.time()) - wk.RECLAIM_COOLDOWN_S - 1, PID))
    fresh = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    assert _run(fresh._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "reclaim_request_close"


def test_repeated_reclaim_failures_hard_halt_reclaim(caplog):
    _wedge()
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK,
                                "reclaim/request-close": {
                                    "ok": False, "attempted": True,
                                    "error": "rpc exploded"}})
    for _ in range(wk.RECLAIM_ERROR_STRIKES_TO_HALT):
        with host_store._get_pool().connection() as conn:
            conn.execute("UPDATE wallet_ops SET ts=%s WHERE pid=%s",
                         (int(time.time()) - wk.RECLAIM_COOLDOWN_S - 1, PID))
        with caplog.at_level("ERROR"):
            assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == \
                "reclaim_request_close_failed"
    assert host_store.wallet_halted(PID, "reclaim") is True
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "halted"


def test_reclaim_harvests_elapsed_channels_before_starting_new_closes():
    _wedge()
    scan = {**_SCAN_OK, "channels": [
        {"id": "c1", "reclaimable": "4.0", "closeRequested": False},
        {"id": "c2", "reclaimable": "3.0", "closeRequested": True},
    ]}
    k = _FakeControl(responses={"reclaim/scan": scan})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "reclaim_withdraw"
    assert k.ops == ["reclaim/scan", "reclaim/withdraw"]
    assert k.calls[1][1] == {"ids": ["c2"]}


def test_reclaim_caps_the_batch_instead_of_declining_it_entirely():
    """INVERTED behaviour, same invariant. The old code DECLINED any batch over
    the cap, because the sidecar acted on every eligible channel however few the
    keeper had chosen — the cap could only be honoured by doing nothing. That
    made a large channel set permanently unreclaimable by automation, the same
    dead end as the shortness test. Now the sidecar is told which channels to act
    on, so the cap binds: exactly MAX_TX_PER_CYCLE fire, richest first, and the
    rest follow next cycle."""
    _wedge()
    scan = {**_SCAN_OK, "channels": [
        {"id": f"c{i}", "reclaimable": f"{i + 1}.0", "closeRequested": False}
        for i in range(wk.MAX_TX_PER_CYCLE + 3)]}
    k = _FakeControl(responses={"reclaim/scan": scan})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "reclaim_request_close"
    (_op, body) = k.calls[1]
    assert len(body["ids"]) == wk.MAX_TX_PER_CYCLE, "the cap BINDS the sidecar"
    # richest first: c10 (11.0) down to c3 (4.0)
    assert body["ids"][0] == f"c{wk.MAX_TX_PER_CYCLE + 2}"


def test_an_empty_channel_selection_never_reaches_the_wire():
    # The dangerous coercion: the sidecar reads "no ids" as "act on EVERY
    # eligible channel", so an empty selection sent as no selection would widen
    # an intentionally-empty batch to unbounded. Both halves refuse — here, and
    # in antseed/ids.js (see its "an EMPTY list is an error" test).
    k = _FakeControl()
    out = _run(k._fire_reclaim_phase(PID, "reclaim/withdraw", "reclaim_withdraw",
                                     reason="x", pre_available=0.5, ids=[]))
    assert out == "empty_selection"
    assert k.calls == [] and host_store.wallet_ops_recent(PID) == []


def test_set_operator_sends_no_id_list_because_it_touches_no_channel():
    _wedge()
    scan = {**_SCAN_OK, "operatorIsSelf": False}
    k = _FakeControl(responses={"reclaim/scan": scan})
    _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0))
    assert k.calls[1] == ("reclaim/set-operator", None)


def test_an_unnamed_channel_declines_rather_than_firing_an_unbounded_batch():
    # A scan row with no id cannot be named, and an unnamed batch falls back to
    # the sidecar's act-on-everything path — which is the whole bug.
    _wedge()
    scan = {**_SCAN_OK, "channels": [
        {"id": "c1", "reclaimable": "4.0", "closeRequested": False},
        {"reclaimable": "9.0", "closeRequested": False}]}
    k = _FakeControl(responses={"reclaim/scan": scan})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "scan_malformed"
    assert k.ops == ["reclaim/scan"], "no transaction fired"


def test_reclaim_halts_below_the_gas_floor():
    _wedge()
    _chain(usdc=30.0, eth=0.0001)
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "gas_floor"
    assert k.calls == []


def test_an_unreadable_gas_balance_does_NOT_block_a_reclaim():
    """The deliberate asymmetry with the deposit path's USDC floor. Failing
    closed on an action that moves money OUT of the wallet is safe; failing
    closed on the one action that moves money BACK IN would let an RPC outage
    strand the escrow permanently — which is H-1 again, by another route. A
    reclaim fired without gas simply does not confirm."""
    _wedge()
    _no_chain()
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "reclaim_request_close"


def test_reclaim_refuses_to_fire_without_an_intent_row(monkeypatch):
    _wedge()
    monkeypatch.setattr(host_store, "wallet_op_begin", lambda *a, **kw: None)
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "ledger_unavailable"
    assert k.ops == ["reclaim/scan"]


# ---- reclaim BEFORE top-up ---------------------------------------------------

def test_a_cycle_reclaims_before_it_tops_up():
    # Topping up a ratchet without first unwinding it just refills the leak, so
    # the ordering is a correctness property, not a preference.
    _wedge()
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="20.0")
    settings.validate_and_write({"antseed.keeper_enabled": 1})
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    state = _run(k.cycle())
    assert k.ops == ["reclaim/scan", "reclaim/request-close", "deposit"]
    prov = state["providers"][PID]
    assert prov["reclaim"] == "reclaim_request_close" and prov["topup"] == "topup_fired"


def test_a_pump_halt_stops_the_topup_but_not_the_reclaim():
    # A pump halt says "stop putting money IN". Reclaim moves money the other
    # way — it is the REMEDY for a ratchet, not another symptom of one — so it
    # deliberately does not honour the topup halt (it has its own breaker).
    _wedge()
    # pre_reserved matches the status below: the reserve is flat across the
    # window, so the escrow genuinely did not move and the deposits are misses.
    _settled_topup(pre=0.4, amount=5.0, post=0.45, ago_s=300, pre_reserved=20.0)
    _settled_topup(pre=0.45, amount=5.0, post=0.5, ago_s=200, pre_reserved=20.0)
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="20.0")
    settings.validate_and_write({"antseed.keeper_enabled": 1})
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    prov = _run(k.cycle())["providers"][PID]
    assert prov["topup"] == "pump_halt"
    assert prov["reclaim"] == "reclaim_request_close"
    assert "deposit" not in k.ops


def test_cycle_does_nothing_without_a_configured_control_plane(monkeypatch):
    monkeypatch.delenv("ANTSEED_CONTROL_URL", raising=False)
    settings.validate_and_write({"antseed.keeper_enabled": 1})
    seed_buyer_status(PID, deposits_available="0.1", deposits_reserved="20.0")
    k = _FakeControl()
    state = _run(k.cycle())
    assert k.calls == [] and "not configured" in state["error"]


def test_one_provider_failing_does_not_stop_the_others(monkeypatch):
    seed_buyer_status("antseed_a", deposits_available="50.0", deposits_reserved="0.0")
    seed_buyer_status("antseed_b", deposits_available="50.0", deposits_reserved="0.0")
    settings.validate_and_write({"antseed.keeper_enabled": 1})
    k = _FakeControl(pids=("antseed_a", "antseed_b"))
    real = k.cycle_provider

    async def _flaky(pid, knobs):
        if pid == "antseed_a":
            raise RuntimeError("boom")
        return await real(pid, knobs)

    monkeypatch.setattr(k, "cycle_provider", _flaky)
    state = _run(k.cycle())
    assert state["providers"]["antseed_a"]["decision"] == "error"
    assert state["providers"]["antseed_b"]["decision"] == "acted"


# ---- catalog wiring ----------------------------------------------------------

def test_antseed_provider_ids_match_the_sources_predicate():
    catalog = {"providers": {
        "antseed": {"discovery": "marketplace", "discovery_id": "antseed"},
        "antseed_cheap": {"discovery": "marketplace", "discovery_id": "antseed_x"},
        "openrouter": {"discovery": "static"},
        "other_market": {"discovery": "marketplace", "discovery_id": "elsewhere"},
    }}
    assert sorted(wk.antseed_provider_ids(catalog)) == ["antseed", "antseed_cheap"]
    assert wk.start({"providers": {"openrouter": {"discovery": "static"}}}) is None


def test_a_reclaim_intent_row_never_dangles_when_the_call_raises():
    # There is no later pass that reconciles reclaim rows (unlike top-ups), so a
    # transport blow-up must still close out the intent as `unknown` rather than
    # leaving a permanently `pending` row in the audit trail.
    _wedge()

    class _Boom(_FakeControl):
        async def _control_post(self, op, body, timeout):
            self.calls.append((op, body))
            if op == "reclaim/scan":
                return _SCAN_OK
            raise RuntimeError("socket exploded")

    k = _Boom()
    with pytest.raises(RuntimeError):
        _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0))
    (row,) = host_store.wallet_ops_recent(PID)
    assert row["outcome"] == "unknown" and "socket exploded" in row["detail"]
    assert host_store.wallet_ops_open(PID) == []
