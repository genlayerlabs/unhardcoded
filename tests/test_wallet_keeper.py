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
    yield


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
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    k = _FakeControl()
    assert _run(k._maybe_topup(PID, knobs, 0.5)) == "daily_cap"
    assert k.calls == []


def test_topup_blocked_when_it_would_breach_the_hot_wallet_floor(monkeypatch):
    # The hot-wallet balance is an UNTRUSTED public-RPC read: veto-only.
    import sources as _sources
    _sources.SOURCE_STATE["antseed"] = {"balances": {PID: {
        "kind": "deposits_usdc", "value": 0.5,
        "detail": {"wallet_usdc": 5.2, "wallet_eth": 0.01}}}}
    try:
        k = _FakeControl()
        # 5.2 - 5.0 = 0.2, below the 1.0 floor -> refuse.
        assert _fire_one_topup(k) == "wallet_floor"
        assert k.calls == []
    finally:
        _sources.SOURCE_STATE.clear()


def test_unknown_hot_wallet_balance_does_not_authorize_or_block(monkeypatch):
    # No RPC reading at all (the source cache is empty, or the read is disabled).
    # The keeper proceeds: the CLI is the authority on whether a deposit can be
    # funded, and a failed deposit is recorded, not retried blindly.
    import sources as _sources
    _sources.SOURCE_STATE.clear()
    k = _FakeControl()
    assert _fire_one_topup(k) == "topup_fired"


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


def test_failed_deposit_is_recorded_as_failed_and_does_not_consume_the_cap():
    k = _FakeControl(responses={"deposit": {"ok": False, "error": "cli exploded"}})
    seed_buyer_status(PID, deposits_available="0.5", deposits_reserved="0.0")
    assert _run(k._maybe_topup(PID, _knobs(topup_cooldown_s=0), 0.5)) == "topup_failed"
    (row,) = host_store.wallet_ops_recent(PID)
    assert row["outcome"] == "failed" and "exploded" in row["detail"]
    spend = host_store.wallet_op_spend_since(PID, "topup", 0)
    assert spend["spent_usdc"] == 0.0, "a failed deposit spent nothing"


# ---- the money-pump breaker --------------------------------------------------

def _settled_topup(pre, amount, post, ago_s=120):
    """A previously fired deposit, old enough to be judged."""
    op_id = host_store.wallet_op_begin(PID, "topup", amount_usdc=amount,
                                       pre_available=pre)
    host_store.wallet_op_finish(op_id, "fired")
    with host_store._get_pool().connection() as conn:
        conn.execute("UPDATE wallet_ops SET ts=%s, updated_at=%s WHERE id=%s",
                     (int(time.time()) - ago_s, int(time.time()) - ago_s, op_id))
    return op_id


def test_effective_deposit_is_recorded_as_effective():
    _settled_topup(pre=0.5, amount=5.0, post=5.5)
    k = _FakeControl()
    assert k._settle_topups(PID, 5.5) is None       # 0.5 -> 5.5 = +5.0 of 5.0
    (row,) = host_store.wallet_ops_settled(PID, "topup", limit=5)
    assert row["outcome"] == "effective" and row["post_available"] == 5.5


def test_deposit_that_moves_less_than_80_percent_is_ineffective():
    _settled_topup(pre=0.5, amount=5.0, post=3.0)
    k = _FakeControl()
    k._settle_topups(PID, 3.0)                      # +2.5 of 5.0 = 50%
    (row,) = host_store.wallet_ops_settled(PID, "topup", limit=5)
    assert row["outcome"] == "ineffective"


def test_two_consecutive_ineffective_deposits_hard_halt_topups():
    # The money pump: deposits keep landing somewhere the router cannot spend
    # them. One strike is noise; two in a row means STOP and get a human.
    _settled_topup(pre=0.5, amount=5.0, post=0.6, ago_s=300)
    k = _FakeControl()
    k._settle_topups(PID, 0.6)
    assert host_store.wallet_halted(PID, "topup") is False, "one strike is not a halt"
    _settled_topup(pre=0.6, amount=5.0, post=0.7, ago_s=120)
    assert k._settle_topups(PID, 0.7) == "pump_halt"
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


def test_pending_intent_is_reconciled_as_unknown_on_restart():
    # The keeper died between writing the intent and hearing back. Whether the
    # transaction landed is unknowable: count it as spent, never as a strike.
    op_id = host_store.wallet_op_begin(PID, "topup", amount_usdc=5.0,
                                       pre_available=0.5)
    assert host_store.wallet_ops_open(PID, "topup")[0]["outcome"] == "pending"
    k = _FakeControl()
    assert k._settle_topups(PID, 0.5) is None       # NOT a pump strike
    (row,) = host_store.wallet_ops_recent(PID)
    assert row["id"] == op_id and row["outcome"] == "unknown"
    assert host_store.wallet_halted(PID, "topup") is False


def test_a_deposit_is_not_judged_before_a_status_cycle_has_passed():
    _settled_topup(pre=0.5, amount=5.0, post=0.5, ago_s=5)   # fired 5s ago
    k = _FakeControl()
    k._settle_topups(PID, 0.5)
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
    fully wedged" evidence the reclaim trigger requires."""
    seed_route_obs(pid, family, peer, ok=False, n=n)


def test_reclaim_requires_short_funds_reserved_escrow_and_a_wedged_provider():
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    knobs = _knobs()
    # funded -> no reclaim
    assert _run(k._maybe_reclaim(PID, knobs, available=9.0, reserved=20.0)) == "not_short"
    # short but little reserved -> nothing to unwind
    assert _run(k._maybe_reclaim(PID, knobs, 0.5, 1.0)) == "nothing_reserved"
    # short + reserved, but no traffic at all -> IDLE, not wedged
    assert _run(k._maybe_reclaim(PID, knobs, 0.5, 20.0)) == "not_wedged"
    assert k.calls == []


def test_reclaim_never_churns_a_provider_that_is_still_succeeding():
    seed_route_obs(PID, "qwen3", "peerA", ok=False, n=9)
    seed_route_obs(PID, "qwen3", "peerA", ok=True, n=1)      # one success
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "not_wedged"
    assert k.calls == []


def test_reclaim_requests_close_on_wedged_provider_and_skips_dust_channels():
    _wedge()
    k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "reclaim_request_close"
    assert k.ops == ["reclaim/scan", "reclaim/request-close"]
    (row,) = host_store.wallet_ops_recent(PID)
    # only c1 (4.0 USDC) counts; c2 holds 0.01, under MIN_CHANNEL_RECLAIMABLE
    assert row["op"] == "reclaim_request_close" and "1 channels" in row["reason"]


def test_reclaim_bootstraps_the_operator_once_then_stops_for_the_cycle():
    _wedge()
    scan = {**_SCAN_OK, "operatorIsSelf": False, "operatorSet": False}
    k = _FakeControl(responses={"reclaim/scan": scan})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "reclaim_set_operator"
    # set-operator only: the next cycle re-scans against the new chain state.
    assert k.ops == ["reclaim/scan", "reclaim/set-operator"]
    (row,) = host_store.wallet_ops_recent(PID)
    assert row["op"] == "reclaim_set_operator" and row["outcome"] == "ok"


def test_reclaim_harvests_elapsed_channels_before_starting_new_closes():
    _wedge()
    scan = {**_SCAN_OK, "channels": [
        {"id": "c1", "reclaimable": "4.0", "closeRequested": False},
        {"id": "c2", "reclaimable": "3.0", "closeRequested": True},
    ]}
    k = _FakeControl(responses={"reclaim/scan": scan})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "reclaim_withdraw"
    assert k.ops == ["reclaim/scan", "reclaim/withdraw"]


def test_reclaim_declines_a_batch_over_the_per_cycle_transaction_cap():
    # The sidecar's phases act on EVERY eligible channel in one invocation, so
    # the only way to honour a transaction cap is to decline and say so.
    _wedge()
    scan = {**_SCAN_OK, "channels": [
        {"id": f"c{i}", "reclaimable": "1.0", "closeRequested": False}
        for i in range(wk.MAX_TX_PER_CYCLE + 1)]}
    k = _FakeControl(responses={"reclaim/scan": scan})
    assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "tx_cap"
    assert k.ops == ["reclaim/scan"], "no transaction fired"


def test_reclaim_halts_below_the_gas_floor():
    _wedge()
    import sources as _sources
    _sources.SOURCE_STATE["antseed"] = {"balances": {PID: {
        "kind": "deposits_usdc", "value": 0.5,
        "detail": {"wallet_eth": 0.0001, "wallet_usdc": 30.0}}}}
    try:
        k = _FakeControl(responses={"reclaim/scan": _SCAN_OK})
        assert _run(k._maybe_reclaim(PID, _knobs(), 0.5, 20.0)) == "gas_floor"
        assert k.calls == []
    finally:
        _sources.SOURCE_STATE.clear()


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
    _wedge()
    _settled_topup(pre=0.4, amount=5.0, post=0.45, ago_s=300)
    _settled_topup(pre=0.45, amount=5.0, post=0.5, ago_s=200)
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
