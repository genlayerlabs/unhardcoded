"""AntSeed funding autonomy — the wallet keeper loop.

The AntSeed buyer pays for every routed call out of an on-chain USDC ESCROW, and
that escrow ratchets: opening a payment channel moves ~1 USDC from
`depositsAvailable` to `depositsReserved`, and a channel that never settles never
gives it back. Two manual operations kept it alive — `buyer deposit` (wallet ->
escrow) and channel reclaim (channel -> escrow) — and when nobody ran them the
provider spent a full day answering 402 `insufficient_deposits` to every single
call. This module closes that loop.

WHY IT LIVES IN THE ROUTER, not a CronJob and not the sidecar:
  * the buyer identity + its sqlite channel store live on an RWO EBS PVC bound to
    the router pod (`replicas: 1`, `Recreate`), so no separate pod can reach the
    state a funding decision depends on;
  * the sidecar's control server is deliberately pod-local (`127.0.0.1:8379`,
    no Service), and the router is the only process that already carries
    ANTSEED_CONTROL_URL / ANTSEED_CONTROL_TOKEN.

THE SHAPE OF ONE CYCLE (per buyer proxy):
  1. read state         — `host_store.buyer_status`, written by the sidecar every
                          60s. This is the ONLY trusted balance signal (below).
  2. settle             — measure the effect of the previous deposit; trip the
                          money-pump breaker if deposits stop landing.
  3. RECLAIM            — before top-up, always: topping up a ratchet without
                          first unwinding it just refills the leak.
  4. top-up             — deposit wallet -> escrow, under cap + cooldown + floor.

TRUST BOUNDARY. Decisions are made ONLY from the CLI-reported `buyer_status`.
`wallet_usdc` / `wallet_eth` come from a public Base RPC (`sources/antseed.py`
defaults to https://mainnet.base.org, an untrusted free endpoint) and are used
VETO-ONLY: a low reading can block an action, a high reading can never authorize
one. An attacker who controls that endpoint can therefore stall the keeper, but
never induce a spend.

SAFETY DIRECTION. The offer tourniquet in `sources/antseed.py` fails OPEN (a
read blip must not kill routing); this keeper fails CLOSED (a read blip must not
move money). Every guardrail below is written to that asymmetry.

Ships DARK: `antseed.keeper_enabled` defaults to 0 and must be armed deliberately.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import host_store
import settings
from sources.antseed import as_float

_log = logging.getLogger("unhardcoded.wallet_keeper")

# How often the loop wakes. Matched to the sidecar's 60s buyer_status write: a
# faster cycle would just re-read the same row, and every gate that matters
# (cooldown, daily cap) is time-based and derived from the durable ledger anyway.
CYCLE_S = 60

# A deposit's effect is judged no earlier than this after it fired. control.js
# re-writes buyer_status synchronously after a successful `buyer deposit`, so one
# cycle is already enough; the margin absorbs a slow status write.
SETTLE_AFTER_S = 90

# --- hard limits. Deliberately NOT operator knobs: these bound the blast radius
# of a misconfigured or hostile knob, so they must not be reachable from the same
# surface that sets the knobs (the dashboard Config tab).
MAX_TOPUP_USDC = 50.0          # mirrored server-side in antseed/control.js
MIN_TOPUP_USDC = 0.01          # below this a deposit is dust that only burns gas
GAS_FLOOR_ETH = 0.0005         # reclaim needs gas; below this, do not try
MAX_TX_PER_CYCLE = 8           # on-chain transactions one cycle may cause
MIN_CHANNEL_RECLAIMABLE = 0.05  # a channel holding less is not worth a close tx
NO_SUCCESS_WINDOW_S = 3600     # "fully wedged" = zero successes over this window
PUMP_EFFECTIVE_FRACTION = 0.8  # a deposit must lift `available` by >= this × amount
PUMP_STRIKES_TO_HALT = 2       # consecutive ineffective deposits before a hard halt

# The sidecar control verbs the keeper may call. EXACT-MATCH allowlist.
ALLOWED_CONTROL_OPS = frozenset({
    "deposit",
    "reclaim/scan",           # read-only
    "reclaim/set-operator",   # 1 tx, moves no funds
    "reclaim/request-close",  # 1 tx per eligible channel
    "reclaim/withdraw",       # channel -> escrow; never leaves the escrow
})

# `buyer withdraw` (escrow -> hot wallet) is absent BY CONSTRUCTION and must stay
# absent. It is the only control verb that moves funds out of the system, i.e.
# the exfiltration path if ANTSEED_CONTROL_TOKEN ever leaks — automating it would
# hand anyone holding that token a drain button that needs no human. It stays a
# dashboard-only, human-initiated action. Note `reclaim/withdraw` is a DIFFERENT
# verb (payment channel -> escrow) and is safe: the funds stay in escrow.
FORBIDDEN_CONTROL_OPS = frozenset({"withdraw", "buyer/withdraw", "/withdraw"})


class KnobError(ValueError):
    """The operator's knob set is internally inconsistent — the keeper stands
    down rather than act on a configuration that cannot mean what it says."""


@dataclass(frozen=True)
class Knobs:
    enabled: bool
    min_available_usdc: float
    topup_trigger_usdc: float
    topup_amount_usdc: float
    topup_wallet_floor_usdc: float
    topup_daily_cap_usdc: float
    topup_cooldown_s: int
    reclaim_min_usdc: float


def load_knobs() -> Knobs:
    """Read the antseed keeper knobs and CROSS-VALIDATE them.

    `settings.SCHEMA` validates each knob in isolation (type + range); the
    relations between them can only be checked where they are read together, and
    a broken relation is not academic:
      * trigger <= tourniquet means funding never reacts before routing is
        suppressed — the provider goes dark and stays dark;
      * amount > daily cap means every top-up is rejected by its own cap, so the
        keeper looks armed while doing nothing.
    Either is a silent no-op dressed as automation, so both raise instead."""
    k = Knobs(
        enabled=bool(int(settings.get("antseed.keeper_enabled"))),
        min_available_usdc=float(settings.get("antseed.min_available_usdc")),
        topup_trigger_usdc=float(settings.get("antseed.topup_trigger_usdc")),
        topup_amount_usdc=float(settings.get("antseed.topup_amount_usdc")),
        topup_wallet_floor_usdc=float(settings.get("antseed.topup_wallet_floor_usdc")),
        topup_daily_cap_usdc=float(settings.get("antseed.topup_daily_cap_usdc")),
        topup_cooldown_s=int(settings.get("antseed.topup_cooldown_s")),
        reclaim_min_usdc=float(settings.get("antseed.reclaim_min_usdc")),
    )
    if k.topup_trigger_usdc <= k.min_available_usdc:
        raise KnobError(
            f"antseed.topup_trigger_usdc ({k.topup_trigger_usdc}) must be ABOVE "
            f"antseed.min_available_usdc ({k.min_available_usdc}): funding has to "
            "react before the offer tourniquet suppresses the provider")
    if k.topup_amount_usdc > k.topup_daily_cap_usdc:
        raise KnobError(
            f"antseed.topup_amount_usdc ({k.topup_amount_usdc}) exceeds "
            f"antseed.topup_daily_cap_usdc ({k.topup_daily_cap_usdc}): every "
            "top-up would be rejected by its own cap")
    if k.topup_amount_usdc > MAX_TOPUP_USDC:
        raise KnobError(
            f"antseed.topup_amount_usdc ({k.topup_amount_usdc}) exceeds the hard "
            f"per-deposit limit ({MAX_TOPUP_USDC} USDC)")
    return k


def antseed_provider_ids(catalog: dict) -> list[str]:
    """The AntSeed buyer proxies in the loaded catalog — the same predicate
    `sources.antseed.AntSeedSource` uses, so keeper and source always agree on
    which providers exist."""
    return [pid for pid, p in (catalog.get("providers") or {}).items()
            if isinstance(p, dict) and p.get("discovery") == "marketplace"
            and str(p.get("discovery_id", "")).startswith("antseed")]


# Last cycle's decision per provider, for /x/runtime. Purely observational.
KEEPER_STATE: dict[str, Any] = {"enabled": False, "last_cycle": None,
                                "providers": {}, "error": None}


class WalletKeeper:
    """The autonomous funding loop for one catalog's AntSeed buyer proxies."""

    def __init__(self, provider_ids: list[str], source_name: str = "antseed"):
        self.provider_ids = list(provider_ids)
        self.source_name = source_name

    # ---- sidecar control plane ------------------------------------------

    @staticmethod
    def control_endpoint() -> "tuple[str | None, str | None]":
        url = (os.getenv("ANTSEED_CONTROL_URL") or "").rstrip("/")
        token = os.getenv("ANTSEED_CONTROL_TOKEN") or ""
        return (url, token) if (url and token) else (None, None)

    async def _control_post(self, op: str, body: "dict | None",
                            timeout: float) -> dict:
        """The raw HTTP call to the sidecar control server. Overridden wholesale
        in tests — the allowlist that protects it lives in `control()`, above this
        seam, so a test double cannot widen the set of reachable verbs."""
        url, token = self.control_endpoint()
        if not url:
            return {"ok": False, "error": "wallet control not configured"}
        import httpx
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{url}/{op}", json=body or {},
                                 headers={"x-antseed-control-token": token},
                                 timeout=timeout)
        except httpx.HTTPError as exc:
            return {"ok": False, "error": f"control unreachable: {exc}"}
        try:
            payload = r.json() or {}
        except Exception:  # noqa: BLE001 — a non-JSON body is still a failure
            payload = {}
        if r.status_code != 200:
            return {"ok": False, "error": str(payload.get("error")
                                              or (r.text or "")[:300])}
        payload.setdefault("ok", True)
        return payload

    async def control(self, op: str, body: "dict | None" = None,
                      timeout: float = 250.0) -> dict:
        """Call one ALLOWLISTED sidecar verb. An op outside the allowlist raises
        rather than returning an error: it is a programming mistake, not a runtime
        condition, and the one op that must never be reachable (`withdraw`) is
        worth failing loudly over."""
        if op in FORBIDDEN_CONTROL_OPS:
            raise ValueError(
                f"wallet keeper refuses to automate {op!r}: escrow->wallet "
                "withdrawal is human-initiated only")
        if op not in ALLOWED_CONTROL_OPS:
            raise ValueError(f"wallet keeper: {op!r} is not an allowlisted verb")
        return await self._control_post(op, body, timeout)

    # ---- untrusted reads (alerts and vetoes only) ------------------------

    def _chain_view(self, pid: str) -> dict:
        """The hot-wallet ETH/USDC balances the antseed source read from a PUBLIC
        Base RPC. Never a precondition for spending — see the module docstring."""
        import sources as _sources
        bal = ((_sources.SOURCE_STATE.get(self.source_name) or {})
               .get("balances") or {}).get(pid) or {}
        return bal.get("detail") or {}

    # ---- guardrail: the money-pump breaker -------------------------------

    def _settle_topups(self, pid: str, available: float) -> "str | None":
        """Close out previously fired deposits and trip the breaker if deposits
        stop working.

        A deposit is EFFECTIVE when `deposits_available` rose by at least
        PUMP_EFFECTIVE_FRACTION of the amount within one status cycle. Two
        consecutive ineffective deposits mean the money is going somewhere the
        router cannot spend it — a ratchet consuming every deposit into channel
        reserves, a misdirected wallet, a lying CLI — and the only safe response
        is to STOP and get a human, not to try a third time. The halt is persisted
        (`wallet_ops`), so it survives a restart and only an operator clears it.

        Known false positive, accepted deliberately: if a channel opens in the
        same window the deposit lands, ~1 USDC of the gain shows up as `reserved`
        rather than `available`. That is precisely the ratchet the breaker exists
        to catch, so it is scored as ineffective on purpose."""
        now = int(time.time())
        for row in host_store.wallet_ops_open(pid, "topup"):
            if row["outcome"] == "pending":
                # The process died between writing the intent and hearing back
                # from the control server: whether the transaction landed is
                # unknowable from here. Conservative in BOTH directions — counted
                # as spent for the cap and the cooldown (never re-fire on top of
                # a possible in-flight deposit), never counted as a pump strike
                # (there is no evidence either way).
                host_store.wallet_op_finish(
                    row["id"], "unknown", post_available=available,
                    detail="reconciled: keeper restarted before the outcome was known")
                _log.warning("wallet keeper: reconciled orphaned topup intent %s "
                             "for %s as UNKNOWN (counts against the daily cap)",
                             row["id"], pid)
                continue
            if now - int(row["updated_at"] or row["ts"]) < SETTLE_AFTER_S:
                continue                                  # no fresh reading yet
            pre, amount = row["pre_available"], row["amount_usdc"]
            if pre is None or not amount:
                host_store.wallet_op_finish(row["id"], "unknown",
                                            post_available=available,
                                            detail="no pre-deposit reading to compare")
                continue
            gained = available - float(pre)
            effective = gained >= PUMP_EFFECTIVE_FRACTION * float(amount)
            host_store.wallet_op_finish(
                row["id"], "effective" if effective else "ineffective",
                post_available=available,
                detail=f"available {pre} -> {available} (+{gained:.6f}) "
                       f"for a {amount} USDC deposit")
            if not effective:
                _log.warning("wallet keeper: topup %s on %s was INEFFECTIVE "
                             "(+%.6f of %s USDC reached deposits_available)",
                             row["id"], pid, gained, amount)

        settled = host_store.wallet_ops_settled(pid, "topup",
                                                limit=PUMP_STRIKES_TO_HALT)
        if (len(settled) >= PUMP_STRIKES_TO_HALT
                and all(r["outcome"] == "ineffective" for r in settled)):
            reason = (f"{PUMP_STRIKES_TO_HALT} consecutive deposits failed to lift "
                      f"deposits_available by {PUMP_EFFECTIVE_FRACTION:.0%} of the "
                      "amount — funding halted pending operator review")
            host_store.wallet_halt(pid, "topup", reason)
            _log.error("wallet keeper: HARD HALT on %s — %s", pid, reason)
            return "pump_halt"
        return None

    # ---- reclaim (always before top-up) ----------------------------------

    def _wedged(self, pid: str) -> bool:
        """Zero successful attempts in the last hour, out of a non-zero number of
        attempts. Both halves matter: a provider nobody called is IDLE, not
        wedged, and force-closing its channels would churn healthy capacity for
        nothing. `provider_attempt_counts` reports ok=-1 on a store error, which
        fails this test — no evidence, no force-close."""
        counts = host_store.provider_attempt_counts(
            pid, window_ms=NO_SUCCESS_WINDOW_S * 1000)
        return counts["total"] > 0 and counts["ok"] == 0

    async def _maybe_reclaim(self, pid: str, knobs: Knobs, available: float,
                             reserved: "float | None") -> str:
        """Unwind the escrow ratchet — recover USDC stuck in idle payment
        channels — but only when the provider is provably not using them."""
        if available >= knobs.topup_trigger_usdc:
            return "not_short"
        if reserved is None or reserved <= knobs.reclaim_min_usdc:
            return "nothing_reserved"
        if not self._wedged(pid):
            # Channels that are still serving traffic are working capital, not
            # stuck funds. Never churn them.
            return "not_wedged"
        eth = as_float(self._chain_view(pid).get("wallet_eth"))
        if eth is not None and eth < GAS_FLOOR_ETH:
            _log.error("wallet keeper: reclaim halted on %s — hot wallet has "
                       "%.6f ETH, below the %.4f gas floor. Top up gas.",
                       pid, eth, GAS_FLOOR_ETH)
            return "gas_floor"

        # A scan is a read-only RPC enumeration: no transaction, so no intent row
        # (the ledger is an audit trail of MONEY MOVEMENT, not of queries).
        scan = await self.control("reclaim/scan", timeout=95.0)
        if not scan.get("ok"):
            _log.warning("wallet keeper: reclaim scan failed on %s: %s",
                         pid, scan.get("error"))
            return "scan_failed"

        if not scan.get("operatorIsSelf"):
            # Bootstrap. Without a deposits operator every requestClose/withdraw
            # reverts NotAuthorized(); self-assignment is idempotent (reclaim.mjs
            # skips when already self) and moves no funds. One tx, then stop —
            # the next cycle re-scans against the new on-chain state.
            return await self._fire_reclaim_phase(
                pid, "reclaim/set-operator", "reclaim_set_operator",
                reason="bootstrap: buyer is not its own deposits operator",
                pre_available=available)

        channels = [c for c in (scan.get("channels") or []) if isinstance(c, dict)]

        def _worth_it(c: dict) -> bool:
            # Skip dust: closing a channel costs two transactions' gas, so a
            # channel holding less than MIN_CHANNEL_RECLAIMABLE is left alone.
            return (as_float(c.get("reclaimable")) or 0.0) >= MIN_CHANNEL_RECLAIMABLE

        withdrawable = [c for c in channels if c.get("closeRequested") and _worth_it(c)]
        closable = [c for c in channels
                    if not c.get("closeRequested") and _worth_it(c)]

        # Harvest first: a channel whose challenge window has elapsed returns its
        # funds NOW, whereas request-close only starts a ~15 min clock. One
        # tx-firing phase per cycle keeps the blast radius of a bad scan small.
        for phase, op, batch, why in (
                ("reclaim/withdraw", "reclaim_withdraw", withdrawable,
                 "challenge window elapsed"),
                ("reclaim/request-close", "reclaim_request_close", closable,
                 "start the challenge window on idle channels")):
            if not batch:
                continue
            if len(batch) > MAX_TX_PER_CYCLE:
                # The sidecar's reclaim phases act on EVERY eligible channel in
                # one invocation — there is no per-channel endpoint — so the only
                # way to honour a per-cycle transaction cap is to decline the
                # batch and say so loudly. An operator can still run it by hand.
                _log.error("wallet keeper: %s on %s would fire %d transactions, "
                           "over the %d/cycle cap — declining, run it manually",
                           phase, pid, len(batch), MAX_TX_PER_CYCLE)
                return "tx_cap"
            return await self._fire_reclaim_phase(
                pid, phase, op, reason=f"{why} ({len(batch)} channels)",
                pre_available=available)
        return "no_eligible_channels"

    async def _fire_reclaim_phase(self, pid: str, phase: str, op: str,
                                  reason: str, pre_available: float) -> str:
        """Write the intent, then fire one on-chain reclaim phase. The intent row
        comes FIRST and a failure to persist it aborts the call: an on-chain
        action with no audit row is worse than a missed reclaim."""
        op_id = host_store.wallet_op_begin(pid, op, reason=reason,
                                           pre_available=pre_available)
        if op_id is None:
            _log.error("wallet keeper: cannot persist the %s intent for %s — "
                       "refusing to fire an unlogged transaction", op, pid)
            return "ledger_unavailable"
        try:
            resp = await self.control(phase)
        except asyncio.CancelledError:
            # Shutdown mid-flight: the transaction may or may not have been sent,
            # and there is no later pass that reconciles reclaim rows, so say so
            # in the ledger rather than leaving a permanently `pending` row.
            host_store.wallet_op_finish(op_id, "unknown",
                                        detail="cancelled before the outcome was known")
            raise
        except Exception as exc:  # noqa: BLE001 — the row must never dangle
            host_store.wallet_op_finish(op_id, "unknown",
                                        detail=f"{type(exc).__name__}: {exc}")
            raise
        ok = bool(resp.get("ok"))
        host_store.wallet_op_finish(
            op_id, "ok" if ok else "failed",
            detail=str(resp if ok else resp.get("error"))[:2000])
        if not ok:
            _log.warning("wallet keeper: %s failed on %s: %s",
                         op, pid, resp.get("error"))
        return op if ok else f"{op}_failed"

    # ---- top-up ----------------------------------------------------------

    async def _maybe_topup(self, pid: str, knobs: Knobs, available: float) -> str:
        """Deposit wallet -> escrow, subject to every guardrail. Ordered cheapest
        check first; every one of them is a REFUSAL to spend, never a licence."""
        if available >= knobs.topup_trigger_usdc:
            return "funded"
        if host_store.wallet_halted(pid, "topup"):
            # Sticky, persisted, and NOT self-clearing — a breaker that re-arms
            # itself is not a breaker. `host_store.wallet_halted` also reports
            # True on a store error, so an unreadable ledger stops spending.
            return "halted"

        now = int(time.time())
        spend = host_store.wallet_op_spend_since(pid, "topup", now - 86400)
        last_ts = spend.get("last_ts")
        if last_ts is not None and now - int(last_ts) < knobs.topup_cooldown_s:
            return "cooldown"
        # Full-amount-or-nothing: a partial deposit that squeezes under the
        # remaining cap is usually below one channel reserve, i.e. dust that only
        # burns gas. Wait for the 24h window to roll instead.
        if spend["spent_usdc"] + knobs.topup_amount_usdc > knobs.topup_daily_cap_usdc:
            _log.warning("wallet keeper: %s daily cap reached (%.4f of %.4f USDC "
                         "in 24h) — no top-up", pid, spend["spent_usdc"],
                         knobs.topup_daily_cap_usdc)
            return "daily_cap"

        amount = min(knobs.topup_amount_usdc, MAX_TOPUP_USDC)
        if amount < MIN_TOPUP_USDC:
            return "amount_too_small"

        # VETO-ONLY hot-wallet floor. The balance behind it comes from an
        # untrusted public RPC, so it may block a deposit but never satisfies a
        # precondition for one: an unknown balance proceeds (the CLI is the
        # authority, a failed deposit is recorded as failed, and two ineffective
        # deposits trip the breaker).
        wallet_usdc = as_float(self._chain_view(pid).get("wallet_usdc"))
        if wallet_usdc is not None and \
                wallet_usdc - amount < knobs.topup_wallet_floor_usdc:
            _log.warning("wallet keeper: %s top-up skipped — hot wallet %.6f USDC "
                         "would fall below the %.4f floor after a %.4f deposit",
                         pid, wallet_usdc, knobs.topup_wallet_floor_usdc, amount)
            return "wallet_floor"

        op_id = host_store.wallet_op_begin(
            pid, "topup", amount_usdc=amount,
            reason=f"deposits_available {available} < trigger "
                   f"{knobs.topup_trigger_usdc}",
            pre_available=available)
        if op_id is None:
            _log.error("wallet keeper: cannot persist the topup intent for %s — "
                       "refusing to fire an unlogged deposit", pid)
            return "ledger_unavailable"
        # Amounts go to the CLI as a plain decimal string (control.js validates
        # the shape AND re-caps the value server-side).
        resp = await self.control("deposit", {"amount": f"{amount:.6f}"},
                                  timeout=130.0)
        if resp.get("ok"):
            host_store.wallet_op_finish(op_id, "fired",
                                        detail=str(resp.get("stdout") or "")[:2000])
            _log.info("wallet keeper: deposited %.4f USDC into %s escrow "
                      "(available was %s)", amount, pid, available)
            return "topup_fired"
        host_store.wallet_op_finish(op_id, "failed",
                                    detail=str(resp.get("error"))[:2000])
        _log.warning("wallet keeper: deposit failed on %s: %s", pid, resp.get("error"))
        return "topup_failed"

    # ---- one cycle -------------------------------------------------------

    async def cycle_provider(self, pid: str, knobs: Knobs) -> dict:
        status = host_store.buyer_status(pid)
        available = as_float((status or {}).get("deposits_available"))
        reserved = as_float((status or {}).get("deposits_reserved"))
        if available is None:
            # FAIL CLOSED. Unlike the offer tourniquet — which fails OPEN so a
            # transient read error cannot black out routing — every action here
            # spends money, and money must never move on a state we cannot read.
            return {"decision": "status_unreadable"}
        out: dict[str, Any] = {"available": available, "reserved": reserved}
        halted = self._settle_topups(pid, available)
        out["reclaim"] = await self._maybe_reclaim(pid, knobs, available, reserved)
        out["topup"] = "pump_halt" if halted else await self._maybe_topup(
            pid, knobs, available)
        out["decision"] = "acted"
        return out

    async def cycle(self) -> dict:
        """One full pass. Never raises: a keeper that crashes the app is worse
        than a keeper that misses a tick."""
        state: dict[str, Any] = {"last_cycle": int(time.time()), "providers": {},
                                 "error": None}
        try:
            knobs = load_knobs()
        except KnobError as exc:
            # A contradictory knob set is an operator error, not a transient one:
            # say so every cycle rather than silently doing nothing.
            _log.error("wallet keeper stood down: %s", exc)
            KEEPER_STATE.update({**state, "enabled": False, "error": str(exc)})
            return KEEPER_STATE
        state["enabled"] = knobs.enabled
        if not knobs.enabled:
            KEEPER_STATE.update(state)
            return KEEPER_STATE
        url, _token = self.control_endpoint()
        if not url:
            state["error"] = "ANTSEED_CONTROL_URL/TOKEN not configured"
            KEEPER_STATE.update(state)
            return KEEPER_STATE
        for pid in self.provider_ids:
            try:
                state["providers"][pid] = await self.cycle_provider(pid, knobs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — isolation is the contract
                _log.exception("wallet keeper cycle failed for %s", pid)
                state["providers"][pid] = {"decision": "error",
                                           "error": f"{type(exc).__name__}: {exc}"}
        KEEPER_STATE.update(state)
        return KEEPER_STATE

    async def run(self) -> None:
        while True:
            try:
                await self.cycle()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — never let the loop die
                _log.exception("wallet keeper cycle raised")
            await asyncio.sleep(CYCLE_S)


def start(catalog: dict) -> "asyncio.Task | None":
    """Start the keeper loop for this catalog's AntSeed proxies, or None when the
    catalog has none. Call from an async context (the app lifespan). The loop
    itself re-reads `antseed.keeper_enabled` every cycle, so the kill switch works
    without a restart."""
    pids = antseed_provider_ids(catalog)
    if not pids:
        return None
    return asyncio.create_task(WalletKeeper(pids).run())
