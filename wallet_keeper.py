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
                          60s. This is the ONLY trusted balance signal (below),
                          and it is rejected once it goes STALE: a row that
                          stopped being written is not evidence of anything.
  2. settle             — measure the effect of the previous deposit; trip the
                          money-pump breaker if deposits stop landing, and the
                          error breaker if they stop being answerable at all.
  3. RECLAIM            — before top-up, always: topping up a ratchet without
                          first unwinding it just refills the leak.
  4. top-up             — deposit wallet -> escrow, under cap + cooldown + floor.

TRUST BOUNDARY. Decisions are made ONLY from the CLI-reported `buyer_status`.
`wallet_usdc` / `wallet_eth` come from a public Base RPC (`sources/antseed.py`
defaults to https://mainnet.base.org, an untrusted free endpoint) and are used
VETO-ONLY: a reading can block an action, it can never authorize one. An attacker
who controls that endpoint can therefore stall the keeper, but never induce a
spend — which requires that an ABSENT reading also blocks, or the attacker just
takes the endpoint down and the floor disappears with it. So an unreadable or
stale `wallet_usdc` VETOES a deposit (`CHAIN_READ_MAX_AGE_S`).

The gas floor is the deliberate exception and runs the other way: an unreadable
`wallet_eth` does NOT block a reclaim. The asymmetry is the point — failing
closed on an action that moves money OUT of the wallet is safe, but failing
closed on the one action that moves money BACK IN would let an RPC outage strand
the escrow permanently.

TIMEOUTS COMPOSE OR THEY LIE. A client timeout below the server's worst case is
not a timeout: the sidecar goes on executing a request the keeper has already
written off, and the CLI spends while the ledger records nothing. So the budgets
below are DERIVED from `antseed/control.js`'s own published budgets rather than
guessed, and every outcome the keeper cannot rule out is recorded as `unknown`
and COUNTED AS SPENT.

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
from sources.antseed import STALE_AFTER_S, as_float

_log = logging.getLogger("unhardcoded.wallet_keeper")

# How often the loop wakes. Matched to the sidecar's 60s buyer_status write: a
# faster cycle would just re-read the same row, and every gate that matters
# (cooldown, daily cap) is time-based and derived from the durable ledger anyway.
CYCLE_S = 60

# A deposit's effect is judged no earlier than this after it fired. control.js
# re-writes buyer_status synchronously after a successful `buyer deposit`, so one
# cycle is already enough; the margin absorbs a slow status write.
SETTLE_AFTER_S = 90

# How old `buyer_status` may be and still be acted on. The sidecar rewrites it
# every 60s, so this is 15 missed writes — by then the sidecar is gone, and the
# escrow the row reports has had 15 minutes of settlements against it. Shared
# with the market book's window (sources.antseed.STALE_AFTER_S) because both
# answer the same question: is the sidecar still telling us things?
STATUS_MAX_AGE_S = STALE_AFTER_S

# How old the untrusted chain reading may be and still veto. `AntSeedSource`
# polls every 300s, so this is three missed polls. Beyond it the reading is not
# a weak signal, it is a DIFFERENT wallet state, and treating it as current is
# how a six-hour-old balance authorizes a deposit the floor would have blocked.
CHAIN_READ_MAX_AGE_S = 900

# --- hard limits. Deliberately NOT operator knobs: these bound the blast radius
# of a misconfigured or hostile knob, so they must not be reachable from the same
# surface that sets the knobs (the dashboard Config tab).
MAX_TOPUP_USDC = 50.0          # mirrored server-side in antseed/control.js
MIN_TOPUP_USDC = 0.01          # below this a deposit is dust that only burns gas
GAS_FLOOR_ETH = 0.0005         # reclaim needs gas; below this, do not try
MAX_TX_PER_CYCLE = 8           # on-chain transactions one cycle may cause
MIN_CHANNEL_RECLAIMABLE = 0.05  # a channel holding less is not worth a close tx
NO_SUCCESS_WINDOW_S = 3600     # "fully wedged" = zero successes over this window
PUMP_EFFECTIVE_FRACTION = 0.8  # a deposit must lift the escrow by >= this × amount
PUMP_STRIKES_TO_HALT = 2       # consecutive ineffective deposits before a hard halt

# `route_observations` is written through a lossy queue that DROPS rows when full
# (host_store._OBS_Q), so a small sample is not evidence of anything — least of
# all evidence for force-closing payment channels. Below this many attempts the
# wedge test abstains.
WEDGE_MIN_ATTEMPTS = 10

# The ERROR breaker, the twin of the money-pump breaker above. `ineffective` is a
# MEASURED failure (the deposit landed somewhere unspendable); `failed`/`unknown`
# are unmeasurable ones (the sidecar never answered, or answered in a way that
# cannot rule out a broadcast transaction). They used to escape every guardrail,
# so a permanently failing deposit re-fired every 60s forever. Weaker evidence
# than a measured miss, hence one more strike before the same hard halt.
TOPUP_ERROR_STRIKES_TO_HALT = 3
# Retry backoff between error strikes. The floor exists because the cooldown knob
# can legitimately be 0 (an operator wanting prompt refunding), and 0 × any
# backoff is still 0 — which is the hammering this exists to stop.
TOPUP_BACKOFF_BASE_S = 300
TOPUP_BACKOFF_CAP_S = 3600

# Reclaim's own rate limit. It fires REAL transactions and had none: at CYCLE_S
# it re-ran request-close every 60s, and re-fired set-operator every cycle until
# the assignment confirmed (reclaim.mjs reads confirmed state only). One
# challenge window is the natural spacing — nothing it starts can finish sooner.
RECLAIM_COOLDOWN_S = 900
RECLAIM_ERROR_STRIKES_TO_HALT = 3

# Reclaim is not only for a SHORT escrow. Once a top-up lifts `available` above
# the trigger, a shortness-only test locks reclaim out forever and the keeper
# becomes a money pump in one direction — which is exactly the prod state this
# module was written for ($0.23 spendable, $15.63 stranded in channels). A
# reserve this many times the spendable escrow is a ratchet worth unwinding
# whatever the balance says.
RATCHET_RESERVED_RATIO = 2.0

# --- client budgets, DERIVED from antseed/control.js's own worst case ---------
# Every one of these must strictly EXCEED the server's budget for the same
# endpoint, or the timeout is a lie: the sidecar keeps executing, the CLI spends,
# and the keeper records a request it believes never happened. control.js
# publishes its budgets on /budgets and the numbers below mirror them; the
# margin absorbs connection setup and the response write.
CONTROL_SLACK_S = 20.0
CONTROL_QUEUE_WAIT_S = 30.0     # control.js QUEUE_WAIT_BUDGET_MS
CONTROL_DB_S = 10.0             # control.js DB_TIMEOUT_MS
CONTROL_STATUS_S = 30.0         # control.js STATUS_TIMEOUT_MS
CONTROL_DEPOSIT_S = 120.0       # control.js DEPOSIT_TIMEOUT_MS
CONTROL_RECLAIM_SCAN_S = 90.0   # control.js RECLAIM_SCAN_TIMEOUT_MS
CONTROL_RECLAIM_TX_S = 240.0    # control.js RECLAIM_TX_TIMEOUT_MS

DEPOSIT_TIMEOUT_S = (CONTROL_QUEUE_WAIT_S + CONTROL_DEPOSIT_S + CONTROL_STATUS_S
                     + CONTROL_DB_S + CONTROL_SLACK_S)          # 210s
RECLAIM_SCAN_TIMEOUT_S = CONTROL_RECLAIM_SCAN_S + CONTROL_SLACK_S   # 110s
RECLAIM_TX_TIMEOUT_S = (CONTROL_QUEUE_WAIT_S + CONTROL_RECLAIM_TX_S
                        + CONTROL_STATUS_S + CONTROL_DB_S + CONTROL_SLACK_S)  # 330s

# HTTP statuses from the control server that PROVE the buyer CLI never ran, so
# the op cost nothing and may be retried freely. Everything else — a read
# timeout, a reset, a 502 from a CLI that exited non-zero, a 504 from a CLI we
# killed mid-broadcast — is inconclusive and must be recorded as `unknown`.
# control.js states this per-branch via `attempted`; the status list is the
# fallback for a response that predates it or comes from something in between.
NOT_ATTEMPTED_STATUSES = frozenset({400, 401, 404, 405, 429})

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
        seam, so a test double cannot widen the set of reachable verbs.

        Every return carries `attempted`, and it is the field that decides
        whether real money may have moved. FALSE means the buyer CLI provably
        never ran (no endpoint, a malformed URL, a 400 from the amount
        validator, a 429 from the sidecar's queue gate) — the op cost nothing.
        TRUE means the request reached the wire and the outcome is unknowable
        from here: a read timeout, a reset connection, a 502 from a CLI that
        exited non-zero, a 504 from a CLI killed mid-broadcast. Callers must
        record the second kind as `unknown` and count it as SPENT.

        The default on any unrecognised failure is TRUE. Being wrong in that
        direction burns a slot of the daily cap; being wrong the other way moves
        USDC on Base mainnet with the ledger recording nothing."""
        url, token = self.control_endpoint()
        if not url:
            return {"ok": False, "attempted": False,
                    "error": "wallet control not configured"}
        try:
            import httpx
        except ImportError as exc:                  # nothing left the process
            return {"ok": False, "attempted": False,
                    "error": f"httpx unavailable: {exc}"}
        try:
            async with httpx.AsyncClient() as c:
                r = await c.post(f"{url}/{op}", json=body or {},
                                 headers={"x-antseed-control-token": token},
                                 timeout=timeout)
        except (httpx.InvalidURL, httpx.UnsupportedProtocol) as exc:
            # Raised while BUILDING the request — and not subclasses of
            # HTTPError, so they used to escape this function entirely and leave
            # the caller's intent row dangling as `pending`.
            return {"ok": False, "attempted": False,
                    "error": f"control endpoint misconfigured: {exc}"}
        except Exception as exc:  # noqa: BLE001 — timeout, reset, DNS, TLS, ...
            return {"ok": False, "attempted": True,
                    "error": f"control unreachable ({type(exc).__name__}): {exc}"}
        try:
            payload = r.json() or {}
        except Exception:  # noqa: BLE001 — a non-JSON body is still a failure
            payload = {}
        if r.status_code != 200:
            attempted = payload.get("attempted")
            if not isinstance(attempted, bool):
                attempted = r.status_code not in NOT_ATTEMPTED_STATUSES
            return {"ok": False, "attempted": attempted,
                    "status": r.status_code,
                    "error": str(payload.get("error") or (r.text or "")[:300])}
        payload.setdefault("ok", True)
        payload.setdefault("attempted", True)
        return payload

    async def control(self, op: str, body: "dict | None" = None,
                      timeout: float = DEPOSIT_TIMEOUT_S) -> dict:
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

    @staticmethod
    def _outcome_for(resp: dict) -> str:
        """`fired` / `unknown` / `failed` for a control response that is not ok.

        The whole point of C1: only a response that PROVES nothing was attempted
        is `failed`, because `failed` consumes neither the daily cap nor the
        cooldown. Anything else is `unknown`, which does."""
        if resp.get("ok"):
            return "fired"
        return "failed" if resp.get("attempted") is False else "unknown"

    # ---- untrusted reads (alerts and vetoes only) ------------------------

    def _chain_view(self, pid: str) -> dict:
        """The hot-wallet ETH/USDC balances the antseed source read from a PUBLIC
        Base RPC. Never a precondition for spending — see the module docstring.

        AGE-BOUNDED. The reading carries the tick that produced it, and a reading
        older than CHAIN_READ_MAX_AGE_S is discarded rather than reused: a stale
        balance is not a weaker signal, it is a different wallet state, and a
        veto that consults a six-hour-old number is not a veto. An empty dict
        here therefore means "no usable reading", which the deposit path treats
        as a refusal (and the reclaim gas floor deliberately does not)."""
        import sources as _sources
        bal = ((_sources.SOURCE_STATE.get(self.source_name) or {})
               .get("balances") or {}).get(pid) or {}
        fetched_at = bal.get("fetched_at")
        if not isinstance(fetched_at, (int, float)) or isinstance(fetched_at, bool):
            return {}
        if time.time() - float(fetched_at) > CHAIN_READ_MAX_AGE_S:
            _log.warning("wallet keeper: chain reading for %s is %.0fs old "
                         "(max %ds) — discarding it",
                         pid, time.time() - float(fetched_at), CHAIN_READ_MAX_AGE_S)
            return {}
        return bal.get("detail") or {}

    # ---- strike counters (shared by both breakers) -----------------------

    @staticmethod
    def _consecutive(rows: list[dict], outcomes: "tuple[str, ...]") -> int:
        """How many of the NEWEST closed-out ops share one of `outcomes`. Counts
        from the newest backwards and stops at the first row that does not — one
        success in between means the run is broken, which is the whole point of a
        consecutive-failure breaker."""
        n = 0
        for row in rows:
            if row["outcome"] not in outcomes:
                break
            n += 1
        return n

    def _error_strikes(self, pid: str, op: str, limit: int) -> int:
        return self._consecutive(
            host_store.wallet_ops_terminal(pid, op, limit=limit),
            host_store.WALLET_OP_ERROR_OUTCOMES)

    # ---- guardrail: the money-pump breaker -------------------------------

    def _reconcile_orphans(self, pid: str) -> None:
        """Close out `pending` rows for the ops that have no settlement pass of
        their own (everything except `topup`). A reclaim intent whose control
        call died leaves a row that nothing would ever revisit, so the audit
        trail grew permanently-`pending` entries that no reader could interpret
        and the reclaim breaker below could not count."""
        for row in host_store.wallet_ops_open(pid):
            if row["op"] == "topup" or row["outcome"] != "pending":
                continue
            host_store.wallet_op_finish(
                row["id"], "unknown",
                detail="reconciled: keeper restarted before the outcome was known")
            _log.warning("wallet keeper: reconciled orphaned %s intent %s for %s "
                         "as UNKNOWN", row["op"], row["id"], pid)

    def _settle_topups(self, pid: str, available: float,
                       reserved: "float | None") -> "str | None":
        """Close out previously fired deposits and trip either breaker if deposits
        stop working.

        A deposit is EFFECTIVE when the ESCROW — `deposits_available` plus
        `deposits_reserved` — rose by at least PUMP_EFFECTIVE_FRACTION of the
        amount within one status cycle. Two consecutive ineffective deposits mean
        the money is going somewhere the escrow never sees — a misdirected
        wallet, a lying CLI — and the only safe response is to STOP and get a
        human, not to try a third time. The halt is persisted (`wallet_ops`), so
        it survives a restart and only an operator clears it.

        WHY THE SUM, not `available` alone. Scoring on the spendable half made a
        deposit look ineffective whenever a channel opened in the same ~90s
        window — the channel RESERVES ~1 USDC, so the money arrived and simply
        moved one column right. Under load that is the normal case, and two busy
        cycles would hard-halt a perfectly healthy system; symmetrically, a
        reclaim landing in the window inflated `available` and masked a deposit
        that genuinely never arrived. Both errors came from measuring one column
        of a two-column ledger. The ratchet the old comment claimed to be
        catching here is caught properly by the reclaim path instead, which acts
        on the reserve directly rather than inferring it from a deposit's shadow.

        The second breaker counts deposits whose effect could never be measured
        at all (`failed` / `unknown`). Those bypassed the money-pump breaker
        entirely, so a deposit that failed every single time re-fired every 60s
        forever with no backoff and no strike count anywhere."""
        now = int(time.time())
        for row in host_store.wallet_ops_open(pid, "topup"):
            if row["outcome"] == "pending":
                # The process died between writing the intent and hearing back
                # from the control server: whether the transaction landed is
                # unknowable from here. Counted as spent for the cap and the
                # cooldown (never re-fire on top of a possible in-flight
                # deposit), and — since C1 — counted toward the ERROR breaker,
                # which is the only guardrail that can see a deposit nobody was
                # ever able to measure.
                host_store.wallet_op_finish(
                    row["id"], "unknown", post_available=available,
                    post_reserved=reserved,
                    detail="reconciled: keeper restarted before the outcome was known")
                _log.warning("wallet keeper: reconciled orphaned topup intent %s "
                             "for %s as UNKNOWN (counts against the daily cap)",
                             row["id"], pid)
                continue
            if now - int(row["updated_at"] or row["ts"]) < SETTLE_AFTER_S:
                continue                                  # no fresh reading yet
            pre, amount = row["pre_available"], row["amount_usdc"]
            pre_reserved = row["pre_reserved"]
            if pre is None or not amount:
                host_store.wallet_op_finish(row["id"], "unknown",
                                            post_available=available,
                                            post_reserved=reserved,
                                            detail="no pre-deposit reading to compare")
                continue
            # Fall back to the spendable half only when a reserved reading is
            # missing on either end (a row written before the columns existed, or
            # a status the buyer reported without them).
            if pre_reserved is not None and reserved is not None:
                gained = (available + reserved) - (float(pre) + float(pre_reserved))
                basis = (f"escrow {float(pre) + float(pre_reserved):.6f} -> "
                         f"{available + reserved:.6f}")
            else:
                gained = available - float(pre)
                basis = f"available {pre} -> {available} (no reserved reading)"
            effective = gained >= PUMP_EFFECTIVE_FRACTION * float(amount)
            host_store.wallet_op_finish(
                row["id"], "effective" if effective else "ineffective",
                post_available=available, post_reserved=reserved,
                detail=f"{basis} (+{gained:.6f}) for a {amount} USDC deposit")
            if not effective:
                _log.warning("wallet keeper: topup %s on %s was INEFFECTIVE "
                             "(+%.6f of %s USDC reached the escrow)",
                             row["id"], pid, gained, amount)

        settled = host_store.wallet_ops_settled(pid, "topup",
                                                limit=PUMP_STRIKES_TO_HALT)
        if (len(settled) >= PUMP_STRIKES_TO_HALT
                and all(r["outcome"] == "ineffective" for r in settled)):
            return self._halt_topups(pid, "pump_halt",
                f"{PUMP_STRIKES_TO_HALT} consecutive deposits failed to lift the "
                f"escrow by {PUMP_EFFECTIVE_FRACTION:.0%} of the amount")

        strikes = self._error_strikes(pid, "topup", TOPUP_ERROR_STRIKES_TO_HALT)
        if strikes >= TOPUP_ERROR_STRIKES_TO_HALT:
            return self._halt_topups(pid, "error_halt",
                f"{strikes} consecutive deposits could not be completed or "
                "confirmed (failed/unknown) — each one may have moved USDC")
        return None

    @staticmethod
    def _halt_topups(pid: str, decision: str, why: str) -> str:
        reason = f"{why} — funding halted pending operator review"
        if host_store.wallet_halt(pid, "topup", reason):
            _log.error("wallet keeper: HARD HALT on %s — %s", pid, reason)
        else:
            # `wallet_halt` reports False only when it could not PERSIST. Saying
            # "HARD HALT" anyway would describe a breaker that evaporates on the
            # next restart. The cycle still stands down: the cap and cooldown
            # readers fail closed on the same broken store.
            _log.error("wallet keeper: %s on %s could NOT be persisted (%s) — "
                       "standing down this cycle, but the halt is NOT durable; "
                       "fix the store", decision, pid, reason)
        return decision

    # ---- reclaim (always before top-up) ----------------------------------

    def _wedged(self, pid: str, knobs: Knobs, available: float) -> bool:
        """Is this provider provably not using its payment channels?

        Two independent proofs, either of which is enough:

        1. THE TOURNIQUET IS ARMED. `available` is below the offer gate in
           sources/antseed.py, so that gate is suppressing every antseed offer
           right now — no offers means no attempts, by construction. This is a
           STRONGER signal than a failed attempt, not a weaker one, and leaving
           it out is what made reclaim structurally unreachable in exactly the
           state it was written for: the tourniquet suppressed all traffic, so
           the attempt-based test below saw `total == 0`, read it as "idle", and
           returned `not_wedged` forever while $15.63 sat stranded in channels.

        2. ZERO SUCCESSES DESPITE TRAFFIC. At least WEDGE_MIN_ATTEMPTS attempts
           in the last hour and not one succeeded. Both halves matter: a provider
           nobody called is IDLE, not wedged, and force-closing its channels
           would churn healthy capacity for nothing. The minimum sample exists
           because `route_observations` is written through a lossy queue that
           drops rows under load — a handful of rows is not evidence for
           force-closing anything.

        `provider_attempt_counts` reports ok=-1 on a store error, which fails
        this test — no evidence, no force-close."""
        if available < knobs.min_available_usdc:
            return True
        counts = host_store.provider_attempt_counts(
            pid, window_ms=NO_SUCCESS_WINDOW_S * 1000)
        return counts["total"] >= WEDGE_MIN_ATTEMPTS and counts["ok"] == 0

    async def _maybe_reclaim(self, pid: str, knobs: Knobs, available: float,
                             reserved: "float | None") -> str:
        """Unwind the escrow ratchet — recover USDC stuck in idle payment
        channels — but only when the provider is provably not using them."""
        if host_store.wallet_halted(pid, "reclaim"):
            # Reclaim's own breaker. Deliberately NOT tied to the top-up halt:
            # a top-up halt means "stop putting money IN", and reclaim moves
            # money the other way — it is the remedy for a ratchet, not another
            # symptom of one. Stopping it there would strand the funds a halt
            # exists to protect.
            return "halted"
        # SHORT, or RATCHETED. A shortness-only test locks reclaim out the moment
        # a top-up lifts `available` past the trigger — permanently, since escrow
        # only comes back down by spending — so the keeper would pump money in
        # and never pull any back. A reserve that dwarfs the spendable escrow is
        # worth unwinding whatever the balance says.
        short = available < knobs.topup_trigger_usdc
        ratcheted = (reserved is not None and reserved >= RATCHET_RESERVED_RATIO
                     * max(available, knobs.topup_trigger_usdc))
        if not short and not ratcheted:
            return "not_short"
        if reserved is None or reserved <= knobs.reclaim_min_usdc:
            return "nothing_reserved"
        if not self._wedged(pid, knobs, available):
            # Channels that are still serving traffic are working capital, not
            # stuck funds. Never churn them.
            return "not_wedged"
        eth = as_float(self._chain_view(pid).get("wallet_eth"))
        if eth is not None and eth < GAS_FLOOR_ETH:
            _log.error("wallet keeper: reclaim halted on %s — hot wallet has "
                       "%.6f ETH, below the %.4f gas floor. Top up gas.",
                       pid, eth, GAS_FLOOR_ETH)
            return "gas_floor"
        # An UNREADABLE gas balance does not block, unlike the deposit path's
        # USDC floor. Failing closed on the one action that recovers money would
        # let an RPC outage strand the escrow permanently; a reclaim fired
        # without gas simply does not confirm.

        # A scan is a read-only RPC enumeration: no transaction, so no intent row
        # (the ledger is an audit trail of MONEY MOVEMENT, not of queries).
        scan = await self.control("reclaim/scan", timeout=RECLAIM_SCAN_TIMEOUT_S)
        if not scan.get("ok"):
            _log.warning("wallet keeper: reclaim scan failed on %s: %s",
                         pid, scan.get("error"))
            return "scan_failed"

        # Reclaim's cooldown, applied AFTER the read-only scan so /x/runtime
        # still shows current channel state every cycle. It fires real
        # transactions and had no rate limit at all: request-close re-ran every
        # 60s, and set-operator re-fired every cycle until the assignment
        # confirmed, because reclaim.mjs's getOperator reads confirmed state
        # only. One challenge window is the natural spacing — nothing reclaim
        # starts can finish sooner than that anyway.
        now = int(time.time())
        last = self._last_reclaim_ts(pid)
        if last is not None and now - last < RECLAIM_COOLDOWN_S:
            return "cooldown"

        if not scan.get("operatorIsSelf"):
            # Bootstrap. Without a deposits operator every requestClose/withdraw
            # reverts NotAuthorized(); self-assignment is idempotent (reclaim.mjs
            # skips when already self) and moves no funds. One tx, then stop —
            # the next cycle re-scans against the new on-chain state.
            return await self._fire_reclaim_phase(
                pid, "reclaim/set-operator", "reclaim_set_operator",
                reason="bootstrap: buyer is not its own deposits operator",
                pre_available=available, pre_reserved=reserved)

        channels = [c for c in (scan.get("channels") or []) if isinstance(c, dict)]

        def _worth_it(c: dict) -> bool:
            # Skip dust: closing a channel costs two transactions' gas, so a
            # channel holding less than MIN_CHANNEL_RECLAIMABLE is left alone.
            return (as_float(c.get("reclaimable")) or 0.0) >= MIN_CHANNEL_RECLAIMABLE

        def _ids(batch: list[dict]) -> list[str]:
            return [str(c["id"]) for c in batch if c.get("id")]

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
            # The cap now BINDS: the sidecar is told which channels to act on
            # (reclaim.mjs takes an id list), so taking the first
            # MAX_TX_PER_CYCLE fires exactly that many transactions. It used to
            # decline the whole batch instead, because the sidecar acted on every
            # eligible channel whatever the keeper decided — which made a large
            # channel set permanently unreclaimable by automation, the same
            # dead end as the shortness test above. Reclaimable-first so the
            # money that matters most comes back first; the rest follow next
            # cycle.
            batch = sorted(batch, key=lambda c: -(as_float(c.get("reclaimable")) or 0.0))
            ids = _ids(batch)
            if len(ids) != len(batch):
                # A scan row with no id cannot be named, and an unnamed batch
                # would fall back to the sidecar's act-on-everything path.
                _log.error("wallet keeper: %s on %s — the scan returned a channel "
                           "with no id; declining rather than firing an unbounded "
                           "batch", phase, pid)
                return "scan_malformed"
            capped, over = ids[:MAX_TX_PER_CYCLE], max(0, len(ids) - MAX_TX_PER_CYCLE)
            if over:
                _log.warning("wallet keeper: %s on %s has %d eligible channels; "
                             "acting on %d this cycle (%d/cycle cap), %d to follow",
                             phase, pid, len(ids), len(capped), MAX_TX_PER_CYCLE, over)
            return await self._fire_reclaim_phase(
                pid, phase, op,
                reason=f"{why} ({len(capped)} of {len(channels)} channels)",
                pre_available=available, pre_reserved=reserved, ids=capped)
        return "no_eligible_channels"

    RECLAIM_OPS = ("reclaim_set_operator", "reclaim_request_close",
                   "reclaim_withdraw")

    @classmethod
    def _last_reclaim_ts(cls, pid: str) -> "int | None":
        """When any reclaim phase last FIRED, from the durable ledger (so a pod
        restart does not reset the cooldown). None when none ever has."""
        return host_store.wallet_ops_last_ts(pid, cls.RECLAIM_OPS)

    async def _fire_reclaim_phase(self, pid: str, phase: str, op: str,
                                  reason: str, pre_available: float,
                                  pre_reserved: "float | None" = None,
                                  ids: "list[str] | None" = None) -> str:
        """Write the intent, then fire one on-chain reclaim phase on the NAMED
        channels. The intent row comes FIRST and a failure to persist it aborts
        the call: an on-chain action with no audit row is worse than a missed
        reclaim."""
        if ids is not None and not ids:
            # An EMPTY selection must NEVER reach the wire. The sidecar reads
            # "no ids" as "act on every eligible channel" (the human-by-hand
            # path), so sending one would widen the batch to unbounded — the
            # exact opposite of what the caller decided. Unreachable from
            # `_maybe_reclaim`, which only calls this with a non-empty batch;
            # here so that it stays unreachable.
            _log.error("wallet keeper: refusing to fire %s on %s with an EMPTY "
                       "channel selection — that would act on every channel",
                       op, pid)
            return "empty_selection"
        op_id = host_store.wallet_op_begin(pid, op, reason=reason,
                                           pre_available=pre_available,
                                           pre_reserved=pre_reserved)
        if op_id is None:
            _log.error("wallet keeper: cannot persist the %s intent for %s — "
                       "refusing to fire an unlogged transaction", op, pid)
            return "ledger_unavailable"
        try:
            body = {"ids": ids} if ids is not None else None
            resp = await self.control(phase, body, timeout=RECLAIM_TX_TIMEOUT_S)
        except asyncio.CancelledError:
            # Shutdown mid-flight: the transaction may or may not have been sent,
            # so say so in the ledger rather than leaving a permanently `pending`
            # row (`_reconcile_orphans` is the backstop when even this misses).
            host_store.wallet_op_finish(op_id, "unknown",
                                        detail="cancelled before the outcome was known")
            raise
        except Exception as exc:  # noqa: BLE001 — the row must never dangle
            host_store.wallet_op_finish(op_id, "unknown",
                                        detail=f"{type(exc).__name__}: {exc}")
            raise
        outcome = self._outcome_for(resp)
        host_store.wallet_op_finish(
            op_id, "ok" if outcome == "fired" else outcome,
            detail=str(resp if outcome == "fired" else resp.get("error"))[:2000])
        if outcome != "fired":
            _log.warning("wallet keeper: %s %s on %s: %s",
                         op, outcome, pid, resp.get("error"))
            strikes = self._error_strikes(pid, op, RECLAIM_ERROR_STRIKES_TO_HALT)
            if strikes >= RECLAIM_ERROR_STRIKES_TO_HALT:
                reason = (f"{strikes} consecutive {op} phases could not be "
                          "completed or confirmed — reclaim halted pending "
                          "operator review")
                if host_store.wallet_halt(pid, "reclaim", reason):
                    _log.error("wallet keeper: HARD HALT on %s reclaim — %s",
                               pid, reason)
                else:
                    _log.error("wallet keeper: reclaim halt on %s could NOT be "
                               "persisted (%s) — it is not durable; fix the store",
                               pid, reason)
            return f"{op}_failed"
        return op

    # ---- top-up ----------------------------------------------------------

    def _topup_cooldown_s(self, pid: str, knobs: Knobs) -> int:
        """The cooldown to enforce right now: the operator's knob, doubled per
        consecutive unmeasurable failure.

        Without this a deposit that fails every time re-fires on the plain
        cooldown forever. The base floor applies only once there IS a strike,
        because the knob may legitimately be 0 and 0 doubled is still 0."""
        strikes = self._error_strikes(pid, "topup", TOPUP_ERROR_STRIKES_TO_HALT)
        if not strikes:
            return knobs.topup_cooldown_s
        base = max(knobs.topup_cooldown_s, TOPUP_BACKOFF_BASE_S)
        return int(min(base * (2 ** (strikes - 1)), TOPUP_BACKOFF_CAP_S))

    async def _maybe_topup(self, pid: str, knobs: Knobs, available: float,
                           reserved: "float | None" = None) -> str:
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
        if last_ts is not None and now - int(last_ts) < self._topup_cooldown_s(pid, knobs):
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

        # VETO-ONLY hot-wallet floor, and it vetoes on ABSENCE too. The balance
        # behind it comes from an untrusted public RPC, so it can only ever block
        # a deposit — but a floor that is skipped whenever the reading is missing
        # is not a floor at all: `_fetch_chain_balances` returns {} on any
        # failure, so an RPC outage (or whoever controls the default public
        # endpoint) removed the guardrail by simply not answering. Failing closed
        # here is cheap: the reading refreshes every poll tick, and the cost of
        # waiting for one is a delayed deposit, not a lost one.
        wallet_usdc = as_float(self._chain_view(pid).get("wallet_usdc"))
        if wallet_usdc is None:
            _log.warning("wallet keeper: %s top-up skipped — no usable hot-wallet "
                         "reading (absent, stale, or the RPC is down), so the "
                         "%.4f USDC floor cannot be checked",
                         pid, knobs.topup_wallet_floor_usdc)
            return "wallet_unreadable"
        if wallet_usdc - amount < knobs.topup_wallet_floor_usdc:
            _log.warning("wallet keeper: %s top-up skipped — hot wallet %.6f USDC "
                         "would fall below the %.4f floor after a %.4f deposit",
                         pid, wallet_usdc, knobs.topup_wallet_floor_usdc, amount)
            return "wallet_floor"

        op_id = host_store.wallet_op_begin(
            pid, "topup", amount_usdc=amount,
            reason=f"deposits_available {available} < trigger "
                   f"{knobs.topup_trigger_usdc}",
            pre_available=available, pre_reserved=reserved)
        if op_id is None:
            _log.error("wallet keeper: cannot persist the topup intent for %s — "
                       "refusing to fire an unlogged deposit", pid)
            return "ledger_unavailable"
        try:
            # Amounts go to the CLI as a plain decimal string (control.js
            # validates the shape AND re-caps the value server-side).
            resp = await self.control("deposit", {"amount": f"{amount:.6f}"},
                                      timeout=DEPOSIT_TIMEOUT_S)
        except asyncio.CancelledError:
            host_store.wallet_op_finish(op_id, "unknown",
                                        detail="cancelled before the outcome was known")
            raise
        except Exception as exc:  # noqa: BLE001 — the row must never dangle
            # `_control_post` converts transport failures into responses, so
            # reaching here means something unforeseen — an import failure, a
            # library raising outside its own hierarchy. The guard mirrors
            # `_fire_reclaim_phase`, which had it from the start.
            host_store.wallet_op_finish(op_id, "unknown",
                                        detail=f"{type(exc).__name__}: {exc}")
            raise
        outcome = self._outcome_for(resp)
        if outcome == "fired":
            host_store.wallet_op_finish(op_id, "fired",
                                        detail=str(resp.get("stdout") or "")[:2000])
            _log.info("wallet keeper: deposited %.4f USDC into %s escrow "
                      "(available was %s)", amount, pid, available)
            return "topup_fired"
        host_store.wallet_op_finish(op_id, outcome,
                                    detail=str(resp.get("error"))[:2000])
        if outcome == "unknown":
            # The request reached the wire. The CLI may have broadcast a Base
            # mainnet transaction and been killed before it could say so, so this
            # counts against the cap and the cooldown exactly like a deposit that
            # succeeded — re-firing on top of a possible in-flight deposit is the
            # one mistake that actually loses money.
            _log.error("wallet keeper: deposit on %s is UNRESOLVED (%s) — "
                       "recorded as UNKNOWN and counted as spent; the transaction "
                       "may have landed", pid, resp.get("error"))
            return "topup_unknown"
        _log.warning("wallet keeper: deposit failed on %s without reaching the "
                     "buyer CLI: %s", pid, resp.get("error"))
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
        # ...and a row nobody is writing any more is not a reading either. The
        # sidecar rewrites it every 60s; once it stops, the escrow this row
        # reports keeps being spent against by traffic the keeper cannot see, so
        # an old row showing a healthy balance is precisely the state in which
        # depositing (or declining to reclaim) is worst. Same fail-closed
        # direction, applied to age instead of absence.
        age_s = self._status_age_s(status)
        if age_s is None or age_s > STATUS_MAX_AGE_S:
            _log.warning("wallet keeper: %s buyer_status is %s (max %ds) — "
                         "standing down; is the sidecar alive?", pid,
                         "undated" if age_s is None else f"{age_s:.0f}s old",
                         STATUS_MAX_AGE_S)
            return {"decision": "status_stale", "age_s": age_s}
        out: dict[str, Any] = {"available": available, "reserved": reserved}
        self._reconcile_orphans(pid)
        halted = self._settle_topups(pid, available, reserved)
        out["reclaim"] = await self._maybe_reclaim(pid, knobs, available, reserved)
        out["topup"] = halted if halted else await self._maybe_topup(
            pid, knobs, available, reserved)
        out["decision"] = "acted"
        return out

    @staticmethod
    def _status_age_s(status: "dict | None") -> "float | None":
        """Seconds since the sidecar wrote this row, or None when it carries no
        usable stamp (`fetched_at` is epoch MILLISECONDS). A row from the future
        is treated as age 0 — clock skew is not evidence of staleness."""
        raw = (status or {}).get("fetched_at")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return None
        return max(0.0, time.time() - float(raw) / 1000.0)

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
        except Exception as exc:  # noqa: BLE001 — anything else from the knobs
            # A non-KnobError (an unreadable settings store, a knob that will not
            # coerce) used to propagate straight out of `cycle()`, leaving
            # KEEPER_STATE holding the LAST successful cycle — so /x/runtime went
            # on describing a keeper that had not run since. Report it in the
            # same place every other stand-down is reported.
            _log.exception("wallet keeper stood down: knobs unreadable")
            KEEPER_STATE.update({**state, "enabled": False,
                                 "error": f"knobs unreadable: "
                                          f"{type(exc).__name__}: {exc}"})
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
