"""
Codex source: never PROBES the codex endpoint (that would burn the quota it
measures). It observes real traffic — the codex adapter pushes an observation
per call (status + any ratelimit/usage/quota response headers) via ingest() —
and self-refreshes on a local tick (no endpoint hit) so the scarcity price
decays back once the rate-limit pressure ages out.

Scarcity pricing: codex is imputed $0 (sunk-cost subscription), so a cost-led
policy would route ALL of a family's traffic to it until it 429s, then oscillate.
So as the subscription gets strained the host imputes a RISING ranking price so
paid routes take over before the 429 wall — and it decays back as pressure eases.
Three signals feed the ramp: the `*used-percent*` quota header when codex
exposes one, a `*reset*` / `retry-after` header on a 429 (which pins the demote
until the quota actually rolls over, instead of letting it decay and re-probe an
exhaustion already known), AND recently observed 429s (the only signal when
neither header is present). Billing stays $0 (executed cost) — ranking-only.
"""
from __future__ import annotations

import time
from collections import deque

import settings
import sources as _sources
from sources import Balance, Price

# Knobs (demote start, imputed prices, 429 window/shed) are operator-tunable from
# the dashboard Config tab — read live via settings.get so an override applies
# without a restart. The 429-driven ramp engages when there is no quota header:
# N recent 429s within the window ramp the price to full demote; they age out of
# the window so it recovers.


def _reset_at(headers: dict, observed_ts: int | None) -> int | None:
    """Epoch seconds at which the codex quota is said to reset, or None.

    The value is accepted in either shape without knowing the vendor's choice:
    a large number is an absolute epoch, a small one is seconds-from-observation
    (so an old event's short reset is correctly already past). Unparseable or
    non-positive values yield None and the caller falls back to the 429 ramp.
    """
    for name, raw in (headers or {}).items():
        n = str(name).lower()
        if "reset" not in n and n != "retry-after":
            continue
        try:
            v = float(str(raw).strip())
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        if v > 10**9:                      # absolute epoch
            return int(v)
        base = observed_ts if observed_ts is not None else int(time.time())
        return int(base + v)               # seconds from when it was observed
    return None


class CodexSource:
    name = "codex"
    # A local tick (NOT an endpoint probe): re-imputes the scarcity price so it
    # decays back even when no fresh codex traffic is arriving to drive ingest().
    poll_interval_s = 30

    def __init__(self, provider_id: str):
        self.provider_ids = [provider_id]
        self._events: deque[dict] = deque(maxlen=500)
        self._host = None
        self._families: list[str] = []

    def bind(self, host, families: list[str]) -> None:
        """Give the source a push channel: on every observed signal it
        re-imputes scarcity prices for the codex-served families."""
        self._host = host
        self._families = list(families)

    def ingest(self, provider_id: str, signal: dict) -> None:
        self._events.append(signal)
        # publish synchronously on the call (responsive); the poll tick handles
        # decay when traffic stops.
        state = _sources.SOURCE_STATE.setdefault(self.name, {
            "last_ok": None, "error": None, "prices_pushed": 0, "balances": {},
        })
        state["balances"] = self._balances_sync()
        state["last_ok"] = int(time.time())
        self._push_scarcity_prices()

    # ---- scarcity ------------------------------------------------------

    def _demote_frac(self) -> float:
        """0 (codex free, wins) → 1 (fully demoted) from the quota header, a
        known reset time, and/or recently observed 429s, whichever is higher."""
        bal = (self._balances_sync().get(self.provider_ids[0]) or {})
        used = bal.get("value")
        detail = bal.get("detail") or {}
        recent_429 = detail.get("recent_429_count") or 0
        start = settings.get("codex.quota_demote_start")
        shed = settings.get("codex.quota_429_shed")
        header_frac = (max(0.0, (float(used) - start) / (1.0 - start))
                       if used is not None and start < 1.0 else 0.0)
        rl_frac = (min(1.0, recent_429 / shed) if shed > 0 else 0.0)

        # A 429 says the subscription is out; a reset header says until WHEN.
        # Without the second the 429 ramp decays after quota_429_window_s and
        # the router re-probes an exhaustion it already knows about, paying a
        # failed call at the head of the cascade every window. While the reset
        # is still ahead, stay fully demoted so paid routes carry the traffic.
        reset_at = detail.get("reset_at")
        reset_frac = 1.0 if reset_at is not None and time.time() < reset_at else 0.0

        return max(0.0, min(1.0, max(header_frac, rl_frac, reset_frac)))

    def _push_scarcity_prices(self) -> None:
        if self._host is None or not self._families:
            return
        frac = self._demote_frac()
        pin = settings.get("codex.imputed_price_in") * frac
        pout = settings.get("codex.imputed_price_out") * frac
        now = int(time.time())
        for family in self._families:
            self._host.update_metrics(self.provider_ids[0], family, {
                "price_in": pin, "price_out": pout, "price_refreshed_at": now,
            })

    def _balances_sync(self) -> dict[str, Balance]:
        events = list(self._events)
        if not events:
            return {}
        used_fraction = None
        observed: dict[str, str] = {}
        last_429 = None
        recent_429 = 0
        reset_at = None
        now_ts = int(time.time())
        for e in events:
            for k, v in (e.get("headers") or {}).items():
                observed[k] = str(v)
                if "used-percent" in k:
                    try:
                        used_fraction = float(v) / 100.0
                    except (TypeError, ValueError):
                        pass
            if e.get("status") == 429:
                last_429 = e.get("ts")
                ts = e.get("ts")
                if ts is None or (now_ts - ts) <= settings.get("codex.quota_429_window_s"):
                    recent_429 += 1
                # Only a 429 carries "you are out until X"; the same header on a
                # healthy 200 just reports when the window rolls over and must
                # not be read as exhaustion.
                at = _reset_at(e.get("headers") or {}, ts)
                if at is not None and (reset_at is None or at > reset_at):
                    reset_at = at
        return {self.provider_ids[0]: {
            "kind": "quota_window",
            "value": used_fraction,
            "detail": {"recent_429_count": recent_429, "last_429_at": last_429,
                       "reset_at": reset_at,
                       "observed_headers": observed, "events": len(events)},
            "fetched_at": int(time.time()),
        }}

    # ---- ProviderSource ------------------------------------------------

    async def pricing(self) -> list[Price]:
        # Local recompute (no endpoint probe) so the refresh loop decays/recovers
        # the scarcity price without needing fresh traffic.
        frac = self._demote_frac()
        pin = settings.get("codex.imputed_price_in") * frac
        pout = settings.get("codex.imputed_price_out") * frac
        return [{"provider_id": self.provider_ids[0], "served_model_id": fam,
                 "model_family": fam,
                 "price_in_usd_per_mtok": pin, "price_out_usd_per_mtok": pout}
                for fam in self._families]

    async def balances(self) -> dict[str, Balance]:
        return self._balances_sync()
