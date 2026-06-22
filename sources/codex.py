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
Two signals feed the ramp: the `*used-percent*` quota header when codex exposes
one, AND recently observed 429s (the only signal when it doesn't). Billing stays
$0 (executed cost) — this is ranking-only.
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
        """0 (codex free, wins) → 1 (fully demoted) from the quota header and/or
        recently observed 429s, whichever is higher."""
        bal = (self._balances_sync().get(self.provider_ids[0]) or {})
        used = bal.get("value")
        recent_429 = (bal.get("detail") or {}).get("recent_429_count") or 0
        start = settings.get("codex.quota_demote_start")
        shed = settings.get("codex.quota_429_shed")
        header_frac = (max(0.0, (float(used) - start) / (1.0 - start))
                       if used is not None and start < 1.0 else 0.0)
        rl_frac = (min(1.0, recent_429 / shed) if shed > 0 else 0.0)
        return max(0.0, min(1.0, max(header_frac, rl_frac)))

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
        return {self.provider_ids[0]: {
            "kind": "quota_window",
            "value": used_fraction,
            "detail": {"recent_429_count": recent_429, "last_429_at": last_429,
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
