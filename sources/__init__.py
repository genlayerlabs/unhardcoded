"""
Provider sources: read-only feeds of live pricing and balances per provider.

Strictly off the request path — a source being down never affects routing;
the router coasts on last-known prices (or the metrics seed before the
first refresh). See docs/superpowers/specs/2026-06-10-provider-sources-design.md.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Literal, Protocol, TypedDict

_log = logging.getLogger("unhardcoded.sources")


class Price(TypedDict):
    provider_id: str
    served_model_id: str          # the provider's wire id, e.g. "openai/gpt-5.5"
    model_family: str | None      # mapped curated family; None = unmapped
    price_in_usd_per_mtok: float
    price_out_usd_per_mtok: float


class Balance(TypedDict):
    kind: Literal["credits_usd", "deposits_usdc", "quota_window"]
    value: float | None
    detail: dict
    fetched_at: int


class ProviderSource(Protocol):
    name: str
    provider_ids: list[str]
    poll_interval_s: int | None   # None = passive-only (no refresh task)

    async def pricing(self) -> list[Price]: ...
    async def balances(self) -> dict[str, Balance]: ...


# source name -> {last_ok, error, prices_pushed, balances}
# Serialized (without secrets — there are none here) by the shim's /x/runtime.
SOURCE_STATE: dict[str, dict[str, Any]] = {}


def _served_pairs(catalog: dict) -> set[tuple[str, str]]:
    """Every (provider_id, family) pair the catalog routes. Marketplace
    providers can serve ANY curated family (their candidates come from
    offers), so they pair with every model."""
    pairs = set()
    families = list((catalog.get("models") or {}).keys())
    for family, model in (catalog.get("models") or {}).items():
        for served in model.get("served_by") or []:
            if served.get("provider"):
                pairs.add((served["provider"], family))
    for pid, p in (catalog.get("providers") or {}).items():
        if isinstance(p, dict) and p.get("discovery") == "marketplace":
            for family in families:
                pairs.add((pid, family))
    return pairs


def push_prices(host: Any, catalog: dict, prices: list[Price]) -> int:
    """Write mapped prices into the core's metrics store (the one ranking
    and price-ceiling filters read). Unmapped or un-cataloged prices are
    skipped — sources never widen the catalog.

    Prices are stored raw. The host/core boundary applies the current
    `<provider>.price_multiplier` knob at selection time so changing the
    multiplier does not require waiting for the next source refresh."""
    pairs = _served_pairs(catalog)
    now = int(time.time())
    pushed = 0
    for p in prices:
        family = p.get("model_family")
        provider = p.get("provider_id")
        if not family or (provider, family) not in pairs:
            continue
        host.update_metrics(provider, family, {
            "price_in": p["price_in_usd_per_mtok"],
            "price_out": p["price_out_usd_per_mtok"],
            "price_refreshed_at": now,
        })
        pushed += 1
    return pushed


# Balance kinds that denominate SPENDABLE MONEY, and so map onto the engine's
# pointwise `credits` field. `quota_window` (a subscription's used-fraction) does
# not — publishing it as credits would compare a ratio against a dollar amount.
_CREDIT_BALANCE_KINDS = ("deposits_usdc", "credits_usd")


def push_credits(host: Any, balances: "dict[str, Balance]") -> int:
    """Publish each provider's spendable balance into the core's `__credits|<pid>`
    metrics slot, which the algebra reads back pointwise as `credits`.

    This is what makes a funding gate expressible in a policy term at all: without
    it `credits` is whatever metrics.live.lua seeded (or the field default, 0) and
    a `cmp(credits, ge, N)` clause is decorative. The host envelope uses it to keep
    AntSeed out of ranking when its escrow cannot pay — see config.live.lua."""
    pushed = 0
    for pid, bal in (balances or {}).items():
        if bal.get("kind") not in _CREDIT_BALANCE_KINDS:
            continue
        value = bal.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        host.update_metrics("__credits", pid,
                            {"free_credits_remaining_usd": float(value)})
        pushed += 1
    return pushed


def seed_credits(host: Any, registry: "list[ProviderSource]") -> int:
    """COLD START: publish last-known credits before the app serves traffic.

    The host envelope's AntSeed clause (`credits >= 1.0`) fails CLOSED — the
    engine's `credits` default is 0, so a pod that has not yet completed a
    balances refresh rejects every AntSeed candidate for up to one poll interval.
    A source that can answer from durable state (`credits_seed()` — no network)
    closes that window. Never raises: a seed failure just leaves the cold-start
    gate closed until the first refresh tick, which is the safe direction."""
    seeded = 0
    for source in registry:
        fn = getattr(source, "credits_seed", None)
        if fn is None:
            continue
        try:
            for pid, value in (fn() or {}).items():
                host.update_metrics("__credits", pid,
                                    {"free_credits_remaining_usd": float(value)})
                seeded += 1
        except Exception as exc:  # noqa: BLE001 — isolation is the contract
            _log.warning("credits seed failed for %s: %s", source.name, exc)
    return seeded


async def refresh_once(host: Any, catalog: dict, source: ProviderSource) -> None:
    """One refresh tick. Never raises: failures land in SOURCE_STATE and the
    last-known data stays in place."""
    state = SOURCE_STATE.setdefault(source.name, {
        "last_ok": None, "error": None, "prices_pushed": 0, "balances": {},
    })
    try:
        prices = await source.pricing()
        state["prices_pushed"] = push_prices(host, catalog, prices)
        state["balances"] = await source.balances()
        # Balances are also the credit signal the algebra gates on — push them
        # every tick, not just into the dashboard view.
        state["credits_pushed"] = push_credits(host, state["balances"])
        if hasattr(source, "market_book"):
            state["book"] = source.market_book()
        # Per-source counters (offers kept/dropped/suppressed, wallet health).
        # snapshot_stats had no non-test caller; surfacing it here is what puts
        # it on /x/runtime + /x/market alongside `balances`.
        if hasattr(source, "snapshot_stats"):
            state["stats"] = source.snapshot_stats()
        state["last_ok"] = int(time.time())
        state["error"] = None
    except Exception as exc:  # noqa: BLE001 — isolation is the contract
        state["error"] = f"{type(exc).__name__}: {exc}"


async def _run_source(host: Any, catalog: dict, source: ProviderSource) -> None:
    while True:
        await refresh_once(host, catalog, source)
        # jitter ±10% so multiple sources don't sync up
        await asyncio.sleep(source.poll_interval_s * (0.9 + 0.2 * random.random()))


def start_refresh_tasks(host: Any, catalog: dict,
                        registry: list[ProviderSource]) -> list[asyncio.Task]:
    """Call from an async context (FastAPI startup). First refresh runs
    immediately; cadence is per-source."""
    return [asyncio.create_task(_run_source(host, catalog, s))
            for s in registry if s.poll_interval_s]
